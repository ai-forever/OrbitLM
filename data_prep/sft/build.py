"""Build one version of the OrbitLM SFT mix.

A version yaml lists streams, their recipe weights, and which streams are
capped at a few epochs. `--stats` counts packed tokens. `--build` writes MDS
streams under the version's out_root. `--resolve` turns the counts into
sampling weights and writes, next to the streams, a resolved yaml and the
data block of the training config.

Values in a version yaml may name environment variables as ${NAME}; the
release version uses ${ORBIT_RAW}, ${ORBIT_DATA} and ${ORBIT_TOKENIZERS}.
They are expanded when the yaml is loaded, and an unset variable stops the
script.

  python build.py --version versions/v5_smoltalk2.yaml --stats
  python build.py --version versions/v5_smoltalk2.yaml --build
  python build.py --version versions/v5_smoltalk2.yaml --resolve
  python build.py --version versions/v5_smoltalk2.yaml --decode OpenThoughts3_1.2M_think
  python build.py --version versions/v5_smoltalk2.yaml --progress

A version with chain_filter drops a dialog whose longest reasoning chain is
longer than max(floor, that percentile of the stream). The percentile is
computed on the dialogs that pass preparation, then the stream is tokenized.

A jsonl or tool_jsonl file larger than 256 MiB is read in line-aligned byte
ranges, one range per worker, up to --jobs. The source file is not copied.
A stream that dedups prompts is not split, and --limit keeps one reader so
the limit still counts rows of the whole file.
"""
import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

import chains
import dialogs
import encode as encode_mod
from weights import addition_weights, epochs_at_budget, part_weights, recipe_sampling_weights
from system_prompts import load_rewrite_table

HERE = os.path.dirname(os.path.abspath(__file__))
# One jsonl below this size stays on one worker. Above it, ranges are at
# least JSONL_MIN_SLICE bytes, and there are at most --jobs of them.
JSONL_KINDS = ("jsonl", "tool_jsonl")
JSONL_SPLIT_MIN_FILE = 256 * 1024 * 1024
JSONL_MIN_SLICE = 32 * 1024 * 1024


def jsonl_piece_count(size, jobs, kind, limit=0, dedup=False):
    """How many byte ranges a jsonl file should be read in.

    One range unless the file is a large jsonl, the run has several workers,
    and the stream does not dedup prompts. A positive limit is a smoke test
    and stays on one reader, so the limit is not multiplied by the ranges.
    """
    if kind not in JSONL_KINDS or dedup or limit or jobs <= 1 or size < JSONL_SPLIT_MIN_FILE:
        return 1
    return max(1, min(int(jobs), int(size) // JSONL_MIN_SLICE))


_ENV_REFERENCE = re.compile(r"\$\{(\w+)\}")


def expand_env(value):
    """Replaces every ${NAME} in the strings of a loaded yaml with that environment variable."""
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def lookup(match):
        name = match.group(1)
        if not os.environ.get(name):
            raise SystemExit(f"set {name} first; data_prep/README.md describes the ORBIT_* roots")
        return os.environ[name]

    return _ENV_REFERENCE.sub(lookup, value)


def load_version(path):
    with open(path, "r", encoding="utf-8") as handle:
        version = yaml.safe_load(handle)
    # The training config refers to the streams through the unexpanded out_root, so that it
    # stays free of machine paths.
    out_root_text = version["out_root"]
    version = expand_env(version)
    version["_out_root_text"] = out_root_text
    version["_path"] = path
    return version


def stream_files(spec):
    if spec["kind"] == "bench":
        return [None]
    return dialogs.expand_paths(spec["path"])


def version_rewrite_path(version):
    """Absolute path of the system-prompt rewrite table, or None when this mix has none.

    A relative path in the version yaml is taken relative to the yaml. A table that the version
    names but that does not exist stops the build, because building without it would silently
    keep the prompts that the table is there to replace.
    """
    raw = version.get("rewrite_table") or ""
    if not raw:
        return None
    if not os.path.isabs(raw):
        raw = os.path.join(os.path.dirname(os.path.abspath(version["_path"])), raw)
    if not os.path.isfile(raw):
        raise SystemExit(
            f"rewrite table not found: {raw}\n"
            "Pass --rewrite-table FILE to use another table; data_prep/README.md describes the table and how "
            "collect_system_prompts.py writes one.")
    return raw


def version_system_turns(version):
    """True when the mix keeps a dataset system prompt as its own system turn.

    v5 sets system_turns in its yaml. A rewrite table also implies system turns, because the
    table replaces whole system turns. Without either, the build folds a task system prompt into
    the first user turn.
    """
    return bool(version.get("system_turns") or version.get("rewrite_table"))


def collect_prompt_keys(version, limit, only=None, rewrite=None):
    """User-prompt keys of streams that later streams dedup against."""
    keys = set()
    selected = version["streams"]
    if only:
        selected = [spec for spec in selected if spec["id"] in only]
    if not any(spec.get("dedup_prompts") for spec in selected):
        return keys
    producers = {spec["id"] for spec in version["streams"] if spec.get("record_prompts")}
    if not producers:
        return keys
    for spec in version["streams"]:
        if spec["id"] not in producers:
            continue
        copied = dict(spec)
        copied["record_prompts"] = True
        copied["dedup_prompts"] = False
        for _turns in dialogs.iter_stream_dialogs(copied, limit=limit, seen_prompts=keys, rewrite=rewrite):
            pass
        print(f"prompt keys from {spec['id']}: {len(keys)}", flush=True)
    return keys


def p95_or_none(value):
    if value is None:
        return None
    return int(value)


def _sum_stats(parts):
    total = defaultdict(int)
    for part in parts:
        for key, value in part.items():
            if isinstance(value, int):
                total[key] += value
    return dict(total)


def merge_mds(part_dirs, out_dir):
    """Move shard files from part directories into one stream directory."""
    shards = []
    template = None
    number = 0
    os.makedirs(out_dir, exist_ok=True)
    for part in part_dirs:
        index_path = os.path.join(part, "index.json")
        if not os.path.isfile(index_path):
            continue
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        if template is None:
            template = index
        for shard in index.get("shards") or []:
            basename = shard["raw_data"]["basename"]
            # shard.00000.mds or shard.00000.mds.zstd: keep everything after the index.
            tail = basename.split(".", 2)[-1]
            new_base = f"shard.{number:05d}.{tail}"
            shutil.move(os.path.join(part, basename), os.path.join(out_dir, new_base))
            shard["raw_data"]["basename"] = new_base
            shards.append(shard)
            number += 1
        shutil.rmtree(part, ignore_errors=True)
    if template is None:
        return 0
    template["shards"] = shards
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as handle:
        json.dump(template, handle)
    return sum(shard.get("samples", 0) for shard in shards)


def _ranges_for_file(spec, path, jobs, limit):
    """Byte ranges of one input. (None, None) reads the whole file."""
    if spec["kind"] not in JSONL_KINDS or not path or not os.path.isfile(path):
        return [(None, None)]
    pieces = jsonl_piece_count(
        os.path.getsize(path), jobs, spec["kind"], limit=limit, dedup=bool(spec.get("dedup_prompts")),
    )
    if pieces <= 1:
        return [(None, None)]
    slices = dialogs.jsonl_slices(path, pieces, min_bytes=JSONL_MIN_SLICE)
    if len(slices) <= 1:
        return [(None, None)]
    print(
        f"  slice {spec['id']} {os.path.basename(path)} into {len(slices)} parts",
        flush=True,
    )
    return slices


def _tasks_for_stream(spec, version, out_root, limit, write, keys_path, jobs):
    context = int(version["seq_len"]) + 1
    files = stream_files(spec)
    tasks = []
    for path in files:
        ranges = _ranges_for_file(spec, path, jobs, limit)
        width = len(ranges)
        for slice_index, (start, end) in enumerate(ranges, start=1):
            part_dir = None
            if write:
                part_dir = os.path.join(out_root, spec["id"], "parts", f"{len(tasks):05d}")
            task = {
                "kind": spec["kind"],
                "path": path,
                "raw_root": spec.get("raw_root"),
                "sources": spec.get("sources"),
                "dedup_prompts": bool(spec.get("dedup_prompts")),
                "out_dir": part_dir,
                "context_size": context,
                "min_resp_tokens": int(version.get("min_resp_tokens", 2)),
                "prompt_keys_path": keys_path if spec.get("dedup_prompts") else None,
                "limit": limit,
                "tokenizer": version["tokenizer"],
                "enable_thinking": spec.get("enable_thinking", False),
                "loss_last_assistant": bool(spec.get("loss_last_assistant")),
                "stream_id": spec["id"],
                "progress_path": os.path.join(out_root, "progress_current.json"),
            }
            if start is not None:
                task["byte_start"] = start
                task["byte_end"] = end
                task["part_label"] = f"{os.path.basename(path)} [{slice_index}/{width}]"
            tasks.append(task)
    return tasks


def _map_tasks(tasks, fn, version, jobs, rewrite_path, on_done=None):
    """Run fn on each file. One file stays in this process."""
    system_turns = version_system_turns(version)
    if len(tasks) == 1 or jobs <= 1:
        encode_mod.init_worker(version["tokenizer"], rewrite_path, system_turns)
        parts = []
        for index, task in enumerate(tasks, start=1):
            part = fn(task)
            parts.append(part)
            if on_done:
                on_done(index, len(tasks), part)
        return parts
    from concurrent.futures import as_completed
    import multiprocessing
    context = multiprocessing.get_context("spawn")
    parts = []
    with ProcessPoolExecutor(
        max_workers=min(jobs, len(tasks)),
        mp_context=context,
        initializer=encode_mod.init_worker,
        initargs=(version["tokenizer"], rewrite_path, system_turns),
    ) as pool:
        futures = [pool.submit(fn, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            part = future.result()
            parts.append(part)
            if on_done:
                on_done(index, len(tasks), part)
    return parts


def _apply_chain_filter(tasks, version, jobs, rewrite_path, spec):
    """Set chain_max on every file of this stream. None when the version has no filter."""
    raw = version.get("chain_filter")
    if not raw:
        return None, None
    floor = int(raw.get("floor", 8192))
    rank = float(raw.get("percentile", 95))

    def on_done(index, total, part):
        print(
            f"  chain scan {spec['id']} {index}/{total} "
            f"dialogs={part['n_dialogs']} with_chain={int(part['lengths'].size)}",
            flush=True,
        )

    measured = _map_tasks(tasks, encode_mod.measure_chains, version, jobs, rewrite_path, on_done)
    arrays = [part["lengths"] for part in measured if part["lengths"].size]
    lengths = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int32)
    threshold, p95 = chains.chain_threshold(lengths, floor, rank)
    for task in tasks:
        task["chain_max"] = threshold
    shown = "none" if p95 is None else str(p95)
    print(
        f"  chain filter {spec['id']}: with_chain={int(lengths.size)} p95={shown} threshold={threshold}",
        flush=True,
    )
    return p95, threshold


def run_stream(spec, version, out_root, limit, write, jobs, keys_path, rewrite_path, on_file=None):
    tasks = _tasks_for_stream(spec, version, out_root, limit, write, keys_path, jobs)
    if not tasks:
        return {"blocks": 0, "packed_tokens": 0}
    chain_p95, chain_max = _apply_chain_filter(tasks, version, jobs, rewrite_path, spec)

    def on_done(index, total, part):
        print(f"  {spec['id']} {index}/{total} blocks={part.get('blocks')} kept={part.get('kept')}", flush=True)
        if on_file:
            on_file(index, total)

    parts = _map_tasks(tasks, encode_mod.encode_file, version, jobs, rewrite_path, on_done)
    stats = _sum_stats(parts)
    if chain_max is not None:
        stats["chain_p95"] = p95_or_none(chain_p95)
        stats["chain_max"] = int(chain_max)
    context = int(version["seq_len"]) + 1
    stats["packed_tokens"] = stats.get("blocks", 0) * context
    if write:
        part_dirs = [task["out_dir"] for task in tasks if task["out_dir"]]
        out_dir = os.path.join(out_root, spec["id"])
        samples = merge_mds(part_dirs, out_dir)
        stats["samples"] = samples
        with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=1)
    return stats


def stream_done(out_root, spec):
    return os.path.isfile(os.path.join(out_root, spec["id"], "index.json"))


def saved_stats(out_root, spec):
    path = os.path.join(out_root, spec["id"], "stats.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def progress_paths(out_root):
    return (
        os.path.join(out_root, "progress.json"),
        os.path.join(out_root, "progress_current.json"),
    )


def write_json_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def write_progress(out_root, name, total, finished, running, files_done=None, files_total=None):
    """How many streams of this run are tokenized.

    progress.json changes when a stream starts or finishes. progress_current.json
    is rewritten by the worker every few thousand rendered dialogs of the
    stream that is running.
    """
    payload = {
        "name": name,
        "total": total,
        "done": len(finished),
        "running": running,
        "finished": list(finished),
    }
    if files_done is not None:
        payload["files_done"] = files_done
        payload["files_total"] = files_total
    path, _current = progress_paths(out_root)
    write_json_atomic(path, payload)


def format_progress(out_root):
    path, current_path = progress_paths(out_root)
    if not os.path.isfile(path):
        return f"no progress yet: {path}"
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    lines = [f"{payload.get('name')}  {payload.get('done')}/{payload.get('total')} tokenized"]
    running = payload.get("running")
    if running:
        files = ""
        if payload.get("files_total"):
            files = f"  files {payload.get('files_done')}/{payload.get('files_total')}"
        lines.append(f"running: {running}{files}")
    if running and os.path.isfile(current_path):
        with open(current_path, "r", encoding="utf-8") as handle:
            current = json.load(handle)
        if current.get("stream") == running:
            verb = "scanned" if current.get("phase") == "chain" else "rendered"
            lines.append(f"  {current.get('file')}: {verb} {current.get('rendered')} dialogs")
    finished = payload.get("finished") or []
    if finished:
        lines.append("finished: " + ", ".join(finished))
    return "\n".join(lines)


def run_version(version, out_root, limit, write, jobs, overwrite, only):
    os.makedirs(out_root, exist_ok=True)
    rewrite_path = version_rewrite_path(version)
    rewrite = load_rewrite_table(rewrite_path, version_system_turns(version))
    keys = collect_prompt_keys(version, limit, only, rewrite=rewrite)
    keys_path = None
    if keys:
        keys_path = os.path.join(out_root, "twix_user_prompt_keys.npy")
        np.save(keys_path, np.asarray(sorted(keys), dtype=np.uint64))
    planned = [spec for spec in version["streams"] if not only or spec["id"] in only]
    finished = []
    write_progress(out_root, version["name"], len(planned), finished, None)
    results = {}
    for spec in planned:
        if write and stream_done(out_root, spec) and not overwrite:
            stats = saved_stats(out_root, spec)
            print(f"keep [{len(finished) + 1}/{len(planned)}] {spec['id']}: {stats}", flush=True)
            results[spec["id"]] = stats or {"blocks": 0, "packed_tokens": 0}
            finished.append(spec["id"])
            write_progress(out_root, version["name"], len(planned), finished, None)
            continue
        if write:
            target = os.path.join(out_root, spec["id"])
            if os.path.islink(target):
                os.unlink(target)
            else:
                shutil.rmtree(target, ignore_errors=True)
        _progress, current = progress_paths(out_root)
        if os.path.isfile(current):
            os.remove(current)
        print(f"start [{len(finished) + 1}/{len(planned)}] {spec['id']}", flush=True)
        write_progress(out_root, version["name"], len(planned), finished, spec["id"])

        def on_file(done_files, total_files, spec_id=spec["id"]):
            write_progress(
                out_root, version["name"], len(planned), finished, spec_id,
                files_done=done_files, files_total=total_files,
            )

        stats = run_stream(
            spec, version, out_root, limit, write, jobs, keys_path, rewrite_path, on_file=on_file,
        )
        results[spec["id"]] = stats
        finished.append(spec["id"])
        write_progress(out_root, version["name"], len(planned), finished, None)
        extra = ""
        if stats.get("chain_max") is not None:
            extra = (
                f" chain_max={stats.get('chain_max')} dropped_chain={stats.get('dropped_chain')}"
            )
        print(f"done [{len(finished)}/{len(planned)}] {spec['id']}: blocks={stats.get('blocks')} "
              f"kept={stats.get('kept')} long={stats.get('dropped_long')} "
              f"asst={stats.get('assistant_tokens')}{extra}", flush=True)
    stats_path = os.path.join(out_root, "stats.json")
    # Merge with any streams skipped by --streams so a partial run does not wipe them.
    if os.path.isfile(stats_path) and only:
        with open(stats_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        previous.update(results)
        results = previous
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    return results


def load_resolve_stats(version, out_root):
    """Packed-token counts for --resolve.

    stats_root holds streams that were not rebuilt. Counts written under
    out_root replace those, so a new stream packed into this version is used.

    A version that names a stats_root can also be built in full. Then stats_root
    may not exist, and every count comes from out_root. check_resolve_stats
    stops later if a stream has a count in neither file.
    """
    stats_root = version.get("stats_root") or out_root
    stats = {}
    read = set()
    for path in (os.path.join(stats_root, "stats.json"), os.path.join(out_root, "stats.json")):
        if not os.path.isfile(path) or os.path.realpath(path) in read:
            continue
        read.add(os.path.realpath(path))
        with open(path, "r", encoding="utf-8") as handle:
            stats.update(json.load(handle))
    if not read:
        raise SystemExit(f"no stats.json in {out_root} or in stats_root {stats_root}; run --build first")
    return stats


def check_resolve_stats(version, stats):
    """Stops when a stream of the version has no entry in stats.json.

    Without this check such a stream gets weight 0 and silently drops out of the mix.
    That happens, for example, when stats.json was written under older stream names.
    A stream that has an entry with 0 blocks was built and came out empty, so it stays allowed.
    """
    missing = [spec["id"] for spec in version["streams"] if spec["id"] not in stats]
    if missing:
        raise SystemExit(
            f"stats.json has no entry for {len(missing)} stream(s) of {version['name']}: "
            f"{', '.join(missing)}. Build them first, or rename the stats keys if the streams were renamed."
        )


def resolve_weights(version, stats):
    check_resolve_stats(version, stats)
    if version.get("parts"):
        return resolve_part_weights(version, stats)
    recipe = {}
    packed_by_stream = {}
    auto_tokens = {}
    context = int(version["seq_len"]) + 1
    for spec in version["streams"]:
        info = stats.get(spec["id"]) or {}
        packed = int(info.get("packed_tokens") or (info.get("blocks") or 0) * context)
        packed_by_stream[spec["id"]] = packed
        if spec.get("weight_mode") == "epoch_cap":
            auto_tokens[spec["id"]] = packed
        else:
            recipe[spec["id"]] = float(spec["weight"])
    # Recipe weight 1 means "the whole split, once per pass". As a sampling
    # weight that is one copy of its packed tokens, so every such split is
    # seen equally often. A smaller recipe weight keeps a smaller copy.
    explicit = recipe_sampling_weights(recipe, packed_by_stream)
    auto = addition_weights(
        explicit,
        auto_tokens,
        int(version["token_budget"]),
        float(version["max_epochs"]),
        float(version["additions_fraction"]),
    )
    weights = dict(explicit)
    weights.update(auto)
    weights = {name: value for name, value in weights.items() if value > 0 and (stats.get(name) or {}).get("blocks", 0)}
    packed = {
        name: int((stats.get(name) or {}).get("packed_tokens") or 0)
        for name in weights
    }
    return weights, epochs_at_budget(weights, packed, int(version["token_budget"])), []


def resolve_part_weights(version, stats):
    context = int(version["seq_len"]) + 1
    packed_by_stream = {}
    for spec in version["streams"]:
        info = stats.get(spec["id"]) or {}
        packed_by_stream[spec["id"]] = int(info.get("packed_tokens") or (info.get("blocks") or 0) * context)
    weights, exceeded = part_weights(
        version["parts"],
        version["streams"],
        packed_by_stream,
        int(version["token_budget"]),
    )
    weights = {
        name: value for name, value in weights.items()
        if value > 0 and (stats.get(name) or {}).get("blocks", 0)
    }
    packed = {name: int((stats.get(name) or {}).get("packed_tokens") or 0) for name in weights}
    epochs = epochs_at_budget(weights, packed, int(version["token_budget"]))
    # part_weights reports epochs against a weight total of 1. Recompute the
    # suggested fraction from the epochs actually written, which match that
    # total once empty streams are dropped.
    reported = []
    for item in exceeded:
        if item["stream"] not in epochs:
            continue
        epochs_now = epochs[item["stream"]]
        cap = item["epoch_cap"]
        if epochs_now <= cap + 1e-6:
            continue
        reported.append({
            "part": item["part"],
            "stream": item["stream"],
            "epochs": float(f"{epochs_now:.4f}"),
            "epoch_cap": cap,
            "fraction": item["fraction"],
            "fraction_at_cap": float(f"{cap / epochs_now * item['fraction']:.6f}"),
        })
    return weights, epochs, reported


def write_resolved(version, stats, weights, epochs, exceeded=None):
    resolved = {
        "name": version["name"],
        "seq_len": version["seq_len"],
        "token_budget": version["token_budget"],
        "out_root": version["out_root"],
        "weights": {name: float(f"{value:.8f}") for name, value in weights.items()},
        "epochs": {name: float(f"{value:.4f}") for name, value in epochs.items()},
        "packed_tokens": {
            name: int((stats.get(name) or {}).get("packed_tokens") or 0) for name in weights
        },
    }
    if version.get("parts"):
        resolved["parts"] = [
            {
                "id": part["id"],
                "fraction": float(part["fraction"]),
                "weight_mode": part["weight_mode"],
                "epoch_cap": float(part["epoch_cap"]),
            }
            for part in version["parts"]
        ]
        resolved["epoch_cap_exceeded"] = list(exceeded or [])
    else:
        resolved["max_epochs"] = version["max_epochs"]
        resolved["additions_fraction"] = version["additions_fraction"]
    path = os.path.join(version["out_root"], f"{version['name']}.resolved.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    print(f"wrote {path}", flush=True)
    data_path = write_data_block(version, weights, epochs, stats)
    print(f"wrote {data_path}", flush=True)
    return path


def write_data_block(version, weights, epochs, stats):
    """Writes the data block of the training config, with proportions that sum to 1.

    The block goes to <out_root>/<name>_data.yaml and is copied into the SFT config under
    configs/train/sft/. stage1_root keeps the ${ORBIT_DATA} form of out_root.
    """
    total = sum(weights.values())
    names = [spec["id"] for spec in version["streams"] if spec["id"] in weights]
    proportions = {name: weights[name] / total for name in names}
    largest = max(names, key=lambda name: proportions[name])
    proportions[largest] += 1.0 - sum(proportions.values())
    if version.get("parts"):
        part_bits = ", ".join(f"{part['id']} {float(part['fraction']):.6g}" for part in version["parts"])
        mix_note = [
            "# Every proportion is a sampling weight. Each part has its own fraction",
            f"# of the token budget ({part_bits}). A recipe weight of 1 means the same",
            "# number of epochs on every split in that part. An epoch_cap overrun is",
            "# reported and does not shrink the fraction.",
        ]
    else:
        mix_note = [
            "# Every proportion is a sampling weight. SmolTalk recipe fractions are",
            "# converted as recipe_weight * packed_tokens, so a recipe weight of 1",
            "# means the same number of epochs on every split. Russian streams,",
            "# T-Wix, Aya-ru, GrandMaster and bench_train share at most 15% of the",
            "# 10B budget and are capped at 5 epochs (data_prep/sft/weights.py).",
        ]
    lines = [
        f"# SFT {version['name']}: the data block written by data_prep/sft/build.py --resolve.",
        *mix_note,
        "# Proportions are relative; the loader renormalizes them. They sum to 1 here.",
        "labels_premasked: true",
        "stage1_token_dtype: uint32",
        f"stage1_root: {version['_out_root_text']}",
        "stage1_mixing_mode: mosaic_random",
        "",
        "stage1_streams:",
    ]
    for name in names:
        lines.append(f"  {name}: {name}")
    lines.append("")
    lines.append("stage1_dataset_proportions:")
    for name in names:
        epoch = epochs.get(name, 0.0)
        packed = int((stats.get(name) or {}).get("packed_tokens") or 0)
        lines.append(f"  {name}: {proportions[name]:.8f}  # epochs={epoch:.3f} packed_tokens={packed}")
    path = os.path.join(version["out_root"], f"{version['name']}_data.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def decode_one(out_root, stream, tokenizer_path):
    """Print one packed block and check the mask against the trainer's label shift."""
    import sys
    from streaming import StreamingDataset
    from transformers import AutoTokenizer

    repo = os.path.dirname(os.path.dirname(HERE))
    sys.path.insert(0, os.path.join(repo, "src"))
    from orbit.data import build_xy_from_sample, tokens_bytes_to_ids
    from orbit.modelling_ouro import build_document_cu_seqlens
    import torch

    stream_dir = os.path.join(out_root, stream)
    dataset = StreamingDataset(local=stream_dir, remote=None, shuffle=False, batch_size=1)
    sample = dataset[0]
    ids = tokens_bytes_to_ids(sample)
    mask = np.frombuffer(sample["loss_mask"], dtype=np.uint8)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
    print(text[:1500])
    print("---")
    print(f"dtype={sample['dtype']} tokens={len(ids)} mask_sum={int(mask.sum())} "
          f"eos_count={int((ids == 151643).sum())}")
    if "You are a helpful assistant" in text:
        raise SystemExit("the default Qwen system prompt leaked into the block")
    x, y = build_xy_from_sample(
        {"loss_mask": sample["loss_mask"]},
        ids,
        block_size=len(ids) - 1,
        pad_id=151643,
        eos_id=151643,
    )
    ignored = (y == -100).numpy()
    # y[k] is ignored when mask[k+1] is 0.
    if not np.array_equal(ignored, mask[1:] == 0):
        raise SystemExit("loss mask does not match build_xy_from_sample")
    batch = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    cu, max_len = build_document_cu_seqlens(batch, 151643)
    print(f"document_masking=1 boundary_token_id=151643 segments={len(cu) - 1} max_seg={max_len}")
    print(f"x={tuple(x.shape)} y={tuple(y.shape)} supervised_labels={int((y != -100).sum())}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="a yaml under data_prep/sft/versions/")
    parser.add_argument("--stats", action="store_true", help="count tokens, do not write MDS")
    parser.add_argument("--build", action="store_true", help="write MDS streams")
    parser.add_argument("--resolve", action="store_true", help="write sampling weights from stats.json")
    parser.add_argument("--progress", action="store_true", help="print how many streams are tokenized and exit")
    parser.add_argument("--decode", default="", help="decode the first block of this stream id")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="max rows per input file, 0 = all")
    parser.add_argument("--streams", default="", help="comma-separated stream ids")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-root", default="", help="override the version yaml out_root")
    parser.add_argument("--rewrite-table", default="",
                        help="the system-prompt rewrite table (default: the one the version yaml names, if any)")
    args = parser.parse_args()

    version = load_version(args.version)
    if args.out_root:
        version["out_root"] = version["_out_root_text"] = args.out_root
    if args.rewrite_table:
        version["rewrite_table"] = os.path.abspath(args.rewrite_table)
    out_root = version["out_root"]
    only = {item for item in args.streams.split(",") if item}
    if args.progress:
        print(format_progress(out_root))
        return
    if args.decode:
        decode_one(out_root, args.decode, version["tokenizer"])
        return
    if args.resolve:
        stats = load_resolve_stats(version, out_root)
        weights, epochs, exceeded = resolve_weights(version, stats)
        write_resolved(version, stats, weights, epochs, exceeded)
        for name, epoch in sorted(epochs.items(), key=lambda item: -item[1]):
            print(f"  {name}: weight={weights[name]:.6f} epochs={epoch:.3f}", flush=True)
        for item in exceeded:
            print(
                f"  epoch_cap {item['part']}/{item['stream']}: "
                f"epochs={item['epochs']:.3f} cap={item['epoch_cap']} "
                f"fraction={item['fraction']} would fit at {item['fraction_at_cap']}",
                flush=True,
            )
        return
    if not args.stats and not args.build:
        parser.error("pass --stats, --build, --resolve, --progress or --decode")
    run_version(version, out_root, args.limit, write=args.build, jobs=args.jobs, overwrite=args.overwrite, only=only)


if __name__ == "__main__":
    main()
