"""Packs documents into one stream of a stage: <stage root>/streams/<name>.

Documents come from one of two kinds of sources:
  --store NAME  a document store written by a build_*.py script (<store root>/<NAME>);
  --src DIR     already-tokenized 4K MDS streams, rebuilt into documents run by run
                (see mds.split_runs). --src can be given several times.

Every document goes through these steps, in this order:
  1. it is dropped if it holds a special token anywhere except its final EOS. Literal strings
     such as "<filename>" in PDF text are encoded as control tokens and must not reach
     training. <think> and </think> are allowed only with --allow-think;
  2. with --truncate-doc-tokens, a longer document is cut to that many tokens plus EOS, so that
     one huge document, such as a whole journal issue, does not fill dozens of blocks in a row;
  3. it is dropped if it is shorter than --min-doc-tokens or longer than --max-doc-tokens;
  4. it is dropped if an identical document was already kept in this part;
  5. it is packed into blocks of exactly the stage's block length (mds.Packer).
--length-filter-only skips steps 1 and 4. The 16K stage repacked the 4K pretraining streams
that way, so rebuilding those streams needs it.

A stream is built in parts that run as separate processes: part P of N takes every Nth store
file or 4K run, and `merge` then joins the parts under one index.json. Store files and runs are
visited in sorted order, or in a seeded random order when the stage sets SHUFFLE_SEED (the 32K
stage does) or --seed is given. --target-tokens stops each part once it has kept
target / nparts tokens; with a random order, a capped stream is a random sample of its pool.
A stream that is already merged is never packed again in place, because a training run may be
reading it.

Run selection, only for --src:
  --sample-tokens T          uses only a seeded random sample of whole runs (--sample-seed)
                             that reaches T estimated tokens (the 16K nemotron_cc);
  --exclude-sample-tokens T  skips exactly the runs that --sample-tokens T would use with
                             --exclude-sample-seed (the 32K nemotron_cc skips the 16K's runs).

Usage:
  pack.py pack --stage 32k --name NAME --store STORE [--store-root DIR] [filters] --part P --nparts N
  pack.py pack --stage 16k --name NAME --src DIR [--src DIR] [run selection] [filters] --part P --nparts N
  pack.py merge --stage 32k --name NAME --nparts N
The arguments of every stream are in stage_16k.py and stage_32k.py.
"""
import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict

import numpy as np

import documents
import mds
import stages

# Runs are sized from their sample counts when a sample of runs is drawn. The estimate has to
# stay 4,096 tokens per 4K sample, because any other estimate changes which runs a seeded sample
# picks.
TOKENS_PER_4K_SAMPLE = 4096


# ---------------------------------------------------------------- document sources

def store_files_of_part(a, seed):
    """Returns this part's share of the store files, in the order they are read."""
    files = documents.docstore_files(os.path.join(a.store_root, a.store))
    if seed is not None:
        random.Random(seed).shuffle(files)
    return files[a.part::a.nparts]


def store_documents(a, seed, stats):
    """Yields the documents of this part's share of the store files."""
    store = os.path.join(a.store_root, a.store)
    mine = store_files_of_part(a, seed)
    stats["store_files"] = len(mine)
    for name in mine:
        yield from documents.iter_docstore_file(store, name)


def list_runs(srcs):
    """Lists (shard directory, shard entries, run, estimated tokens) for every run, in source order."""
    runs = []
    for src in srcs:
        root, shards = mds.read_shards(src)
        for run in mds.split_runs(shards):
            runs.append((root, shards, tuple(run), sum(shards[i]["samples"] for i in run) * TOKENS_PER_4K_SAMPLE))
    return runs


def sample_runs(runs, target_tokens, seed):
    """Returns a seeded random sample of whole runs that reaches target_tokens, in source order.

    Runs are taken in a seeded random order until their estimated tokens reach the target, so
    the chosen documents are a random sample of raw files and the choice is reproducible.
    """
    order = list(range(len(runs)))
    random.Random(seed).shuffle(order)
    keep, total = [], 0
    for i in order:
        if total >= target_tokens:
            break
        keep.append(i)
        total += runs[i][3]
    return [runs[i] for i in sorted(keep)]


def runs_of_part(a, seed, stats):
    """Returns this part's share of the 4K runs, after the sample and exclusion options."""
    runs = list_runs(a.src)
    if a.exclude_sample_tokens:
        used = {(r[0], r[2]) for r in sample_runs(runs, a.exclude_sample_tokens, a.exclude_sample_seed)}
        runs = [r for r in runs if (r[0], r[2]) not in used]
        stats["runs_excluded"] = len(used)
    if a.sample_tokens:
        runs = sample_runs(runs, a.sample_tokens, a.sample_seed)
    if seed is not None:
        random.Random(seed).shuffle(runs)
    return runs[a.part::a.nparts]


def run_documents(a, seed, stats):
    """Yields the documents of this part's share of the 4K runs."""
    mine = runs_of_part(a, seed, stats)
    stats["runs_available"] = len(mine)
    for root, shards, run, _ in mine:
        stats["runs_read"] += 1
        yield from mds.iter_run_docs(root, shards, run, stats)


# ---------------------------------------------------------------- filtering and packing

def filter_document(doc, a, seen, stats):
    """Applies steps 1 to 4 of the module docstring; returns the document to pack, or None."""
    if not a.length_filter_only:
        body = doc[:-1]
        special = body[body >= documents.FIRST_ADDED_TOKEN_ID]
        if special.size and not (a.allow_think and np.all(np.isin(special, documents.THINK_TOKEN_IDS))):
            stats["special_tokens"] += 1
            return None
    if a.truncate_doc_tokens and len(doc) - 1 > a.truncate_doc_tokens:
        doc = np.append(doc[:a.truncate_doc_tokens], np.uint32(documents.EOS))
        stats["truncated"] += 1
    if len(doc) - 1 < a.min_doc_tokens:
        stats["too_short"] += 1
        return None
    if a.max_doc_tokens and len(doc) - 1 > a.max_doc_tokens:
        stats["too_long"] += 1
        return None
    if not a.length_filter_only:
        key = hashlib.blake2b(doc.tobytes(), digest_size=8).digest()
        if key in seen:
            stats["duplicates"] += 1
            return None
        seen.add(key)
    return doc


def pack(a, stage):
    stream_dir = os.path.join(stage.ROOT, "streams", a.name)
    # A merged stream may be in use by a training run, so it is never rebuilt in place by accident.
    if os.path.exists(os.path.join(stream_dir, "index.json")):
        raise SystemExit(f"{stream_dir} is already merged; remove it to rebuild")
    out = os.path.join(stream_dir, f"part_{a.part:02d}")
    if os.path.exists(os.path.join(out, "index.json")):
        raise SystemExit(f"{out} already finished; remove it to rebuild")
    seed = stage.SHUFFLE_SEED if a.seed is None else a.seed
    stats = defaultdict(int)
    docs = store_documents(a, seed, stats) if a.store else run_documents(a, seed, stats)
    writer = mds.StreamWriter(out, stage.BLOCK_TOKENS)
    packer = mds.Packer(writer, stage.BLOCK_TOKENS, buffer_docs=a.buffer_docs)
    target = a.target_tokens / a.nparts if a.target_tokens else 0
    seen = set()
    start = last_report = time.time()
    for doc in docs:
        stats["docs"] += 1
        doc = filter_document(doc, a, seen, stats)
        if doc is None:
            continue
        stats["kept"] += 1
        stats["kept_tokens"] += len(doc)
        packer.add(doc)
        if time.time() - last_report > 60:
            last_report = time.time()
            print(f"{time.strftime('%H:%M:%S')} {a.name} part {a.part}: {json.dumps(stats)}", flush=True)
        if target and stats["kept_tokens"] >= target:
            break
    packer.flush()
    writer.close()
    stats.update(packer.stats)
    stats["blocks_written"] = writer.blocks
    stats["tokens_written"] = writer.blocks * stage.BLOCK_TOKENS
    stats["minutes"] = round((time.time() - start) / 60, 1)
    manifest_dir = os.path.join(stage.ROOT, "manifests", "pack")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, f"{a.name}.part_{a.part:02d}.json"), "w") as f:
        json.dump({"args": vars(a), "stats": stats}, f, indent=1)
    print("DONE", json.dumps(stats), flush=True)


def merge(a, stage):
    stream_dir = os.path.join(stage.ROOT, "streams", a.name)
    parts = [os.path.join(stream_dir, f"part_{i:02d}") for i in range(a.nparts)]
    missing = [p for p in parts if not os.path.exists(os.path.join(p, "index.json"))]
    if missing:
        raise SystemExit(f"parts not finished: {missing}")
    samples = mds.merge_parts(stream_dir, parts)
    print(f"merged {a.name}: {samples} samples, {samples * stage.BLOCK_TOKENS / 1e9:.3f}B tokens")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pack", "merge"])
    stages.add_stage_argument(ap)
    ap.add_argument("--name", required=True, help="the stream to build: <stage root>/streams/NAME")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    source = ap.add_argument_group("source (exactly one kind)")
    source.add_argument("--store", default="", help="document store name under --store-root")
    source.add_argument("--store-root", default="", help="where the stores are (default: the stage's STORE_ROOT)")
    source.add_argument("--src", action="append", default=[], help="a 4K MDS stream; can be repeated")
    order = ap.add_argument_group("order and run selection")
    order.add_argument("--seed", type=int, default=None, help="visit files and runs in this seeded order "
                                                              "(default: the stage's SHUFFLE_SEED)")
    order.add_argument("--sample-tokens", type=float, default=0)
    order.add_argument("--sample-seed", type=int, default=16384)
    order.add_argument("--exclude-sample-tokens", type=float, default=0)
    order.add_argument("--exclude-sample-seed", type=int, default=16384)
    filters = ap.add_argument_group("filters")
    filters.add_argument("--min-doc-tokens", type=int, default=0)
    filters.add_argument("--max-doc-tokens", type=int, default=0)
    filters.add_argument("--truncate-doc-tokens", type=int, default=0)
    filters.add_argument("--allow-think", action="store_true", help="keep documents with <think> and </think>")
    filters.add_argument("--length-filter-only", action="store_true",
                         help="skip the special-token check and the deduplication")
    filters.add_argument("--target-tokens", type=float, default=0, help="kept tokens of the whole stream")
    ap.add_argument("--buffer-docs", type=int, default=4096, help="documents the packer buffers for best fit")
    a = ap.parse_args()
    stage = stages.get(a.stage)
    if a.cmd == "merge":
        merge(a, stage)
        return
    if bool(a.store) == bool(a.src):
        raise SystemExit("give exactly one source kind: --store or --src")
    if a.store and (a.sample_tokens or a.exclude_sample_tokens):
        raise SystemExit("--sample-tokens and --exclude-sample-tokens select 4K runs and need --src")
    a.store_root = a.store_root or stage.STORE_ROOT
    pack(a, stage)


if __name__ == "__main__":
    main()
