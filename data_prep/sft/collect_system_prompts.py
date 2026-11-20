"""Collect the unique system prompts of a mix and write a rewrite table for v5.

Reads the streams of a version yaml plus the two SmolTalk SystemChats
splits. Tool-calling sets are not included. Writes a summary, the identity
rewrites, the disputed prompts, REVIEW.md and a rewrite table that maps an
identity card to its scrubbed text into --out. Clean prompts are absent from
the table and stay verbatim. A rewrite that a person stored in the --saved
table wins over the classifier, so a table comes out again with its manual
edits. By default --saved is the table that the version yaml names, if it
names one. The v5 build reads only the table that its version yaml names,
or the one given with build.py --rewrite-table, so a new table changes the
mix only once it is put there.

  python data_prep/sft/collect_system_prompts.py --out /tmp/system_prompts \
      --version data_prep/sft/versions/v5_smoltalk2.yaml
"""
import argparse
import glob
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq

import dialogs
from build import load_version
from system_prompts import classify_system, row_has_tools, system_texts_from_row

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEMCHATS = (
    "smoltalk_smollm3_systemchats_30k_no_think",
    "smoltalk_systemchats_Qwen3_32B_think",
)


def _parquet_columns(path):
    names = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [name for name in ("messages", "conversation", "chat_template_kwargs", "subset") if name in names]
    return columns or None


def _scan_rows(stream_id, rows):
    counts = Counter()
    n_rows = 0
    n_with_system = 0
    n_with_tools = 0
    for row in rows:
        n_rows += 1
        if row_has_tools(row):
            n_with_tools += 1
        texts = system_texts_from_row(row)
        if not texts:
            continue
        n_with_system += 1
        for text in texts:
            counts[text] += 1
    return {
        "stream": stream_id,
        "n_rows": n_rows,
        "n_with_system": n_with_system,
        "n_with_tools": n_with_tools,
        "counts": counts,
    }


def _scan_parquet(stream_id, path):
    columns = _parquet_columns(path)
    rows = dialogs.iter_parquet([path], columns, limit=0)
    return _scan_rows(stream_id, rows)


def _scan_jsonl(stream_id, path):
    return _scan_rows(stream_id, dialogs.iter_jsonl([path], limit=0))


def _scan_job(job):
    stream_id, kind, path = job
    if kind == "jsonl":
        return _scan_jsonl(stream_id, path)
    return _scan_parquet(stream_id, path)


def stream_jobs(version):
    jobs = []
    for spec in version["streams"]:
        if spec["kind"] in ("bench", "tool_jsonl"):
            continue
        if spec["kind"] == "jsonl":
            for path in dialogs.expand_paths(spec["path"]):
                jobs.append((spec["id"], "jsonl", path))
            continue
        for path in dialogs.expand_paths(spec["path"]):
            jobs.append((spec["id"], spec["kind"], path))
    smoltalk_dir = version.get("smoltalk_dir")
    listed = {spec["id"] for spec in version["streams"]}
    if smoltalk_dir:
        # v5 lists the SystemChats splits as streams already, and they are read only once.
        for name in SYSTEMCHATS:
            if name in listed:
                continue
            matches = sorted(glob.glob(os.path.join(smoltalk_dir, name + "-*.parquet")))
            if not matches:
                raise FileNotFoundError(f"no SystemChats files for {name}")
            for path in matches:
                jobs.append((name, "smoltalk", path))
    return jobs


def merge_scans(parts):
    counts = {}
    streams = []
    for part in parts:
        streams.append({
            "stream": part["stream"],
            "n_rows": part["n_rows"],
            "n_with_system": part["n_with_system"],
            "n_with_tools": part["n_with_tools"],
            "n_unique": len(part["counts"]),
        })
        for text, count in part["counts"].items():
            slot = counts.get(text)
            if slot is None:
                slot = {"count": 0, "streams": Counter()}
                counts[text] = slot
            slot["count"] += count
            slot["streams"][part["stream"]] += count
    # Files of one stream are separate jobs. Fold them.
    folded = {}
    for row in streams:
        slot = folded.get(row["stream"])
        if slot is None:
            folded[row["stream"]] = dict(row)
            continue
        for key in ("n_rows", "n_with_system", "n_with_tools"):
            slot[key] += row[key]
    return counts, list(folded.values())


def classify_counts(counts):
    identity = []
    disputed = []
    buckets = Counter()
    for text, slot in counts.items():
        verdict = classify_system(text)
        buckets[verdict["bucket"]] += 1
        row = {
            "count": slot["count"],
            "streams": dict(slot["streams"]),
            "names": verdict["names"],
            "reason": verdict["reason"],
            "text": text,
        }
        if verdict["bucket"] == "identity":
            row["rewritten"] = verdict["rewritten"]
            identity.append(row)
        elif verdict["bucket"] == "disputed":
            disputed.append(row)
    identity.sort(key=lambda row: -row["count"])
    disputed.sort(key=lambda row: -row["count"])
    table = {row["text"]: row["rewritten"] for row in identity}
    return identity, disputed, buckets, table


def keep_saved_rewrites(identity, disputed, buckets, table, saved, counts):
    """Keep a rewrite that a person already stored in the saved table.

    The classifier result is used when the saved value matches it. A different
    saved value wins, including a prompt the classifier left untouched. A
    saved original that this scan no longer sees is dropped.
    """
    if not saved:
        return identity, disputed, buckets, table
    identity = list(identity)
    disputed = list(disputed)
    for original, rewritten in saved.items():
        if original not in counts:
            continue
        if table.get(original) == rewritten:
            continue
        row = None
        for collection, bucket_name in ((identity, "identity"), (disputed, "disputed")):
            for index, item in enumerate(collection):
                if item["text"] != original:
                    continue
                row = collection.pop(index)
                if bucket_name != "identity":
                    buckets[bucket_name] -= 1
                    buckets["identity"] += 1
                break
            if row is not None:
                break
        if row is None:
            slot = counts[original]
            row = {
                "count": slot["count"],
                "streams": dict(slot["streams"]),
                "names": [],
                "text": original,
            }
            buckets["clean"] -= 1
            buckets["identity"] += 1
        row["rewritten"] = rewritten
        row["reason"] = "manual"
        identity.append(row)
        table[original] = rewritten
    identity.sort(key=lambda row: -row["count"])
    disputed.sort(key=lambda row: -row["count"])
    return identity, disputed, buckets, table


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _preview(text, limit=240):
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def write_review(out_dir, stream_rows, buckets, identity, disputed, n_unique):
    lines = [
        "# System prompts for v5",
        "",
        f"Unique system strings: {n_unique}",
        "",
        "| Bucket | Unique strings |",
        "|---|---:|",
    ]
    for name in ("clean", "identity", "disputed"):
        lines.append(f"| {name} | {buckets.get(name, 0)} |")
    lines += ["", "## Streams", "", "| Stream | Rows | With system | Unique | Tool rows |", "|---|---:|---:|---:|---:|"]
    for row in sorted(stream_rows, key=lambda item: -item["n_with_system"]):
        lines.append(
            f"| {row['stream']} | {row['n_rows']} | {row['n_with_system']} | {row['n_unique']} | {row['n_with_tools']} |"
        )
    lines += ["", "## Identity cards", "", "The scrubbed text is what training would use.", ""]
    for row in identity[:40]:
        lines.append(f"- {row['count']} × `{', '.join(row['names'])}`")
        lines.append(f"  - from: {_preview(row['text'])}")
        lines.append(f"  - to: {_preview(row['rewritten'])}")
    if len(identity) > 40:
        lines.append(f"- … {len(identity) - 40} more in identity.jsonl")
    lines += ["", "## Disputed", "", "These stay verbatim until edited.", ""]
    for row in disputed[:80]:
        lines.append(
            f"- {row['count']} × `{', '.join(row['names'])}` ({row['reason']}): {_preview(row['text'])}"
        )
    if len(disputed) > 80:
        lines.append(f"- … {len(disputed) - 80} more in disputed.jsonl")
    lines.append("")
    with open(os.path.join(out_dir, "REVIEW.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=os.path.join(HERE, "versions", "v5_smoltalk2.yaml"))
    parser.add_argument("--out", required=True, help="directory for the new table and the review files")
    parser.add_argument("--saved", default="",
                        help="the table whose manual rewrites are kept (default: the one the version yaml names, if it exists)")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    version = load_version(args.version)
    if not args.saved and version.get("rewrite_table"):
        named = version["rewrite_table"]
        if not os.path.isabs(named):
            named = os.path.join(os.path.dirname(os.path.abspath(args.version)), named)
        args.saved = named
    jobs = stream_jobs(version)
    print(f"scanning {len(jobs)} files", flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        for index, part in enumerate(pool.map(_scan_job, jobs), start=1):
            parts.append(part)
            print(
                f"  {index}/{len(jobs)} {part['stream']}: "
                f"rows={part['n_rows']} systems={part['n_with_system']} unique={len(part['counts'])}",
                flush=True,
            )
    counts, stream_rows = merge_scans(parts)
    # A prompt can occur in several files of one stream, so the number of
    # unique prompts per stream is counted from the merged map instead of
    # adding up the per-file counts.
    per_stream_unique = Counter()
    for slot in counts.values():
        for stream in slot["streams"]:
            per_stream_unique[stream] += 1
    for row in stream_rows:
        row["n_unique"] = per_stream_unique[row["stream"]]
    identity, disputed, buckets, table = classify_counts(counts)
    buckets["clean"] = len(counts) - len(identity) - len(disputed)
    saved = {}
    if args.saved and os.path.isfile(args.saved):
        with open(args.saved, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
    identity, disputed, buckets, table = keep_saved_rewrites(
        identity, disputed, buckets, table, saved, counts,
    )
    os.makedirs(args.out, exist_ok=True)
    summary = {
        "n_unique": len(counts),
        "buckets": dict(buckets),
        "streams": stream_rows,
        "n_identity": len(identity),
        "n_disputed": len(disputed),
    }
    _write_json(os.path.join(args.out, "summary.json"), summary)
    _write_jsonl(os.path.join(args.out, "identity.jsonl"), identity)
    _write_jsonl(os.path.join(args.out, "disputed.jsonl"), disputed)
    _write_json(os.path.join(args.out, "rewrite_table.json"), table)
    write_review(args.out, stream_rows, buckets, identity, disputed, len(counts))
    print(
        f"unique={len(counts)} clean={buckets['clean']} "
        f"identity={buckets['identity']} disputed={buckets['disputed']}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
