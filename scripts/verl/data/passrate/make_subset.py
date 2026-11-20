"""Builds the prompt subsets that passrate.sh measures, in the same jsonl format as its input.

Two commands:

  sample  draws a random subset of a source jsonl (instruction, response, source, ...). Every row
          gets a unique_id of the form <src stem>:<row number in the src file>, unless it already
          has one, so a measured prompt can always be traced back to its source row.
  shard   splits a subset into consecutive shards of a fixed size. Each shard becomes one verl
          val-only launch, so a crash loses at most one shard and a relaunch skips finished ones.

The two OMR filters work on the rows whose source is "OMR", the label that marks a row from
nvidia/OpenMathReasoning. They read the prior that omr_join.py writes: for each such row, the
pass_rate_72b_tir that OpenMathReasoning publishes for the problem. A row whose problem text differs
from the dataset's has no published pass rate. Such rows are almost never kept, so dropping them before
the measurement saves GPU time. The prior is keyed by row numbers, so it fits only the
file it was built from; its .meta.json records that file's sha256, and sample refuses any other --src.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prior_problem(prior: str | None, src: str) -> str | None:
    """Why the OMR prior cannot be used with ``src``, or None when it was built from exactly that file."""
    if not prior:
        return "the OMR prior filters need --prior, the jsonl that omr_join.py writes"
    try:
        with open(f"{prior}.meta.json", encoding="utf-8") as handle:
            recorded = json.load(handle)["src_sha256"]
    except (OSError, ValueError, KeyError):
        return f"{prior}.meta.json is missing or has no src_sha256; rebuild the prior with omr_join.py"
    if recorded != file_sha256(src):
        return f"the prior {prior} was built from another file than {src}; its row numbers would not match"
    return None


def read_ids(paths: list[str]) -> set[str]:
    """unique_ids listed in earlier subsets or results, which a new sample must not repeat."""
    ids: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    ids.add(json.loads(line)["unique_id"])
    return ids


def sample(args: argparse.Namespace) -> int:
    stem = os.path.splitext(os.path.basename(args.src))[0]
    prior: dict[int, float | None] = {}
    if args.drop_omr_unmatched or args.drop_omr_prior0:
        problem = prior_problem(args.prior, args.src)
        if problem:
            print(f"make_subset: {problem}", file=sys.stderr)
            return 2
        with open(args.prior, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                prior[int(row["row"])] = row["pass_rate_72b_tir"]
    excluded = read_ids(args.exclude)

    rows, dropped = [], {"excluded": 0, "omr_unmatched": 0, "omr_prior0": 0, "source": 0}
    with open(args.src, encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            row = json.loads(line)
            row.setdefault("unique_id", f"{stem}:{number}")
            if row["unique_id"] in excluded:
                dropped["excluded"] += 1
                continue
            if args.sources and row.get("source") not in args.sources:
                dropped["source"] += 1
                continue
            if row.get("source") == "OMR":
                value = prior.get(number, "absent") if prior else "absent"
                if args.drop_omr_unmatched and value is None:
                    dropped["omr_unmatched"] += 1
                    continue
                if args.drop_omr_prior0 and value == 0.0:
                    dropped["omr_prior0"] += 1
                    continue
            rows.append(row)
    random.Random(args.seed).shuffle(rows)
    if args.count is not None:
        rows = rows[: args.count]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f"{args.out}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)
    meta = {"src": os.path.abspath(args.src), "src_sha256": file_sha256(args.src), "seed": args.seed,
            "count": len(rows), "requested": args.count, "dropped": dropped, "sources": args.sources,
            "drop_omr_unmatched": args.drop_omr_unmatched, "drop_omr_prior0": args.drop_omr_prior0,
            "exclude": [os.path.abspath(p) for p in args.exclude]}
    with open(f"{args.out}.meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=1)
    print(f"make_subset: {len(rows)} rows -> {args.out} (dropped {dropped})")
    return 0


def shard(args: argparse.Namespace) -> int:
    with open(args.subset, encoding="utf-8") as handle:
        rows = [line for line in handle if line.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    count = 0
    for start in range(0, len(rows), args.shard_size):
        path = os.path.join(args.out_dir, f"s{start // args.shard_size:03d}.jsonl")
        chunk = rows[start : start + args.shard_size]
        # An existing shard is kept only when it holds exactly these rows, so a relaunch with the
        # same subset reuses it and a changed subset can never be mixed with old shards.
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                if handle.read() != "".join(chunk):
                    print(f"make_subset: {path} exists with other rows; remove the shard dir first",
                          file=sys.stderr)
                    return 2
        else:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("".join(chunk))
        count += 1
    print(f"make_subset: {len(rows)} rows in {count} shards of up to {args.shard_size} -> {args.out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_subset.py")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=None, help="rows to keep after shuffling; default all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", nargs="*", default=[], help="jsonl files whose unique_ids are skipped")
    p.add_argument("--sources", nargs="*", default=None, help="keep only these values of the source field")
    p.add_argument("--prior", default=None, help="the OMR prior of omr_join.py; needed by the two filters below")
    p.add_argument("--drop-omr-unmatched", action="store_true")
    p.add_argument("--drop-omr-prior0", action="store_true")
    p.set_defaults(func=sample)
    s = sub.add_parser("shard")
    s.add_argument("--subset", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--shard-size", type=int, required=True)
    s.set_defaults(func=shard)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
