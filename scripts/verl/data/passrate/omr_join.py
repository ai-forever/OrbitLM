"""Builds the OpenMathReasoning prior that the OMR filters of make_subset.py read.

    $PY -I omr_join.py --omr-dir <OpenMathReasoning>/data --src <math jsonl> --out <prior.jsonl> [--workers 24]

A row of ``--src`` that comes from nvidia/OpenMathReasoning carries the source label "OMR". For every
such row, the tool looks up the problem's ``pass_rate_72b_tir`` in the ``cot`` parquet files of
nvidia/OpenMathReasoning. The lookup tries the
exact problem text first and then the text with whitespace collapsed. The output has one line per OMR
row: ``{"row": <0-based line number in --src>, "match": "exact"|"normalized"|"missing",
"pass_rate_72b_tir": <float or null>}``.

``<out>.meta.json`` records the sha256 of ``--src``, because the prior is keyed by row numbers and fits
no other file. make_subset.py refuses the prior for any other source. The same file holds a summary:
how many rows matched, and a histogram of the prior values.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool

import pyarrow.parquet as pq


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_one(path):
    """Distinct pass_rate_72b_tir values per problem text in one parquet file."""
    table = pq.read_table(path, columns=["problem", "pass_rate_72b_tir"])
    seen = {}
    for problem, rate in zip(table.column("problem").to_pylist(), table.column("pass_rate_72b_tir").to_pylist()):
        seen.setdefault(problem, set()).add(rate)
    return seen, table.num_rows


def read_rates(omr_dir, workers):
    """Every problem of the cot split with its prior, the number of parquet rows and of conflicting problems."""
    files = sorted(glob.glob(os.path.join(omr_dir, "cot-*.parquet")))
    if not files:
        raise SystemExit(f"omr_join: no cot-*.parquet files in {omr_dir}")
    rates, rows = {}, 0
    with Pool(workers) as pool:
        for seen, n in pool.imap_unordered(read_one, files):
            rows += n
            for problem, values in seen.items():
                rates.setdefault(problem, set()).update(values)
    # The same problem text can come from several forum sources, each with its own 72B pass rate. The
    # prior is then the mean of the distinct numeric values. "n/a" counts only when nothing else is known.
    merged, conflicts = {}, 0
    for problem, values in rates.items():
        numbers = sorted(float(v) for v in values if v not in (None, "n/a"))
        if len(numbers) > 1:
            conflicts += 1
        merged[problem] = (sum(numbers) / len(numbers)) if numbers else "n/a"
    print(f"OMR cot: {len(files)} files, {rows} rows, {len(merged)} unique problems, "
          f"{conflicts} problems with several distinct rates")
    return merged, rows, conflicts


def main(argv=None):
    parser = argparse.ArgumentParser(prog="omr_join.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--omr-dir", required=True, help="the data/ dir of nvidia/OpenMathReasoning (cot-*.parquet)")
    parser.add_argument("--src", required=True, help="the math jsonl whose OMR rows get a prior")
    parser.add_argument("--out", required=True, help="the prior jsonl to write; <out>.meta.json goes next to it")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args(argv)

    rates, cot_rows, conflicts = read_rates(args.omr_dir, args.workers)
    by_norm = {}
    for problem, rate in rates.items():
        by_norm.setdefault(norm(problem), rate)
    prefixes = {norm(problem)[:60] for problem in rates}

    exact = normed = missing = prefix_hits = 0
    buckets = collections.Counter()
    out_rows = []
    with open(args.src, encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            row = json.loads(line)
            if row.get("source") != "OMR":
                continue
            question = row["instruction"]
            if question in rates:
                rate, how = rates[question], "exact"
                exact += 1
            elif norm(question) in by_norm:
                rate, how = by_norm[norm(question)], "normalized"
                normed += 1
            else:
                rate, how = None, "missing"
                missing += 1
                # A 60-character prefix that still matches hints at a problem whose text was edited.
                prefix_hits += norm(question)[:60] in prefixes
            value = None
            if rate not in (None, "n/a"):
                try:
                    value = float(rate)
                except ValueError:
                    value = None
            if value is None:
                buckets["n/a" if how != "missing" else "unmatched"] += 1
            else:
                low = min(int(value * 10), 9) / 10
                buckets[f"{low:.1f}-{low + 0.1:.1f}"] += 1
                if value == 0.0:
                    buckets["exactly 0"] += 1
                if value == 1.0:
                    buckets["exactly 1"] += 1
            out_rows.append({"row": number, "match": how, "pass_rate_72b_tir": value})

    summary = {"omr_rows_in_src": exact + normed + missing, "exact": exact, "normalized": normed,
               "missing": missing, "omr_unique_problems": len(rates),
               "missing_but_60char_prefix_found": prefix_hits, "omr_cot_rows": cot_rows,
               "problems_with_several_rates": conflicts, "buckets": dict(sorted(buckets.items()))}
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for item in out_rows:
            handle.write(json.dumps(item) + "\n")
    meta = {"src": os.path.abspath(args.src), "src_sha256": file_sha256(args.src),
            "omr_dir": os.path.abspath(args.omr_dir), "summary": summary}
    with open(f"{args.out}.meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
