"""Renders multiple-choice items into the bench_mcq document store.

Every item becomes one plain-text document in the evaluation style of bench_train:
  "Question: ...\\nA. ...\\nB. ...\\nAnswer: B"

Source: allenai/qasc train, a question with eight choices, written as part 0 of the store.

Usage:
  build_bench_mcq.py qasc [--limit K] [--overwrite]
  build_bench_mcq.py manifest
--limit reads only the first K items of the input file, for tests.
"""
import argparse
import hashlib
import json
import os
import time
from collections import defaultdict

import paths
from build_bench_math import Output
from build_bench_train import Drop, iter_rows, labelled_mcq

ROOT = paths.CTX16K_ROOT
QASC = os.path.join(paths.RAW_HF, "allenai__qasc", "data", "train-00000-of-00001.parquet")
QASC_PART = 0


def write_manifest(a, source, part, stats, out, t0):
    totals = out.close()
    manifest = {
        "builder": "build_bench_mcq.py",
        "stream": "bench_mcq",
        "source": source,
        "store": a.out,
        "part": part,
        "args": vars(a),
        "examples_read": stats["read"],
        "drops": {k: v for k, v in stats.items() if k.startswith("drop_")},
        "docs_written": totals["docs"],
        "tokens_written": totals["tokens"],
        "stats": dict(stats),
        "first_300_chars_of_3_examples": [t[:300] for t in out.examples],
        "minutes": round((time.time() - t0) / 60, 2),
    }
    # Each part writes its own manifest; the "manifest" command joins them into
    # manifests/build/bench_mcq.json.
    mdir = os.path.join(a.work, "manifests_mcq")
    os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, f"{source}_part_{part:04d}.json")
    json.dump(manifest, open(path + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    print("DONE", json.dumps({k: manifest[k] for k in ("source", "part", "examples_read", "drops", "docs_written", "tokens_written")}), flush=True)


def dedup(key, seen, st):
    """Returns True the first time a prompt key is seen, and counts the duplicate otherwise."""
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=12).digest()
    if h in seen:
        st["drop_duplicate"] += 1
        return False
    seen.add(h)
    return True


# ---------------------------------------------------------------- QASC

def cmd_qasc(a):
    t0 = time.time()
    part = QASC_PART
    out = Output(a.out, part, a.overwrite)
    st, seen = defaultdict(int), set()
    for r in iter_rows(QASC, a.limit):
        st["read"] += 1
        try:
            text, key = labelled_mcq(r["question"], r["choices"], r["answerKey"])
        except Drop as e:
            st[f"drop_{e}"] += 1
            continue
        if dedup(key, seen, st):
            out.add(text, st)
    write_manifest(a, "qasc", part, st, out, t0)


# ---------------------------------------------------------------- stream manifest

def cmd_manifest(a):
    """Joins the per-part manifests into manifests/build/bench_mcq.json."""
    mdir = os.path.join(a.work, "manifests_mcq")
    parts = [json.load(open(os.path.join(mdir, f))) for f in sorted(os.listdir(mdir)) if f.endswith(".json")]
    per_source = defaultdict(lambda: defaultdict(int))
    for m in parts:
        ps = per_source[m["source"]]
        ps["parts"] += 1
        for k in ("examples_read", "docs_written", "tokens_written"):
            ps[k] += m[k]
        for k, v in m["drops"].items():
            ps[k] += v
    examples = []
    for m in parts:
        examples += m["first_300_chars_of_3_examples"][:2]
    manifest = {
        "builder": "build_bench_mcq.py",
        "stream": "bench_mcq",
        "store": a.out,
        "examples_read": sum(m["examples_read"] for m in parts),
        "drops": {k: sum(m["drops"].get(k, 0) for m in parts) for k in sorted({k for m in parts for k in m["drops"]})},
        "docs_written": sum(m["docs_written"] for m in parts),
        "tokens_written": sum(m["tokens_written"] for m in parts),
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "first_300_chars_of_3_examples": examples[:3],
        "parts": parts,
    }
    path = os.path.join(ROOT, "manifests", "build", "bench_mcq.json") if a.manifest is None else a.manifest
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(manifest, open(path + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    print(json.dumps({k: manifest[k] for k in ("docs_written", "tokens_written", "per_source")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["qasc", "manifest"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "docs", "bench_mcq"))
    ap.add_argument("--work", default=os.path.join(ROOT, "work", "bench2"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    {"qasc": cmd_qasc, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    main()
