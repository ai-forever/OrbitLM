#!/usr/bin/env python3
"""Turns the pool that build_pool.py writes (v1) into the pool the distillation runs used (v2).

    $PY scripts/verl/data/fix_pool.py --pool <v1 prompts.jsonl> --hf-hub <dir> --out <dir> \
        [--flagged <ids.json>]

Three fixes, applied in this order, each recorded per id in <out>/fix_log.json:

1. With ``--flagged``, drop every id it lists. The file is a JSON object ``{"ids": {reason: [id, ...]}}``,
   for example the result of reading v1 by hand. Such a review finds what the filters miss: Russian
   translations of MATH problems, prompts derived from HumanEval, math that needs a figure or states its
   answer, code rows without a statement, remnants of multi-turn dialogs, identity questions the regexes
   missed, doubled text, prompts over 1,024 tokens once the template is applied. Without ``--flagged``
   this step drops nothing.
2. Drop Russian math prompts whose ordered formula-token 4-grams match a problem of the MATH test,
   MATH-500 or AIME 2024/2025 with a Jaccard of 0.4 or more. The exact formula signatures of
   build_pool.py miss translations whose formulas were typeset again. The threshold is low on purpose,
   because a false positive costs only one prompt.
3. Remove the added Python 3 instruction from AceReason code prompts that already carry their own
   output-format instruction, so each keeps exactly one.

The output is <out>/prompts.jsonl with the input's fields and order, and <out>/fix_log.json with the
counts, every dropped or edited id, the shares after the fix and the md5 of the new pool.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

import pandas as pd

# Formula spans: $$...$$, $...$, \(...\) and \[...\].
SPAN = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)
# Formula tokens: LaTeX commands, numbers, single letters and operators.
TOK = re.compile(r"\\[a-zA-Z]+|\d+(?:[.,]\d+)?|[a-zA-Z]|[=+\-*/^<>!|]")
JACCARD_MIN = 0.4
# A prompt with fewer distinct 4-grams says too little to match on.
MIN_GRAMS = 8
# 4-grams that occur in this many eval problems or more are too common to point at one problem.
MAX_POSTINGS = 300
TOP_CANDIDATES = 30

# AceReason rows that already ask for a format of their own.
OWN = re.compile(r"Please place the solution code in the following format|Write Python code to solve the problem|"
                 r"You must use ```python|Please use python programming language only|Solve the problem starting with")
# The two prefixes and two suffixes that build_pool.wrap_statement may have added.
PRE = re.compile(r"^(Solve the following programming problem in Python 3\. Put the complete solution in one ```python code block\.|"
                 r"Write a Python 3 program that solves the problem below\. Give the full code in a single ```python block\.)\n\n")
SUF = re.compile(r"\n\n(Solve this problem in Python 3 and put the complete solution in one ```python code block\.|"
                 r"Write a Python 3 solution for this problem\. Present the whole program in a single ```python code block\.)$")


def toks(text):
    """The set of ordered formula-token 4-grams of a text, or of its numbers when it has no formula."""
    spans = " ".join("".join(g for g in m.groups() if g) for m in SPAN.finditer(text))
    base = spans if spans.strip() else " ".join(re.findall(r"\d+(?:[.,]\d+)?", text))
    seq = TOK.findall(base.replace("{", " ").replace("}", " "))
    return frozenset(tuple(seq[i:i + 4]) for i in range(len(seq) - 3))


def math_evals(hub):
    """(eval name, problem) pairs of the MATH test, MATH-500 and AIME from an offline hub cache."""
    def files(pattern):
        return sorted(glob.glob(f"{hub}/{pattern}", recursive=True))

    evals = []
    for f in files("datasets--EleutherAI--hendrycks_math/snapshots/*/**/test-*.parquet"):
        evals += [("MATH-test", p) for p in pd.read_parquet(f, columns=["problem"])["problem"]]
    for f in (files("datasets--HuggingFaceH4--MATH-500/snapshots/*/**/*.jsonl")
              + files("datasets--HuggingFaceH4--MATH-500/snapshots/*/**/*.parquet")):
        df = pd.read_json(f, lines=True) if f.endswith(".jsonl") else pd.read_parquet(f)
        evals += [("MATH-500", p) for p in df["problem"]]
    for pattern, col in [("datasets--Maxwell-Jia--AIME_2024/snapshots/*/**/*.parquet", "Problem"),
                         ("datasets--Maxwell-Jia--AIME_2024/snapshots/*/**/*.jsonl", "Problem"),
                         ("datasets--math-ai--aime25/snapshots/*/**/*.parquet", "problem"),
                         ("datasets--math-ai--aime25/snapshots/*/**/*.jsonl", "problem")]:
        for f in files(pattern):
            df = pd.read_json(f, lines=True) if f.endswith(".jsonl") else pd.read_parquet(f)
            c = col if col in df.columns else [x for x in df.columns if x.lower() == "problem"][0]
            evals += [("AIME", p) for p in df[c]]
    return evals


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fix_pool.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", required=True, help="prompts.jsonl of build_pool.py (v1)")
    parser.add_argument("--flagged", default=None,
                        help='optional: JSON {"ids": {reason: [id, ...]}} of ids to drop, e.g. from a review by hand')
    parser.add_argument("--hf-hub", required=True, help="offline Hugging Face hub cache with MATH, MATH-500 and AIME")
    parser.add_argument("--out", required=True, help="output dir for prompts.jsonl and fix_log.json")
    args = parser.parse_args(argv)
    out_pool = os.path.join(args.out, "prompts.jsonl")
    if os.path.abspath(out_pool) == os.path.abspath(args.pool):
        sys.exit("fix_pool: --out must not hold the input pool; v1 stays as it is")

    evals = math_evals(args.hf_hub)
    print("eval items", collections.Counter(s for s, _ in evals))
    ev = [(s, p, toks(p)) for s, p in evals]
    inv = collections.defaultdict(set)
    for i, (_, _, t) in enumerate(ev):
        for x in t:
            inv[x].add(i)

    with open(args.pool, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    flag = {}
    if args.flagged:
        with open(args.flagged, encoding="utf-8") as handle:
            flag = json.load(handle)["ids"]
    drop = {}
    for reason, ids in flag.items():
        for i in ids:
            drop.setdefault(i, reason)

    for r in rows:
        if r["domain"] != "math" or r["lang"] != "ru" or r["id"] in drop:
            continue
        t = toks(r["prompt"])
        if len(t) < MIN_GRAMS:
            continue
        cand = collections.Counter()
        for x in t:
            if len(inv[x]) < MAX_POSTINGS:
                for i in inv[x]:
                    cand[i] += 1
        best = max((c / len(t | ev[i][2]) for i, c in cand.most_common(TOP_CANDIDATES)), default=0)
        if best >= JACCARD_MIN:
            drop[r["id"]] = "ru_math_fuzzy_eval_translation"

    unwrapped, out = [], []
    for r in rows:
        if r["id"] in drop:
            continue
        if r["domain"] == "code" and r["source"].startswith("AceReason") and OWN.search(r["prompt"]):
            p = SUF.sub("", PRE.sub("", r["prompt"], count=1), count=1)
            if p != r["prompt"]:
                r = dict(r, prompt=p)
                unwrapped.append(r["id"])
        out.append(r)

    os.makedirs(args.out, exist_ok=True)
    tmp = f"{out_pool}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        for r in out:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_pool)
    cells = collections.Counter((r["domain"], r["lang"], r["mode"]) for r in out)
    domains = collections.Counter(r["domain"] for r in out)
    with open(out_pool, "rb") as handle:
        md5 = hashlib.md5(handle.read()).hexdigest()
    log = {"version": 2, "rows_in": len(rows), "rows_out": len(out),
           "dropped_by_reason": dict(collections.Counter(drop.values())), "dropped_ids": drop,
           "unwrapped_double_instruction": len(unwrapped), "unwrapped_ids": unwrapped,
           "domain_share": {k: round(v / len(out), 4) for k, v in domains.items()},
           "ru_share": round(sum(1 for r in out if r["lang"] == "ru") / len(out), 4),
           "think_share": round(sum(1 for r in out if r["mode"] == "think") / len(out), 4),
           "cells": {"/".join(k): v for k, v in sorted(cells.items())},
           "md5": md5}
    with open(os.path.join(args.out, "fix_log.json"), "w", encoding="utf-8") as handle:
        json.dump(log, handle, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in log.items() if k not in ("dropped_ids", "unwrapped_ids", "cells")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
