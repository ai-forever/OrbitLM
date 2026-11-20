"""Renders math question/answer sources into the bench_math and bench_math_dm document stores.

Every item becomes one plain-text document in the evaluation style, not a chat:
  "Question: <problem>\\nAnswer: <solution or answer>"

Sources, their stores and their part numbers (each process writes its own part):
  omr       nvidia/OpenMathReasoning, cot split: up to 4 solutions per problem,
            bench_math, parts 0-99.
  bigmath   SynthLabsAI/Big-Math-RL-Verified: problem and final answer, bench_math, part 200.
  deepmind  deepmind/math_dataset, train splits: question and short answer,
            bench_math_dm, parts 100-199.
DeepMind has a store of its own, so that its share of the mix is set in the yaml instead of
following from how many rows were rendered.

OpenMathReasoning has about ten solutions per problem, spread over all 144 files, so choosing
solutions per problem needs a global view. It therefore runs in three steps:
  omr-scan --part P --nparts N   reads its files and records, for every row, the problem hash,
                                 whether the solution's boxed answer matches the expected
                                 answer, and a hash of the solution text (work/bench2/omr_scan/);
  omr-select                     picks the solutions of every problem over all files
                                 (work/bench2/omr_select/);
  omr --part P --nparts N        renders the picked rows of its files.
The first solution of a problem is a matching one when any matches, and otherwise a random but
reproducible one. Up to three more are added, and those must match the expected answer and
differ in text from the solutions already picked (--max-solutions sets the total).
Only the final solution after </think> is kept; the reasoning trace is dropped.

bigmath must run after omr-select, because it drops problems that OpenMathReasoning already
has (same normalized text), using the problem hashes that omr-select writes.

Usage:
  build_bench_math.py omr-scan --part P --nparts N [--limit K]
  build_bench_math.py omr-select
  build_bench_math.py omr --part P --nparts N [--limit K] [--overwrite]
  build_bench_math.py deepmind --part P --nparts N [--per-module ROWS] [--limit K] [--overwrite]
  build_bench_math.py bigmath [--limit K] [--overwrite]
  build_bench_math.py manifest
--part/--nparts split the source's input files (each part takes every Nth file).
--limit reads only the first K rows of every file, for tests; --out and --work redirect the
store and the intermediate files, also for tests.
"""
import argparse
import codecs
import glob
import hashlib
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

import documents
import paths

ROOT = paths.CTX16K_ROOT
RAW = paths.RAW_HF
MAX_TOKENS = 16384
OMR_GLOB = os.path.join(RAW, "nvidia__OpenMathReasoning", "data", "cot-*.parquet")
DM_GLOB = os.path.join(RAW, "deepmind__math_dataset", "*", "train", "*.parquet")
BIGMATH = os.path.join(RAW, "SynthLabsAI__Big-Math-RL-Verified", "data", "train-00000-of-00001.parquet")
PART_BASE = {"omr": 0, "deepmind": 100, "bigmath": 200}
STREAM = {"omr": "bench_math", "bigmath": "bench_math", "deepmind": "bench_math_dm"}


def norm(s):
    return " ".join((s or "").lower().split())


def h64(s):
    """A 64-bit hash of a string, used as the exact-dedup key of a normalized problem."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


# ---------------------------------------------------------------- answer checking for OpenMathReasoning

def last_boxed(s):
    """Returns the content of the last \\boxed{...} in s, with nested braces, or None."""
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        c = s[j]
        depth += 1 if c == "{" else -1 if c == "}" else 0
        j += 1
    return s[i + 7:j - 1] if depth == 0 else None


def answer_key(s):
    """Normalizes a final answer so that trivially different spellings compare equal.

    It unwraps \\boxed{} and math delimiters and removes spacing commands, spaces, \\left and
    \\right, degree marks and a trailing period. The check only decides which solution of a
    problem is preferred, so a missed match costs nothing but the preference.
    """
    s = (s or "").strip()
    b = last_boxed(s)
    if b is not None:
        s = b
    for a, z in (("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$"), ("$", "$")):
        if s.startswith(a) and s.endswith(z) and len(s) >= len(a) + len(z):
            s = s[len(a):len(s) - len(z)]
    for t in ("\\left", "\\right", "\\displaystyle", "\\!", "\\,", "\\;", "\\:", "\\ ", " ", "\n"):
        s = s.replace(t, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("^{\\circ}", "")
    return s.rstrip(".")


def final_solution(generated):
    """Returns the text after the last </think>, or None when the reasoning never closed."""
    k = generated.rfind("</think>")
    if k < 0:
        return None
    return generated[k + len("</think>"):].strip() or None


# ---------------------------------------------------------------- helpers

def iter_batches(path, columns, limit, batch_size=2000):
    """Yields column dicts of a parquet file in small batches, stopping after limit rows."""
    pf = pq.ParquetFile(path)
    n = 0
    for b in pf.iter_batches(batch_size=batch_size, columns=columns):
        d = b.to_pydict()
        if limit and n + b.num_rows > limit:
            d = {k: v[:limit - n] for k, v in d.items()}
        yield d
        n += len(d[columns[0]])
        if limit and n >= limit:
            return


class Output:
    """Tokenizes rendered texts in chunks, drops over-long ones and writes the rest to the store."""

    def __init__(self, store, part, overwrite):
        prefix = f"part_{part:04d}_"
        os.makedirs(store, exist_ok=True)
        mine = [f for f in os.listdir(store) if f.startswith(prefix)]
        if mine and not overwrite:
            raise SystemExit(f"{store} already has {len(mine)} files of part {part}; pass --overwrite to rebuild them")
        for f in mine:
            os.remove(os.path.join(store, f))
        self.writer = documents.DocStoreWriter(store, part)
        self.pending = []
        self.examples = []

    def add(self, text, st):
        self.pending.append((text, st))
        if len(self.pending) >= 4000:
            self.flush()

    def flush(self):
        texts = [t for t, _ in self.pending]
        for (text, st), toks in zip(self.pending, documents.encode_docs(texts)):
            if len(toks) - 1 > MAX_TOKENS:
                st["drop_too_long"] += 1
                continue
            self.writer.add(toks)
            st["docs"] += 1
            st["tokens"] += len(toks)
            st["max_doc_tokens"] = max(st["max_doc_tokens"], len(toks))
            if len(self.examples) < 3:
                self.examples.append(text)
        self.pending = []

    def close(self):
        self.flush()
        return self.writer.close()


def write_manifest(a, source, part, stats, out, t0):
    totals = out.close()
    manifest = {
        "builder": "build_bench_math.py",
        "stream": STREAM[source],
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
    # Each part writes its own manifest next to the others; the "manifest" command joins them
    # into one manifest per stream in manifests/build/.
    mdir = os.path.join(a.work, "manifests")
    os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, f"{source}_part_{part:04d}.json")
    json.dump(manifest, open(path + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    print("DONE", json.dumps({k: manifest[k] for k in ("source", "part", "examples_read", "drops", "docs_written", "tokens_written")}), flush=True)


# ---------------------------------------------------------------- OpenMathReasoning

def omr_files():
    return sorted(glob.glob(OMR_GLOB))


def cmd_omr_scan(a):
    sdir = os.path.join(a.work, "omr_scan")
    os.makedirs(sdir, exist_ok=True)
    for path in omr_files()[a.part::a.nparts]:
        t0 = time.time()
        hashes, scores, ties = [], [], []
        cols = ["problem", "expected_answer", "generated_solution"]
        for d in iter_batches(path, cols, a.limit):
            for p, ea, g in zip(d["problem"], d["expected_answer"], d["generated_solution"]):
                hashes.append(h64(norm(p)))
                fin = final_solution(g or "") if (p or "").strip() else None
                if fin is None:
                    scores.append(0)  # unusable: no problem text, or no final solution
                    ties.append(0)
                    continue
                boxed = last_boxed(fin)
                match = boxed is not None and answer_key(boxed) == answer_key(ea)
                scores.append(2 if match else 1)
                # The tie-break among equally good solutions is a hash of the solution itself,
                # which picks a random but reproducible one.
                ties.append(h64(fin))
        out = os.path.join(sdir, os.path.basename(path).replace(".parquet", ".npz"))
        np.savez(out + ".tmp.npz", hash=np.array(hashes, np.uint64), score=np.array(scores, np.uint8),
                 tie=np.array(ties, np.uint64))
        os.replace(out + ".tmp.npz", out)
        sc = np.array(scores)
        print(f"{time.strftime('%H:%M:%S')} scanned {os.path.basename(path)}: rows {len(sc)}, "
              f"match {int((sc == 2).sum())}, no match {int((sc == 1).sum())}, unusable {int((sc == 0).sum())}, "
              f"{time.time() - t0:.0f}s", flush=True)


def cmd_omr_select(a):
    sdir = os.path.join(a.work, "omr_scan")
    files = omr_files()
    missing = [f for f in files if not os.path.exists(os.path.join(sdir, os.path.basename(f).replace(".parquet", ".npz")))]
    if missing:
        raise SystemExit(f"{len(missing)} files not scanned yet, e.g. {missing[0]}")
    H, S, T, F, I = [], [], [], [], []
    for fi, f in enumerate(files):
        z = np.load(os.path.join(sdir, os.path.basename(f).replace(".parquet", ".npz")))
        n = len(z["hash"])
        H.append(z["hash"]); S.append(z["score"]); T.append(z["tie"])
        F.append(np.full(n, fi, np.int32)); I.append(np.arange(n, dtype=np.int64))
    H, S, T, F, I = (np.concatenate(x) for x in (H, S, T, F, I))
    # Sort by problem, then best score first, then the tie-break hash. The first row of every
    # problem is then its first pick: a matching solution when there is one.
    order = np.lexsort((T, -S.astype(np.int16), H))
    Hs, Ss, Ts = H[order], S[order], T[order]
    first = np.ones(len(order), bool)
    first[1:] = Hs[1:] != Hs[:-1]
    # Inside a problem, identical solution texts have the same hash and sit next to each other,
    # so a row whose hash differs from the row before is the first copy of its text.
    distinct = first.copy()
    distinct[1:] |= Ts[1:] != Ts[:-1]
    # The matching distinct solutions of a problem come first in its rows, so numbering them
    # inside the problem and keeping numbers below the limit takes the first pick plus the
    # extra ones. A problem without any matching solution keeps only its first row.
    match = (Ss == 2) & distinct
    before = np.cumsum(match) - match
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    rank = before - before[group_start]
    keep = (match & (rank < a.max_solutions)) | (first & (Ss == 1))
    pick = order[keep]
    per_problem = np.bincount(np.cumsum(first)[keep] - 1, minlength=int(first.sum()))
    stats = {"rows": int(len(H)), "problems": int(first.sum()), "picked": int(len(pick)),
             "max_solutions": a.max_solutions,
             "picked_with_matching_answer": int((S[pick] == 2).sum()),
             "problems_by_solutions_picked": {str(k): int((per_problem == k).sum()) for k in range(a.max_solutions + 1)},
             "problems_without_usable_solution": int((first & (Ss == 0)).sum())}
    odir = os.path.join(a.work, "omr_select")
    os.makedirs(odir, exist_ok=True)
    for fi, f in enumerate(files):
        rows = np.sort(I[pick][F[pick] == fi])
        np.save(os.path.join(odir, os.path.basename(f).replace(".parquet", ".npy")), rows)
    # Big-Math drops problems whose normalized text OpenMathReasoning already has. Every
    # problem counts here, including the ones without a usable solution.
    np.save(os.path.join(a.work, "omr_problem_hashes.npy"), np.unique(H))
    json.dump(stats, open(os.path.join(odir, "select_stats.json"), "w"), indent=1)
    print("DONE", json.dumps(stats), flush=True)


def cmd_omr(a):
    t0 = time.time()
    part = PART_BASE["omr"] + a.part
    out = Output(a.out, part, a.overwrite)
    st = defaultdict(int)
    odir = os.path.join(a.work, "omr_select")
    for path in omr_files()[a.part::a.nparts]:
        picked = np.load(os.path.join(odir, os.path.basename(path).replace(".parquet", ".npy")))
        picked = set(picked.tolist())
        row = 0
        for d in iter_batches(path, ["problem", "generated_solution"], a.limit):
            for p, g in zip(d["problem"], d["generated_solution"]):
                st["read"] += 1
                if row in picked:
                    st["picked"] += 1
                    out.add(f"Question: {p.strip()}\nAnswer: {final_solution(g)}", st)
                row += 1
        st["drop_not_picked"] = st["read"] - st["picked"]
        print(f"{time.strftime('%H:%M:%S')} omr {os.path.basename(path)}: {dict(st)}", flush=True)
    write_manifest(a, "omr", part, st, out, t0)


# ---------------------------------------------------------------- DeepMind mathematics dataset

def dm_decode(s):
    """Turns the stored text back into the question: the parquet holds Python bytes literals
    such as "b'Solve 2*x = 4 for x.\\\\n'", so the b'' wrapper and the escapes are undone."""
    s = s or ""
    if len(s) >= 3 and s[0] == "b" and s[1] in "'\"" and s[-1] == s[1]:
        s = codecs.escape_decode(s[2:-1].encode("utf-8"))[0].decode("utf-8")
    return s.strip()


def dm_keep_mask(path, limit, cap):
    """Returns which rows of one module file to keep: the first copy of every question, and
    at most cap of them when a cap is given.

    The mask is computed from a compact array of 64-bit question hashes before any rendering.
    A Python set of the hashes of 28 modules read side by side would take several GB.
    """
    hashes = []
    for d in iter_batches(path, ["question"], limit, batch_size=20000):
        hashes.extend(h64(norm(dm_decode(q))) for q in d["question"])
    h = np.array(hashes, np.uint64)
    keep = np.zeros(len(h), bool)
    keep[np.unique(h, return_index=True)[1]] = True
    unique = int(keep.sum())
    if cap:
        keep &= np.cumsum(keep) <= cap
    return keep, len(h) - unique, unique - int(keep.sum())


def cmd_deepmind(a):
    t0 = time.time()
    part = PART_BASE["deepmind"] + a.part
    files = sorted(glob.glob(DM_GLOB))[a.part::a.nparts]
    out = Output(a.out, part, a.overwrite)
    st = defaultdict(int)
    rng = random.Random(16384 + a.part)
    masks, kept = {}, {}
    for f in files:
        mod = f.split(os.sep)[-3]
        masks[mod], dup, over = dm_keep_mask(f, a.limit, a.per_module)
        st["drop_duplicate"] += dup
        st["drop_over_module_cap"] += over
        kept[mod] = int(masks[mod].sum())
        print(f"{time.strftime('%H:%M:%S')} deepmind {mod}: rows {len(masks[mod])}, duplicates {dup}, "
              f"over the cap {over}, kept {kept[mod]}", flush=True)
    # All modules of the part are read side by side, one batch from each in turn, and every
    # round is shuffled before writing. A packed block then mixes many modules instead of
    # holding hundreds of copies of one question template.
    readers = {f.split(os.sep)[-3]: iter_batches(f, ["question", "answer"], a.limit, batch_size=1000) for f in files}
    pos = defaultdict(int)
    while readers:
        pool = []
        for mod in list(readers):
            d = next(readers[mod], None)
            if d is None:
                del readers[mod]
                continue
            keep = masks[mod]
            for q, ans in zip(d["question"], d["answer"]):
                st["read"] += 1
                i = pos[mod]
                pos[mod] += 1
                if not keep[i]:
                    continue
                q, ans = dm_decode(q), dm_decode(ans)
                if not q or not ans:
                    st["drop_empty_field"] += 1
                    continue
                pool.append(f"Question: {q}\nAnswer: {ans}")
        rng.shuffle(pool)
        for text in pool:
            out.add(text, st)
    st["modules"] = len(files)
    st["kept_per_module"] = kept
    write_manifest(a, "deepmind", part, st, out, t0)


# ---------------------------------------------------------------- Big-Math

def cmd_bigmath(a):
    import re
    t0 = time.time()
    part = PART_BASE["bigmath"]
    omr = np.load(os.path.join(a.work, "omr_problem_hashes.npy"))
    out = Output(a.out, part, a.overwrite)
    st = defaultdict(int)
    seen = set()
    # About 1,400 answers were escaped twice and read "\\\\frac{1}{5}". A double backslash
    # directly before a letter is never a LaTeX line break, so it is turned back into one.
    double_bs = re.compile(r"\\\\(?=[A-Za-z])")
    for d in iter_batches(BIGMATH, ["problem", "answer"], a.limit):
        for p, ans in zip(d["problem"], d["answer"]):
            st["read"] += 1
            p, ans = (p or "").strip(), (ans or "").strip()
            if not p or not ans:
                st["drop_empty_field"] += 1
                continue
            k = h64(norm(p))
            if k in seen:
                st["drop_duplicate"] += 1
                continue
            seen.add(k)
            i = np.searchsorted(omr, np.uint64(k))
            if i < len(omr) and omr[i] == np.uint64(k):
                st["drop_in_openmathreasoning"] += 1
                continue
            fixed = double_bs.sub(r"\\", ans)
            st["answers_unescaped"] += int(fixed != ans)
            out.add(f"Question: {p}\nAnswer: {fixed}", st)
    write_manifest(a, "bigmath", part, st, out, t0)


# ---------------------------------------------------------------- stream manifest

def cmd_manifest(a):
    """Joins the per-part manifests into manifests/build/bench_math.json and bench_math_dm.json."""
    mdir = os.path.join(a.work, "manifests")
    parts = [json.load(open(os.path.join(mdir, f))) for f in sorted(os.listdir(mdir)) if f.endswith(".json")]
    select = os.path.join(a.work, "omr_select", "select_stats.json")
    for stream in sorted(set(STREAM.values())):
        mine = [m for m in parts if STREAM[m["source"]] == stream]
        per_source = defaultdict(lambda: defaultdict(int))
        for m in mine:
            ps = per_source[m["source"]]
            ps["parts"] += 1
            for k in ("examples_read", "docs_written", "tokens_written"):
                ps[k] += m[k]
            for k, v in m["drops"].items():
                ps[k] += v
        examples = []
        for m in mine:
            examples += m["first_300_chars_of_3_examples"][:3 if len(per_source) == 1 else 2]
        manifest = {
            "builder": "build_bench_math.py",
            "stream": stream,
            "store": sorted({m["store"] for m in mine}),
            "examples_read": sum(m["examples_read"] for m in mine),
            "drops": {k: sum(m["drops"].get(k, 0) for m in mine) for k in sorted({k for m in mine for k in m["drops"]})},
            "docs_written": sum(m["docs_written"] for m in mine),
            "tokens_written": sum(m["tokens_written"] for m in mine),
            "per_source": {k: dict(v) for k, v in per_source.items()},
            "first_300_chars_of_3_examples": examples[:3],
            "parts": mine,
        }
        if "omr" in per_source and os.path.exists(select):
            manifest["omr_selection"] = json.load(open(select))
        path = os.path.join(a.manifest_dir, f"{stream}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(manifest, open(path + ".tmp", "w"), indent=1, ensure_ascii=False)
        os.replace(path + ".tmp", path)
        print(stream, json.dumps({k: manifest[k] for k in ("docs_written", "tokens_written", "per_source")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["omr-scan", "omr-select", "omr", "deepmind", "bigmath", "manifest"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-module", type=int, default=0, help="deepmind: keep at most this many rows per module")
    ap.add_argument("--max-solutions", type=int, default=4, help="omr-select: solutions kept per problem")
    ap.add_argument("--out", default=None, help="the store; by default work/docs/bench_math, or bench_math_dm for deepmind")
    ap.add_argument("--work", default=os.path.join(ROOT, "work", "bench2"))
    ap.add_argument("--manifest-dir", default=os.path.join(ROOT, "manifests", "build"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(ROOT, "work", "docs", STREAM.get(a.cmd, "bench_math"))
    {"omr-scan": cmd_omr_scan, "omr-select": cmd_omr_select, "omr": cmd_omr, "deepmind": cmd_deepmind,
     "bigmath": cmd_bigmath, "manifest": cmd_manifest}[a.cmd](a)


if __name__ == "__main__":
    main()
