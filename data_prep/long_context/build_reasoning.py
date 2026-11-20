"""Builds the acereason and openthoughts3 document stores for the 16K context-extension stage.

The work is split into these subcommands, run in this order:

  scan      Reads every source file once and saves per-row facts to work/reasoning/scan/<stream>/:
            the normalized question hash, the answer hash, whether the <think> block is well
            formed, and the length in characters. For openthoughts3 it also tokenizes each usable
            example to get its exact length. About a fifth of the finished OpenThoughts3 answers
            sit just above the 16,384-token limit, and a character estimate cannot tell those
            apart from the ones just below it.
  select    Decides, over the whole source at once, which rows become documents: at most two
            answers per question, no repeated answers, and for openthoughts3 no question that
            AceReason already has. The decision has to be global, because the answers to one
            question are spread over many files.
  build     Renders and tokenizes the selected rows of each file and writes them to
            work/docs/<stream>. Each input file gets its own document-store part, numbered by the
            file index, so parallel runs never collide and a finished file is never redone.
  recheck   Drops documents whose added tokens are wrong from parts that are already written
            (the first acereason parts were built with the early version of the shared
            library and needed it; see recheck below).
  manifest  Sums the per-file statistics into manifests/build/<stream>.json. build also does
            this after every file, so the manifest is always current.

The first build kept two answers per question (tier r0-2). Raising that to FINAL_CAPS answers
adds new tiers as new parts, without touching the parts that already exist:

  anskey              Hashes every answer as it will be rendered, so that two answers which
                      differ only in whitespace around </think> count as the same answer.
  build --ranks 2:C   Builds the answers ranked 2 to C-1 (tier r2-C), leaving out repeated
                      answers. It also saves the exact token count of every row it tokenized
                      under work/reasoning/count/.
  count --label r0-2  Saves the exact token counts of the rows the first tier selected, so that
                      extend can tell which of them were written and which were dropped.
  extend              Checks those counts against the documents already in the store, then
                      finds the questions that are short of the cap because an answer was too
                      long or a repeat. Their later answers need exact counts too; extend lists
                      them, count --label s2 measures them, and extend runs again (asking for
                      s3 and so on in the rare case that it needs more). The last run picks the
                      backfill answers (tier bC) and predicts every new tier exactly.
  build --tier backfill  Builds tier bC.
  verify              Checks every document of the new tiers against the format rules and
                      against the prediction of extend, and looks for identical documents
                      anywhere in the store.

scan, count and build take --part P --nparts N. A part takes every Nth file of one fixed shuffled
order, so whatever a single process has finished at any moment is a random sample of the
source, and the remaining files can be fanned out to more processes later. Files that are
already finished are skipped.

Usage:
  build_reasoning.py scan     --stream S [--part P --nparts N]
  build_reasoning.py select   --stream S
  build_reasoning.py anskey   --stream S [--part P --nparts N]
  build_reasoning.py build    --stream S [--part P --nparts N] [--limit-files K] [--ranks 0:2 | --tier backfill] [--test]
  build_reasoning.py count    --stream S --label r0-2|sN [--part P --nparts N] [--limit-files K] [--test]
  build_reasoning.py extend   --stream S
  build_reasoning.py verify   --stream S
  build_reasoning.py recheck  --stream S
  build_reasoning.py manifest --stream S
"""
import argparse
import glob
import hashlib
import json
import os
import random
import time

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

import documents
import paths

ROOT = paths.CTX16K_ROOT
WORK = os.path.join(ROOT, "work", "reasoning")
SOURCES = {
    "acereason": os.path.join(paths.RAW_HF, "nvidia__AceReason-1.1-SFT/sft_data.parquet/data-*.arrow"),
    "openthoughts3": os.path.join(paths.RAW_HF, "open-thoughts__OpenThoughts3-1.2M/data/*.parquet"),
}
MAX_TOKENS = 16384
# The first build kept two answers per question. The finished stores keep up to FINAL_CAPS
# answers per question; the extra answers live in their own tiers (see the module docstring).
CAP = 2
FINAL_CAPS = {"acereason": 4, "openthoughts3": 8}
SEED = 20260930
ENCODE_CHUNK = 256

# Median characters per rendered token, measured on 1,000 AceReason and 900 OpenThoughts3
# examples with the OrbitLM tokenizer. AceReason gets no exact token counts before the build,
# because tokenizing its 25B tokens twice would take too long. select uses these ratios only to
# prefer, among the answers to one question, the ones that probably fit into 16,384 tokens.
DOMAINS = {"math": 0, "code": 1, "science": 2}
CHARS_PER_TOKEN = np.array([2.85, 3.55, 3.9, 3.3])

OK, UNCLOSED, MALFORMED, EMPTY, BAD_TURNS, IN_ACEREASON, DUPLICATE, TOO_LONG = range(8)
STATUS_NAMES = ["ok", "unclosed_think", "malformed_think", "empty_question_reasoning_or_answer",
                "bad_turns", "question_in_acereason", "duplicate_answer", "too_long"]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def source_files(stream):
    return sorted(glob.glob(SOURCES[stream]))


def shuffled_order(n):
    # A fixed seed keeps the order identical in every process, which is what lets parts be
    # split and resumed without coordination.
    order = list(range(n))
    random.Random(SEED).shuffle(order)
    return order


def hash64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")


def question_hash(q):
    # Questions are compared after lowercasing and collapsing whitespace, so copies that differ
    # only in spacing or capitalization count as the same question.
    return hash64(" ".join((q or "").lower().split()))


def row_random(files, rows):
    """Returns a fixed pseudo-random 64-bit number per (file, row), used to break ties fairly."""
    x = (files.astype(np.uint64) << np.uint64(32)) | rows.astype(np.uint64)
    x = x ^ np.uint64(SEED)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x


def parse_answer(out):
    """Splits an answer into its reasoning and its final content, or says why it is unusable.

    An answer is usable only when it opens with exactly one <think> and closes it with exactly
    one </think>. Some OpenThoughts3 answers close the reasoning, keep reasoning, and close it a
    second time. Such an answer has no single correct split, and keeping the extra tags inside
    the text would teach the model a broken format, so it is dropped.
    """
    s = (out or "").strip()
    if "</think>" not in s:
        return UNCLOSED, None, None
    if not s.startswith("<think>") or s.count("<think>") != 1 or s.count("</think>") != 1:
        return MALFORMED, None, None
    j = s.index("</think>")
    reasoning = s[len("<think>"):j].strip()
    content = s[j + len("</think>"):].strip()
    if not reasoning or not content:
        return EMPTY, None, None
    return OK, reasoning, content


def render(question, reasoning, content):
    return documents.render_chat([{"role": "user", "content": question},
                                  {"role": "assistant", "content": content, "reasoning": reasoning}])


THINK_OPEN, THINK_CLOSE = documents.THINK_TOKEN_IDS
FIRST_ADDED_ID = documents.FIRST_ADDED_TOKEN_ID

# Rendered texts are tokenized with documents.encode_docs, which keeps the <think> and </think>
# tags that render_chat inserts and removes every other added-token string. render_chat has
# already cleaned the question, reasoning and answer, so the cleaning in encode_docs changes
# nothing here, and the first build, which tokenized the rendered text directly, got the same
# tokens.

# The first acereason parts were built with an early version of the shared library, which
# removed only these three control-token strings from document text.
EARLY_LIBRARY_REMOVED = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
_early_library_kept = None


def old_library_bad(parts):
    """True when the early version of the shared library would have spoiled this example.

    That version removed only EARLY_LIBRARY_REMOVED from the text. Any other added-token
    string, such as a literal <filename>, stayed in the text, became a control token, and the
    document was dropped (by build or by recheck). The current library removes those strings
    instead. Knowing which rows were affected lets extend match the first tier's documents to
    their rows, because the parts built with the early version are missing exactly these rows.
    """
    global _early_library_kept
    if _early_library_kept is None:
        _early_library_kept = [s for s in documents.added_token_strings() if s not in EARLY_LIBRARY_REMOVED]
    return any(s in p for p in parts for s in _early_library_kept)


def special_tokens_ok(doc):
    """True when the only added tokens in a document are one <think>, one </think> and the final EOS.

    This catches both ways the tags can go wrong: removed by over-eager cleaning, or extra
    control tokens left inside the text by cleaning that is too weak.
    """
    special = doc[:-1][doc[:-1] >= FIRST_ADDED_ID]
    return len(special) == 2 and special[0] == THINK_OPEN and special[1] == THINK_CLOSE


def iter_file_batches(stream, path):
    """Yields lists of (question, answer, domain, turns_ok) for consecutive rows of one file."""
    if stream == "acereason":
        # The files are datasets save_to_disk shards, which are Arrow IPC streams. Memory-mapping
        # keeps only the batch being converted in memory, not the whole 500 MB file.
        with pa.memory_map(path) as src:
            for b in ipc.open_stream(src):
                n = b.num_rows
                yield list(zip(b.column("input").to_pylist(), b.column("output").to_pylist(),
                               b.column("category").to_pylist(), [True] * n))
    else:
        for b in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=["conversations", "domain"]):
            rows = []
            for conv, dom in zip(b.column("conversations").to_pylist(), b.column("domain").to_pylist()):
                conv = conv or []
                turns_ok = len(conv) == 2 and conv[0]["from"] == "human" and conv[1]["from"] == "gpt"
                q = conv[0]["value"] if conv else ""
                a = conv[-1]["value"] if conv else ""
                rows.append((q, a, dom, turns_ok))
            yield rows


def row_status(question, out, turns_ok):
    if not turns_ok:
        return BAD_TURNS, None, None
    status, reasoning, content = parse_answer(out)
    if status == OK and not (question or "").strip():
        return EMPTY, None, None
    return status, reasoning, content


def iter_encoded(stream, path, take):
    """Renders and tokenizes the rows of one file where take is True, in row order.

    Yields (row, rendered text, document with EOS, old_library_bad) for each selected row.
    Rows are tokenized in chunks, because one call per row is much slower.
    """
    pending = []

    def flush():
        docs = documents.encode_docs([p[1] for p in pending])
        out = [(row, text, doc, bad) for (row, text, bad), doc in zip(pending, docs)]
        pending.clear()
        return out

    row = 0
    for batch in iter_file_batches(stream, path):
        for q, out, dom, turns_ok in batch:
            if take[row]:
                s, reasoning, content = row_status(q, out, turns_ok)
                assert s == OK, (path, row, s)
                pending.append((row, render(q, reasoning, content), old_library_bad((q, reasoning, content))))
            row += 1
        if len(pending) >= ENCODE_CHUNK:
            yield from flush()
    if pending:
        yield from flush()
    assert row == len(take), f"{path} has {row} rows, the scan saw {len(take)}"


def count_dir(stream, label, test=False):
    return os.path.join(WORK, "test" if test else "", "count", stream, label)


def anskey_dir(stream):
    return os.path.join(WORK, "anskey", stream)


def answer_key(reasoning, content):
    """Hashes an answer exactly as render_chat will write it into the document."""
    return hash64(documents.clean_text(reasoning).strip() + "\0" + documents.clean_text(content).strip())


def answer_repeats(stream, status, rank, qhash):
    """Marks every answer that repeats, as rendered, a better-ranked answer to the same question.

    select compares the raw answers. Two answers that differ only in the whitespace after
    </think>, or in control-token strings that rendering removes, both pass that comparison,
    yet they make identical documents. Of each such group only the best-ranked answer is kept.
    On the documents of the first two tiers, this found two pairs.
    """
    files = source_files(stream)
    key_files = [os.path.join(anskey_dir(stream), f"f{fi:04d}.npy") for fi in range(len(files))]
    missing = [p for p in key_files if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"the answer keys are not finished ({len(missing)} files missing); run anskey first")
    keys = np.concatenate([np.load(p) for p in key_files])
    ranked = np.flatnonzero(status == OK)
    o = ranked[np.lexsort((rank[ranked], keys[ranked], qhash[ranked]))]
    same = (qhash[o][1:] == qhash[o][:-1]) & (keys[o][1:] == keys[o][:-1])
    repeat = np.zeros(len(status), bool)
    repeat[o[1:][same]] = True
    return repeat


def all_qhash(stream):
    files = source_files(stream)
    return np.concatenate([np.load(os.path.join(WORK, "scan", stream, f"f{fi:04d}.npz"))["qhash"] for fi in range(len(files))])


class Counts:
    """Collects the exact length and the added-token facts of each tokenized row of one file.

    ntok is the rendered length without the final EOS, the number the 16,384 limit applies to.
    """

    def __init__(self):
        self.rows, self.ntok, self.special_ok, self.old_bad = [], [], [], []

    def add(self, row, doc, bad):
        self.rows.append(row)
        self.ntok.append(len(doc) - 1)
        self.special_ok.append(special_tokens_ok(doc))
        self.old_bad.append(bad)

    def save(self, path):
        tmp = path + ".tmp.npz"
        np.savez(tmp, rows=np.array(self.rows, np.int32), ntok=np.array(self.ntok, np.int32),
                 special_ok=np.array(self.special_ok, bool), old_bad=np.array(self.old_bad, bool))
        os.replace(tmp, path)


# ---------------------------------------------------------------- scan

# AceReason appends one of these two fixed instructions to its code problems, while
# OpenThoughts3 has the same problems without them. On the pinned downloads, comparing whole
# questions found none of the 125,840 OpenThoughts3 OpenCodeReasoning rows in AceReason, and
# comparing without these instructions found 108,496 of them.
ACEREASON_CODE_SUFFIXES = (
    "Write Python code to solve the problem. Please place the solution code in the following format:\n"
    "```python\n# Your solution code here\n```",
    "Please place the solution code in the following format:\n```python\n# Your solution code here\n```",
)


def strip_code_suffix(q):
    q = (q or "").rstrip()
    for s in ACEREASON_CODE_SUFFIXES:
        if q.endswith(s):
            return q[:-len(s)]
    return q


def acereason_questions():
    """Returns the sorted hashes of every AceReason question, in two forms.

    The first form is the hash the acereason scan stored. The second is the hash of the question
    without AceReason's fixed code instruction, so that an OpenThoughts3 question matches when
    it is the same problem without that instruction. Reading the questions again takes about
    two minutes, so the result is saved and reused.
    """
    cache = os.path.join(WORK, "acereason_questions.u64.npy")
    if os.path.exists(cache):
        return np.load(cache)
    files = source_files("acereason")
    d = os.path.join(WORK, "scan", "acereason")
    scan_files = [os.path.join(d, f"f{i:04d}.npz") for i in range(len(files))]
    missing = [p for p in scan_files if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"the acereason scan is not finished ({len(missing)} files missing); run it first")
    stripped = []
    for path in files:
        with pa.memory_map(path) as src:
            for b in ipc.open_stream(src):
                stripped.extend(question_hash(strip_code_suffix(q)) for q in b.column("input").to_pylist())
    hashes = np.unique(np.concatenate([np.load(p)["qhash"] for p in scan_files] + [np.array(stripped, np.uint64)]))
    np.save(cache + ".tmp.npy", hashes)
    os.replace(cache + ".tmp.npy", cache)
    return hashes


def scan(a):
    files = source_files(a.stream)
    out_dir = os.path.join(WORK, "scan", a.stream)
    os.makedirs(out_dir, exist_ok=True)
    # Only openthoughts3 is tokenized here: it is small, and its exact lengths matter (see the
    # module docstring). Its questions that AceReason already has are dropped anyway, so they
    # are not tokenized.
    exact = a.stream == "openthoughts3"
    ace = set(acereason_questions().tolist()) if exact else None
    mine = shuffled_order(len(files))[a.part::a.nparts]
    if a.limit_files:
        mine = mine[:a.limit_files]
    t_start, done_here = time.time(), 0
    for k, fi in enumerate(mine):
        dst = os.path.join(out_dir, f"f{fi:04d}.npz")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        qh, oh, st, ch, nt, dm = [], [], [], [], [], []
        for batch in iter_file_batches(a.stream, files[fi]):
            texts, where = [], []
            for q, out, dom, turns_ok in batch:
                h = question_hash(q)
                status, reasoning, content = row_status(q, out, turns_ok)
                qh.append(h)
                oh.append(hash64(out or ""))
                st.append(status)
                ch.append(len(q or "") + len(out or ""))
                nt.append(-1)
                dm.append(DOMAINS.get(dom, 3))
                if exact and status == OK and h not in ace:
                    texts.append(render(q, reasoning, content))
                    where.append(len(nt) - 1)
            for i in range(0, len(texts), ENCODE_CHUNK):
                for j, d in zip(where[i:i + ENCODE_CHUNK], documents.encode_docs(texts[i:i + ENCODE_CHUNK])):
                    nt[j] = len(d) - 1
        tmp = os.path.join(out_dir, f"f{fi:04d}.tmp.npz")
        np.savez(tmp, qhash=np.array(qh, np.uint64), ohash=np.array(oh, np.uint64), status=np.array(st, np.int8),
                 chars=np.array(ch, np.int64), ntok=np.array(nt, np.int32), domain=np.array(dm, np.int8))
        os.replace(tmp, dst)
        done_here += 1
        left = len(mine) - k - 1
        eta = (time.time() - t_start) / done_here * left / 60
        log(f"scan {a.stream} part {a.part}/{a.nparts}: file {fi} ({k + 1}/{len(mine)}) {len(st)} rows, "
            f"ok {st.count(OK)}, {time.time() - t0:.0f}s, about {eta:.0f} min left")
    log(f"scan {a.stream} part {a.part}/{a.nparts}: DONE")


# ---------------------------------------------------------------- select

def select_path(stream):
    return os.path.join(WORK, f"{stream}_select.npz")


def select(a):
    files = source_files(a.stream)
    scan_dir = os.path.join(WORK, "scan", a.stream)
    cols = {k: [] for k in ("qhash", "ohash", "status", "chars", "ntok", "domain")}
    file_start = [0]
    for fi in range(len(files)):
        p = os.path.join(scan_dir, f"f{fi:04d}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"the scan is not finished: {p} is missing")
        z = np.load(p)
        for k in cols:
            cols[k].append(z[k])
        file_start.append(file_start[-1] + len(z["status"]))
    c = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(c["status"])
    fidx = np.repeat(np.arange(len(files)), np.diff(file_start))
    rows = np.arange(n) - np.array(file_start[:-1])[fidx]
    rnd = row_random(fidx, rows)
    status = c["status"].copy()
    summary = {"stream": a.stream, "files": len(files), "rows": int(n),
               "unique_questions": int(len(np.unique(c["qhash"])))}

    if a.stream == "openthoughts3":
        in_ace = np.isin(c["qhash"], acereason_questions())
        summary["rows_whose_question_is_in_acereason"] = int(in_ace.sum())
        summary["questions_in_acereason"] = int(len(np.unique(c["qhash"][in_ace])))
        status[(status == OK) & in_ace] = IN_ACEREASON
        assert not np.any((status == OK) & (c["ntok"] < 0)), "openthoughts3 rows without exact token counts"
    status[(status == OK) & (c["ntok"] > MAX_TOKENS)] = TOO_LONG

    # A repeated answer to the same question is dropped; which copy stays is decided by the fixed
    # per-row random number, so the choice does not depend on file order.
    ok = np.flatnonzero(status == OK)
    o = ok[np.lexsort((rnd[ok], c["ohash"][ok], c["qhash"][ok]))]
    same = (c["qhash"][o][1:] == c["qhash"][o][:-1]) & (c["ohash"][o][1:] == c["ohash"][o][:-1])
    status[o[1:][same]] = DUPLICATE

    # Every question's remaining answers are ranked. Answers that probably fit into 16,384
    # tokens come first, so that a question does not waste one of its two places on an answer
    # the build will drop. Inside each group the order is the fixed per-row random number.
    # Exact counts are used where scan made them; elsewhere the length is estimated.
    est = np.where(c["ntok"] >= 0, c["ntok"], c["chars"] / CHARS_PER_TOKEN[c["domain"]]).astype(np.float32)
    ok = np.flatnonzero(status == OK)
    likely_long = est[ok] > MAX_TOKENS
    o = ok[np.lexsort((rnd[ok], likely_long, c["qhash"][ok]))]
    q = c["qhash"][o]
    starts = np.r_[True, q[1:] != q[:-1]]
    group = np.cumsum(starts) - 1
    first = np.flatnonzero(starts)
    sizes = np.diff(np.r_[first, len(o)])
    rank = np.full(n, -1, np.int32)
    rank[o] = np.arange(len(o)) - first[group]
    ncand = np.zeros(n, np.int32)
    ncand[o] = sizes[group]

    summary["status_counts"] = {STATUS_NAMES[s]: int((status == s).sum()) for s in range(len(STATUS_NAMES))}
    summary["questions_with_usable_answers"] = int(len(first))
    hist = np.bincount(np.minimum(sizes, 17))
    summary["usable_answers_per_question"] = {("17+" if i == 17 else str(i)): int(v) for i, v in enumerate(hist) if v}
    summary["estimate_note"] = ("token numbers below are exact for openthoughts3; for acereason they are estimated from "
                                "characters, and build reports the measured tokens-to-estimate ratio")
    summary["by_cap"] = {}
    for cap in (1, 2, 3, 4, 8, 16, 10 ** 9):
        m = (status == OK) & (rank < cap)
        summary["by_cap"]["all" if cap == 10 ** 9 else str(cap)] = {
            "docs": int(m.sum()), "est_tokens": int(est[m].sum()),
            "est_likely_too_long": int((m & (est > MAX_TOKENS)).sum())}
    summary["cap"] = CAP
    tmp = select_path(a.stream) + ".tmp.npz"
    np.savez(tmp, status=status, rank=rank, ncand=ncand, est=est, file_start=np.array(file_start, np.int64))
    os.replace(tmp, select_path(a.stream))
    json.dump(summary, open(os.path.join(WORK, f"{a.stream}_select.json"), "w"), indent=1)
    log(f"select {a.stream}: " + json.dumps(summary))


# ---------------------------------------------------------------- build

BACKFILL = 2  # the tier code extend gives to backfill answers (0 is tier r0-2, 1 is tier r2-C)


def build(a):
    files = source_files(a.stream)
    z = np.load(select_path(a.stream))
    status, rank, ncand, est, file_start = z["status"], z["rank"], z["ncand"], z["est"], z["file_start"]
    if a.tier == "backfill":
        # The backfill answers all rank at or above the final cap, so their parts are numbered
        # from cap * 1000, next to the parts of the rank tiers and never overlapping them.
        cap = FINAL_CAPS[a.stream]
        x = np.load(extend_path(a.stream))
        if not x["final"]:
            raise SystemExit("extend has not made its final selection yet; run the count stage it asked for, then extend again")
        tier, want_ntok = x["tier"], x["ntok"]
        lo, hi, tag = cap, None, f"b{cap}"
    else:
        lo, hi = (int(x) for x in a.ranks.split(":"))
        tag = f"r{lo}-{hi}"
        repeat = answer_repeats(a.stream, status, rank, all_qhash(a.stream))
    base = os.path.join(WORK, "test") if a.test else os.path.join(ROOT, "work")
    docs_dir = os.path.join(base, "docs", a.stream)
    done_dir = os.path.join(WORK, "test" if a.test else "", "done", a.stream, tag)
    os.makedirs(done_dir, exist_ok=True)
    # A rank tier also saves the exact length of every row it tokenized, including the rows it
    # drops. extend needs those lengths to find the questions that lost a place.
    counts = None if hi is None else count_dir(a.stream, tag, a.test)
    if counts:
        os.makedirs(counts, exist_ok=True)
    mine = shuffled_order(len(files))[a.part::a.nparts]
    if a.limit_files:
        mine = mine[:a.limit_files]
    t_start, done_here = time.time(), 0
    for k, fi in enumerate(mine):
        marker = os.path.join(done_dir, f"f{fi:04d}.json")
        if os.path.exists(marker):
            continue
        # The part number is the file index, shifted by the lowest rank of the tier, so that a
        # later run over higher ranks (for example --ranks 2:4) writes new parts instead of
        # overwriting these.
        pnum = lo * 1000 + fi
        # A process killed in the middle of a file may have left a half-written part behind;
        # the file is redone from scratch, so those pieces are removed first.
        for stale in glob.glob(os.path.join(docs_dir, f"part_{pnum:04d}_*")):
            os.remove(stale)
        t0 = time.time()
        s0, s1 = int(file_start[fi]), int(file_start[fi + 1])
        st, rk, nc = status[s0:s1], rank[s0:s1], ncand[s0:s1]
        if hi is None:
            take = tier[s0:s1] == BACKFILL
        else:
            take = (st == OK) & (rk >= lo) & (rk < hi)
        writer = documents.DocStoreWriter(docs_dir, pnum)
        fs = {"file": os.path.basename(files[fi]), "file_index": fi, "rows": s1 - s0,
              "status_counts": {STATUS_NAMES[s]: int((st == s).sum()) for s in range(len(STATUS_NAMES))},
              "selected": int(take.sum()), "selected_est_tokens": float(est[s0:s1][take].sum()),
              "too_long": 0, "too_long_with_unused_answers": 0, "bad_special_tokens": 0,
              "kept_est_tokens": 0.0, "examples": []}
        if hi is None:
            fs["length_differs_from_extend"] = 0
        else:
            fs["over_cap"] = int(((st == OK) & (rk >= hi)).sum())
            fs["below_rank_range"] = int(((st == OK) & (rk < lo)).sum())
            fs["answer_repeats"] = 0
        measured = Counts()
        for j, text, d, bad in iter_encoded(a.stream, files[fi], take):
            if len(fs["examples"]) < 3:
                fs["examples"].append(text[:300])
            if counts:
                measured.add(j, d, bad)
            elif len(d) - 1 != want_ntok[s0 + j]:
                fs["length_differs_from_extend"] += 1
            # A repeated answer is still measured above, because extend expects exact counts
            # for every row of the rank range, but it is not written.
            if hi is not None and repeat[s0 + j]:
                fs["answer_repeats"] += 1
                continue
            if not special_tokens_ok(d):
                fs["bad_special_tokens"] += 1
                continue
            if len(d) - 1 > MAX_TOKENS:
                fs["too_long"] += 1
                # This question had more usable answers than it got places, so the one that
                # turned out too long could have been replaced; counted to size that loss.
                if hi is not None and nc[j] > hi:
                    fs["too_long_with_unused_answers"] += 1
                continue
            writer.add(d)
            fs["kept_est_tokens"] += float(est[s0 + j])
        fs.update(writer.close())
        if counts:
            measured.save(os.path.join(counts, f"f{fi:04d}.npz"))
        fs["seconds"] = round(time.time() - t0, 1)
        tmp = marker + ".tmp"
        json.dump(fs, open(tmp, "w"), indent=1)
        os.replace(tmp, marker)
        done_here += 1
        left = len(mine) - k - 1
        eta = (time.time() - t_start) / done_here * left / 3600
        log(f"build {a.stream} {tag} part {a.part}/{a.nparts}: file {fi} ({k + 1}/{len(mine)}) selected {fs['selected']}, "
            f"too long {fs['too_long']}, docs {fs['docs']}, tokens {fs['tokens'] / 1e6:.1f}M, {fs['seconds']:.0f}s, "
            f"about {eta:.1f} h left")
        if not a.test:
            write_manifest(a.stream)
    log(f"build {a.stream} {tag} part {a.part}/{a.nparts}: DONE")


# ---------------------------------------------------------------- count

def stage_path(stream, label):
    return os.path.join(WORK, f"{stream}_count_{label}.npz")


def stage_labels(stream):
    """Returns the labels of the extra count stages extend has asked for so far: s2, s3, ..."""
    labels = []
    while os.path.exists(stage_path(stream, f"s{len(labels) + 2}")):
        labels.append(f"s{len(labels) + 2}")
    return labels


def count(a):
    """Saves exact token counts for the rows that one label names, without writing documents.

    The label r0-2 names the rows the first tier selected. The labels s2, s3, ... name the rows
    that extend listed: the later answers of the questions that are still short of the cap.
    """
    files = source_files(a.stream)
    z = np.load(select_path(a.stream))
    status, rank, file_start = z["status"], z["rank"], z["file_start"]
    if a.label == "r0-2":
        mask = (status == OK) & (rank >= 0) & (rank < CAP)
    else:
        mask = np.load(stage_path(a.stream, a.label))["mask"]
    out_dir = count_dir(a.stream, a.label, a.test)
    os.makedirs(out_dir, exist_ok=True)
    mine = shuffled_order(len(files))[a.part::a.nparts]
    if a.limit_files:
        mine = mine[:a.limit_files]
    t_start, done_here = time.time(), 0
    for k, fi in enumerate(mine):
        dst = os.path.join(out_dir, f"f{fi:04d}.npz")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        s0, s1 = int(file_start[fi]), int(file_start[fi + 1])
        take = mask[s0:s1]
        measured = Counts()
        # Most files have no row in the s2 list, and reading such a file would be wasted time.
        if take.any():
            for j, text, d, bad in iter_encoded(a.stream, files[fi], take):
                measured.add(j, d, bad)
        measured.save(dst)
        done_here += 1
        left = len(mine) - k - 1
        eta = (time.time() - t_start) / done_here * left / 60
        log(f"count {a.stream} {a.label} part {a.part}/{a.nparts}: file {fi} ({k + 1}/{len(mine)}) rows {len(measured.rows)}, "
            f"tokens {sum(measured.ntok) / 1e6:.1f}M, {time.time() - t0:.0f}s, about {eta:.0f} min left")
    log(f"count {a.stream} {a.label} part {a.part}/{a.nparts}: DONE")


# ---------------------------------------------------------------- anskey

def anskey(a):
    """Saves, for every usable row, the hash of its answer as rendered (see answer_repeats).

    Rows that select did not mark as usable get 0. No tokenizing is needed, only reading.
    """
    files = source_files(a.stream)
    z = np.load(select_path(a.stream))
    status, file_start = z["status"], z["file_start"]
    out_dir = anskey_dir(a.stream)
    os.makedirs(out_dir, exist_ok=True)
    mine = shuffled_order(len(files))[a.part::a.nparts]
    if a.limit_files:
        mine = mine[:a.limit_files]
    for k, fi in enumerate(mine):
        dst = os.path.join(out_dir, f"f{fi:04d}.npy")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        s0, s1 = int(file_start[fi]), int(file_start[fi + 1])
        usable = status[s0:s1] == OK
        keys = np.zeros(s1 - s0, np.uint64)
        row = 0
        for batch in iter_file_batches(a.stream, files[fi]):
            for q, out, dom, turns_ok in batch:
                if usable[row]:
                    s, reasoning, content = row_status(q, out, turns_ok)
                    assert s == OK, (fi, row, s)
                    keys[row] = answer_key(reasoning, content)
                row += 1
        assert row == s1 - s0, f"file {fi} has {row} rows, the scan saw {s1 - s0}"
        tmp = os.path.join(out_dir, f"f{fi:04d}.tmp.npy")
        np.save(tmp, keys)
        os.replace(tmp, dst)
        log(f"anskey {a.stream} part {a.part}/{a.nparts}: file {fi} ({k + 1}/{len(mine)}) {time.time() - t0:.0f}s")
    log(f"anskey {a.stream} part {a.part}/{a.nparts}: DONE")


# ---------------------------------------------------------------- extend

def extend_path(stream):
    return os.path.join(WORK, f"{stream}_extend.npz")


def stored_doc_lengths(stream, pnum):
    """Returns the lengths of the documents in one part of the real store, in the order written."""
    docs_dir = os.path.join(ROOT, "work", "docs", stream)
    idx = sorted(glob.glob(os.path.join(docs_dir, f"part_{pnum:04d}_*.idx.npy")))
    return np.concatenate([np.diff(np.load(p)) for p in idx]) if idx else np.zeros(0, np.int64)


def extend(a):
    """Chooses the answers that raise a stream to FINAL_CAPS answers per question.

    The existing tier r0-2 stays as it is. For each question, the answers it already has in
    that tier count against the cap, and the remaining places are filled with its next answers
    in the order select ranked them, skipping every answer that is too long. Answers ranked
    below the cap form tier r2-C, and answers ranked at or above it form the backfill tier bC.
    Those replace places that an answer too long for 16,384 tokens took in the earlier tiers.

    An answer that repeats, as rendered, a better-ranked answer to the same question is never
    added (see answer_repeats); the question's next answer takes its place instead.

    Every decision uses exact token counts. Rows ranked below the cap are all counted before
    this step. Later rows are counted only for the questions that are still short of the cap,
    so a run that finds such rows without counts lists them for the next count stage
    (count --label s2, then s3 if needed), and the run after that stage continues.
    """
    stream, cap = a.stream, FINAL_CAPS[a.stream]
    files = source_files(stream)
    nf = len(files)
    z = np.load(select_path(stream))
    status, rank, est, file_start = z["status"], z["rank"], z["est"], z["file_start"]
    n = len(status)
    qhash = all_qhash(stream)
    assert len(qhash) == n
    repeat = answer_repeats(stream, status, rank, qhash)
    ntok = np.full(n, -1, np.int32)
    special_ok = np.zeros(n, bool)
    old_bad = np.zeros(n, bool)
    stages = stage_labels(stream)
    for label in ["r0-2", f"r2-{cap}"] + stages:
        d = count_dir(stream, label)
        for fi in range(nf):
            p = os.path.join(d, f"f{fi:04d}.npz")
            if not os.path.exists(p):
                raise SystemExit(f"the {label} counts are not finished: {p} is missing")
            c = np.load(p)
            g = int(file_start[fi]) + c["rows"].astype(np.int64)
            ntok[g], special_ok[g], old_bad[g] = c["ntok"], c["special_ok"], c["old_bad"]
    ok = status == OK
    measured = ntok >= 0
    assert not (ok & (rank < cap) & ~measured).any(), f"some rows ranked below {cap} have no exact count"
    for label in stages:
        assert measured[np.load(stage_path(stream, label))["mask"]].all()
    fits = measured & (ntok <= MAX_TOKENS) & special_ok

    # Tier r0-2 is already written, and extend has to know which of its rows made it into the
    # store. Those are the rows that fit. In parts built with the older library, the rows that
    # library spoiled are missing as well. Whichever of the two rules reproduces every stored
    # document length of a part exactly is the one that part was built with.
    first = ok & (rank < CAP)
    kept0 = np.zeros(n, bool)
    alignment = {"new_library": 0, "old_library": 0, "either": 0}
    unmatched = []
    for fi in range(nf):
        s0, s1 = int(file_start[fi]), int(file_start[fi + 1])
        rows = s0 + np.flatnonzero(first[s0:s1])
        lens = stored_doc_lengths(stream, fi)
        by_new = rows[fits[rows]]
        by_old = by_new[~old_bad[by_new]]
        m_new = np.array_equal(ntok[by_new].astype(np.int64) + 1, lens)
        m_old = np.array_equal(ntok[by_old].astype(np.int64) + 1, lens)
        if not (m_new or m_old):
            unmatched.append({"file": fi, "stored_docs": len(lens), "fitting_rows": len(by_new)})
            continue
        alignment["either" if m_new and m_old else "new_library" if m_new else "old_library"] += 1
        kept0[by_new if m_new else by_old] = True
    if unmatched:
        raise SystemExit(f"tier r0-2 does not match its rows in {len(unmatched)} files: {unmatched[:5]}")

    # Each question's remaining places are filled in rank order. have is the number of
    # documents the question already has in tier r0-2, and pos is the place of an answer
    # among the question's remaining answers that fit and are not repeats.
    uq, qi = np.unique(qhash, return_inverse=True)
    have = np.bincount(qi[kept0], minlength=len(uq))
    usable = fits & ~repeat
    cand = np.flatnonzero(ok & (rank >= CAP) & usable)
    o = cand[np.lexsort((rank[cand], qi[cand]))]
    g = qi[o]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]) if len(o) else np.zeros(0, np.int64)
    pos = np.arange(len(o)) - np.repeat(starts, np.diff(np.r_[starts, len(o)]))
    chosen = o[have[g] + pos < cap]
    tier = np.full(n, -1, np.int8)
    tier[kept0] = 0
    tier[chosen] = np.where(rank[chosen] < cap, 1, BACKFILL)
    # Every answer ranked 2 to cap-1 that fits and is not a repeat always gets a place, because
    # a question has at most two documents in tier r0-2. So tier r2-C is exactly what
    # build --ranks 2:C keeps.
    rank_tier = ok & (rank >= CAP) & (rank < cap)
    assert np.array_equal(tier == 1, rank_tier & usable)

    per_q = np.bincount(qi[tier >= 0], minlength=len(uq))
    short = per_q < cap
    need = ok & (rank >= cap) & ~measured & short[qi]
    final = not need.any()
    next_label = f"s{len(stages) + 2}"
    if not final:
        tmp = stage_path(stream, next_label) + ".tmp.npz"
        np.savez(tmp, mask=need)
        os.replace(tmp, stage_path(stream, next_label))

    def total(mask):
        return {"docs": int(mask.sum()), "tokens": int((ntok[mask].astype(np.int64) + 1).sum())}

    usable_q = np.bincount(qi[ok], minlength=len(uq)) > 0
    hist = np.bincount(per_q[usable_q], minlength=cap + 1)
    rank_tier_repeats = rank_tier & fits & repeat
    summary = {
        "stream": stream, "cap": cap, "final": final,
        "tier_r0-2_alignment_by_file": alignment,
        "tier_r0-2_rows_not_in_store": {
            "too_long": int((first & (ntok > MAX_TOKENS)).sum()),
            "control_tokens_under_old_library": int((first & ~kept0 & (ntok <= MAX_TOKENS)).sum())},
        "tiers": {"r0-2": total(tier == 0), f"r2-{cap}": total(tier == 1), f"b{cap}": total(tier == BACKFILL)},
        "all_tiers": total(tier >= 0),
        "rows_ranked_2_to_cap_too_long": int((rank_tier & measured & (ntok > MAX_TOKENS)).sum()),
        "answer_repeats": {
            "all_usable_rows": int(repeat.sum()),
            "both_copies_in_tier_r0-2_already": int((repeat & kept0).sum()),
            "left_out_of_tier_r2-cap": int(rank_tier_repeats.sum()),
            "files_of_those": sorted(set((np.searchsorted(file_start, np.flatnonzero(rank_tier_repeats), side="right") - 1).tolist())),
            "left_out_of_backfill": int((ok & (rank >= cap) & fits & repeat).sum())},
        "questions_with_usable_answers": int(usable_q.sum()),
        "questions_by_answers_kept": {str(i): int(v) for i, v in enumerate(hist)},
        "questions_with_backfill": int(len(np.unique(qi[tier == BACKFILL]))),
        "questions_short_of_cap_with_no_fitting_answer_left": int((usable_q & short).sum()) if final else None,
    }
    for label in stages:
        m = np.load(stage_path(stream, label))["mask"]
        summary[f"{label}_rows_counted"] = int(m.sum())
        summary[f"{label}_rows_too_long"] = int((m & (ntok > MAX_TOKENS)).sum())
    if not final:
        summary[f"{next_label}_rows_to_count"] = int(need.sum())
        summary[f"{next_label}_estimated_tokens"] = int(est[need].sum())
    tmp = extend_path(stream) + ".tmp.npz"
    np.savez(tmp, tier=tier, ntok=ntok, repeat=repeat, final=np.array(final))
    os.replace(tmp, extend_path(stream))
    json.dump(summary, open(os.path.join(WORK, f"{stream}_extend.json"), "w"), indent=1)
    log(f"extend {stream}: " + json.dumps(summary))
    if not final:
        log(f"extend {stream}: run count --stream {stream} --label {next_label}, then extend again")


# ---------------------------------------------------------------- verify

def check_docs(toks, lens):
    """Returns the list of format problems in a run of documents with the given lengths."""
    problems = []
    ends = np.cumsum(lens)
    if len(lens) and lens.max() > MAX_TOKENS + 1:
        problems.append(f"{int((lens > MAX_TOKENS + 1).sum())} documents longer than {MAX_TOKENS + 1} tokens")
    # Each document ends with EOS and has no other EOS exactly when the EOS positions are the
    # last positions of the documents.
    if not np.array_equal(np.flatnonzero(toks == documents.EOS), ends - 1):
        problems.append("EOS positions differ from the document ends")
    tag_pos = {}
    for name, tid in (("<think>", THINK_OPEN), ("</think>", THINK_CLOSE)):
        p = np.flatnonzero(toks == tid)
        # One tag per document means one position per document, and each falls in its own one.
        if len(p) != len(lens) or not np.array_equal(np.searchsorted(ends, p, side="right"), np.arange(len(lens))):
            problems.append(f"{name} does not occur exactly once in every document")
        tag_pos[name] = p
    if len(tag_pos["<think>"]) == len(tag_pos["</think>"]) and not np.all(tag_pos["<think>"] < tag_pos["</think>"]):
        problems.append("</think> before <think>")
    other = (toks >= FIRST_ADDED_ID) & (toks != documents.EOS) & (toks != THINK_OPEN) & (toks != THINK_CLOSE)
    if other.any():
        problems.append(f"{int(other.sum())} other added tokens")
    return problems


def doc_hashes(toks, lens):
    """Returns a 64-bit hash of each document in a run of documents with the given lengths."""
    raw = memoryview(toks.view(np.uint8))
    ends = np.cumsum(lens) * 4
    starts = ends - np.asarray(lens) * 4
    return np.array([int.from_bytes(hashlib.blake2b(raw[s:e], digest_size=8).digest(), "little")
                     for s, e in zip(starts.tolist(), ends.tolist())], np.uint64)


def verify(a):
    """Checks every document of the new tiers of one stream, file by file.

    Each document must end with its only EOS, be at most 16,385 tokens long with that EOS,
    and contain exactly one <think> and one </think>, in that order, and no other added token.
    Each part must also hold exactly the documents extend predicted, with the predicted lengths,
    which shows that no answer was lost or added between extend and build. Finally, no
    document of a new tier may be identical to any other document in the store.
    """
    stream, cap = a.stream, FINAL_CAPS[a.stream]
    files = source_files(stream)
    x = np.load(extend_path(stream))
    if not x["final"]:
        raise SystemExit("extend has not made its final selection yet")
    tier, ntok = x["tier"], x["ntok"]
    file_start = np.load(select_path(stream))["file_start"]
    docs_dir = os.path.join(ROOT, "work", "docs", stream)
    report = {"stream": stream, "checked": time.strftime("%Y-%m-%d %H:%M:%S"), "tiers": {}}
    hashes, where = [], []
    for code, tag, lo in ((1, f"r2-{cap}", CAP), (BACKFILL, f"b{cap}", cap)):
        # A tier that extend left empty is never built (openthoughts3 needs no backfill,
        # because its lengths were exact from the start), so there is nothing to check.
        if not (tier == code).any() and not os.path.isdir(os.path.join(WORK, "done", stream, tag)):
            report["tiers"][tag] = {"docs": 0, "tokens": 0, "ok": True, "note": "extend chose no answers for this tier"}
            continue
        res = {"files": 0, "docs": 0, "tokens": 0, "max_doc_tokens": 0, "problems": []}
        marker_docs = marker_tokens = 0
        for fi in range(len(files)):
            s0, s1 = int(file_start[fi]), int(file_start[fi + 1])
            want = ntok[s0:s1][tier[s0:s1] == code].astype(np.int64) + 1
            marker = os.path.join(WORK, "done", stream, tag, f"f{fi:04d}.json")
            if not os.path.exists(marker):
                res["problems"].append(f"file {fi}: not built")
                continue
            m = json.load(open(marker))
            marker_docs += m["docs"]
            marker_tokens += m["tokens"]
            names = sorted(p[:-4] for p in glob.glob(os.path.join(docs_dir, f"part_{lo * 1000 + fi:04d}_*.u32")))
            toks = [np.fromfile(p + ".u32", dtype=np.uint32) for p in names]
            lens = [np.diff(np.load(p + ".idx.npy")) for p in names]
            toks = np.concatenate(toks) if toks else np.zeros(0, np.uint32)
            lens = np.concatenate(lens) if lens else np.zeros(0, np.int64)
            if lens.sum() != len(toks):
                res["problems"].append(f"file {fi}: offsets do not cover the tokens")
                continue
            if not np.array_equal(lens, want):
                res["problems"].append(f"file {fi}: {len(lens)} documents stored, {len(want)} predicted, or lengths differ")
            res["problems"] += [f"file {fi}: {p}" for p in check_docs(toks, lens)]
            hashes.append(doc_hashes(toks, lens))
            where += [(tag, fi, i) for i in range(len(lens))]
            res["files"] += 1
            res["docs"] += len(lens)
            res["tokens"] += int(len(toks))
            res["max_doc_tokens"] = max(res["max_doc_tokens"], int(lens.max()) if len(lens) else 0)
        if (marker_docs, marker_tokens) != (res["docs"], res["tokens"]):
            res["problems"].append(f"markers say {marker_docs} docs and {marker_tokens} tokens")
        res["ok"] = not res["problems"]
        report["tiers"][tag] = res
        log(f"verify {stream} {tag}: " + json.dumps(res)[:2000])

    # Tier r0-2 is hashed too, so that a new document repeating one of its documents is caught.
    for fi in range(len(files)):
        for p in sorted(glob.glob(os.path.join(docs_dir, f"part_{fi:04d}_*.u32"))):
            lens = np.diff(np.load(p[:-4] + ".idx.npy"))
            hashes.append(doc_hashes(np.fromfile(p, dtype=np.uint32), lens))
            where += [("r0-2", fi, i) for i in range(len(lens))]
    h = np.concatenate(hashes) if hashes else np.zeros(0, np.uint64)
    u, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    groups = {}
    for j in np.flatnonzero(cnt[inv] > 1):
        groups.setdefault(int(inv[j]), []).append(where[j])
    new_groups = [g for g in groups.values() if any(t != "r0-2" for t, _, _ in g)]
    report["identical_documents"] = {
        "docs_hashed": int(len(h)),
        "groups_involving_new_tiers": len(new_groups),
        "groups_only_in_tier_r0-2": len(groups) - len(new_groups),
        "examples": [[f"{t} file {fi} doc {i}" for t, fi, i in g] for g in list(groups.values())[:10]]}
    report["ok"] = all(t["ok"] for t in report["tiers"].values()) and not new_groups
    log(f"verify {stream}: identical documents " + json.dumps(report["identical_documents"]) + f", ok {report['ok']}")
    json.dump(report, open(os.path.join(WORK, f"{stream}_verify.json"), "w"), indent=1)
    write_manifest(stream)


# ---------------------------------------------------------------- recheck

def recheck(a):
    """Drops documents with wrong added tokens from an already written store and fixes the markers.

    The first acereason build ran with the early version of the shared library, whose cleaning
    removed only three control-token strings. A literal string such as <filename> inside a
    question or answer was therefore encoded as a control token. The build now drops such
    documents itself; this step applies the same rule to parts written before that. Running it
    again changes nothing.
    """
    docs_dir = os.path.join(ROOT, "work", "docs", a.stream)
    total_dropped = 0
    for name in documents.docstore_files(docs_dir):
        # Only the first tier (parts below 1000) was ever built with the older library. The
        # later tiers drop such documents while they are built, and verify checks them.
        if int(name.split("_")[1]) >= 1000:
            continue
        toks = np.fromfile(os.path.join(docs_dir, name + ".u32"), dtype=np.uint32)
        offs = np.load(os.path.join(docs_dir, name + ".idx.npy"))
        keep = np.array([special_tokens_ok(toks[offs[i]:offs[i + 1]]) for i in range(len(offs) - 1)], bool)
        if keep.all():
            continue
        lens = np.diff(offs)
        dropped, dropped_tokens = int((~keep).sum()), int(lens[~keep].sum())
        new_offs = np.zeros(int(keep.sum()) + 1, np.int64)
        new_offs[1:] = np.cumsum(lens[keep])
        new_toks = np.concatenate([toks[offs[i]:offs[i + 1]] for i in np.flatnonzero(keep)])
        base = os.path.join(docs_dir, name)
        new_toks.tofile(base + ".u32.tmp")
        np.save(base + ".idx.tmp.npy", new_offs)
        os.replace(base + ".u32.tmp", base + ".u32")
        os.replace(base + ".idx.tmp.npy", base + ".idx.npy")
        # The part number encodes the rank range and the file (see build), which locates the marker.
        pnum = int(name.split("_")[1])
        lo, fi = pnum // 1000, pnum % 1000
        for marker in glob.glob(os.path.join(WORK, "done", a.stream, f"r{lo}-*", f"f{fi:04d}.json")):
            m = json.load(open(marker))
            m["docs"] -= dropped
            m["tokens"] -= dropped_tokens
            m["bad_special_tokens"] = m.get("bad_special_tokens", 0) + dropped
            json.dump(m, open(marker + ".tmp", "w"), indent=1)
            os.replace(marker + ".tmp", marker)
        total_dropped += dropped
        log(f"recheck {a.stream}: {name} dropped {dropped} documents ({dropped_tokens} tokens)")
    write_manifest(a.stream)
    log(f"recheck {a.stream}: DONE, {total_dropped} documents dropped")


# ---------------------------------------------------------------- manifest

def write_manifest(stream):
    files = source_files(stream)
    summary_path = os.path.join(WORK, f"{stream}_select.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    order = shuffled_order(len(files))
    pos = {fi: i for i, fi in enumerate(order)}
    tiers = {}

    def tier_order(tdir):
        # Rank tiers come in rank order (r0-2, r2-4, ...), and the backfill tier comes last.
        name = os.path.basename(tdir)
        return (name[0] == "b", int(name[1:].split("-")[0]))

    for tdir in sorted(glob.glob(os.path.join(WORK, "done", stream, "[rb]*")), key=tier_order):
        markers = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(tdir, "f*.json")))]
        markers.sort(key=lambda m: pos[m["file_index"]])
        tot = {"files_done": len(markers), "files_total": len(files)}
        # Markers written before a statistic existed simply lack it, so missing keys count as zero.
        for key in ("rows", "over_cap", "below_rank_range", "selected", "too_long", "too_long_with_unused_answers",
                    "bad_special_tokens", "answer_repeats", "length_differs_from_extend", "docs", "tokens",
                    "selected_est_tokens",
                    "kept_est_tokens", "seconds"):
            tot[key] = sum(m.get(key, 0) for m in markers)
        tot["drops_in_done_files"] = {k: sum(m["status_counts"][k] for m in markers)
                                      for k in STATUS_NAMES if k != "ok"}
        tot["drops_in_done_files"]["over_cap"] = tot["over_cap"]
        tot["drops_in_done_files"]["too_long_after_tokenizing"] = tot["too_long"]
        tot["drops_in_done_files"]["bad_special_tokens"] = tot["bad_special_tokens"]
        tot["drops_in_done_files"]["answer_repeats"] = tot["answer_repeats"]
        if markers:
            tot["seconds_per_file"] = round(tot["seconds"] / len(markers), 1)
            # AceReason lengths were only estimated before the build; the measured ratio turns
            # the estimate for the whole source into a projection of what a full build yields.
            if tot["kept_est_tokens"] and os.path.basename(tdir).startswith("r"):
                ratio = tot["tokens"] / tot["kept_est_tokens"]
                tot["tokens_per_estimated_token"] = round(ratio, 4)
                # A tier covers ranks LO..HI-1, so its share of the source is the estimate for
                # cap HI minus the estimate for cap LO.
                lo, hi = os.path.basename(tdir)[1:].split("-")
                by_cap = summary.get("by_cap", {})
                est_all = None
                if hi in by_cap and (lo == "0" or lo in by_cap):
                    est_all = by_cap[hi]["est_tokens"] - (by_cap[lo]["est_tokens"] if lo != "0" else 0)
                if est_all and tot["selected_est_tokens"]:
                    keep = tot["kept_est_tokens"] / tot["selected_est_tokens"]
                    tot["projected_tokens_full_source"] = int(est_all * keep * ratio)
        tot["examples_first_300_chars"] = [m["examples"][0] for m in markers[:3] if m["examples"]]
        tiers[os.path.basename(tdir)] = tot
    cap = FINAL_CAPS[stream]
    extras = {}
    for key, name in (("extend_summary", f"{stream}_extend.json"), ("verify", f"{stream}_verify.json")):
        p = os.path.join(WORK, name)
        if os.path.exists(p):
            extras[key] = json.load(open(p))
    manifest = {"stream": stream, "docstore": os.path.join(ROOT, "work", "docs", stream),
                "rules": {"max_tokens": MAX_TOKENS, "answers_per_question": cap,
                          "tiers": {"r0-2": "the first build: answers ranked 0 and 1, parts 0000-0999",
                                    f"r2-{cap}": f"answers ranked 2 to {cap - 1} that fit, parts 2000-2999",
                                    f"b{cap}": (f"backfill, parts {cap}000-{cap}999: for a question that lost places in "
                                                f"the earlier tiers to answers longer than {MAX_TOKENS} tokens, its next "
                                                f"answers that fit, in rank order, until it has {cap} (exact token counts)")},
                          "question_normalization": "lowercase, whitespace collapsed, blake2b-64",
                          "duplicate_answers": "an answer repeated exactly for the same question is kept once",
                          "openthoughts3_vs_acereason": ("an openthoughts3 question is dropped when it matches an AceReason "
                                                         "question as is or without AceReason's fixed code instruction"),
                          "think": "exactly one <think> at the start and one </think>; reasoning and answer non-empty",
                          "file_order_seed": SEED},
                "totals_all_tiers": {"docs": sum(t["docs"] for t in tiers.values()),
                                     "tokens": sum(t["tokens"] for t in tiers.values()),
                                     "complete": all(t["files_done"] == t["files_total"] for t in tiers.values())},
                "select_summary": summary, "build": tiers, **extras, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = os.path.join(ROOT, "manifests", "build", f"{stream}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.{os.getpid()}.tmp"
    json.dump(manifest, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, out)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "select", "anskey", "build", "count", "extend", "verify", "recheck", "manifest"])
    ap.add_argument("--stream", required=True, choices=sorted(SOURCES))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--ranks", default=f"0:{CAP}", help="answer ranks to build, LO:HI; the default keeps two per question")
    ap.add_argument("--tier", choices=["backfill"], help="build the backfill tier chosen by extend instead of a rank range")
    ap.add_argument("--label", help="for count: which rows to count, r0-2 or a stage that extend asked for (s2, s3, ...)")
    ap.add_argument("--test", action="store_true", help="write to work/reasoning/test instead of the real store")
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(a)
    elif a.cmd == "select":
        select(a)
    elif a.cmd == "build":
        build(a)
    elif a.cmd == "anskey":
        anskey(a)
    elif a.cmd == "count":
        if not (a.label == "r0-2" or (a.label or "").startswith("s") and a.label[1:].isdigit()):
            ap.error("count needs --label r0-2 or --label sN")
        count(a)
    elif a.cmd == "extend":
        extend(a)
    elif a.cmd == "verify":
        verify(a)
    elif a.cmd == "recheck":
        recheck(a)
    else:
        print(json.dumps(write_manifest(a.stream), indent=1, ensure_ascii=False))
