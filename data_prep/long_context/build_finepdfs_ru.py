"""Builds the finepdfs_ru document store: long, clean Russian PDF documents from FinePDFs rus_Cyrl.

Every document goes through the filters below in this order, and the first filter that fails is
its drop reason. The cheap metadata filters run before the text is touched, and tokenizing runs
last, so the expensive steps only see documents that can still be kept.

  metadata:  language is rus_Cyrl with a confident document score; most pages are Russian;
             the Llama-3.2 token_count is long enough to possibly reach 8,192 Qwen3 tokens;
             the mean FineWeb-Edu score is high enough (the main quality and size knob).
  text:      Cyrillic share of letters; characters from no expected script (broken fonts,
             private-use glyphs); Latin-1 letters typical of a wrong code page; bureaucratic
             documents (work programs, orders, regulations) by their title page and by the
             density of officialese; repeated lines; markdown or HTML tables; clause numbers
             destroyed by the IP anonymizer; zstd ratio (loops and templates compress too
             well); digit share.
  exact dedup by a hash of the lowercased, whitespace-collapsed text, within the process.
  length:    at least 8,192 Qwen3 tokens after tokenizing.

The thresholds were chosen on probes of four files; the numbers behind each one are in NOTES
below, which the build manifest repeats.

Each input file is written as its own doc-store part (part id = the file's index in the sorted
list of 36 files). That keeps part ids stable whatever --nparts is, so parallel processes never
collide, and a restarted process skips the files it already finished.

Usage:
  build_finepdfs_ru.py --part P --nparts N [--max-files K] [--out-name NAME]
  build_finepdfs_ru.py --manifest [--out-name NAME]   (sums the per-file stats into manifests/build)
"""
import argparse
import glob
import hashlib
import json
import os
import time
from collections import Counter

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

import documents
import paths
from finepdfs_features import text_features

ROOT = paths.CTX16K_ROOT
SRC = os.path.join(paths.RAW_HF, "HuggingFaceFW__finepdfs/data/rus_Cyrl/train")
WORK = os.path.join(ROOT, "work/finepdfs")
MIN_TOKENS = 8192

# Metadata thresholds.
LID_MIN = 0.90      # full_doc_lid_score, which GlotLID computed on the first 40K characters
PLID_MIN = 0.80     # page_average_lid_score; documents with many non-Russian pages fall below it
# Qwen3 gives 1.03 to 1.28 times as many tokens as the Llama-3.2 token_count (1st to 99th
# percentile, median 1.10). Below 6,500 Llama tokens only 0.04% of documents reach 8,192 Qwen3
# tokens, so the prefilter loses almost nothing and saves tokenizing short documents.
LLAMA_MIN = 6500
# The mean fw_edu score is the size knob: at 0.9, the text filters below leave about 15B Qwen3
# tokens over the 36 files.
EDU_MIN = 0.90

# Text filters, in the order they are applied. Each takes the dict from text_features().
TEXT_FILTERS = [
    ("low_cyrillic", lambda f: f["cyr_share"] < 0.80),
    ("garbage_chars", lambda f: f["garbage"] > 0.01),
    ("wrong_codepage", lambda f: f["latx_share"] > 0.01),
    # A work program or curriculum title is enough on its own. An approval stamp ("УТВЕРЖДАЮ",
    # "Протокол №") also appears on some textbooks, so it only counts together with officialese.
    ("officialese_head", lambda f: f["head_program"] or (f["head_stamp"] and f["official_per_10k"] > 20)),
    ("officialese_body", lambda f: f["official_per_10k"] > 40),
    ("repeated_lines", lambda f: f["dup_line_frac"] > 0.15),
    ("tables", lambda f: f["table_frac"] > 0.25 or f["html_tags"] * 1000 > f["chars"]),
    ("fake_ip", lambda f: f["fake_ip_per_10k"] > 0.5),
    ("repetitive_zstd", lambda f: f["zstd_ratio"] < 0.16),
    ("digits", lambda f: f["digit_share"] > 0.12),
]
# The measurements behind the thresholds, copied into the manifest so a reader can judge them.
NOTES = {
    "metadata_scan": "36 files, 8.05M documents, 71.8B Llama tokens; 1.69M documents have >= 7,000 Llama tokens (59.9B).",
    "scores": "dclm_scores, ocr_quality_scores and fw_edu_v2_scores are empty for rus_Cyrl; fw_edu_scores is the only quality score.",
    "qwen_per_llama": "median 1.104, 1st percentile 1.03, 99th percentile 1.28, over 10,255 tokenized documents.",
    "size": "18.64B Llama tokens pass the metadata cuts at EDU_MIN 0.9; the text filters keep 73% of them on the probe "
            "files, which predicts about 15.1B Qwen3 tokens.",
    "text_filter_token_shares_on_probe": "officialese_head 15.4%, tables 5.0%, low_cyrillic 2.1%, repeated_lines 1.8%, "
            "fake_ip 0.9%, repetitive_zstd 0.6%, garbage_chars 0.6%, officialese_body 0.2%, wrong_codepage 0.1%, digits 0.1%.",
    "long_tail": "more than half of the kept tokens are in documents over 128K tokens (journal issues, books, collections).",
}
META_REASONS = ["not_rus", "page_lang", "short_prefilter", "low_edu"]
REASONS = META_REASONS + [n for n, _ in TEXT_FILTERS] + ["duplicate", "short_tokens"]
N_EXAMPLES = 5

COLS = ["text", "token_count", "full_doc_lid", "full_doc_lid_score", "page_average_lid_score", "fw_edu_scores"]
# The tokenizer's Encoding objects hold token strings and offsets besides the ids, about 150 bytes
# per token, so one encode_batch call is kept to about 3M tokens (8M characters).
TOKENIZE_BATCH_CHARS = 8_000_000


def mean_list(col):
    """Mean of each row's list of chunk scores; rows without scores get -1 so they fail the cut."""
    lens = pc.list_value_length(col).to_numpy(zero_copy_only=False)
    flat = pc.list_flatten(col).to_numpy(zero_copy_only=False)
    out = np.full(len(lens), -1.0)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    nz = lens > 0
    if nz.any():
        out[nz] = np.add.reduceat(flat, starts[nz]) / lens[nz]
    return out


def text_key(s):
    """64-bit hash of the lowercased text with all whitespace runs collapsed to one space."""
    d = hashlib.blake2b(" ".join(s.lower().split()).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "little")


class FileBuilder:
    """Processes one parquet file into one doc-store part, collecting stats and example heads."""

    def __init__(self, store, part, seen):
        self.w = documents.DocStoreWriter(store, part, flush_tokens=100_000_000)
        self.seen = seen
        self.st = Counter()
        self.ex = {"kept": [], **{r: [] for r in REASONS}}
        self.hashes = []
        self.pending, self.pending_chars = [], 0
        self.qwen_llama = []   # (qwen tokens, llama tokens) of every tokenized document
        self.t = Counter()     # seconds spent per step

    def note(self, reason, text):
        self.st["drop_" + reason] += 1
        if len(self.ex[reason]) < N_EXAMPLES:
            self.ex[reason].append(text[:200])

    def batch(self, b):
        n = b.num_rows
        self.st["read"] += n
        tokc = b.column("token_count").to_numpy()
        rus = pc.equal(b.column("full_doc_lid"), "rus_Cyrl").to_numpy(zero_copy_only=False)
        lid = b.column("full_doc_lid_score").to_numpy(zero_copy_only=False)
        plid = b.column("page_average_lid_score").to_numpy(zero_copy_only=False)
        edu = mean_list(b.column("fw_edu_scores"))
        # The checks are assigned from the last to the first, so the first failing check is the
        # one that remains as the drop reason.
        reason = np.full(n, "", dtype=object)
        reason[edu < EDU_MIN] = "low_edu"
        reason[tokc < LLAMA_MIN] = "short_prefilter"
        reason[plid < PLID_MIN] = "page_lang"
        reason[~rus | (lid < LID_MIN)] = "not_rus"
        text_col = b.column("text")
        for r in META_REASONS:
            idx = np.flatnonzero(reason == r)
            self.st["drop_" + r] += int(idx.size)
            # Only the few example texts are pulled out of Arrow, not the whole batch.
            for i in idx[:max(0, N_EXAMPLES - len(self.ex[r]))]:
                self.ex[r].append((text_col[int(i)].as_py() or "")[:200])
        keep_idx = np.flatnonzero(reason == "")
        if not keep_idx.size:
            return
        texts = b.column("text").take(keep_idx).to_pylist()
        for i, s in zip(keep_idx, texts):
            s = documents.clean_text(s)
            t0 = time.time()
            f = text_features(s)
            bad = next((name for name, fn in TEXT_FILTERS if fn(f)), None)
            self.t["features"] += time.time() - t0
            if bad:
                self.note(bad, s)
                continue
            k = text_key(s)
            if k in self.seen:
                self.note("duplicate", s)
                continue
            self.seen.add(k)
            self.pending.append((s, int(tokc[i]), k))
            self.pending_chars += len(s)
            if self.pending_chars >= TOKENIZE_BATCH_CHARS:
                self.flush_pending()

    def flush_pending(self):
        if not self.pending:
            return
        t0 = time.time()
        docs = documents.encode_docs([p[0] for p in self.pending])
        self.t["tokenize"] += time.time() - t0
        for (s, tl, k), d in zip(self.pending, docs):
            ntok = len(d) - 1
            self.qwen_llama.append((ntok, tl))
            if ntok < MIN_TOKENS:
                self.note("short_tokens", s)
                continue
            self.w.add(d)
            self.hashes.append(k)
            self.st["docs_written"] += 1
            self.st["tokens_written"] += len(d)
            if len(self.ex["kept"]) < N_EXAMPLES:
                self.ex["kept"].append(s[:200])
        self.pending, self.pending_chars = [], 0

    def close(self):
        self.flush_pending()
        self.w.close()


def run(a):
    files = sorted(glob.glob(os.path.join(SRC, "*.parquet")))
    store = os.path.join(ROOT, "work/docs", a.out_name)
    stats_dir = os.path.join(WORK, "stats", a.out_name)
    os.makedirs(stats_dir, exist_ok=True)
    mine = list(enumerate(files))[a.part::a.nparts][:a.max_files or None]
    seen = set()
    # Documents already written by this part's finished files still count as seen, so a
    # restarted process does not write their duplicates again.
    for i, _ in mine:
        hp = os.path.join(stats_dir, f"file_{i:04d}.hashes.npy")
        if os.path.exists(os.path.join(stats_dir, f"file_{i:04d}.json")) and os.path.exists(hp):
            seen.update(int(x) for x in np.load(hp))
    for i, fp in mine:
        name = os.path.basename(fp)[:-8]
        done = os.path.join(stats_dir, f"file_{i:04d}.json")
        if os.path.exists(done):
            print(f"{name}: already done, skipping", flush=True)
            continue
        # An interrupted file may have left flushed sub-parts behind. The file is processed again
        # from the start, so those leftovers are removed first; they are this script's own output.
        for old in glob.glob(os.path.join(store, f"part_{i:04d}_*")):
            os.remove(old)
        t0 = time.time()
        fb = FileBuilder(store, i, seen)
        pf = pq.ParquetFile(fp)
        for b in pf.iter_batches(batch_size=2000, columns=COLS):
            fb.batch(b)
        fb.close()
        ql = np.array(fb.qwen_llama, dtype=np.int64).reshape(-1, 2)
        out = {"file": name, "part_id": i, "seconds": round(time.time() - t0),
               "seconds_features": round(fb.t["features"]), "seconds_tokenize": round(fb.t["tokenize"]),
               **dict(fb.st),
               "qwen_per_llama_median": round(float(np.median(ql[:, 0] / ql[:, 1])), 4) if len(ql) else None,
               "examples": fb.ex}
        np.save(os.path.join(stats_dir, f"file_{i:04d}.hashes.npy"), np.array(fb.hashes, dtype=np.uint64))
        np.save(os.path.join(stats_dir, f"file_{i:04d}.qwen_llama.npy"), ql)
        json.dump(out, open(done + ".tmp", "w"), ensure_ascii=False, indent=1)
        os.replace(done + ".tmp", done)
        print(f"{time.strftime('%H:%M:%S')} {name}: {json.dumps({k: v for k, v in out.items() if k != 'examples'})}", flush=True)


def manifest(a):
    stats_dir = os.path.join(WORK, "stats", a.out_name)
    per_file = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(stats_dir, "file_*.json")))]
    tot = Counter()
    for s in per_file:
        tot.update({k: v for k, v in s.items() if isinstance(v, int) and not k.startswith(("part_id", "seconds"))})
    # Duplicates across processes are not caught during the run, so they are counted here.
    allh = [np.load(p) for p in sorted(glob.glob(os.path.join(stats_dir, "file_*.hashes.npy")))]
    allh = np.concatenate(allh) if allh else np.zeros(0, np.uint64)
    cross_dups = int(allh.size - np.unique(allh).size)
    # The first 300 characters of three kept documents, decoded back from the doc store, so the
    # manifest shows what was really written.
    tok = documents.tokenizer()
    store = os.path.join(ROOT, "work/docs", a.out_name)
    heads = []
    for nm in documents.docstore_files(store)[:3]:
        d = next(documents.iter_docstore_file(store, nm))
        heads.append(tok.decode(d[:300].tolist())[:300])
    examples = {}
    for s in per_file:
        for r, lst in (s.get("examples") or {}).items():
            examples.setdefault(r, [])
            examples[r] += lst[:N_EXAMPLES - len(examples[r])]
    m = {"stream": a.out_name, "source": SRC, "files_total": len(glob.glob(os.path.join(SRC, "*.parquet"))),
         "files_done": len(per_file),
         "thresholds": {"LID_MIN": LID_MIN, "PLID_MIN": PLID_MIN, "LLAMA_MIN": LLAMA_MIN, "EDU_MIN": EDU_MIN,
                        "MIN_TOKENS": MIN_TOKENS, "text_filters_in_order": [n for n, _ in TEXT_FILTERS]},
         "notes": NOTES,
         "examples_read": tot["read"], "drops": {r: tot["drop_" + r] for r in REASONS},
         "cross_process_duplicates_not_removed": cross_dups,
         "docs_written": tot["docs_written"], "tokens_written": tot["tokens_written"],
         "rendered_examples_first300": heads, "example_heads_200": examples}
    os.makedirs(os.path.join(ROOT, "manifests/build"), exist_ok=True)
    dst = os.path.join(ROOT, "manifests/build", a.out_name + ".json")
    json.dump(m, open(dst + ".tmp", "w"), ensure_ascii=False, indent=1)
    os.replace(dst + ".tmp", dst)
    print(json.dumps({k: v for k, v in m.items() if k not in ("rendered_examples_first300", "example_heads_200", "notes")},
                     ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--out-name", default="finepdfs_ru")
    ap.add_argument("--manifest", action="store_true")
    a = ap.parse_args()
    manifest(a) if a.manifest else run(a)
