"""Builds the long-document streams olmocr_science, olmocr_health, olmocr_literature and arxiv.

Every source is a set of raw .jsonl.zst files (read-only). For each document, in this order:
  1. a cheap character-length prefilter drops documents that cannot reach 8,192 tokens;
  2. exact deduplication by a hash of the whitespace-normalized text (the Dolma mix repeats
     documents, arXiv papers two or three times, always next to each other in the same file);
  3. the strict quality filters and the span cleaning described below;
  4. tokenization with the Qwen3 tokenizer; documents of at least 8,192 tokens are written to
     the document store work/docs/<stream>, which pack.py later packs into 16K blocks.

The filter thresholds come from a data-quality audit that was made before this build. A sample
of each source (the olmOCR health, science and literature folders, and arXiv) was scored with
the metrics below, and every threshold was chosen on those scores. The metric functions here use
the audit's definitions exactly, so that each threshold keeps the meaning it had on the sample.
The comments below say which part of the audit each rule comes from, and the build manifest
repeats the rules.

Every run works on one part of the selected files (--part P --nparts N takes every Nth file),
so parts are independent and can run as separate processes or on separate machines.
Deduplication is exact within a part; the part also saves the hash of every written document,
and `crossdup` counts the copies that ended up in different parts.

Usage:
  build_pdf_arxiv.py build --stream olmocr_literature --part 0 --nparts 1 [--max-files K] [--test]
  build_pdf_arxiv.py manifest --stream olmocr_literature --nparts 1
  build_pdf_arxiv.py crossdup --stream olmocr_literature
  build_pdf_arxiv.py files --stream olmocr_science        (prints the selected files)
"""
import argparse
import hashlib
import io
import json
import os
import random
import re
import time
from collections import Counter

import numpy as np
import zstandard

import documents
import paths

ROOT = paths.CTX16K_ROOT
RAW = paths.DOLMA3_MIX
WORK = os.path.join(ROOT, "work", "pdf_arxiv")
MIN_TOKENS = 8192
SELECTION_SEED = 20260930

# The audit found proofpile-arXiv documents only in these ten files; the other ten files of the
# folder hold other Proof-Pile-2 parts (web math, code, a validation split). Every document's id
# is still checked, so a stray non-arXiv document is dropped even inside these files.
ARXIV_FILES = ["arxiv-train-0000.jsonl.zst", "arxiv-train-0001.jsonl.zst", "arxiv-train-0003.jsonl.zst",
               "arxiv-train-0004.jsonl.zst", "arxiv-train-0005.jsonl.zst", "arxiv-train-0007.jsonl.zst",
               "arxiv-train-0008.jsonl.zst", "arxiv-train-0013.jsonl.zst", "arxiv-train-0015.jsonl.zst",
               "arxiv-train-0016.jsonl.zst"]

# Per stream: source folders, the share of files to take, and the lowest characters-per-token
# ratio we expect. The ratio sets the length prefilter: a document shorter than
# MIN_TOKENS * ratio characters is assumed to be too short and is never tokenized. The ratios are
# below the measured averages (about 4.3 for English prose, 3.9 for science PDFs with formulas,
# 3.0 for LaTeX) so that dense documents are not lost; test runs check this with --probe.
STREAMS = {
    "olmocr_science": {"dirs": ["olmocr_science_pdfs-science_math_and_technology-part1",
                                "olmocr_science_pdfs-science_math_and_technology-part2"],
                       "share": 0.40, "min_cpt": 3.0, "kind": "pdf"},
    "olmocr_health": {"dirs": ["olmocr_science_pdfs-health"], "share": 0.60, "min_cpt": 3.0, "kind": "pdf"},
    "olmocr_literature": {"dirs": ["olmocr_science_pdfs-literature"], "share": 1.0, "min_cpt": 3.0, "kind": "pdf"},
    "arxiv": {"dirs": ["rpj-proofpile-arxiv"], "share": 1.0, "min_cpt": 2.2, "kind": "arxiv"},
}


# ---------------------------------------------------------------- shared text metrics

WORD_RE = re.compile(r"\w+", re.UNICODE)
HYPHEN_BREAK_RE = re.compile(r"\w-\n\w")
BULLET_RE = re.compile(r"^\s*([-*•●▪]|\d+[.)])\s")
_P64 = np.uint64(0x100000001B3)


def _bmp_table(test):
    return np.array([test(chr(c)) for c in range(65536)], dtype=bool)


# Lookup tables of str.isspace / isalpha / isdigit / isalnum for every character of the Basic
# Multilingual Plane. The audit called these methods once per character, which is the slowest part
# of scoring a long document; indexing a table with the whole document at once gives the same
# answers. The few characters outside the plane (for example math letters such as 𝑥) are still
# checked with the string methods.
_IS_SPACE = _bmp_table(lambda ch: re.match(r"\s", ch) is not None)
_IS_ALPHA = _bmp_table(str.isalpha)
_IS_DIGIT = _bmp_table(str.isdigit)
_IS_ALNUM = _bmp_table(str.isalnum)


def char_classes(text):
    """Counts of non-space characters, and of letters, digits and alphanumerics among them."""
    cp = np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype=np.uint32)
    bmp = cp < 65536
    c = cp[bmp]
    keep = ~_IS_SPACE[c]
    ns = int(keep.sum())
    alpha = int(_IS_ALPHA[c][keep].sum())
    digit = int(_IS_DIGIT[c][keep].sum())
    alnum = int(_IS_ALNUM[c][keep].sum())
    if not bmp.all():
        for ch in (chr(x) for x in cp[~bmp]):
            if re.match(r"\s", ch) is None:
                ns += 1
                alpha += ch.isalpha()
                digit += ch.isdigit()
                alnum += ch.isalnum()
    return ns, alpha, digit, alnum


def dup13_frac(words):
    """Share of word 13-grams that occur more than once in the document.

    This is the audit's dup13_frac. The audit joined every 13 words into a string and counted the
    strings, which is too slow for hundreds of gigabytes, so here each word becomes a 64-bit hash
    and each 13-gram a rolling combination of 13 word hashes. Two different 13-grams get the same
    value only by a 64-bit collision, so the result is the same number in practice.
    """
    n = len(words) - 12
    if n <= 0:
        return 0.0
    hs = np.fromiter(map(hash, words), dtype=np.int64, count=len(words)).view(np.uint64)
    g = np.zeros(n, np.uint64)
    with np.errstate(over="ignore"):
        for k in range(13):
            g = g * _P64 + hs[k:k + n]
    _, counts = np.unique(g, return_counts=True)
    return float(counts[counts > 1].sum()) / n


def metrics(text):
    """The audit's quality metrics, limited to the ones the rules use.

    Values are rounded to three decimals before the rules compare them, as in the audit, and
    the Gopher-like check uses the unrounded values, also as in the audit.
    """
    n = max(len(text), 1)
    nonempty = [l for l in text.split("\n") if l.strip()]
    words = WORD_RE.findall(text)
    nw = max(len(words), 1)
    line_counts = Counter(l.strip() for l in nonempty)
    dup_line_chars = sum(len(l) * (c - 1) for l, c in line_counts.items() if c > 1 and len(l) > 10)
    dup13 = dup13_frac(words)
    n_ns, n_alpha, n_digit, n_alnum = char_classes(text)
    ns = max(n_ns, 1)
    alpha = n_alpha / ns
    digits = n_digit / ns
    symbols = (n_ns - n_alnum) / ns
    mean_word = len("".join(words)) / nw
    short_lines = sum(1 for l in nonempty if len(l.strip()) < 20) / max(len(nonempty), 1)
    bullets = sum(1 for l in nonempty if BULLET_RE.match(l)) / max(len(nonempty), 1)
    gopher_ok = (3 <= mean_word <= 10 and symbols < 0.35 and bullets < 0.9 and alpha > 0.6 and dup13 < 0.2)
    return {
        "dup_line_char_frac": round(dup_line_chars / n, 3),
        "dup13_frac": round(dup13, 3),
        "alpha_frac": round(alpha, 3),
        "digit_frac": round(digits, 3),
        "symbol_frac": round(symbols, 3),
        "short_line_frac": round(short_lines, 3),
        "bullet_line_frac": round(bullets, 3),
        "hyphen_breaks_per_1k_words": round(1000 * len(HYPHEN_BREAK_RE.findall(text)) / nw, 2),
        "mean_word_len": round(mean_word, 2),
        "gopher_like_pass": gopher_ok,
    }


# ---------------------------------------------------------------- olmOCR PDFs

TABLE_BLOCK = re.compile(r"<table.*?</table>", re.S | re.I)
# A table-of-contents line with dot leaders (". . . . 12"), a run of ellipses, or a fill-in blank.
DOT_LEADER = re.compile(r"(\.\s?){6,}|…{3,}|_{6,}")


def is_dot_line(line):
    """True for a line that matches DOT_LEADER.

    A match needs at least six dots, an ellipsis character or six underscores, so the cheap
    substring checks skip the regex for almost every line without changing the result.
    """
    return (line.count(".") >= 6 or "…" in line or "______" in line) and DOT_LEADER.search(line) is not None
REF_HEAD = re.compile(r"\n\s*(#+\s*|\*\*)?\s*(\d+\.?\s*)?(references|reference list|bibliography|literature cited|works cited|"
                      r"referencias|bibliografía|références|literatur|referências)\s*:?\s*(\*\*)?\s*\n", re.I)
PAGENUM = re.compile(r"^\s*(page\s+)?\d{1,4}(\s+of\s+\d{1,4})?\s*$", re.I | re.M)
SEPARATOR = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,}|(\*\s){3,}\*?)\s*$", re.M)
SUPERSCRIPT = re.compile(r"\\textsuperscript\{([^}]*)\}")
MIDBREAK = re.compile(r"[a-z,]\n[a-z]")
MIDBREAK_JOIN = re.compile(r"([a-z,])\n([a-z])")
LATIN_WORD = re.compile(r"[A-Za-zÀ-ÿ]+")
EN_STOP = set("the of and to in a is that for are with as be on by this it or from at an was were which not have has "
              "their can these more also".split())


def meta_first(md, key):
    """Reads the document-level value of a Dolma span attribute such as the English score."""
    v = md.get(key)
    if isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) == 3:
        return v[0][2]
    return None


def pdf_raw_features(t):
    """Features the health audit computed on the text as stored, before any cleaning."""
    n = max(len(t), 1)
    words = LATIN_WORD.findall(t)
    nw = max(len(words), 1)
    lines = [l for l in t.split("\n") if l.strip()]
    nl = max(len(lines), 1)
    return {
        "html_table_char_frac": round(sum(len(m.group()) for m in TABLE_BLOCK.finditer(t)) / n, 3),
        "dotleader_line_frac": round(sum(1 for l in lines if is_dot_line(l)) / nl, 3),
        "midbreak_per_1k_words": round(1000 * len(MIDBREAK.findall(t)) / nw, 2),
        "latex_per_1k_chars": round(1000 * (t.count("\\(") + t.count("$") + t.count("\\[")) / n, 3),
    }


def en_stop_frac(t):
    """Share of Latin-script words that are common English function words (health audit).

    Counting distinct words first and lowercasing only those is much faster than lowercasing
    every word, and gives the same number.
    """
    counts = Counter(LATIN_WORD.findall(t))
    stop = sum(v for w, v in counts.items() if w.lower() in EN_STOP)
    return stop / max(sum(counts.values()), 1)


def clean_pdf(t, join_breaks):
    """The audit's span cleaning for olmOCR text.

    It removes HTML tables, Markdown table rows, table-of-contents lines with dot leaders, lone
    page numbers and separator lines, unwraps \\textsuperscript{}, and cuts the text at the last
    references heading when that heading sits after the first 40% of the document (a heading
    earlier than that is more likely a section title than the start of the bibliography).
    join_breaks also joins a line break that splits a sentence (lowercase or comma, newline,
    lowercase). That join is applied to health only, because in literature it would merge the
    lines of verse.
    """
    t = TABLE_BLOCK.sub("", t)
    t = "\n".join(l for l in t.split("\n") if not l.lstrip().startswith("|") and not is_dot_line(l))
    refs = list(REF_HEAD.finditer(t))
    if refs and refs[-1].start() > 0.4 * len(t):
        t = t[:refs[-1].start()]
    t = PAGENUM.sub("", t)
    t = SEPARATOR.sub("", t)
    t = SUPERSCRIPT.sub(r"\1", t)
    if join_breaks:
        t = MIDBREAK_JOIN.sub(r"\1 \2", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def pdf_rules(stream, raw, m, stop_frac, md):
    """Returns the names of the strict rules the document fails (empty list: it passes).

    The base set is the union of the two olmOCR audits, taking the stricter threshold where they
    differ: the health audit (non-English, tables, table of contents, broken lines, short lines,
    bullets, symbols) and the science audit (duplicate 13-grams above 0.08 and duplicate lines
    above 0.03 instead of health's 0.10 and 0.05, hyphenated line breaks). On the science sample
    this set keeps about 44% of tokens, which is the share the strict filter was meant to keep.

    Literature adds the thresholds of the literature audit, which was stricter still: its
    survivors had no fallback pages, an English score of at least 0.8, few digits and formulas.
    """
    r = []
    if stop_frac < 0.20:
        r.append("non_english")
    if raw["html_table_char_frac"] > 0.15:
        r.append("tables")
    if raw["dotleader_line_frac"] > 0.05:
        r.append("toc")
    if raw["midbreak_per_1k_words"] > 10:
        r.append("midbreaks")
    if m["dup13_frac"] > 0.08:
        r.append("dup13")
    if m["dup_line_char_frac"] > 0.03:
        r.append("dup_lines")
    if m["short_line_frac"] > 0.45:
        r.append("short_lines")
    if m["bullet_line_frac"] > 0.30:
        r.append("bullets")
    if m["symbol_frac"] > 0.12 or m["alpha_frac"] < 0.85:
        r.append("symbols")
    if m["hyphen_breaks_per_1k_words"] > 1.0:
        r.append("hyphen_breaks")
    if not m["gopher_like_pass"]:
        r.append("gopher")
    if stream == "olmocr_literature":
        en = meta_first(md, "ft_lang_id_en_doc_v2__ft_lang_id_en_doc_v2__en")
        if (md.get("total-fallback-pages") or 0) > 0:
            r.append("lit_fallback_pages")
        if en is None or en < 0.8:
            r.append("lit_en_score")
        if m["dup13_frac"] > 0.05:
            r.append("lit_dup13")
        if m["short_line_frac"] > 0.35:
            r.append("lit_short_lines")
        if m["symbol_frac"] > 0.10 or m["alpha_frac"] < 0.88:
            r.append("lit_symbols")
        if m["digit_frac"] > 0.03:
            r.append("lit_digits")
        if raw["latex_per_1k_chars"] > 2:
            r.append("lit_latex")
    return r


# ---------------------------------------------------------------- arXiv LaTeX

# The arXiv audit's LaTeX measurements, copied unchanged so that the thresholds mean the same.
CMD = re.compile(r"\\[a-zA-Z@]+\*?")
INLINE = re.compile(r"(?<!\\)\$\$.+?(?<!\\)\$\$|(?<!\\)\$[^$]+?(?<!\\)\$|\\\[.+?\\\]|\\\(.+?\\\)", re.S)
ENVMATH = re.compile(r"\\begin\{(equation|align|eqnarray|gather|multline|displaymath|split|array)\*?\}.*?\\end\{\1\*?\}", re.S)
TAB = re.compile(r"\\begin\{(tabular|table)\*?\}.*?\\end\{\1\*?\}", re.S)
FIG = re.compile(r"\\begin\{(figure|picture|tikzpicture|pspicture)\*?\}.*?\\end\{\1\*?\}", re.S)
PRE = re.compile(r"\\(documentclass|usepackage|newcommand|renewcommand|def\\|DeclareMathOperator|newtheorem)")
# The cleaning of an arXiv paper removes its table environments, and also its bibliography when
# that starts after the first 40% of the paper.
TABLE_ENV = re.compile(r"\\begin\{(table\*?|tabular\*?|tabularx|longtable|sidewaystable)\}.*?\\end\{\1\}", re.S)
BIB = re.compile(r"\\begin\{thebibliography\}|\\bibliography\{|\\section\*?\{\s*(References|Bibliography)\s*\}|\n\s*(References|REFERENCES|Bibliography)\s*\n")

# LaTeX commands that occur in at least 3 of the audit's 260 sampled papers. The papers come
# without their preamble, so any other command is most likely an author macro that the model will
# never see defined. The rate of such commands per 1,000 characters is the audit's "rare" measure.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_arxiv_latex_cmds.json")) as _f:
    COMMON_CMDS = frozenset(json.load(_f))


def latex_metrics(t):
    n = max(len(t), 1)
    math_chars = sum(len(m.group(0)) for m in ENVMATH.finditer(t))
    t2 = ENVMATH.sub(" ", t)
    math_chars += sum(len(m.group(0)) for m in INLINE.finditer(t2))
    t3 = INLINE.sub(" ", t2)
    tab_chars = sum(len(m.group(0)) for m in TAB.finditer(t3))
    t4 = TAB.sub(" ", t3)
    t4 = FIG.sub(" ", t4)
    cmd_chars = sum(len(m.group(0)) for m in CMD.finditer(t4))
    t5 = CMD.sub(" ", t4)
    t5 = re.sub(r"[{}\[\]\\&%~^_]", " ", t5)
    prose_chars = len(re.sub(r"\s+", " ", t5))
    cmds = CMD.findall(t)
    rare = sum(1 for c in cmds if c not in COMMON_CMDS)
    return {
        "math_char_frac": round(math_chars / n, 3),
        "table_char_frac": round(tab_chars / n, 3),
        "cmd_char_frac_outside_math": round(cmd_chars / n, 3),
        "prose_char_frac": round(prose_chars / n, 3),
        "preamble_macros": len(PRE.findall(t)),
        "pct_lines_comment": round(sum(1 for l in t.split("\n") if l.lstrip().startswith("%")) / max(t.count("\n") + 1, 1), 4),
        "rare_cmds_per_1k_chars": 1000 * rare / n,
    }


def arxiv_rules(m, lm):
    """Returns the names of the strict rules the paper fails (empty list: it passes).

    The audit recorded a base filter (about 50% of tokens) and a "strict filter" of 28%, but did
    not save the strict rule itself. This set was rebuilt from the audit's per-paper records: it
    keeps only papers that also passed the base filter, and on the audit sample it keeps 29% of
    tokens. Its thresholds: little repetition, at least half of the characters prose, formulas at
    most 45%, at most one undefined author macro per 1,000 characters, no leftover preamble, few
    tables and comment lines, and the Gopher-like check.
    """
    r = []
    if not m["gopher_like_pass"]:
        r.append("gopher")
    if m["dup13_frac"] > 0.10:
        r.append("dup13")
    if m["dup_line_char_frac"] > 0.05:
        r.append("dup_lines")
    if m["symbol_frac"] > 0.31 or m["alpha_frac"] < 0.66:
        r.append("symbols")
    if lm["prose_char_frac"] < 0.5:
        r.append("low_prose")
    if lm["math_char_frac"] > 0.45:
        r.append("math_heavy")
    if lm["rare_cmds_per_1k_chars"] > 1.0:
        r.append("undefined_macros")
    if lm["preamble_macros"] > 3:
        r.append("preamble")
    if lm["table_char_frac"] > 0.15:
        r.append("tables")
    if lm["cmd_char_frac_outside_math"] > 0.08:
        r.append("commands")
    if lm["pct_lines_comment"] > 0.02:
        r.append("comments")
    return r


def clean_arxiv(t):
    t = TABLE_ENV.sub("", t)
    ms = list(BIB.finditer(t))
    if ms and ms[0].start() > 0.4 * len(t):
        t = t[:ms[0].start()]
    return re.sub(r"\n{3,}", "\n\n", t).strip()


# ---------------------------------------------------------------- file selection and reading

def all_files(stream):
    cfg = STREAMS[stream]
    if stream == "arxiv":
        return [os.path.join(RAW, cfg["dirs"][0], f) for f in ARXIV_FILES]
    files = []
    for d in cfg["dirs"]:
        files += [os.path.join(RAW, d, f) for f in sorted(os.listdir(os.path.join(RAW, d))) if f.endswith(".jsonl.zst")]
    return files


def selected_files(stream):
    """The files of the stream, subsampled at random with a fixed seed when STREAMS takes a share.

    The selection is sorted, so part P of N (every Nth file) is the same list on every machine.
    """
    files = all_files(stream)
    share = STREAMS[stream]["share"]
    if share < 1.0:
        files = sorted(random.Random(SELECTION_SEED).sample(files, round(share * len(files))))
    return files


def iter_jsonl_zst(path):
    with open(path, "rb") as fh:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in reader:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------- the build

def part_output_paths(stream, part, test):
    """Where a part writes its documents, its statistics and the hashes of its documents.

    Test runs write under work/pdf_arxiv/test, so they never touch the real document store.
    """
    base = os.path.join(WORK, "test") if test else WORK
    store = os.path.join(base, "docs", stream) if test else os.path.join(ROOT, "work", "docs", stream)
    return (store, os.path.join(base, "stats", f"{stream}.part_{part:04d}.json"),
            os.path.join(base, "hashes", stream, f"part_{part:04d}.npy"))


class Part:
    """Processes one part of a stream and keeps its statistics."""

    def __init__(self, a):
        self.a = a
        self.stream = a.stream
        self.kind = STREAMS[a.stream]["kind"]
        self.min_chars = int(MIN_TOKENS * STREAMS[a.stream]["min_cpt"])
        self.join_breaks = a.stream == "olmocr_health"
        self.store, self.stats_path, self.hash_path = part_output_paths(a.stream, a.part, a.test)
        self.seen = set()
        self.kept_hashes = []
        self.pending = []  # (cleaned text, hash) waiting for batched tokenization
        self.pending_chars = 0
        self.st = Counter()
        self.reasons_first = Counter()
        self.reasons_any = Counter()
        self.examples = []
        self.probe = Counter()

    def prepare_outputs(self):
        """Refuses to mix with an earlier run of the same part, unless --overwrite removes it.

        A crashed run leaves some of its store files behind; a new run would write its own
        files next to them and the part would hold some documents twice.
        """
        os.makedirs(self.store, exist_ok=True)
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.hash_path), exist_ok=True)
        prefix = f"part_{self.a.part:04d}_"
        old = [f for f in os.listdir(self.store) if f.startswith(prefix)]
        if old or os.path.exists(self.stats_path):
            if not self.a.overwrite:
                raise SystemExit(f"part {self.a.part} of {self.stream} already has outputs ({len(old)} store files); "
                                 f"pass --overwrite to rebuild it")
            for f in old:
                os.remove(os.path.join(self.store, f))
            for p in (self.stats_path, self.hash_path):
                if os.path.exists(p):
                    os.remove(p)

    def doc_text(self, d):
        """Returns the cleaned text of a document that passes every filter, or None.

        Every drop is counted under the first rule the document fails, and every failed rule is
        also counted on its own, so the manifest shows both what removed documents and how the
        rules overlap.
        """
        t = d.get("text") or ""
        self.st["examples_read"] += 1
        self.st["chars_read"] += len(t)
        if self.kind == "arxiv" and not str(d.get("id", "")).startswith("proofpile-arXiv"):
            self.st["drop_not_arxiv"] += 1
            return None
        if len(t) < self.min_chars:
            self.st["drop_short_chars_raw"] += 1
            if self.a.probe and len(t) >= 0.75 * self.min_chars:
                self.probe_short(t, d.get("metadata") or {})
            return None
        key = hashlib.blake2b(" ".join(t.split()).encode("utf-8", "ignore"), digest_size=8).digest()
        if key in self.seen:
            self.st["drop_duplicate"] += 1
            self.st["duplicate_chars"] += len(t)
            return None
        self.seen.add(key)
        self.st["unique_long_docs"] += 1
        self.st["unique_long_chars"] += len(t)
        c, reasons = self.filter_doc(t, d.get("metadata") or {}, True)
        if reasons:
            self.reasons_first[reasons[0]] += 1
            self.reasons_any.update(reasons)
            self.st["drop_filters"] += 1
            return None
        self.st["passed_filters"] += 1
        self.pending.append((c, key))
        self.pending_chars += len(c)
        return c

    def filter_doc(self, t, md, check_len):
        """Cleans one document and returns (cleaned text, names of the rules it fails).

        check_len also fails a document whose cleaned text is shorter than the length prefilter;
        the prefilter probe turns it off to see what the prefilter alone would have lost.
        """
        if self.kind == "pdf":
            raw = pdf_raw_features(t)
            c = clean_pdf(t, self.join_breaks)
            if check_len and len(c) < self.min_chars:
                return c, ["short_chars_after_clean"]
            return c, pdf_rules(self.stream, raw, metrics(c), en_stop_frac(c), md)
        reasons = arxiv_rules(metrics(t), latex_metrics(t))
        c = clean_arxiv(t) if not reasons else ""
        if check_len and not reasons and len(c) < self.min_chars:
            reasons = ["short_chars_after_clean"]
        return c, reasons

    def probe_short(self, t, md):
        """Test mode only: checks what the raw length prefilter throws away.

        Documents just below the prefilter length are tokenized raw. The ones that reach 8,192
        tokens are then cleaned and filtered like any other document, and the manifest reports
        how many of them would have been written without the prefilter.
        """
        n = len(documents.encode_docs([t])[0]) - 1
        self.probe["below_prefilter_tokenized"] += 1
        if n < MIN_TOKENS:
            return
        self.probe["below_prefilter_reaching_8k_raw"] += 1
        c, reasons = self.filter_doc(t, md, False)
        if not reasons and len(documents.encode_docs([c])[0]) - 1 >= MIN_TOKENS:
            self.probe["below_prefilter_would_be_written"] += 1

    def flush(self, writer):
        if not self.pending:
            return
        docs = documents.encode_docs([c for c, _ in self.pending])
        for (c, key), doc in zip(self.pending, docs):
            ntok = len(doc) - 1
            self.st["tokens_of_passed"] += ntok
            self.st["chars_of_passed"] += len(c)
            if ntok < MIN_TOKENS:
                self.st["drop_short_tokens"] += 1
                continue
            writer.add(doc)
            self.kept_hashes.append(int.from_bytes(key, "little"))
            self.st["docs_written"] += 1
            self.st["tokens_written"] += len(doc)
            self.st["chars_written"] += len(c)
            if len(self.examples) < 3:
                self.examples.append(c[:300])
        self.pending, self.pending_chars = [], 0

    def save_stats(self, files, done, t0):
        out = {"stream": self.stream, "part": self.a.part, "nparts": self.a.nparts, "done": done,
               "files_total": len(files), "files_done": self.st["files_done"],
               "minutes": round((time.time() - t0) / 60, 2), "min_chars_prefilter": self.min_chars,
               "stats": dict(self.st), "drop_first_reason": dict(self.reasons_first),
               "fail_any_reason": dict(self.reasons_any), "probe": dict(self.probe),
               "examples": self.examples, "file_seconds": self.file_seconds}
        tmp = self.stats_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
        os.replace(tmp, self.stats_path)

    def run(self):
        files = selected_files(self.stream)[self.a.part::self.a.nparts]
        if self.a.max_files:
            files = files[:self.a.max_files]
        self.prepare_outputs()
        writer = documents.DocStoreWriter(self.store, self.a.part, flush_tokens=100_000_000)
        self.file_seconds = []
        t0 = time.time()
        for fi, path in enumerate(files):
            tf = time.time()
            for d in iter_jsonl_zst(path):
                self.doc_text(d)
                if len(self.pending) >= 64 or self.pending_chars >= 20_000_000:
                    self.flush(writer)
                if self.a.max_docs and self.st["examples_read"] >= self.a.max_docs:
                    break
            self.flush(writer)
            self.st["files_done"] += 1
            self.file_seconds.append([os.path.basename(path), round(time.time() - tf, 1)])
            self.save_stats(files, False, t0)
            print(f"{time.strftime('%H:%M:%S')} {self.stream} part {self.a.part}: file {fi + 1}/{len(files)} "
                  f"{os.path.basename(path)} {time.time() - tf:.0f}s; read {self.st['examples_read']}, "
                  f"written {self.st['docs_written']} docs / {self.st['tokens_written'] / 1e6:.1f}M tok", flush=True)
        res = writer.close()
        np.save(self.hash_path, np.array(self.kept_hashes, dtype=np.uint64))
        assert res["docs"] == self.st["docs_written"]
        self.save_stats(files, True, t0)
        print("DONE", json.dumps({"stats": dict(self.st), "drop_first_reason": dict(self.reasons_first)}), flush=True)


# ---------------------------------------------------------------- manifest and checks

RULES_TEXT = {
    "pdf": "clean: strip <table> blocks, '|' table rows, dot-leader TOC lines, lone page-number and separator lines, "
           "unwrap \\textsuperscript, cut at the last references heading if past 40% of the text, join mid-sentence "
           "line breaks (health only). drop (health+science audits, stricter threshold where they differ): English "
           "stopword share < 0.20, raw HTML-table char share > 0.15, raw dot-leader line share > 0.05, raw mid-sentence "
           "breaks > 10 per 1k words, dup13 > 0.08, duplicate-line chars > 0.03, short lines > 0.45, bullet lines > "
           "0.30, symbols > 0.12 or letters < 0.85, hyphen breaks > 1 per 1k words, Gopher-like fail; literature also "
           "(literature audit): fallback pages > 0, fastText English < 0.8, dup13 > 0.05, short lines > 0.35, symbols > "
           "0.10 or letters < 0.88, digits > 0.03, raw LaTeX marks > 2 per 1k chars. keep >= 8192 tokens.",
    "arxiv": "only proofpile-arXiv ids; drop: Gopher-like fail, dup13 > 0.10, duplicate-line chars > 0.05, symbols > 0.31 "
             "or letters < 0.66, prose chars < 0.5, math chars > 0.45, undefined-macro rate > 1 per 1k chars, "
             "preamble macros > 3, table chars > 0.15, command chars outside math > 0.08, comment lines > 2%; clean: strip "
             "table/tabular environments, cut a bibliography tail past 40%. keep >= 8192 tokens.",
}


def manifest(a):
    parts = []
    for p in range(a.nparts):
        path = part_output_paths(a.stream, p, a.test)[1]
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        parts.append(json.load(open(path)))
    not_done = [x["part"] for x in parts if not x["done"]]
    tot, first, anyr = Counter(), Counter(), Counter()
    examples = []
    for x in parts:
        tot.update(x["stats"])
        first.update(x["drop_first_reason"])
        anyr.update(x["fail_any_reason"])
        examples += x["examples"]
    out = {
        "stream": a.stream, "source_dirs": [os.path.join(RAW, d) for d in STREAMS[a.stream]["dirs"]],
        "files_selected": len(selected_files(a.stream)), "file_share": STREAMS[a.stream]["share"],
        "selection_seed": SELECTION_SEED, "nparts": a.nparts, "parts_not_done": not_done,
        "examples_read": tot["examples_read"], "docs_written": tot["docs_written"], "tokens_written": tot["tokens_written"],
        "drops": {k[5:]: v for k, v in tot.items() if k.startswith("drop_")},
        "drop_filters_first_reason": dict(first.most_common()), "fail_any_reason": dict(anyr.most_common()),
        "stats": dict(tot), "rules": RULES_TEXT[STREAMS[a.stream]["kind"]],
        "doc_store": os.path.join(ROOT, "work", "docs", a.stream), "examples_first_300_chars": examples[:3],
        "minutes_per_part": [x["minutes"] for x in parts],
    }
    mdir = os.path.join(WORK, "test") if a.test else os.path.join(ROOT, "manifests", "build")
    os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, f"{a.stream}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    print(json.dumps({k: out[k] for k in ("examples_read", "docs_written", "tokens_written", "drops", "parts_not_done")}))


def crossdup(a):
    """Counts written documents whose text also appears in another part of the same stream."""
    d = os.path.dirname(part_output_paths(a.stream, 0, a.test)[2])
    arrs = [np.load(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".npy")]
    allh = np.concatenate(arrs) if arrs else np.zeros(0, np.uint64)
    u = np.unique(allh)
    print(json.dumps({"parts": len(arrs), "docs": int(allh.size), "unique": int(u.size), "cross_part_duplicates": int(allh.size - u.size)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "manifest", "crossdup", "files"])
    ap.add_argument("--stream", required=True, choices=sorted(STREAMS))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--max-files", type=int, default=0, help="test runs: only the first K files of the part")
    ap.add_argument("--max-docs", type=int, default=0, help="test runs: stop a file after this many documents in total")
    ap.add_argument("--test", action="store_true", help="write to work/pdf_arxiv/test instead of the real store")
    ap.add_argument("--probe", action="store_true", help="tokenize documents just below the length prefilter")
    ap.add_argument("--overwrite", action="store_true", help="remove this part's earlier outputs first")
    a = ap.parse_args()
    if a.cmd == "build":
        Part(a).run()
    elif a.cmd == "manifest":
        manifest(a)
    elif a.cmd == "crossdup":
        crossdup(a)
    else:
        fs = selected_files(a.stream)
        print(len(fs), "of", len(all_files(a.stream)), "files selected")
        for f in fs[:5]:
            print(" ", f)
