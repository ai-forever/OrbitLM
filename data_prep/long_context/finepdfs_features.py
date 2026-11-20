"""Cheap text statistics for FinePDFs documents.

The thresholds of build_finepdfs_ru.py were chosen by measuring these statistics on four sample
files of the source. The build computes them with this same code, so each threshold means
exactly what it meant on those samples.

Characters are classified with a lookup table over code points in numpy, because a Python loop
over hundreds of millions of characters would take hours.
"""
import re

import numpy as np
import zstandard

# Character classes. A class is one small integer, and the lookup table maps every code point of
# the Basic Multilingual Plane to its class. Code points above it count as OTHER.
WS, CYR, LAT, LATX, DIG, PUN, GRK, MTH, OTHER = range(9)
NCLS = 9


def _build_lut():
    lut = np.full(0x10000, OTHER, np.uint8)
    for c in (9, 10, 11, 12, 13, 32, 0xA0, 0x202F, 0x3000):
        lut[c] = WS
    lut[0x2000:0x200C] = WS
    lut[0x0400:0x0500] = CYR
    lut[0x41:0x5B] = LAT
    lut[0x61:0x7B] = LAT
    # Latin-1 and Latin Extended letters. In a Russian document a lot of them usually means the
    # text layer was decoded with the wrong code page (for example "Ïðîåêò" instead of "Проект").
    lut[0xC0:0x250] = LATX
    lut[0x30:0x3A] = DIG
    for lo, hi in ((0x21, 0x30), (0x3A, 0x41), (0x5B, 0x61), (0x7B, 0x7F), (0xA1, 0xC0),
                   (0x2010, 0x2060), (0x20A0, 0x20D0), (0x2100, 0x2150), (0x25A0, 0x2600)):
        lut[lo:hi] = PUN
    lut[0xD7] = PUN   # the multiplication sign sits inside the Latin-1 letter block
    lut[0xF7] = PUN   # and so does the division sign
    lut[0x0370:0x0400] = GRK
    for lo, hi in ((0x2070, 0x20A0), (0x2150, 0x2190), (0x2190, 0x2400)):
        lut[lo:hi] = MTH
    return lut


LUT = _build_lut()

# The FinePDFs anonymizer replaced every public-range IPv4-looking string with one of these six
# addresses. In Russian documents such strings are mostly clause numbers like "4.12.1.3", so a
# document full of these addresses has had its numbering destroyed.
FAKE_IPS = ("22.214.171.124", "126.96.36.199", "188.8.131.52", "184.108.40.206", "220.127.116.11", "18.104.22.168")

_ZC = zstandard.ZstdCompressor(level=3)

# Bureaucratic documents (school and university work programs, orders, regulations, reports) get
# high edu scores because they are about education, but they are templates full of officialese.
# Their title page carries one of these phrases.
HEAD_CHARS = 3000
OFFICIAL_WINDOW = 10000
RX_PROGRAM = re.compile(r"рабоч\w* программ|программ\w* (учебной )?дисциплин|фонд\w* оценочных|"
                        r"календарно-тематич|тематическое планирование|учебн\w* план")
RX_STAMP = re.compile(r"утверждаю|утвержден\w*|рассмотрен\w*|согласован\w*|принят\w* на (заседании|педагогическом)|"
                      r"протокол\w*\s*(№|от)|приказ\w*\s*(№|от)|постановлени\w*\s*(№|от)")
# Officialese vocabulary counted over the whole text; its density separates regulations and
# templates from textbooks and articles that merely mention a law once.
RX_OFFICIAL = re.compile(r"в соответствии с|российской федерации|федеральн\w* закон|постановлени\w*|приказ\w*|"
                         r"утвержд\w*|обучающи\w*|компетенци\w*|планируемы\w* результат|фгос|нормативн\w*|"
                         r"настоящ\w* (положени|приказ|постановлени|договор|правил|порядк|стандарт)|"
                         r"не позднее|осуществля\w*|заявител\w*|муниципальн\w*|учреждени\w*|"
                         r"должностн\w* лиц|органами? (местного|государственн)|в установленном порядке")

FEATURES = ("chars", "nonws", "cyr_share", "letter_share", "digit_share", "garbage", "latx_share",
            "math_share", "lines", "mean_line", "dup_line_frac", "short_line_frac", "table_frac",
            "html_tags", "fake_ip", "fake_ip_per_10k", "zstd_ratio", "dollar_per_10k",
            "head_program", "head_stamp", "official_per_10k")


def text_features(s):
    """Returns a dict of the statistics named in FEATURES for one document text."""
    cp = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32)
    big = cp >= 0x10000
    cls = LUT[np.minimum(cp, 0xFFFF)]
    if big.any():
        cls = cls.copy()
        cls[big] = OTHER
    cnt = np.bincount(cls, minlength=NCLS)
    n = int(cp.size)
    nonws = max(n - int(cnt[WS]), 1)
    letters = int(cnt[CYR] + cnt[LAT] + cnt[LATX])
    f = {
        "chars": n,
        "nonws": nonws,
        "cyr_share": cnt[CYR] / max(letters, 1),
        "letter_share": (letters + cnt[GRK]) / nonws,
        "digit_share": cnt[DIG] / nonws,
        "garbage": cnt[OTHER] / nonws,
        "latx_share": cnt[LATX] / max(letters, 1),
        "math_share": (cnt[GRK] + cnt[MTH]) / nonws,
    }
    # Line statistics. A line that repeats an earlier line of the same document is counted as
    # duplicated; running headers, OCR loops and form templates show up here.
    seen = set()
    nl = dup_chars = line_chars = short = table = 0
    for ln in s.split("\n"):
        t = ln.strip()
        if not t:
            continue
        nl += 1
        L = len(t)
        line_chars += L
        if L < 25:
            short += 1
        if t[0] == "|":
            table += L
        if L >= 10:
            if t in seen:
                dup_chars += L
            else:
                seen.add(t)
    f["lines"] = nl
    f["mean_line"] = line_chars / max(nl, 1)
    f["dup_line_frac"] = dup_chars / max(line_chars, 1)
    f["short_line_frac"] = short / max(nl, 1)
    f["table_frac"] = table / max(line_chars, 1)
    f["html_tags"] = s.count("<td") + s.count("<tr") + s.count("<th")
    ips = sum(s.count(ip) for ip in FAKE_IPS)
    f["fake_ip"] = ips
    f["fake_ip_per_10k"] = ips * 10000 / max(n, 1)
    raw = s.encode("utf-8")
    f["zstd_ratio"] = len(_ZC.compress(raw)) / max(len(raw), 1)
    f["dollar_per_10k"] = s.count("$") * 10000 / max(n, 1)
    head = s[:HEAD_CHARS].lower()
    f["head_program"] = int(RX_PROGRAM.search(head) is not None)
    f["head_stamp"] = int(RX_STAMP.search(head) is not None)
    # The officialese density is measured on three 10K-character windows (start, middle and
    # three quarters in) instead of the whole text: running the regex over megabyte-long
    # documents was the slowest step, and these documents keep one register throughout.
    if n <= 3 * OFFICIAL_WINDOW:
        sample = s
    else:
        sample = "\n".join(s[p:p + OFFICIAL_WINDOW] for p in (0, n // 2, 3 * n // 4))
    f["official_per_10k"] = sum(1 for _ in RX_OFFICIAL.finditer(sample.lower())) * 10000 / max(len(sample), 1)
    return f
