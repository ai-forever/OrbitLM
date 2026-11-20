"""Builds the code_repos stream: one long document per source-code repository.

The sources are the Python and Java parts of Stack-Edu in the dolma mix. There, every file is a
separate record and the files of one repository are scattered across hundreds of input files, so
the build has two passes.

  route   (pass 1) reads input files, drops repeats and files with int_score below 3, and appends
          every kept file to one of NB buckets chosen by a hash of its repository name. All files
          of a repository therefore end up in the same bucket.
  build   (pass 2) reads one bucket at a time, groups its files by repository, orders them by
          path, joins each repository into one document with a header line per file, tokenizes
          it, drops files that look like encoded or generated data, cuts very large repositories,
          and keeps repositories of at least 8,192 tokens. The limits are explained at their
          constants below.
  manifest  sums the statistics of both passes into manifests/build/code_repos.json.

Both passes split with --part P --nparts N: route takes every Nth input file of one language,
build takes every Nth bucket of both languages. A build unit (one bucket) writes its own doc
store part, numbered by the unit, and a small statistics file; units that already have their
statistics file are skipped, so a crashed build can simply be started again.

Usage:
  build_code_repos.py route --lang Python --part P --nparts N
  build_code_repos.py build --part P --nparts N
  build_code_repos.py manifest
"""
import argparse
import glob
import hashlib
import io
import json
import os
import resource
import time
from collections import Counter, defaultdict

import numpy as np
import zstandard

import documents
import paths

ROOT = paths.CTX16K_ROOT
SRC = paths.DOLMA3_EXCLUDED
LANGS = ("Python", "Java")
HEADER = {"Python": "# File: {}", "Java": "// File: {}"}
NB = 256
MIN_SCORE = 3
MIN_TOKENS = 8192

# Real code averages 3.7 (Python) to 4.1 (Java) bytes per token with this tokenizer. Files under
# 1.5 bytes per token are dropped: they are encoded binary blobs (get-pip.py alone is a 1M-token
# base85 string), resource dumps, or long tables of numbers, whose digits are one token each.
# Code with CJK comments stays above this line, at about 1.9 bytes per token.
MIN_BYTES_PER_TOKEN = 1.5

# Files over 100,000 tokens (about 400 KB) are dropped as generated data. In a sample of the
# largest repositories these were animation frames written out as code, translation tables and
# game data tables; hand-written modules in the sample stayed under 75,000 tokens.
MAX_FILE_TOKENS = 100_000

# A repository document is cut after the last whole file that fits in 1,048,576 tokens (64
# blocks). Without the cap one repository of 58,000 Code Jam solutions alone was 21M tokens,
# 3.5% of the stream, and would have filled 1,300 consecutive blocks with near-duplicate programs.
# The build options --max-file-tokens and --max-repo-tokens change both limits (0 turns one off).
MAX_REPO_TOKENS = 1 << 20

# Because every kept file has at least 1.5 bytes per token, a repository under
# 8,192 * 1.5 = 12,288 bytes can never reach 8,192 tokens. Such repositories are dropped without
# tokenizing them, which skips most of the tokenizer work. The build --audit flag tokenizes them
# anyway and counts how many would have passed, which must be zero.
MIN_BYTES = 12288

# Pass 1 keeps one output buffer per bucket and writes it as an independent zstd frame once it
# reaches this size. Concatenated frames form a valid zstd stream, and 256 buffers of 2 MB keep
# the process well under its memory cap.
BUCKET_FLUSH = 2 << 20

# Pass 2 tokenizes the files of many repositories in one batch, so the tokenizer's threads have
# enough independent pieces of work. The batch is bounded by text size to bound its memory.
TOKENIZE_BATCH_BYTES = 16 << 20


def h64(b):
    return hashlib.blake2b(b, digest_size=8).digest()


def input_files(lang):
    return sorted(glob.glob(os.path.join(SRC, f"stack_edu-{lang}", "*.jsonl.zst")))


def iter_zst_lines(path):
    with open(path, "rb") as f:
        reader = io.BufferedReader(zstandard.ZstdDecompressor(max_window_size=1 << 31).stream_reader(f), 1 << 20)
        for line in reader:
            yield line


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------- pass 1: route files into buckets

def route(a):
    files = input_files(a.lang)[a.part::a.nparts]
    if a.limit_files:
        files = files[:a.limit_files]
    tag = f"p{a.part:04d}of{a.nparts:04d}"
    stats_dir = os.path.join(a.work, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    bdirs = [os.path.join(a.work, "buckets", a.lang, f"b{b:03d}") for b in range(NB)]
    for d in bdirs:
        os.makedirs(d, exist_ok=True)
    # Outputs are written under a .tmp name and renamed only when the whole part has finished,
    # so pass 2 never reads a bucket file that a crashed or running pass 1 left half written.
    outs = [open(os.path.join(d, f"{tag}.jsonl.zst.tmp"), "wb") for d in bdirs]
    bufs = [[] for _ in range(NB)]
    sizes = [0] * NB
    cctx = zstandard.ZstdCompressor(level=3)

    def flush(b):
        if bufs[b]:
            outs[b].write(cctx.compress(b"".join(bufs[b])))
            bufs[b].clear()
            sizes[b] = 0

    st = Counter()
    seen = set()
    t0 = time.time()
    for fi, path in enumerate(files):
        prev = None
        for line in iter_zst_lines(path):
            st["lines"] += 1
            # The raw data repeats every record 2 to 4 times in a row. Comparing a hash of the
            # raw line with the previous one skips those repeats without parsing their JSON.
            lh = h64(line)
            if lh == prev:
                st["drop_repeated_line"] += 1
                continue
            prev = lh
            d = json.loads(line)
            m = d.get("metadata") or d  # Java records keep the fields under "metadata"
            repo, fpath, score, text = m.get("repo_name"), m.get("path"), m.get("int_score"), d.get("text")
            if not repo or not fpath or score is None or text is None:
                st["drop_missing_fields"] += 1
                continue
            if score < MIN_SCORE:
                st["drop_low_score"] += 1
                continue
            if not text.strip():
                st["drop_empty_file"] += 1
                continue
            try:
                tb = text.encode("utf-8")
            except UnicodeEncodeError:
                # Lone surrogates from the JSON escapes cannot be encoded or tokenized.
                text = text.encode("utf-8", "replace").decode("utf-8")
                tb = text.encode("utf-8")
                st["fixed_surrogates"] += 1
            th = h64(tb)
            # Exact deduplication of file contents. The source turned out to be deduplicated by
            # content already (no repeated text across repositories in a 950K-file sample), so
            # this set only guards against leftovers seen by the same process.
            key = int.from_bytes(th, "little")
            if key in seen:
                st["drop_duplicate_text"] += 1
                continue
            seen.add(key)
            b = int.from_bytes(h64(repo.encode("utf-8", "replace")), "little") % NB
            rec = json.dumps({"r": repo, "p": fpath, "h": th.hex(), "t": text}, ensure_ascii=False)
            rb = rec.encode("utf-8", "replace") + b"\n"
            bufs[b].append(rb)
            sizes[b] += len(rb)
            if sizes[b] >= BUCKET_FLUSH:
                flush(b)
            st["kept_files"] += 1
            st["kept_text_bytes"] += len(tb)
        st["input_files_done"] += 1
        el = time.time() - t0
        print(f"{time.strftime('%H:%M:%S')} route {a.lang} {tag}: {fi + 1}/{len(files)} files, "
              f"{el / 60:.1f} min, ETA {el / (fi + 1) * (len(files) - fi - 1) / 60:.0f} min, {json.dumps(st)}", flush=True)
    for b in range(NB):
        flush(b)
        outs[b].close()
        final = os.path.join(bdirs[b], f"{tag}.jsonl.zst")
        os.replace(final + ".tmp", final)
    atomic_json({"lang": a.lang, "part": a.part, "nparts": a.nparts, "files": [os.path.basename(f) for f in files],
                 "limit_files": a.limit_files, "minutes": round((time.time() - t0) / 60, 1), "stats": dict(st)},
                os.path.join(stats_dir, f"route_{a.lang}_{tag}.json"))
    print("DONE", json.dumps(st), flush=True)


def check_route_complete(work, lang, allow_partial):
    """Refuses to build a language whose pass 1 has not covered every input file."""
    stats = [json.load(open(p)) for p in glob.glob(os.path.join(work, "stats", f"route_{lang}_p*.json"))]
    if not stats:
        raise SystemExit(f"no finished pass-1 parts for {lang}")
    done = sorted(f for s in stats for f in s["files"])
    expected = sorted(os.path.basename(f) for f in input_files(lang))
    if done != expected and not allow_partial:
        raise SystemExit(f"pass 1 for {lang} covered {len(done)} of {len(expected)} input files "
                         f"(or covered some twice); finish it, or pass --allow-partial for a test")
    return len(done), len(expected)


# ---------------------------------------------------------------- pass 2: build repository documents

def render_file(lang, path, text):
    """Returns one file's part of the repository document: a header line with the path, then the
    file text without its leading blank lines and trailing whitespace."""
    body = text.lstrip("\r\n").rstrip()
    return documents.clean_text(HEADER[lang].format(path.lstrip("/")) + "\n" + body)


# The files of a repository are joined with a blank line. Each file is tokenized on its own, with
# the blank line appended, because the tokenizer's pre-splitting always cuts between trailing
# newlines and the next header's first character. The concatenated token ids are therefore the
# same as for the whole joined text (checked on 200 repositories), no huge string has to be
# tokenized at once, and single files can be dropped after tokenizing without redoing the rest.
SEP = "\n\n"


def build_unit(a, lang, b, uid, tok):
    bdir = os.path.join(a.work, "buckets", lang, f"b{b:03d}")
    st = Counter()
    hist = Counter()
    samples = []
    repos = defaultdict(list)
    for fpath in sorted(glob.glob(os.path.join(bdir, "*.jsonl.zst"))):
        for line in iter_zst_lines(fpath):
            d = json.loads(line)
            repos[d["r"]].append((d["p"], d["h"], d["t"]))
    # Stale output of an earlier crashed attempt at this unit is removed, because the store is
    # read as "every part file in the directory" and a leftover file would duplicate documents.
    for old in glob.glob(os.path.join(a.store, f"part_{uid:04d}_*")):
        os.remove(old)
    writer = documents.DocStoreWriter(a.store, uid)
    pending, pending_bytes = [], 0

    def tokenize_pending():
        encs = tok.encode_batch_fast([c + SEP for _, ch, _ in pending for c in ch], add_special_tokens=False)
        k = 0
        for name, ch, text_bytes in pending:
            ids, kept = [], []
            for j, c in enumerate(ch):
                x = np.asarray(encs[k + j].ids, dtype=np.uint32)
                if len(x) and len((c + SEP).encode("utf-8")) / len(x) < MIN_BYTES_PER_TOKEN:
                    st["drop_low_bytes_per_token_files"] += 1
                    st["drop_low_bytes_per_token_tokens"] += len(x)
                    continue
                if a.max_file_tokens and len(x) > a.max_file_tokens:
                    st["drop_huge_files"] += 1
                    st["drop_huge_files_tokens"] += len(x)
                    continue
                ids.append(x)
                kept.append(c)
            k += len(ch)
            if a.max_repo_tokens:
                total, cut = 0, len(ids)
                for j, x in enumerate(ids):
                    if total + len(x) > a.max_repo_tokens - 1:  # one token is left for the EOS
                        cut = j
                        break
                    total += len(x)
                if cut < len(ids):
                    st["truncated_repos"] += 1
                    st["truncated_files"] += len(ids) - cut
                    st["truncated_tokens"] += sum(len(x) for x in ids[cut:])
                    ids, kept = ids[:cut], kept[:cut]
            if kept:
                # The last file must not end with the separator. Its tokens are recomputed from
                # the bare text, because the separator's newlines can merge into the token before.
                ids[-1] = np.asarray(tok.encode(kept[-1], add_special_tokens=False).ids, dtype=np.uint32)
            n = int(sum(len(x) for x in ids))
            if text_bytes < MIN_BYTES:  # only reached with --audit
                st["audit_short_bytes_repos_tokenized"] += 1
                if n >= MIN_TOKENS:
                    st["audit_short_bytes_would_pass"] += 1
                continue
            if n < MIN_TOKENS:
                st["drop_short_tokens_repos"] += 1
                st["drop_short_tokens_tokens"] += n
                continue
            ids.append(np.array([documents.EOS], dtype=np.uint32))
            doc = np.concatenate(ids)
            writer.add(doc)
            st["kept_repos"] += 1
            st["kept_files"] += len(kept)
            st["kept_tokens"] += len(doc)
            st["max_repo_tokens"] = max(st["max_repo_tokens"], len(doc))
            hist[size_bin(len(doc))] += 1
            if len(samples) < 3:
                samples.append({"repo": name, "tokens": len(doc), "head": SEP.join(kept[:3])[:300]})
        pending.clear()

    for name in sorted(repos):
        recs = sorted(repos[name])
        st["repos_in"] += 1
        st["files_in"] += len(recs)
        files, seen, prev_path = [], set(), None
        for path, th, text in recs:
            # Identical contents inside one repository (a file copied to a second path, or a
            # record that reached this bucket from two pass-1 parts) are kept once, at the first
            # path in sorted order.
            if th in seen:
                st["drop_duplicate_in_repo_files"] += 1
                continue
            seen.add(th)
            # Two different contents under one path are two versions of the same file. Only
            # one is kept, because the second would be a near copy under a repeated header.
            if path == prev_path:
                st["drop_same_path_files"] += 1
                continue
            prev_path = path
            files.append((path, text))
        chunks = [render_file(lang, p, t) for p, t in files]
        text_bytes = sum(len(c.encode("utf-8")) for c in chunks) + len(SEP) * (len(chunks) - 1)
        if text_bytes < MIN_BYTES:
            st["drop_short_bytes_repos"] += 1
            st["drop_short_bytes_files"] += len(files)
            st["drop_short_bytes_bytes"] += text_bytes
            if not a.audit:
                continue
        else:
            st["tokenized_repos"] += 1
        pending.append((name, chunks, text_bytes))
        pending_bytes += text_bytes
        if pending_bytes >= TOKENIZE_BATCH_BYTES:
            tokenize_pending()
            pending_bytes = 0
    if pending:
        tokenize_pending()
    del repos
    res = writer.close()
    assert res["docs"] == st["kept_repos"]
    return st, hist, samples


def size_bin(n):
    for lim, name in ((16384, "8K-16K"), (32768, "16K-32K"), (65536, "32K-64K"), (131072, "64K-128K"),
                      (262144, "128K-256K"), (1 << 20, "256K-1M"), (4 << 20, "1M-4M")):
        if n < lim:
            return name
    return ">=4M"


def build(a):
    for lang in LANGS:
        done, expected = check_route_complete(a.work, lang, a.allow_partial)
        print(f"pass 1 for {lang}: {done}/{expected} input files routed", flush=True)
    units = [(li, lang, b) for li, lang in enumerate(LANGS) for b in range(NB)][a.part::a.nparts]
    if a.limit_units:
        units = units[:a.limit_units]
    stats_dir = a.stats_dir or os.path.join(a.work, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(a.store, exist_ok=True)
    tok = documents.tokenizer()
    t0 = time.time()
    tot = Counter()
    todo = [u for u in units if not os.path.exists(os.path.join(stats_dir, f"build_{u[1]}_b{u[2]:03d}.json"))]
    print(f"build part {a.part}/{a.nparts}: {len(units)} units, {len(units) - len(todo)} already done", flush=True)
    for i, (li, lang, b) in enumerate(todo):
        uid = li * NB + b
        ts = time.time()
        st, hist, samples = build_unit(a, lang, b, uid, tok)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        atomic_json({"lang": lang, "bucket": b, "store_part": uid, "audit": a.audit,
                     "max_file_tokens": a.max_file_tokens, "max_repo_tokens": a.max_repo_tokens, "minutes": round((time.time() - ts) / 60, 2), "max_rss_mb": rss,
                     "stats": dict(st), "hist": dict(hist), "samples": samples},
                    os.path.join(stats_dir, f"build_{lang}_b{b:03d}.json"))
        tot.update({k: v for k, v in st.items() if k != "max_repo_tokens"})  # a maximum does not add up
        el = time.time() - t0
        print(f"{time.strftime('%H:%M:%S')} build part {a.part}: {i + 1}/{len(todo)} units ({lang} b{b:03d} "
              f"{time.time() - ts:.0f}s), {el / 60:.1f} min, ETA {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min, "
              f"kept {tot['kept_repos']} repos {tot['kept_tokens'] / 1e9:.3f}B tokens, max RSS {rss} MB", flush=True)
    print("DONE", json.dumps(tot), flush=True)


# ---------------------------------------------------------------- manifest

def manifest(a):
    route_dir = os.path.join(a.work, "stats")
    build_dir = a.stats_dir or route_dir
    route_st, build_st, hist, samples = {}, {}, {}, []
    # The limits each build unit actually used; more than one value means mixed runs.
    limits = {"max_file_tokens": set(), "max_repo_tokens": set()}
    for lang in LANGS:
        rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(route_dir, f"route_{lang}_p*.json")))]
        bs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(build_dir, f"build_{lang}_b*.json")))]
        route_st[lang] = dict(sum((Counter(s["stats"]) for s in rs), Counter()), input_files_expected=len(input_files(lang)))
        c = Counter()
        mx = 0
        for s in bs:
            for k in limits:
                limits[k].add(s.get(k))
            mx = max(mx, s["stats"].get("max_repo_tokens", 0))
            c.update({k: v for k, v in s["stats"].items() if k != "max_repo_tokens"})
        build_st[lang] = dict(c, max_repo_tokens=mx, units_done=len(bs), units_expected=NB)
        hist[lang] = dict(sum((Counter(s["hist"]) for s in bs), Counter()))
        samples.append([dict(x, lang=lang) for s in bs for x in s["samples"][:1]])
    # Three examples, alternating languages: Python, Java, Python.
    samples = [x for pair in zip(*samples) for x in pair][:3] or [x for lst in samples for x in lst][:3]
    docs = sum(v.get("kept_repos", 0) for v in build_st.values())
    toks = sum(v.get("kept_tokens", 0) for v in build_st.values())
    out = {
        "stream": "code_repos",
        "store": a.store,
        "sources": {lang: os.path.join(SRC, f"stack_edu-{lang}") for lang in LANGS},
        "rules": {"min_int_score": MIN_SCORE, "min_repo_tokens_excluding_eos": MIN_TOKENS,
                  "drop_files_below_bytes_per_token": MIN_BYTES_PER_TOKEN,
                  "drop_files_above_tokens": sorted(limits["max_file_tokens"]),
                  "cut_repos_above_tokens": sorted(limits["max_repo_tokens"]),
                  "skip_tokenizing_below_bytes": MIN_BYTES, "buckets": NB,
                  "dedup": "adjacent repeated lines; exact text hash over each pass-1 process (global when route runs with --nparts 1 per language); exact text hash within each repository; one version per path",
                  "document": "files of one repository ordered by path, each preceded by '# File: <path>' (Python) or '// File: <path>' (Java), joined by blank lines, one EOS at the end"},
        "examples_read": sum(v.get("lines", 0) for v in route_st.values()),
        "route": route_st,
        "build": build_st,
        "repo_token_histogram": hist,
        "docs_written": docs,
        "tokens_written": toks,
        "samples": samples,
    }
    path = a.manifest
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_json(out, path)
    print(json.dumps({k: out[k] for k in ("examples_read", "docs_written", "tokens_written")}), "->", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["route", "build", "manifest"])
    ap.add_argument("--lang", choices=LANGS)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--work", default=os.path.join(ROOT, "work", "code"))
    ap.add_argument("--store", default=os.path.join(ROOT, "work", "docs", "code_repos"))
    ap.add_argument("--stats-dir", default="", help="where build writes its per-unit statistics (default WORK/stats; a test uses its own)")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "manifests", "build", "code_repos.json"),
                    help="where the manifest goes (a test writes it into its own work directory)")
    ap.add_argument("--limit-files", type=int, default=0, help="route only the first K input files of the part (tests)")
    ap.add_argument("--limit-units", type=int, default=0, help="build only the first K buckets of the part (tests)")
    ap.add_argument("--allow-partial", action="store_true", help="build although pass 1 is incomplete (tests)")
    ap.add_argument("--audit", action="store_true", help="also tokenize repositories under MIN_BYTES and count would-be passes")
    ap.add_argument("--max-file-tokens", type=int, default=MAX_FILE_TOKENS, help="drop files longer than this (0: no limit)")
    ap.add_argument("--max-repo-tokens", type=int, default=MAX_REPO_TOKENS, help="cut repositories after the last whole file that fits (0: no limit)")
    a = ap.parse_args()
    if a.cmd == "route":
        if not a.lang:
            ap.error("route needs --lang")
        route(a)
    elif a.cmd == "build":
        build(a)
    else:
        manifest(a)
