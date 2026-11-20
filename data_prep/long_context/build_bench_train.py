"""Renders the train splits of the evaluation benchmarks into the bench_train document store.

Every benchmark item becomes one plain-text document in the evaluation style, not a chat:
  multiple choice:  "Question: ...\\nA. ...\\nB. ...\\nAnswer: B"
  GSM8K and MATH:   "Question: ...\\nAnswer: <full solution>" (GSM8K keeps its "#### N" last line)
  DROP:             "Passage: ...\\nQuestion: ...\\nAnswer: ...\\n\\nQuestion: ...\\nAnswer: ..."
DROP asks about a dozen questions per passage. Writing the passage once with all of its
questions keeps the passage from being repeated a dozen times in the stream.
The 16K mix (stage_16k.MIX) gives this store about five passes, so each item is written exactly
once here and the repetition comes from the mix.

Items are deduplicated exactly inside each source by a hash of the normalized prompt (the
question with its options, or the DROP passage; DROP questions are deduplicated inside their
passage). Documents longer than 16,384 tokens are dropped and counted. The documents of all
sources are shuffled together with a fixed seed before writing, so that a packed block later
holds a mix of benchmarks instead of a run of one benchmark.

Usage:
  build_bench_train.py [--part P --nparts N] [--sources gsm8k,math,...] [--limit K] [--out DIR]
                 [--overwrite]
--part/--nparts split the list of input files (each part takes every Nth file). Deduplication
only sees the files of its own part, so pack.py's exact document dedup catches the rest.
--limit reads only the first K rows of every file, for tests.
"""
import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import defaultdict

import pyarrow.parquet as pq

import documents
import paths

ROOT = paths.CTX16K_ROOT
RAW = paths.RAW_HF
MAX_TOKENS = 16384
LETTERS = "ABCDEFGHIJ"


class Drop(Exception):
    """Raised by a renderer to skip an item; the message is the drop reason for the statistics."""


def norm(s):
    return " ".join((s or "").lower().split())


def mcq(question, options, answer_index):
    """Renders one multiple-choice item. The answer is given by its position in options."""
    question = (question or "").strip()
    options = [(o or "").strip() for o in options]
    if not question or not options or any(not o for o in options):
        raise Drop("empty_field")
    if not (0 <= answer_index < len(options)) or len(options) > len(LETTERS):
        raise Drop("bad_label")
    lines = [f"Question: {question}"]
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    lines.append(f"Answer: {LETTERS[answer_index]}")
    return "\n".join(lines), norm(question) + "\x00" + "\x00".join(norm(o) for o in options)


def labelled_mcq(question, choices, answer_key):
    """Renders an item whose choices carry their own labels (ARC, OpenBookQA, CommonsenseQA).

    Some ARC items are labelled 1-4 instead of A-D. The options are always re-lettered A, B, C
    by position, and the answer letter follows the position of the answer key among the labels.
    """
    labels = list(choices["label"])
    if answer_key not in labels:
        raise Drop("bad_label")
    return mcq(question, choices["text"], labels.index(answer_key))


# ---------------------------------------------------------------- renderers, one per source

_CALC = re.compile(r"<<[^<>]*>>")


def r_gsm8k(r):
    # GSM8K solutions carry calculator annotations like <<48/2=24>> that are not part of the
    # written solution, so they are removed. The last line "#### N" is kept as it is, because
    # lm-eval's strict-match scoring looks for exactly that line.
    q, a = (r["question"] or "").strip(), (r["answer"] or "").strip()
    body, sep, final = a.rpartition("####")
    if not q or not sep or not final.strip():
        raise Drop("no_final_answer")
    body = _CALC.sub("", body).strip()
    text = f"Question: {q}\nAnswer: {body}\n#### {final.strip()}"
    return text, norm(q)


def r_math(r):
    q, s = (r["problem"] or "").strip(), (r["solution"] or "").strip()
    if not q or not s:
        raise Drop("empty_field")
    return f"Question: {q}\nAnswer: {s}", norm(q)


def r_mmlu(r):
    return mcq(r["question"], r["choices"], int(r["answer"]))


def r_arc(r):
    return labelled_mcq(r["question"], r["choices"], r["answerKey"])


def r_obqa(r):
    return labelled_mcq(r["question_stem"], r["choices"], r["answerKey"])


def r_csqa(r):
    return labelled_mcq(r["question"], r["choices"], r["answerKey"])


def _hs_clean(text):
    # The same cleanup lm-eval applies to HellaSwag: WikiHow items carry markers such as
    # [title] and [step] that are not part of the text.
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def r_hellaswag(r):
    # Some items have a lone "." as the first half of the context, which would otherwise
    # render as "Painting: . A lady ..."; such a placeholder is dropped.
    ctx_a = (r["ctx_a"] or "").strip()
    ctx_a = "" if ctx_a in (".", "") else ctx_a
    ctx = _hs_clean(f"{r['activity_label']}: {ctx_a} {(r['ctx_b'] or '').capitalize()}")
    if not str(r["label"]).strip().isdigit():
        raise Drop("bad_label")
    return mcq(ctx, [_hs_clean(e) for e in r["endings"]], int(r["label"]))


def r_winogrande(r):
    if str(r["answer"]) not in ("1", "2"):
        raise Drop("bad_label")
    return mcq(r["sentence"], [r["option1"], r["option2"]], int(r["answer"]) - 1)


def r_piqa(r):
    return mcq(r["goal"], [r["sol1"], r["sol2"]], int(r["label"]))


def drop_passages(rows, st):
    """Yields one DROP document per passage: the passage, then all of its question/answer pairs.

    The first pair follows the passage on the next line, as in the single-question evaluation
    prompt, and further pairs are separated by blank lines. Questions are deduplicated inside
    their passage. A passage whose document would exceed 16,384 tokens has its questions split
    into halves, each half written with its own copy of the passage.
    """
    groups = {}  # normalized passage -> (passage, {normalized question: pair text})
    for r in rows:
        st["read"] += 1
        passage, q = (r["passage"] or "").strip(), (r["question"] or "").strip()
        # A multi-span answer lists every span, the way lm-eval joins them for DROP targets.
        spans = [s.strip() for s in (r["answers_spans"] or {}).get("spans") or [] if s and s.strip()]
        spans = list(dict.fromkeys(spans))
        if not passage or not q:
            st["drop_empty_field"] += 1
            continue
        if not spans:
            st["drop_no_answer"] += 1
            continue
        g = groups.setdefault(norm(passage), (passage, {}))
        if norm(q) in g[1]:
            st["drop_duplicate_question"] += 1
            continue
        g[1][norm(q)] = f"Question: {q}\nAnswer: {', '.join(spans)}"

    def render(passage, pairs):
        return f"Passage: {passage}\n" + "\n\n".join(pairs)

    for key, (passage, qa) in groups.items():
        pending = [list(qa.values())]
        k = 0
        while pending:
            pairs = pending.pop(0)
            text = render(passage, pairs)
            if len(pairs) > 1 and len(documents.encode_docs([text])[0]) - 1 > MAX_TOKENS:
                st["split_passages"] += 1
                half = len(pairs) // 2
                pending[:0] = [pairs[:half], pairs[half:]]
                continue
            st["questions"] += len(pairs)
            yield text, f"{key}\x00{k}"
            k += 1


def per_row(render):
    """Turns a one-row renderer into an item generator that counts reads and drops."""
    def items(rows, st):
        for r in rows:
            st["read"] += 1
            try:
                yield render(r)
            except Drop as e:
                st[f"drop_{e}"] += 1
    return items


MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]

# Each source has an item generator, which turns rows into (text, dedup key) pairs, and a list
# of input files relative to raw_hf. MMLU is read from the flat copy under all/, which holds
# the same 99,842 rows as auxiliary_train/ without the extra struct wrapper.
SOURCES = {
    "gsm8k": (per_row(r_gsm8k), ["openai__gsm8k/main/train-00000-of-00001.parquet"]),
    "math": (per_row(r_math), [f"EleutherAI__hendrycks_math/{s}/train-00000-of-00001.parquet" for s in MATH_SUBJECTS]),
    "mmlu_aux": (per_row(r_mmlu), ["cais__mmlu/all/auxiliary_train-00000-of-00001.parquet"]),
    "arc": (per_row(r_arc), ["allenai__ai2_arc/ARC-Challenge/train-00000-of-00001.parquet",
                    "allenai__ai2_arc/ARC-Easy/train-00000-of-00001.parquet"]),
    "hellaswag": (per_row(r_hellaswag), ["Rowan__hellaswag/data/train-00000-of-00001.parquet"]),
    "winogrande": (per_row(r_winogrande), ["allenai__winogrande/winogrande_xl/train-00000-of-00001.parquet"]),
    "piqa": (per_row(r_piqa), ["ybisk__piqa/plain_text/train/0000.parquet"]),
    "openbookqa": (per_row(r_obqa), ["allenai__openbookqa/main/train-00000-of-00001.parquet"]),
    "commonsense_qa": (per_row(r_csqa), ["tau__commonsense_qa/data/train-00000-of-00001.parquet"]),
    "drop": (drop_passages, ["ucinlp__drop/data/train-00000-of-00001.parquet"]),
}


def iter_rows(path, limit):
    pf = pq.ParquetFile(path)
    n = 0
    for batch in pf.iter_batches(batch_size=2000):
        for r in batch.to_pylist():
            yield r
            n += 1
            if limit and n >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "docs", "bench_train"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "manifests", "build", "bench_train.json"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=16384)
    a = ap.parse_args()

    # Each part looks at, and removes, only the files of its own part number. Parts can therefore
    # be rebuilt one at a time while the files of the other parts stay in place.
    prefix = f"part_{a.part:04d}_"
    os.makedirs(a.out, exist_ok=True)
    mine = [f for f in os.listdir(a.out) if f.startswith(prefix)]
    if mine and not a.overwrite:
        raise SystemExit(f"{a.out} already has {len(mine)} files of part {a.part}; pass --overwrite to rebuild them")
    for f in mine:
        os.remove(os.path.join(a.out, f))

    names = [s for s in a.sources.split(",") if s]
    files = [(s, f) for s in names for f in SOURCES[s][1]][a.part::a.nparts]
    stats = defaultdict(lambda: defaultdict(int))
    examples = {}
    seen = defaultdict(set)  # prompt hashes per source, shared by all files of that source
    docs = []  # (source, text); benchmarks are small, so the whole part fits in memory
    t0 = time.time()
    for src, rel in files:
        items = SOURCES[src][0]
        st = stats[src]
        for text, key in items(iter_rows(os.path.join(RAW, rel), a.limit), st):
            h = hashlib.blake2b(key.encode("utf-8"), digest_size=12).digest()
            if h in seen[src]:
                st["drop_duplicate"] += 1
                continue
            seen[src].add(h)
            docs.append((src, text))
            examples.setdefault(src, text)
        print(f"{time.strftime('%H:%M:%S')} {src} {rel}: {dict(st)}", flush=True)

    random.Random(a.seed + a.part).shuffle(docs)
    writer = documents.DocStoreWriter(a.out, a.part)
    for i in range(0, len(docs), 2000):
        chunk = docs[i:i + 2000]
        for (src, _), toks in zip(chunk, documents.encode_docs([t for _, t in chunk])):
            st = stats[src]
            if len(toks) - 1 > MAX_TOKENS:
                st["drop_too_long"] += 1
                continue
            writer.add(toks)
            st["docs"] += 1
            st["tokens"] += len(toks)
            st["max_doc_tokens"] = max(st["max_doc_tokens"], len(toks))
    totals = writer.close()

    first3 = [t[:300] for _, t in docs[:3]]
    manifest = {
        "builder": "build_bench_train.py",
        "stream": "bench_train",
        "store": a.out,
        "part": a.part,
        "nparts": a.nparts,
        "limit": a.limit,
        "examples_read": sum(s["read"] for s in stats.values()),
        "drops": {k: sum(s.get(k, 0) for s in stats.values())
                  for k in sorted({k for s in stats.values() for k in s if k.startswith("drop_")})},
        "docs_written": totals["docs"],
        "tokens_written": totals["tokens"],
        "per_source": {k: dict(v) for k, v in stats.items()},
        "first_300_chars_of_3_examples": first3,
        "first_example_per_source": {k: v[:600] for k, v in examples.items()},
        "minutes": round((time.time() - t0) / 60, 2),
    }
    os.makedirs(os.path.dirname(a.manifest), exist_ok=True)
    tmp = a.manifest + ".tmp"
    json.dump(manifest, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, a.manifest)
    print("DONE", json.dumps({k: manifest[k] for k in ("examples_read", "drops", "docs_written", "tokens_written")}), flush=True)


if __name__ == "__main__":
    main()
