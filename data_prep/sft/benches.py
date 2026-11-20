"""Train splits of the OrbitLM eval benchmarks, as SFT dialogs.

The question (and, for multiple choice, the options) is the user turn. The
answer is the assistant turn, so the loss covers the answer only. The wording
matches the evaluation prompts used by the long-context bench store: GSM8K
keeps its final "#### N" line, calculator notes are removed, HellaSwag markers
are cleaned the way lm-eval cleans them. Documents are not decontaminated.
"""
import os
import re

import pyarrow.parquet as pq

from render import is_degenerate_text, is_echo

LETTERS = "ABCDEFGHIJ"
_CALC = re.compile(r"<<[^<>]*>>")

MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]


class Drop(Exception):
    pass


def _norm(text):
    return " ".join((text or "").lower().split())


def mcq(question, options, answer_index):
    question = (question or "").strip()
    options = [(option or "").strip() for option in options]
    if not question or not options or any(not option for option in options):
        raise Drop("empty_field")
    if not (0 <= answer_index < len(options)) or len(options) > len(LETTERS):
        raise Drop("bad_label")
    lines = [f"Question: {question}"]
    lines += [f"{LETTERS[i]}. {option}" for i, option in enumerate(options)]
    user = "\n".join(lines)
    assistant = f"Answer: {LETTERS[answer_index]}"
    key = _norm(question) + "\x00" + "\x00".join(_norm(option) for option in options)
    return user, assistant, key


def labelled_mcq(question, choices, answer_key):
    labels = list(choices["label"])
    if answer_key not in labels:
        raise Drop("bad_label")
    return mcq(question, choices["text"], labels.index(answer_key))


def r_gsm8k(row):
    question = (row["question"] or "").strip()
    answer = (row["answer"] or "").strip()
    body, sep, final = answer.rpartition("####")
    if not question or not sep or not final.strip():
        raise Drop("no_final_answer")
    body = _CALC.sub("", body).strip()
    return f"Question: {question}", f"{body}\n#### {final.strip()}", _norm(question)


def r_math(row):
    question = (row["problem"] or "").strip()
    solution = (row["solution"] or "").strip()
    if not question or not solution:
        raise Drop("empty_field")
    return f"Question: {question}", solution, _norm(question)


def r_mmlu(row):
    return mcq(row["question"], row["choices"], int(row["answer"]))


def r_arc(row):
    return labelled_mcq(row["question"], row["choices"], row["answerKey"])


def r_obqa(row):
    return labelled_mcq(row["question_stem"], row["choices"], row["answerKey"])


def r_csqa(row):
    return labelled_mcq(row["question"], row["choices"], row["answerKey"])


def _hs_clean(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def r_hellaswag(row):
    ctx_a = (row["ctx_a"] or "").strip()
    ctx_a = "" if ctx_a in (".", "") else ctx_a
    ctx = _hs_clean(f"{row['activity_label']}: {ctx_a} {(row['ctx_b'] or '').capitalize()}")
    if not str(row["label"]).strip().isdigit():
        raise Drop("bad_label")
    return mcq(ctx, [_hs_clean(ending) for ending in row["endings"]], int(row["label"]))


def r_winogrande(row):
    if str(row["answer"]) not in ("1", "2"):
        raise Drop("bad_label")
    return mcq(row["sentence"], [row["option1"], row["option2"]], int(row["answer"]) - 1)


def r_piqa(row):
    return mcq(row["goal"], [row["sol1"], row["sol2"]], int(row["label"]))


def iter_rows(path, limit):
    parquet = pq.ParquetFile(path)
    seen = 0
    for batch in parquet.iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            yield row
            seen += 1
            if limit and seen >= limit:
                return


def drop_dialogs(rows, limit):
    """One dialog per DROP question. The passage stays in the user turn."""
    groups = {}
    for row in rows:
        passage = (row["passage"] or "").strip()
        question = (row["question"] or "").strip()
        spans = [span.strip() for span in (row["answers_spans"] or {}).get("spans") or [] if span and span.strip()]
        spans = list(dict.fromkeys(spans))
        if not passage or not question or not spans:
            continue
        group = groups.setdefault(_norm(passage), (passage, {}))
        group[1].setdefault(_norm(question), (question, ", ".join(spans)))
        if limit and sum(len(qa) for _, qa in groups.values()) >= limit:
            break
    for passage_key, (passage, questions) in groups.items():
        for question_key, (question, answer) in questions.items():
            user = f"Passage: {passage}\nQuestion: {question}"
            yield user, answer, f"{passage_key}\x00{question_key}"


SOURCES = {
    "gsm8k": (r_gsm8k, ["openai__gsm8k/main/train-00000-of-00001.parquet"]),
    "math": (r_math, [f"EleutherAI__hendrycks_math/{subject}/train-00000-of-00001.parquet" for subject in MATH_SUBJECTS]),
    "mmlu_aux": (r_mmlu, ["cais__mmlu/all/auxiliary_train-00000-of-00001.parquet"]),
    "arc": (r_arc, [
        "allenai__ai2_arc/ARC-Challenge/train-00000-of-00001.parquet",
        "allenai__ai2_arc/ARC-Easy/train-00000-of-00001.parquet",
    ]),
    "hellaswag": (r_hellaswag, ["Rowan__hellaswag/data/train-00000-of-00001.parquet"]),
    "winogrande": (r_winogrande, ["allenai__winogrande/winogrande_xl/train-00000-of-00001.parquet"]),
    "piqa": (r_piqa, ["ybisk__piqa/plain_text/train/0000.parquet"]),
    "openbookqa": (r_obqa, ["allenai__openbookqa/main/train-00000-of-00001.parquet"]),
    "commonsense_qa": (r_csqa, ["tau__commonsense_qa/data/train-00000-of-00001.parquet"]),
    "drop": (None, ["ucinlp__drop/data/train-00000-of-00001.parquet"]),
}


def _render_rows(path, render, limit):
    for row in iter_rows(path, limit):
        try:
            yield render(row)
        except Drop:
            continue


def iter_bench_dialogs(raw_root, sources=None, limit=0):
    """Yield prepared {role, content} dialogs, deduplicated inside each benchmark."""
    names = sources or list(SOURCES)
    for name in names:
        render, rels = SOURCES[name]
        seen = set()
        produced = 0
        for rel in rels:
            path = os.path.join(raw_root, rel)
            if name == "drop":
                pairs = drop_dialogs(iter_rows(path, limit), limit)
            else:
                pairs = _render_rows(path, render, limit)
            for user, assistant, key in pairs:
                if key in seen:
                    continue
                if is_degenerate_text(user) or is_degenerate_text(assistant) or is_echo(user, assistant):
                    continue
                seen.add(key)
                yield [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
                produced += 1
                if limit and produced >= limit:
                    break
            if limit and produced >= limit:
                break
