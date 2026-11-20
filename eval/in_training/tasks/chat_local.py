"""Prompts, datasets, scorers and detectors of the SFT (chat) evaluation.

The itr_chat_*.yaml task files name these functions with !function. The same functions serve the model
subclass of the chat evaluation (eval/in_training/itr_chat.py), which loads this file by its path, and
the CPU tests, so the scoring and the gates share one implementation. Every heavy import (datasets,
math_verify, lm-eval's IFEval checks) happens inside the function that needs it, so the module itself
loads in any Python.

The prompts follow the Qwen3 model cards: 0-shot, one user turn, and an instruction that fixes the
answer format. The response of a think-mode prompt holds the reasoning, then "</think>", then the
answer. Everything is scored on the answer alone, the text after the last "</think>", and a think block
that never closed has no answer and scores 0.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Optional

HERE = Path(__file__).resolve().parent

# The subclass of the chat evaluation puts the prompt mode of its process here. The scorers need it,
# because a think-mode response that never closed its think block and a no_think response look the
# same as text: both have no "</think>". One process always runs one mode.
MODE_ENV = "ORBIT_ITR_MODE"

# Token ids of the Qwen3 vocabulary that the prompt contract uses (README.md). They
# are the defaults of token_facts. The evaluation processes pass the ids of the tokenizer in use
# (vocabulary_ids), so that a model with another vocabulary is measured on its own ids. For every
# OrbitLM SFT run both give the same numbers.
END_OF_TEXT = 151643
IM_END = 151645
THINK_OPEN = 151667
THINK_CLOSE = 151668
NEWLINE = 198
QWEN3_IDS = {"end_of_text": END_OF_TEXT, "im_end": IM_END, "think_open": THINK_OPEN, "think_close": THINK_CLOSE,
             "newline": NEWLINE}
# Gate G3 asks that a model without the prefill opens "<think>\n" and then writes at least this many
# tokens before anything that would end its reasoning.
NP_WINDOW = 32

THINK_OPEN_TEXT = "<think>"
THINK_CLOSE_TEXT = "</think>"
END_MARKERS = ("<|im_end|>", "<|endoftext|>")

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
MATH_INSTRUCTION_RU = "Рассуждай пошагово и запиши окончательный ответ в \\boxed{}."
MCQ_INSTRUCTION = 'Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.'
CHOICE_LETTERS = "ABCD"

# Datasets that the task files do not name themselves, pinned to the revisions that the document
# counts in src/orbit/in_training_eval/benchmarks.py were taken from.
MMLU_REDUX_PATH = "fxmarty/mmlu-redux-2.0-ok"
MMLU_REDUX_REVISION = "8dd06e599d20d57e83b590d497c95869a2d18781"
# Thirty questions per subject give 1,710 questions, a standard error of about 1.2 points, at a third
# of the cost of the full 5,330. The smallest subject (virology) has 43.
MMLU_REDUX_PER_SUBJECT = 30
MMLU_REDUX_SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering",
    "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy",
    "prehistory", "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
)
# The chat probe: root prompts of the OpenAssistant OASST2 validation split (Apache-2.0), which no
# benchmark of the suite covers, plus trivial prompts written for this suite.
OASST2 = ("OpenAssistant/oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75")
PROBE_EN = 200
PROBE_TRIVIAL_FILE = HERE / "chat_probe_trivial.jsonl"


# ---------------------------------------------------------------------------
# The answer part
# ---------------------------------------------------------------------------


def current_mode() -> str:
    return os.environ.get(MODE_ENV, "think")


def strip_end_markers(text: str) -> str:
    """Removes a trailing end-of-turn or end-of-text marker that the decoder may have left in the text."""
    text = text or ""
    changed = True
    while changed:
        changed = False
        stripped = text.rstrip()
        for marker in END_MARKERS:
            if stripped.endswith(marker):
                text = stripped[: -len(marker)]
                changed = True
    return text


def split_answer(text: str, mode: str) -> Optional[str]:
    """The answer part of a response: the text after the last "</think>".

    In think and no-prefill mode a response without "</think>" has no answer (None): its think block
    never closed, whether it ran into the budget or stopped inside. In no_think mode a response without
    the tag is all answer.
    """
    text = strip_end_markers(text)
    if THINK_CLOSE_TEXT in text:
        return text.rsplit(THINK_CLOSE_TEXT, 1)[1]
    return text if mode == "no_think" else None


# ---------------------------------------------------------------------------
# Prompts and documents
# ---------------------------------------------------------------------------


def _with_cell(doc: dict, lang: str, domain: str, gold: Any) -> dict:
    return {**doc, "itr_lang": lang, "itr_domain": domain, "itr_gold": "" if gold is None else str(gold)}


def process_math500(dataset):
    # The gold of MATH-500 is its worked solution, whose last \boxed{} math_verify reads, as in the
    # base suite's minerva tasks.
    return dataset.map(lambda doc: _with_cell(doc, "en", "math", doc["solution"]))


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def process_gsm8k(dataset):
    return dataset.map(lambda doc: _with_cell(doc, "en", "math", gsm8k_gold(doc["answer"])))


def process_mgsm_ru(dataset):
    def gold(doc):
        number = doc["answer_number"]
        if isinstance(number, float) and number.is_integer():
            number = int(number)
        return number

    return dataset.map(lambda doc: _with_cell(doc, "ru", "math", gold(doc)))


def process_ifeval(dataset):
    return dataset.map(lambda doc: _with_cell(doc, "en", "if", None))


def math_prompt(doc: dict) -> str:
    return f"{doc['problem']}\n{MATH_INSTRUCTION}"


def gsm8k_prompt(doc: dict) -> str:
    return f"{doc['question']}\n{MATH_INSTRUCTION}"


def mgsm_ru_prompt(doc: dict) -> str:
    return f"{doc['question']}\n{MATH_INSTRUCTION_RU}"


def mmlu_redux_prompt(doc: dict) -> str:
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(CHOICE_LETTERS, doc["choices"]))
    return f"{doc['question']}\n{options}\n{MCQ_INSTRUCTION}"


def plain_prompt(doc: dict) -> str:
    return doc["prompt"]


def _dataset_dict(rows: list[dict]):
    import datasets

    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})


def load_mmlu_redux(**_metadata):
    """MMLU-Redux 2.0 ("ok" questions), thirty per subject, as one task.

    The thirty are the questions with the smallest sha256 of their text, which picks the same questions
    whatever order the dataset stores them in.
    """
    import datasets

    rows = []
    for subject in MMLU_REDUX_SUBJECTS:
        split = datasets.load_dataset(MMLU_REDUX_PATH, subject, split="test", revision=MMLU_REDUX_REVISION)
        order = sorted(range(len(split)), key=lambda i: hashlib.sha256(split[i]["question"].encode("utf-8")).hexdigest())
        for index in sorted(order[:MMLU_REDUX_PER_SUBJECT]):
            row = split[index]
            rows.append(_with_cell({
                "subject": subject,
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),
            }, "en", "mcq", CHOICE_LETTERS[int(row["answer"])]))
    return _dataset_dict(rows)


def read_trivial_prompts(path: Path = PROBE_TRIVIAL_FILE) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            rows.append({"prompt": entry["prompt"], "itr_lang": entry["lang"], "itr_domain": "trivial", "source": "trivial"})
    return rows


def select_probe_prompts(messages: list[dict]) -> list[dict]:
    """The OASST2 part of the chat probe: the first 200 English and every Russian root prompt.

    Root prompts are the messages without a parent. They are ordered by message id, which is stable
    across downloads of the pinned revision.
    """
    roots = sorted(
        (m for m in messages if m.get("parent_id") is None and m.get("role") == "prompter" and m.get("lang") in ("en", "ru")),
        key=lambda m: m["message_id"],
    )
    english = [m for m in roots if m["lang"] == "en"][:PROBE_EN]
    russian = [m for m in roots if m["lang"] == "ru"]
    return [
        {"prompt": m["text"], "itr_lang": m["lang"], "itr_domain": "chat", "source": f"oasst2:{m['message_id']}"}
        for m in english + russian
    ]


def load_chat_probe(**_metadata):
    import datasets

    validation = datasets.load_dataset(OASST2[0], split="validation", revision=OASST2[1])
    columns = ("message_id", "parent_id", "role", "lang", "text")
    messages = [{key: row[key] for key in columns} for row in validation]
    rows = select_probe_prompts(messages) + read_trivial_prompts()
    return _dataset_dict([{**row, "itr_gold": ""} for row in rows])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def math_correct(answer: Optional[str], gold: str) -> int:
    """1 when math_verify finds the gold in the answer part, 0 otherwise or without an answer."""
    if answer is None or not answer.strip():
        return 0
    try:
        from math_verify import parse, verify

        return int(bool(verify(gold=parse(gold if "\\boxed" in gold else f"${gold}$"), target=parse(answer))))
    except Exception:
        # A parser crash on an odd answer counts as a wrong answer, never as a failed process.
        return 0


# Qwen's instruction asks for "answer": "X", but Qwen3-1.7B writes that exact JSON form in only a few
# percent of its answers. Most answers say answer: X, **Answer**: `X`, "the answer is (X)" or
# \boxed{\text{X}}, so all of these are read; json_format reports the share in the requested form.
# The keys and phrases match in any case, the letter only as a capital that no letter or digit
# follows, so "the answer is a bit unclear" and "Answer: Avogadro" hold no choice.
_JSON_CHOICE = re.compile(r'(?i:"answer")\s*:\s*"\s*\(?([A-D])\)?\s*"')
_KEYED_CHOICE = re.compile(r"(?i:\b(?:answer|choice))\W{0,6}[:：]\W{0,6}\(?([A-D])(?![A-Za-z0-9])")
_ANSWER_IS = re.compile(r"(?i:\b(?:answer|choice|option) is)\W{0,6}\(?([A-D])(?![A-Za-z0-9])")
_BOXED_CHOICE = re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?\(?([A-D])(?![A-Za-z0-9])")
_FINAL_LINE = re.compile(r"^\W{0,4}\(?([A-D])(?:[.)]|\W{0,4}$)")


def extract_choice(answer: Optional[str]) -> tuple[Optional[str], bool]:
    """The chosen letter, and whether the answer used Qwen's JSON form "answer": "X".

    The letter is the last one that any of the forms above names, because a model often discusses
    other options before it commits. Only when no form matches does a final line that starts with a
    letter, such as "**B. True, False**", count. A looser fallback, such as the last capital A-D
    anywhere, would read the article "A" at the start of a sentence.
    """
    if not answer:
        return None, False
    strict = bool(_JSON_CHOICE.search(answer))
    last: Optional[tuple[int, str]] = None
    for pattern in (_JSON_CHOICE, _KEYED_CHOICE, _ANSWER_IS, _BOXED_CHOICE):
        for match in pattern.finditer(answer):
            if last is None or match.start(1) > last[0]:
                last = (match.start(1), match.group(1))
    if last is not None:
        return last[1], strict
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    if lines:
        match = _FINAL_LINE.match(lines[-1])
        if match:
            return match.group(1), strict
    return None, strict


def _require_langdetect() -> None:
    """Imports langdetect for lm-eval's IFEval checks and makes its answers repeatable.

    lm-eval's IFEval module imports langdetect at load time, and a missing package would fail there
    with a bare ModuleNotFoundError, so this says what is missing and where it belongs.

    langdetect draws random numbers while it detects, so the same short or mixed text can get a
    different language on another run. A fixed seed gives one answer per text, which keeps the IFEval
    score of a checkpoint the same when it is evaluated again. lm-eval itself sets no seed.
    """
    try:
        import langdetect
    except ImportError as exc:
        raise ImportError(
            "IFEval needs langdetect for its language checks (95 of the 541 prompts); install "
            "requirements/eval.txt into the evaluation environment, or eval/in_training/requirements-sft-eval.txt "
            "into the folder of eval_extra_pythonpath"
        ) from exc
    langdetect.DetectorFactory.seed = 0


def ifeval_scores(doc: dict, answer: Optional[str]) -> dict:
    """lm-eval's IFEval checks of one answer, with Python's random numbers seeded for the call.

    Some IFEval checks draw a parameter at random when the dataset's value is not usable. Prompt 19
    asks for at least four "#", and the letter check replaces "#" by a random letter. Without a seed
    the same answer could pass in lm-eval's scoring and fail in the records, or pass at one
    checkpoint and fail at the next. The caller's random state is restored afterwards.
    """
    _require_langdetect()
    from lm_eval.tasks.ifeval import utils as ifeval_utils

    state = random.getstate()
    random.seed(0)
    try:
        return ifeval_utils.process_results(doc, [answer or ""])
    finally:
        random.setstate(state)


TASK_KINDS = {
    "itr_chat_math500": "math",
    "itr_chat_gsm8k": "math",
    "itr_chat_mgsm_ru": "math",
    "itr_chat_ifeval": "ifeval",
    "itr_chat_mmlu_redux": "mcq",
    "itr_chat_probe": "probe",
}


def score_response(task: str, doc: dict, text: str, mode: str) -> Optional[float]:
    """The correctness of one response (1.0 or 0.0), or None for a task without a score."""
    kind = TASK_KINDS.get(task)
    answer = split_answer(text, mode)
    if kind == "math":
        return float(math_correct(answer, doc["itr_gold"]))
    if kind == "mcq":
        letter, _ = extract_choice(answer)
        return float(letter == doc["itr_gold"])
    if kind == "ifeval":
        return float(bool(ifeval_scores(doc, answer)["prompt_level_strict_acc"]))
    return None


def process_math(doc: dict, results: list) -> dict:
    return {"acc": math_correct(split_answer(results[0], current_mode()), doc["itr_gold"])}


def process_mmlu_redux(doc: dict, results: list) -> dict:
    letter, strict = extract_choice(split_answer(results[0], current_mode()))
    return {"acc": int(letter == doc["itr_gold"]), "json_format": int(strict)}


def process_ifeval_results(doc: dict, results: list) -> dict:
    return ifeval_scores(doc, split_answer(results[0], current_mode()))


def agg_inst_level_acc(items):
    flat = [item for sublist in items for item in sublist]
    return sum(flat) / len(flat) if flat else 0.0


def process_probe(doc: dict, results: list) -> dict:
    answer = split_answer(results[0], current_mode())
    return {"answered": int(bool(answer and answer.strip()))}


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def rep_ngram(ids: list[int], n: int = 16) -> float:
    """1 - distinct n-grams / all n-grams: 0 for text without repeats, close to 1 for a loop."""
    if len(ids) < n + 1:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / float(len(grams))


def trimmed_length(ids: list[int], is_blank: Callable[[int], bool]) -> int:
    start, end = 0, len(ids)
    while start < end and is_blank(ids[start]):
        start += 1
    while end > start and is_blank(ids[end - 1]):
        end -= 1
    return end - start


def _last_index(ids: list[int], token: int) -> Optional[int]:
    for position in range(len(ids) - 1, -1, -1):
        if ids[position] == token:
            return position
    return None


def vocabulary_ids(tokenizer: Any) -> dict[str, int]:
    """The ids of the texts that token_facts looks for, in the vocabulary of this tokenizer.

    Each text must be a single token. A vocabulary that splits "</think>" into pieces could not show
    where a think block closes, so such a tokenizer is refused here instead of being measured wrongly.
    """
    texts = {"end_of_text": "<|endoftext|>", "im_end": "<|im_end|>", "think_open": THINK_OPEN_TEXT,
             "think_close": THINK_CLOSE_TEXT, "newline": "\n"}
    found = {}
    for name, text in texts.items():
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not a single token in this vocabulary (ids {ids})")
        found[name] = int(ids[0])
    return found


def token_facts(gen_ids: list[int], mode: str, finish: Optional[str], is_blank: Callable[[int], bool],
                vocabulary: Optional[dict[str, int]] = None) -> dict:
    """What the generated ids of one response say, before any threshold is applied.

    The thresholds (the short-think length, the loop cut-offs) are applied by the trainer, from these
    numbers, so they can be recalibrated without touching the evaluation processes. "Reasoning" is the
    ids before the first "</think>"; the answer is the ids after the last one, without the final stop
    token. ``vocabulary`` holds the ids of the tokenizer in use (vocabulary_ids); without it the ids
    are the Qwen3 vocabulary's.
    """
    v = vocabulary or QWEN3_IDS
    end_of_text, im_end, think_open, think_close = v["end_of_text"], v["im_end"], v["think_open"], v["think_close"]
    ids = [int(token) for token in gen_ids]
    body = ids[:-1] if ids and ids[-1] in (im_end, end_of_text) else ids
    facts: dict[str, Any] = {
        "n_gen": len(ids),
        "finish": finish,
        "last_id": ids[-1] if ids else None,
        "n_open": ids.count(think_open),
        "n_close": ids.count(think_close),
        "rep16_tail": round(rep_ngram(ids[-1024:]), 4),
        "rep16_w4k": round(rep_ngram(ids[-4096:]), 4),
    }
    start = 0
    if mode == "no_prefill":
        opened = ids[:2] == [think_open, v["newline"]]
        window = ids[2:2 + NP_WINDOW]
        facts["np_opened"] = opened
        facts["np_ok"] = bool(
            opened and len(window) == NP_WINDOW and not any(t in (think_close, im_end, end_of_text) for t in window)
        )
        start = 2 if opened else 0
    last_close = _last_index(body, think_close)
    answer = body[last_close + 1:] if last_close is not None else ([] if mode != "no_think" else body)
    facts["answer_len"] = len(answer)
    facts["answer_nonblank"] = any(not is_blank(token) for token in answer)
    if mode == "no_think":
        facts["closed"] = None
        facts["reasoning_len"] = None
        facts["format_ok"] = bool(
            facts["n_open"] == 0 and facts["n_close"] == 0 and finish == "stop" and facts["last_id"] == im_end
            and facts["answer_nonblank"]
        )
        facts["stop_inside"] = False
        return facts
    first_close = ids.index(think_close, start) if think_close in ids[start:] else None
    reasoning = ids[start:first_close] if first_close is not None else body[start:]
    facts["closed"] = first_close is not None
    facts["reasoning_len"] = trimmed_length(reasoning, is_blank)
    # In no-prefill mode the model's own opening tag is the one "<think>" a well-formed response has.
    allowed_open = 1 if mode == "no_prefill" and facts.get("np_opened") else 0
    facts["format_ok"] = bool(
        facts["n_close"] == 1 and facts["n_open"] == allowed_open and finish == "stop"
        and facts["last_id"] == im_end and facts["answer_nonblank"]
    )
    facts["stop_inside"] = bool(first_close is None and finish == "stop")
    return facts


_MARKERS = [re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bwait\b", r"\bhmm+\b", r"\blet me double[- ]check\b", r"\blet me re-?check\b", r"\blet me verify\b",
    r"\bon second thought\b",
    r"\bподожди\b", r"\bпогоди\b", r"\bхм+\b", r"\bстоп\b", r"\bперепроверю\b", r"\bдавай(?:те)? проверим\b",
)]
_OPENER = re.compile(r"^(?:Okay|Alright|Хорошо),")
_NOT_PROSE = [re.compile(pattern, re.DOTALL) for pattern in (
    r"```.*?(?:```|$)",
    r"`[^`]*`",
    r"\$\$.*?\$\$",
    r"\$[^$]*\$",
    r"\\\(.*?\\\)",
    r"\\\[.*?\\\]",
    r"\\boxed\{(?:[^{}]|\{[^{}]*\})*\}",
    r"https?://\S+",
)]
_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
# The answer language is determined only from at least this many letters; shorter answers (a number,
# a formula) count as "undetermined" and stay out of gate G11.
LANG_MIN_LETTERS = 20


def count_markers(text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in _MARKERS)


def letter_counts(text: str) -> tuple[int, int]:
    """(Latin letters, Cyrillic letters) of the prose, with formulas, code and links removed."""
    for pattern in _NOT_PROSE:
        text = pattern.sub(" ", text)
    return len(_LATIN.findall(text)), len(_CYRILLIC.findall(text))


def answer_language(text: Optional[str]) -> Optional[str]:
    latin, cyrillic = letter_counts(text or "")
    if latin + cyrillic < LANG_MIN_LETTERS:
        return None
    return "ru" if cyrillic / float(latin + cyrillic) >= 0.5 else "en"


def text_facts(text: str, mode: str) -> dict:
    """What the text of one response says: the answer's markers, opener and language."""
    answer = split_answer(text, mode)
    prose = answer or ""
    latin, cyrillic = letter_counts(prose)
    return {
        "has_answer": answer is not None,
        "markers": count_markers(prose),
        "opener": bool(_OPENER.match(prose.lstrip())),
        "ans_latin": latin,
        "ans_cyr": cyrillic,
        "lang": answer_language(prose) if answer is not None else None,
    }
