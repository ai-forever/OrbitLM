#!/usr/bin/env python3
"""Watch a distillation or RL run on verl and check its stop rules.

    $PY scripts/verl/watch.py <run_dir>                          # distillation: one report
    $PY scripts/verl/watch.py <run_dir> --follow 600             # check again every 10 minutes
    $PY scripts/verl/watch.py <run_dir> --json                   # the report as one JSON object
    $PY scripts/verl/watch.py <run_dir> --rl --val-dir <dir>     # RL: rules W1-W8
    $PY scripts/verl/watch.py <run_dir> --rl --val-dir <dir> --follow 600

``<run_dir>`` is ``$ORBIT_VERL_RUNS/runs/<run_name>``. The script only reads files there and uses
only the standard library, so it runs under any python, also while the run trains. A rule that fires
means: look at the run now. Stopping a run and choosing its checkpoint are separate decisions, and a
fire never says which checkpoint is best.

What it reads:

- ``metrics.jsonl``, plus the ``metrics.<date-time>.jsonl`` files that earlier launches of a resumed
  run left behind. Per training step it takes the teacher KL (``actor/distillation/loss``),
  ``actor/grad_norm``, the mean response length, and the share of responses cut by the length limit.
  verl logs that share as ``response_length/clip_ratio``, the share of responses as long as the
  longest one of the batch. It equals the cut share when the longest response reached the limit
  (``data.max_response_length`` of the run's ``config.yaml``) and is 0 otherwise. At validation
  steps it takes the reward's exact per-source flags ``acc``, ``truncated``, ``think_unclosed`` and
  ``looped``.
- ``rollouts/<step>.jsonl``, verl's dump of the training rollouts. It holds only decoded text, so
  the mode is read from the end of the prompt ("<think>" is think, "</think>" is no_think), lengths
  are in characters, and the loop flag is the SFT evaluation's detector (``orbit_verl.data.loops``)
  applied to words and punctuation marks. Training loop shares are therefore estimates. A think
  answer that holds the stop-thinking sentence of the thinking budget counts as forced; its
  thinking and its answer are then measured on either side of the inserted text.
- ``val_generations/<step>.jsonl``, the same dump for validation. Its rows carry the reward's flags,
  so they are exact. Validation answers are never forced, so a forced one is reported in a note.

Distillation rules (default mode). A training window is ``--window`` consecutive steps (5 by
default), and only complete windows count. The first value of a series is the step-0 validation, or
the first training window.

1. Truncated share above 2x its first value, and at least 0.02 above it.
2. Looped share rising at 3 checks in a row, by at least 0.01 in total.
3. ``actor/grad_norm`` above 5x the median of the first 20 steps for 3 steps in a row. Until 20
   steps exist, the median of the steps so far stands in, once there are at least 5. Steps 1-5
   never count as high, because a run starts with high norms that decay.
4. Mean length of the think answers above 1.3x its first value, in training or in validation.
5. Teacher KL, as a mean over the last 5 steps (``--kl-smooth``), rising 20 steps in a row.
6. Russian must not break: on training answers to Russian math or chat prompts, the Cyrillic share
   of the thinking or of the answer falls below 0.85x its first value, and by at least 0.05. The
   share is counted on prose only (formulas, code and LaTeX commands are cut out first). Code
   prompts are left out, because their thinking is about English identifiers.

Exit codes: 0 no rule fired, 2 the run dir does not exist, 3 a rule fired.

RL rules (``--rl``). The script prints one status line per training step and per validation, and
one FIRE line per rule. A fire has one of three levels: "notify" (look at the run), "confirmed" (a
notify that held on two checks in a row) and "hard-stop" (the run should stop now). Every rule reads
step 0 as its reference unless it says otherwise. Validation rows are matched to their set by the
exact question, against the frozen validation files (``--val-dir`` or ``--val-file ROLE=PATH``),
because verl's dump does not record the data source.

- W1 val drop: the mean of a primary source's accuracy over each validation and the one before it
  (MATH-500 think by default; ``--w1`` adds sources). Notify when it is at least 0.03 below its best
  on two validations in a row, or at least 0.05 below once. Hard stop at 0.08 below on two in a row.
  LCB-78 has no drop rule, because its noise (SD about 0.04) is as large as the drops.
- W2 grad: ``actor/grad_norm`` above 2x the median of steps 1-20 for 3 steps in a row, steps 1-5
  not counted. Notify.
- W3 KL: the 5-step mean of ``actor/kl_loss`` above 0.045 notifies; above ``--kl-hard`` (0.06) it
  is a hard stop.
- W4 format: (a) the looping share of training answers, as a 3-step mean, above 0.05; (b) the
  training truncated share, as a 5-step mean, at or above max(2x, +0.05) of the lowest earlier
  5-step mean, 3 steps in a row; (c) the mean response length below 0.5x its steps 1-5 mean; (d) a
  validation source's think_unclosed at or above max(0.5, step 0 + 0.15), with only the relative
  part for code sources. (a)-(c) notify. (d) notifies on the first validation and is a hard stop on
  the second in a row.
- W5 exit drift: the 5-step mean of ``actor/rltt/expected_exit_step`` more than 0.03 below its
  highest earlier value. Notify.
- W6 Russian: (a) MGSM-ru think Cyrillic below 0.95, or more than 2% of its rows below 0.5;
  (b) the Russian chat probe's think Cyrillic at or below step 0 - 0.05; (c) its answer Cyrillic at
  or below step 0 - 0.03, or its share of answers below 0.5 Cyrillic up by 0.03; (d) the paired gap
  acc(MGSM-ru) - acc(GSM8K-en, the same 250 problems in English) at or below its step-0 value - 0.06;
  (e) the share of English training answers with Cyrillic words above 3x the first window and 0.01
  above it; (f) distillation rule 6 on Russian training prompts, by window. Each notifies on its
  first failing check and is confirmed on the second in a row. Two confirmed W6 rules make a hard
  stop.
- W7 chat probe, Russian and English separately: the looped or think_unclosed share at or above
  step 0 + 0.05 (notify, confirmed on the second check), or the mean answer length outside 0.5-1.5x
  of step 0 (notify).
- W8 loop gap: the largest gap between an intermediate loop's mean log-prob of the sampled tokens
  and the final loop's (``actor/rltt/loop_logp_t*``) above 1.5x its median over steps 1-20 for 3
  steps in a row, steps 1-5 not counted. The gap can grow a few steps before grad_norm does, so it
  is an early sign of the same trouble. Notify.

Only these are hard stops: W3 above ``--kl-hard``, W4d on two validations in a row, W1's large drop
on two validations in a row, and two confirmed W6 rules. The watcher never stops a run itself. Its exit
code is the hook for automation: with ``--follow`` it keeps checking until a hard stop fires and then
exits with 4, so a wrapper can stop the run and pick its checkpoint.

RL exit codes: 0 nothing fired, 2 a usage error, 3 only notify fires, 4 a hard stop fired.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# scripts/verl/watch.py -> parents[2] is the repo root, whatever the current directory is.
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Both modules use only the standard library, so the script stays runnable without verl.
from orbit_verl.data.loops import LOOP_MIN_TOKENS, LOOP_REP16, LOOP_TAIL_TOKENS, tail_rep16  # noqa: E402
from orbit_verl.data.response_info import STOP_THINKING_TEXT, STOP_THINKING_TEXTS, THINK_CLOSE, THINK_OPEN  # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_STOP = 3

THINK = "think"
NO_THINK = "no_think"
UNKNOWN = "unknown"
ALL = "all"
MODES = (THINK, NO_THINK)

# The keys of verl's metrics.jsonl that the rules read. The teacher KL is the distillation loss
# that verl logs with use_policy_gradient: the clamped k1 averaged over the tokens of the batch.
KL_KEYS = ("actor/distillation/loss", "distillation/loss")
GRAD_KEY = "actor/grad_norm"
LENGTH_KEY = "response_length/mean"
LENGTH_MAX_KEY = "response_length/max"
CLIP_KEY = "response_length/clip_ratio"
SCORE_KEY = "critic/score/mean"

# verl's validation metrics: val-core or val-aux, then the data source (which may hold "/"), the
# reward key and the statistic. Only the means are read.
_VAL_KEY = re.compile(r"^val-(?:core|aux)/(?P<source>.+)/(?P<var>acc|truncated|think_unclosed|looped)/mean@(?P<n>\d+)$")
# The units of the text-based loop estimate: runs of word characters, and every other character
# that is not a space. They stand in for token ids and come out at about the same count.
_UNIT = re.compile(r"\w+|[^\w\s]")
# What is cut out of a text before its Cyrillic share is counted: code blocks, display and inline
# formulas, LaTeX environments and commands, and inline code. Their Latin letters say nothing about
# the language of the prose around them.
_FORMULAS = tuple(re.compile(pattern, re.S) for pattern in (
    r"```.*?```", r"\$\$.*?\$\$", r"\\\[.*?\\\]", r"\\\(.*?\\\)", r"\$[^$]*\$",
    r"\\begin\{.*?\}.*?\\end\{.*?\}", r"\\[A-Za-z]+", r"`[^`]*`"))
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")
# The English instruction that the distillation prompts append to every math problem. A Russian math
# prompt is recognised by the question before it.
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within"
# Russian math problems get the Russian instruction of sft_think_ru.yaml instead ("Reason step by step
# and write the final answer in \boxed{}"), and code problems get the instruction of sft_think_code.yaml.
# Both are copied from those templates, so that the kind of a decoded prompt can be read from its text.
MATH_INSTRUCTION_RU = "Рассуждай пошагово и запиши окончательный ответ в"
CODE_INSTRUCTION = "Write Python code to solve the problem."

# The kinds of prompt that rule 6 tells apart. A prompt with none of the instructions is a chat
# prompt: chat, instruction following and every other pool row that is not math or code.
MATH = "math"
CODE = "code"
CHAT = "chat"
KIND_INSTRUCTIONS = ((MATH, MATH_INSTRUCTION), (MATH, MATH_INSTRUCTION_RU), (CODE, CODE_INSTRUCTION))
# The dump groups keep these kinds next to the modes, under their own keys, so a kind is never
# mistaken for a mode. Rule 6 reads the math and chat groups.
KIND_KEYS = {kind: f"kind:{kind}" for kind in (MATH, CODE, CHAT)}
RU_GUARD_KINDS = (MATH, CHAT)
# What remains of the SFT chat template in a decoded prompt: "user\n" before the question, and the
# assistant turn after it. The think markers follow the assistant turn.
_USER_PREFIX = "user\n"
_ASSISTANT_TURN = "\nassistant\n"
# A Russian word of at least two letters inside an answer to an English prompt. An RL run that
# degrades can mix such words into its English reasoning, which is what W6e watches.
_CYRILLIC_WORD = re.compile(r"[А-Яа-яЁё]{2,}")


def prompt_kind(prompt: str) -> str:
    """Whether a decoded prompt is a math, code or chat prompt, from the instruction it carries."""
    return next((kind for kind, instruction in KIND_INSTRUCTIONS if instruction in prompt), CHAT)


def prose(text: str) -> str:
    """``text`` without its formulas, code and LaTeX commands (``_FORMULAS``)."""
    for pattern in _FORMULAS:
        text = pattern.sub(" ", text)
    return text


def cyrillic_share(text: str, minimum: int = 20) -> float | None:
    """The share of Cyrillic among the Cyrillic and Latin letters of the prose of ``text``.

    None when the prose has fewer than ``minimum`` such letters, because a share of a handful of
    letters says nothing.
    """
    letters = prose(text)
    cyrillic, latin = len(_CYRILLIC.findall(letters)), len(_LATIN.findall(letters))
    return cyrillic / (cyrillic + latin) if cyrillic + latin >= minimum else None


def question_text(prompt: str) -> str:
    """The question of a decoded prompt, without the decoded chat template and without the instruction.

    verl decodes a prompt with the special tokens dropped, so the SFT template leaves the words
    "user" and "assistant" and the think markers around the question:
    "user\\n{question}\\n{instruction}\\nassistant\\n<think>\\n". The question ends where any known
    instruction starts, and the template's words are cut off as well. Both carry Latin letters that
    would outweigh the Cyrillic of a short Russian question such as "Привет!" ("Hi!").
    """
    question = prompt[len(_USER_PREFIX):] if prompt.startswith(_USER_PREFIX) else prompt
    end = question.rfind(_ASSISTANT_TURN)
    if end >= 0:
        question = question[:end]
    for _, instruction in KIND_INSTRUCTIONS:
        question = question.split(instruction)[0]
    return question.strip()


def russian_prompt(prompt: str) -> bool:
    """Whether a decoded prompt asks in Russian: the prose of its question is mostly Cyrillic.

    One letter is enough once the template is cut off, because a short greeting is still a Russian
    prompt. On the 109 prompts of the Russian chat probe this finds all of them, and it finds none
    of the English ones.
    """
    share = cyrillic_share(question_text(prompt), minimum=1)
    return share is not None and share >= 0.5


# The sentence of the stop-thinking text that the agent loop inserts at the thinking budget. The
# training dump holds only decoded text, so a forced answer is recognised by this sentence.
STOP_SENTENCE = STOP_THINKING_TEXT.split(THINK_CLOSE)[0].strip()
# The sentences of every language's stop-thinking text (English, and Russian for Russian rows), each
# with the inserted text around it: a blank line before it, and the line break, "</think>" and a
# blank line after it.
_STOP_PARTS = tuple(
    (sentence, *text.partition(sentence)[::2])
    for text in STOP_THINKING_TEXTS.values()
    for sentence in [text.split(THINK_CLOSE)[0].strip()]
)


@dataclass(frozen=True)
class Thresholds:
    """The numbers of the stop rules; the module docstring explains each rule."""

    truncated_factor: float = 2.0
    min_share_increase: float = 0.02
    loop_rises: int = 3
    min_loop_rise: float = 0.01
    grad_factor: float = 5.0
    grad_baseline_steps: int = 20
    grad_run: int = 3
    min_grad_baseline: int = 5
    grad_warmup: int = 5
    length_factor: float = 1.3
    ru_factor: float = 0.85
    min_ru_drop: float = 0.05
    kl_rises: int = 20
    kl_smooth: int = 5
    window: int = 5


# --------------------------------------------------------------------------- reading files


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """The JSON objects of a jsonl file, one per line.

    A line that does not parse is skipped. verl writes these files while the run trains, and the
    last line can be half written when the script reads it.
    """
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def _number(value: Any) -> float | None:
    """``value`` as a float when it is a finite number, else None. JSON true and false are not numbers here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def metrics_files(run_dir: Path) -> list[Path]:
    """``metrics.jsonl`` and the files of earlier launches, oldest first.

    The launcher renames the file of an earlier launch to ``metrics.<YYYYmmdd-HHMMSS>.jsonl``
    before verl truncates it, so the names sort in time order.
    """
    current = run_dir / "metrics.jsonl"
    return sorted(run_dir.glob("metrics.*.jsonl")) + ([current] if current.is_file() else [])


def read_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    """The numeric metrics of every step, merged over all metrics files.

    A step can appear in two files after a resume, once with its training metrics and once with
    the validation that runs right after the restart. Their keys are merged, and a later file wins
    for a key that both have.
    """
    steps: dict[int, dict[str, float]] = {}
    for path in metrics_files(run_dir):
        for record in read_jsonl(path):
            step, data = record.get("step"), record.get("data")
            if isinstance(step, bool) or not isinstance(step, int) or not isinstance(data, dict):
                continue
            values = {key: number for key, raw in data.items() if (number := _number(raw)) is not None}
            steps.setdefault(step, {}).update(values)
    return steps


def yaml_scalar(text: str, keys: Sequence[str]) -> str | None:
    """The scalar at ``keys`` in block-style YAML as omegaconf writes it, or None when it is not there.

    Only direct children are matched: the first line inside a matched key sets the indentation of
    its children, and deeper lines are skipped. The run's config.yaml is the only input, and this
    small reader keeps the script free of a YAML dependency. A key that omegaconf writes in quotes
    is matched without them: it quotes ``n``, which YAML 1.1 would read as false.
    """
    level, parent_indent, child_indent = 0, -1, 0
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent <= parent_indent:
            return None
        if child_indent is None:
            child_indent = indent
        if indent != child_indent:
            continue
        key, colon, rest = stripped.partition(":")
        if not colon or key.strip().strip("'\"") != keys[level]:
            continue
        if level == len(keys) - 1:
            return rest.strip().strip("'\"") or None
        level, parent_indent, child_indent = level + 1, indent, None
    return None


def response_limit(run_dir: Path, given: int | None) -> tuple[int | None, str]:
    """The response length limit of the run in tokens, and where it came from."""
    if given is not None:
        return given, "--max-response-length"
    try:
        text = (run_dir / "config.yaml").read_text(encoding="utf-8")
    except OSError:
        return None, "unknown: no config.yaml in the run dir; pass --max-response-length"
    value = yaml_scalar(text, ("data", "max_response_length"))
    try:
        return int(str(value)), "config.yaml"
    except ValueError:
        return None, "unknown: config.yaml has no data.max_response_length; pass --max-response-length"


# --------------------------------------------------------------------------- one dumped answer


@dataclass(frozen=True)
class RowFacts:
    """What the script learns from one dumped answer.

    ``think_chars`` is the length of the thinking block of a think answer, None for other modes.
    For a forced answer it is the length of the model's own thinking, before the inserted text.
    ``forced`` is True when the agent loop closed the thinking block at the thinking budget.
    ``truncated`` is None when the row does not say it, as in verl's training dump.
    ``looped_exact`` is False when the loop flag is the text-based estimate.
    """

    mode: str
    chars: int
    think_chars: int | None
    unclosed: bool
    forced: bool
    looped: bool
    looped_exact: bool
    truncated: bool | None
    score: float | None
    russian: bool = False
    ru_think: float | None = None
    ru_answer: float | None = None
    kind: str = CHAT
    fragment: bool = False


def answer_mode(row: Mapping[str, Any]) -> str:
    """The thinking mode of a dumped answer: from a ``mode`` key, else from the end of the decoded prompt.

    The SFT think prompt ends with "<think>\\n" and the no-think prompt with "<think>\\n\\n</think>\\n\\n".
    """
    mode = row.get("mode")
    if mode in MODES:
        return str(mode)
    prompt = str(row.get("input") or "").rstrip()
    if prompt.endswith(THINK_CLOSE):
        return NO_THINK
    if prompt.endswith(THINK_OPEN):
        return THINK
    return UNKNOWN


def think_block(text: str, starts_inside: bool) -> tuple[bool, int | None]:
    """Whether the answer ends inside a thinking block, and how many characters its thinking takes.

    This is ``response_info.split_think`` on text: the block ends at the last ``</think>``, and a
    ``<think>`` after it opens a new block that never closes. ``starts_inside`` is true for a think
    prompt, whose answer begins inside the block. The length is None when there is no block.
    """
    close, opened = text.rfind(THINK_CLOSE), text.rfind(THINK_OPEN)
    if close >= 0:
        if opened > close:
            return True, len(text)
        return False, close
    if opened >= 0 or starts_inside:
        return True, len(text)
    return False, None


def text_loops(text: str) -> bool:
    """The evaluation's loop detector applied to the words and punctuation marks of ``text``.

    It asks for at least 512 units and a rep16 of at least 0.5 over the last 1,024 units, the
    evaluation's thresholds for token ids. A repeated passage repeats in any unit, so a real loop
    is found. Russian text has fewer units than tokens, so a short Russian loop can be missed.
    """
    if len(text) < LOOP_MIN_TOKENS:
        return False
    units = _UNIT.findall(text)
    if len(units) < LOOP_MIN_TOKENS:
        return False
    vocabulary: dict[str, int] = {}
    tail = [vocabulary.setdefault(unit, len(vocabulary)) for unit in units[-LOOP_TAIL_TOKENS:]]
    return tail_rep16(tail) >= LOOP_REP16


def forced_split(text: str) -> tuple[str, str] | None:
    """The model's thinking and its answer on either side of the stop-thinking text, or None without it.

    The agent loop inserts the text once, right after the thinking that the budget stopped, so the
    first occurrence of the sentence is the inserted one. The thinking ends where the blank line
    before the sentence starts, and the answer starts after the "</think>" and blank line after it.
    """
    found = [(text.find(sentence), sentence, before, after) for sentence, before, after in _STOP_PARTS]
    found = [item for item in found if item[0] >= 0]
    if not found:
        return None
    at, sentence, before, after = min(found)
    start = at - len(before) if text[:at].endswith(before) else at
    end = at + len(sentence)
    if text.startswith(after, end):
        end += len(after)
    return text[:start], text[end:]


def _flag(value: Any) -> bool | None:
    """A 0/1 flag of a dump row as a bool, or None when the row has no usable value."""
    if isinstance(value, bool):
        return value
    number = _number(value)
    return None if number is None else number >= 0.5


def row_facts(row: Mapping[str, Any], russian: bool | None = None) -> RowFacts:
    """Read one dumped answer. Flags that the row carries win over what the text suggests.

    Only a think answer can be forced, because the agent loop forces only the modes whose prompt
    opens a thinking block. A forced answer is split at the stop-thinking text, so that its thinking
    length and its loop estimate describe what the model wrote, not the inserted text.
    ``russian`` says whether the prompt is Russian when the caller already knows it, as the RL
    mode does for validation rows matched to a Russian set; otherwise the prompt text decides.
    """
    text = str(row.get("output") or "")
    mode = answer_mode(row)
    unclosed, think_chars = think_block(text, starts_inside=mode == THINK)
    split = forced_split(text) if mode == THINK else None
    given_forced = _flag(row.get("think_forced"))
    forced = split is not None if given_forced is None else given_forced
    given_unclosed = _flag(row.get("think_unclosed"))
    given_loop = _flag(row.get("looped"))
    if given_loop is not None:
        looped = given_loop
    elif forced and split is not None:
        # The thinking is checked first, as the agent loop checks the ids. The answer after the
        # forced close is checked on its own, because it can loop until the length limit as well.
        thinking, after = split
        looped = text_loops(thinking) or text_loops(after)
    else:
        looped = text_loops(text)
    if forced and split is not None:
        think_chars = len(split[0])
    prompt = str(row.get("input") or "")
    russian = russian_prompt(prompt) if russian is None else russian
    ru_think = ru_answer = None
    if russian:
        if forced and split is not None:
            thinking, answer = split
        elif mode == THINK:
            thinking, closed, answer = text.rpartition(THINK_CLOSE)
            if not closed:
                thinking, answer = text, ""
        else:
            thinking, answer = "", text
        ru_think, ru_answer = cyrillic_share(thinking), cyrillic_share(answer)
    return RowFacts(
        mode=mode,
        chars=len(text),
        think_chars=think_chars if mode == THINK else None,
        unclosed=unclosed if given_unclosed is None else given_unclosed,
        forced=forced,
        looped=looped,
        looped_exact=given_loop is not None,
        truncated=_flag(row.get("truncated")),
        score=_number(row.get("score")),
        russian=russian,
        ru_think=ru_think,
        ru_answer=ru_answer,
        kind=prompt_kind(prompt),
        fragment=not russian and _CYRILLIC_WORD.search(text) is not None,
    )


@dataclass
class Group:
    """Sums over the answers of one mode, in one step or in one window of steps."""

    rows: int = 0
    chars: int = 0
    think_rows: int = 0
    think_chars: int = 0
    unclosed: int = 0
    forced: int = 0
    looped: int = 0
    loop_estimates: int = 0
    truncated: int = 0
    truncated_known: int = 0
    scored: int = 0
    score: float = 0.0
    ru_rows: int = 0
    ru_think_n: int = 0
    ru_think: float = 0.0
    ru_answer_n: int = 0
    ru_answer: float = 0.0
    # Russian rows whose thinking or answer prose is less than half Cyrillic, and answers to
    # English prompts that hold a Russian word. The RL rules W6a, W6c and W6e read them.
    ru_think_low: int = 0
    ru_answer_low: int = 0
    fragments: int = 0

    def add(self, fact: RowFacts) -> None:
        self.rows += 1
        self.chars += fact.chars
        if fact.think_chars is not None:
            self.think_rows += 1
            self.think_chars += fact.think_chars
        self.unclosed += int(fact.unclosed)
        self.forced += int(fact.forced)
        self.looped += int(fact.looped)
        self.loop_estimates += int(not fact.looped_exact)
        if fact.truncated is not None:
            self.truncated_known += 1
            self.truncated += int(fact.truncated)
        if fact.score is not None:
            self.scored += 1
            self.score += fact.score
        if fact.russian:
            self.ru_rows += 1
            if fact.ru_think is not None:
                self.ru_think_n += 1
                self.ru_think += fact.ru_think
                self.ru_think_low += int(fact.ru_think < 0.5)
            if fact.ru_answer is not None:
                self.ru_answer_n += 1
                self.ru_answer += fact.ru_answer
                self.ru_answer_low += int(fact.ru_answer < 0.5)
        self.fragments += int(fact.fragment)

    def merged(self, other: Group) -> Group:
        return Group(**{name: getattr(self, name) + getattr(other, name) for name in self.__dataclass_fields__})

    def summary(self) -> dict[str, Any]:
        """Means and shares; a value that no row of the group provides is None."""
        rows = self.rows or 1
        return {
            "rows": self.rows,
            "chars": self.chars / rows,
            "think_chars": self.think_chars / self.think_rows if self.think_rows else None,
            "unclosed": self.unclosed / rows,
            "forced": self.forced / rows,
            "looped": self.looped / rows,
            "looped_estimated": self.loop_estimates > 0,
            "truncated": self.truncated / self.truncated_known
            if self.truncated_known and self.truncated_known == self.rows else None,
            "score": self.score / self.scored if self.scored else None,
            "ru_rows": self.ru_rows,
            "ru_think": self.ru_think / self.ru_think_n if self.ru_think_n else None,
            "ru_answer": self.ru_answer / self.ru_answer_n if self.ru_answer_n else None,
            "ru_think_low": self.ru_think_low / self.ru_think_n if self.ru_think_n else None,
            "ru_answer_low": self.ru_answer_low / self.ru_answer_n if self.ru_answer_n else None,
            "fragments": self.fragments / (self.rows - self.ru_rows) if self.rows > self.ru_rows else None,
        }


# Parsed dump files, keyed by path, with the modification time and size they were parsed at. In
# --follow mode only new or changed files are read again.
_DUMP_CACHE: dict[str, tuple[tuple[int, int], dict[str, Group]]] = {}


def read_dump_dir(directory: Path) -> dict[int, dict[str, Group]]:
    """The answers of every ``<step>.jsonl`` in ``directory``, summed per mode, for all modes and per prompt kind."""
    steps: dict[int, dict[str, Group]] = {}
    if not directory.is_dir():
        return steps
    for path in directory.glob("*.jsonl"):
        if not path.stem.isdigit():
            continue
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        cached = _DUMP_CACHE.get(str(path))
        if cached is None or cached[0] != stamp:
            groups: dict[str, Group] = {}
            for row in read_jsonl(path):
                fact = row_facts(row)
                groups.setdefault(fact.mode, Group()).add(fact)
                groups.setdefault(ALL, Group()).add(fact)
                groups.setdefault(KIND_KEYS[fact.kind], Group()).add(fact)
            cached = (stamp, groups)
            _DUMP_CACHE[str(path)] = cached
        steps[int(path.stem)] = cached[1]
    return steps


# --------------------------------------------------------------------------- series and windows

Points = list[tuple[int, float]]


@dataclass
class Series:
    """One named sequence of checks: (step, value) pairs in step order."""

    name: str
    points: Points


def window_end(step: int, width: int) -> int:
    """The last step of the window that holds ``step``. Windows are steps 1..w, w+1..2w, and so on."""
    return ((step - 1) // width + 1) * width


def window_means(points: Points, width: int) -> Points:
    """The mean of every complete window of training steps, labelled by its last step.

    A window is complete once a step at or after its end exists. Step 0 holds no training.
    """
    training = [(step, value) for step, value in points if step >= 1]
    if not training:
        return []
    last = max(step for step, _ in training)
    buckets: dict[int, list[float]] = {}
    for step, value in training:
        end = window_end(step, width)
        if end <= last:
            buckets.setdefault(end, []).append(value)
    return [(end, statistics.fmean(values)) for end, values in sorted(buckets.items())]


def window_groups(steps: Mapping[int, Mapping[str, Group]], width: int) -> dict[int, dict[str, Group]]:
    """The dumped answers of every complete window, summed per mode; see ``window_means``."""
    training = {step: groups for step, groups in steps.items() if step >= 1}
    if not training:
        return {}
    last = max(training)
    windows: dict[int, dict[str, Group]] = {}
    for step in sorted(training):
        end = window_end(step, width)
        if end > last:
            continue
        target = windows.setdefault(end, {})
        for mode, group in training[step].items():
            target[mode] = target.get(mode, Group()).merged(group)
    return windows


def rolling_mean(points: Points, width: int) -> Points:
    """The mean of each point and the ``width - 1`` points before it, from the first full stretch on."""
    width = max(1, width)
    return [(points[i][0], statistics.fmean(value for _, value in points[i - width + 1:i + 1]))
            for i in range(width - 1, len(points))]


def train_truncated(metrics: Mapping[int, Mapping[str, float]], limit: int | None) -> Points:
    """The exact share of training responses cut by the length limit, per step.

    verl's clip ratio is the share of responses as long as the longest one of the batch. When the
    longest one reached the limit, that is the cut share. When it did not, nothing was cut.
    """
    if limit is None:
        return []
    points = []
    for step in sorted(metrics):
        row = metrics[step]
        if step < 1 or CLIP_KEY not in row or LENGTH_MAX_KEY not in row:
            continue
        points.append((step, row[CLIP_KEY] if row[LENGTH_MAX_KEY] >= limit else 0.0))
    return points


def metric_points(metrics: Mapping[int, Mapping[str, float]], key: str | None) -> Points:
    """The training steps that logged ``key``, with its value."""
    if key is None:
        return []
    return [(step, metrics[step][key]) for step in sorted(metrics) if step >= 1 and key in metrics[step]]


def validation_metrics(metrics: Mapping[int, Mapping[str, float]]) -> dict[str, dict[str, Points]]:
    """Validation means per data source and reward key, as ``{source: {key: [(step, value)]}}``.

    When a source was validated with several samples per prompt, the mean over the most samples is kept.
    """
    best: dict[tuple[str, str, int], tuple[int, float]] = {}
    for step in sorted(metrics):
        for key, value in metrics[step].items():
            match = _VAL_KEY.match(key)
            if match is None:
                continue
            slot = (match["source"], match["var"], step)
            samples = int(match["n"])
            if slot not in best or samples > best[slot][0]:
                best[slot] = (samples, value)
    out: dict[str, dict[str, Points]] = {}
    for (source, var, step), (_, value) in sorted(best.items(), key=lambda item: item[0][2]):
        out.setdefault(source, {}).setdefault(var, []).append((step, value))
    return out


# --------------------------------------------------------------------------- rules


@dataclass
class RuleResult:
    """The outcome of one stop rule: "ok", "FIRED" or "n/a", with the numbers behind it."""

    rule: str
    status: str
    detail: str
    step: int | None = None


def first_above_start(points: Points, factor: float, min_increase: float) -> tuple[int, float] | None:
    """The first check above ``factor`` times the first value and ``min_increase`` above it, or None."""
    base = points[0][1]
    for step, value in points[1:]:
        if value > factor * base and value - base >= min_increase:
            return step, value
    return None


def first_below_start(points: Points, factor: float, min_decrease: float) -> tuple[int, float] | None:
    """The first point below ``factor`` times the first one and at least ``min_decrease`` under it."""
    if len(points) < 2:
        return None
    start = points[0][1]
    for step, value in points[1:]:
        if value < factor * start and start - value >= min_decrease:
            return step, value
    return None


def first_rise_run(points: Points, rises: int, min_total: float) -> tuple[int, list[float]] | None:
    """The first check that ends ``rises`` rises in a row whose sum is at least ``min_total``, or None."""
    for i in range(rises, len(points)):
        values = [value for _, value in points[i - rises:i + 1]]
        if all(later > earlier for earlier, later in zip(values, values[1:])) and values[-1] - values[0] >= min_total:
            return points[i][0], values
    return None


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _combine(rule: str, outcomes: list[tuple[int | None, str]], missing: str) -> RuleResult:
    """One rule's result from the outcome of each series: (step where it fired or None, text)."""
    if not outcomes:
        return RuleResult(rule, "n/a", missing)
    fired = [(step, text) for step, text in outcomes if step is not None]
    if fired:
        return RuleResult(rule, "FIRED", "; ".join(text for _, text in fired), min(step for step, _ in fired))
    return RuleResult(rule, "ok", "; ".join(text for _, text in outcomes))


def rule_above_start(rule: str, series: Sequence[Series], factor: float, min_increase: float,
                     fmt: Callable[[float], str], missing: str) -> RuleResult:
    """Rules 1 and 4: some series climbed above ``factor`` times its first value."""
    outcomes = []
    for item in series:
        if len(item.points) < 2:
            continue
        (start, base), (last, now) = item.points[0], item.points[-1]
        hit = first_above_start(item.points, factor, min_increase)
        if hit is None:
            outcomes.append((None, f"{item.name} {fmt(base)} at step {start}, {fmt(now)} at step {last}"))
        else:
            outcomes.append((hit[0], f"{item.name} {fmt(base)} at step {start} -> {fmt(hit[1])} at step {hit[0]}"))
    return _combine(rule, outcomes, missing)


def rule_below_start(rule: str, series: Sequence[Series], factor: float, min_decrease: float,
                     missing: str) -> RuleResult:
    """Fires when any series falls below ``factor`` times its first value (``first_below_start``)."""
    outcomes = []
    for item in series:
        if len(item.points) < 2:
            continue
        hit = first_below_start(item.points, factor, min_decrease)
        if hit is None:
            outcomes.append((None, f"{item.name} {item.points[0][1]:.3f} at step {item.points[0][0]}, "
                                   f"last {item.points[-1][1]:.3f}"))
        else:
            outcomes.append((hit[0], f"{item.name} {item.points[0][1]:.3f} at step {item.points[0][0]} -> "
                                     f"{hit[1]:.3f} at step {hit[0]}"))
    return _combine(rule, outcomes, missing)


def rule_rising(rule: str, series: Sequence[Series], rises: int, min_total: float, missing: str) -> RuleResult:
    """Rule 2: some series rose at ``rises`` checks in a row."""
    outcomes = []
    for item in series:
        if len(item.points) < 2:
            continue
        hit = first_rise_run(item.points, rises, min_total)
        if hit is None:
            values = ", ".join(_fmt(value) for _, value in item.points[-(rises + 1):])
            outcomes.append((None, f"{item.name} last checks {values}"))
        else:
            outcomes.append((hit[0], f"{item.name} rose to step {hit[0]}: {', '.join(_fmt(v) for v in hit[1])}"))
    return _combine(rule, outcomes, missing)


def run_above_median(points: Points, factor: float, baseline_steps: int, run: int,
                     warmup: int) -> tuple[int, Points] | None:
    """The first ``run`` points in a row above ``factor`` times the median of the first ``baseline_steps``.

    Steps up to ``warmup`` never count as high. Returns the step that completes the run and the
    points of the run, or None.
    """
    median = statistics.median(value for _, value in points[:baseline_steps])
    streak: Points = []
    for step, value in points:
        streak = streak + [(step, value)] if value > factor * median and step > warmup else []
        if len(streak) >= run:
            return step, streak[-run:]
    return None


def rule_grad_norm(points: Points, th: Thresholds) -> RuleResult:
    """Rule 3: ``grad_run`` steps in a row above ``grad_factor`` times the median of the first steps."""
    rule = (f"grad_norm above {th.grad_factor:g}x the median of the first {th.grad_baseline_steps} steps "
            f"for {th.grad_run} steps in a row")
    baseline = points[:th.grad_baseline_steps]
    if len(baseline) < th.min_grad_baseline:
        return RuleResult(rule, "n/a", f"{len(points)} steps with {GRAD_KEY}; the median needs {th.min_grad_baseline}")
    median = statistics.median(value for _, value in baseline)
    if median <= 0:
        return RuleResult(rule, "n/a", f"the median of {GRAD_KEY} over the first steps is {median}")
    provisional = "" if len(baseline) >= th.grad_baseline_steps else f" (median of only {len(baseline)} steps so far)"
    hit = run_above_median(points, th.grad_factor, th.grad_baseline_steps, th.grad_run, th.grad_warmup)
    if hit is not None:
        step, run = hit
        values = ", ".join(f"{v:.4g}" for _, v in run)
        return RuleResult(rule, "FIRED", f"steps {run[0][0]}-{step}: {values} above "
                          f"{th.grad_factor:g} x {median:.4g}{provisional}", step)
    peak = max(points, key=lambda point: point[1])
    return RuleResult(rule, "ok", f"median {median:.4g}{provisional}, highest {peak[1]:.4g} at step {peak[0]}, "
                      f"last {points[-1][1]:.4g}")


def rule_kl(points: Points, key: str | None, th: Thresholds) -> RuleResult:
    """Rule 5: the teacher KL, averaged over ``kl_smooth`` steps, rose ``kl_rises`` steps in a row."""
    rule = f"teacher KL rising {th.kl_rises} steps in a row (mean over {th.kl_smooth} steps)"
    if key is None:
        return RuleResult(rule, "n/a", f"metrics.jsonl has none of {', '.join(KL_KEYS)} yet")
    smooth = rolling_mean(points, th.kl_smooth)
    if len(smooth) <= th.kl_rises:
        return RuleResult(rule, "n/a", f"{len(points)} steps with {key}; the rule needs "
                          f"{th.kl_rises + th.kl_smooth}")
    hit = first_rise_run(smooth, th.kl_rises, 0.0)
    if hit is not None:
        return RuleResult(rule, "FIRED", f"{key} rose from {hit[1][0]:.4g} to {hit[1][-1]:.4g} up to step {hit[0]}",
                          hit[0])
    rises, longest = 0, 0
    for (_, earlier), (_, later) in zip(smooth, smooth[1:]):
        rises = rises + 1 if later > earlier else 0
        longest = max(longest, rises)
    return RuleResult(rule, "ok", f"{key} first {smooth[0][1]:.4g}, last {smooth[-1][1]:.4g}, "
                      f"longest rise {longest} steps")


# --------------------------------------------------------------------------- the report


@dataclass
class Report:
    """Everything one check found: the tables and the rule results."""

    run_dir: str
    limit: int | None
    limit_source: str
    kl_key: str | None
    window: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)
    validation_answers: list[dict[str, Any]] = field(default_factory=list)
    rules: list[RuleResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def fired(self) -> list[RuleResult]:
        return [result for result in self.rules if result.status == "FIRED"]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["fired"] = [result.rule for result in self.fired]
        return out


def _kl_key(metrics: Mapping[int, Mapping[str, float]], given: str | None) -> str | None:
    if given is not None:
        return given
    present = {key for row in metrics.values() for key in row}
    return next((key for key in KL_KEYS if key in present), None)


def _mode_series(windows: Mapping[int, Mapping[str, Group]], name: str, key: str,
                 modes: Sequence[str] = MODES) -> list[Series]:
    """One series per mode of a summary value over the training windows; windows without the value are skipped."""
    series = []
    for mode in modes:
        points = []
        for end in sorted(windows):
            group = windows[end].get(mode)
            value = None if group is None or not group.rows else group.summary()[key]
            if value is not None:
                points.append((end, float(value)))
        if points:
            series.append(Series(f"{name} {mode}", points))
    return series


def build_report(run_dir: Path, th: Thresholds = Thresholds(), *, limit: int | None = None,
                 kl_key: str | None = None) -> Report:
    """Read the run dir once and evaluate every rule."""
    metrics = read_metrics(run_dir)
    rollouts = read_dump_dir(run_dir / "rollouts")
    answers = read_dump_dir(run_dir / "val_generations")
    response_cap, cap_source = response_limit(run_dir, limit)
    key = _kl_key(metrics, kl_key)
    report = Report(str(run_dir), response_cap, cap_source, key, th.window)

    kl, grad = metric_points(metrics, key), metric_points(metrics, GRAD_KEY)
    cut = dict(train_truncated(metrics, response_cap))
    for step in sorted(set(step for step in metrics if step >= 1) | set(step for step in rollouts if step >= 1)):
        row = metrics.get(step, {})
        report.steps.append({
            "step": step,
            "teacher_kl": row.get(key) if key else None,
            "grad_norm": row.get(GRAD_KEY),
            "response_length": row.get(LENGTH_KEY),
            "truncated": cut.get(step),
            "score": row.get(SCORE_KEY),
            "modes": {mode: group.summary() for mode, group in sorted(rollouts.get(step, {}).items())},
        })
    by_source = validation_metrics(metrics)
    for source, values in sorted(by_source.items()):
        for step in sorted({step for points in values.values() for step, _ in points}):
            entry = {"step": step, "source": source}
            entry.update({var: dict(points).get(step) for var, points in values.items()})
            report.validation.append(entry)
    report.validation.sort(key=lambda entry: (entry["step"], entry["source"]))
    for step in sorted(answers):
        for mode, group in sorted(answers[step].items()):
            if mode != ALL and mode not in KIND_KEYS.values():
                report.validation_answers.append({"step": step, "mode": mode, **group.summary()})

    windows = window_groups(rollouts, th.window)
    # Rule 1. Validation sources first, because their first value is the start model at step 0.
    val_cut = [Series(f"val {source}", values["truncated"]) for source, values in by_source.items()
               if "truncated" in values]
    train_cut = [Series("train all rows", window_means(sorted(cut.items()), th.window))]
    missing_cut = ("no validation and no complete training window yet" if response_cap is not None else
                   f"no validation yet, and the training share needs the response limit ({cap_source})")
    report.rules.append(rule_above_start(
        f"truncated share above {th.truncated_factor:g}x its first value",
        [*val_cut, *train_cut, *_mode_series(windows, "train", "truncated")],
        th.truncated_factor, th.min_share_increase, _fmt, missing_cut))

    # Rule 2. The training series are text estimates; the validation series are exact.
    val_loop = [Series(f"val {source}", values["looped"]) for source, values in by_source.items() if "looped" in values]
    report.rules.append(rule_rising(
        f"looped share rising {th.loop_rises} checks in a row", [*val_loop, *_mode_series(windows, "train", "looped")],
        th.loop_rises, th.min_loop_rise, "no validation and no complete training window yet"))

    # Rule 3.
    report.rules.append(rule_grad_norm(grad, th) if grad else RuleResult(
        "grad_norm above the median of the first steps", "n/a", f"metrics.jsonl has no {GRAD_KEY} yet"))

    # Rule 4. The validation answers of the think sets answer the same problems at every check.
    val_think = [(step, float(groups[THINK].summary()["chars"])) for step, groups in sorted(answers.items())
                 if THINK in groups and groups[THINK].rows]
    report.rules.append(rule_above_start(
        f"mean think answer length above {th.length_factor:g}x its first value (characters)",
        [*_mode_series(windows, "train", "chars", (THINK,)), *([Series("val think", val_think)] if val_think else [])],
        th.length_factor, 0.0, lambda value: f"{value:.0f}", "no think answers in the dumps yet"))

    # Rule 5.
    report.rules.append(rule_kl(kl, key, th))

    # Rule 6. Per prompt kind over all modes, because a Russian prompt can be in either mode and a window has
    # few of each.
    guarded = tuple(KIND_KEYS[kind] for kind in RU_GUARD_KINDS)
    report.rules.append(rule_below_start(
        f"Cyrillic share on Russian {' and '.join(RU_GUARD_KINDS)} prompts below {th.ru_factor:g}x its first value",
        [*_mode_series(windows, "train thinking", "ru_think", guarded),
         *_mode_series(windows, "train answer", "ru_answer", guarded)],
        th.ru_factor, th.min_ru_drop, "no Russian prompts in a complete training window yet"))

    unknown = sum(groups[UNKNOWN].rows for groups in rollouts.values() if UNKNOWN in groups)
    if unknown:
        report.notes.append(f"{unknown} rollout answers have a prompt that ends in neither <think> nor </think>, "
                            "so their mode is unknown and they count only in the all-rows numbers.")
    if any(group.loop_estimates for groups in rollouts.values() for group in groups.values()):
        report.notes.append("The loop shares of training rollouts are estimated from the decoded text; the "
                            "validation numbers are the reward's exact flags.")
    # The agent loop applies the thinking budget to training rows only, so that validation measures
    # how the model closes its thinking by itself. A forced validation answer means that this broke.
    forced_answers = sum(groups[ALL].forced for groups in answers.values() if ALL in groups)
    if forced_answers:
        report.notes.append(f"{forced_answers} validation answers count as forced (the reward's think_forced flag, "
                            "or the stop-thinking sentence in the text), although the agent loop never forces "
                            "validation rows. Check extra_info.split of the validation sets. A model that writes "
                            "the sentence by itself also shows up here.")
    return report


# --------------------------------------------------------------------------- printing


def _cell(value: Any, width: int, digits: int = 3) -> str:
    if value is None:
        text = "-"
    elif isinstance(value, float):
        text = f"{value:.{digits}f}"
    else:
        text = str(value)
    return text.rjust(width)


def render(report: Report, last: int = 20) -> str:
    """The report as text tables. ``last`` limits the training table to its last rows; 0 shows all."""
    lines = [f"watch {report.run_dir}",
             f"  response limit: {report.limit if report.limit is not None else '-'} tokens ({report.limit_source})",
             f"  teacher KL: {report.kl_key or '-'}; training window: {report.window} steps", ""]
    steps = report.steps[-last:] if last > 0 else report.steps
    lines.append(f"training: {len(steps)} of {len(report.steps)} steps (per-mode columns from rollouts/, "
                 "chars = characters of the decoded answer, forced = share closed at the thinking budget, ru = "
                 "Russian prompts of each kind and the Cyrillic share of their thinking and answer prose)")
    header = (f"{'step':>5} {'kl':>8} {'grad':>8} {'len':>7} {'cut':>6} {'score':>6} |"
              + "".join(f" {mode + ' n':>10} {'chars':>7} {'uncl':>6} {'loop':>6} {'forced':>6} |" for mode in MODES)
              + "".join(f" {kind + ' ru n':>10} {'thk':>6} {'ans':>6} |" for kind in RU_GUARD_KINDS))
    lines.append(header)
    for row in steps:
        text = (f"{row['step']:>5} {_cell(row['teacher_kl'], 8, 4)} {_cell(row['grad_norm'], 8, 4)} "
                f"{_cell(row['response_length'], 7, 0)} {_cell(row['truncated'], 6)} {_cell(row['score'], 6)} |")
        for mode in MODES:
            stats = row["modes"].get(mode)
            if stats is None:
                text += f" {'-':>10} {'-':>7} {'-':>6} {'-':>6} {'-':>6} |"
            else:
                text += (f" {stats['rows']:>10} {_cell(stats['chars'], 7, 0)} {_cell(stats['unclosed'], 6)} "
                         f"{_cell(stats['looped'], 6)} {_cell(stats['forced'], 6)} |")
        for kind in RU_GUARD_KINDS:
            stats = row["modes"].get(KIND_KEYS[kind])
            if stats is None:
                text += f" {'-':>10} {'-':>6} {'-':>6} |"
            else:
                text += f" {stats['ru_rows']:>10} {_cell(stats['ru_think'], 6)} {_cell(stats['ru_answer'], 6)} |"
        lines.append(text)
    if report.validation:
        lines += ["", "validation (metrics.jsonl, the reward's exact flags):",
                  f"{'step':>5}  {'data source':<28} {'acc':>6} {'cut':>6} {'uncl':>6} {'loop':>6}"]
        for entry in report.validation:
            lines.append(f"{entry['step']:>5}  {entry['source']:<28} {_cell(entry.get('acc'), 6)} "
                         f"{_cell(entry.get('truncated'), 6)} {_cell(entry.get('think_unclosed'), 6)} "
                         f"{_cell(entry.get('looped'), 6)}")
    if report.validation_answers:
        lines += ["", "validation answers (val_generations/):",
                  f"{'step':>5}  {'mode':<9} {'rows':>6} {'chars':>7} {'think':>7}"]
        for entry in report.validation_answers:
            lines.append(f"{entry['step']:>5}  {entry['mode']:<9} {entry['rows']:>6} {_cell(entry['chars'], 7, 0)} "
                         f"{_cell(entry['think_chars'], 7, 0)}")
    lines += ["", "stop rules:"]
    for result in report.rules:
        lines.append(f"  {result.status:<6} {result.rule}")
        lines.append(f"         {result.detail}")
    for note in report.notes:
        lines.append(f"note: {note}")
    fired = report.fired
    lines.append("")
    lines.append("STOP RULE FIRED: " + "; ".join(result.rule for result in fired) if fired else "no stop rule fired")
    return "\n".join(lines)


# --------------------------------------------------------------------------- RL mode (--rl)

# The training metrics that only the RL rules read. ``actor/kl_loss`` is the low_var_kl term to the
# reference model. The RLTT loss logs each loop's mean log-prob of the sampled tokens; the highest
# index is the final loop, the one that vLLM samples from.
RL_KL_KEY = "actor/kl_loss"
EXIT_KEY = "actor/rltt/expected_exit_step"
_LOOP_LOGP_KEY = re.compile(r"^actor/rltt/loop_logp_t(\d+)$")

NOTIFY = "notify"
CONFIRMED = "confirmed"
HARD = "hard-stop"
EXIT_HARD = 4

# The validation sources whose accuracy W1 watches by default, with the drops for (two evals in a
# row, one eval, hard stop on two evals in a row). A run whose MATH-500 think set carries no tag
# reports it as plain "orbit/math", so that source is watched as well. LCB is left out: see W1.
W1_DEFAULT = {"orbit/math#think": (0.03, 0.05, 0.08), "orbit/math": (0.03, 0.05, 0.08)}
# MGSM-ru and the same 250 GSM8K problems in English, whose accuracy gap W6d watches.
MGSM_RU_SOURCE = "orbit/math#ru"
MGSM_EN_SOURCE = "orbit/math#mgsm_en"
# LCB starts with most answers unclosed, so W4d gives these sources no absolute floor.
CODE_SOURCE_PREFIX = "orbit/code"

# The frozen validation files of the RL recipe, by the role the rules give them. --val-dir looks for
# these names, and --val-file ROLE=PATH sets one file. The question is the jsonl field that the
# dataset builder (distill_data.QUESTION_FIELD) reads.
VAL_FILES = {"math500": "math500.jsonl", "gsm8k_en": "gsm8k_en_paired250.jsonl", "mgsm_ru": "mgsm_ru.jsonl",
             "lcb": "lcb_val78.jsonl", "lcb200": "lcb_val200.jsonl", "chat_ru": "chat_probe_ru.jsonl",
             "chat_en": "chat_probe_en.jsonl"}
QUESTION_FIELD = "instruction"
RU_ROLES = ("mgsm_ru", "chat_ru")
CHAT_ROLES = ("chat_ru", "chat_en")


@dataclass(frozen=True)
class RLThresholds:
    """The numbers of the rules W1-W8; the module docstring explains each rule."""

    grad_factor: float = 2.0
    grad_baseline_steps: int = 20
    grad_run: int = 3
    warmup: int = 5
    kl_smooth: int = 5
    kl_notify: float = 0.045
    kl_hard: float = 0.06
    loop_smooth: int = 3
    loop_share: float = 0.05
    cut_smooth: int = 5
    cut_factor: float = 2.0
    cut_margin: float = 0.05
    cut_run: int = 3
    length_steps: int = 5
    length_factor: float = 0.5
    unclosed_floor: float = 0.5
    unclosed_rise: float = 0.15
    exit_smooth: int = 5
    exit_drop: float = 0.03
    mgsm_cyrillic: float = 0.95
    mgsm_low_share: float = 0.02
    chat_think_drop: float = 0.05
    chat_answer_drop: float = 0.03
    chat_low_rise: float = 0.03
    gap_drop: float = 0.06
    fragment_factor: float = 3.0
    fragment_margin: float = 0.01
    chat_flag_rise: float = 0.05
    chat_length_low: float = 0.5
    chat_length_high: float = 1.5
    loop_gap_factor: float = 1.5
    window: int = 5


@dataclass(frozen=True)
class Fire:
    """One rule firing: the rule, its level (notify, confirmed or hard-stop), the step and the numbers."""

    rule: str
    level: str
    step: int
    series: str
    detail: str

    def line(self) -> str:
        return f"FIRE {self.rule} {self.level} at step {self.step}: {self.series}: {self.detail}"


# One check of an RL rule: the step, whether the check fails, and the numbers behind it.
Check = tuple[int, bool, str]


def streak_fires(rule: str, series: str, checks: Sequence[Check], *, notify_after: int | None = 1,
                 confirm_after: int | None = 2, confirmed_level: str = CONFIRMED) -> list[Fire]:
    """The fires of a rule that judges each check on its own.

    ``notify_after`` failing checks in a row give the notify fire, and ``confirm_after`` failing
    checks in a row give the confirmed fire (or the hard stop, with ``confirmed_level``). None turns
    a level off. Only the first fire of each level is kept, because the step where a problem began
    is what matters.
    """
    fires: list[Fire] = []
    streak = 0
    for step, failing, detail in checks:
        streak = streak + 1 if failing else 0
        if notify_after is not None and streak == notify_after and not any(f.level == NOTIFY for f in fires):
            fires.append(Fire(rule, NOTIFY, step, series, detail))
        if confirm_after is not None and streak == confirm_after:
            fires.append(Fire(rule, confirmed_level, step, series, detail))
            break
    return fires


def _normal(question: str) -> str:
    """A question with its whitespace collapsed, so that a decoded prompt and its source file compare equal."""
    return " ".join(question.split())


def load_val_index(files: Mapping[str, Path]) -> dict[str, str]:
    """Every question of the frozen validation files, normalized, mapped to the role of its file.

    MATH-500 think and no_think share one file, and both copies of the Russian chat probe share one
    as well, so a role can stand for several data sources. The rules tell think from no_think by the
    prompt, and they pool the two probe copies on purpose.
    """
    index: dict[str, str] = {}
    for role, path in files.items():
        for record in read_jsonl(path):
            question = record.get(QUESTION_FIELD)
            if isinstance(question, str) and question.strip():
                index.setdefault(_normal(question), role)
    return index


def val_role(prompt: str, index: Mapping[str, str]) -> tuple[str, bool]:
    """The role of a decoded validation prompt, and whether it was found in the frozen files.

    A prompt that matches no file is classified from its template: the instruction gives the kind,
    and the Cyrillic of the question tells Russian from English.
    """
    role = index.get(_normal(question_text(prompt)))
    if role is not None:
        return role, True
    kind, russian = prompt_kind(prompt), russian_prompt(prompt)
    if kind == MATH:
        return ("mgsm_ru" if russian else "math"), False
    if kind == CODE:
        return "code", False
    return ("chat_ru" if russian else "chat_en"), False


# Parsed validation dumps, keyed by path, with the file stamp and the index they were parsed with.
_ROLE_CACHE: dict[str, tuple[tuple[int, int, int], dict[str, Group], int]] = {}


def read_val_roles(directory: Path, index: Mapping[str, str]) -> tuple[dict[int, dict[str, Group]], dict[int, int]]:
    """The validation answers of every ``<step>.jsonl``, summed per role, and per step the rows that matched no file."""
    steps: dict[int, dict[str, Group]] = {}
    unmatched: dict[int, int] = {}
    if not directory.is_dir():
        return steps, unmatched
    fingerprint = hash(frozenset(index.items()))
    for path in directory.glob("*.jsonl"):
        if not path.stem.isdigit():
            continue
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size, fingerprint)
        cached = _ROLE_CACHE.get(str(path))
        if cached is None or cached[0] != stamp:
            groups: dict[str, Group] = {}
            missed = 0
            for row in read_jsonl(path):
                role, found = val_role(str(row.get("input") or ""), index)
                missed += int(not found)
                groups.setdefault(role, Group()).add(row_facts(row, russian=role in RU_ROLES))
            cached = (stamp, groups, missed)
            _ROLE_CACHE[str(path)] = cached
        steps[int(path.stem)] = cached[1]
        unmatched[int(path.stem)] = cached[2]
    return steps, unmatched


def loop_gap_points(metrics: Mapping[int, Mapping[str, float]]) -> Points:
    """Per training step, the largest gap between an intermediate loop's mean log-prob and the final loop's."""
    points = []
    for step in sorted(metrics):
        loops = {int(match[1]): value for key, value in metrics[step].items() if (match := _LOOP_LOGP_KEY.match(key))}
        if step >= 1 and len(loops) >= 2:
            final = loops[max(loops)]
            points.append((step, max(abs(value - final) for t, value in loops.items() if t != max(loops))))
    return points


def role_points(roles: Mapping[int, Mapping[str, Group]], role: str, key: str) -> Points:
    """One summary value of a role over the validation steps; steps without it are skipped."""
    points = []
    for step in sorted(roles):
        group = roles[step].get(role)
        value = None if group is None or not group.rows else group.summary()[key]
        if value is not None:
            points.append((step, float(value)))
    return points


def rule_w1(source: str, points: Points, twice: float, once: float, hard: float) -> list[Fire]:
    """W1: the 2-eval running mean of a source's accuracy against its best so far.

    One eval of MATH-500 has an SD of about 0.012, so a single low eval is often noise. The mean of
    each eval and the one before it halves that variance, and the hard stop needs the large drop on
    two running means in a row.
    """
    means = [(points[i][0], (points[i - 1][1] + points[i][1]) / 2) for i in range(1, len(points))]
    fires: list[Fire] = []
    best = -math.inf
    low_before = hard_before = False
    for step, mean in means:
        best = max(best, mean)
        drop = best - mean
        # A small tolerance, so that a drop of exactly the threshold counts despite float rounding.
        low, large = drop >= twice - 1e-9, drop >= hard - 1e-9
        detail = f"2-eval mean {mean:.3f}, best {best:.3f}, drop {drop:.3f}"
        if not fires and (drop >= once - 1e-9 or (low and low_before)):
            fires.append(Fire("W1", NOTIFY, step, f"val {source} acc", detail))
        if large and hard_before:
            fires.append(Fire("W1", HARD, step, f"val {source} acc", detail + f" (>= {hard:g} twice)"))
            break
        low_before, hard_before = low, large
    return fires


def _f(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


@dataclass
class RLReport:
    """What one RL check found: status lines in step order, the fires, and notes."""

    run_dir: str
    limit: int | None
    limit_source: str
    lines: list[tuple[int, int, str]] = field(default_factory=list)
    fires: list[Fire] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def first(self, levels: Sequence[str]) -> Fire | None:
        found = [fire for fire in self.fires if fire.level in levels]
        return min(found, key=lambda fire: fire.step) if found else None

    def as_dict(self) -> dict[str, Any]:
        notify, hard = self.first((NOTIFY, CONFIRMED, HARD)), self.first((HARD,))
        return {"run_dir": self.run_dir, "limit": self.limit, "lines": [text for _, _, text in self.lines],
                "fires": [asdict(fire) for fire in self.fires], "notes": self.notes,
                "first_notify": notify.step if notify else None, "first_hard": hard.step if hard else None}


def status_lines(metrics: Mapping[int, Mapping[str, float]], dumps: Mapping[int, Mapping[str, Group]],
                 roles: Mapping[int, Mapping[str, Group]], by_source: Mapping[str, Mapping[str, Points]],
                 cut: Points, th: RLThresholds) -> list[tuple[int, int, str]]:
    """One line per training step (order 0) and one per validation (order 1), sorted by step.

    A training line exists once verl logged the step's metrics, and a validation line once the
    validation metrics of the step are logged. verl writes both dumps before it logs, so a line
    never shows a half-written dump.
    """
    kl5 = dict(rolling_mean(metric_points(metrics, RL_KL_KEY), th.kl_smooth))
    cuts, cut5, gaps = dict(cut), dict(rolling_mean(cut, th.cut_smooth)), dict(loop_gap_points(metrics))
    lines = []
    for step in sorted(step for step in metrics if step >= 1 and GRAD_KEY in metrics[step]):
        row, groups = metrics[step], dumps.get(step, {})
        scores = " ".join(f"{kind} {_f(groups[KIND_KEYS[kind]].summary()['score'])}"
                          for kind in (MATH, CODE) if KIND_KEYS[kind] in groups)
        every = groups[ALL].summary() if ALL in groups else {}
        lines.append((step, 0, (
            f"step {step:>4} score {_f(row.get(SCORE_KEY))} [{scores}] kl {_f(row.get(RL_KL_KEY), 4)} "
            f"kl5 {_f(kl5.get(step), 4)} grad {_f(row.get(GRAD_KEY), 4)} len {_f(row.get(LENGTH_KEY), 0)} "
            f"cut {_f(cuts.get(step))} cut5 {_f(cut5.get(step))} loop {_f(every.get('looped'))} "
            f"frag {_f(every.get('fragments'))} gap {_f(gaps.get(step))} exit {_f(row.get(EXIT_KEY))}")))
    for step in sorted({step for values in by_source.values() for points in values.values() for step, _ in points}):
        parts = []
        for source, values in sorted(by_source.items()):
            acc, unclosed = dict(values.get("acc", [])).get(step), dict(values.get("think_unclosed", [])).get(step)
            if acc is not None or unclosed is not None:
                parts.append(f"{source.removeprefix('orbit/')} {_f(acc)}/{_f(unclosed)}")
        groups = roles.get(step, {})
        for role in (*RU_ROLES, "chat_en"):
            if role in groups:
                s = groups[role].summary()
                parts.append(f"{role} n {s['rows']} thk {_f(s['ru_think'])} ans {_f(s['ru_answer'])} "
                             f"ans<.5 {_f(s['ru_answer_low'])} loop {_f(s['looped'])} uncl {_f(s['unclosed'])} "
                             f"chars {_f(s['chars'], 0)}")
        lines.append((step, 1, f"val  {step:>4} acc/uncl " + " | ".join(parts)))
    return sorted(lines)


def train_rules(metrics: Mapping[int, Mapping[str, float]], dumps: Mapping[int, Mapping[str, Group]],
                cut: Points, th: RLThresholds) -> list[Fire]:
    """W2, W3, W4a-c, W5 and W8: the rules on training metrics and training rollouts."""
    fires: list[Fire] = []
    grad, gap = metric_points(metrics, GRAD_KEY), loop_gap_points(metrics)
    # W2 and W8 wait for the full median of steps 1-20, which includes the warmup steps that the
    # streak itself skips.
    for rule, series, points, factor in (("W2", GRAD_KEY, grad, th.grad_factor),
                                         ("W8", "max |loop_logp_t - loop_logp_final|", gap, th.loop_gap_factor)):
        if len(points) < th.grad_baseline_steps:
            continue
        hit = run_above_median(points, factor, th.grad_baseline_steps, th.grad_run, th.warmup)
        if hit is not None:
            median = statistics.median(value for _, value in points[:th.grad_baseline_steps])
            fires.append(Fire(rule, NOTIFY, hit[0], series, f"{', '.join(f'{v:.4g}' for _, v in hit[1])} above "
                              f"{factor:g} x {median:.4g}, the median of steps 1-{th.grad_baseline_steps}"))
    # W3. The notify and the hard stop are two levels of the same mean.
    kl = rolling_mean(metric_points(metrics, RL_KL_KEY), th.kl_smooth)
    series = f"{RL_KL_KEY} {th.kl_smooth}-step mean"
    fires += streak_fires("W3", series, [(step, value > th.kl_notify, f"{value:.4f} above {th.kl_notify:g}")
                                         for step, value in kl], confirm_after=None)
    if th.kl_hard > 0:
        fires += streak_fires("W3", series, [(step, value > th.kl_hard, f"{value:.4f} above {th.kl_hard:g}")
                                             for step, value in kl], notify_after=None, confirm_after=1,
                              confirmed_level=HARD)
    # W4a. The loop flag of a training answer is the text estimate.
    loops = [(step, float(groups[ALL].summary()["looped"])) for step, groups in sorted(dumps.items())
             if step >= 1 and ALL in groups and groups[ALL].rows]
    fires += streak_fires("W4a", f"train looped share {th.loop_smooth}-step mean",
                          [(step, value > th.loop_share, f"{value:.3f} above {th.loop_share:g}")
                           for step, value in rolling_mean(loops, th.loop_smooth)], confirm_after=None)
    # W4b. Each 5-step mean is compared with the lowest 5-step mean before it.
    checks, lowest = [], math.inf
    for step, value in rolling_mean(cut, th.cut_smooth):
        bar = max(th.cut_factor * lowest, lowest + th.cut_margin)
        checks.append((step, value >= bar, f"{value:.3f} at or above {bar:.3f}, max({th.cut_factor:g}x, "
                                           f"+{th.cut_margin:g}) of the lowest earlier mean {lowest:.3f}"))
        lowest = min(lowest, value)
    fires += streak_fires("W4b", f"train truncated share {th.cut_smooth}-step mean, {th.cut_run} steps in a row",
                          checks, notify_after=th.cut_run, confirm_after=None)
    # W4c.
    length = metric_points(metrics, LENGTH_KEY)
    start = [value for step, value in length if step <= th.length_steps]
    if len(start) >= th.length_steps:
        base = statistics.fmean(start)
        fires += streak_fires("W4c", LENGTH_KEY, [
            (step, value < th.length_factor * base, f"{value:.0f} below {th.length_factor:g} x {base:.0f}, "
                                                    f"the mean of steps 1-{th.length_steps}")
            for step, value in length], confirm_after=None)
    # W5.
    checks, highest = [], -math.inf
    for step, value in rolling_mean(metric_points(metrics, EXIT_KEY), th.exit_smooth):
        checks.append((step, value < highest - th.exit_drop,
                       f"{value:.3f} more than {th.exit_drop:g} below the highest earlier mean {highest:.3f}"))
        highest = max(highest, value)
    fires += streak_fires("W5", f"{EXIT_KEY} {th.exit_smooth}-step mean", checks, confirm_after=None)
    return fires


def val_rules(by_source: Mapping[str, Mapping[str, Points]], w1: Mapping[str, tuple[float, float, float]],
              th: RLThresholds) -> list[Fire]:
    """W1, W4d and W6d: the rules on the validation metrics that verl logs per data source."""
    fires: list[Fire] = []
    for source, (twice, once, hard) in w1.items():
        if "acc" in by_source.get(source, {}):
            fires += rule_w1(source, by_source[source]["acc"], twice, once, hard)
    # W4d is relative to each source's first validation, which is step 0 on a fresh start.
    for source, values in sorted(by_source.items()):
        points = values.get("think_unclosed", [])
        if len(points) < 2:
            continue
        (start, base), bar = points[0], points[0][1] + th.unclosed_rise
        if not source.startswith(CODE_SOURCE_PREFIX):
            bar = max(th.unclosed_floor, bar)
        fires += streak_fires("W4d", f"val {source} think_unclosed", [
            (step, value >= bar, f"{value:.3f} at or above {bar:.3f} (step {start}: {base:.3f})")
            for step, value in points[1:]], confirmed_level=HARD)
    ru, en = dict(by_source.get(MGSM_RU_SOURCE, {}).get("acc", [])), dict(by_source.get(MGSM_EN_SOURCE, {}).get("acc", []))
    paired = [(step, ru[step] - en[step]) for step in sorted(set(ru) & set(en))]
    if len(paired) >= 2:
        (start, base) = paired[0]
        fires += streak_fires("W6d", "val acc(MGSM-ru) - acc(GSM8K-en paired)", [
            (step, value <= base - th.gap_drop, f"{value:+.3f}, step {start} {base:+.3f}, alarm at -{th.gap_drop:g}")
            for step, value in paired[1:]])
    return fires


def probe_rules(roles: Mapping[int, Mapping[str, Group]], dumps: Mapping[int, Mapping[str, Group]],
                th: RLThresholds) -> list[Fire]:
    """W6a-c, W6e-f and W7: the Russian and chat rules, read from the validation and rollout dumps."""
    fires: list[Fire] = []
    # W6a has absolute thresholds, because Russian thinking on MGSM-ru starts at 0.996.
    think, low = dict(role_points(roles, "mgsm_ru", "ru_think")), dict(role_points(roles, "mgsm_ru", "ru_think_low"))
    fires += streak_fires("W6a", "val mgsm_ru think Cyrillic", [
        (step, think[step] < th.mgsm_cyrillic or low.get(step, 0.0) > th.mgsm_low_share,
         f"mean {think[step]:.3f} (alarm below {th.mgsm_cyrillic:g}), rows below 0.5 {_f(low.get(step))} "
         f"(alarm above {th.mgsm_low_share:g})") for step in sorted(think)])
    # W6b and W6c read both copies of the Russian probe, pooled into one role.
    think = role_points(roles, "chat_ru", "ru_think")
    if len(think) >= 2:
        (start, base) = think[0]
        fires += streak_fires("W6b", "val chat_ru think Cyrillic", [
            (step, value <= base - th.chat_think_drop, f"{value:.3f}, step {start} {base:.3f}, "
                                                       f"alarm at -{th.chat_think_drop:g}")
            for step, value in think[1:]])
    answer, low = dict(role_points(roles, "chat_ru", "ru_answer")), dict(role_points(roles, "chat_ru", "ru_answer_low"))
    steps = sorted(set(answer) & set(low))
    if len(steps) >= 2:
        start = steps[0]
        fires += streak_fires("W6c", "val chat_ru answer Cyrillic", [
            (step, answer[step] <= answer[start] - th.chat_answer_drop or low[step] >= low[start] + th.chat_low_rise,
             f"mean {answer[step]:.3f} (step {start} {answer[start]:.3f}), answers below 0.5 {low[step]:.3f} "
             f"(step {start} {low[start]:.3f})") for step in steps[1:]])
    # W6e and W6f read complete training windows.
    windows = window_groups(dumps, th.window)
    for series in _mode_series(windows, "train", "fragments", (ALL,)):
        base = series.points[0][1]
        fires += streak_fires("W6e", "train share of English answers with Cyrillic words, per window", [
            (step, value > th.fragment_factor * base and value - base >= th.fragment_margin,
             f"{value:.3f}, steps 1-{th.window} {base:.3f}") for step, value in series.points[1:]])
    rule6 = Thresholds()
    guarded = tuple(KIND_KEYS[kind] for kind in RU_GUARD_KINDS)
    for series in [*_mode_series(windows, "train thinking", "ru_think", guarded),
                   *_mode_series(windows, "train answer", "ru_answer", guarded)]:
        base = series.points[0][1]
        fires += streak_fires("W6f", f"{series.name} Cyrillic of Russian prompts, per window", [
            (step, value < rule6.ru_factor * base and base - value >= rule6.min_ru_drop,
             f"{value:.3f}, first window {base:.3f}") for step, value in series.points[1:]])
    # W7, for each language of the chat probe.
    for role in CHAT_ROLES:
        for key in ("looped", "unclosed"):
            points = role_points(roles, role, key)
            if len(points) >= 2:
                (start, base) = points[0]
                fires += streak_fires("W7", f"val {role} {key} share", [
                    (step, value >= base + th.chat_flag_rise, f"{value:.3f}, step {start} {base:.3f}")
                    for step, value in points[1:]])
        points = role_points(roles, role, "chars")
        if len(points) >= 2 and points[0][1] > 0:
            (start, base) = points[0]
            fires += streak_fires("W7", f"val {role} mean response chars", [
                (step, not th.chat_length_low * base <= value <= th.chat_length_high * base,
                 f"{value:.0f}, step {start} {base:.0f}, allowed {th.chat_length_low:g}-{th.chat_length_high:g}x")
                for step, value in points[1:]], confirm_after=None)
    return fires


def build_rl_report(run_dir: Path, th: RLThresholds = RLThresholds(), *, limit: int | None = None,
                    index: Mapping[str, str] | None = None,
                    w1: Mapping[str, tuple[float, float, float]] = W1_DEFAULT,
                    rollouts: bool = True) -> RLReport:
    """Read the run dir once, write the status lines and evaluate W1-W8."""
    metrics = read_metrics(run_dir)
    dumps = read_dump_dir(run_dir / "rollouts") if rollouts else {}
    roles, unmatched = read_val_roles(run_dir / "val_generations", index or {})
    response_cap, cap_source = response_limit(run_dir, limit)
    by_source = validation_metrics(metrics)
    cut = train_truncated(metrics, response_cap)
    report = RLReport(str(run_dir), response_cap, cap_source, status_lines(metrics, dumps, roles, by_source, cut, th))
    fires = [*train_rules(metrics, dumps, cut, th), *val_rules(by_source, w1, th), *probe_rules(roles, dumps, th)]
    # Two confirmed Russian rules make a hard stop, at the step where the second one is confirmed.
    confirmed: dict[str, Fire] = {}
    for fire in sorted((f for f in fires if f.rule.startswith("W6") and f.level == CONFIRMED), key=lambda f: f.step):
        confirmed.setdefault(fire.rule, fire)
        if len(confirmed) == 2:
            fires.append(Fire("W6", HARD, fire.step, "two confirmed Russian rules",
                              ", ".join(f"{f.rule} at step {f.step}" for f in confirmed.values())))
            break
    report.fires = sorted(fires, key=lambda fire: (fire.step, fire.rule))
    if response_cap is None:
        report.notes.append(f"The training truncated share needs the response limit ({cap_source}); W4b is off.")
    missed = {step: count for step, count in sorted(unmatched.items()) if count}
    if index and missed:
        report.notes.append(f"Validation rows that match no frozen file, per step: {missed}. They were classified "
                            "from their template; check that --val-dir holds the run's validation files.")
    return report


def val_files(val_dir: str | None, items: Sequence[str]) -> dict[str, Path]:
    """The frozen validation files by role: those of --val-dir that exist, then each --val-file ROLE=PATH."""
    files = {}
    if val_dir:
        files = {role: Path(val_dir) / name for role, name in VAL_FILES.items() if (Path(val_dir) / name).is_file()}
    for item in items:
        role, _, path = item.partition("=")
        if role not in VAL_FILES or not path:
            raise SystemExit(f"--val-file wants ROLE=PATH with ROLE one of {', '.join(VAL_FILES)}, got {item!r}")
        files[role] = Path(path)
    return files


def w1_sources(items: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    """W1's sources: the defaults and each --w1 SOURCE:TWICE:ONCE[:HARD]."""
    found = dict(W1_DEFAULT)
    for item in items:
        source, *numbers = item.split(":")
        if len(numbers) not in (2, 3):
            raise SystemExit(f"--w1 wants SOURCE:TWICE:ONCE[:HARD], got {item!r}")
        values = [float(number) for number in numbers]
        found[source] = (values[0], values[1], values[2] if len(values) == 3 else 0.08)
    return found


def run_rl(args: argparse.Namespace, run_dir: Path) -> int:
    """The RL mode: check once, or follow the run and print only what is new."""
    th = RLThresholds(kl_hard=args.kl_hard, window=args.window)
    files = val_files(args.val_dir, args.val_file)
    index = load_val_index(files)
    w1 = w1_sources(args.w1)
    shown: set[tuple[int, int]] = set()
    fired: set[tuple[str, str, str]] = set()
    first = True
    try:
        while True:
            report = build_rl_report(run_dir, th, limit=args.max_response_length, index=index, w1=w1,
                                     rollouts=not args.no_rollouts)
            hard = report.first((HARD,))
            if args.json:
                print(json.dumps(report.as_dict(), ensure_ascii=False), flush=True)
            else:
                if first:
                    print(f"rl watch {run_dir}: response limit {report.limit} ({report.limit_source}); "
                          f"val files: {', '.join(f'{role}={path}' for role, path in files.items()) or 'none'}")
                train = [line for line in report.lines if line[1] == 0]
                keep = {line[:2] for line in train[-args.last:]} if first and args.last > 0 else None
                for step, order, text in report.lines:
                    if (step, order) not in shown and (keep is None or order == 1 or (step, order) in keep):
                        print(text)
                    shown.add((step, order))
                for fire in report.fires:
                    if (fire.rule, fire.level, fire.series) not in fired:
                        print(fire.line())
                        fired.add((fire.rule, fire.level, fire.series))
                for note in report.notes if first else ():
                    print(f"note: {note}")
                if not args.follow:
                    notify = report.first((NOTIFY, CONFIRMED, HARD))
                    print(f"first notify: {f'step {notify.step} ({notify.rule})' if notify else 'none'}; "
                          f"first hard stop: {f'step {hard.step} ({hard.rule})' if hard else 'none'}")
                sys.stdout.flush()
            first = False
            # A hard stop also ends the follow loop. The run should stop now, and the exit code is how a
            # wrapper learns that.
            if not args.follow or hard is not None:
                return EXIT_HARD if hard else EXIT_STOP if report.fires else EXIT_OK
            time.sleep(args.follow)
    except KeyboardInterrupt:
        return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/verl/watch.py",
        description="Report an on-policy distillation run and check its stop rules 1-6, or, with --rl, an RL "
                    "run and the rules W1-W8 (the module docstring explains each rule). Exit code 3 means that a "
                    "rule fired, and 4 that an RL hard-stop rule fired.",
    )
    parser.add_argument("run_dir", help="The run dir, $ORBIT_VERL_RUNS/runs/<run_name>.")
    parser.add_argument("--follow", type=float, default=0.0, metavar="SECONDS",
                        help="Check again every SECONDS until a rule fires (exit 3), with --rl until a hard stop "
                             "fires (exit 4), or until Ctrl-C (exit 0).")
    parser.add_argument("--json", action="store_true", help="Print the report as one JSON object per check.")
    parser.add_argument("--last", type=int, default=20, help="Training steps to print; 0 prints all (default 20).")
    parser.add_argument("--max-response-length", type=int, default=None,
                        help="The run's response limit in tokens. Default: data.max_response_length of config.yaml.")
    parser.add_argument("--kl-key", default=None,
                        help=f"The metrics.jsonl key of the teacher KL. Default: the first of {', '.join(KL_KEYS)}.")
    parser.add_argument("--window", type=int, default=Thresholds.window,
                        help="Training steps per check of the rollout-based rules (default 5).")
    parser.add_argument("--kl-smooth", type=int, default=Thresholds.kl_smooth,
                        help="Steps in the moving average of the teacher KL rule; 1 uses the raw values (default 5).")
    rl = parser.add_argument_group("RL mode", "Rules W1-W8 for an RL run; see the module docstring.")
    rl.add_argument("--rl", action="store_true", help="Watch an RL run instead of a distillation run.")
    rl.add_argument("--val-dir", default=None,
                    help=f"The dir of the frozen validation files, found by their names: {', '.join(VAL_FILES.values())}.")
    rl.add_argument("--val-file", action="append", default=[], metavar="ROLE=PATH",
                    help=f"One frozen validation file; ROLE is one of {', '.join(VAL_FILES)}. Repeatable.")
    rl.add_argument("--w1", action="append", default=[], metavar="SOURCE:TWICE:ONCE[:HARD]",
                    help="Another W1 source and its drops, for example an LCB-200 set. Repeatable.")
    rl.add_argument("--kl-hard", type=float, default=RLThresholds.kl_hard,
                    help="W3's hard stop on the 5-step mean of actor/kl_loss; 0 makes W3 notify only (default 0.06).")
    rl.add_argument("--no-rollouts", action="store_true",
                    help="Skip the training dumps, which are slow to read; W4a, W6e and W6f are then off.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_dir():
        print(f"watch: {run_dir} is not a directory", file=sys.stderr)
        return EXIT_USAGE
    if args.window < 1 or args.kl_smooth < 1:
        print("watch: --window and --kl-smooth must be at least 1", file=sys.stderr)
        return EXIT_USAGE
    if args.rl:
        return run_rl(args, run_dir)
    thresholds = Thresholds(window=args.window, kl_smooth=args.kl_smooth)
    try:
        while True:
            report = build_report(run_dir, thresholds, limit=args.max_response_length, kl_key=args.kl_key)
            if args.json:
                print(json.dumps(report.as_dict(), ensure_ascii=False), flush=True)
            else:
                if args.follow:
                    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                print(render(report, args.last), flush=True)
            if report.fired:
                return EXIT_STOP
            if not args.follow:
                return EXIT_OK
            time.sleep(args.follow)
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
