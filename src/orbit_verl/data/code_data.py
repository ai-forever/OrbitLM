"""Build the code-RL dataset: stdin problems with tests, written as train.jsonl and val.jsonl.

CLI::

    export PYTHONPATH=<repo>/src
    $PY -m orbit_verl.data.code_data build --out-dir DIR \\
        [--codeforces DIR] [--code-contests DIR] [--taco DIR] [--primeintellect DIR] [--lcb DIR] \\
        [--val-from 2025-02-01] [--val-size 200] [--max-cases 30] [--min-cases 3] \\
        [--max-test-bytes 4000000] [--validate] [--workers N] [--limit-per-source N] [--seed 0] [--force]

What it writes into ``--out-dir``:

- ``train.jsonl`` and ``val.jsonl``, one problem per line, with these fields:
  ``instruction`` is the problem statement as the model sees it. It carries no answer-format
  instruction, because the prompt template adds one.
  ``response`` is the ground truth as a JSON string:
  ``{"kind": "stdin", "tests": [{"input": ..., "output": ...}], "judge_imports": true, "float_tolerance": 1e-06}``.
  ``source``, ``id``, ``difficulty``, ``n_cases`` and ``validated`` are metadata for people reading the data.
  ``id`` is ``<source>/<row number in the HF dataset>``, or the judge's id for open-r1 and LCB (``codeforces/2063/A``,
  ``lcb/abc390_c``). ``difficulty`` is the Codeforces rating when the source knows it, and otherwise the
  source's own label, such as TACO's ``MEDIUM``, a CodeContests difficulty name or LCB's ``easy``.
  Both files are local sources for ``prepare_data`` (``--src DIR/train.jsonl --val-src DIR/val.jsonl``). It
  reads ``instruction`` and ``response`` by default and copies the response string unchanged into
  ``reward_model.ground_truth``.
- ``report.json``: counts per source and per drop reason, the settings and the timings. It is written last,
  and ``--force`` replaces only a directory that holds one.

The train sources are Hugging Face cache directories. Their ``*-train-*.arrow`` files are read directly with
pyarrow, so neither the ``datasets`` library nor the network is needed. Any directory above the shard files
works, for example ``<hf_cache>/datasets/BAAI___taco``, as long as it holds only one cached copy of the data.

- ``--codeforces``: open-r1/codeforces. It has the complete official tests but no reference solutions.
- ``--code-contests``: deepmind/code_contests, with public, private and generated tests and Python 3 solutions.
- ``--taco``: BAAI/TACO, with its tests in ``input_output`` and Python solutions.
- ``--primeintellect``: the PrimeIntellect part of agentica-org/DeepCoder-Preview-Dataset, a directory of
  ``train-*.parquet`` files. Its tests were generated and checked against solutions by its authors, each problem
  has a Python reference, and the statements carry a fixed wrapper ("Solve the following coding problem ...")
  that is cut off, because our prompt adds its own instruction.

``--lcb`` is a LiveCodeBench ``code_generation_lite`` directory (``test*.jsonl``). It is an evaluation set, and
nothing in it is trained on. Every LCB problem is used to decontaminate the train problems. The stdin problems
(every platform except LeetCode) whose contest is on or after ``--val-from`` become ``val.jsonl``: a sample of
``--val-size`` of them, chosen with ``--seed``. Without ``--lcb``, ``val.jsonl`` holds ``--val-size`` problems
taken out of the train problems, and ``report.json`` says so.

What happens to every problem, in order. Each drop is counted in ``report.json`` under the name in brackets.

1. Filters. The reward runs a program on stdin and compares its stdout with one expected output, so only
   problems that such a judge can grade stay in.
   - [not_stdin] Function-call problems (a ``fn_name`` or starter code) cannot be graded on stdin.
   - [file_io] Problems that read and write named files cannot be graded on stdin either.
   - [interactive] An interactive problem needs a judge that answers the program, and the reward has none.
   - [needs_checker] A problem with a special checker accepts several outputs, so one expected output
     would fail correct programs.
   - [incomplete_tests] An open-r1 problem without its complete official tests can pass wrong programs.
   - [has_images] A statement that refers to a picture may leave out what the model needs to solve it.
   - [empty_statement], [bad_tests], [non_string_tests] The row cannot be turned into a statement and tests.
2. Tests. Example and public cases come first, then the others in source order. A case whose input repeats an
   earlier one is skipped, and line ends are normalised to LF.
   - [conflicting_tests] Two cases with the same input and different outputs show that the answer is not
     unique.
   - At most ``--max-cases`` cases are kept, because during RL every case is one sandbox run.
   - Trailing cases are dropped until the tests take at most ``--max-test-bytes`` as JSON, because the tests
     travel inside every dataset row and every reward call. [tests_too_large] counts the problems that this
     cap leaves with too few cases.
   - [too_few_cases] With fewer than ``--min-cases`` cases, a wrong program passes too easily.
   - [constant_output] When every case expects the same output, a program that only prints it passes them all.
3. Deduplication. A problem that is in several sources would otherwise be trained on more often than the rest.
   - [duplicate_id] Problems that share a Codeforces id (contest and index, open-r1 aliases included) or an
     AtCoder task id keep one copy. open-r1 wins over CodeContests, and CodeContests wins over TACO;
     open-r1 has the complete official tests.
   - [duplicate_statement] Problems whose statements are equal after lowercasing and collapsing whitespace
     keep one copy, with the same priority.
4. Decontamination against LCB, so that the evaluation measures problems the model has not trained on.
   - [lcb_id] The problem has the Codeforces or AtCoder id of an LCB problem.
   - [lcb_ngram] At least half of the statement's word 13-grams occur in some LCB statement.
5. ``--validate`` (optional). Up to three Python 3 reference solutions of a TACO, CodeContests or
   PrimeIntellect problem, the shortest first, run through ``code_sandbox.execute`` with exactly the reward's
   settings. The problem stays only if one of them passes. This also removes problems whose answers are not
   unique, and broken tests.
   - [no_python_reference] The problem has no Python 3 solution to check it with.
   - [validation_failed] No reference passed. ``report.json`` lists why they failed.
   open-r1 problems have no solutions and cannot be checked. They stay, with ``validated: false``, because
   their tests are the complete official tests and they need no checker. Without ``--validate`` every problem
   has ``validated: false``.
6. The problems are shuffled with ``--seed`` and written through a staging directory, so an interrupted run
   leaves no half-written dataset under the target name.

The ground truth asks the reward for ``judge_imports`` and a float tolerance of 1e-6 for every source. Both
were measured on LCB with the sandbox. LCB runs stdin programs after its import prelude and recursion limit,
and correct answers to AtCoder-style fractional problems differ from the reference outputs in the last digits.
Using the same settings everywhere means that the reward never needs to know where a problem came from.

``--validate`` is expensive. One sandbox run costs about 40 ms of CPU, so checking costs roughly
problems x cases x 40 ms for every reference tried: 20,000 problems with 20 cases each take about 4.5 CPU
hours. Run it on a machine with many CPUs and with many ``--workers``.

pyarrow is imported only when arrow files are read. Everything else is standard library plus
``orbit_verl.code_sandbox``, which is standard library too.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import dataclasses
import glob
import io
import json
import os
import pickle
import random
import re
import shutil
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from orbit_verl import common
from orbit_verl.code_sandbox import Limits, available_cpus, execute, extract_code, outputs_match

TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
REPORT_FILE = "report.json"
REPORT_KIND = "orbit_verl_code_data"

# The train sources in priority order: when two sources hold the same problem, the earlier one keeps it.
CODEFORCES = "codeforces"
CODE_CONTESTS = "code_contests"
TACO = "taco"
PRIMEINTELLECT = "primeintellect"
LCB = "lcb"
# The order is also the priority of deduplication: of two copies of a problem, the earlier source's stays.
TRAIN_SOURCES = (CODEFORCES, CODE_CONTESTS, TACO, PRIMEINTELLECT)
_PRIORITY = {name: rank for rank, name in enumerate(TRAIN_SOURCES)}

# Many AtCoder tasks accept a fractional answer within 1e-6 or tighter, and LCB's reference outputs print more
# digits than a correct program usually does. Exact comparison would fail those programs, so stdin problems allow
# this much absolute or relative difference on fractional tokens.
LCB_FLOAT_TOLERANCE = 1e-6

# The settings the reward uses for every problem of this dataset. Validation runs the references with the
# same values, so a problem that passes here can also be passed by a policy program during RL.
JUDGE_IMPORTS = True
FLOAT_TOLERANCE = LCB_FLOAT_TOLERANCE

# Validation tries at most this many reference solutions per problem, the shortest first. Every reference
# that fails costs a full run over the kept tests, so trying more would mostly add cost on problems whose
# tests are broken.
MAX_REFERENCES = 3

# A train statement counts as an LCB copy when at least this share of its word n-grams occurs in LCB.
NGRAM = 13
NGRAM_SHARE = 0.5

# Validation prints a progress line after every this many problems.
PROGRESS_EVERY = 500

# Rows are turned into Python objects this many at a time. A record batch of a Hugging Face arrow file can
# hold a thousand rows, and CodeContests rows carry up to hundreds of generated tests each, so converting a
# whole batch at once could need gigabytes.
_ROWS_PER_CHUNK = 8

# CodeContests stores the language of a solution and its difficulty as class-label ids. The names below are
# the ClassLabel names in the dataset's own arrow metadata (deepmind/code_contests).
CC_PYTHON3 = 3
CC_DIFFICULTIES = ("UNKNOWN_DIFFICULTY", "EASY", "MEDIUM", "HARD", "HARDER", "HARDEST", "EXTERNAL",
                   *"ABCDEFGHIJKLMNOPQRSTUV")

DEFAULT_VAL_FROM = "2025-02-01"
DEFAULT_VAL_SIZE = 200
DEFAULT_MAX_CASES = 30
DEFAULT_MIN_CASES = 3
DEFAULT_MAX_TEST_BYTES = 4_000_000


@dataclass(frozen=True)
class CaseCaps:
    """How many test cases a problem keeps, how large they may be, and how few are too few."""

    max_cases: int = DEFAULT_MAX_CASES
    min_cases: int = DEFAULT_MIN_CASES
    max_test_bytes: int = DEFAULT_MAX_TEST_BYTES


@dataclass
class Problem:
    """One problem on its way through the builder.

    ``cf_ids`` and ``atcoder_id`` are the judge ids that deduplication and decontamination compare.
    ``solutions`` holds the reference programs that validation may run, the shortest first.
    ``order`` is the row number in the source, which keeps every step deterministic.
    """

    source: str
    id: str
    statement: str
    difficulty: str
    tests: list[dict[str, str]]
    solutions: list[str] = field(default_factory=list)
    cf_ids: tuple[str, ...] = ()
    atcoder_id: str | None = None
    order: int = 0
    validated: bool = False


# ----------------------------------------------------------------------------------------------------------
# Statements, tests and ids
# ----------------------------------------------------------------------------------------------------------

def _lf(text: str) -> str:
    return text.replace("\r\n", "\n")


def normalise_statement(text: Any) -> str:
    """The statement as the model sees it: LF line ends and no surrounding whitespace."""
    return _lf(str(text or "")).strip()


def statement_key(text: str) -> str:
    """The key under which two statements count as the same: lowercase, with all whitespace collapsed."""
    return " ".join(text.lower().split())


_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(")


def has_images(statement: str) -> bool:
    """True when the statement points at a picture: CodeContests' ``<image>`` marker or a Markdown image."""
    return "<image>" in statement.lower() or _MARKDOWN_IMAGE.search(statement) is not None


def _input_key(stdin: str) -> str:
    # Two inputs that differ only in trailing spaces or trailing blank lines give a program that reads lines or
    # tokens the same data, so they count as one case.
    lines = [line.rstrip() for line in stdin.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _case_bytes(case: dict[str, str]) -> int:
    return len(json.dumps(case, ensure_ascii=False).encode("utf-8"))


def tests_json_bytes(tests: list[dict[str, str]]) -> int:
    """The size of ``json.dumps(tests, ensure_ascii=False)`` in UTF-8 bytes, which ``--max-test-bytes`` caps."""
    return len(json.dumps(tests, ensure_ascii=False).encode("utf-8"))


def select_tests(groups: Iterable[Iterable[Any]], caps: CaseCaps) -> list[dict[str, str]] | str:
    """The test cases a problem keeps, or the reason it is dropped.

    ``groups`` lists the case groups in priority order, such as the examples and then the official tests.
    Each case is a dict with ``input`` and ``output``. Step 2 of the module docstring describes the rules.

    Reading stops once ``max_cases`` distinct cases are found. A CodeContests problem can carry hundreds of
    generated cases, and the ones after the cap would never reach the reward. A conflict among those later
    cases therefore goes unnoticed, which is harmless because the reward never sees them.
    """
    seen: dict[str, str] = {}
    cases: list[dict[str, str]] = []
    for group in groups:
        for case in group:
            if not isinstance(case, dict):
                return "bad_tests"
            stdin, expected = case.get("input"), case.get("output")
            if not isinstance(stdin, str) or not isinstance(expected, str):
                return "non_string_tests"
            stdin, expected = _lf(stdin), _lf(expected)
            key = _input_key(stdin)
            if key in seen:
                # The reward compares outputs with outputs_match, so two expected outputs that it would treat
                # as equal are not a conflict.
                if not outputs_match(expected, seen[key], FLOAT_TOLERANCE):
                    return "conflicting_tests"
                continue
            seen[key] = expected
            cases.append({"input": stdin, "output": expected})
            if len(cases) >= caps.max_cases:
                break
        if len(cases) >= caps.max_cases:
            break

    enough_before_size_cap = len(cases) >= caps.min_cases
    # The JSON of a list is "[" + the items joined by ", " + "]", so the total is computed from the item sizes
    # once, instead of dumping the whole list again after every removal.
    sizes = [_case_bytes(case) for case in cases]
    total = 2 + sum(sizes) + 2 * max(0, len(cases) - 1)
    while cases and total > caps.max_test_bytes:
        total -= sizes.pop() + (2 if len(cases) > 1 else 0)
        cases.pop()
    if len(cases) < caps.min_cases:
        return "tests_too_large" if enough_before_size_cap else "too_few_cases"
    if len(cases) > 1 and len({" ".join(case["output"].split()) for case in cases}) == 1:
        return "constant_output"
    return cases


def ground_truth(tests: list[dict[str, str]]) -> str:
    """The ``response`` field: the JSON string that the reward parses."""
    return json.dumps(
        {"kind": "stdin", "tests": tests, "judge_imports": JUDGE_IMPORTS, "float_tolerance": FLOAT_TOLERANCE},
        ensure_ascii=False,
    )


_CF_ID = re.compile(r"(\d+)\s*[/_]\s*([A-Za-z]\d*)")
_CF_URLS = (
    re.compile(r"codeforces\.com/problemset/problem/(\d+)/([A-Za-z]\d*)(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"codeforces\.com/contest/(\d+)/problem/([A-Za-z]\d*)(?![A-Za-z0-9])", re.IGNORECASE),
)
_ATCODER_URL = re.compile(r"atcoder\.jp/contests/[^/\s]+/tasks/([A-Za-z0-9_]+)", re.IGNORECASE)


def codeforces_id(contest: Any, index: Any) -> str | None:
    """``"1873/A"`` for contest 1873 and index ``a``; None when either part is missing."""
    contest_text, index_text = str(contest or "").strip(), str(index or "").strip().upper()
    if not contest_text.isdigit() or int(contest_text) <= 0 or not re.fullmatch(r"[A-Z]\d*", index_text):
        return None
    return f"{int(contest_text)}/{index_text}"


def codeforces_id_from_text(text: Any) -> str | None:
    """The Codeforces id in ``"2063/A"`` (open-r1) or ``"1873_A"`` (LCB question ids)."""
    match = _CF_ID.fullmatch(str(text or "").strip())
    return codeforces_id(match.group(1), match.group(2)) if match else None


def ids_from_url(url: Any) -> tuple[tuple[str, ...], str | None]:
    """The Codeforces ids and the AtCoder task id that a TACO problem URL names."""
    text = str(url or "")
    cf_ids = tuple(
        dict.fromkeys(
            found for pattern in _CF_URLS for match in pattern.finditer(text)
            if (found := codeforces_id(match.group(1), match.group(2)))
        )
    )
    atcoder = _ATCODER_URL.search(text)
    return cf_ids, atcoder.group(1).lower() if atcoder else None


def lcb_eval_ids(question_id: str, platform: str) -> tuple[str | None, str | None]:
    """The (Codeforces id, AtCoder task id) of an LCB problem; LeetCode problems have neither.

    LCB writes a Codeforces problem as ``1873_A`` and an AtCoder task by its task id, such as ``abc301_e``.
    """
    if platform == "codeforces":
        return codeforces_id_from_text(question_id), None
    if platform == "atcoder":
        return None, str(question_id).strip().lower() or None
    return None, None


def _references(programs: Iterable[Any]) -> list[str]:
    """The reference programs that validation runs: distinct, not empty, the shortest first, at most three.

    Short references are tried first because they finish their runs sooner and fail less often for reasons
    that have nothing to do with the tests, such as an unusual library.
    """
    unique = list(dict.fromkeys(p for p in programs if isinstance(p, str) and p.strip()))
    return sorted(unique, key=len)[:MAX_REFERENCES]


def _as_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


# ----------------------------------------------------------------------------------------------------------
# One source row -> Problem, or the reason it is dropped
# ----------------------------------------------------------------------------------------------------------

def taco_problem(row: dict[str, Any], number: int, caps: CaseCaps, with_solutions: bool) -> Problem | str:
    """A BAAI/TACO row. Only stdin problems stay: no ``fn_name``, no starter code, string inputs and outputs."""
    statement = normalise_statement(row.get("question"))
    if not statement:
        return "empty_statement"
    if str(row.get("starter_code") or "").strip():
        return "not_stdin"
    try:
        spec = json.loads(row.get("input_output") or "")
    except ValueError:
        return "bad_tests"
    if not isinstance(spec, dict):
        return "bad_tests"
    if spec.get("fn_name"):
        return "not_stdin"
    if _as_int(row.get("picture_num")) > 0 or has_images(statement):
        return "has_images"
    inputs, outputs = spec.get("inputs"), spec.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        return "bad_tests"
    # TACO stores the arguments of function-call problems as lists. Such a value anywhere in the tests marks a
    # problem that is not really read from stdin, even when the first cases happen to be strings.
    if not all(isinstance(value, str) for value in inputs + outputs):
        return "non_string_tests"
    tests = select_tests([[{"input": i, "output": o} for i, o in zip(inputs, outputs)]], caps)
    if isinstance(tests, str):
        return tests
    solutions: list[str] = []
    if with_solutions:
        try:
            programs = json.loads(row.get("solutions") or "[]")
        except ValueError:
            programs = []
        solutions = _references(programs if isinstance(programs, list) else [])
    cf_ids, atcoder_id = ids_from_url(row.get("url"))
    return Problem(
        source=TACO, id=f"{TACO}/{number}", statement=statement, difficulty=str(row.get("difficulty") or ""),
        tests=tests, solutions=solutions, cf_ids=cf_ids, atcoder_id=atcoder_id, order=number,
    )


# The wrapper that DeepCoder-Preview puts around every PrimeIntellect statement. Our prompt mode adds its own
# instruction, so the model would otherwise read two different instructions.
_PI_PREFIX = re.compile(r"^\s*Solve the following coding problem using the programming language python:\s*")
_PI_SUFFIX = re.compile(
    r"\s*The input will be given via stdin and the output should be printed to stdout by your code\.\s*"
    r"Now solve the problem by providing the code\.\s*$")


def primeintellect_problem(row: dict[str, Any], number: int, caps: CaseCaps, with_solutions: bool) -> Problem | str:
    """A PrimeIntellect row of DeepCoder-Preview. Only problems whose every test is a stdin/stdout case stay."""
    statement = normalise_statement(_PI_SUFFIX.sub("", _PI_PREFIX.sub("", str(row.get("problem") or ""))))
    if not statement:
        return "empty_statement"
    try:
        cases = json.loads(row.get("tests") or "")
    except ValueError:
        return "bad_tests"
    if not isinstance(cases, list) or not cases or not all(isinstance(case, dict) for case in cases):
        return "bad_tests"
    if any(case.get("type", "stdin_stdout") != "stdin_stdout" for case in cases):
        return "not_stdin"
    if has_images(statement):
        return "has_images"
    if not all(isinstance(case.get("input"), str) and isinstance(case.get("output"), str) for case in cases):
        return "non_string_tests"
    tests = select_tests([[{"input": case["input"], "output": case["output"]} for case in cases]], caps)
    if isinstance(tests, str):
        return tests
    solutions: list[str] = []
    if with_solutions:
        # The references are markdown answers with the program in a fenced block, so the code is cut out the
        # same way the reward cuts it out of a model's answer.
        solutions = _references(extract_code(str(answer)) for answer in (row.get("solutions") or []))
    return Problem(
        source=PRIMEINTELLECT, id=f"{PRIMEINTELLECT}/{number}", statement=statement, difficulty="",
        tests=tests, solutions=solutions, order=number,
    )


def _struct_cases(value: Any) -> list[dict[str, Any]]:
    """CodeContests tests: a struct of two lists, ``{"input": [...], "output": [...]}``, as a list of cases.

    A list of ``{"input", "output"}`` dicts is accepted too. Raises ValueError when the lists differ in length.
    """
    if not value:
        return []
    if isinstance(value, list):
        return value
    inputs, outputs = value.get("input") or [], value.get("output") or []
    if len(inputs) != len(outputs):
        raise ValueError("inputs and outputs differ in length")
    return [{"input": i, "output": o} for i, o in zip(inputs, outputs)]


def _code_contests_solutions(value: Any) -> list[str]:
    """The Python 3 programs among CodeContests ``solutions``; Python 2 (language 1) cannot run here."""
    if not value:
        return []
    if isinstance(value, list):
        pairs = [(item.get("language"), item.get("solution")) for item in value]
    else:
        pairs = list(zip(value.get("language") or [], value.get("solution") or []))
    return [program for language, program in pairs if language == CC_PYTHON3]


def code_contests_problem(
    row: dict[str, Any], number: int, caps: CaseCaps, with_solutions: bool
) -> Problem | str:
    """A deepmind/code_contests row. Tests are public, then private, then generated."""
    statement = normalise_statement(row.get("description"))
    if not statement:
        return "empty_statement"
    if str(row.get("input_file") or "").strip() or str(row.get("output_file") or "").strip():
        return "file_io"
    if has_images(statement):
        return "has_images"
    try:
        groups = [_struct_cases(row.get(name)) for name in ("public_tests", "private_tests", "generated_tests")]
    except (AttributeError, ValueError):
        return "bad_tests"
    tests = select_tests(groups, caps)
    if isinstance(tests, str):
        return tests
    cf_id = codeforces_id(row.get("cf_contest_id"), row.get("cf_index"))
    rating = _as_int(row.get("cf_rating"))
    label = row.get("difficulty")
    if rating > 0:
        difficulty = str(rating)
    elif isinstance(label, int) and 0 <= label < len(CC_DIFFICULTIES):
        difficulty = CC_DIFFICULTIES[label]
    else:
        difficulty = "" if label is None else str(label)
    solutions = _references(_code_contests_solutions(row.get("solutions"))) if with_solutions else []
    return Problem(
        source=CODE_CONTESTS, id=f"{CODE_CONTESTS}/{number}", statement=statement, difficulty=difficulty,
        tests=tests, solutions=solutions, cf_ids=(cf_id,) if cf_id else (), order=number,
    )


def codeforces_statement(row: dict[str, Any]) -> str:
    """The statement of an open-r1/codeforces row, laid out like the Codeforces page without the title.

    The description is followed by the Input and Output sections, the examples and the note. A section that
    the row leaves empty is left out.
    """
    parts = [normalise_statement(row.get("description"))]
    for heading, column in (("Input", "input_format"), ("Output", "output_format")):
        text = normalise_statement(row.get(column))
        if text:
            parts.append(f"{heading}\n\n{text}")
    examples = [example for example in row.get("examples") or [] if isinstance(example, dict)]
    if examples:
        parts.append("Examples")
        for example in examples:
            stdin = _lf(str(example.get("input") or "")).strip("\n")
            expected = _lf(str(example.get("output") or "")).strip("\n")
            parts.append(f"Input\n\n{stdin}\n\nOutput\n\n{expected}")
    note = normalise_statement(row.get("note"))
    if note:
        parts.append(f"Note\n\n{note}")
    return "\n\n".join(part for part in parts if part).strip()


def codeforces_problem(row: dict[str, Any], number: int, caps: CaseCaps, with_solutions: bool) -> Problem | str:
    """An open-r1/codeforces row. It has no reference solutions, so ``with_solutions`` changes nothing."""
    if row.get("input_mode") != "stdio":
        return "not_stdin"
    if str(row.get("interaction_format") or "").strip():
        return "interactive"
    if row.get("generated_checker"):
        return "needs_checker"
    if not row.get("official_tests_complete"):
        return "incomplete_tests"
    statement = codeforces_statement(row)
    if not normalise_statement(row.get("description")):
        return "empty_statement"
    if has_images(statement):
        return "has_images"
    tests = select_tests([row.get("examples") or [], row.get("official_tests") or []], caps)
    if isinstance(tests, str):
        return tests
    problem_id = str(row.get("id") or "").strip() or str(number)
    # A problem given in both divisions of a round has one id per division. open-r1 lists the other ids as
    # aliases, and another source may hold the problem under any of them.
    ids = [codeforces_id_from_text(problem_id)] + [codeforces_id_from_text(a) for a in row.get("aliases") or []]
    rating = _as_int(row.get("rating"))
    return Problem(
        source=CODEFORCES, id=f"{CODEFORCES}/{problem_id}", statement=statement,
        difficulty=str(rating) if rating > 0 else "", tests=tests,
        cf_ids=tuple(dict.fromkeys(i for i in ids if i)), order=number,
    )


@dataclass(frozen=True)
class _SourceSpec:
    """How to read one train source: its shard file prefix, the columns to load and the row converter."""

    file_prefix: str
    columns: tuple[str, ...]
    optional: tuple[str, ...]
    convert: Callable[[dict[str, Any], int, CaseCaps, bool], Problem | str]
    has_references: bool
    # True for a directory of train-*.parquet files, as the Hugging Face hub stores DeepCoder-Preview. The other
    # sources are arrow shards from the datasets cache.
    parquet: bool = False


_SOURCE_SPECS: dict[str, _SourceSpec] = {
    CODEFORCES: _SourceSpec(
        file_prefix="codeforces",
        columns=("id", "description", "input_format", "output_format", "examples", "official_tests",
                 "official_tests_complete", "input_mode", "generated_checker", "interaction_format"),
        optional=("note", "rating", "aliases"),
        convert=codeforces_problem,
        has_references=False,
    ),
    CODE_CONTESTS: _SourceSpec(
        file_prefix="code_contests",
        columns=("description", "public_tests", "private_tests", "generated_tests", "input_file", "output_file",
                 "cf_contest_id", "cf_index"),
        optional=("difficulty", "cf_rating"),
        convert=code_contests_problem,
        has_references=True,
    ),
    TACO: _SourceSpec(
        file_prefix="taco",
        columns=("question", "starter_code", "input_output"),
        optional=("difficulty", "url", "picture_num"),
        convert=taco_problem,
        has_references=True,
    ),
    PRIMEINTELLECT: _SourceSpec(
        file_prefix="train",
        columns=("problem", "tests"),
        optional=(),
        convert=primeintellect_problem,
        has_references=True,
        parquet=True,
    ),
}


# ----------------------------------------------------------------------------------------------------------
# Reading arrow shards and LCB
# ----------------------------------------------------------------------------------------------------------

def find_arrow_files(root: str | os.PathLike, prefix: str, split: str = "train") -> list[Path]:
    """The ``<prefix>-<split>[-NNNNN-of-NNNNN].arrow`` shards under ``root``, in shard order.

    Raises ValueError when the shards sit in more than one directory. That happens when the cache holds two
    revisions or two configs of a dataset, and reading both would count every problem twice.
    """
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"{base} is not a directory.")
    name = re.compile(rf"{re.escape(prefix)}-{re.escape(split)}(-\d+-of-\d+)?\.arrow")
    files = sorted(path for path in base.rglob(f"{prefix}-{split}*.arrow") if name.fullmatch(path.name))
    if not files:
        raise FileNotFoundError(f"No {prefix}-{split}-*.arrow files under {base}.")
    dirs = sorted({str(path.parent) for path in files})
    if len(dirs) > 1:
        raise ValueError(
            f"{prefix} shards are in {len(dirs)} directories under {base}: {dirs}. Pass the one directory "
            "to read, so that no problem is read twice."
        )
    return files


def find_parquet_files(root: str | os.PathLike, split: str = "train") -> list[Path]:
    """The ``<split>-NNNNN-of-NNNNN.parquet`` files directly in ``root``, in shard order."""
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"{base} is not a directory.")
    name = re.compile(rf"{re.escape(split)}-\d+-of-\d+\.parquet")
    files = sorted(path for path in base.iterdir() if name.fullmatch(path.name))
    if not files:
        raise FileNotFoundError(f"No {split}-*-of-*.parquet files in {base}.")
    return files


def iter_parquet_rows(path: str | os.PathLike, columns: Iterable[str], optional: Iterable[str] = ()):
    """Yields the rows of one parquet file as dicts, with the same column rules as ``iter_arrow_rows``."""
    import pyarrow.parquet as pq

    columns, optional = tuple(columns), tuple(optional)
    handle = pq.ParquetFile(str(path))
    names = list(handle.schema_arrow.names)
    missing = [name for name in columns if name not in names]
    if missing:
        raise ValueError(f"{path} has no column {missing}; its columns are {names}.")
    wanted = [name for name in dict.fromkeys(columns + optional) if name in names]
    absent = dict.fromkeys(name for name in optional if name not in names)
    for batch in handle.iter_batches(columns=wanted, batch_size=_ROWS_PER_CHUNK):
        for record in batch.to_pylist():
            record.update(absent)
            yield record


def _open_arrow(path: Path):
    """Opens a Hugging Face arrow file, which is normally an Arrow stream, and as a fallback an Arrow file.

    Returns the memory map and the reader. The caller closes the memory map.
    """
    import pyarrow as pa

    errors = []
    for opener in (pa.ipc.open_stream, pa.ipc.open_file):
        source = pa.memory_map(str(path), "r")
        try:
            return source, opener(source)
        except (pa.ArrowException, ValueError, OSError) as exc:
            source.close()
            errors.append(f"{opener.__name__}: {exc}")
    raise ValueError(f"{path} is neither an Arrow stream nor an Arrow file ({'; '.join(errors)}).")


def _record_batches(reader) -> Iterator[Any]:
    # A file reader gives random access to its batches, and a stream reader only iterates over them.
    if hasattr(reader, "num_record_batches") and hasattr(reader, "get_batch"):
        for index in range(reader.num_record_batches):
            yield reader.get_batch(index)
    else:
        yield from reader


def iter_arrow_rows(path: str | os.PathLike, columns: Iterable[str], optional: Iterable[str] = ()):
    """Yields the rows of one arrow file as dicts that hold only ``columns`` and ``optional``.

    A missing column of ``columns`` raises ValueError. A missing column of ``optional`` reads as None.
    """
    columns, optional = tuple(columns), tuple(optional)
    source, reader = _open_arrow(Path(path))
    try:
        names = list(reader.schema.names)
        missing = [name for name in columns if name not in names]
        if missing:
            raise ValueError(f"{path} has no column {missing}; its columns are {names}.")
        wanted = [name for name in dict.fromkeys(columns + optional) if name in names]
        absent = dict.fromkeys(name for name in optional if name not in names)
        for batch in _record_batches(reader):
            positions = [batch.schema.get_field_index(name) for name in wanted]
            for start in range(0, batch.num_rows, _ROWS_PER_CHUNK):
                chunk = batch.slice(start, _ROWS_PER_CHUNK)
                values = [chunk.column(position).to_pylist() for position in positions]
                for row in zip(*values):
                    record = dict(zip(wanted, row))
                    record.update(absent)
                    yield record
    finally:
        source.close()


def read_source(
    name: str,
    root: str | os.PathLike,
    caps: CaseCaps,
    *,
    with_solutions: bool,
    limit: int | None = None,
) -> tuple[list[Problem], dict[str, Any]]:
    """Reads one train source and applies steps 1 and 2. Returns the problems and the source's report entry.

    ``limit`` stops after that many rows, which is meant for quick trial runs.
    """
    spec = _SOURCE_SPECS[name]
    files = find_parquet_files(root, spec.file_prefix) if spec.parquet else find_arrow_files(root, spec.file_prefix)
    # The reference solutions are large, so they are loaded only when validation will run them.
    columns = spec.columns + (("solutions",) if with_solutions and spec.has_references else ())
    iter_rows = iter_parquet_rows if spec.parquet else iter_arrow_rows

    def rows() -> Iterator[dict[str, Any]]:
        for path in files:
            yield from iter_rows(path, columns, spec.optional)

    started = time.monotonic()
    problems: list[Problem] = []
    dropped: Counter = Counter()
    number = 0
    for number, row in enumerate(islice(rows(), limit), start=1):
        result = spec.convert(row, number - 1, caps, with_solutions)
        if isinstance(result, str):
            dropped[result] += 1
        else:
            problems.append(result)
    record = {
        "path": str(Path(root).resolve()),
        "files": [path.name for path in files],
        "directory": str(files[0].parent),
        "rows_read": number,
        "after_filters": len(problems),
        "dropped": dropped,
        "read_s": round(time.monotonic() - started, 1),
    }
    return problems, record


def iter_lcb_rows(lcb_dir: str | os.PathLike) -> Iterator[dict[str, Any]]:
    """The raw rows of every ``test*.jsonl`` in an LCB ``code_generation_lite`` directory, file by file.

    The tests are not decoded here. Decoding the private tests is slow, so ``lcb_test_groups`` decodes them only
    for the problems that may become validation problems.
    """
    files = sorted(glob.glob(os.path.join(str(lcb_dir), "test*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No test*.jsonl files in the LCB directory {lcb_dir}.")
    for path in files:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


class _NoGlobalsUnpickler(pickle.Unpickler):
    """Loads LCB's private tests, which are stored as a pickled JSON string.

    Refusing every global means the pickle can only build plain values such as str, list and dict. Loading it
    therefore cannot call any function, whatever the file contains.
    """

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name} in LCB test data")


def _lcb_private_cases(blob):
    """Decode LCB's private test cases: base64 of a zlib-compressed pickle of a JSON string."""
    raw = _NoGlobalsUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(blob.encode("ascii"))))).load()
    return json.loads(raw) if isinstance(raw, str) else raw


def lcb_test_groups(row: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """The public and the private test cases of one LCB row, in that order, as LCB grades them.

    The private cases are a pickled JSON string, decoded with ``_lcb_private_cases``, whose unpickler refuses
    every global. A row that stores them as plain JSON is read as JSON.
    """
    public = json.loads(row.get("public_test_cases") or "[]")
    blob = row.get("private_test_cases") or ""
    private = json.loads(blob) if blob.lstrip().startswith("[") else (_lcb_private_cases(blob) if blob else [])
    return [public, private]


@dataclass
class EvalIndex:
    """What the train problems are decontaminated against: LCB's judge ids and the hashes of its 13-grams.

    The n-grams are kept as Python string hashes rather than as word tuples, which keeps the set small. The
    hashes are salted per process, so the set is never saved and is only used in the process that built it.
    """

    cf_ids: set[str] = field(default_factory=set)
    atcoder_ids: set[str] = field(default_factory=set)
    ngrams: set[int] = field(default_factory=set)
    problems: int = 0

    def add(self, question_id: str, platform: str, statement: str) -> None:
        cf_id, atcoder_id = lcb_eval_ids(question_id, platform)
        if cf_id:
            self.cf_ids.add(cf_id)
        if atcoder_id:
            self.atcoder_ids.add(atcoder_id)
        self.ngrams.update(ngram_hashes(statement))
        self.problems += 1


_WORD = re.compile(r"\w+")


def ngram_hashes(text: str, n: int = NGRAM) -> list[int]:
    """The hashes of the lowercase word n-grams of a text, in order; empty for a text shorter than n words."""
    words = _WORD.findall(text.lower())
    return [hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)]


def ngram_share(text: str, ngrams: set[int], n: int = NGRAM) -> float:
    """The share of the text's word n-grams that occur in ``ngrams``; 0 for a text shorter than n words."""
    grams = ngram_hashes(text, n)
    return sum(gram in ngrams for gram in grams) / len(grams) if grams else 0.0


def read_lcb(
    lcb_dir: str | os.PathLike, caps: CaseCaps, val_from: str
) -> tuple[EvalIndex, list[Problem], dict[str, Any]]:
    """Reads LCB once. Returns the decontamination index, the validation candidates and the report entry.

    A candidate is a stdin problem (platform other than LeetCode) whose contest date is on or after
    ``val_from``. Its tests go through the same ``select_tests`` caps as the train problems, so a validation
    reward costs what a train reward costs.
    """
    started = time.monotonic()
    index = EvalIndex()
    candidates: list[Problem] = []
    dropped: Counter = Counter()
    platforms: Counter = Counter()
    eligible = 0
    for number, row in enumerate(iter_lcb_rows(lcb_dir)):
        question_id, platform = str(row.get("question_id") or ""), str(row.get("platform") or "")
        statement = normalise_statement(row.get("question_content"))
        index.add(question_id, platform, statement)
        platforms[platform] += 1
        # ISO dates such as "2025-03-01T00:00:00" compare correctly as strings.
        if platform == "leetcode" or str(row.get("contest_date") or "")[:10] < val_from:
            continue
        eligible += 1
        if not statement:
            dropped["empty_statement"] += 1
            continue
        try:
            groups = lcb_test_groups(row)
        except (ValueError, TypeError, AttributeError, zlib.error, binascii.Error, pickle.UnpicklingError):
            dropped["bad_tests"] += 1
            continue
        tests = select_tests(groups, caps)
        if isinstance(tests, str):
            dropped[tests] += 1
            continue
        candidates.append(Problem(
            source=LCB, id=f"{LCB}/{question_id}", statement=statement, difficulty=str(row.get("difficulty") or ""),
            tests=tests, order=number,
        ))
    record = {
        "path": str(Path(lcb_dir).resolve()),
        "files": [os.path.basename(p) for p in sorted(glob.glob(os.path.join(str(lcb_dir), "test*.jsonl")))],
        "problems": index.problems,
        "platforms": platforms,
        "codeforces_ids": len(index.cf_ids),
        "atcoder_ids": len(index.atcoder_ids),
        "ngrams": len(index.ngrams),
        "val_candidates": eligible,
        "val_dropped": dropped,
        "read_s": round(time.monotonic() - started, 1),
    }
    return index, candidates, record


# ----------------------------------------------------------------------------------------------------------
# Deduplication and decontamination
# ----------------------------------------------------------------------------------------------------------

def _priority_order(problems: Iterable[Problem]) -> list[Problem]:
    return sorted(problems, key=lambda problem: (_PRIORITY.get(problem.source, len(_PRIORITY)), problem.order))


def dedup_by_id(problems: Iterable[Problem]) -> tuple[list[Problem], list[Problem]]:
    """Keeps one problem per Codeforces or AtCoder id, by source priority. Returns (kept, dropped)."""
    seen: set[tuple[str, str]] = set()
    kept, dropped = [], []
    for problem in _priority_order(problems):
        keys = [("cf", cf_id) for cf_id in problem.cf_ids]
        if problem.atcoder_id:
            keys.append(("atcoder", problem.atcoder_id))
        if any(key in seen for key in keys):
            dropped.append(problem)
            continue
        seen.update(keys)
        kept.append(problem)
    return kept, dropped


def dedup_by_statement(problems: Iterable[Problem]) -> tuple[list[Problem], list[Problem]]:
    """Keeps one problem per normalised statement, by source priority. Returns (kept, dropped)."""
    seen: set[str] = set()
    kept, dropped = [], []
    for problem in _priority_order(problems):
        key = statement_key(problem.statement)
        if key in seen:
            dropped.append(problem)
            continue
        seen.add(key)
        kept.append(problem)
    return kept, dropped


def decontaminate(problems: Iterable[Problem], index: EvalIndex) -> tuple[list[Problem], list[tuple[Problem, str]]]:
    """Drops the problems that LCB also has. Returns the kept problems and (problem, reason) for the rest."""
    kept: list[Problem] = []
    dropped: list[tuple[Problem, str]] = []
    for problem in problems:
        if any(cf_id in index.cf_ids for cf_id in problem.cf_ids) or (
            problem.atcoder_id is not None and problem.atcoder_id in index.atcoder_ids
        ):
            dropped.append((problem, "lcb_id"))
        elif index.ngrams and ngram_share(problem.statement, index.ngrams) >= NGRAM_SHARE:
            dropped.append((problem, "lcb_ngram"))
        else:
            kept.append(problem)
    return kept, dropped


# ----------------------------------------------------------------------------------------------------------
# Validation with reference solutions
# ----------------------------------------------------------------------------------------------------------

RunFn = Callable[..., dict]


def validate_problem(problem: Problem, run: RunFn = execute) -> dict[str, Any]:
    """Runs the problem's references, the shortest first, until one passes the kept tests.

    ``run`` has the signature of ``code_sandbox.execute`` and is called with the reward's settings. Returns
    ``passed``, ``reason`` ("passed", "no_python_reference" or the last failure), the failure ``reasons`` of
    every reference tried, ``attempts``, ``cpu_s`` and the ``detail`` of the last failure.
    """
    if not problem.solutions:
        return {"passed": False, "reason": "no_python_reference", "reasons": [], "attempts": 0, "cpu_s": 0.0,
                "detail": ""}
    reasons: list[str] = []
    detail = ""
    cpu = 0.0
    for attempt, program in enumerate(problem.solutions[:MAX_REFERENCES], start=1):
        try:
            result = run(program, problem.tests, kind="stdin", judge_imports=JUDGE_IMPORTS,
                         float_tolerance=FLOAT_TOLERANCE)
        except Exception as exc:  # noqa: BLE001 - one broken run must not stop a build that takes hours
            result = {"passed": False, "reason": "harness_exception", "detail": repr(exc), "cpu_time": 0.0}
        cpu += float(result.get("cpu_time") or 0.0)
        if result.get("passed"):
            return {"passed": True, "reason": "passed", "reasons": reasons, "attempts": attempt, "cpu_s": cpu,
                    "detail": ""}
        reasons.append(str(result.get("reason") or "unknown"))
        detail = str(result.get("detail") or "")[:300]
    return {"passed": False, "reason": reasons[-1], "reasons": reasons, "attempts": len(reasons), "cpu_s": cpu,
            "detail": detail}


def validate_all(
    problems: list[Problem], run: RunFn, workers: int, log: Callable[[str], None]
) -> list[dict[str, Any]]:
    """Validates every problem in a thread pool and returns the results in the order of ``problems``.

    Threads are enough: ``execute`` spends nearly all its time waiting for the sandbox process.
    """
    results: list[dict[str, Any] | None] = [None] * len(problems)
    started = time.monotonic()
    passed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(validate_problem, problem, run): position for position, problem in enumerate(problems)}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[futures[future]] = result
            passed += bool(result["passed"])
            if done % PROGRESS_EVERY == 0 or done == len(problems):
                elapsed = time.monotonic() - started
                remaining = elapsed / done * (len(problems) - done)
                log(f"validated {done}/{len(problems)} problems, {passed} passed, "
                    f"{elapsed:.0f} s elapsed, about {remaining:.0f} s left")
    return [result for result in results if result is not None]


# ----------------------------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------------------------

def output_record(problem: Problem) -> dict[str, Any]:
    """One line of train.jsonl or val.jsonl."""
    return {
        "instruction": problem.statement,
        "response": ground_truth(problem.tests),
        "source": problem.source,
        "id": problem.id,
        "difficulty": problem.difficulty,
        "n_cases": len(problem.tests),
        "validated": bool(problem.validated),
    }


def _check_out_dir(out_dir: Path, force: bool) -> None:
    """Refuses to overwrite anything but a directory this tool wrote itself."""
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise FileExistsError(f"{out_dir} exists and is not a directory.")
    if not any(out_dir.iterdir()):
        return
    if not force:
        raise FileExistsError(f"{out_dir} is not empty. Pass --force to replace a dataset this tool wrote.")
    try:
        kind = json.loads((out_dir / REPORT_FILE).read_text(encoding="utf-8")).get("kind")
    except (OSError, ValueError, AttributeError):
        kind = None
    if kind != REPORT_KIND:
        raise FileExistsError(
            f"{out_dir} has no {REPORT_FILE} of kind {REPORT_KIND}. --force replaces only directories this tool "
            "wrote; remove it by hand if it is safe to."
        )


def write_dataset(out_dir: Path, train: list[Problem], val: list[Problem], report: dict[str, Any]) -> dict[str, Any]:
    """Writes train.jsonl, val.jsonl and then report.json through a staging directory. Returns the report.

    The staging directory sits next to the target and is moved into place at the end, so an interrupted run
    leaves no half-written dataset under the target name. The caller has checked the target with
    ``_check_out_dir``.
    """
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.parent / f".{out_dir.name}.staging-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        sizes = {}
        for name, problems in ((TRAIN_FILE, train), (VAL_FILE, val)):
            with open(staging / name, "w", encoding="utf-8") as stream:
                for problem in problems:
                    stream.write(json.dumps(output_record(problem), ensure_ascii=False) + "\n")
            sizes[name] = (staging / name).stat().st_size
        report = dict(report, bytes=sizes)
        written = common.write_manifest(staging / REPORT_FILE, REPORT_KIND, report)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return written


# ----------------------------------------------------------------------------------------------------------
# The whole build
# ----------------------------------------------------------------------------------------------------------

def _log(message: str) -> None:
    print(f"code_data: {message}", flush=True)


def build(
    out_dir: str | os.PathLike,
    *,
    taco: str | os.PathLike | None = None,
    code_contests: str | os.PathLike | None = None,
    codeforces: str | os.PathLike | None = None,
    primeintellect: str | os.PathLike | None = None,
    lcb: str | os.PathLike | None = None,
    val_from: str = DEFAULT_VAL_FROM,
    val_size: int = DEFAULT_VAL_SIZE,
    max_cases: int = DEFAULT_MAX_CASES,
    min_cases: int = DEFAULT_MIN_CASES,
    max_test_bytes: int = DEFAULT_MAX_TEST_BYTES,
    validate: bool = False,
    workers: int | None = None,
    limit_per_source: int | None = None,
    seed: int = 0,
    force: bool = False,
    run: RunFn = execute,
    log: Callable[[str], None] = _log,
) -> dict[str, Any]:
    """Builds ``out_dir`` as the module docstring describes and returns the written report.

    ``run`` replaces ``code_sandbox.execute`` for validation in tests. ``limit_per_source`` caps the rows read
    from each train source, not from LCB, because a smaller LCB would let contaminated problems through.
    """
    roots = {CODEFORCES: codeforces, CODE_CONTESTS: code_contests, TACO: taco, PRIMEINTELLECT: primeintellect}
    if not any(roots.values()):
        raise ValueError("Give at least one train source: --codeforces, --code-contests, --taco or --primeintellect.")
    try:
        date.fromisoformat(val_from)
    except ValueError:
        raise ValueError(f"--val-from {val_from!r} is not a date like 2025-02-01.") from None
    if val_size < 1:
        # verl builds a validation dataloader even with validation turned off, and an empty val file makes
        # that dataloader fail at start-up.
        raise ValueError("--val-size must be at least 1.")
    if min_cases < 1 or max_cases < min_cases:
        raise ValueError("--min-cases must be at least 1 and --max-cases at least --min-cases.")
    if max_test_bytes < 1:
        raise ValueError("--max-test-bytes must be positive.")
    if limit_per_source is not None and limit_per_source < 1:
        raise ValueError("--limit-per-source must be positive.")
    if validate:
        workers = workers or available_cpus()
        if workers < 1:
            raise ValueError("--workers must be at least 1.")
    out = Path(out_dir).resolve()
    _check_out_dir(out, force)
    caps = CaseCaps(max_cases=max_cases, min_cases=min_cases, max_test_bytes=max_test_bytes)

    started = time.monotonic()
    timings: dict[str, Any] = {}
    sources: dict[str, dict[str, Any]] = {}

    index = EvalIndex()
    lcb_candidates: list[Problem] = []
    if lcb:
        index, lcb_candidates, sources[LCB] = read_lcb(lcb, caps, val_from)
        timings["lcb_s"] = sources[LCB]["read_s"]
        log(f"lcb: {index.problems} problems in {timings['lcb_s']} s, {len(lcb_candidates)} stdin problems "
            f"from {val_from} on can be validation problems")

    problems: list[Problem] = []
    for name in TRAIN_SOURCES:
        if not roots[name]:
            continue
        found, sources[name] = read_source(name, roots[name], caps, with_solutions=validate, limit=limit_per_source)
        problems.extend(found)
        record = sources[name]
        timings[f"read_{name}_s"] = record["read_s"]
        log(f"{name}: read {record['rows_read']} rows in {record['read_s']} s, {len(found)} pass the filters; "
            f"dropped {dict(record['dropped'])}")

    def drop(problem: Problem, reason: str) -> None:
        sources[problem.source]["dropped"][reason] += 1

    step = time.monotonic()
    problems, dropped = dedup_by_id(problems)
    for problem in dropped:
        drop(problem, "duplicate_id")
    problems, dropped = dedup_by_statement(problems)
    for problem in dropped:
        drop(problem, "duplicate_statement")
    timings["dedup_s"] = round(time.monotonic() - step, 1)

    step = time.monotonic()
    problems, contaminated = decontaminate(problems, index)
    for problem, reason in contaminated:
        drop(problem, reason)
    timings["decontaminate_s"] = round(time.monotonic() - step, 1)
    log(f"{len(problems)} problems after deduplication and decontamination")

    if validate:
        step = time.monotonic()
        checkable = [problem for problem in problems if _SOURCE_SPECS[problem.source].has_references]
        log(f"validating {len(checkable)} problems with {workers} workers")
        results = validate_all(checkable, run, workers, log)
        failed_ids = set()
        for problem, result in zip(checkable, results):
            entry = sources[problem.source].setdefault("validation", {
                "checked": 0, "passed": 0, "failed": 0, "no_reference": 0, "attempts": 0, "cpu_s": 0.0,
                "failure_reasons": Counter(), "failed_examples": [],
            })
            entry["checked"] += 1
            entry["attempts"] += result["attempts"]
            entry["cpu_s"] = round(entry["cpu_s"] + result["cpu_s"], 1)
            if result["passed"]:
                entry["passed"] += 1
                problem.validated = True
                continue
            failed_ids.add(id(problem))
            if result["reason"] == "no_python_reference":
                entry["no_reference"] += 1
                drop(problem, "no_python_reference")
                continue
            entry["failed"] += 1
            entry["failure_reasons"].update(result["reasons"])
            if len(entry["failed_examples"]) < 20:
                entry["failed_examples"].append(
                    {"id": problem.id, "reasons": result["reasons"], "detail": result["detail"]}
                )
            drop(problem, "validation_failed")
        problems = [problem for problem in problems if id(problem) not in failed_ids]
        timings["validate_s"] = round(time.monotonic() - step, 1)

    # The order before shuffling must not depend on dict or set iteration, so that a seed gives one dataset.
    train = _priority_order(problems)
    random.Random(seed).shuffle(train)
    if lcb:
        val_source = LCB
        val_note = (f"LCB stdin problems (every platform except LeetCode) with a contest date on or after "
                    f"{val_from}, sampled with the seed. They are not validated: LCB has no reference solutions.")
        val = sorted(lcb_candidates, key=lambda problem: problem.order)
        random.Random(seed).shuffle(val)
        val = val[:val_size]
        sources[LCB]["val"] = len(val)
    else:
        val_source = "train_holdout"
        val_note = (f"No --lcb was given, so val.jsonl holds {val_size} train problems that train.jsonl leaves "
                    "out. It measures fit to the train distribution, not generalisation to new problems.")
        val, train = train[:val_size], train[val_size:]
    if not train or not val:
        hint = " Use an earlier --val-from." if lcb and not val else ""
        raise ValueError(f"Nothing left to write: {len(train)} train and {len(val)} val problems.{hint}")

    for name in TRAIN_SOURCES:
        if name in sources:
            sources[name]["train"] = sum(problem.source == name for problem in train)
            sources[name]["val"] = sum(problem.source == name for problem in val)
    dropped_total: Counter = Counter()
    for record in sources.values():
        dropped_total.update(record.get("dropped") or {})

    report = {
        "rows": {"train": len(train), "val": len(val)},
        "val_source": val_source,
        "val_note": val_note,
        "sources": sources,
        "dropped": dropped_total,
        "settings": {
            "val_from": val_from,
            "val_size": val_size,
            "max_cases": max_cases,
            "min_cases": min_cases,
            "max_test_bytes": max_test_bytes,
            "validate": validate,
            "workers": workers if validate else None,
            "limit_per_source": limit_per_source,
            "seed": seed,
            "judge_imports": JUDGE_IMPORTS,
            "float_tolerance": FLOAT_TOLERANCE,
            "max_references": MAX_REFERENCES,
            "ngram": NGRAM,
            "ngram_share": NGRAM_SHARE,
            "limits": dataclasses.asdict(Limits()) if validate else None,
        },
        "timings": timings,
    }
    # The total covers everything but writing the files, because report.json is part of what is written.
    timings["total_s"] = round(time.monotonic() - started, 1)
    return write_dataset(out, train, val, report)


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orbit_verl.data.code_data",
        description="Build the code-RL dataset (train.jsonl, val.jsonl, report.json) from cached HF datasets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build", help="Build the dataset. The module docstring explains every step.")
    command.add_argument("--out-dir", required=True, help="Output dir for train.jsonl, val.jsonl and report.json.")
    command.add_argument("--codeforces", help="HF cache dir of open-r1/codeforces (codeforces-train-*.arrow).")
    command.add_argument("--code-contests", help="HF cache dir of deepmind/code_contests.")
    command.add_argument("--taco", help="HF cache dir of BAAI/TACO (taco-train-*.arrow).")
    command.add_argument("--primeintellect",
                         help="The primeintellect dir of DeepCoder-Preview-Dataset (train-*-of-*.parquet).")
    command.add_argument("--lcb", help="LiveCodeBench code_generation_lite dir with test*.jsonl (evaluation set).")
    command.add_argument("--val-from", default=DEFAULT_VAL_FROM,
                         help="LCB stdin problems from this contest date on become val.jsonl.")
    command.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE, help="Problems in val.jsonl.")
    command.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES, help="Test cases kept per problem.")
    command.add_argument("--min-cases", type=int, default=DEFAULT_MIN_CASES,
                         help="Drop problems with fewer test cases than this.")
    command.add_argument("--max-test-bytes", type=int, default=DEFAULT_MAX_TEST_BYTES,
                         help="Largest JSON size of one problem's tests; trailing cases are dropped to fit.")
    command.add_argument("--validate", action="store_true",
                         help="Keep a TACO/CodeContests/PrimeIntellect problem only if a reference solution passes "
                              "in the sandbox.")
    command.add_argument("--workers", type=int, help="Validation threads. Default: the CPU quota of this machine.")
    command.add_argument("--limit-per-source", type=int, help="Read at most this many rows of each train source.")
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--force", action="store_true", help="Replace an output dir that this tool wrote.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = build(
            args.out_dir,
            taco=args.taco,
            code_contests=args.code_contests,
            codeforces=args.codeforces,
            primeintellect=args.primeintellect,
            lcb=args.lcb,
            val_from=args.val_from,
            val_size=args.val_size,
            max_cases=args.max_cases,
            min_cases=args.min_cases,
            max_test_bytes=args.max_test_bytes,
            validate=args.validate,
            workers=args.workers,
            limit_per_source=args.limit_per_source,
            seed=args.seed,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"code_data: error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out_dir).resolve()
    print(f"code_data: wrote {out}")
    print(f"  rows: train {report['rows']['train']}, val {report['rows']['val']} (val from {report['val_source']})")
    for name, record in report["sources"].items():
        kept = f"train {record.get('train', 0)}, val {record.get('val', 0)}"
        dropped = ", ".join(f"{key} {value}" for key, value in sorted((record.get("dropped") or {}).items()))
        print(f"  {name}: {kept}; dropped: {dropped or 'none'}")
    print("  timings: " + ", ".join(f"{key} {value}" for key, value in report["timings"].items()))
    print(f"  next: prepare_data with --src {out / TRAIN_FILE} --val-src {out / VAL_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
