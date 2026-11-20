"""Build the dataset of an on-policy distillation run, with a prompt mode chosen per row.

CLI::

    $PY -m orbit_verl.data.distill_data --pool POOL.jsonl --val SPEC [--val SPEC ...] [--gold SPEC ...] \\
        --tokenizer DIR --max-prompt-length N --out-dir DIR [--runs-root DIR] [--think-share 0.5] \\
        [--seed 0] [--limit N] [--force]

``prepare_data`` writes datasets of one prompt mode and one reward. A distillation run trains on a
mix: math, code, instructions, chat and refusals, about half of them with thinking and half without,
and most of them without a gold answer. verl takes the agent loop of every row from the row's
``agent_name`` column, so each row can have its own prompt mode. This tool writes such a dataset.
Its files have the columns of ``prepare_data`` plus ``teacher_route``, and its manifest has the
same kind, so the launcher checks it like any other dataset. The manifest records ``prompt_mode``
as "mixed" and counts the rows of each mode.

Inputs:

- ``--pool PATH``: the distillation prompt pool, a jsonl file with one prompt per line. Its fields
  are ``id``, ``prompt`` (the raw first user turn, with no template and no leading <|endoftext|>),
  ``mode`` (think or no_think), ``domain`` (math, code, if, chat or refusal), ``lang`` and
  ``source``. Pool rows have no gold answer, so their data_source is orbit/none and their reward
  is 0. The domain picks the prompt format (``DOMAIN_FORMATS``): math problems get the boxed-answer
  instruction of sft_think, code problems the code instruction of sft_think_code, and every other
  prompt is sent as the user wrote it (sft_chat_think). The mode picks the variant with or without
  thinking. A row whose mode is null gets one drawn with ``--think-share``.
- ``--gold PATH:DATA_SOURCE:FAMILY[:COUNT]``, repeatable: rows with a gold answer. They serve the
  variant that adds the task reward to the distillation loss, and they give accuracy numbers on
  training prompts for free. PATH is a jsonl or parquet file with ``instruction`` and ``response``
  columns, like the RL data. FAMILY is math, math_ru or code
  and picks the prompt format. math_ru is for Russian math problems: they get the Russian
  instruction of sft_think_ru, and their rows record the domain math and the language ru, as pool
  rows of Russian math do. Each row's mode is drawn with ``--think-share``. COUNT rows are sampled
  after filtering; without COUNT every row is kept.
- ``--val PATH:DATA_SOURCE:PROMPT_MODE[:LIMIT]``, repeatable, at least one: validation sets, each in
  one prompt mode, with the first LIMIT rows that pass the filters. verl groups validation metrics by
  data_source, so a DATA_SOURCE with a tag after "#", such as ``orbit/math#no_think``, keeps the
  sets of different modes apart. The reward is chosen by the part before the tag.

What the tool does, in order:

1. Refuses a tokenizer that does not encode the markup of the SFT prompts as single tokens.
2. Reads the validation sets. Rows with an empty question, an empty gold answer (except for
   orbit/none sets), chat markup inside the question, or a repeated question are dropped, and so
   are rows whose rendered prompt is longer than ``max_prompt_length`` tokens.
3. Reads each gold input, shuffles it with the seed, and keeps rows until COUNT are kept. The same
   filters apply, and a row whose question is also in a validation set is dropped as well.
4. Reads the pool in file order with the same filters. A pool prompt that repeats a gold question
   is dropped, so the gold copy with its answer stays.
5. Shuffles the train rows with the seed and keeps the first ``limit`` of them.
6. Writes both parquet files, then ``manifest.json`` last, as ``prepare_data`` does.

Each row's prompt length is measured with ``prompts.prompt_token_ids``, the function the rollout
uses, so the filter and the rollout count the same tokens.

The ``teacher_route`` column holds the row's thinking mode, "think" or "no_think". With
``distillation.teacher_key: teacher_route`` verl sends each row to the teacher of its mode, so the
think rows and the no_think rows can have different teachers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orbit_verl import common
from orbit_verl.data import prepare_data, sources
from orbit_verl.data.prompts import (
    DATA_SOURCES,
    PROMPT_MODES,
    PromptMode,
    load_template,
    missing_tokens,
    prompt_token_ids,
    render_messages,
    reward_of,
)

# The value of orbit.prepare.data.builder that selects this tool in the launcher, also recorded
# in the manifest.
BUILDER = "distill"
# The manifest's prompt_mode. It names no single mode, because every row has its own.
MIXED = "mixed"
# The column that verl's distillation can route rows to teachers by (distillation.teacher_key).
TEACHER_ROUTE = "teacher_route"

THINK = "think"
NO_THINK = "no_think"
THINKING_MODES = (THINK, NO_THINK)

# The data_source of rows without a gold answer. Its reward is always 0.
NONE_SOURCE = "orbit/none"

# The prompt format of each pool domain. Instruction, chat and refusal prompts are sent as the user
# wrote them, because the math instruction would change what they ask for.
DOMAIN_FORMATS: dict[str, str] = {"math": "math", "code": "code", "if": "chat", "chat": "chat", "refusal": "chat"}

# The prompt format of a pool domain for a row in another language than English, where it differs
# from DOMAIN_FORMATS. A Russian math problem gets the Russian instruction (prompts.sft_think_ru):
# with the English one the teacher thought in English on about half of them, and distillation would
# pull the student's Russian thinking towards English too. Code keeps the English instruction of
# prompts.sft_think_code, and chat prompts carry no instruction at all.
LANG_FORMATS: dict[tuple[str, str], str] = {("math", "ru"): "math_ru"}
# The version of the format rules, recorded in the manifest. A dataset built under other rules does
# not match the config, so the launcher never reuses it by its name alone. 2: LANG_FORMATS.
FORMAT_VERSION = 2

# The prompt mode of each format, with and without thinking.
FORMAT_MODES: dict[tuple[str, str], str] = {
    ("math", THINK): "sft_think",
    ("math", NO_THINK): "sft_no_think",
    ("math_ru", THINK): "sft_think_ru",
    ("math_ru", NO_THINK): "sft_no_think_ru",
    ("code", THINK): "sft_think_code",
    ("code", NO_THINK): "sft_no_think_code",
    ("chat", THINK): "sft_chat_think",
    ("chat", NO_THINK): "sft_chat_no_think",
}
# The other direction: the format and the thinking mode of each of those prompt modes. A validation
# set names its prompt mode, and its rows need both values.
MODE_FORMATS: dict[str, tuple[str, str]] = {mode: key for key, mode in FORMAT_MODES.items()}

# The prompt formats that a gold input may use. Chat prompts have no gold answers to check.
# math_ru is the format of LANG_FORMATS for Russian math problems, so a set of Russian problems with
# gold answers is trained in the same prompt format as the Russian math of the pool.
GOLD_FAMILIES = ("math", "math_ru", "code")

# The markup of the SFT prompt templates. The tokenizer turns this text into special tokens wherever
# it appears, also inside a question. A question with <|im_end|> in it would then end the user turn
# early, so such rows are dropped.
MARKUP: tuple[str, ...] = tuple(sorted(set().union(*(PROMPT_MODES[mode].required_tokens
                                                     for mode in FORMAT_MODES.values()))))

# The columns of a gold or validation file: the question and its reference answer.
QUESTION_FIELD = "instruction"
ANSWER_FIELD = "response"

# The reasons a row can be dropped, in the order the filters run.
DROP_REASONS = ("empty", "markup", "val_overlap", "duplicate", "overlong")


def _split_spec(text: str, fields: int, flag: str, form: str) -> tuple[list[str], int | None]:
    """Split an input spec into its path, ``fields`` more fields, and an optional trailing number.

    The fields are taken from the right. Everything before them is the path, so a path that holds a
    colon itself stays whole.
    """
    parts = str(text).split(":")
    number = None
    if len(parts) > fields + 1 and parts[-1].strip().isdigit():
        number = int(parts.pop())
    path = ":".join(parts[:-fields])
    if len(parts) < fields + 1 or not path.strip() or not all(part.strip() for part in parts[-fields:]):
        raise ValueError(f"{flag} {text!r} does not have the form {form}.")
    return [path, *(part.strip() for part in parts[-fields:])], number


@dataclass(frozen=True)
class GoldSpec:
    """One ``--gold`` input: rows with a gold answer, rendered in the prompt format of ``family``."""

    path: str
    data_source: str
    family: str
    count: int | None = None

    def canonical(self) -> str:
        """The spec as manifests record it, with the source in the form ``sources.canonical_src`` gives.

        Two spellings of the same input, such as a relative and an absolute path, give the same string,
        so the launcher can compare a config with a manifest.
        """
        text = f"{sources.canonical_src(self.path)}:{self.data_source}:{self.family}"
        return text if self.count is None else f"{text}:{self.count}"


@dataclass(frozen=True)
class ValSpec:
    """One ``--val`` input: a validation set in one prompt mode."""

    path: str
    data_source: str
    prompt_mode: str
    limit: int | None = None

    def canonical(self) -> str:
        """The spec as manifests record it; see ``GoldSpec.canonical``."""
        text = f"{sources.canonical_src(self.path)}:{self.data_source}:{self.prompt_mode}"
        return text if self.limit is None else f"{text}:{self.limit}"


def parse_gold_spec(spec: str | GoldSpec) -> GoldSpec:
    """Read a ``--gold`` spec ``PATH:DATA_SOURCE:FAMILY[:COUNT]``. Raises ValueError for a spec that cannot work.

    The data source must name a reward that checks gold answers, and code rows need both the code
    family and the code reward. A mismatch would send code problems to the MATH check, or math
    problems to the sandbox, and every row would score 0 without an error.
    """
    if isinstance(spec, GoldSpec):
        return spec
    (path, data_source, family), count = _split_spec(spec, 2, "--gold", "PATH:DATA_SOURCE:FAMILY[:COUNT]")
    reward = reward_of(data_source)
    if family not in GOLD_FAMILIES:
        raise ValueError(f"--gold {spec!r} has family {family!r}; known families: {list(GOLD_FAMILIES)}.")
    known = sorted(source for source, name in DATA_SOURCES.items() if name != "none")
    if reward is None:
        raise ValueError(f"--gold {spec!r} has the unknown data_source {data_source!r}; use one of {known}, "
                         "optionally followed by a '#' tag.")
    if reward == "none":
        raise ValueError(f"--gold {spec!r} has data_source {data_source!r}, which checks no gold answer. "
                         f"Use one of {known}.")
    if (reward == "code") != (family == "code"):
        raise ValueError(f"--gold {spec!r}: the {family} family does not fit the {reward} reward of "
                         f"{data_source!r}. Code problems need orbit/code and the code family.")
    if count is not None and count < 1:
        raise ValueError(f"--gold {spec!r} asks for {count} rows; COUNT must be positive or left out.")
    return GoldSpec(path, data_source, family, count)


def parse_val_spec(spec: str | ValSpec) -> ValSpec:
    """Read a ``--val`` spec ``PATH:DATA_SOURCE:PROMPT_MODE[:LIMIT]``. Raises ValueError for a spec that cannot work.

    The prompt mode must be one of the SFT modes of ``FORMAT_MODES``. A code check needs a code
    prompt and the other checks need a prompt that is not a code prompt. orbit/none fits any mode.
    """
    if isinstance(spec, ValSpec):
        return spec
    (path, data_source, prompt_mode), limit = _split_spec(spec, 2, "--val", "PATH:DATA_SOURCE:PROMPT_MODE[:LIMIT]")
    if prompt_mode not in MODE_FORMATS:
        raise ValueError(f"--val {spec!r} has prompt mode {prompt_mode!r}; distillation datasets use "
                         f"{sorted(MODE_FORMATS)}.")
    reward = reward_of(data_source)
    if reward is None:
        raise ValueError(f"--val {spec!r} has the unknown data_source {data_source!r}; known sources: "
                         f"{sorted(DATA_SOURCES)}, each optionally followed by a '#' tag.")
    if reward != "none" and (reward == "code") != (MODE_FORMATS[prompt_mode][0] == "code"):
        raise ValueError(f"--val {spec!r}: prompt mode {prompt_mode} does not fit the {reward} reward of "
                         f"{data_source!r}. Code problems need orbit/code and a code prompt mode.")
    if limit is not None and limit < 1:
        raise ValueError(f"--val {spec!r} keeps {limit} rows; LIMIT must be positive or left out.")
    return ValSpec(path, data_source, prompt_mode, limit)


@dataclass
class _Row:
    """One prompt on its way into the dataset. ``number`` is its row number in the input it came from."""

    question: str
    answer: str
    data_source: str
    mode: PromptMode
    domain: str
    thinking: str
    id: str
    number: int
    lang: str | None = None
    source: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    prompt_tokens: int = 0


def _drop_counter() -> Counter:
    return Counter(dict.fromkeys(DROP_REASONS, 0))


def _admit(question: str, answer: str | None, seen: set[str], blocked: set[str], dropped: Counter) -> bool:
    """Apply the filters that need no tokenizer. Returns True for a row that may go on.

    ``answer`` is None for a row that needs no gold answer. ``seen`` holds the questions taken so far
    and receives this one. ``blocked`` holds the validation questions, which no train row may repeat.
    The reason for a drop is counted in ``dropped``.
    """
    key = question.strip()
    if not key or (answer is not None and not answer.strip()):
        dropped["empty"] += 1
        return False
    if any(marker in question for marker in MARKUP):
        dropped["markup"] += 1
        return False
    if key in blocked:
        dropped["val_overlap"] += 1
        return False
    if key in seen:
        dropped["duplicate"] += 1
        return False
    seen.add(key)
    return True


def _fits(row: _Row, tokenizer: Any, max_prompt_length: int, dropped: Counter) -> bool:
    """Render the row's prompt and measure it as the rollout will. Returns False for a prompt over the limit."""
    row.messages = render_messages(row.mode, row.question)
    row.prompt_tokens = len(prompt_token_ids(tokenizer, row.messages))
    if row.prompt_tokens > max_prompt_length:
        dropped["overlong"] += 1
        return False
    return True


def _input_name(path: str) -> str:
    """A short name for an input in row ids and the ``source`` field: the file name."""
    return Path(path).name


def _domain_and_lang(prompt_format: str) -> tuple[str, str | None]:
    """The domain and the language that a row in this prompt format records in its extra_info.

    A format of LANG_FORMATS stands for its domain in one language, so math_ru gives ("math", "ru").
    Any other format is its own domain, with no language.
    """
    return next(((domain, lang) for (domain, lang), fmt in LANG_FORMATS.items() if fmt == prompt_format),
                (prompt_format, None))


def _draw(rng: random.Random, think_share: float) -> str:
    return THINK if rng.random() < think_share else NO_THINK


def _pool_row_problem(row: Any) -> str | None:
    """What is wrong with one parsed pool line, or None when it is usable."""
    if not isinstance(row, dict):
        return "is not a JSON object"
    if row.get("id") is None or not str(row["id"]).strip():
        return "has no id"
    if not isinstance(row.get("prompt"), str):
        return "has no prompt string"
    if row.get("domain") not in DOMAIN_FORMATS:
        return f"has domain {row.get('domain')!r}; known domains: {sorted(DOMAIN_FORMATS)}"
    if row.get("mode") is not None and row["mode"] not in THINKING_MODES:
        return f"has mode {row['mode']!r}; known modes: {list(THINKING_MODES)} or null"
    for key in ("lang", "source"):
        if row.get(key) is not None and not isinstance(row[key], str):
            return f"has a {key} that is not a string"
    return None


def read_pool(path: str | os.PathLike) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The rows of a prompt pool, and the record of the file for the manifest.

    A malformed line raises ValueError with its line number. Such a line means the pool builder has
    a bug, and skipping it quietly would change the mix without anyone noticing. An empty prompt is
    not malformed; the converter drops it and counts it.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Prompt pool {target} does not exist.")
    rows = []
    with open(target, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{target}:{line_number} is not valid JSON ({exc}).") from None
            problem = _pool_row_problem(row)
            if problem:
                raise ValueError(f"{target}:{line_number} {problem}.")
            rows.append(row)
    record = {"kind": "file", "path": str(target.resolve()), "sha256": common.sha256_file(target), "rows": len(rows)}
    return rows, record


def _record(row: _Row, split: str) -> dict[str, Any]:
    return {
        "data_source": row.data_source,
        "prompt": row.messages,
        "agent_name": row.mode.agent_loop,
        "ability": row.domain,
        "reward_model": {"style": "rule", "ground_truth": row.answer},
        "extra_info": {
            "index": int(row.number),
            "split": split,
            "prompt_mode": row.mode.name,
            "question": row.question,
            "prompt_tokens": int(row.prompt_tokens),
            "id": row.id,
            "domain": row.domain,
            "lang": row.lang,
            "source": row.source,
            "mode": row.thinking,
        },
        TEACHER_ROUTE: row.thinking,
    }


def parquet_schema():
    """The schema of ``prepare_data`` with five more ``extra_info`` fields and the ``teacher_route`` column.

    The fields are added to ``prepare_data``'s schema instead of being listed again, so a change of
    the common columns reaches both tools.
    """
    import pyarrow as pa

    base = prepare_data.parquet_schema()
    position = base.get_field_index("extra_info")
    info = base.field("extra_info").type
    fields = [info.field(i) for i in range(info.num_fields)]
    fields += [pa.field(name, pa.string()) for name in ("id", "domain", "lang", "source", "mode")]
    return base.set(position, pa.field("extra_info", pa.struct(fields))).append(pa.field(TEACHER_ROUTE, pa.string()))


def _counts(rows: list[_Row]) -> dict[str, dict[str, dict[str, int]]]:
    """Rows per domain, then per thinking mode, then per data_source."""
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        by_source = counts.setdefault(row.domain, {}).setdefault(row.thinking, {})
        by_source[row.data_source] = by_source.get(row.data_source, 0) + 1
    return counts


def _check_tokenizer(tokenizer: Any) -> None:
    """Refuse a tokenizer that splits the markup of the SFT prompts, and fail early on a broken template file."""
    for name in sorted(MODE_FORMATS):
        load_template(name)
        missing = missing_tokens(name, tokenizer)
        if missing:
            raise ValueError(
                f"Prompt mode {name!r} needs {missing} as single tokens, but the tokenizer splits them into "
                "pieces. Distillation datasets are written in the SFT chat format, which only the SFT models "
                "of the release line (Qwen3 vocabulary) were trained on."
            )


def _check_inputs_exist(pool: str | None, gold: list[GoldSpec], val: list[ValSpec]) -> None:
    """Raise FileNotFoundError for an input file that does not exist.

    The check runs before the tokenizer is loaded, so a typo in a path fails in a second instead of
    after the slow part.
    """
    paths = ([pool] if pool is not None else []) + [spec.path for spec in [*gold, *val]]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist: {', '.join(missing)}.")


def _check_val_sources(specs: list[ValSpec]) -> None:
    """Refuse two validation sets that share a data_source but differ in prompt mode.

    verl averages the validation metrics per data_source, so the two sets would be reported as one
    number. A tag such as ``orbit/math#no_think`` keeps them apart.
    """
    modes: dict[str, set[str]] = {}
    for spec in specs:
        modes.setdefault(spec.data_source, set()).add(spec.prompt_mode)
    shared = {source: sorted(names) for source, names in modes.items() if len(names) > 1}
    if shared:
        raise ValueError(f"Validation sets in different prompt modes share a data_source: {shared}. verl would "
                         "report them as one set. Tag each one, as in orbit/math#think and orbit/math#no_think.")


def build(
    *,
    pool: str | None,
    gold: Sequence[GoldSpec],
    val: Sequence[ValSpec],
    tokenizer: Any,
    max_prompt_length: int,
    think_share: float,
    seed: int,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return the train records, the val records and the statistics for the manifest. Writes nothing.

    The steps are those of the module docstring. ``convert`` checks the arguments and writes the files.
    """
    _check_tokenizer(tokenizer)
    _check_val_sources(list(val))
    rng = random.Random(seed)

    val_rows: list[_Row] = []
    val_dropped, val_sources = [], []
    for spec in val:
        raw, record = sources.read_rows(spec.path, QUESTION_FIELD, ANSWER_FIELD)
        fmt, thinking = MODE_FORMATS[spec.prompt_mode]
        domain, lang = _domain_and_lang(fmt)
        needs_answer = reward_of(spec.data_source) != "none"
        name, dropped, seen, kept = _input_name(spec.path), _drop_counter(), set(), []
        for number, question, answer in raw:
            if spec.limit is not None and len(kept) >= spec.limit:
                break
            if not _admit(question, answer if needs_answer else None, seen, set(), dropped):
                continue
            row = _Row(question, answer, spec.data_source, PROMPT_MODES[spec.prompt_mode], domain, thinking,
                       id=f"{name}:{number}", number=number, lang=lang, source=name)
            if _fits(row, tokenizer, max_prompt_length, dropped):
                kept.append(row)
        if not kept:
            raise ValueError(f"--val {spec.canonical()} kept no rows (dropped {dict(dropped)}).")
        val_rows += kept
        val_dropped.append(dict(dropped))
        val_sources.append(record)
    blocked = {row.question.strip() for row in val_rows}

    # Train questions taken so far, from every input, so that one question is trained on once.
    seen: set[str] = set()
    train_rows: list[_Row] = []
    gold_dropped, gold_sources = [], []
    for spec in gold:
        raw, record = sources.read_rows(spec.path, QUESTION_FIELD, ANSWER_FIELD)
        order = list(raw)
        rng.shuffle(order)
        name, dropped, kept = _input_name(spec.path), _drop_counter(), []
        # A math_ru row records the domain math and the language ru, as a Russian math row of the pool
        # does. The manifest then counts it as math, and its extra_info still says that it is Russian.
        domain, lang = _domain_and_lang(spec.family)
        for number, question, answer in order:
            if spec.count is not None and len(kept) >= spec.count:
                break
            if not _admit(question, answer, seen, blocked, dropped):
                continue
            thinking = _draw(rng, think_share)
            row = _Row(question, answer, spec.data_source, PROMPT_MODES[FORMAT_MODES[(spec.family, thinking)]],
                       domain, thinking, id=f"{name}:{number}", number=number, lang=lang, source=name)
            if _fits(row, tokenizer, max_prompt_length, dropped):
                kept.append(row)
        if spec.count is not None and len(kept) < spec.count:
            raise ValueError(f"--gold {spec.canonical()} asks for {spec.count} rows, but only {len(kept)} are left "
                             f"after filtering (dropped {dict(dropped)}). Lower COUNT or leave it out.")
        train_rows += kept
        gold_dropped.append(dict(dropped))
        gold_sources.append(record)

    pool_record, pool_dropped, modes_drawn = None, None, 0
    if pool is not None:
        raw_pool, pool_record = read_pool(pool)
        dropped = _drop_counter()
        for number, item in enumerate(raw_pool):
            if not _admit(item["prompt"], None, seen, blocked, dropped):
                continue
            thinking = item.get("mode")
            if thinking is None:
                thinking = _draw(rng, think_share)
                modes_drawn += 1
            domain = item["domain"]
            prompt_format = LANG_FORMATS.get((domain, item.get("lang")), DOMAIN_FORMATS[domain])
            row = _Row(item["prompt"], "", NONE_SOURCE, PROMPT_MODES[FORMAT_MODES[(prompt_format, thinking)]],
                       domain, thinking, id=str(item["id"]), number=number, lang=item.get("lang"),
                       source=item.get("source"))
            if _fits(row, tokenizer, max_prompt_length, dropped):
                train_rows.append(row)
        pool_dropped = dict(dropped)

    rng.shuffle(train_rows)
    if limit is not None:
        train_rows = train_rows[:limit]
    if not train_rows:
        raise ValueError(f"Nothing left to train on after filtering (pool dropped {pool_dropped}, "
                         f"gold dropped {gold_dropped}).")

    lengths = sorted(row.prompt_tokens for row in train_rows + val_rows)
    stats = {
        "pool": sources.canonical_src(pool) if pool is not None else None,
        "pool_source": pool_record,
        "pool_modes_drawn": modes_drawn,
        "gold": [spec.canonical() for spec in gold],
        "gold_sources": gold_sources,
        "val": [spec.canonical() for spec in val],
        "val_sources": val_sources,
        "rows": {"train": len(train_rows), "val": len(val_rows)},
        "dropped": {"pool": pool_dropped, "gold": gold_dropped, "val": val_dropped},
        "counts": {"train": _counts(train_rows), "val": _counts(val_rows)},
        "prompt_modes": {
            "train": dict(sorted(Counter(row.mode.name for row in train_rows).items())),
            "val": dict(sorted(Counter(row.mode.name for row in val_rows).items())),
        },
        "prompt_tokens": {
            "max": int(lengths[-1]),
            "p50": prepare_data._percentile(lengths, 0.50),
            "p99": prepare_data._percentile(lengths, 0.99),
        },
    }
    return [_record(row, "train") for row in train_rows], [_record(row, "val") for row in val_rows], stats


def convert(
    pool: str | None,
    out_dir: str,
    *,
    tokenizer_dir: str,
    max_prompt_length: int,
    val: Sequence[str | ValSpec],
    gold: Sequence[str | GoldSpec] = (),
    think_share: float = 0.5,
    seed: int = 0,
    limit: int | None = None,
    force: bool = False,
    runs_root: str | None = None,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Build ``out_dir`` with train.parquet, val.parquet and manifest.json. Returns the manifest.

    ``pool`` may be None when ``gold`` gives the train rows. ``val`` and ``gold`` take specs as
    strings or parsed. ``tokenizer`` replaces the tokenizer loaded from ``tokenizer_dir``, for tests;
    the manifest still fingerprints the files in ``tokenizer_dir``. ``runs_root`` places the caches
    as in ``prepare_data.convert``. ``force`` replaces an existing dataset that one of the two tools
    wrote, and nothing else.
    """
    gold_specs = [parse_gold_spec(spec) for spec in gold]
    val_specs = [parse_val_spec(spec) for spec in val]
    if pool is None and not gold_specs:
        raise ValueError("Nothing to train on: give a prompt pool, gold inputs, or both.")
    if not val_specs:
        # verl builds a validation dataloader even with validation turned off, and an empty
        # val.parquet makes that dataloader fail at start-up.
        raise ValueError("At least one validation set (--val) is needed.")
    if max_prompt_length < 1:
        raise ValueError("max_prompt_length must be positive.")
    if not 0.0 <= float(think_share) <= 1.0:
        raise ValueError(f"think_share must be between 0 and 1, not {think_share}.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive.")
    _check_inputs_exist(None if pool is None else str(pool), gold_specs, val_specs)
    _check_val_sources(val_specs)
    out = Path(out_dir).resolve()
    prepare_data._check_out_dir(out, force)

    common.apply_cache_env(None if runs_root is None else str(runs_root))
    import pyarrow as pa
    import pyarrow.parquet as pq

    if tokenizer is None:
        tokenizer = prepare_data._load_tokenizer(tokenizer_dir)
    train, val_records, stats = build(
        pool=None if pool is None else str(pool),
        gold=gold_specs,
        val=val_specs,
        tokenizer=tokenizer,
        max_prompt_length=int(max_prompt_length),
        think_share=float(think_share),
        seed=int(seed),
        limit=limit,
    )
    payload = {
        "builder": BUILDER,
        "format_version": FORMAT_VERSION,
        "prompt_mode": MIXED,
        **stats,
        "max_prompt_length": int(max_prompt_length),
        "tokenizer_dir": str(Path(tokenizer_dir).resolve()),
        "tokenizer_sha256": common.tokenizer_sha256(tokenizer_dir),
        "tokenizer_family": prepare_data.tokenizer_family(tokenizer_dir),
        "think_share": float(think_share),
        "seed": int(seed),
        "limit": limit,
    }

    # As in prepare_data: everything goes to a staging directory next to the target, which is moved
    # into place at the end, so an interrupted run leaves no half-written dataset under the target name.
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.staging-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        schema = parquet_schema()
        for name, records in ((prepare_data.TRAIN_FILE, train), (prepare_data.VAL_FILE, val_records)):
            pq.write_table(pa.Table.from_pylist(records, schema=schema), staging / name)
        manifest = common.write_manifest(staging / common.DATA_MANIFEST, common.DATA_MANIFEST_KIND, payload)
        if out.exists():
            # _check_out_dir has already confirmed that this is empty or a dataset of ours.
            shutil.rmtree(out)
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orbit_verl.data.distill_data",
        description="Build a distillation dataset: a prompt pool, optional gold rows and validation sets, "
                    "each row in its own SFT prompt mode.",
    )
    parser.add_argument("--pool", help="Prompt pool jsonl (id, prompt, mode, domain, lang, source).")
    parser.add_argument(
        "--gold", action="append", default=[], metavar="PATH:DATA_SOURCE:FAMILY[:COUNT]",
        help="Rows with gold answers (instruction/response), family math, math_ru or code. Repeatable.",
    )
    parser.add_argument(
        "--val", action="append", default=[], metavar="PATH:DATA_SOURCE:PROMPT_MODE[:LIMIT]",
        help="A validation set in one prompt mode, e.g. math500.jsonl:orbit/math#think:sft_think. Repeatable; "
             "at least one is needed. A orbit/none set still needs the response field, which may be empty.",
    )
    parser.add_argument("--tokenizer", required=True, help="Tokenizer dir, normally the prepared model dir.")
    parser.add_argument(
        "--max-prompt-length", required=True, type=int,
        help="Drop prompts longer than this many tokens. Must equal data.max_prompt_length of the run.",
    )
    parser.add_argument("--out-dir", required=True, help="Output dir for train.parquet, val.parquet, manifest.json.")
    parser.add_argument(
        "--runs-root",
        help="Root of the runtime outputs; every cache goes under <runs-root>/cache. "
             f"Default: ${common.RUNS_ROOT_ENV}.",
    )
    parser.add_argument("--think-share", type=float, default=0.5,
                        help="Share of think rows among the gold rows and the pool rows without a mode.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Keep the first N train rows after filtering and shuffling.")
    parser.add_argument("--force", action="store_true", help="Replace an output dir that a data tool wrote.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        runs_root = common.resolve_runs_root(args.runs_root)
    except common.RunsRootError:
        print(f"distill_data: error: {common.runs_root_help(('--runs-root PATH',))}", file=sys.stderr)
        return 2
    try:
        manifest = convert(
            args.pool,
            args.out_dir,
            tokenizer_dir=args.tokenizer,
            max_prompt_length=args.max_prompt_length,
            val=args.val,
            gold=args.gold,
            think_share=args.think_share,
            seed=args.seed,
            limit=args.limit,
            force=args.force,
            runs_root=runs_root,
        )
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        print(f"distill_data: error: {exc}", file=sys.stderr)
        return 2
    rows, tokens = manifest["rows"], manifest["prompt_tokens"]
    print(f"distill_data: wrote {Path(args.out_dir).resolve()}")
    print(f"  tokenizer family {manifest['tokenizer_family'] or 'unknown'}")
    print(f"  rows: train {rows['train']}, val {rows['val']}")
    for split in ("train", "val"):
        modes = ", ".join(f"{name} {count}" for name, count in manifest["prompt_modes"][split].items())
        print(f"  {split} prompt modes: {modes}")
    print(f"  dropped: {json.dumps(manifest['dropped'], sort_keys=True)}")
    print(
        f"  prompt tokens: max {tokens['max']}, p50 {tokens['p50']}, p99 {tokens['p99']} "
        f"(limit {manifest['max_prompt_length']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
