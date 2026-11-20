#!/usr/bin/env python3
"""Build the validation files of the RL and distillation runs from offline copies of public sets.

    # The chat probes, from an offline copy of the OASST2 validation split.
    $PY scripts/verl/data/build_val.py chat --out-dir DIR [--english 100] [--chat-local FILE] \\
        --oasst2 <hf_cache>/hub/datasets--OpenAssistant--oasst2/snapshots/<revision>/data/validation-<hash>.parquet
    # LiveCodeBench validation problems.
    $PY scripts/verl/data/build_val.py lcb --lcb DIR --val-from 2025-02-01 --out FILE [--val-size 200] [--seed 0]
    # MATH-500, GSM8K (the 250 problems that MGSM translates) and the Russian part of MGSM.
    $PY scripts/verl/data/build_val.py math500 --out math500.jsonl \\
        --math500 <hf_cache>/hub/datasets--HuggingFaceH4--MATH-500/snapshots/<revision>/test.jsonl
    $PY scripts/verl/data/build_val.py gsm8k --out gsm8k_en_paired250.jsonl \\
        --gsm8k <hf_cache>/hub/datasets--openai--gsm8k/snapshots/<revision>/main/test-00000-of-00001.parquet
    $PY scripts/verl/data/build_val.py mgsm-ru --out mgsm_ru.jsonl \\
        --mgsm <hf_cache>/hub/datasets--juletxara--mgsm/snapshots/<revision>/ru/test-00000-of-00001.parquet

``chat`` writes ``chat_probe_ru.jsonl`` and ``chat_probe_en.jsonl`` into ``--out-dir``. They hold the
prompts of the chat probe of the offline SFT evaluation (task itr_chat_probe), so the in-run validation
and the offline evaluation ask the model the same questions. The prompts are chosen by the probe's own
functions in chat_local.py, which this tool imports by path (default: the repo's
eval/in_training/tasks/chat_local.py): the root prompts of the OASST2 validation split and the trivial
prompts of chat_probe_trivial.jsonl. The Russian file gets every Russian prompt of both. The English
file gets the first ``--english`` English OASST2 prompts in the probe's order. The English trivial
prompts are left out, so the English probe holds OASST2 prompts only.

The probe loads OASST2 with ``datasets.load_dataset``. This tool reads the parquet file of the same
pinned revision with pyarrow instead, so it does not need the datasets library and writes no cache.
The parquet path must name the revision that chat_local.py pins, so a file of another revision cannot
be used by mistake.

A prompt that repeats an earlier one is written once. Prompts are compared after stripping whitespace,
as distill_data does, which would otherwise drop the copy itself and leave the file and the dataset with
different row counts. "Привет!" is both an OASST2 prompt and a trivial prompt, so the Russian file has
109 rows instead of 110. Each row is ``{"instruction", "response", "lang", "domain", "source"}``. The
response is empty, because a orbit/none validation set has no gold answer; distill_data reads only
``instruction`` and ``response``, and the other fields are there for people reading the file.

``lcb`` writes LiveCodeBench stdin problems as a code validation set. It reads LCB with code_data's own
``read_lcb`` and draws the sample exactly as ``code_data.build`` draws its val.jsonl. With the settings of
a code_data build it therefore writes that build's val.jsonl byte for byte. With the default test caps,
``--val-from 2025-02-01`` gives the 78 problems of the code validation set of the RL runs (lcb_val78.jsonl).

``math500``, ``gsm8k`` and ``mgsm-ru`` write the math validation sets. Each row is ``instruction`` (the
problem as the dataset states it) and ``response`` (the gold answer as ``\\boxed{<answer>}``), which the
orbit/math reward compares with the last ``\\boxed{}`` of an answer. The rows keep the dataset's order,
which matters because the RL recipe validates its no-think set on the first 200 rows of MATH-500.

- ``math500``: every row of MATH-500, with the dataset's ``answer`` as the gold answer and its
  ``unique_id`` kept.
- ``gsm8k``: the first 250 problems of the GSM8K test split, the problems that MGSM translates, so the
  Russian and the English validation ask the same questions in the same order. The gold answer is the
  number after "####" in the solution, with the thousands commas removed. ``unique_id`` is
  ``gsm8k_test_<row number>``.
- ``mgsm-ru``: the 250 problems of the Russian part of MGSM, with ``answer_number`` as the gold answer.

Each input path must name the dataset revision that the SFT evaluation pins (eval/in_training/tasks),
as for the chat probes. With those revisions the three commands write the files of the release recipes
byte for byte.

Every command refuses to replace an existing file, so a validation file that runs have already read is
never changed under them. Each command prints the checksums of the files it wrote.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

# scripts/verl/data -> the repo root is three levels up. The SFT evaluation's probe lives there.
CHAT_LOCAL_DEFAULT = Path(__file__).resolve().parents[3] / "eval" / "in_training" / "tasks" / "chat_local.py"
RU_FILE = "chat_probe_ru.jsonl"
EN_FILE = "chat_probe_en.jsonl"
DEFAULT_ENGLISH = 100
# The columns that chat_local.load_chat_probe takes from each OASST2 message.
OASST2_COLUMNS = ("message_id", "parent_id", "role", "lang", "text")
# The dataset revisions that the SFT evaluation reads (eval/in_training/tasks/itr_chat_*.yaml). The math
# validation files of the release recipes were written from exactly these.
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MGSM_REVISION = "b2f13d426afe3be8d69a7e739b36724db8b66bbc"
# MGSM translates the first 250 problems of the GSM8K test split.
MGSM_PAIRED_ROWS = 250


def load_chat_local(path: str | Path):
    """Import the offline probe's chat_local.py from its path. Its module level imports only the standard library."""
    spec = importlib.util.spec_from_file_location("chat_local_probe", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_revision(path: str | Path, revision: str, pinned_by: str) -> None:
    """Refuse a file that is not in the Hugging Face snapshot of ``revision``.

    The path is checked as given, without resolving it. The files of a Hugging Face snapshot are links
    into blobs/, and only the link's path names the revision.
    """
    if revision not in Path(path).parts:
        raise ValueError(f"{path} is not in the snapshot of revision {revision}, which {pinned_by} pins.")


def read_oasst2(path: str | Path, revision: str) -> list[dict[str, Any]]:
    """The OASST2 messages of one parquet file, as dicts of the columns the probe uses."""
    check_revision(path, revision, "chat_local.py")
    import pyarrow.parquet as pq

    return pq.read_table(str(path), columns=list(OASST2_COLUMNS)).to_pylist()


def unique_prompts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into the first copy of each prompt and the repeats, keeping the order."""
    seen: set[str] = set()
    kept, repeats = [], []
    for row in rows:
        key = row["prompt"].strip()
        (repeats if key in seen else kept).append(row)
        seen.add(key)
    return kept, repeats


def _record(row: dict[str, Any]) -> dict[str, Any]:
    return {"instruction": row["prompt"], "response": "", "lang": row["itr_lang"], "domain": row["itr_domain"],
            "source": row["source"]}


def _write_new(path: Path, lines: list[str]) -> str:
    """Write a file that must not exist yet, and return its sha256."""
    data = "".join(lines).encode("utf-8")
    with open(path, "xb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def build_chat(chat_local_path: str | Path, oasst2_path: str | Path, out_dir: str | Path,
               english: int = DEFAULT_ENGLISH) -> dict[str, Any]:
    """Write the Russian and the English probe files and return a summary of what went into them."""
    if english < 1:
        raise ValueError("--english must be at least 1.")
    chat_local = load_chat_local(chat_local_path)
    messages = read_oasst2(oasst2_path, chat_local.OASST2[1])
    # The probe's own rows in its own order, as load_chat_probe puts them together.
    rows = chat_local.select_probe_prompts(messages) + chat_local.read_trivial_prompts()
    russian, ru_repeats = unique_prompts([row for row in rows if row["itr_lang"] == "ru"])
    english_rows, en_repeats = unique_prompts(
        [row for row in rows if row["itr_lang"] == "en" and row["itr_domain"] == "chat"])
    if len(english_rows) < english:
        raise ValueError(f"The probe has only {len(english_rows)} distinct English OASST2 prompts, "
                         f"fewer than --english {english}.")
    english_rows = english_rows[:english]
    out = Path(out_dir)
    summary: dict[str, Any] = {"oasst2_messages": len(messages), "probe_rows": len(rows), "files": {}}
    for name, kept, repeats in ((RU_FILE, russian, ru_repeats), (EN_FILE, english_rows, en_repeats)):
        lines = [json.dumps(_record(row), ensure_ascii=False) + "\n" for row in kept]
        domains: dict[str, int] = {}
        for row in kept:
            domains[row["itr_domain"]] = domains.get(row["itr_domain"], 0) + 1
        summary["files"][name] = {
            "rows": len(kept),
            "domains": domains,
            "repeats_dropped": [row["prompt"][:80] for row in repeats],
            "sha256": _write_new(out / name, lines),
        }
    return summary


def _write_math(out: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Write math validation rows as jsonl and return their count and checksums."""
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    sha256 = _write_new(Path(out), [data.decode("utf-8")])
    return {"rows": len(rows), "sha256": sha256, "md5": hashlib.md5(data).hexdigest()}


def boxed(answer: str) -> str:
    """The gold answer in the form the orbit/math reward reads."""
    return "\\boxed{" + answer + "}"


def build_math500(test_jsonl: str | Path, out: str | Path) -> dict[str, Any]:
    """Write MATH-500 in its own order: the problem, the boxed answer and the dataset's unique_id."""
    check_revision(test_jsonl, MATH500_REVISION, "the SFT evaluation")
    rows = []
    with open(test_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows.append({"instruction": item["problem"], "response": boxed(item["answer"]),
                             "unique_id": item["unique_id"]})
    return _write_math(out, rows)


def gsm8k_answer(solution: str) -> str:
    """The final answer of a GSM8K solution: the number after "####", without thousands commas."""
    if "####" not in solution:
        raise ValueError(f"a GSM8K solution without '####': {solution[-80:]!r}")
    return solution.split("####")[-1].strip().replace(",", "")


def build_gsm8k(test_parquet: str | Path, out: str | Path, rows: int = MGSM_PAIRED_ROWS) -> dict[str, Any]:
    """Write the first ``rows`` problems of the GSM8K test split, the ones that MGSM translates."""
    check_revision(test_parquet, GSM8K_REVISION, "the SFT evaluation")
    import pyarrow.parquet as pq

    items = pq.read_table(str(test_parquet), columns=["question", "answer"]).to_pylist()[:rows]
    return _write_math(out, [{"instruction": item["question"], "response": boxed(gsm8k_answer(item["answer"])),
                              "unique_id": f"gsm8k_test_{number}"} for number, item in enumerate(items)])


def build_mgsm_ru(test_parquet: str | Path, out: str | Path) -> dict[str, Any]:
    """Write the Russian part of MGSM in its own order, with answer_number as the gold answer."""
    check_revision(test_parquet, MGSM_REVISION, "the SFT evaluation")
    import pyarrow.parquet as pq

    items = pq.read_table(str(test_parquet), columns=["question", "answer_number"]).to_pylist()
    return _write_math(out, [{"instruction": item["question"], "response": boxed(str(int(item["answer_number"])))}
                             for item in items])


def build_lcb(lcb_dir: str | Path, out: str | Path, *, val_from: str, val_size: int = 200, seed: int = 0,
              max_cases: int | None = None, min_cases: int | None = None,
              max_test_bytes: int | None = None) -> dict[str, Any]:
    """Write a LiveCodeBench validation set the way code_data.build writes its val.jsonl, and return a summary.

    A test cap left as None takes code_data's default.
    """
    # The repo's src/ is put on the path, so the tool runs without PYTHONPATH.
    src = str(Path(__file__).resolve().parents[3] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from orbit_verl.data import code_data as cd

    caps = cd.CaseCaps(max_cases=cd.DEFAULT_MAX_CASES if max_cases is None else max_cases,
                       min_cases=cd.DEFAULT_MIN_CASES if min_cases is None else min_cases,
                       max_test_bytes=cd.DEFAULT_MAX_TEST_BYTES if max_test_bytes is None else max_test_bytes)
    _, candidates, record = cd.read_lcb(lcb_dir, caps, val_from)
    # These three steps are the val sample of code_data.build. Sorting by row number first makes the
    # shuffle independent of the order read_lcb found the problems in.
    val = sorted(candidates, key=lambda problem: problem.order)
    random.Random(seed).shuffle(val)
    val = val[:val_size]
    if not val:
        raise ValueError(f"LCB has no stdin problem from {val_from} on that passes the test caps.")
    lines = [json.dumps(cd.output_record(problem), ensure_ascii=False) + "\n" for problem in val]
    return {
        "rows": len(val),
        "val_candidates": record["val_candidates"],
        "val_dropped": dict(record["val_dropped"]),
        "sha256": _write_new(Path(out), lines),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="write chat_probe_ru.jsonl and chat_probe_en.jsonl")
    chat.add_argument("--chat-local", default=str(CHAT_LOCAL_DEFAULT) if CHAT_LOCAL_DEFAULT.is_file() else None,
                      required=not CHAT_LOCAL_DEFAULT.is_file(),
                      help="the offline probe's tasks/chat_local.py (default: the repo's copy)")
    chat.add_argument("--oasst2", required=True, help="the OASST2 validation parquet of the pinned revision")
    chat.add_argument("--out-dir", required=True)
    chat.add_argument("--english", type=int, default=DEFAULT_ENGLISH)
    lcb = commands.add_parser("lcb", help="write LCB stdin problems as a code validation set")
    lcb.add_argument("--lcb", required=True, help="a LiveCodeBench code_generation_lite directory")
    lcb.add_argument("--out", required=True)
    lcb.add_argument("--val-from", required=True)
    lcb.add_argument("--val-size", type=int, default=200)
    lcb.add_argument("--seed", type=int, default=0)
    # Left out, the test caps are code_data's defaults, which the RL runs' code data was built with.
    lcb.add_argument("--max-cases", type=int)
    lcb.add_argument("--min-cases", type=int)
    lcb.add_argument("--max-test-bytes", type=int)
    math500 = commands.add_parser("math500", help="write MATH-500 as a math validation set")
    math500.add_argument("--math500", required=True, help="test.jsonl of the pinned MATH-500 snapshot")
    math500.add_argument("--out", required=True)
    gsm8k = commands.add_parser("gsm8k", help="write the 250 GSM8K test problems that MGSM translates")
    gsm8k.add_argument("--gsm8k", required=True, help="main/test-*.parquet of the pinned GSM8K snapshot")
    gsm8k.add_argument("--out", required=True)
    mgsm = commands.add_parser("mgsm-ru", help="write the Russian part of MGSM as a math validation set")
    mgsm.add_argument("--mgsm", required=True, help="ru/test-*.parquet of the pinned MGSM snapshot")
    mgsm.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "chat":
            summary = build_chat(args.chat_local, args.oasst2, args.out_dir, args.english)
        elif args.command == "math500":
            summary = build_math500(args.math500, args.out)
        elif args.command == "gsm8k":
            summary = build_gsm8k(args.gsm8k, args.out)
        elif args.command == "mgsm-ru":
            summary = build_mgsm_ru(args.mgsm, args.out)
        else:
            summary = build_lcb(args.lcb, args.out, val_from=args.val_from, val_size=args.val_size, seed=args.seed,
                                max_cases=args.max_cases, min_cases=args.min_cases,
                                max_test_bytes=args.max_test_bytes)
    except (OSError, ValueError) as exc:
        print(f"build_val: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
