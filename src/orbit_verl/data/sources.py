"""Where the question and answer rows of a dataset come from: a local jsonl or parquet file.

A source is one path. It is the ``--src`` of ``prepare_data``, ``orbit.prepare.data.src`` in a
config, and the PATH part of the gold and validation specs of ``distill_data``. The question and
the gold answer are read from the columns ``question_field`` and ``answer_field`` (``instruction``
and ``response`` in every data file of the release).

A parquet file is read with pyarrow. Any other file is read as jsonl, one JSON object per line,
which covers ``.jsonl``, ``.json`` and plain text exports. Every value is turned into a string, as
the reward and the prompt templates expect.

Only the standard library is imported at module level.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from orbit_verl import common


def canonical_src(spec: str | os.PathLike) -> str:
    """The form of a source that manifests record: the resolved absolute path."""
    return str(Path(spec).resolve())


def same_source(a: str | os.PathLike | None, b: str | os.PathLike | None) -> bool:
    """True when two source paths name the same file, compared by their resolved paths.

    A pre-flight that compares a manifest's ``src`` with a config uses this instead of comparing the
    strings, so a relative and an absolute path to the same file count as the same source.
    """
    if a is None or b is None:
        return a is None and b is None
    return canonical_src(a) == canonical_src(b)


def _iter_file_rows(src: str, question_field: str, answer_field: str):
    """Yield (question, answer) pairs of a local parquet or jsonl file."""
    if src.endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(src, columns=[question_field, answer_field])
        for question, answer in zip(table.column(question_field).to_pylist(),
                                    table.column(answer_field).to_pylist()):
            yield str(question), str(answer)
        return
    with open(src, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield str(row[question_field]), str(row[answer_field])


def read_rows(
    spec: str | os.PathLike,
    question_field: str,
    answer_field: str,
) -> tuple[list[tuple[int, str, str]], dict[str, Any]]:
    """Return ``(source row number, question, gold answer)`` for every row, and a record of the source.

    The record goes into the dataset manifest. It says exactly which rows were read: the file's
    path and sha256, the two field names and the row count.
    """
    src = str(spec)
    if not Path(src).is_file():
        raise FileNotFoundError(f"Source file {src} does not exist.")
    try:
        rows = [(number, question, answer)
                for number, (question, answer) in enumerate(_iter_file_rows(src, question_field, answer_field))]
    except KeyError as exc:
        raise KeyError(f"{src} has no field {exc}. Pick the fields with --question-field / --answer-field.") from None
    record = {
        "kind": "file",
        "path": canonical_src(src),
        "sha256": common.sha256_file(src),
        "question_field": question_field,
        "answer_field": answer_field,
        "rows": len(rows),
    }
    return rows, record
