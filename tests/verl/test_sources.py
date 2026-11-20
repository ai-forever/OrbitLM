"""CPU tests for orbit_verl.data.sources: reading question and answer rows from local files.

Run with PYTHONPATH=<repo>/src. The parquet test needs pyarrow and skips without it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orbit_verl import common
from orbit_verl.data import sources


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def test_canonical_and_same_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rows.jsonl").write_text("{}\n")
    assert sources.canonical_src("rows.jsonl") == str((tmp_path / "rows.jsonl").resolve())
    assert sources.same_source("rows.jsonl", tmp_path / "rows.jsonl")
    (tmp_path / "sub").mkdir()
    assert sources.same_source("./sub/../rows.jsonl", str(tmp_path / "rows.jsonl"))
    assert not sources.same_source("rows.jsonl", "other.jsonl")
    assert sources.same_source(None, None) and not sources.same_source(None, "rows.jsonl")
    # A symlink names the same rows as its target.
    (tmp_path / "link.jsonl").symlink_to(tmp_path / "rows.jsonl")
    assert sources.same_source("link.jsonl", "rows.jsonl")


def test_read_rows_of_a_jsonl_file(tmp_path):
    path = _jsonl(tmp_path / "rows.jsonl", [
        {"instruction": "Q?", "response": "4", "source": "extra fields are ignored"},
        {"instruction": "Сколько?", "response": 5},
    ])
    # Blank lines are skipped, and every value is read as a string.
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rows, record = sources.read_rows(str(path), "instruction", "response")
    assert rows == [(0, "Q?", "4"), (1, "Сколько?", "5")]
    assert record == {"kind": "file", "path": str(path.resolve()), "sha256": common.sha256_file(path),
                      "question_field": "instruction", "answer_field": "response", "rows": 2}


def test_read_rows_with_other_field_names(tmp_path):
    path = _jsonl(tmp_path / "rows.jsonl", [{"problem": "P", "answer": "A"}])
    rows, record = sources.read_rows(path, "problem", "answer")
    assert rows == [(0, "P", "A")] and (record["question_field"], record["answer_field"]) == ("problem", "answer")


def test_read_rows_of_a_parquet_file(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"instruction": ["Q1", "Q2"], "response": ["1", "2"], "other": [1, 2]}), path)
    rows, record = sources.read_rows(str(path), "instruction", "response")
    assert rows == [(0, "Q1", "1"), (1, "Q2", "2")] and record["rows"] == 2


def test_read_rows_refuses_a_missing_file_and_a_missing_field(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.jsonl"):
        sources.read_rows(str(tmp_path / "missing.jsonl"), "instruction", "response")
    path = _jsonl(tmp_path / "rows.jsonl", [{"instruction": "Q?", "response": "4"}])
    with pytest.raises(KeyError, match="question-field"):
        sources.read_rows(str(path), "question", "response")
    # A directory is not a source file either.
    with pytest.raises(FileNotFoundError):
        sources.read_rows(str(tmp_path), "instruction", "response")


def test_import_stays_light():
    # The module is imported by the converters and by the launcher's checks, so it must not pull in
    # pyarrow, datasets or transformers before a file is read.
    code = ("import sys, orbit_verl.data.sources; "
            "print(*(name in sys.modules for name in ('pyarrow', 'datasets', 'transformers')))")
    env = dict(os.environ, PYTHONPATH=str(Path(sources.__file__).resolve().parents[2]), PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert out.split() == ["False", "False", "False"]
