"""CPU tests of the pass-rate tool's Python parts: the aggregators and make_subset.py.

The aggregators read verl's validation dump, so each test writes one by hand: the decoded prompt the
way verl dumps it (special tokens dropped), the response, the ground truth and the reward's keys.
No tokenizer is loaded, so response lengths are counted in characters. Only the standard library is
needed, so these tests run anywhere.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "scripts" / "verl" / "data" / "passrate"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


agg = _load("passrate_agg")
agg_code = _load("passrate_agg_code")
make_subset = _load("make_subset")

CODE_FRAME = ("user\n", "\n\nWrite Python code to solve the problem. Please place the solution code in the following "
                        "format:\n```python\n# Your solution code here\n```\nassistant\n<think>\n")


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _samples(prompt: str, correct: int, n: int = 4, gts: str | None = None) -> list[dict]:
    rows = []
    for i in range(n):
        row = {"input": prompt, "output": f"answer {i}", "score": 1.0 if i < correct else 0.0,
               "answer_correct": 1.0 if i < correct else 0.0, "truncated": 0.0, "think_unclosed": 0.0}
        if gts is not None:
            row["gts"] = gts
        rows.append(row)
    return rows


@pytest.mark.parametrize("mode", ["sft_think", "sft_think_ru"])
def test_math_shard_and_merge_keep_the_mixed_prompts(tmp_path, mode):
    questions = ["What is 1+1?", "What is 2+2?", "Сколько будет 3+3?"]
    subset = _write_jsonl(tmp_path / "subset.jsonl", [
        {"instruction": q, "response": "\\boxed{2}", "unique_id": f"src:{i}", "source": "OMR"}
        for i, q in enumerate(questions)])
    run = tmp_path / "run"
    dump = []
    for question, correct in zip(questions, (0, 2, 4)):
        dump += _samples(agg.PREFIX + question + agg.SUFFIXES[mode], correct)
    _write_jsonl(run / "val_generations" / "0.jsonl", dump)

    assert agg.main(["shard", "--run-dir", str(run), "--subset", str(subset), "--prompt-mode", mode]) == 0
    rows = {row["unique_id"]: row for row in _read_jsonl(run / "passrate.jsonl")}
    assert [rows[f"src:{i}"]["n_correct"] for i in range(3)] == [0, 2, 4]
    assert rows["src:1"]["pass_rate"] == 0.5 and rows["src:1"]["len_unit"] == "chars"

    out = tmp_path / "results"
    assert agg.main(["merge", "--subset", str(subset), "--out-dir", str(out), str(run)]) == 0
    keep = _read_jsonl(out / "keep.jsonl")
    # The default keep rule is 1..n-1, so only the prompt solved 2 times out of 4 is kept, with its source fields.
    assert [row["unique_id"] for row in keep] == ["src:1"]
    assert keep[0]["instruction"] == questions[1] and keep[0]["n_correct"] == 2
    assert json.loads((out / "summary.json").read_text())["kept"] == 1


def test_math_shard_flags_prompts_outside_the_subset(tmp_path):
    subset = _write_jsonl(tmp_path / "subset.jsonl", [{"instruction": "Q1", "response": "1", "unique_id": "s:0"}])
    run = tmp_path / "run"
    # The Russian frame does not fit an English shard, so its prompts are not cut and do not match.
    _write_jsonl(run / "val_generations" / "0.jsonl",
                 _samples(agg.PREFIX + "Q1" + agg.SUFFIXES["sft_think"], 1)
                 + _samples(agg.PREFIX + "Q2" + agg.SUFFIXES["sft_think"], 1))
    assert agg.main(["shard", "--run-dir", str(run), "--subset", str(subset)]) == 3
    summary = json.loads((run / "passrate_summary.json").read_text())
    assert summary["unmatched_prompts"] == 1


def test_code_shard_matches_by_tests_text_and_frame(tmp_path):
    tests = {key: json.dumps({"kind": "stdin", "tests": [{"input": key, "output": key}]}) for key in "ABC"}
    statements = {"A": "Read two numbers and print their sum.", "B": "Print hello.", "C": "Read a line. Print hello."}
    subset = _write_jsonl(tmp_path / "subset.jsonl", [
        {"instruction": statements[key], "response": tests[key], "unique_id": f"code:{key}", "source": "taco"}
        for key in "ABC"])

    def prompt(key: str) -> str:
        return CODE_FRAME[0] + statements[key] + CODE_FRAME[1]

    run = tmp_path / "run"
    # A carries its tests, so the tests and the text decide. B and C come without tests: B's statement is
    # found by text alone, and C's prompt contains B's statement as well, which the shared frame settles.
    _write_jsonl(run / "val_generations" / "0.jsonl",
                 _samples(prompt("A"), 1, gts=tests["A"]) + _samples(prompt("B"), 2) + _samples(prompt("C"), 0))
    assert agg_code.main(["shard", "--run-dir", str(run), "--subset", str(subset)]) == 0
    rows = {row["unique_id"]: row for row in _read_jsonl(run / "passrate.jsonl")}
    assert {key: rows[key]["match"] for key in rows} == {"code:A": "truth+text", "code:B": "text", "code:C": "frame"}
    assert [rows[f"code:{key}"]["n_correct"] for key in "ABC"] == [1, 2, 0]

    out = tmp_path / "results"
    assert agg_code.main(["merge", "--subset", str(subset), "--out-dir", str(out), str(run)]) == 0
    assert [row["unique_id"] for row in _read_jsonl(out / "keep.jsonl")] == ["code:A", "code:B"]


def test_code_shard_refuses_a_statement_with_other_tests(tmp_path):
    good = json.dumps({"kind": "stdin", "tests": [{"input": "1", "output": "1"}]})
    other = json.dumps({"kind": "stdin", "tests": [{"input": "2", "output": "2"}]})
    subset = _write_jsonl(tmp_path / "subset.jsonl",
                          [{"instruction": "Echo the number.", "response": good, "unique_id": "code:0"}])
    run = tmp_path / "run"
    _write_jsonl(run / "val_generations" / "0.jsonl",
                 _samples(CODE_FRAME[0] + "Echo the number." + CODE_FRAME[1], 1, gts=other))
    assert agg_code.main(["shard", "--run-dir", str(run), "--subset", str(subset)]) == 3
    summary = json.loads((run / "passrate_summary.json").read_text())
    assert [item["reason"] for item in summary["refused"]] == ["truth_mismatch"]


def test_make_subset_samples_shards_and_guards_the_prior(tmp_path):
    src = _write_jsonl(tmp_path / "math.jsonl", [
        {"instruction": f"Q{i}", "response": str(i), "source": "OMR" if i % 2 else "other"} for i in range(6)])
    out = tmp_path / "subsets" / "t.jsonl"
    assert make_subset.main(["sample", "--src", str(src), "--out", str(out), "--seed", "1"]) == 0
    rows = _read_jsonl(out)
    assert sorted(row["unique_id"] for row in rows) == [f"math:{i}" for i in range(6)]

    shards = tmp_path / "subsets" / "t.shards"
    assert make_subset.main(["shard", "--subset", str(out), "--out-dir", str(shards), "--shard-size", "4"]) == 0
    assert sorted(path.name for path in shards.iterdir()) == ["s000.jsonl", "s001.jsonl"]
    # A shard that already holds other rows is never mixed with a changed subset.
    (shards / "s001.jsonl").write_text("{}\n", encoding="utf-8")
    assert make_subset.main(["shard", "--subset", str(out), "--out-dir", str(shards), "--shard-size", "4"]) == 2

    # The OMR filters need a prior built from exactly this source file.
    prior = _write_jsonl(tmp_path / "prior.jsonl", [{"row": 1, "match": "exact", "pass_rate_72b_tir": 0.0},
                                                    {"row": 3, "match": "missing", "pass_rate_72b_tir": None},
                                                    {"row": 5, "match": "exact", "pass_rate_72b_tir": 0.5}])
    args = ["sample", "--src", str(src), "--out", str(tmp_path / "f.jsonl"), "--drop-omr-prior0", "--drop-omr-unmatched"]
    assert make_subset.main(args) == 2
    assert make_subset.main(args + ["--prior", str(prior)]) == 2
    (tmp_path / "prior.jsonl.meta.json").write_text(json.dumps({"src_sha256": make_subset.file_sha256(str(src))}))
    assert make_subset.main(args + ["--prior", str(prior)]) == 0
    kept = {row["unique_id"] for row in _read_jsonl(tmp_path / "f.jsonl")}
    assert kept == {"math:0", "math:2", "math:4", "math:5"}
