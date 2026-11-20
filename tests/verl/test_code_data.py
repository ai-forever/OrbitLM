"""CPU tests for orbit_verl.data.code_data: readers, filters, test caps, dedup, decontamination, validation, output.

Each source is mimicked by tiny arrow files that pyarrow writes into tmp_path, with the directory layout and the
column names of the Hugging Face cache. LCB is mimicked by a test6.jsonl whose private tests are encoded the way
LCB encodes them. Validation runs with a fake run function, except in the last test, which starts real sandboxes
and runs on Linux only.

Run with the verl python and PYTHONPATH=<repo>/src (the last test needs Linux):

    $PY -m pytest -q tests/verl/test_code_data.py
"""
from __future__ import annotations

import base64
import json
import pickle
import sys
import threading
import zlib
from pathlib import Path

import pytest

from orbit_verl.data import code_data as cd
from orbit_verl.data.code_data import CaseCaps, Problem

CAPS = CaseCaps(max_cases=30, min_cases=3, max_test_bytes=4_000_000)
REWARD_SETTINGS = {"kind": "stdin", "judge_imports": True, "float_tolerance": 1e-6}


def _pa():
    return pytest.importorskip("pyarrow")


# ----------------------------------------------------------------------------------------------------------
# Tiny data
# ----------------------------------------------------------------------------------------------------------

def _text(tag: str, words: int = 40) -> str:
    """A statement made of words that only this tag uses, so two tags never share an n-gram by accident."""
    return " ".join(f"{tag}w{i}" for i in range(words))


def _cases(tag: str, n: int = 4) -> list[dict[str, str]]:
    """n stdin cases with distinct inputs and distinct outputs."""
    return [{"input": f"{i} {tag}\n", "output": f"{i}{tag}\n"} for i in range(n)]


def _write_arrow(path: Path, rows: list[dict], schema, *, stream: bool = True) -> Path:
    """Writes rows like the Hugging Face cache does: an Arrow stream, here in batches of two rows."""
    pa = _pa()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    with pa.OSFile(str(path), "wb") as sink:
        opener = pa.ipc.new_stream if stream else pa.ipc.new_file
        with opener(sink, schema) as writer:
            writer.write_table(table, max_chunksize=2)
    return path


def _taco_schema():
    pa = _pa()
    names = ("question", "solutions", "starter_code", "input_output", "difficulty", "raw_tags", "name", "source",
             "tags", "skill_types", "url", "Expected Auxiliary Space", "time_limit", "date", "picture_num",
             "memory_limit", "Expected Time Complexity")
    return pa.schema([(name, pa.string()) for name in names])


def _code_contests_schema():
    pa = _pa()
    tests = pa.struct([("input", pa.list_(pa.string())), ("output", pa.list_(pa.string()))])
    solutions = pa.struct([("language", pa.list_(pa.int64())), ("solution", pa.list_(pa.string()))])
    return pa.schema([
        ("name", pa.string()), ("description", pa.string()),
        ("public_tests", tests), ("private_tests", tests), ("generated_tests", tests),
        ("source", pa.int64()), ("difficulty", pa.int64()),
        ("solutions", solutions), ("incorrect_solutions", solutions),
        ("cf_contest_id", pa.int64()), ("cf_index", pa.string()), ("cf_points", pa.float32()),
        ("cf_rating", pa.int32()), ("cf_tags", pa.list_(pa.string())),
        ("is_description_translated", pa.bool_()), ("untranslated_description", pa.string()),
        ("time_limit", pa.struct([("seconds", pa.int64()), ("nanos", pa.int64())])),
        ("memory_limit_bytes", pa.int64()), ("input_file", pa.string()), ("output_file", pa.string()),
    ])


def _codeforces_schema():
    pa = _pa()
    case = pa.struct([("input", pa.string()), ("output", pa.string())])
    return pa.schema([
        ("id", pa.string()), ("aliases", pa.list_(pa.string())), ("contest_id", pa.string()),
        ("title", pa.string()), ("description", pa.string()), ("input_format", pa.string()),
        ("output_format", pa.string()), ("interaction_format", pa.string()), ("note", pa.string()),
        ("examples", pa.list_(case)), ("editorial", pa.string()), ("rating", pa.int64()),
        ("official_tests", pa.list_(case)), ("official_tests_complete", pa.bool_()), ("input_mode", pa.string()),
        ("generated_checker", pa.string()), ("executable", pa.bool_()),
    ])


def _taco_row(tag: str, *, question: str | None = None, cases: list[dict] | None = None, **extra) -> dict:
    cases = _cases(tag) if cases is None else cases
    spec = {"inputs": [c["input"] for c in cases], "outputs": [c["output"] for c in cases]}
    row = {"question": question or _text(tag), "solutions": json.dumps([]), "starter_code": "",
           "input_output": json.dumps(spec), "difficulty": "MEDIUM", "source": "codeforces", "url": "",
           "picture_num": "0"}
    row.update(extra)
    return row


def _struct(cases: list[dict]) -> dict:
    return {"input": [c["input"] for c in cases], "output": [c["output"] for c in cases]}


def _cc_row(tag: str, *, description: str | None = None, cases: list[dict] | None = None, cf=(0, ""),
            python3: tuple[str, ...] = (), **extra) -> dict:
    cases = _cases(tag) if cases is None else cases
    row = {"name": tag, "description": description or _text(tag), "public_tests": _struct(cases[:1]),
           "private_tests": _struct(cases[1:2]), "generated_tests": _struct(cases[2:]), "difficulty": 8,
           "solutions": {"language": [3] * len(python3), "solution": list(python3)},
           "cf_contest_id": cf[0], "cf_index": cf[1], "cf_rating": 0, "input_file": "", "output_file": ""}
    row.update(extra)
    return row


def _cf_row(problem_id: str, tag: str, *, cases: list[dict] | None = None, **extra) -> dict:
    cases = _cases(tag) if cases is None else cases
    row = {"id": problem_id, "aliases": [], "title": f"Title {tag}", "description": _text(tag),
           "input_format": f"The input of {tag}.", "output_format": f"The output of {tag}.",
           "interaction_format": None, "note": None, "examples": cases[:1], "rating": 1200,
           "official_tests": cases, "official_tests_complete": True, "input_mode": "stdio",
           "generated_checker": None, "executable": True}
    row.update(extra)
    return row


def _lcb_private(cases: list[dict]) -> str:
    """LCB's encoding of private tests: base64 of zlib of a pickled JSON string."""
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode("ascii")


def _lcb_row(question_id: str, platform: str, day: str, content: str, cases: list[dict]) -> dict:
    typed = [dict(case, testtype="stdin") for case in cases]
    return {
        "question_title": question_id, "question_content": content, "platform": platform,
        "question_id": question_id, "contest_id": question_id.split("_")[0], "contest_date": f"{day}T00:00:00",
        "starter_code": "", "difficulty": "medium", "public_test_cases": json.dumps(typed[:1]),
        "private_test_cases": _lcb_private(typed[1:]), "metadata": "{}",
    }


def _write_sources(root: Path, *, codeforces=(), code_contests=(), taco=(), taco_split_at=None) -> dict[str, Path]:
    """Writes the given rows in the Hugging Face cache layout and returns the dataset dirs to pass to build."""
    dirs = {}
    if codeforces:
        dirs["codeforces"] = root / "open-r1___codeforces"
        _write_arrow(dirs["codeforces"] / "default" / "0.0.0" / "fbe3f6e9" / "codeforces-train-00000-of-00001.arrow",
                     list(codeforces), _codeforces_schema())
    if code_contests:
        dirs["code_contests"] = root / "deepmind___code_contests"
        _write_arrow(dirs["code_contests"] / "default" / "0.0.0" / "802411c3" /
                     "code_contests-train-00000-of-00001.arrow", list(code_contests), _code_contests_schema())
    if taco:
        dirs["taco"] = root / "BAAI___taco"
        shard_dir = dirs["taco"] / "ALL" / "0.0.0" / "6c79de44"
        rows = list(taco)
        split = len(rows) // 2 if taco_split_at is None else taco_split_at
        _write_arrow(shard_dir / "taco-train-00000-of-00002.arrow", rows[:split], _taco_schema())
        _write_arrow(shard_dir / "taco-train-00001-of-00002.arrow", rows[split:], _taco_schema())
    return dirs


def _write_lcb(root: Path, rows: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "test6.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return root


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _check_contract(out: Path) -> tuple[list[dict], list[dict], dict]:
    """Checks what every dataset must satisfy; returns the train rows, the val rows and the report."""
    train, val = _read_jsonl(out / cd.TRAIN_FILE), _read_jsonl(out / cd.VAL_FILE)
    report = json.loads((out / cd.REPORT_FILE).read_text(encoding="utf-8"))
    assert report["kind"] == cd.REPORT_KIND
    assert report["rows"] == {"train": len(train), "val": len(val)}
    for row in train + val:
        assert set(row) == {"instruction", "response", "source", "id", "difficulty", "n_cases", "validated"}
        assert row["instruction"] == row["instruction"].strip() and row["instruction"]
        # The response is a JSON string, and the reward reads exactly these four keys from it.
        assert isinstance(row["response"], str)
        truth = json.loads(row["response"])
        assert list(truth) == ["kind", "tests", "judge_imports", "float_tolerance"]
        assert truth["kind"] == "stdin" and truth["judge_imports"] is True and truth["float_tolerance"] == 1e-6
        assert len(truth["tests"]) == row["n_cases"] >= report["settings"]["min_cases"]
        assert row["n_cases"] <= report["settings"]["max_cases"]
        for case in truth["tests"]:
            assert set(case) == {"input", "output"}
            assert isinstance(case["input"], str) and isinstance(case["output"], str)
            assert "\r\n" not in case["input"] + case["output"]
        assert cd.tests_json_bytes(truth["tests"]) <= report["settings"]["max_test_bytes"]
    ids = [row["id"] for row in train + val]
    assert len(ids) == len(set(ids))
    return train, val, report


# ----------------------------------------------------------------------------------------------------------
# Tests and ids
# ----------------------------------------------------------------------------------------------------------

def test_select_tests_keeps_public_cases_first_and_skips_repeated_inputs():
    public = [{"input": "1 2\r\n", "output": "3\r\n"}]
    private = [
        {"input": "1 2  \n\n", "output": "3"},  # the same input with trailing spaces and a blank line
        {"input": "5 5\n", "output": "10\n"},
        {"input": "2 2\n", "output": "4\n"},
    ]
    tests = cd.select_tests([public, private], CAPS)
    assert tests == [
        {"input": "1 2\n", "output": "3\n"},
        {"input": "5 5\n", "output": "10\n"},
        {"input": "2 2\n", "output": "4\n"},
    ]


def test_select_tests_caps_count_and_bytes():
    cases = _cases("cap", 10)
    assert cd.select_tests([cases], CaseCaps(max_cases=4, min_cases=3)) == cases[:4]
    # The byte cap drops trailing cases until the JSON of the tests fits, and it counts exactly.
    budget = cd.tests_json_bytes(cases[:5])
    kept = cd.select_tests([cases], CaseCaps(max_cases=10, min_cases=3, max_test_bytes=budget))
    assert kept == cases[:5]
    assert cd.select_tests([cases], CaseCaps(max_cases=10, min_cases=3, max_test_bytes=budget - 1)) == cases[:4]
    assert len(json.dumps(kept, ensure_ascii=False).encode("utf-8")) == budget
    # A problem that has enough cases but loses them to the byte cap is told apart from one that never had them.
    tiny = cd.tests_json_bytes(cases[:2])
    assert cd.select_tests([cases], CaseCaps(max_cases=10, min_cases=3, max_test_bytes=tiny)) == "tests_too_large"
    assert cd.select_tests([cases[:2]], CAPS) == "too_few_cases"
    assert cd.select_tests([[]], CAPS) == "too_few_cases"


def test_select_tests_drop_reasons():
    assert cd.select_tests([[{"input": "1\n", "output": "1\n"}, {"input": "1\n", "output": "2\n"}]], CAPS) == (
        "conflicting_tests"
    )
    # Outputs that the reward treats as equal are not a conflict.
    same = [{"input": "1\n", "output": "0.5\n"}, {"input": "1\n", "output": "0.50000\n"}] + _cases("ok", 3)
    assert len(cd.select_tests([same], CAPS)) == 4
    constant = [{"input": f"{i}\n", "output": "YES\n"} for i in range(5)]
    assert cd.select_tests([constant], CAPS) == "constant_output"
    assert cd.select_tests([[{"input": ["1"], "output": "1"}]], CAPS) == "non_string_tests"
    assert cd.select_tests([["1 2"]], CAPS) == "bad_tests"


def test_ids_from_urls_and_lcb_question_ids():
    assert cd.ids_from_url("https://codeforces.com/problemset/problem/1234/a") == (("1234/A",), None)
    assert cd.ids_from_url("https://codeforces.com/contest/1234/problem/E1") == (("1234/E1",), None)
    assert cd.ids_from_url("https://atcoder.jp/contests/abc301/tasks/abc301_e") == ((), "abc301_e")
    assert cd.ids_from_url("https://www.codechef.com/problems/XYZ") == ((), None)
    assert cd.ids_from_url(None) == ((), None)
    assert cd.codeforces_id_from_text("2063/A") == "2063/A"
    assert cd.codeforces_id_from_text("1873_a") == "1873/A"
    assert cd.codeforces_id_from_text("abc301_e") is None
    assert cd.codeforces_id(0, "A") is None and cd.codeforces_id(12, "") is None
    assert cd.lcb_eval_ids("1873_A", "codeforces") == ("1873/A", None)
    assert cd.lcb_eval_ids("ABC301_E", "atcoder") == (None, "abc301_e")
    assert cd.lcb_eval_ids("two-sum", "leetcode") == (None, None)


def test_codeforces_statement_layout():
    row = _cf_row("1/A", "s", cases=[{"input": "1 2\r\n", "output": "3\r\n"}, {"input": "4 5\n", "output": "9\n"}])
    row.update(description="  Add two numbers.\r\n", input_format="Two integers.", output_format="Their sum.",
               note="The sum of 1 and 2 is 3.", examples=row["official_tests"])
    assert cd.codeforces_statement(row) == (
        "Add two numbers.\n\nInput\n\nTwo integers.\n\nOutput\n\nTheir sum.\n\nExamples\n\n"
        "Input\n\n1 2\n\nOutput\n\n3\n\nInput\n\n4 5\n\nOutput\n\n9\n\nNote\n\nThe sum of 1 and 2 is 3."
    )
    row.update(note=None, examples=[])
    assert cd.codeforces_statement(row) == "Add two numbers.\n\nInput\n\nTwo integers.\n\nOutput\n\nTheir sum."


def test_row_converters_apply_the_source_filters():
    taco = cd.taco_problem
    assert taco(_taco_row("a", starter_code="def f(x):\n"), 0, CAPS, False) == "not_stdin"
    function_call = _taco_row("a")
    function_call["input_output"] = json.dumps({"inputs": [[1]], "outputs": [[1]], "fn_name": "f"})
    assert taco(function_call, 0, CAPS, False) == "not_stdin"
    assert taco(_taco_row("a", picture_num="2"), 0, CAPS, False) == "has_images"
    assert taco(_taco_row("a", input_output="not json"), 0, CAPS, False) == "bad_tests"
    lists = _taco_row("a", cases=_cases("a", 3) + [{"input": ["1"], "output": "x"}])
    assert taco(lists, 0, CAPS, False) == "non_string_tests"
    good = taco(_taco_row("a", url="https://atcoder.jp/contests/abc1/tasks/abc1_a",
                          solutions=json.dumps(["print(1)\n" * 3, "print(1)", "", "print(1)", "x = 1\n" * 9])),
                7, CAPS, True)
    assert isinstance(good, Problem) and good.id == "taco/7" and good.atcoder_id == "abc1_a"
    # Distinct non-empty references, the shortest first, at most three.
    assert good.solutions == ["print(1)", "print(1)\n" * 3, "x = 1\n" * 9]

    cc = cd.code_contests_problem
    assert cc(_cc_row("b", input_file="in.txt"), 0, CAPS, False) == "file_io"
    assert cc(_cc_row("b", description="Look at the picture <image> below."), 0, CAPS, False) == "has_images"
    problem = cc(_cc_row("b", cf=(100, "a"), python3=["print(2)", "print(1)"], cf_rating=1500), 3, CAPS, True)
    assert problem.cf_ids == ("100/A",) and problem.difficulty == "1500" and problem.id == "code_contests/3"
    # Public cases come first, then private, then generated.
    assert problem.tests == _cases("b")
    python2 = _cc_row("b")
    python2["solutions"] = {"language": [1, 3, 2], "solution": ["print 1", "print(1)", "int main(){}"]}
    assert cc(python2, 0, CAPS, True).solutions == ["print(1)"]
    assert cc(_cc_row("b"), 0, CAPS, True).difficulty == "B"

    cf = cd.codeforces_problem
    assert cf(_cf_row("1/A", "c", interaction_format="Interact with the judge."), 0, CAPS, False) == "interactive"
    assert cf(_cf_row("1/A", "c", generated_checker="def check(): ..."), 0, CAPS, False) == "needs_checker"
    assert cf(_cf_row("1/A", "c", official_tests_complete=False), 0, CAPS, False) == "incomplete_tests"
    assert cf(_cf_row("1/A", "c", input_mode="file"), 0, CAPS, False) == "not_stdin"
    crlf = [{"input": f"{i}\r\n", "output": f"{i * 2}\r\n"} for i in range(4)]
    problem = cf(_cf_row("2/B", "c", cases=crlf, aliases=["3/D"]), 0, CAPS, False)
    assert problem.id == "codeforces/2/B" and problem.cf_ids == ("2/B", "3/D") and problem.difficulty == "1200"
    assert problem.tests == [{"input": f"{i}\n", "output": f"{i * 2}\n"} for i in range(4)]


def _pi_row(tag: str, *, cases: list[dict] | None = None, solutions: tuple[str, ...] = ("print(1)",), **extra) -> dict:
    """A PrimeIntellect row of DeepCoder-Preview, with its statement wrapper and fenced reference answers."""
    cases = _cases(tag) if cases is None else cases
    problem = ("Solve the following coding problem using the programming language python:\n\n" + _text(tag)
               + "\n\nThe input will be given via stdin and the output should be printed to stdout by your code."
               "\n\nNow solve the problem by providing the code.")
    row = {"problem": problem, "tests": json.dumps([{"type": "stdin_stdout", **case} for case in cases]),
           "solutions": [f"```python\n{code}\n```" for code in solutions]}
    row.update(extra)
    return row


def test_primeintellect_rows():
    pi = cd.primeintellect_problem
    problem = pi(_pi_row("p", solutions=("print(22)", "print(1)")), 5, CAPS, True)
    # The wrapper of the dataset is cut off, so only the problem itself reaches our prompt.
    assert isinstance(problem, Problem) and problem.statement == _text("p") and problem.id == "primeintellect/5"
    assert problem.tests == _cases("p")
    # References are cut out of their markdown fences, the shortest first.
    assert problem.solutions == ["print(1)\n", "print(22)\n"]
    calls = _pi_row("p")
    calls["tests"] = json.dumps([{"type": "function_call", "fn_name": "f", "input": [1], "output": [1]}])
    assert pi(calls, 0, CAPS, False) == "not_stdin"
    assert pi(_pi_row("p", tests="nope"), 0, CAPS, False) == "bad_tests"
    assert pi(_pi_row("p", cases=_cases("p", 2)), 0, CAPS, False) == "too_few_cases"


def test_parquet_reader_and_files(tmp_path):
    pa = _pa()
    import pyarrow.parquet as pq

    root = tmp_path / "primeintellect"
    root.mkdir()
    rows = [_pi_row(f"r{i}") for i in range(5)]
    for shard in range(2):
        pq.write_table(pa.Table.from_pylist(rows[shard::2]), root / f"train-0000{shard}-of-00002.parquet")
    (root / "README.md").write_text("not data")
    files = cd.find_parquet_files(root)
    assert [path.name for path in files] == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    read = [row for path in files for row in cd.iter_parquet_rows(path, ("problem", "tests"), ("missing",))]
    assert len(read) == 5 and all(row["missing"] is None for row in read)
    problems, record = cd.read_source(cd.PRIMEINTELLECT, root, CAPS, with_solutions=True)
    assert len(problems) == 5 and record["rows_read"] == 5 and problems[0].solutions == ["print(1)\n"]
    with pytest.raises(ValueError):
        list(cd.iter_parquet_rows(files[0], ("no_such_column",)))


def test_dedup_priority_and_statements():
    def make(source, order, statement, cf_ids=(), atcoder_id=None):
        return Problem(source=source, id=f"{source}/{order}", statement=statement, difficulty="",
                       tests=_cases("x"), cf_ids=cf_ids, atcoder_id=atcoder_id, order=order)

    taco = make("taco", 0, "t0", cf_ids=("1/A",))
    contests = make("code_contests", 5, "c5", cf_ids=("1/A",))
    codeforces = make("codeforces", 9, "f9", cf_ids=("2/A", "1/A"))
    first_atcoder = make("taco", 1, "t1", atcoder_id="abc1_a")
    second_atcoder = make("taco", 2, "t2", atcoder_id="abc1_a")
    kept, dropped = cd.dedup_by_id([taco, contests, first_atcoder, codeforces, second_atcoder])
    assert [p.id for p in kept] == ["codeforces/9", "taco/1"]
    assert {p.id for p in dropped} == {"taco/0", "code_contests/5", "taco/2"}

    upper = make("taco", 3, "  ADD   two\nNumbers ")
    lower = make("code_contests", 4, "add two numbers")
    other = make("taco", 4, "add three numbers")
    kept, dropped = cd.dedup_by_statement([upper, other, lower])
    assert [p.id for p in kept] == ["code_contests/4", "taco/4"] and [p.id for p in dropped] == ["taco/3"]


def test_ngram_share_and_decontamination():
    lcb_text = _text("lcb", 40)
    index = cd.EvalIndex()
    index.add("1873_A", "codeforces", "unrelated words " * 3)
    index.add("abc301_e", "atcoder", lcb_text)
    index.add("two-sum", "leetcode", "")
    assert index.problems == 3 and index.cf_ids == {"1873/A"} and index.atcoder_ids == {"abc301_e"}

    heavy = " ".join(lcb_text.split()[:30] + _text("own", 10).split())  # 18 of 28 13-grams come from LCB
    light = " ".join(lcb_text.split()[:20] + _text("own", 30).split())  # 8 of 38
    assert cd.ngram_share(heavy, index.ngrams) == pytest.approx(18 / 28)
    assert cd.ngram_share(light, index.ngrams) == pytest.approx(8 / 38)
    assert cd.ngram_share("too short to have a single thirteen gram", index.ngrams) == 0.0

    def make(order, statement, cf_ids=(), atcoder_id=None):
        return Problem(source="taco", id=f"taco/{order}", statement=statement, difficulty="", tests=_cases("x"),
                       cf_ids=cf_ids, atcoder_id=atcoder_id, order=order)

    problems = [make(0, _text("a"), cf_ids=("1873/A",)), make(1, _text("b"), atcoder_id="abc301_e"),
                make(2, heavy), make(3, light), make(4, _text("c"))]
    kept, dropped = cd.decontaminate(problems, index)
    assert [p.id for p in kept] == ["taco/3", "taco/4"]
    assert [(p.id, reason) for p, reason in dropped] == [("taco/0", "lcb_id"), ("taco/1", "lcb_id"),
                                                         ("taco/2", "lcb_ngram")]


# ----------------------------------------------------------------------------------------------------------
# Arrow reading
# ----------------------------------------------------------------------------------------------------------

def test_arrow_reader_reads_streams_and_files_and_checks_columns(tmp_path):
    rows = [_taco_row(f"r{i}") for i in range(5)]
    stream = _write_arrow(tmp_path / "s" / "taco-train-00000-of-00001.arrow", rows, _taco_schema())
    as_file = _write_arrow(tmp_path / "f" / "taco-train-00000-of-00001.arrow", rows, _taco_schema(), stream=False)
    for path in (stream, as_file):
        read = list(cd.iter_arrow_rows(path, ("question", "input_output"), ("url", "no_such_column")))
        assert [row["question"] for row in read] == [row["question"] for row in rows]
        assert set(read[0]) == {"question", "input_output", "url", "no_such_column"}
        assert read[0]["no_such_column"] is None
    with pytest.raises(ValueError, match="no column"):
        list(cd.iter_arrow_rows(stream, ("question", "fn_name")))
    not_arrow = tmp_path / "n" / "taco-train-00000-of-00001.arrow"
    not_arrow.parent.mkdir()
    not_arrow.write_bytes(b"this is not arrow data at all")
    with pytest.raises(ValueError, match="neither an Arrow stream nor an Arrow file"):
        list(cd.iter_arrow_rows(not_arrow, ("question",)))


def test_find_arrow_files(tmp_path):
    root = tmp_path / "open-r1___codeforces"
    one = root / "default" / "0.0.0" / "aaa"
    one.mkdir(parents=True)
    for name in ("codeforces-train-00001-of-00002.arrow", "codeforces-train-00000-of-00002.arrow",
                 "codeforces-test-00000-of-00001.arrow", "codeforces-train-00000-of-00002.arrow.lock"):
        (one / name).write_bytes(b"")
    assert [p.name for p in cd.find_arrow_files(root, "codeforces")] == [
        "codeforces-train-00000-of-00002.arrow", "codeforces-train-00001-of-00002.arrow"]
    # A single shard is named without the shard suffix.
    single = tmp_path / "single"
    single.mkdir()
    (single / "codeforces-train.arrow").write_bytes(b"")
    assert [p.name for p in cd.find_arrow_files(single, "codeforces")] == ["codeforces-train.arrow"]
    # Two cached revisions would make every problem count twice, so the reader refuses them.
    two = root / "verifiable" / "0.0.0" / "bbb"
    two.mkdir(parents=True)
    (two / "codeforces-train-00000-of-00001.arrow").write_bytes(b"")
    with pytest.raises(ValueError, match="2 directories"):
        cd.find_arrow_files(root, "codeforces")
    with pytest.raises(FileNotFoundError):
        cd.find_arrow_files(tmp_path / "missing", "taco")
    with pytest.raises(FileNotFoundError):
        cd.find_arrow_files(one, "taco")


def test_lcb_test_groups_decodes_private_cases():
    cases = _cases("l", 3)
    row = _lcb_row("abc1_a", "atcoder", "2025-03-01", "text", cases)
    public, private = cd.lcb_test_groups(row)
    assert [c["input"] for c in public + private] == [c["input"] for c in cases]
    row["private_test_cases"] = json.dumps(cases[1:])  # some releases store them as plain JSON
    assert cd.lcb_test_groups(row)[1] == cases[1:]


# ----------------------------------------------------------------------------------------------------------
# Whole builds
# ----------------------------------------------------------------------------------------------------------

LCB_NGRAM_TEXT = _text("lcbngram", 40)


def _scenario(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    """Every filter, dedup rule and decontamination rule once. The comments name the expected outcome."""
    shared = "Print the sum of two integers given on one line, then print their product on the next line."
    codeforces = [
        _cf_row("100/A", "f0"),  # kept
        _cf_row("101/B", "f1", interaction_format="Ask queries."),  # interactive
        _cf_row("102/C", "f2", generated_checker="checker source"),  # needs_checker
        _cf_row("103/D", "f3", official_tests_complete=False),  # incomplete_tests
        _cf_row("104/E", "f4", input_mode="file"),  # not_stdin
        _cf_row("1873/A", "f5"),  # lcb_id
        _cf_row("105/F", "f6", aliases=["106/D"]),  # kept
    ]
    code_contests = [
        _cc_row("c0", cf=(100, "A")),  # duplicate_id: open-r1 has 100/A
        _cc_row("c1"),  # kept
        _cc_row("c2", output_file="output.txt"),  # file_io
        _cc_row("c3", description="See <image> for the layout."),  # has_images
        _cc_row("c4", cases=_cases("c4", 2)),  # too_few_cases
        _cc_row("c5", description=shared),  # kept, and TACO's copy of this statement is dropped
        _cc_row("c6", cases=[{"input": f"{i}\n", "output": "NO\n"} for i in range(4)]),  # constant_output
    ]
    function_call = _taco_row("t0")
    function_call["input_output"] = json.dumps({"inputs": [[1]], "outputs": [[1]], "fn_name": "f"})
    taco = [
        function_call,  # not_stdin
        _taco_row("t1", starter_code="class Solution:\n"),  # not_stdin
        _taco_row("t2", picture_num="2"),  # has_images
        _taco_row("t3", url="https://codeforces.com/problemset/problem/100/A"),  # duplicate_id
        _taco_row("t4", url="https://codeforces.com/contest/106/problem/D"),  # duplicate_id through an alias
        _taco_row("t5", question="  " + shared.upper().replace(" ", "   ") + "\n"),  # duplicate_statement
        _taco_row("t6", url="https://atcoder.jp/contests/abc301/tasks/abc301_e"),  # lcb_id
        _taco_row("t7", question=" ".join(LCB_NGRAM_TEXT.split()[:30] + _text("t7", 10).split())),  # lcb_ngram
        _taco_row("t8", question=" ".join(LCB_NGRAM_TEXT.split()[:20] + _text("t8", 30).split())),  # kept
        _taco_row("t9"),  # kept
        _taco_row("t10", cases=_cases("t10", 3) + [{"input": ["x"], "output": "y"}]),  # non_string_tests
        _taco_row("t11", url="https://atcoder.jp/contests/abc200/tasks/abc200_a"),  # kept
        _taco_row("t12", url="https://atcoder.jp/contests/abc200/tasks/abc200_a"),  # duplicate_id within TACO
    ]
    lcb = _write_lcb(tmp_path / "lcb", [
        _lcb_row("1873_A", "codeforces", "2023-09-21", _text("l0"), _cases("l0")),
        _lcb_row("abc301_e", "atcoder", "2023-05-13", _text("l1"), _cases("l1")),
        _lcb_row("abc390_c", "atcoder", "2025-02-08", _text("l2"), _cases("l2")),  # val
        _lcb_row("abc391_d", "atcoder", "2025-03-01", _text("l3"), _cases("l3", 6)),  # val
        _lcb_row("2063_B", "codeforces", "2025-01-20", _text("l4"), _cases("l4")),  # before --val-from
        _lcb_row("lc_1", "leetcode", "2025-05-01", _text("l5"), _cases("l5")),  # LeetCode is never val
        _lcb_row("abc395_a", "atcoder", "2025-04-01", _text("l6"), _cases("l6", 2)),  # too few cases for val
        _lcb_row("abc302_b", "atcoder", "2023-05-20", LCB_NGRAM_TEXT, _cases("l7")),
    ])
    dirs = _write_sources(tmp_path / "hf", codeforces=codeforces, code_contests=code_contests, taco=taco)
    return dirs, lcb


def test_build_end_to_end(tmp_path):
    dirs, lcb = _scenario(tmp_path)
    out = tmp_path / "out" / "code_v1"
    logged = []
    report = cd.build(out, lcb=lcb, log=logged.append, **dirs)
    train, val, written = _check_contract(out)
    assert report["rows"] == written["rows"]

    assert sorted(row["id"] for row in train) == sorted([
        "codeforces/100/A", "codeforces/105/F", "code_contests/1", "code_contests/5", "taco/8", "taco/9",
        "taco/11",
    ])
    assert sorted(row["id"] for row in val) == ["lcb/abc390_c", "lcb/abc391_d"]
    assert written["val_source"] == "lcb"
    assert all(row["validated"] is False for row in train + val)

    sources = written["sources"]
    assert sources["codeforces"]["rows_read"] == 7 and sources["code_contests"]["rows_read"] == 7
    assert sources["taco"]["rows_read"] == 13
    assert sources["codeforces"]["dropped"] == {"interactive": 1, "needs_checker": 1, "incomplete_tests": 1,
                                                "not_stdin": 1, "lcb_id": 1}
    assert sources["code_contests"]["dropped"] == {"duplicate_id": 1, "file_io": 1, "has_images": 1,
                                                   "too_few_cases": 1, "constant_output": 1}
    assert sources["taco"]["dropped"] == {"not_stdin": 2, "has_images": 1, "duplicate_id": 3,
                                          "duplicate_statement": 1, "lcb_id": 1, "lcb_ngram": 1,
                                          "non_string_tests": 1}
    assert {name: sources[name]["train"] for name in cd.TRAIN_SOURCES if name in sources} == {
        "codeforces": 2, "code_contests": 2, "taco": 3}
    assert sources["lcb"]["problems"] == 8 and sources["lcb"]["val_candidates"] == 3
    assert sources["lcb"]["val_dropped"] == {"too_few_cases": 1} and sources["lcb"]["val"] == 2
    assert written["dropped"]["duplicate_id"] == 4 and written["dropped"]["lcb_id"] == 2
    assert set(written["timings"]) >= {"lcb_s", "dedup_s", "decontaminate_s", "total_s", "read_taco_s"}
    assert written["settings"]["max_cases"] == 30 and written["settings"]["validate"] is False

    by_id = {row["id"]: row for row in train + val}
    # The open-r1 statement is built from its parts, and its examples come first among the tests.
    statement = by_id["codeforces/100/A"]["instruction"]
    assert statement.startswith(_text("f0") + "\n\nInput\n\nThe input of f0.") and "\n\nExamples\n\n" in statement
    assert json.loads(by_id["codeforces/100/A"]["response"])["tests"] == _cases("f0")
    assert by_id["code_contests/5"]["instruction"].startswith("Print the sum")
    assert by_id["codeforces/105/F"]["difficulty"] == "1200" and by_id["taco/9"]["difficulty"] == "MEDIUM"
    # An LCB validation problem keeps its public case first and its private cases after it.
    assert json.loads(by_id["lcb/abc391_d"]["response"])["tests"] == _cases("l3", 6)
    assert by_id["lcb/abc391_d"]["instruction"] == _text("l3") and by_id["lcb/abc391_d"]["source"] == "lcb"
    assert any("lcb:" in line for line in logged)


def test_build_is_deterministic_and_the_seed_shuffles(tmp_path):
    dirs, lcb = _scenario(tmp_path)
    first = cd.build(tmp_path / "a", lcb=lcb, seed=3, log=lambda _: None, **dirs)
    second = cd.build(tmp_path / "b", lcb=lcb, seed=3, log=lambda _: None, **dirs)
    assert first["rows"] == second["rows"]
    assert (tmp_path / "a" / cd.TRAIN_FILE).read_bytes() == (tmp_path / "b" / cd.TRAIN_FILE).read_bytes()
    orders = set()
    for seed in range(6):
        cd.build(tmp_path / f"s{seed}", lcb=lcb, seed=seed, log=lambda _: None, **dirs)
        orders.add(tuple(row["id"] for row in _read_jsonl(tmp_path / f"s{seed}" / cd.TRAIN_FILE)))
    assert len(orders) > 1


def test_val_from_and_val_size_pick_the_lcb_problems(tmp_path):
    dirs, lcb = _scenario(tmp_path)
    cd.build(tmp_path / "late", lcb=lcb, val_from="2025-03-01", log=lambda _: None, **dirs)
    assert [row["id"] for row in _read_jsonl(tmp_path / "late" / cd.VAL_FILE)] == ["lcb/abc391_d"]
    cd.build(tmp_path / "one", lcb=lcb, val_size=1, log=lambda _: None, **dirs)
    assert len(_read_jsonl(tmp_path / "one" / cd.VAL_FILE)) == 1
    # Problems from before --val-from are still evaluation problems: the train copy of 1873/A stays out.
    train_ids = {row["id"] for row in _read_jsonl(tmp_path / "late" / cd.TRAIN_FILE)}
    assert "codeforces/1873/A" not in train_ids
    with pytest.raises(ValueError, match="earlier --val-from"):
        cd.build(tmp_path / "none", lcb=lcb, val_from="2030-01-01", log=lambda _: None, **dirs)


def test_without_lcb_val_is_held_out_from_train(tmp_path):
    dirs = _write_sources(tmp_path / "hf", taco=[_taco_row(f"h{i}") for i in range(6)])
    report = cd.build(tmp_path / "out", val_size=2, log=lambda _: None, **dirs)
    train, val, written = _check_contract(tmp_path / "out")
    assert report["val_source"] == written["val_source"] == "train_holdout"
    assert "No --lcb" in written["val_note"]
    assert len(train) == 4 and len(val) == 2
    assert {row["id"] for row in train} | {row["id"] for row in val} == {f"taco/{i}" for i in range(6)}
    # Row numbers continue across the two TACO shards, so an id points at one row of the dataset.
    assert written["sources"]["taco"]["files"] == ["taco-train-00000-of-00002.arrow",
                                                   "taco-train-00001-of-00002.arrow"]


def test_limit_per_source_and_argument_checks(tmp_path):
    dirs, lcb = _scenario(tmp_path)
    report = cd.build(tmp_path / "out", lcb=lcb, limit_per_source=2, val_size=1, log=lambda _: None,
                      codeforces=dirs["codeforces"], taco=dirs["taco"])
    assert report["sources"]["codeforces"]["rows_read"] == 2 and report["sources"]["taco"]["rows_read"] == 2
    assert report["sources"]["lcb"]["problems"] == 8
    with pytest.raises(ValueError, match="at least one train source"):
        cd.build(tmp_path / "x", lcb=lcb)
    with pytest.raises(ValueError, match="not a date"):
        cd.build(tmp_path / "x", taco=dirs["taco"], val_from="Feb 2025")
    with pytest.raises(ValueError, match="--max-cases"):
        cd.build(tmp_path / "x", taco=dirs["taco"], min_cases=5, max_cases=4)


class _FakeRun:
    """Stands in for code_sandbox.execute: a program passes when it contains GOOD and raises when it is BOOM."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, program, tests, **kwargs):
        with self.lock:
            self.calls.append((program, tests, kwargs))
        if program == "BOOM":
            raise RuntimeError("the harness broke")
        passed = "GOOD" in program
        return {"passed": passed, "reason": "passed" if passed else "wrong_answer", "cpu_time": 0.05,
                "detail": "" if passed else "case 0: expected '0', got '1'"}


def test_validation_with_a_fake_run(tmp_path):
    dirs = _write_sources(
        tmp_path / "hf",
        codeforces=[_cf_row("7/A", "x0")],
        code_contests=[
            # The Python 2 program would pass but must never run; the shorter Python 3 one fails first.
            dict(_cc_row("v0", python3=["bad", "print('GOOD')"]),
                 solutions={"language": [1, 3, 3], "solution": ["GOOD py2", "bad", "print('GOOD')"]}),
            _cc_row("v1", python3=["bad1", "bad22", "bad333", "bad4444"]),  # all three tried references fail
        ],
        taco=[
            _taco_row("w0", solutions=json.dumps(["GOOD"])),
            _taco_row("w1", solutions=json.dumps([])),  # nothing to check with
            _taco_row("w2", solutions=json.dumps(["BOOM", "GOOD second"])),  # a harness error is not fatal
        ],
        taco_split_at=1,
    )
    run = _FakeRun()
    report = cd.build(tmp_path / "out", validate=True, workers=2, run=run, val_size=1, log=lambda _: None, **dirs)
    train, val, written = _check_contract(tmp_path / "out")
    rows = {row["id"]: row for row in train + val}
    assert set(rows) == {"codeforces/7/A", "code_contests/0", "taco/0", "taco/2"}
    assert rows["codeforces/7/A"]["validated"] is False  # open-r1 has no references to run
    assert all(rows[name]["validated"] is True for name in ("code_contests/0", "taco/0", "taco/2"))

    programs = [program for program, _, _ in run.calls]
    assert "GOOD py2" not in programs and "bad4444" not in programs
    assert programs.count("bad") == 1 and programs.count("print('GOOD')") == 1
    # Validation uses exactly the reward's settings and the tests the dataset keeps.
    assert all(kwargs == REWARD_SETTINGS for _, _, kwargs in run.calls)
    kept_tests = json.loads(rows["code_contests/0"]["response"])["tests"]
    assert [tests for program, tests, _ in run.calls if program == "print('GOOD')"] == [kept_tests]

    cc, taco = written["sources"]["code_contests"], written["sources"]["taco"]
    assert cc["dropped"] == {"validation_failed": 1}
    assert cc["validation"]["checked"] == 2 and cc["validation"]["passed"] == 1 and cc["validation"]["failed"] == 1
    assert cc["validation"]["failure_reasons"] == {"wrong_answer": 3}
    assert cc["validation"]["failed_examples"][0]["id"] == "code_contests/1"
    assert taco["dropped"] == {"no_python_reference": 1}
    assert taco["validation"]["checked"] == 3 and taco["validation"]["passed"] == 2
    assert taco["validation"]["no_reference"] == 1
    assert "validation" not in written["sources"]["codeforces"]
    assert written["settings"]["validate"] is True and written["settings"]["workers"] == 2
    assert written["settings"]["limits"]["time_s"] == 6.0 and "validate_s" in written["timings"]
    assert report["rows"] == {"train": 3, "val": 1}


def test_validate_problem_reports_attempts_and_reasons():
    run = _FakeRun()
    problem = Problem(source="taco", id="taco/0", statement="s", difficulty="", tests=_cases("p"),
                      solutions=["BOOM", "bad", "GOOD"])
    result = cd.validate_problem(problem, run)
    assert result["passed"] and result["attempts"] == 3 and result["reasons"] == ["harness_exception", "wrong_answer"]
    result = cd.validate_problem(Problem(source="taco", id="taco/1", statement="s", difficulty="",
                                         tests=_cases("p"), solutions=["bad"]), run)
    assert not result["passed"] and result["reason"] == "wrong_answer" and result["detail"].startswith("case 0")
    empty = cd.validate_problem(Problem(source="taco", id="taco/2", statement="s", difficulty="", tests=[]), run)
    assert empty["reason"] == "no_python_reference" and empty["attempts"] == 0


# ----------------------------------------------------------------------------------------------------------
# Output directory handling and the CLI
# ----------------------------------------------------------------------------------------------------------

def test_force_staging_and_foreign_dirs(tmp_path, monkeypatch):
    dirs = _write_sources(tmp_path / "hf", taco=[_taco_row(f"g{i}") for i in range(4)])
    quiet = dict(val_size=1, log=lambda _: None, **dirs)
    out = tmp_path / "data" / "code"
    cd.build(out, **quiet)
    first = (out / cd.REPORT_FILE).read_text()
    with pytest.raises(FileExistsError, match="--force"):
        cd.build(out, **quiet)
    cd.build(out, force=True, seed=1, **quiet)
    assert (out / cd.REPORT_FILE).read_text() != first

    # A failure while writing leaves the old dataset in place and no staging dir behind.
    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    before = (out / cd.TRAIN_FILE).read_bytes()
    monkeypatch.setattr(cd.common, "write_manifest", broken)
    with pytest.raises(RuntimeError, match="disk full"):
        cd.build(out, force=True, seed=2, **quiet)
    with pytest.raises(RuntimeError, match="disk full"):
        cd.build(tmp_path / "data" / "fresh", **quiet)
    monkeypatch.undo()
    assert (out / cd.TRAIN_FILE).read_bytes() == before
    assert not (tmp_path / "data" / "fresh").exists()
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["code"]

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="remove it by hand"):
        cd.build(foreign, force=True, **quiet)
    assert (foreign / "notes.txt").read_text() == "keep me"
    empty = tmp_path / "empty"
    empty.mkdir()
    cd.build(empty, **quiet)
    assert (empty / cd.REPORT_FILE).is_file()
    a_file = tmp_path / "a_file"
    a_file.write_text("x")
    with pytest.raises(FileExistsError, match="not a directory"):
        cd.build(a_file, force=True, **quiet)


def test_cli(tmp_path, capsys):
    dirs, lcb = _scenario(tmp_path)
    out = tmp_path / "cli"
    argv = ["build", "--out-dir", str(out), "--taco", str(dirs["taco"]), "--code-contests",
            str(dirs["code_contests"]), "--codeforces", str(dirs["codeforces"]), "--lcb", str(lcb),
            "--max-cases", "5", "--seed", "1"]
    assert cd.main(argv) == 0
    printed = capsys.readouterr().out
    assert "rows: train 7, val 2 (val from lcb)" in printed and "taco: train" in printed
    _, _, written = _check_contract(out)
    assert written["settings"]["max_cases"] == 5 and written["settings"]["seed"] == 1
    assert cd.main(argv) == 2
    assert "--force" in capsys.readouterr().err
    assert cd.main(["build", "--out-dir", str(tmp_path / "x")]) == 2
    assert "at least one train source" in capsys.readouterr().err


# ----------------------------------------------------------------------------------------------------------
# The real sandbox
# ----------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the sandbox needs Linux")
def test_validation_in_the_real_sandbox():
    tests = [{"input": f"{a} {b}\n", "output": f"{a + b}\n"} for a, b in ((1, 2), (5, 7), (10, 20))]
    correct = "a, b = map(int, input().split())\nprint(a + b)\n"
    wrong = "a, b = map(int, input().split())\nprint(a - b)\n"

    def problem(solutions, cases=tests):
        return Problem(source="taco", id="taco/0", statement="s", difficulty="", tests=cases,
                       solutions=cd._references(solutions))

    assert cd.validate_problem(problem([correct]))["passed"]
    failed = cd.validate_problem(problem([wrong]))
    assert not failed["passed"] and failed["reason"] == "wrong_answer"
    # print(0) is shorter, so it runs first and fails; the correct reference then passes.
    both = cd.validate_problem(problem([correct, "print(0)\n"]))
    assert both["passed"] and both["attempts"] == 2 and both["reasons"] == ["wrong_answer"]
    # The reward's settings: gcd comes from the judge prelude, and 1/3 matches a shorter expected value.
    judged = [{"input": f"{n}\n", "output": f"{g}\n{1 / 3:.7f}\n"} for n, g in ((4, 2), (9, 3), (5, 1))]
    assert cd.validate_problem(problem(["n = int(input())\nprint(gcd(n, 6))\nprint(1 / 3)\n"], judged))["passed"]
