"""CPU tests of scripts/verl/data/build_val.py and of the RL validation specs it writes files for.

Run with PYTHONPATH=<repo>/src. The chat tests use a small stand-in for the offline probe's
chat_local.py and a tiny OASST2 parquet, so they need pyarrow but no downloaded data. The LCB test
compares the tool with code_data.build on the synthetic LCB scenario of test_code_data.py. The math
tests write tiny copies of MATH-500, GSM8K and MGSM in the layout of a Hugging Face hub snapshot.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parents[1] / "scripts" / "verl" / "data" / "build_val.py"
REVISION = "rev0"


def _load(name: str, path: Path):
    """Import a file as a module. dataclasses look the module up in sys.modules, so it goes there first."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load("build_val", SCRIPT)

# The parts of chat_local.py that the tool calls, with the same row shape and the same order rules.
FAKE_CHAT_LOCAL = f'''
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OASST2 = ("OpenAssistant/oasst2", "{REVISION}")


def read_trivial_prompts():
    rows = []
    for line in (HERE / "chat_probe_trivial.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        rows.append({{"prompt": entry["prompt"], "itr_lang": entry["lang"], "itr_domain": "trivial",
                      "source": "trivial"}})
    return rows


def select_probe_prompts(messages):
    roots = sorted((m for m in messages if m["parent_id"] is None and m["role"] == "prompter"),
                   key=lambda m: m["message_id"])
    english = [m for m in roots if m["lang"] == "en"][:200]
    russian = [m for m in roots if m["lang"] == "ru"]
    return [{{"prompt": m["text"], "itr_lang": m["lang"], "itr_domain": "chat", "source": "oasst2:" + m["message_id"]}}
            for m in english + russian]
'''


@pytest.fixture
def probe(tmp_path):
    """A probe folder and an OASST2 snapshot with one Russian and one English repeat."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "chat_local.py").write_text(FAKE_CHAT_LOCAL, encoding="utf-8")
    trivial = [{"lang": "en", "prompt": "Hi!"}, {"lang": "ru", "prompt": "Привет!"}, {"lang": "ru", "prompt": "Как дела?"}]
    (tasks / "chat_probe_trivial.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trivial), encoding="utf-8")
    messages = [
        ("m3", None, "prompter", "en", "Explain recursion."),
        ("m1", None, "prompter", "ru", "Привет!"),  # the trivial file has it too
        ("m2", None, "prompter", "en", "Write a poem."),
        ("m4", "m2", "assistant", "en", "A reply, which is not a root prompt."),
        ("m5", None, "prompter", "en", " Write a poem.\n"),  # repeats m2 after stripping
        ("m6", None, "prompter", "de", "Hallo!"),
        ("m7", None, "prompter", "en", "Name three rivers."),
    ]
    table = pa.Table.from_pylist([dict(zip(tool.OASST2_COLUMNS, row)) for row in messages])
    parquet = tmp_path / "snapshots" / REVISION / "data" / "validation.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(table, parquet)
    return tasks / "chat_local.py", parquet


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_chat_files_hold_the_probe_prompts_once(tmp_path, probe):
    chat_local, parquet = probe
    out = tmp_path / "out"
    out.mkdir()
    summary = tool.build_chat(chat_local, parquet, out, english=2)

    russian = _rows(out / tool.RU_FILE)
    assert [(row["instruction"], row["domain"], row["source"]) for row in russian] == [
        ("Привет!", "chat", "oasst2:m1"), ("Как дела?", "trivial", "trivial")]
    english = _rows(out / tool.EN_FILE)
    # Message id order, the repeat m5 dropped before the cap, and no trivial prompt.
    assert [row["source"] for row in english] == ["oasst2:m2", "oasst2:m3"]
    assert all(row["response"] == "" for row in russian + english)
    assert summary["files"][tool.RU_FILE]["repeats_dropped"] == ["Привет!"]
    assert summary["files"][tool.EN_FILE]["repeats_dropped"] == [" Write a poem.\n"]

    # The files are frozen inputs, so a second build must not replace them.
    with pytest.raises(FileExistsError):
        tool.build_chat(chat_local, parquet, out, english=2)


def test_chat_refuses_another_revision_and_too_few_prompts(tmp_path, probe):
    chat_local, parquet = probe
    other = tmp_path / "other.parquet"
    other.write_bytes(parquet.read_bytes())
    with pytest.raises(ValueError, match="revision rev0"):
        tool.build_chat(chat_local, other, tmp_path)
    with pytest.raises(ValueError, match="only 3 distinct English"):
        tool.build_chat(chat_local, parquet, tmp_path, english=4)


def test_rl_val_specs_parse_and_the_ru_probe_counts_twice(tmp_path):
    """The RL runs' validation specs pass distill_data, and a file listed under two tags gives two samples per prompt."""
    from orbit_verl.data import distill_data as dd
    from test_distill_data import CharTokenizer

    files = {name: tmp_path / name for name in ("math500.jsonl", "lcb.jsonl", "mgsm_ru.jsonl", "gsm8k.jsonl",
                                                "chat_ru.jsonl", "chat_en.jsonl", "gold.jsonl")}
    code = json.dumps({"kind": "stdin", "tests": [{"input": "1\n", "output": "1\n"}]})
    contents = {
        "math500.jsonl": [("What is 1+1?", "2"), ("What is 2+2?", "4"), ("What is 3+3?", "6")],
        "lcb.jsonl": [("Print the input.", code)],
        "mgsm_ru.jsonl": [("Сколько будет 2+2?", "4")],
        "gsm8k.jsonl": [("How much is 2+2?", "4")],
        "chat_ru.jsonl": [("Привет!", ""), ("Как дела?", "")],
        "chat_en.jsonl": [("Hello!", "")],
        "gold.jsonl": [("What is 5+5?", "10")],
    }
    for name, rows in contents.items():
        files[name].write_text("".join(json.dumps({"instruction": q, "response": a}, ensure_ascii=False) + "\n"
                                       for q, a in rows), encoding="utf-8")
    specs = [
        f"{files['math500.jsonl']}:orbit/math#think:sft_think",
        f"{files['math500.jsonl']}:orbit/math#no_think:sft_no_think:2",
        f"{files['lcb.jsonl']}:orbit/code#think:sft_think_code",
        f"{files['mgsm_ru.jsonl']}:orbit/math#ru:sft_think_ru",
        f"{files['gsm8k.jsonl']}:orbit/math#mgsm_en:sft_think",
        f"{files['chat_ru.jsonl']}:orbit/none#chat_ru:sft_chat_think",
        f"{files['chat_ru.jsonl']}:orbit/none#chat_ru2:sft_chat_think",
        f"{files['chat_en.jsonl']}:orbit/none#chat_en:sft_chat_think",
    ]
    val = [dd.parse_val_spec(spec) for spec in specs]
    _, records, stats = dd.build(pool=None, gold=[dd.parse_gold_spec(f"{files['gold.jsonl']}:orbit/math:math")],
                                 val=val, tokenizer=CharTokenizer(), max_prompt_length=256, think_share=1.0, seed=0,
                                 limit=None)
    per_source: dict[str, list[str]] = {}
    for record in records:
        per_source.setdefault(record["data_source"], []).append(record["extra_info"]["question"])
    assert per_source["orbit/none#chat_ru"] == per_source["orbit/none#chat_ru2"] == ["Привет!", "Как дела?"]
    assert len(per_source["orbit/math#no_think"]) == 2
    assert {record["extra_info"]["prompt_mode"] for record in records
            if record["data_source"].startswith("orbit/none")} == {"sft_chat_think"}
    assert all(not any(dropped.values()) for dropped in stats["dropped"]["val"])


def test_lcb_sample_equals_the_val_file_of_code_data_build(tmp_path):
    from orbit_verl.data import code_data as cd

    scenario = _load("build_val_code_data_cases", HERE / "test_code_data.py")
    dirs, lcb = scenario._scenario(tmp_path)
    for name, settings in {"all": {}, "late": {"val_from": "2025-03-01"}, "one": {"val_size": 1, "seed": 3}}.items():
        cd.build(tmp_path / name, lcb=lcb, log=lambda _: None, **settings, **dirs)
        out = tmp_path / f"{name}.jsonl"
        tool.build_lcb(lcb, out, val_from=settings.get("val_from", cd.DEFAULT_VAL_FROM),
                       val_size=settings.get("val_size", cd.DEFAULT_VAL_SIZE), seed=settings.get("seed", 0))
        assert out.read_bytes() == (tmp_path / name / cd.VAL_FILE).read_bytes()


def _snapshot(tmp_path: Path, dataset: str, revision: str, relative: str) -> Path:
    """A file path inside a Hugging Face hub snapshot of ``revision``, with its folders made."""
    path = tmp_path / "hub" / f"datasets--{dataset}" / "snapshots" / revision / relative
    path.parent.mkdir(parents=True)
    return path


def test_math500_keeps_the_dataset_order_and_boxes_the_answer(tmp_path):
    source = _snapshot(tmp_path, "HuggingFaceH4--MATH-500", tool.MATH500_REVISION, "test.jsonl")
    items = [{"problem": "Find $x$ if $2x=4$.", "solution": "...", "answer": "2", "subject": "Algebra", "level": 1,
              "unique_id": "test/algebra/1.json"},
             {"problem": "Compute $\\frac{1}{2}$ of 3.", "solution": "...", "answer": "\\frac{3}{2}",
              "subject": "Prealgebra", "level": 2, "unique_id": "test/prealgebra/7.json"}]
    source.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    out = tmp_path / "math500.jsonl"
    summary = tool.build_math500(source, out)
    assert _rows(out) == [
        {"instruction": "Find $x$ if $2x=4$.", "response": "\\boxed{2}", "unique_id": "test/algebra/1.json"},
        {"instruction": "Compute $\\frac{1}{2}$ of 3.", "response": "\\boxed{\\frac{3}{2}}",
         "unique_id": "test/prealgebra/7.json"}]
    assert summary["rows"] == 2 and len(summary["md5"]) == 32
    with pytest.raises(FileExistsError):
        tool.build_math500(source, out)


def test_gsm8k_takes_the_problems_that_mgsm_translates(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = _snapshot(tmp_path, "openai--gsm8k", tool.GSM8K_REVISION, "main/test-00000-of-00001.parquet")
    items = [{"question": f"Question {number}’s text?", "answer": f"Work.\n#### {1000 + number:,}"}
             for number in range(tool.MGSM_PAIRED_ROWS + 3)]
    pq.write_table(pa.Table.from_pylist(items), source)
    out = tmp_path / "gsm8k.jsonl"
    summary = tool.build_gsm8k(source, out)
    rows = _rows(out)
    assert summary["rows"] == len(rows) == tool.MGSM_PAIRED_ROWS
    # The thousands comma goes, the curly apostrophe stays as it is.
    assert rows[0] == {"instruction": "Question 0’s text?", "response": "\\boxed{1000}", "unique_id": "gsm8k_test_0"}
    assert rows[-1]["unique_id"] == "gsm8k_test_249" and "’" in out.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="####"):
        tool.gsm8k_answer("no final answer")


def test_mgsm_ru_boxes_the_answer_number(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    source = _snapshot(tmp_path, "juletxara--mgsm", tool.MGSM_REVISION, "ru/test-00000-of-00001.parquet")
    table = pa.table({"question": ["Сколько будет 2+2?", "Сколько будет 9*9?"], "answer": [None, None],
                      "answer_number": pa.array([4, 81], type=pa.int32()), "equation_solution": [None, None]})
    pq.write_table(table, source)
    out = tmp_path / "mgsm_ru.jsonl"
    tool.build_mgsm_ru(source, out)
    assert _rows(out) == [{"instruction": "Сколько будет 2+2?", "response": "\\boxed{4}"},
                          {"instruction": "Сколько будет 9*9?", "response": "\\boxed{81}"}]
    assert "Сколько" in out.read_text(encoding="utf-8")


def test_math_sets_refuse_another_revision(tmp_path):
    other = tmp_path / "test.jsonl"
    other.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match=f"revision {tool.MATH500_REVISION}"):
        tool.build_math500(other, tmp_path / "out.jsonl")
    with pytest.raises(ValueError, match=f"revision {tool.GSM8K_REVISION}"):
        tool.build_gsm8k(other, tmp_path / "out.jsonl")
    with pytest.raises(ValueError, match=f"revision {tool.MGSM_REVISION}"):
        tool.build_mgsm_ru(other, tmp_path / "out.jsonl")
    assert not (tmp_path / "out.jsonl").exists()
