"""CPU tests for orbit_verl.data.distill_data and the launcher's distill builder.

Run with the verl python and PYTHONPATH=<repo>/src.
The tests use tiny jsonl inputs and a character-level stand-in for the Qwen3 tokenizer, so they need
no model files. The tests that write parquet need pyarrow, the launcher tests need omegaconf, and the
verl loading test needs verl; each of them skips when its package is missing.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from orbit_verl import common

# The cache variables must be set before transformers or datasets is imported in this process.
# They go under the runs root of the person running the tests, or into a temporary dir.
RUNS_ROOT = os.environ.get("ORBIT_VERL_RUNS") or tempfile.mkdtemp(prefix="orbit_verl_tests_")
common.apply_cache_env(RUNS_ROOT)

from orbit_verl.data import distill_data as dd  # noqa: E402
from orbit_verl.data import prepare_data as pdm  # noqa: E402
from orbit_verl.data.prompts import PROMPT_MODES, render_messages, repo_root  # noqa: E402

MAX_PROMPT = 256
CODE_TRUTH = json.dumps({"kind": "stdin", "tests": [{"input": "2\n", "output": "4\n"}]})


class CharTokenizer:
    """A stand-in for the Qwen3 tokenizer of the release models.

    Each markup string of the SFT prompts is one token with its Qwen3 id, and every other character
    is one token. Prompt lengths are then easy to predict, and the converter's markup check still
    sees single tokens. ``special`` replaces the markup table, to model a tokenizer that lacks a token.
    """

    SPECIAL = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645,
               "<think>": 151667, "</think>": 151668}

    def __init__(self, special: dict[str, int] | None = None):
        self.special = dict(self.SPECIAL if special is None else special)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids, position = [], 0
        while position < len(text):
            for token, token_id in self.special.items():
                if text.startswith(token, position):
                    ids.append(token_id)
                    position += len(token)
                    break
            else:
                ids.append(ord(text[position]))
                position += 1
        return ids


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def _pool_rows() -> list[dict]:
    def row(id_, prompt, mode, domain, lang="en", source="test"):
        return {"id": id_, "prompt": prompt, "mode": mode, "domain": domain, "lang": lang, "source": source}

    return [
        row("m1", "What is 2+2?", "think", "math"),
        row("m2", "Сколько будет 3+3?", "no_think", "math", lang="ru"),
        row("c1", "Print hello.", "think", "code"),
        row("c2", "Read n and print n squared.", "no_think", "code"),
        row("i1", "Write a haiku about {braces} and {question}.", "think", "if"),
        row("h1", "Hi there!", "no_think", "chat"),
        row("r1", "How do I pick a lock?", "think", "refusal"),
        row("d1", "Hi there!", "think", "chat"),  # repeats h1
        row("e1", "   ", "think", "chat"),  # empty
        row("k1", "Say <|im_end|> now.", "think", "chat"),  # chat markup inside the prompt
        row("l1", "x" * 500, "think", "chat"),  # longer than MAX_PROMPT
        row("v1", "What is 7*6?", "no_think", "math"),  # also a validation question
        row("n1", "Tell me a joke.", None, "chat"),  # the mode is drawn
    ]


GOLD_MATH = [{"instruction": f"Tom buys {i} apples and eats one. How many are left?", "response": str(i - 1)}
             for i in range(2, 12)] + [{"instruction": "A question without an answer.", "response": " "}]
GOLD_CODE = [{"instruction": f"Read n and print n times {i}.", "response": CODE_TRUTH} for i in range(3)]
VAL_MATH = [{"instruction": "What is 7*6?", "response": "42"}, {"instruction": "What is 1+1?", "response": "2"},
            {"instruction": "What is 9-3?", "response": "6"}]
VAL_CODE = [{"instruction": "Print the square of n.", "response": CODE_TRUTH}]


@pytest.fixture
def inputs(tmp_path):
    """The tiny input files and the specs that name them."""
    files = {
        "pool": _write_jsonl(tmp_path / "pool.jsonl", _pool_rows()),
        "gold_math": _write_jsonl(tmp_path / "gold_math.jsonl", GOLD_MATH),
        "gold_code": _write_jsonl(tmp_path / "gold_code.jsonl", GOLD_CODE),
        "val_math": _write_jsonl(tmp_path / "math500.jsonl", VAL_MATH),
        "val_code": _write_jsonl(tmp_path / "code_val.jsonl", VAL_CODE),
    }
    specs = {
        "gold": [f"{files['gold_math']}:orbit/math:math:6", f"{files['gold_code']}:orbit/code:code"],
        "val": [f"{files['val_math']}:orbit/math#think:sft_think",
                f"{files['val_math']}:orbit/math#no_think:sft_no_think:2",
                f"{files['val_code']}:orbit/code#think:sft_think_code"],
    }
    return files, specs


def _build(tmp_path, files, specs, **overrides):
    args = dict(
        pool=str(files["pool"]),
        gold=[dd.parse_gold_spec(spec) for spec in specs["gold"]],
        val=[dd.parse_val_spec(spec) for spec in specs["val"]],
        tokenizer=CharTokenizer(),
        max_prompt_length=MAX_PROMPT,
        think_share=0.5,
        seed=0,
        limit=None,
    )
    args.update(overrides)
    return dd.build(**args)


def _check_record(record: dict, split: str) -> None:
    """What every row must satisfy, whichever input it came from."""
    info = record["extra_info"]
    mode = PROMPT_MODES[info["prompt_mode"]]
    assert record["prompt"] == render_messages(mode, info["question"])
    assert record["agent_name"] == mode.agent_loop
    assert record[dd.TEACHER_ROUTE] == info["mode"] == dd.MODE_FORMATS[mode.name][1]
    assert info["split"] == split
    text = "".join(message["content"] for message in record["prompt"])
    assert info["prompt_tokens"] == len(CharTokenizer().encode(text)) <= MAX_PROMPT
    assert record["reward_model"]["style"] == "rule"


# ---- the tables and the specs ------------------------------------------------------------------


def test_every_format_has_a_registered_prompt_mode():
    assert set(dd.FORMAT_MODES.values()) == {"sft_think", "sft_no_think", "sft_think_code", "sft_no_think_code",
                                             "sft_chat_think", "sft_chat_no_think", "sft_think_ru", "sft_no_think_ru"}
    assert set(dd.FORMAT_MODES.values()) <= set(PROMPT_MODES)
    assert set(dd.DOMAIN_FORMATS.values()) == {"math", "code", "chat"}
    assert {fmt for fmt, _ in dd.FORMAT_MODES} == set(dd.DOMAIN_FORMATS.values()) | set(dd.LANG_FORMATS.values())
    # A language format is a variant of its domain's format, so it exists for both thinking modes.
    for (domain, _lang), fmt in dd.LANG_FORMATS.items():
        assert domain in dd.DOMAIN_FORMATS and {(fmt, t) for t in dd.THINKING_MODES} <= set(dd.FORMAT_MODES)
    for name, (fmt, thinking) in dd.MODE_FORMATS.items():
        assert dd.FORMAT_MODES[(fmt, thinking)] == name
    assert dd.MARKUP == ("</think>", "<think>", "<|endoftext|>", "<|im_end|>", "<|im_start|>")


def test_gold_specs(tmp_path):
    spec = dd.parse_gold_spec("data/math.jsonl:orbit/math:math:500")
    assert spec == dd.GoldSpec("data/math.jsonl", "orbit/math", "math", 500)
    assert spec.canonical() == f"{Path('data/math.jsonl').resolve()}:orbit/math:math:500"
    # The fields are taken from the right, so a path with a colon stays whole, and COUNT is optional.
    colon = dd.parse_gold_spec("data/a:b.jsonl:orbit/math:math")
    assert (colon.path, colon.data_source, colon.family, colon.count) == ("data/a:b.jsonl", "orbit/math", "math", None)
    assert colon.canonical() == f"{Path('data/a:b.jsonl').resolve()}:orbit/math:math"
    assert dd.parse_gold_spec("c.jsonl:orbit/code#train:code").data_source == "orbit/code#train"
    ru = dd.parse_gold_spec("ru.jsonl:orbit/math#ru_train:math_ru:800")
    assert (ru.data_source, ru.family, ru.count) == ("orbit/math#ru_train", "math_ru", 800)
    bad = {
        "m.jsonl:orbit/math": "does not have the form",
        "m.jsonl:orbit/math:chat": "family",
        "m.jsonl:orbit/none:math": "checks no gold answer",
        "m.jsonl:openai/gsm8k:math": "unknown data_source",
        "m.jsonl:orbit/gsm8k:math": "unknown data_source",
        "c.jsonl:orbit/math:code": "does not fit",
        "c.jsonl:orbit/code:math": "does not fit",
        "ru.jsonl:orbit/code:math_ru": "does not fit",
        "m.jsonl:orbit/math:math:0": "COUNT",
    }
    for text, message in bad.items():
        with pytest.raises(ValueError, match=message):
            dd.parse_gold_spec(text)


def test_val_specs():
    spec = dd.parse_val_spec("m.jsonl:orbit/math#no_think:sft_no_think:50")
    assert spec == dd.ValSpec("m.jsonl", "orbit/math#no_think", "sft_no_think", 50)
    assert spec.canonical().endswith("m.jsonl:orbit/math#no_think:sft_no_think:50")
    # A set without gold answers fits any mode; it reports the response diagnostics only.
    assert dd.parse_val_spec("if.jsonl:orbit/none#if:sft_chat_no_think").limit is None
    bad = {
        "m.jsonl:orbit/math:math_fewshot": "distillation datasets use",
        "m.jsonl:orbit/math:no_such_mode": "distillation datasets use",
        "m.jsonl:orbit/maths#x:sft_think": "unknown data_source",
        "c.jsonl:orbit/code:sft_think": "does not fit",
        "m.jsonl:orbit/math:sft_think_code": "does not fit",
        "m.jsonl:orbit/math:sft_think:0": "LIMIT",
        ":orbit/math:sft_think": "does not have the form",
    }
    for text, message in bad.items():
        with pytest.raises(ValueError, match=message):
            dd.parse_val_spec(text)


# ---- building the rows -------------------------------------------------------------------------


def test_build_mixes_modes_per_row(tmp_path, inputs):
    files, specs = inputs
    train, val, stats = _build(tmp_path, files, specs)
    for record in train:
        _check_record(record, "train")
    for record in val:
        _check_record(record, "val")

    pool = [record for record in train if record["data_source"] == dd.NONE_SOURCE]
    by_id = {record["extra_info"]["id"]: record for record in pool}
    # The domain picks the format and the row's mode picks thinking; chat prompts keep their text.
    expected = {"m1": "sft_think", "m2": "sft_no_think_ru", "c1": "sft_think_code", "c2": "sft_no_think_code",
                "i1": "sft_chat_think", "h1": "sft_chat_no_think", "r1": "sft_chat_think"}
    assert {key: by_id[key]["extra_info"]["prompt_mode"] for key in expected} == expected
    assert set(by_id) == set(expected) | {"n1"}
    assert by_id["n1"]["extra_info"]["prompt_mode"] in ("sft_chat_think", "sft_chat_no_think")
    assert stats["pool_modes_drawn"] == 1
    assert by_id["i1"]["prompt"][0]["content"] == (
        "<|endoftext|><|im_start|>user\nWrite a haiku about {braces} and {question}.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n")
    assert by_id["m2"]["extra_info"]["lang"] == "ru" and by_id["m2"]["ability"] == "math"
    # A Russian math problem gets the Russian instruction of the SFT eval, and no English in its user turn.
    m2 = by_id["m2"]["prompt"][0]["content"]
    assert "Сколько будет 3+3?\nРассуждай пошагово и запиши окончательный ответ в \\boxed{}." in m2
    assert "Please reason" not in m2
    assert "Please reason" in by_id["m1"]["prompt"][0]["content"]
    assert by_id["r1"]["ability"] == "refusal" and by_id["r1"]["extra_info"]["source"] == "test"
    assert all(record["reward_model"]["ground_truth"] == "" for record in pool)
    assert stats["dropped"]["pool"] == {"empty": 1, "markup": 1, "val_overlap": 1, "duplicate": 1, "overlong": 1}

    gold = [record for record in train if record["data_source"] != dd.NONE_SOURCE]
    math = [record for record in gold if record["data_source"] == "orbit/math"]
    code = [record for record in gold if record["data_source"] == "orbit/code"]
    assert len(math) == 6 and len(code) == 3
    assert {record["extra_info"]["prompt_mode"] for record in math} <= {"sft_think", "sft_no_think"}
    assert {record["extra_info"]["prompt_mode"] for record in code} <= {"sft_think_code", "sft_no_think_code"}
    assert all(record["reward_model"]["ground_truth"] == CODE_TRUTH for record in code)
    for record in math:
        # The gold answer stays with its question.
        count = int(record["extra_info"]["question"].split()[2])
        assert record["reward_model"]["ground_truth"] == str(count - 1)
        assert record["extra_info"]["id"] == f"gold_math.jsonl:{record['extra_info']['index']}"

    # Validation rows keep the file order and the spec's data_source, mode and limit.
    assert [(record["data_source"], record["extra_info"]["question"]) for record in val] == [
        ("orbit/math#think", "What is 7*6?"), ("orbit/math#think", "What is 1+1?"),
        ("orbit/math#think", "What is 9-3?"),
        ("orbit/math#no_think", "What is 7*6?"), ("orbit/math#no_think", "What is 1+1?"),
        ("orbit/code#think", "Print the square of n."),
    ]
    assert [record[dd.TEACHER_ROUTE] for record in val] == ["think"] * 3 + ["no_think"] * 2 + ["think"]
    val_questions = {record["extra_info"]["question"] for record in val}
    assert not val_questions & {record["extra_info"]["question"] for record in train}

    assert stats["rows"] == {"train": len(train), "val": len(val)}
    for split, records in (("train", train), ("val", val)):
        assert stats["prompt_modes"][split] == dict(Counter(r["extra_info"]["prompt_mode"] for r in records))
        counted = sum(n for by_mode in stats["counts"][split].values() for by_source in by_mode.values()
                      for n in by_source.values())
        assert counted == len(records)
    assert stats["counts"]["train"]["math"]["think"].get(dd.NONE_SOURCE) == 1
    assert stats["gold"] == [dd.parse_gold_spec(spec).canonical() for spec in specs["gold"]]
    assert stats["val"] == [dd.parse_val_spec(spec).canonical() for spec in specs["val"]]
    assert stats["pool_source"]["sha256"] == common.sha256_file(files["pool"])


def test_build_is_seeded(tmp_path, inputs):
    files, specs = inputs
    first, _, _ = _build(tmp_path, files, specs, seed=3)
    again, _, _ = _build(tmp_path, files, specs, seed=3)
    assert first == again
    other, _, _ = _build(tmp_path, files, specs, seed=4)
    assert sorted(r["extra_info"]["id"] for r in other if r["data_source"] == dd.NONE_SOURCE) == sorted(
        r["extra_info"]["id"] for r in first if r["data_source"] == dd.NONE_SOURCE)


def test_think_share_draws_the_gold_modes(tmp_path, inputs):
    files, specs = inputs
    for share, mode in ((1.0, "think"), (0.0, "no_think")):
        train, _, _ = _build(tmp_path, files, specs, think_share=share)
        gold = [record for record in train if record["data_source"] != dd.NONE_SOURCE]
        assert {record[dd.TEACHER_ROUTE] for record in gold} == {mode}


def test_gold_only_with_limit(tmp_path, inputs):
    files, specs = inputs
    train, _, stats = _build(tmp_path, files, specs, pool=None, limit=4)
    assert len(train) == 4 and stats["pool"] is None and stats["dropped"]["pool"] is None
    assert all(record["data_source"] in ("orbit/math", "orbit/code") for record in train)


def test_gold_shortfall_is_refused(tmp_path, inputs):
    files, specs = inputs
    # Ten math rows have an answer, and one of them is empty.
    short = [dd.parse_gold_spec(f"{files['gold_math']}:orbit/math:math:11")]
    with pytest.raises(ValueError, match="asks for 11 rows"):
        _build(tmp_path, files, specs, gold=short)


def test_val_sets_in_two_modes_need_tags(tmp_path, inputs):
    files, specs = inputs
    untagged = [dd.parse_val_spec(f"{files['val_math']}:orbit/math:sft_think"),
                dd.parse_val_spec(f"{files['val_math']}:orbit/math:sft_no_think")]
    with pytest.raises(ValueError, match="share a data_source"):
        _build(tmp_path, files, specs, val=untagged)


def test_val_without_gold_answers_keeps_empty_answers(tmp_path, inputs):
    files, specs = inputs
    chat = _write_jsonl(tmp_path / "chat_val.jsonl", [{"instruction": "Say hi.", "response": ""}])
    _, val, _ = _build(tmp_path, files, specs, val=[dd.parse_val_spec(f"{chat}:orbit/none#chat:sft_chat_no_think")])
    assert [(r["data_source"], r["extra_info"]["prompt_mode"]) for r in val] == [
        ("orbit/none#chat", "sft_chat_no_think")]


def test_tokenizer_that_splits_the_markup_is_refused(tmp_path, inputs):
    files, specs = inputs
    special = {token: token_id for token, token_id in CharTokenizer.SPECIAL.items() if token != "</think>"}
    with pytest.raises(ValueError, match="</think>"):
        _build(tmp_path, files, specs, tokenizer=CharTokenizer(special))


@pytest.mark.parametrize(("row", "message"), [
    ({"id": "a", "prompt": "q", "mode": "think", "domain": "poetry"}, "domain"),
    ({"id": "a", "prompt": "q", "mode": "thinking", "domain": "chat"}, "mode"),
    ({"prompt": "q", "mode": "think", "domain": "chat"}, "no id"),
    ({"id": "a", "prompt": None, "mode": "think", "domain": "chat"}, "prompt"),
    ({"id": "a", "prompt": "q", "mode": "think", "domain": "chat", "lang": 1}, "lang"),
])
def test_malformed_pool_rows_are_refused(tmp_path, row, message):
    path = _write_jsonl(tmp_path / "pool.jsonl", [{"id": "ok", "prompt": "q", "mode": "think", "domain": "chat"}, row])
    with pytest.raises(ValueError, match=f"pool.jsonl:2 .*{message}"):
        dd.read_pool(path)
    (tmp_path / "broken.jsonl").write_text('{"id": "a",\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        dd.read_pool(tmp_path / "broken.jsonl")


# ---- writing the dataset -----------------------------------------------------------------------


def _tokenizer_dir(tmp_path: Path) -> Path:
    """A directory that common.tokenizer_sha256 can fingerprint; the fake tokenizer does the work."""
    path = tmp_path / "tokenizer"
    path.mkdir(exist_ok=True)
    (path / "tokenizer.json").write_text('{"fake": true}', encoding="utf-8")
    return path


def _convert(tmp_path, files, specs, out, **overrides):
    args = dict(tokenizer_dir=str(_tokenizer_dir(tmp_path)), max_prompt_length=MAX_PROMPT, val=specs["val"],
                gold=specs["gold"], runs_root=RUNS_ROOT, tokenizer=CharTokenizer())
    args.update(overrides)
    return dd.convert(str(files["pool"]), str(out), **args)


def test_convert_writes_a_dataset_the_launcher_accepts(tmp_path, inputs):
    pq = pytest.importorskip("pyarrow.parquet")
    files, specs = inputs
    out = tmp_path / "data" / "distill"
    manifest = _convert(tmp_path, files, specs, out)
    assert sorted(path.name for path in out.iterdir()) == ["manifest.json", "train.parquet", "val.parquet"]
    assert not list(out.parent.glob(".distill.staging-*"))
    assert common.read_manifest(out / common.DATA_MANIFEST, common.DATA_MANIFEST_KIND) == manifest
    assert manifest["builder"] == dd.BUILDER and manifest["prompt_mode"] == dd.MIXED
    assert manifest["format_version"] == dd.FORMAT_VERSION == 2
    assert manifest["max_prompt_length"] == MAX_PROMPT and manifest["think_share"] == 0.5
    assert manifest["tokenizer_sha256"] == common.tokenizer_sha256(_tokenizer_dir(tmp_path))
    schema = dd.parquet_schema()
    assert schema.names == [*pdm.parquet_schema().names, dd.TEACHER_ROUTE]
    for name, split in ((pdm.TRAIN_FILE, "train"), (pdm.VAL_FILE, "val")):
        assert pq.read_schema(out / name).remove_metadata().equals(schema)
        rows = pq.read_table(out / name).to_pylist()
        assert len(rows) == manifest["rows"][split]
        for row in rows:
            _check_record(row, split)
    # Replacing a dataset needs force, and force replaces only a directory a data tool wrote.
    with pytest.raises(FileExistsError):
        _convert(tmp_path, files, specs, out)
    assert _convert(tmp_path, files, specs, out, force=True, limit=2)["rows"]["train"] == 2
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("not ours")
    with pytest.raises(FileExistsError):
        _convert(tmp_path, files, specs, foreign, force=True)
    assert (foreign / "keep.txt").read_text() == "not ours"


def test_convert_refuses_bad_arguments(tmp_path, inputs):
    files, specs = inputs
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="validation set"):
        _convert(tmp_path, files, specs, out, val=[])
    with pytest.raises(ValueError, match="think_share"):
        _convert(tmp_path, files, specs, out, think_share=1.5)
    with pytest.raises(ValueError, match="Nothing to train on"):
        dd.convert(None, str(out), tokenizer_dir=str(tmp_path), max_prompt_length=MAX_PROMPT, val=specs["val"])
    assert not out.exists()


def test_cli_reports_errors_with_exit_code_2(tmp_path, inputs, capsys):
    files, specs = inputs
    argv = ["--pool", str(tmp_path / "missing.jsonl"), "--val", specs["val"][0], "--tokenizer", str(tmp_path),
            "--max-prompt-length", "64", "--out-dir", str(tmp_path / "out"), "--runs-root", RUNS_ROOT]
    assert dd.main(argv) == 2
    assert "distill_data: error" in capsys.readouterr().err
    assert dd.main([*argv[:2], "--val", "m.jsonl:orbit/math:no_such_mode", *argv[4:]]) == 2
    assert not (tmp_path / "out").exists()


def test_verl_rlhf_dataset_loads_the_mixed_files(tmp_path, inputs):
    """The files must load through verl's RLHFDataset, with teacher_route reaching each sample."""
    pytest.importorskip("pyarrow")
    verl = pytest.importorskip("verl")
    from omegaconf import OmegaConf
    from verl.utils.dataset.rl_dataset import RLHFDataset

    files, specs = inputs
    out = tmp_path / "distill"
    _convert(tmp_path, files, specs, out)
    config = OmegaConf.load(Path(verl.__file__).parent / "trainer" / "config" / "data" / "legacy_data.yaml")
    config.max_prompt_length = MAX_PROMPT
    config.filter_overlong_prompts = False
    config.cache_dir = str(tmp_path / "verl_cache")
    dataset = RLHFDataset(data_files=[str(out / pdm.TRAIN_FILE)], tokenizer=CharTokenizer(), config=config)
    import pyarrow.parquet as pq

    stored = pq.read_table(out / pdm.TRAIN_FILE).to_pylist()
    assert len(dataset) == len(stored)
    for position in range(len(stored)):
        item, row = dataset[position], stored[position]
        assert item["raw_prompt"] == row["prompt"]
        assert item["agent_name"] == row["agent_name"]
        assert item[dd.TEACHER_ROUTE] == row[dd.TEACHER_ROUTE]
        assert item["data_source"] == row["data_source"]


# ---- the launcher's distill builder ------------------------------------------------------------


def _launch_cfg(tmp_path: Path, data: dict, prompt_mode: str = "sft_think", distillation: dict | None = None):
    """The part of a composed config that the launcher's data checks read."""
    omegaconf = pytest.importorskip("omegaconf")
    block = {"prompt_mode": "${orbit.prompt_mode}", "src": "???", "name": "d", "val_src": None, "limit": None,
             "data_source": None, "question_field": "instruction", "answer_field": "response", "val_size": 64}
    block.update(data)
    cfg = omegaconf.OmegaConf.create({
        "orbit": {"runs_root": RUNS_ROOT, "repo_root": str(repo_root()), "prompt_mode": prompt_mode,
                   "model_dir": str(_tokenizer_dir(tmp_path)), "data_dir": str(tmp_path / "data" / "distill"),
                   "prepare": {"data": block}},
        "data": {"max_prompt_length": MAX_PROMPT},
        "distillation": distillation or {"enabled": False, "teacher_key": "data_source",
                                         "teacher_models": {"teacher_model": {"model_path": "/t"}}},
    })
    omegaconf.OmegaConf.set_struct(cfg, True)
    return cfg


def _distill_block(files, specs, **extra):
    return {"builder": "distill", "pool": str(files["pool"]), "gold": specs["gold"], "val": specs["val"],
            "think_share": 0.5, "seed": 0, **extra}


def test_builder_selection(tmp_path, inputs):
    from orbit_verl.launch import prepare

    files, specs = inputs
    assert prepare.data_builder(_launch_cfg(tmp_path, {})) == "prepare_data"
    assert prepare.data_builder(_launch_cfg(tmp_path, _distill_block(files, specs))) == "distill"
    # "mixed" names no single prompt mode, so only the distill builder can serve it.
    assert prepare.data_builder(_launch_cfg(tmp_path, {}, prompt_mode="mixed")) == "distill"
    problems = prepare.check_data(_launch_cfg(tmp_path, {"builder": "mixer"}))
    assert [p.kind for p in problems] == ["config"] and "mixer" in problems[0].message


def test_teacher_routing_by_mode_needs_the_distill_builder(tmp_path, inputs):
    from orbit_verl.launch import prepare

    files, specs = inputs
    two = {"enabled": True, "teacher_key": dd.TEACHER_ROUTE,
           "teacher_models": {"think": {"key": "think"}, "no_think": {"key": "no_think"}}}
    problems = prepare.check_data(_launch_cfg(tmp_path, {"src": "x.jsonl"}, distillation=two), model_ok=False)
    assert [p.kind for p in problems] == ["config"] and dd.TEACHER_ROUTE in problems[0].message
    # One teacher takes every row whatever the column says, and the distill builder writes the column.
    one = {**two, "teacher_models": {"teacher_model": {"model_path": "/t"}}}
    assert [p.kind for p in prepare.check_data(_launch_cfg(tmp_path, {"src": "x.jsonl"}, distillation=one),
                                               model_ok=False)] == ["data"]
    routed = prepare.check_data(_launch_cfg(tmp_path, _distill_block(files, specs), distillation=two), model_ok=False)
    assert [p.kind for p in routed] == ["data"]


def test_missing_distill_dataset_prints_the_distill_command(tmp_path, inputs):
    from orbit_verl.launch import prepare

    files, specs = inputs
    problems = prepare.check_data(_launch_cfg(tmp_path, _distill_block(files, specs, limit=8)), model_ok=False)
    assert [(p.kind, p.buildable) for p in problems] == [("data", True)]
    command = problems[0].commands[0]
    assert "-m orbit_verl.data.distill_data --pool " in command and "--prompt-mode" not in command
    assert command.count("--val ") == 3 and command.count("--gold ") == 2 and "--limit 8" in command


def test_distill_config_problems(tmp_path, inputs):
    from orbit_verl.launch import prepare

    files, specs = inputs
    cases = {
        "needs a prompt pool": {"pool": None, "gold": []},
        "validation set": {"val": []},
        "family": {"gold": [f"{files['gold_math']}:orbit/math:chat"]},
    }
    for message, change in cases.items():
        problems = prepare.check_data(_launch_cfg(tmp_path, _distill_block(files, specs, **change)))
        assert [p.kind for p in problems] == ["config"] and message in problems[0].message
        # --force-prepare must not build from a block that no tool can build from.
        assert prepare._build_data(_launch_cfg(tmp_path, _distill_block(files, specs, **change)), True,
                                   lambda line: None) == problems


def _fake_distill_dataset(tmp_path: Path, files, specs, **manifest_changes) -> Path:
    """A dataset dir whose manifest records what distill_data records for these inputs."""
    out = tmp_path / "data" / "distill"
    out.mkdir(parents=True, exist_ok=True)
    for name in (pdm.TRAIN_FILE, pdm.VAL_FILE):
        (out / name).write_bytes(b"")
    payload = {
        "builder": dd.BUILDER, "prompt_mode": dd.MIXED, "pool": str(Path(files["pool"]).resolve()),
        "gold": [dd.parse_gold_spec(spec).canonical() for spec in specs["gold"]],
        "val": [dd.parse_val_spec(spec).canonical() for spec in specs["val"]],
        "think_share": 0.5, "seed": 0, "limit": None, "max_prompt_length": MAX_PROMPT,
        "tokenizer_sha256": common.tokenizer_sha256(_tokenizer_dir(tmp_path)),
        "format_version": dd.FORMAT_VERSION,
    }
    payload.update(manifest_changes)
    common.write_manifest(out / common.DATA_MANIFEST, common.DATA_MANIFEST_KIND, payload)
    return out


def test_distill_dataset_is_compared_with_the_config(tmp_path, inputs):
    from orbit_verl.launch import prepare

    files, specs = inputs
    _fake_distill_dataset(tmp_path, files, specs)
    assert prepare.check_data(_launch_cfg(tmp_path, _distill_block(files, specs))) == []
    # A relative path to the same pool is the same input.
    relative = os.path.relpath(files["pool"])
    assert prepare.check_data(_launch_cfg(tmp_path, _distill_block(files, specs, pool=relative))) == []
    changed = {
        "think_share": {"think_share": 0.3},
        "seed": {"seed": 1},
        "limit": {"limit": 5},
        "val": {"val": specs["val"][:2]},
        "gold": {"gold": specs["gold"][::-1]},
        "pool": {"pool": str(files["gold_math"])},
    }
    for key, change in changed.items():
        problems = prepare.check_data(_launch_cfg(tmp_path, _distill_block(files, specs, **change)))
        assert [p.kind for p in problems] == ["data"], key
        said = "the pool" if key == "pool" else f"it has {key}="
        assert said in problems[0].message and "--force" in problems[0].commands[0], key
    # A config that builds with prepare_data does not take a distillation dataset, and the reverse.
    problems = prepare.check_data(_launch_cfg(tmp_path, {"src": "x.jsonl"}))
    assert "mixed distillation dataset" in problems[0].message
    _fake_distill_dataset(tmp_path, files, specs, builder=None, prompt_mode="sft_think")
    problems = prepare.check_data(_launch_cfg(tmp_path, _distill_block(files, specs)))
    assert "built by prepare_data" in problems[0].message


def test_build_data_runs_the_configured_builder(tmp_path, inputs, monkeypatch):
    from orbit_verl.launch import prepare

    files, specs = inputs
    calls = []
    monkeypatch.setattr(dd, "convert", lambda pool, out_dir, **kwargs: calls.append(("distill", pool, out_dir, kwargs)))
    monkeypatch.setattr(pdm, "convert", lambda src, out_dir, **kwargs: calls.append(("prepare_data", src, out_dir)))
    cfg = _launch_cfg(tmp_path, _distill_block(files, specs, limit=8))
    assert prepare._build_data(cfg, False, lambda line: None) == []
    builder, pool, out_dir, kwargs = calls.pop()
    assert (builder, pool, out_dir) == ("distill", str(files["pool"]), str(tmp_path / "data" / "distill"))
    assert kwargs["gold"] == specs["gold"] and kwargs["val"] == specs["val"] and kwargs["limit"] == 8
    assert kwargs["think_share"] == 0.5 and kwargs["seed"] == 0 and kwargs["max_prompt_length"] == MAX_PROMPT
    assert kwargs["force"] is False and kwargs["runs_root"] == RUNS_ROOT
    # Every other config keeps prepare_data.
    assert prepare._build_data(_launch_cfg(tmp_path, {"src": "x.jsonl"}), False, lambda line: None) == []
    assert calls.pop()[:2] == ("prepare_data", "x.jsonl")



def test_a_dataset_of_older_format_rules_is_not_reused(tmp_path, inputs):
    """A dataset built before the Russian math format (no format_version, which means 1) gives Russian
    math problems the English instruction, so the launcher must not reuse it by its name."""
    from orbit_verl.launch import prepare

    files, specs = inputs
    out = _fake_distill_dataset(tmp_path, files, specs)
    manifest = common.read_manifest(out / common.DATA_MANIFEST, common.DATA_MANIFEST_KIND)
    manifest.pop("format_version")
    common.write_manifest(out / common.DATA_MANIFEST, common.DATA_MANIFEST_KIND, manifest)
    problems = prepare.check_data(_launch_cfg(tmp_path, _distill_block(files, specs)))
    assert [p.kind for p in problems] == ["data"] and "prompt formats are version 1" in problems[0].message


def test_a_russian_math_validation_set_keeps_its_domain_and_language(tmp_path, inputs):
    files, specs = inputs
    val_specs = [*specs["val"], f"{files['val_math']}:orbit/math#ru:sft_think_ru:1"]
    _, val, _ = _build(tmp_path, files, {**specs, "val": val_specs})
    (ru,) = [record for record in val if record["data_source"] == "orbit/math#ru"]
    assert ru["extra_info"]["domain"] == "math" and ru["extra_info"]["lang"] == "ru"
    assert ru["extra_info"]["prompt_mode"] == "sft_think_ru"


# ---- Russian math with gold answers (the math_ru family) ---------------------------------------

GOLD_RU = [{"instruction": f"Найдите x, если x - {i} = 1.", "response": f"\\boxed{{{i + 1}}}"} for i in range(4)]


def test_russian_math_gold_rows_keep_domain_and_language(tmp_path, inputs):
    """A math_ru gold row is Russian math in the Russian prompt format, with domain math and language ru.

    Without the language, a Russian training row would look like English math to everything that
    reads extra_info, and a thinking budget would insert the English stop-thinking sentence into it.
    """
    files, specs = inputs
    ru_path = _write_jsonl(tmp_path / "ru_math_keep.jsonl", GOLD_RU)
    truth = {row["instruction"]: row["response"] for row in GOLD_RU}
    for share, mode in ((1.0, "sft_think_ru"), (0.0, "sft_no_think_ru")):
        gold = [dd.parse_gold_spec(specs["gold"][0]), dd.parse_gold_spec(f"{ru_path}:orbit/math#ru_train:math_ru")]
        train, _, stats = _build(tmp_path, files, specs, pool=None, gold=gold, think_share=share)
        for record in train:
            _check_record(record, "train")
        ru = [record for record in train if record["data_source"] == "orbit/math#ru_train"]
        assert len(ru) == len(GOLD_RU)
        for record in ru:
            info = record["extra_info"]
            assert (info["prompt_mode"], info["domain"], info["lang"], record["ability"]) == (mode, "math", "ru", "math")
            assert record["reward_model"]["ground_truth"] == truth[info["question"]]
            content = record["prompt"][0]["content"]
            assert "\nРассуждай пошагово и запиши окончательный ответ в \\boxed{}." in content
            assert "Please reason" not in content
        # English math gold rows keep no language, as before.
        english = [record for record in train if record["data_source"] == "orbit/math"]
        assert english and {record["extra_info"]["lang"] for record in english} == {None}
        thinking = "think" if share == 1.0 else "no_think"
        assert stats["counts"]["train"]["math"][thinking]["orbit/math#ru_train"] == len(GOLD_RU)

