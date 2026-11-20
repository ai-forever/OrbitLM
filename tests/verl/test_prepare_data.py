"""CPU tests for orbit_verl.data.prepare_data: filtering, splits, schema, manifest, verl loading.

Run with the verl python and PYTHONPATH=<repo>/src. Most tests need the Qwen3 tokenizer of an SFT
export (ORBIT_TEST_SFT_EXPORT, see conftest.py) and skip without it.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from orbit_verl import common

# The cache variables must be set before transformers is imported in this process. They go under
# the runs root of the person running the tests, or into a temporary dir.
RUNS_ROOT = os.environ.get("ORBIT_VERL_RUNS") or tempfile.mkdtemp(prefix="orbit_verl_tests_")
common.apply_cache_env(RUNS_ROOT)

import pyarrow.parquet as pq  # noqa: E402

from orbit_verl.data import prepare_data as pdm  # noqa: E402
from orbit_verl.data.prompts import PROMPT_MODES, prompt_token_ids, render_messages  # noqa: E402

# The end-of-text id of the Qwen3 tokenizer, which the prompts start with.
QWEN3_EOT = 151643

MANIFEST_KEYS = {
    "src", "val_src", "source", "val_source", "prompt_mode", "data_source", "agent_name", "question_field",
    "answer_field", "max_prompt_length", "tokenizer_dir", "tokenizer_sha256", "tokenizer_family", "rows",
    "dropped", "prompt_tokens", "seed", "limit", "val_size", "drop_duplicates",
}


@pytest.fixture(scope="module")
def tokenizer_dir(sft_export) -> str:
    """The SFT export, which holds the Qwen3 tokenizer that the datasets are measured with."""
    return str(sft_export)


@pytest.fixture(scope="module")
def tokenizer(qwen3_tokenizer):
    return qwen3_tokenizer


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def _rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def _synthetic_rows() -> list[dict]:
    rows = [
        {"instruction": "", "response": "1"},  # empty question
        {"instruction": "What is 2+3?", "response": " "},  # empty answer
        {"instruction": "What is 2+3?", "response": "5"},
        {"instruction": "What is 2+3?", "response": "\\boxed{5}"},  # repeated question
        {"instruction": "long " * 2000, "response": "1"},  # longer than any prompt limit
    ]
    rows += [
        {"instruction": f"Tom buys {i} apples and eats one. How many apples are left?", "response": str(i - 1)}
        for i in range(2, 22)
    ]
    rows += [{"instruction": "Сколько будет 2+2? Ответ в градусах: 90°.", "response": "4"},
             {"instruction": "Let $f(x) = \\frac{1}{2}x$ and write {question} literally. Find f(2).", "response": "1"}]
    return rows


# The rows of _synthetic_rows that survive the filters: two empty rows, one repeated question and one
# overlong prompt are dropped.
KEPT = len(_synthetic_rows()) - 4


def _check_dataset(out: Path, mode_name: str, max_prompt_length: int, tokenizer, tokenizer_dir: str
                   ) -> tuple[list[dict], list[dict]]:
    """Checks every dataset must pass; returns the train and val rows."""
    mode = PROMPT_MODES[mode_name]
    schema = pdm.parquet_schema()
    for name in (pdm.TRAIN_FILE, pdm.VAL_FILE):
        assert pq.read_schema(out / name).remove_metadata().equals(schema)
    train, val = _rows(out / pdm.TRAIN_FILE), _rows(out / pdm.VAL_FILE)
    manifest = common.read_manifest(out / common.DATA_MANIFEST, common.DATA_MANIFEST_KIND)
    assert MANIFEST_KEYS <= set(manifest)
    assert manifest["rows"] == {"train": len(train), "val": len(val)}
    assert manifest["prompt_mode"] == mode_name and manifest["agent_name"] == mode.agent_loop
    assert manifest["max_prompt_length"] == max_prompt_length
    assert manifest["tokenizer_sha256"] == common.tokenizer_sha256(tokenizer_dir)
    assert set(manifest["dropped"]) == {"overlong", "duplicate", "empty"}
    lengths = [row["extra_info"]["prompt_tokens"] for row in train + val]
    assert manifest["prompt_tokens"]["max"] == max(lengths) <= max_prompt_length
    for split, rows in (("train", train), ("val", val)):
        for row in rows:
            question = row["extra_info"]["question"]
            assert row["prompt"] == render_messages(mode, question)
            assert row["agent_name"] == mode.agent_loop
            assert row["data_source"] == manifest["data_source"]
            assert row["ability"] == ("code" if manifest["data_source"] == "orbit/code" else "math")
            assert row["reward_model"]["style"] == "rule"
            assert row["extra_info"]["split"] == split
            assert row["extra_info"]["prompt_mode"] == mode_name
        assert len({row["extra_info"]["index"] for row in rows}) == len(rows)
    # Lengths must be what the rollout will see; checking a sample keeps the test fast.
    for row in (train + val)[:20]:
        ids = prompt_token_ids(tokenizer, row["prompt"])
        assert row["extra_info"]["prompt_tokens"] == len(ids)
        # The prompt starts with <|endoftext|>, and the tokenizer added nothing of its own.
        assert ids[0] == QWEN3_EOT and QWEN3_EOT not in ids[1:]
    train_questions = {row["extra_info"]["question"].strip() for row in train}
    assert not train_questions & {row["extra_info"]["question"].strip() for row in val}
    return train, val


def test_synthetic_counts_and_schema(tmp_path, tokenizer, tokenizer_dir):
    src = _write_jsonl(tmp_path / "src.jsonl", _synthetic_rows())
    out = tmp_path / "out"
    manifest = pdm.convert(
        str(src), str(out), prompt_mode="sft_think", tokenizer_dir=tokenizer_dir, max_prompt_length=768, val_size=4
    )
    assert manifest["dropped"] == {"empty": 2, "duplicate": 1, "overlong": 1}
    assert manifest["rows"] == {"train": KEPT - 4, "val": 4}
    assert manifest["data_source"] == "orbit/math" and manifest["tokenizer_family"] == "qwen3"
    train, val = _check_dataset(out, "sft_think", 768, tokenizer, tokenizer_dir)
    kept = {row["extra_info"]["question"]: row["reward_model"]["ground_truth"] for row in train + val}
    # The first copy of a repeated question wins, and the gold string is stored unchanged.
    assert kept["What is 2+3?"] == "5"
    assert sorted(path.name for path in out.iterdir()) == ["manifest.json", "train.parquet", "val.parquet"]
    assert not list(tmp_path.glob(".out.staging-*"))


def test_options_limit_seed_keep(tmp_path, tokenizer_dir):
    src = _write_jsonl(tmp_path / "src.jsonl", _synthetic_rows())
    common_args = dict(prompt_mode="sft_think", tokenizer_dir=tokenizer_dir, max_prompt_length=768, val_size=4)
    a = pdm.convert(str(src), str(tmp_path / "a"), limit=5, **common_args)
    assert a["rows"] == {"train": 5, "val": 4} and a["limit"] == 5
    b = pdm.convert(str(src), str(tmp_path / "b"), seed=1, **common_args)
    c = pdm.convert(str(src), str(tmp_path / "c"), seed=1, **common_args)
    assert _rows(tmp_path / "b" / "val.parquet") == _rows(tmp_path / "c" / "val.parquet")
    assert b["rows"] == c["rows"]
    keep = pdm.convert(str(src), str(tmp_path / "keep"), drop_duplicates=False, **common_args)
    assert keep["dropped"]["duplicate"] == 0 and keep["drop_duplicates"] is False
    assert keep["rows"]["train"] + keep["rows"]["val"] == KEPT + 1


def test_val_src_and_overlong(tmp_path, tokenizer, tokenizer_dir):
    src = _write_jsonl(tmp_path / "src.jsonl", _synthetic_rows())
    val_src = _write_jsonl(tmp_path / "val.jsonl", [{"instruction": "What is 7*6?", "response": "42"}])
    # The limit is the median prompt length of the source, so some prompts fit and some do not. The
    # expected drops are measured independently: unique non-empty questions over the limit.
    questions = {row["instruction"] for row in _synthetic_rows() if row["instruction"].strip()}
    lengths = {q: len(prompt_token_ids(tokenizer, render_messages("sft_think", q))) for q in questions}
    limit = sorted(lengths.values())[len(lengths) // 2]
    assert len(prompt_token_ids(tokenizer, render_messages("sft_think", "What is 7*6?"))) <= limit
    manifest = pdm.convert(
        str(src), str(tmp_path / "out"), prompt_mode="sft_think", tokenizer_dir=tokenizer_dir,
        max_prompt_length=limit, val_src=str(val_src),
    )
    assert manifest["rows"]["val"] == 1 and manifest["val_src"] == str(val_src.resolve())
    assert manifest["val_size"] is None and manifest["val_source"]["rows"] == 1
    too_long = [q for q, n in lengths.items() if n > limit]
    assert 1 <= len(too_long) < len(questions)
    assert manifest["dropped"]["overlong"] == len(too_long)
    train, val = _check_dataset(tmp_path / "out", "sft_think", limit, tokenizer, tokenizer_dir)
    assert all(row["prompt"][0]["content"].startswith("<|endoftext|><|im_start|>user\n") for row in train)
    assert val[0]["reward_model"]["ground_truth"] == "42"


def test_code_and_chat_modes_take_their_data_source(tmp_path, tokenizer, tokenizer_dir):
    src = _write_jsonl(tmp_path / "src.jsonl", _synthetic_rows())
    args = dict(tokenizer_dir=tokenizer_dir, max_prompt_length=768, val_size=4)
    code = pdm.convert(str(src), str(tmp_path / "code"), prompt_mode="sft_think_code", **args)
    assert code["data_source"] == "orbit/code" and code["agent_name"] == "orbit_sft_think_code"
    _check_dataset(tmp_path / "code", "sft_think_code", 768, tokenizer, tokenizer_dir)
    chat = pdm.convert(str(src), str(tmp_path / "chat"), prompt_mode="sft_chat_no_think", **args)
    assert chat["data_source"] == "orbit/none"
    # An explicit data source wins over the mode's default.
    explicit = pdm.convert(str(src), str(tmp_path / "math_ru"), prompt_mode="sft_think_ru",
                           data_source="orbit/math", **args)
    assert explicit["data_source"] == "orbit/math"


def test_force_only_replaces_our_dirs(tmp_path, tokenizer_dir):
    src = _write_jsonl(tmp_path / "src.jsonl", _synthetic_rows())
    args = dict(prompt_mode="sft_think", tokenizer_dir=tokenizer_dir, max_prompt_length=768, val_size=4)
    out = tmp_path / "out"
    pdm.convert(str(src), str(out), **args)
    with pytest.raises(FileExistsError):
        pdm.convert(str(src), str(out), **args)
    manifest = pdm.convert(str(src), str(out), force=True, limit=3, **args)
    assert manifest["rows"]["train"] == 3
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("not ours")
    with pytest.raises(FileExistsError):
        pdm.convert(str(src), str(foreign), force=True, **args)
    assert (foreign / "keep.txt").read_text() == "not ours"


def test_bad_arguments(tmp_path, tokenizer_dir):
    src = _write_jsonl(tmp_path / "src.jsonl", _synthetic_rows())
    args = dict(tokenizer_dir=tokenizer_dir, max_prompt_length=768)
    with pytest.raises(ValueError):
        pdm.convert(str(src), str(tmp_path / "o1"), prompt_mode="nope", **args)
    # GSM8K few-shot RL is not part of the release, so its data source is unknown like any other.
    for data_source in ("openai/gsm8k", "orbit/gsm8k"):
        with pytest.raises(ValueError, match="Unknown data_source"):
            pdm.convert(str(src), str(tmp_path / "o2"), prompt_mode="sft_think", data_source=data_source, **args)
    with pytest.raises(FileNotFoundError):
        pdm.convert(str(tmp_path / "missing.jsonl"), str(tmp_path / "o3"), prompt_mode="sft_think", **args)
    with pytest.raises(KeyError, match="question-field"):
        pdm.convert(str(src), str(tmp_path / "o4"), prompt_mode="sft_think", question_field="question", **args)
    assert pdm.main(["--src", str(tmp_path / "missing.jsonl"), "--prompt-mode", "sft_think", "--tokenizer",
                     tokenizer_dir, "--max-prompt-length", "10", "--out-dir", str(tmp_path / "o5"),
                     "--runs-root", RUNS_ROOT]) == 2
    assert not (tmp_path / "o5").exists()


def test_the_cli_offers_the_release_modes():
    parser = pdm._build_parser()
    choices = next(action.choices for action in parser._actions if action.dest == "prompt_mode")
    assert sorted(choices) == sorted(PROMPT_MODES)
    assert not [action for action in parser._actions if "fewshot" in action.dest]


def test_cli_writes_dataset(tmp_path, tokenizer, tokenizer_dir, capsys):
    src = _write_jsonl(tmp_path / "src.jsonl", _synthetic_rows())
    code = pdm.main([
        "--src", str(src), "--prompt-mode", "sft_chat_think", "--tokenizer", tokenizer_dir,
        "--max-prompt-length", "256", "--out-dir", str(tmp_path / "chat"), "--val-size", "2",
        "--runs-root", RUNS_ROOT,
    ])
    assert code == 0
    assert "rows: train" in capsys.readouterr().out
    train, _ = _check_dataset(tmp_path / "chat", "sft_chat_think", 256, tokenizer, tokenizer_dir)
    assert [m["role"] for m in train[0]["prompt"]] == ["user"]
    assert train[0]["data_source"] == "orbit/none"


def test_verl_rlhf_dataset_loads_and_mixes(tmp_path, tokenizer, tokenizer_dir):
    """The files must load through verl's RLHFDataset with the stock data defaults, also mixed."""
    from omegaconf import OmegaConf

    import verl
    from verl.utils.dataset.rl_dataset import RLHFDataset

    src = _write_jsonl(tmp_path / "src.jsonl", _synthetic_rows())
    modes = ("sft_think", "sft_chat_no_think")
    for mode in modes:
        pdm.convert(
            str(src), str(tmp_path / mode), prompt_mode=mode, tokenizer_dir=tokenizer_dir,
            max_prompt_length=768, val_size=4, data_source="orbit/math",
        )
    config = OmegaConf.load(Path(verl.__file__).parent / "trainer" / "config" / "data" / "legacy_data.yaml")
    config.max_prompt_length = 768
    config.cache_dir = str(tmp_path / "verl_cache")
    files = [str(tmp_path / mode / pdm.TRAIN_FILE) for mode in modes]
    dataset = RLHFDataset(data_files=files, tokenizer=tokenizer, config=config)
    stored = _rows(Path(files[0])) + _rows(Path(files[1]))
    assert len(dataset) == len(stored)
    for position in (0, len(stored) - 1):
        item, row = dataset[position], stored[position]
        assert item["raw_prompt"] == row["prompt"]
        assert item["agent_name"] == row["agent_name"]
        assert item["data_source"] == "orbit/math"
        assert item["reward_model"]["ground_truth"] == row["reward_model"]["ground_truth"]
        assert item["index"] == row["extra_info"]["index"]


def test_tokenizer_family_is_recorded(tmp_path, tokenizer_dir):
    src = _write_jsonl(tmp_path / "src.jsonl", _synthetic_rows())
    args = dict(prompt_mode="sft_think", max_prompt_length=768, val_size=4)
    # An export dir has a config.json, whose vocabulary size names the family.
    qwen3 = pdm.convert(str(src), str(tmp_path / "qwen3"), tokenizer_dir=tokenizer_dir, **args)
    assert qwen3["tokenizer_family"] == "qwen3"
    # A bare tokenizer dir has no config.json, so only its fingerprint ties the dataset to it.
    bare_dir = tmp_path / "bare_tokenizer"
    bare_dir.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
                 "chat_template.jinja", "added_tokens.json"):
        if (Path(tokenizer_dir) / name).is_file():
            shutil.copyfile(Path(tokenizer_dir) / name, bare_dir / name)
    bare = pdm.convert(str(src), str(tmp_path / "bare"), tokenizer_dir=str(bare_dir), **args)
    assert bare["tokenizer_family"] is None
    assert bare["tokenizer_sha256"] == qwen3["tokenizer_sha256"]
    # The file source is recorded with its checksum, so a changed file is visible later.
    assert bare["source"] == {"kind": "file", "path": str(src.resolve()), "sha256": common.sha256_file(src),
                              "question_field": "instruction", "answer_field": "response",
                              "rows": len(_synthetic_rows())}


def test_tokenizer_family_without_a_tokenizer(tmp_path):
    """The family comes from a prepared dir's manifest or from config.json, so no tokenizer is needed."""
    prepared = tmp_path / "model"
    prepared.mkdir()
    common.write_manifest(prepared / common.MODEL_MANIFEST, common.MODEL_MANIFEST_KIND, {"preset": "qwen3"})
    assert pdm.tokenizer_family(prepared) == "qwen3"
    export = tmp_path / "export"
    export.mkdir()
    (export / "config.json").write_text(json.dumps({"vocab_size": 151936}), encoding="utf-8")
    assert pdm.tokenizer_family(export) == "qwen3"
    # A vocabulary that no preset knows, such as the old 49152 one, has no family.
    (export / "config.json").write_text(json.dumps({"vocab_size": 49152}), encoding="utf-8")
    assert pdm.tokenizer_family(export) is None
    assert pdm.tokenizer_family(tmp_path / "missing") is None


def test_no_runs_root_is_refused(tmp_path, tokenizer_dir, monkeypatch):
    src = _write_jsonl(tmp_path / "src.jsonl", _synthetic_rows())
    args = ["--src", str(src), "--prompt-mode", "sft_think", "--tokenizer", tokenizer_dir,
            "--max-prompt-length", "768", "--out-dir", str(tmp_path / "out"), "--val-size", "4"]
    monkeypatch.delenv(common.RUNS_ROOT_ENV, raising=False)
    assert pdm.main(args) == 2
    monkeypatch.setenv(common.RUNS_ROOT_ENV, RUNS_ROOT)
    assert pdm.main(args) == 0
    # The API refuses too when nobody placed the caches.
    monkeypatch.delenv(common.RUNS_ROOT_ENV)
    monkeypatch.delenv("HF_MODULES_CACHE")
    with pytest.raises(common.RunsRootError, match="runs root"):
        pdm.convert(str(src), str(tmp_path / "api"), prompt_mode="sft_think", tokenizer_dir=tokenizer_dir,
                    max_prompt_length=768)
