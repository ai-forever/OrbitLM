"""CPU tests of the in-training evaluation (src/orbit/in_training_eval, imported through train_eval.py).

Run from the repository root with the trainer's environment:

    PYTHONPATH=src python -m pytest tests/train/test_train_eval.py

Nothing here needs a GPU, lm-eval or vLLM. The orchestration tests start two CPU processes with
the gloo backend and a stand-in for lm-eval, so the collectives, the file barrier, the
aggregation and the summary run exactly as in training. The local task helpers, run_lm_eval.py
and the lm-eval checks run against small stand-ins for the lm-eval modules they use. The
stand-ins come first on the path and include every lm-eval module that the code under test
imports, so the environment's own lm-eval is never loaded. The export test needs a real tokenizer
folder in ORBIT_TEST_TOKENIZER_DIR and is skipped without one. The tests of leftover evaluation
processes read /proc and are skipped where it does not exist. So are the tests that the
evaluation environment's libstdc++ really gets loaded, by the preload and by run_lm_eval.py. Those
also need a conda interpreter. The preload test is also skipped where torch gets the environment's
libstdc++ even without the preload, because there the preload has nothing to change. Two MBPP tests
need parts of the evaluation environment and are skipped without them: the comparison with the stock
mbpp task needs the lm-eval 0.4.13 folder in ORBIT_TEST_LM_EVAL_SITE, and the run of programs through
evaluate's code_eval needs ORBIT_TEST_EVAL_PYTHON and an offline cache in ORBIT_TEST_EVAL_CACHE.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import types
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from orbit import in_training_eval as te  # noqa: E402
from orbit import train_eval  # noqa: E402
from orbit.experiment_config import load_experiment_config  # noqa: E402
# The modules whose own names some tests replace, because the code under test looks them up there.
from orbit.in_training_eval import environment as te_environment  # noqa: E402
from orbit.in_training_eval import evaluator as te_evaluator  # noqa: E402
from orbit.in_training_eval import preflight as te_preflight  # noqa: E402
from orbit.in_training_eval import settings as te_settings  # noqa: E402
from orbit.runtime import report_scalar  # noqa: E402

PACKAGE_DIR = REPO_ROOT / "src" / "orbit"
TRAIN_CONFIGS = REPO_ROOT / "configs" / "train"
FIXTURES = REPO_ROOT / "tests" / "train" / "fixtures" / "in_training_eval"
BASE_SUITE = REPO_ROOT / "eval" / "in_training" / "suites" / "base.yaml"
SMOKE_SUITE = REPO_ROOT / "eval" / "in_training" / "suites" / "smoke.yaml"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def parse_eval_args(config: dict, argv: list[str] | None = None) -> argparse.Namespace:
    """The evaluation options exactly as train.parse_args builds them, for a flattened config."""
    parser = argparse.ArgumentParser()
    train_eval.add_eval_arguments(parser, lambda key, fallback=None: config.get(key, fallback))
    return parser.parse_args(argv or [])


class ParserError(Exception):
    pass


def raise_error(message: str) -> None:
    raise ParserError(message)


def write_suite(folder: Path, body: str, *, entry: Path | None = None) -> Path:
    tasks = folder / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    entry = entry or (folder / "entry.py")
    if not entry.exists():
        entry.write_text("")
    path = folder / "suite.yaml"
    path.write_text(f"tasks_dir: {tasks}\nentry: {entry}\n{body}")
    return path


@pytest.fixture
def eval_env(tmp_path):
    """A complete, valid set of evaluation settings on the local filesystem."""
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text(json.dumps({
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "bos_token": None,
        "added_tokens_decoder": {
            "151643": {"content": "<|endoftext|>"},
            "151644": {"content": "<|im_start|>"},
            "151645": {"content": "<|im_end|>"},
        },
    }))
    cache = tmp_path / "cache"
    cache.mkdir()
    return {
        "tokenizer": tokenizer,
        "cache": cache,
        "eval_python": Path(sys.executable),
    }


def make_args(eval_env, **overrides) -> argparse.Namespace:
    values = dict(
        eval_every=1000,
        eval_at_start=False,
        eval_suite=BASE_SUITE,
        eval_python=eval_env["eval_python"],
        eval_harness_path=None,
        eval_extra_pythonpath=None,
        eval_max_model_len=None,
        eval_gpu_memory_utilization=0.75,
        eval_timeout_minutes=30.0,
        eval_total_timeout_minutes=120.0,
        eval_weights="raw",
        eval_keep_exports=False,
        eval_cache_dir=eval_env["cache"],
        eval_tokenizer_source=None,
        tokenizer_config=eval_env["tokenizer"] / "tokenizer_config.json",
        save_every=1000,
        block_size=16384,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def make_settings(suite: "te.Suite", root: Path | None = None, **overrides) -> "te.EvalSettings":
    root = root or Path("/nonexistent")
    values = dict(
        every=10, at_start=False, suite=suite, eval_python=Path(sys.executable), harness_path=None,
        extra_pythonpath=None, max_model_len=4096, gpu_memory_utilization=0.5,
        timeout_minutes=1.0, total_timeout_minutes=1.0, weights="raw", keep_exports=False,
        cache_dir=root / "cache", tokenizer_source=root, tokenizer_config=root / "tok.json",
    )
    values.update(overrides)
    return te.EvalSettings(**values)


# ---------------------------------------------------------------------------
# Defaults keep the trainer unchanged
# ---------------------------------------------------------------------------


# The variables that the release configs name. Loading a config only expands them, so the dummy
# values below need not exist.
CONFIG_VARIABLES = ("ORBIT_DATA", "ORBIT_RUNS", "ORBIT_EXPORTS", "ORBIT_TOKENIZERS", "ORBIT_EVAL_CACHE",
                    "ORBIT_LM_EVAL", "ORBIT_EVAL_PYTHON")


def load_config(path: Path) -> dict:
    """Loads a trainer config, with a dummy value for every config variable that is not set."""
    added = [name for name in CONFIG_VARIABLES if not os.environ.get(name)]
    for name in added:
        os.environ[name] = f"/nonexistent/{name.lower()}"
    try:
        return load_experiment_config(path)
    finally:
        for name in added:
            del os.environ[name]


def all_trainer_configs() -> list[Path]:
    """Every trainer config of the repository that does not switch the evaluation on.

    Left out are the config that tests the evaluation itself and the configs that set eval_* keys,
    which are the 32K stage and the SFT stage. Every other config must keep the evaluation off.
    """
    configs = sorted(TRAIN_CONFIGS.rglob("*.yaml")) + sorted((REPO_ROOT / "tests" / "train" / "configs").glob("*.yaml"))
    return [
        path for path in configs
        if "in_training_eval" not in path.name
        and not any(str(key).startswith("eval_") for key in load_config(path))
    ]


def test_the_configs_without_the_evaluation_include_the_pretraining_and_smoke_configs():
    names = {str(path.relative_to(TRAIN_CONFIGS)) for path in all_trainer_configs() if TRAIN_CONFIGS in path.parents}
    assert {"pretrain/stage1_4k.yaml", "pretrain/stage2_4k.yaml", "pretrain/stage3_16k.yaml",
            "smoke/pretrain_4k.yaml", "smoke/context_16k.yaml", "smoke/sft.yaml"} <= names


@pytest.mark.parametrize("config_path", all_trainer_configs(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_existing_configs_leave_the_evaluation_off(config_path):
    config = load_config(config_path)
    args = parse_eval_args(config)
    assert args.eval_every == 0
    assert args.eval_at_start is False
    assert not te.eval_requested(args)
    before = dict(vars(args))
    te.validate_eval_arguments(args, raise_error)
    assert vars(args) == before


def test_the_32k_stage_evaluates_every_1000_steps_and_its_first_and_last_step():
    config = load_config(TRAIN_CONFIGS / "pretrain" / "stage4_32k.yaml")
    args = parse_eval_args(config)
    assert (args.eval_every, args.eval_at_start, args.eval_at_end) == (1000, True, True)
    assert te.eval_requested(args)
    # Every evaluated step must be a checkpoint step.
    assert args.eval_every % int(config["save_every"]) == 0


def test_train_py_touches_the_evaluator_only_when_it_exists():
    """Every use of the evaluator in train.py sits under 'if in_training_eval is not None'.

    With the defaults build_in_training_eval returns None, so this proves that a run without
    the evaluation keys executes exactly the training code it executed before.
    """
    tree = ast.parse((PACKAGE_DIR / "train.py").read_text())
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def guarded(node: ast.AST) -> bool:
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "in_training_eval"
                    and isinstance(test.ops[0], ast.IsNot)
                ):
                    return True
        return False

    uses = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "in_training_eval"
    ]
    assert len(uses) == 3  # attach and maybe_run at start, maybe_run in the loop
    assert len([n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "maybe_run"]) == 2
    assert all(guarded(node) for node in uses)


def test_build_returns_none_without_evaluation(tmp_path):
    args = argparse.Namespace(eval_every=0, eval_at_start=False)
    assert train_eval.build_in_training_eval(
        args, rank=0, local_rank=0, world_size=1, distributed=False, device=None, run_dir=tmp_path
    ) is None


# ---------------------------------------------------------------------------
# Configuration keys and validation
# ---------------------------------------------------------------------------


def test_yaml_values_get_their_types(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "eval_every: '5000'\neval_gpu_memory_utilization: '0.6'\neval_suite: suites/x.yaml\n"
        "eval_max_model_len: 32768\neval_timeout_minutes: 20\neval_at_start: true\n"
    )
    config = load_experiment_config(config_path)
    assert config["eval_every"] == 5000
    assert config["eval_gpu_memory_utilization"] == pytest.approx(0.6)
    assert isinstance(config["eval_timeout_minutes"], float)
    assert config["eval_suite"] == (tmp_path / "suites" / "x.yaml").resolve()
    args = parse_eval_args(config, ["--eval-every", "7000"])
    assert args.eval_every == 7000  # the command line overrides the file
    assert args.eval_at_start is True


def test_valid_settings_fill_the_defaults(eval_env):
    args = make_args(eval_env)
    te.validate_eval_arguments(args, raise_error)
    # The longest RULER length of base.yaml plus room for the answer prefix of RULER prompts.
    assert args.eval_max_model_len == 32768 + te.RULER_PROMPT_MARGIN
    assert args.eval_tokenizer_source == eval_env["tokenizer"].resolve()
    args = make_args(eval_env, block_size=65536)
    te.validate_eval_arguments(args, raise_error)
    assert args.eval_max_model_len == 65536


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(eval_every=-1), "must be 0"),
        (dict(eval_every=1500), "multiple of --save-every"),
        (dict(eval_python=None), "eval_python"),
        (dict(eval_python=Path("/nonexistent/python")), "does not exist"),
        (dict(eval_cache_dir=None), "eval_cache_dir"),
        (dict(eval_gpu_memory_utilization=1.5), "between"),
        (dict(eval_timeout_minutes=0), "positive"),
        (dict(eval_max_model_len=16384), "below the longest RULER length"),
        (dict(eval_max_model_len=32768), "answer prefix"),
        (dict(eval_harness_path=Path("/nonexistent")), "lm_eval"),
    ],
)
def test_bad_settings_are_refused_at_startup(eval_env, overrides, message):
    with pytest.raises(ParserError, match=message):
        te.validate_eval_arguments(make_args(eval_env, **overrides), raise_error)


def test_tokenizer_folder_without_tokenizer_files_is_refused(eval_env, tmp_path):
    package_like = tmp_path / "package"
    package_like.mkdir()
    (package_like / "tokenizer_config.json").write_text("{}")
    args = make_args(eval_env, tokenizer_config=package_like / "tokenizer_config.json")
    with pytest.raises(ParserError, match="eval_tokenizer_source"):
        te.validate_eval_arguments(args, raise_error)


def test_eval_at_start_alone_turns_the_evaluation_on(eval_env):
    args = make_args(eval_env, eval_every=0, eval_at_start=True, eval_python=None)
    with pytest.raises(ParserError, match="eval_python"):
        te.validate_eval_arguments(args, raise_error)


def record_digest(data: bytes) -> str:
    """A file hash in the form that pip writes into RECORD."""
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def write_record(folder: Path, dist_info: Path) -> None:
    """Writes the RECORD file of dist_info the way pip does, for the lm_eval package in folder."""
    rows = []
    for path in sorted([*(folder / "lm_eval").rglob("*"), *dist_info.iterdir()]):
        if path.is_file() and path.name != "RECORD":
            rows.append(f"{path.relative_to(folder).as_posix()},{record_digest(path.read_bytes())},{path.stat().st_size}")
    rows.append(f"{dist_info.name}/RECORD,,")
    (dist_info / "RECORD").write_text("\n".join(rows) + "\n")


def fake_lm_eval(
    folder: Path, version: str, *, dist_info: str | None = None, init: str | None = None,
    files: dict[str, str] | None = None,
) -> Path:
    """A folder laid out like 'pip install --target': an lm_eval package and its metadata folder.

    By default the package only reports its version. The metadata names the release, as pip
    writes it, and its RECORD file lists the hash of every file. ``init`` replaces the package's
    __init__.py, and ``files`` adds more files to the package, by path relative to it.
    """
    package = folder / "lm_eval"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"__version__ = {version!r}\n" if init is None else init)
    for relative, text in (files or {}).items():
        (package / relative).parent.mkdir(parents=True, exist_ok=True)
        (package / relative).write_text(text)
    metadata = folder / f"lm_eval-{dist_info or version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: lm_eval\nVersion: {dist_info or version}\n")
    write_record(folder, metadata)
    return folder


def test_harness_folder_must_hold_the_pinned_lm_eval(eval_env, tmp_path):
    pinned = fake_lm_eval(tmp_path / "pinned", te.LM_EVAL_VERSION)
    args = make_args(eval_env, eval_harness_path=pinned)
    te.validate_eval_arguments(args, raise_error)
    assert args.eval_harness_path == pinned

    # A pip --target install of another release.
    other = fake_lm_eval(tmp_path / "other", "0.4.9.1")
    with pytest.raises(ParserError, match=f"metadata of lm-eval {re.escape(te.LM_EVAL_VERSION)} alone.*0.4.9.1"):
        te.validate_eval_arguments(make_args(eval_env, eval_harness_path=other), raise_error)

    # The pinned release installed over an older one: two metadata folders.
    stacked = fake_lm_eval(tmp_path / "stacked", te.LM_EVAL_VERSION)
    (stacked / "lm_eval-0.4.12.dist-info").mkdir()
    with pytest.raises(ParserError, match="0.4.12.dist-info"):
        te.validate_eval_arguments(make_args(eval_env, eval_harness_path=stacked), raise_error)

    # A bare source checkout has no metadata at all. The message says how to install the release,
    # into a new folder rather than over the checkout, and with the evaluation interpreter.
    checkout = tmp_path / "checkout"
    (checkout / "lm_eval").mkdir(parents=True)
    with pytest.raises(ParserError, match="no lm_eval metadata") as refused:
        te.validate_eval_arguments(make_args(eval_env, eval_harness_path=checkout), raise_error)
    command = te.lm_eval_install_command("<new folder>", eval_env["eval_python"])
    assert command in str(refused.value) and str(checkout) not in command
    assert command.startswith(f"{eval_env['eval_python']} -m pip install")
    assert f"wonderwords=={te.WONDERWORDS_VERSION}" in command


def test_lm_eval_files_must_be_those_that_pip_recorded(tmp_path):
    # Without --no-compile, pip compiles the package and records the bytecode files with hashes.
    release = fake_lm_eval(tmp_path / "site", te.LM_EVAL_VERSION, files={
        "models/vllm_causallms.py": "LLM = None\n", "__pycache__/__init__.cpython-312.pyc": "compiled by pip",
    })
    assert te.verify_lm_eval_files(release) == 2  # the two source files of the package
    # Python may rewrite or add bytecode files at any time, so they are not compared.
    (release / "lm_eval" / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"\x00")
    (release / "lm_eval" / "models" / "__pycache__").mkdir()
    (release / "lm_eval" / "models" / "__pycache__" / "vllm_causallms.cpython-312.pyc").write_bytes(b"\x00")
    assert te.verify_lm_eval_files(release) == 2

    added = release / "lm_eval" / "models" / "patched.py"
    added.write_text("")
    with pytest.raises(ValueError, match="1 files are not part of the release, for example lm_eval/models/patched.py"):
        te.verify_lm_eval_files(release)
    added.unlink()

    edited = release / "lm_eval" / "models" / "vllm_causallms.py"
    edited.write_text("LLM = object\n")
    with pytest.raises(ValueError, match="1 files differ from the release, for example lm_eval/models/vllm_causallms.py"):
        te.verify_lm_eval_files(release)
    edited.unlink()
    with pytest.raises(ValueError, match=re.escape("lm_eval/models/vllm_causallms.py (missing)")):
        te.verify_lm_eval_files(release)


# ---------------------------------------------------------------------------
# Suite files
# ---------------------------------------------------------------------------


def test_repository_suites_load():
    base = te.load_suite(BASE_SUITE)
    assert list(base.benchmarks) == [
        "math", "gsm8k", "mmlu", "mmlu_pro", "hellaswag", "winogrande", "cruxeval", "humaneval", "mbpp", "ruler",
    ]
    assert base.ruler_lengths == (4096, 8192, 16384, 32768)
    assert base.entry.name == "run_lm_eval.py" and base.tasks_dir.is_dir()
    smoke = te.load_suite(SMOKE_SUITE)
    assert smoke.ruler_lengths == (32768,)
    assert smoke.benchmarks["hellaswag"] == te.BenchmarkSettings(limit=8, shards=2)


def test_every_suite_task_has_a_task_file():
    """The local tasks named by the registry exist; the stock MMLU and MMLU-Pro come from lm-eval."""
    suite = te.load_suite(BASE_SUITE)
    for unit in te.expand_units(suite):
        for task in unit.tasks:
            if task.startswith("itr_"):
                assert (suite.tasks_dir / f"{task}.yaml").is_file(), task


@pytest.mark.parametrize(
    "body, message",
    [
        ("benchmarks:\n  mmlu: {}\n  squad: {}\n", "unknown benchmark 'squad'"),
        ("benchmarks:\n  mmlu: {shard: 2}\n", "unknown keys"),
        ("benchmarks:\n  mmlu: {shards: 2}\n", "cannot be sharded"),
        ("benchmarks:\n  hellaswag: {shards: 3, limit: 2}\n", "more shards"),
        ("benchmarks:\n  ruler: {lengths: [5000]}\n", "not among the supported"),
        ("benchmarks:\n  ruler: {lengths: [4096], num_samples: 600}\n", "at most 500"),
        ("benchmarks: {}\n", "non-empty"),
        ("benchmark:\n  mmlu: {}\n", "unknown suite keys"),
    ],
)
def test_bad_suites_are_refused(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        te.load_suite(write_suite(tmp_path, body))


def test_unknown_benchmark_stops_validation(eval_env, tmp_path):
    suite = write_suite(tmp_path, "benchmarks:\n  mmlu: {}\n  arc: {}\n")
    with pytest.raises(ParserError, match="unknown benchmark 'arc'"):
        te.validate_eval_arguments(make_args(eval_env, eval_suite=suite), raise_error)


# ---------------------------------------------------------------------------
# Units and the LPT plan
# ---------------------------------------------------------------------------


def test_units_cover_the_suite():
    units = te.expand_units(te.load_suite(BASE_SUITE))
    by_benchmark: dict[str, list[te.Unit]] = {}
    for unit in units:
        by_benchmark.setdefault(unit.benchmark, []).append(unit)
    assert len(by_benchmark["mmlu"]) == 57
    assert len(by_benchmark["mmlu_pro"]) == 14
    assert len(by_benchmark["math"]) == 7
    assert len(by_benchmark["cruxeval"]) == 2
    assert [(u.tasks, u.docs, u.signature) for u in by_benchmark["mbpp"]] == [(("itr_mbpp",), 500, ())]
    assert len(by_benchmark["ruler"]) == 8 * 4
    shards = by_benchmark["hellaswag"]
    assert len(shards) == 4
    covered = sorted(i for unit in shards for i in range(unit.samples[1], unit.samples[2]))
    assert covered == list(range(10042))  # the shards partition the documents exactly
    assert all(u.signature == ("--num_fewshot", "5") for u in by_benchmark["mmlu"])
    assert len({u.signature for u in by_benchmark["ruler"]}) == 4  # one --metadata per length
    assert len({u.name for u in units}) == len(units)


def test_a_limit_with_shards_splits_the_first_documents(tmp_path):
    suite = te.load_suite(write_suite(tmp_path, "benchmarks:\n  hellaswag: {shards: 2, limit: 9}\n"))
    units = te.expand_units(suite)
    assert [u.samples for u in units] == [("itr_hellaswag", 0, 4), ("itr_hellaswag", 4, 9)]
    # The limit lives in the document lists, because lm-eval refuses --limit with --samples.
    assert all("--limit" not in u.signature for u in units)
    assert sum(u.docs for u in units) == 9


def test_plan_is_deterministic_and_complete():
    suite = te.load_suite(BASE_SUITE)
    units = te.expand_units(suite)
    reference = te.assign_units(units, 64, suite.process_startup_seconds)
    shuffled = list(units)
    random.Random(7).shuffle(shuffled)
    again = te.assign_units(shuffled, 64, suite.process_startup_seconds)
    assert [[u.name for u in r] for r in reference] == [[u.name for u in r] for r in again]
    names = [u.name for rank in reference for u in rank]
    assert sorted(names) == sorted(u.name for u in units)
    assert te.build_plan(suite, 64).to_dict() == te.build_plan(suite, 64).to_dict()


@pytest.mark.parametrize("world_size", [1, 8, 64, 128])
def test_plan_is_balanced(world_size):
    suite = te.load_suite(BASE_SUITE)
    plan = te.build_plan(suite, world_size)
    startup = suite.process_startup_seconds
    total = sum(u.cost for u in plan.units)
    largest = max(u.cost for u in plan.units)
    # A lower bound on any schedule: the work spread evenly, or the largest unit with one
    # startup. LPT stays within 4/3 of the optimum; the startup terms get one more startup of slack.
    bound = max(total / world_size + startup, largest + startup)
    assert plan.makespan_seconds <= 4.0 / 3.0 * bound + startup
    if world_size >= len(plan.units):
        assert plan.makespan_seconds == pytest.approx(largest + startup)


def test_ranks_without_units_have_no_processes():
    suite = te.load_suite(SMOKE_SUITE)
    plan = te.build_plan(suite, 512)
    idle = [rank for rank, procs in enumerate(plan.processes) if not procs]
    assert len(idle) == 512 - sum(1 for procs in plan.processes if procs)
    assert idle and plan.estimated_seconds[idle[0]] == 0.0


def test_one_rank_groups_units_by_process_key():
    suite = te.load_suite(SMOKE_SUITE)
    plan = te.build_plan(suite, 1)
    processes = plan.processes[0]
    keys = [te.process_key(p.units[0]) for p in processes]
    assert len(keys) == len(set(keys))
    assert all(te.process_key(unit) == te.process_key(p.units[0]) for p in processes for unit in p.units)
    # Every task without process-wide arguments shares one process, except MBPP, which runs in a
    # process of its own right after the shared one.
    shared, own = [p for p in processes if p.signature == ("--limit", "4")]
    assert {"itr_minerva_algebra", "itr_gsm8k", "itr_cruxeval_input", "itr_humaneval"} <= set(shared.tasks)
    assert "itr_mbpp" not in shared.tasks
    assert own.tasks == ["itr_mbpp"] and own.index == shared.index + 1


SFT_SUITE = REPO_ROOT / "eval" / "in_training" / "suites" / "sft.yaml"
SFT_SMOKE_SUITE = REPO_ROOT / "eval" / "in_training" / "suites" / "sft_smoke.yaml"
MBPP_SUITES = [BASE_SUITE, SMOKE_SUITE, SFT_SUITE, SFT_SMOKE_SUITE]


@pytest.mark.parametrize("suite_path", MBPP_SUITES, ids=lambda p: p.name)
@pytest.mark.parametrize("world_size", [1, 8, 64])
def test_mbpp_runs_in_a_process_of_its_own(suite_path, world_size):
    """A failure of code_eval during MBPP's scoring cannot take another benchmark's units with it."""
    suite = te.load_suite(suite_path)
    plan = te.build_plan(suite, world_size)
    processes = [process for procs in plan.processes for process in procs]
    mbpp = [process for process in processes if any(unit.benchmark == "mbpp" for unit in process.units)]
    assert len(mbpp) == 1
    assert {unit.benchmark for unit in mbpp[0].units} == {"mbpp"} and mbpp[0].tasks == ["itr_mbpp"]
    command = te.build_lm_eval_command(
        mbpp[0], eval_python=Path("/py"), suite=suite, model_dir=Path("/model"), out_dir=Path("/out"),
        max_model_len=33024, gpu_memory_utilization=0.75, samples_path=None,
    )
    assert command[command.index("--tasks") + 1] == "itr_mbpp"
    assert command[command.index("--model") + 1] == "vllm" and "--confirm_run_unsafe_code" in command
    # The budget of 512 tokens comes from the task file, so the process gets no --gen_kwargs.
    assert "--gen_kwargs" not in command and "--apply_chat_template" not in command
    # Every other process still groups by its signature alone.
    for process in processes:
        if process is not mbpp[0]:
            assert {te.process_key(unit) for unit in process.units} == {(process.signature, "")}


def test_a_benchmark_with_its_own_process_pays_its_own_startup():
    """MBPP and HumanEval have equal signatures, yet one rank runs them in two processes."""
    suite = te.load_suite(BASE_SUITE)
    units = [unit for unit in te.expand_units(suite) if unit.benchmark in ("humaneval", "mbpp")]
    assert {unit.signature for unit in units} == {()}
    pair = replace(suite, benchmarks={name: suite.benchmarks[name] for name in ("humaneval", "mbpp")})
    plan = te.build_plan(pair, 1)
    assert [process.tasks for process in plan.processes[0]] == [["itr_humaneval"], ["itr_mbpp"]]
    startup = suite.process_startup_seconds
    assert plan.makespan_seconds == pytest.approx(sum(unit.cost for unit in units) + 2 * startup)


def test_shards_in_one_process_merge_their_documents():
    units = [
        te.Unit(name=f"hellaswag/itr_hellaswag#{i}of2", benchmark="hellaswag", tasks=("itr_hellaswag",),
                signature=(), docs=5, cost=1.0, samples=("itr_hellaswag", 5 * i, 5 * i + 5))
        for i in range(2)
    ]
    process = te.EvalProcess(rank=0, index=0, units=tuple(units))
    assert process.tasks == ["itr_hellaswag"]
    assert process.samples == {"itr_hellaswag": list(range(10))}


# ---------------------------------------------------------------------------
# Aggregation from real lm-eval results
# ---------------------------------------------------------------------------


def scored(suite_body: str, tmp_path, results_for, world_size=4):
    suite = te.load_suite(write_suite(tmp_path, suite_body))
    plan = te.build_plan(suite, world_size)
    processes = [(p, results_for(p)) for procs in plan.processes for p in procs]
    return te.score_processes(processes)


def by_benchmark(process: te.EvalProcess) -> dict:
    """One results file for the whole process, as lm-eval writes it: the fixtures of every
    benchmark that has a unit in the process, merged."""
    merged: dict = {"results": {}, "n-samples": {}}
    for name in sorted({unit.benchmark for unit in process.units}):
        payload = fixture(name)
        merged["results"].update(payload["results"])
        merged["n-samples"].update(payload["n-samples"])
    return merged


def test_scores_match_lm_eval_group_scores(tmp_path):
    """The size-weighted means equal the group rows that lm-eval itself computed."""
    scores = scored("benchmarks:\n  math: {}\n  mmlu: {}\n  mmlu_pro: {}\n", tmp_path, by_benchmark)
    assert scores.benchmarks["math"] == pytest.approx(fixture("math")["results"]["itr_minerva"]["math_verify,none"], abs=1e-12)
    assert scores.benchmarks["mmlu"] == pytest.approx(fixture("mmlu")["results"]["mmlu"]["acc,none"], abs=1e-12)
    assert scores.benchmarks["mmlu_pro"] == pytest.approx(
        fixture("mmlu_pro")["results"]["mmlu_pro"]["exact_match,custom-extract"], abs=1e-12
    )
    assert scores.samples == {"math": 5000, "mmlu": 14042, "mmlu_pro": 12032}
    assert scores.incomplete == {}


def test_scores_of_single_task_benchmarks_use_the_explicit_metrics(tmp_path):
    body = "benchmarks:\n  gsm8k: {}\n  hellaswag: {}\n  winogrande: {}\n  humaneval: {}\n  cruxeval: {}\n  mbpp: {}\n"
    scores = scored(body, tmp_path, by_benchmark)
    assert scores.benchmarks["gsm8k"] == fixture("gsm8k")["results"]["itr_gsm8k"]["exact_match,strict-match"]
    assert scores.benchmarks["hellaswag"] == fixture("hellaswag")["results"]["itr_hellaswag"]["acc_norm,none"]
    assert scores.benchmarks["winogrande"] == fixture("winogrande")["results"]["itr_winogrande"]["acc,none"]
    assert scores.benchmarks["humaneval"] == fixture("humaneval")["results"]["itr_humaneval"]["pass@1,create_test"]
    # The MBPP fixture is the results file of a real lm-eval run of the local task, so it pins the key.
    assert scores.benchmarks["mbpp"] == fixture("mbpp")["results"]["itr_mbpp"]["pass@1,create_test"]
    assert scores.samples["mbpp"] == 500
    crux = fixture("cruxeval")["results"]
    assert scores.parts["cruxeval_input"] == crux["itr_cruxeval_input"]["pass@1,create_test"]
    assert scores.parts["cruxeval_output"] == crux["itr_cruxeval_output"]["pass@1,create_test"]
    assert scores.benchmarks["cruxeval"] == pytest.approx(
        (crux["itr_cruxeval_input"]["pass@1,create_test"] + crux["itr_cruxeval_output"]["pass@1,create_test"]) / 2
    )


def test_ruler_is_scored_per_length(tmp_path):
    body = "benchmarks:\n  ruler: {lengths: [4096, 8192, 16384], num_samples: 500}\n"
    scores = scored(body, tmp_path, by_benchmark, world_size=8)
    results = fixture("ruler")["results"]
    for length in (4096, 8192, 16384):
        expected = sum(results[t][f"{length},none"] for t in te.RULER_TASKS) / 8
        assert scores.ruler[str(length)] == pytest.approx(expected)
    assert scores.ruler["mean"] == pytest.approx(sum(scores.ruler[str(x)] for x in (4096, 8192, 16384)) / 3)


def test_a_ruler_length_that_failed_does_not_hide_the_others(tmp_path):
    body = "benchmarks:\n  ruler: {lengths: [16384, 32768], num_samples: 500}\n"
    scores = scored(body, tmp_path, by_benchmark, world_size=8)
    results = fixture("ruler")["results"]
    # The fixture has no 32768 metric, so all eight 32K units fail; 16K is still reported, and the
    # mean over lengths is not, because it would silently drop the hardest length.
    assert scores.ruler == {"16384": pytest.approx(sum(results[t]["16384,none"] for t in te.RULER_TASKS) / 8)}
    assert len(scores.incomplete["ruler"]) == 8
    # Only the reported length counts. (The fixture's process built three lengths, so lm-eval
    # counted 1500 samples per task there; in training each process builds one length.)
    counts = fixture("ruler")["n-samples"]
    assert scores.samples["ruler"] == sum(counts[t]["effective"] for t in te.RULER_TASKS)


def test_one_failed_ruler_unit_withholds_only_its_length(tmp_path):
    def results_for(process):
        payload = by_benchmark(process)
        if any(unit.length == 8192 for unit in process.units):
            payload["results"].pop("itr_niah_multivalue", None)
        return payload

    body = "benchmarks:\n  ruler: {lengths: [4096, 8192], num_samples: 500}\n"
    scores = scored(body, tmp_path, results_for, world_size=16)
    assert set(scores.ruler) == {"4096"}
    assert scores.incomplete == {"ruler": ["ruler/itr_niah_multivalue@8192"]}


def test_a_missing_subject_withholds_the_benchmark(tmp_path):
    def results_for(process):
        payload = by_benchmark(process)
        payload["results"].pop("mmlu_anatomy", None)
        return payload

    scores = scored("benchmarks:\n  mmlu: {}\n  math: {}\n", tmp_path, results_for)
    assert "mmlu" not in scores.benchmarks
    assert scores.incomplete == {"mmlu": ["mmlu/mmlu_anatomy"]}
    assert "math" in scores.benchmarks


def test_a_failed_process_withholds_its_benchmarks(tmp_path):
    def results_for(process):
        return None if process.rank == 0 else by_benchmark(process)

    scores = scored("benchmarks:\n  mmlu: {}\n", tmp_path, results_for, world_size=4)
    assert scores.benchmarks == {}
    assert scores.incomplete["mmlu"]


@pytest.mark.parametrize("world_size", [1, 2, 3, 9, 11, 64])
def test_shards_are_weighted_by_the_documents_of_each_process(tmp_path, world_size):
    """The stand-in results below report n-samples "effective" = 10042 for every shard process.

    Older lm-eval releases did exactly that with --samples, and the pinned one reports the
    documents of the shard. The score is right in both cases only when the weights come from the
    processes' own document lists, which is what this checks. At some world sizes two shards
    share one process, which then reports one accuracy over the documents of both.
    """
    suite = te.load_suite(write_suite(tmp_path, "benchmarks:\n  hellaswag: {shards: 4}\n"))
    plan = te.build_plan(suite, world_size)
    accuracy = [0.5, 0.6, 0.7, 0.8]
    shard_of: dict[int, int] = {}
    for unit in plan.units:
        shard = int(unit.name.split("#")[1].split("of")[0]) - 1
        shard_of.update({doc: shard for doc in range(unit.samples[1], unit.samples[2])})

    def results_for(process):
        docs = process.samples["itr_hellaswag"]
        value = sum(accuracy[shard_of[doc]] for doc in docs) / len(docs)
        return {"results": {"itr_hellaswag": {"acc_norm,none": value}},
                "n-samples": {"itr_hellaswag": {"original": 10042, "effective": 10042}}}

    scores = te.score_processes([(p, results_for(p)) for procs in plan.processes for p in procs])
    expected = sum(accuracy[shard_of[doc]] for doc in range(10042)) / 10042
    assert scores.benchmarks["hellaswag"] == pytest.approx(expected, abs=1e-12)
    assert scores.samples["hellaswag"] == 10042


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class FakeClearMLLogger:
    def __init__(self):
        self.scalars: list[tuple[str, str, float, int]] = []
        self.flushed = 0

    def report_scalar(self, title, series, value, iteration):
        self.scalars.append((title, series, float(value), int(iteration)))

    def flush(self):
        self.flushed += 1


class FakeWriter:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []
        self.flushed = 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), int(step)))

    def flush(self):
        self.flushed += 1


def test_reporting_uses_the_eval_titles_with_the_weights_as_series():
    scores = te.Scores(
        benchmarks={"mmlu": 0.53, "cruxeval": 0.29},
        parts={"cruxeval_input": 0.31, "cruxeval_output": 0.28},
        ruler={"4096": 0.9, "32768": 0.5, "mean": 0.7},
    )
    logger, writer = FakeClearMLLogger(), FakeWriter()
    te.report_eval_scalars(
        report_scalar, logger, writer, scores=scores, step=5000, weights="raw",
        timings={"wall_min": 12.5}, failures={"units_failed": 0},
    )
    reported = {(title, series): (value, it) for title, series, value, it in logger.scalars}
    assert reported[("eval_mmlu", "raw")] == (0.53, 5000)
    assert reported[("eval_cruxeval_input", "raw")] == (0.31, 5000)
    assert reported[("eval_ruler", "raw_4096")] == (0.9, 5000)
    assert reported[("eval_ruler", "raw_mean")] == (0.7, 5000)
    assert reported[("eval_timing", "wall_min")] == (12.5, 5000)
    assert reported[("eval_failures", "units_failed")] == (0.0, 5000)
    assert {it for *_, it in logger.scalars} == {5000}
    assert ("eval_mmlu/raw", 0.53, 5000) in writer.scalars
    assert logger.flushed == 1 and writer.flushed == 1


def test_reporting_without_clearml_is_a_no_op():
    te.report_eval_scalars(report_scalar, None, None, scores=te.Scores(benchmarks={"mmlu": 0.5}),
                           step=1, weights="raw", timings={}, failures={})


# ---------------------------------------------------------------------------
# File barrier
# ---------------------------------------------------------------------------


def test_file_barrier_waits_for_late_files(tmp_path):
    paths = [tmp_path / f"rank_{i}.json" for i in range(3)]

    def write_later():
        for path in paths:
            time.sleep(0.1)
            path.write_text("{}")

    thread = threading.Thread(target=write_later)
    thread.start()
    missing = te.wait_for_files(paths, deadline=time.monotonic() + 10, poll_seconds=0.02)
    thread.join()
    assert missing == []


def test_file_barrier_gives_up_at_the_deadline(tmp_path):
    (tmp_path / "a").write_text("")
    started = time.monotonic()
    missing = te.wait_for_files([tmp_path / "a", tmp_path / "b"], deadline=started + 0.3, poll_seconds=0.05)
    assert missing == [tmp_path / "b"]
    assert time.monotonic() - started < 2.0


def test_file_barrier_stops_when_training_is_stopped(tmp_path):
    started = time.monotonic()
    missing = te.wait_for_files([tmp_path / "never"], deadline=started + 60, poll_seconds=0.05,
                                should_stop=lambda: True)
    assert missing and time.monotonic() - started < 1.0


def test_file_barrier_stops_waiting_for_a_dead_writer(tmp_path):
    late = tmp_path / "late"
    threading.Timer(0.2, lambda: late.write_text("{}")).start()
    started = time.monotonic()
    missing = te.wait_for_files([tmp_path / "dead", late], deadline=started + 30, poll_seconds=0.02,
                                given_up=lambda path: path.name == "dead")
    assert missing == [tmp_path / "dead"]
    assert time.monotonic() - started < 5.0


def test_heartbeat_watcher_detects_a_silent_rank(tmp_path):
    now = [0.0]
    watcher = te.HeartbeatWatcher(tmp_path, stale_seconds=10, clock=lambda: now[0])
    beat = te.Heartbeat(tmp_path / "rank_00000", interval=1.0)
    assert not watcher.dead(0)  # no file yet: alive
    beat.beat()
    assert not watcher.dead(0)
    now[0] = 5.0
    beat.beat()
    assert not watcher.dead(0)
    now[0] = 14.0
    assert not watcher.dead(0)  # 9 s without a change
    now[0] = 16.0
    assert watcher.dead(0)  # 11 s without a change
    beat.beat()
    assert not watcher.dead(0)  # it beats again
    now[0] = 1000.0
    assert not watcher.dead(7)  # a file that never appears never counts as dead
    # The beats rewrite the file in place with a fixed width, so they need no new disk blocks.
    assert (tmp_path / "rank_00000").stat().st_size == te.Heartbeat.WIDTH


def test_heartbeat_thread_beats_until_stopped(tmp_path):
    beat = te.Heartbeat(tmp_path / "hb" / "rank_00003", interval=0.05)
    beat.start()
    time.sleep(0.4)
    beat.stop()
    count = int((tmp_path / "hb" / "rank_00003").read_text().split()[0])
    assert count >= 3
    time.sleep(0.2)
    assert int((tmp_path / "hb" / "rank_00003").read_text().split()[0]) == count


def test_json_files_are_written_atomically(tmp_path):
    target = tmp_path / "sub" / "x.json"
    te.write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    assert [p.name for p in target.parent.iterdir()] == ["x.json"]


# ---------------------------------------------------------------------------
# Environment and command of the evaluation processes
# ---------------------------------------------------------------------------


def test_launcher_variables_are_scrubbed(tmp_path):
    base = {
        "RANK": "3", "LOCAL_RANK": "3", "WORLD_SIZE": "64", "LOCAL_WORLD_SIZE": "8", "NODE_RANK": "0",
        "MASTER_ADDR": "10.0.0.1", "MASTER_PORT": "29500", "TORCHELASTIC_RUN_ID": "x",
        "TORCHELASTIC_RESTART_COUNT": "0", "GROUP_RANK": "0", "GROUP_WORLD_SIZE": "8", "ROLE_RANK": "3",
        "ROLE_NAME": "default", "PET_NPROC_PER_NODE": "8", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "PYTHONPATH": "/trainer/src", "PYTHONHOME": "/x", "PATH": "/usr/bin", "KEEP_ME": "1",
        "NCCL_DEBUG": "INFO",
    }
    env = te.build_eval_env(
        base, eval_python=Path("/envs/eval/bin/python"), gpu="3", cache_dir=tmp_path,
        rank_cache=tmp_path / "run" / "rank_00003",
        harness_path=Path("/harness"), extra_pythonpath=os.pathsep.join(["/extra/a", "/extra/b"]),
        parent_pid=4242, gpu_uuid="GPU-abc", compile_threads=6,
    )
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "NODE_RANK", "MASTER_ADDR",
                "MASTER_PORT", "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT", "GROUP_RANK",
                "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_NAME", "PET_NPROC_PER_NODE", "PYTHONHOME"):
        assert key not in env, key
    assert env["KEEP_ME"] == "1" and env["NCCL_DEBUG"] == "INFO"
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["PYTHONPATH"] == os.pathsep.join(["/harness", "/extra/a", "/extra/b"])
    assert env["PATH"].split(os.pathsep)[0] == "/envs/eval/bin"
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_EVALUATE_OFFLINE"):
        assert env[key] == "1"
    assert env["VLLM_USE_AOT_COMPILE"] == "0" and env["VLLM_NO_USAGE_STATS"] == "1"
    assert env["HF_ALLOW_CODE_EVAL"] == "1"
    assert env["HF_HOME"] == str(tmp_path / "hf")
    assert env["VLLM_CACHE_ROOT"] == str(tmp_path / "run" / "rank_00003" / "vllm")
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env[te.EVAL_PARENT_ENV] == "4242" and env[te.EVAL_GPU_ENV] == "GPU-abc"
    assert env["TORCHINDUCTOR_COMPILE_THREADS"] == "6"
    other = te.build_eval_env(base, eval_python=Path("/p"), gpu="", cache_dir=tmp_path,
                              rank_cache=tmp_path / "run" / "rank_00004", harness_path=None, extra_pythonpath=None)
    for key in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "HF_MODULES_CACHE"):
        assert env[key] != other[key]
    assert "PYTHONPATH" not in other  # the trainer's own PYTHONPATH never leaks through
    assert te.EVAL_PARENT_ENV not in other and te.EVAL_GPU_ENV not in other


def fake_libstdcxx(path: Path, glibcxx: str, cxxabi: str) -> Path:
    """A stand-in libstdc++ whose string table defines the given newest GLIBCXX and CXXABI versions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ["GLIBCXX_3.4", f"GLIBCXX_{glibcxx}", "CXXABI_1.3", f"CXXABI_{cxxabi}", "CXXABI_TM_1", "GLIBCXX_FORCE_NEW"]
    path.write_bytes(b"\x7fELF" + b"".join(b"\x00" + name.encode() for name in names) + b"\x00")
    return path


def test_libstdcxx_versions_reads_the_newest_versions_of_a_copy(tmp_path):
    stand_in = fake_libstdcxx(tmp_path / "libstdc++.so.6", "3.4.34", "1.3.15")
    versions = te.libstdcxx_versions(stand_in)
    assert versions == {"GLIBCXX": (3, 4, 34), "CXXABI": (1, 3, 15)}
    assert te.describe_libstdcxx_versions(versions) == "CXXABI_1.3.15, GLIBCXX_3.4.34"
    assert te.libstdcxx_versions(tmp_path / "missing" / "libstdc++.so.6") == {}


def test_the_environments_libstdcxx_is_preloaded(tmp_path, monkeypatch):
    # The system folders of the machine that runs the test must not decide the outcome.
    monkeypatch.setattr(te_environment, "SYSTEM_LIBRARY_DIRS", (str(tmp_path / "system"),))
    fake_libstdcxx(tmp_path / "system" / "libstdc++.so.6", "3.4.30", "1.3.13")
    prefix = tmp_path / "envs" / "eval"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin" / "python"
    python.write_text("")
    libstdcxx = fake_libstdcxx(prefix / "lib" / "libstdc++.so.6", "3.4.34", "1.3.15")
    trainer_env = {"LD_PRELOAD": "/opt/hooks/libhook.so",
                   "LD_LIBRARY_PATH": f"{tmp_path / 'system'}:{tmp_path / 'driver'}"}

    def build(base, eval_python=python):
        return te.build_eval_env(base, eval_python=eval_python, gpu="0", cache_dir=None,
                                 harness_path=None, extra_pythonpath=None)

    env = build(trainer_env)
    # The trainer's own preload stays, and its library path is not touched.
    assert env["LD_PRELOAD"] == f"/opt/hooks/libhook.so:{libstdcxx}"
    assert env["LD_LIBRARY_PATH"] == trainer_env["LD_LIBRARY_PATH"]
    # run_lm_eval.py loads the same copy itself unless this variable says that it was left out.
    assert te.EVAL_NO_LIBSTDCXX_ENV not in env
    assert te.EVAL_NO_LIBSTDCXX_ENV not in build({**trainer_env, te.EVAL_NO_LIBSTDCXX_ENV: "left over"})
    # A trainer that already runs with the preload does not get it twice.
    assert build(env)["LD_PRELOAD"] == env["LD_PRELOAD"]
    assert build({})["LD_PRELOAD"] == str(libstdcxx)
    # The python of a virtual environment links to the interpreter of the environment it was made
    # from, and that environment's copy is the one its libraries need.
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(python)
    assert build({}, venv_python)["LD_PRELOAD"] == str(libstdcxx)
    # An interpreter without a libstdc++ of its own leaves LD_PRELOAD as the trainer's environment set it.
    libstdcxx.unlink()
    assert build(trainer_env)["LD_PRELOAD"] == trainer_env["LD_PRELOAD"]
    without = build({})
    assert "LD_PRELOAD" not in without
    assert "has no lib/libstdc++.so.6" in without[te.EVAL_NO_LIBSTDCXX_ENV]


def test_the_environments_libstdcxx_is_not_preloaded_when_another_copy_is_newer(tmp_path, monkeypatch):
    """A preloaded older copy would fail the libraries that need the newer one, so it stays out."""
    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    own = fake_libstdcxx(tmp_path / "env" / "lib" / "libstdc++.so.6", "3.4.32", "1.3.14")
    # Every stand-in copy gets its own folder, so that no file is rewritten under the cache.
    copies = {
        name: fake_libstdcxx(tmp_path / name / "libstdc++.so.6", glibcxx, cxxabi).parent
        for name, glibcxx, cxxabi in [
            ("older", "3.4.30", "1.3.13"), ("same", "3.4.32", "1.3.14"),
            ("newer_glibcxx", "3.4.33", "1.3.14"), ("newer_cxxabi", "3.4.32", "1.3.15"),
        ]
    }

    def build(system: str, ld_library_path: str = "") -> dict[str, str]:
        monkeypatch.setattr(te_environment, "SYSTEM_LIBRARY_DIRS", (str(copies[system]),))
        base = {"LD_LIBRARY_PATH": ld_library_path} if ld_library_path else {}
        return te.build_eval_env(base, eval_python=python, gpu="0", cache_dir=None,
                                 harness_path=None, extra_pythonpath=None)

    # An older or an equally new copy elsewhere: the environment's copy can serve everything.
    assert build("older").get("LD_PRELOAD") == str(own)
    assert build("same").get("LD_PRELOAD") == str(own)
    assert build("older", str(copies["same"])).get("LD_PRELOAD") == str(own)
    # A copy that defines a newer version of either kind, in the system or on LD_LIBRARY_PATH.
    for system, ld_library_path, newer in [
        ("newer_glibcxx", "", "newer_glibcxx"),
        ("newer_cxxabi", "", "newer_cxxabi"),
        ("older", f"/nonexistent:{copies['newer_glibcxx']}", "newer_glibcxx"),
    ]:
        env = build(system, ld_library_path)
        assert "LD_PRELOAD" not in env
        # The variable tells run_lm_eval.py to leave the copy out as well, and it names the newer copy.
        assert str(copies[newer] / "libstdc++.so.6") in env[te.EVAL_NO_LIBSTDCXX_ENV]
    assert build("newer_cxxabi")[te.EVAL_NO_LIBSTDCXX_ENV] == (
        f"the environment's {own} (CXXABI_1.3.14, GLIBCXX_3.4.32) is not preloaded, because "
        f"{copies['newer_cxxabi'] / 'libstdc++.so.6'} defines the newer CXXABI_1.3.15"
    )


@pytest.mark.skipif(
    not Path("/proc/self/maps").exists() or te.environment_libstdcxx(Path(sys.executable)) is None,
    reason="needs /proc and an interpreter whose environment ships libstdc++ (a conda environment)",
)
def test_the_preload_beats_the_system_libstdcxx_that_torch_would_load(tmp_path):
    """The real mechanism, with this interpreter as the evaluation interpreter.

    torch's libraries find the system libstdc++, here put first on LD_LIBRARY_PATH the way a
    training environment can. Without the preload, the process loads that copy when it imports
    torch first. With the environment of an evaluation process, it loads the environment's copy
    instead. The test needs a system copy that is not newer than the environment's, which is the
    case the preload exists for. It also needs torch's libraries to find the system copy when
    nothing is preloaded, as they do in a conda environment on Ubuntu. Where either does not hold,
    it is skipped.
    """
    system = Path("/usr/lib/x86_64-linux-gnu")
    if not (system / "libstdc++.so.6").exists():
        pytest.skip("no system libstdc++ in /usr/lib/x86_64-linux-gnu")
    own = te.environment_libstdcxx(Path(sys.executable)).resolve()
    base = {**os.environ, "LD_LIBRARY_PATH": str(system)}
    base.pop("LD_PRELOAD", None)
    env = te.build_eval_env(base, eval_python=Path(sys.executable), gpu="", cache_dir=None,
                            harness_path=None, extra_pythonpath=None)
    if "LD_PRELOAD" not in env:
        pytest.skip(f"nothing is preloaded here: {env[te.EVAL_NO_LIBSTDCXX_ENV]}")
    without = dict(env)
    without.pop("LD_PRELOAD")
    loaded = te.probe_eval_environment(Path(sys.executable), without, ["torch", "sqlite3"])["libstdcxx"]
    if loaded != [str((system / "libstdc++.so.6").resolve())]:
        pytest.skip(f"without the preload this process already loads {loaded}, so the preload has nothing to beat")
    report = te.probe_eval_environment(Path(sys.executable), env, ["torch", "sqlite3"])
    assert report["errors"] == {}
    assert report["libstdcxx"] == [str(own)]


def test_compile_threads_share_the_cpus_of_a_node():
    assert te.compile_threads_per_rank(1) >= te.compile_threads_per_rank(8) >= 1
    assert te.compile_threads_per_rank(10 ** 6) == 1


def test_online_environment_for_prefetch(tmp_path):
    env = te.build_eval_env({}, eval_python=Path("/p"), gpu="", cache_dir=tmp_path,
                            harness_path=None, extra_pythonpath=None, offline=False)
    assert "HF_HUB_OFFLINE" not in env


def test_rank_cache_links_the_shared_evaluate_metrics(tmp_path):
    shared = tmp_path / "hf" / "modules" / "evaluate_modules"
    shared.mkdir(parents=True)
    own = tmp_path / "run" / "eval" / "cache" / "rank_00005"
    te.prepare_rank_cache(tmp_path, own)
    te.prepare_rank_cache(tmp_path, own)  # a second evaluation finds the link in place
    link = own / "hf_modules" / "evaluate_modules"
    assert link.is_symlink() and link.resolve() == shared.resolve()


def test_commands_carry_the_process_arguments(tmp_path):
    suite = te.load_suite(BASE_SUITE)
    plan = te.build_plan(suite, 64)
    by_kind = {}
    for procs in plan.processes:
        for process in procs:
            by_kind.setdefault(process.units[0].benchmark, process)
    model_dir = tmp_path / "model"

    ruler = te.build_lm_eval_command(
        by_kind["ruler"], eval_python=Path("/py"), suite=suite, model_dir=model_dir,
        out_dir=tmp_path / "out", max_model_len=32768, gpu_memory_utilization=0.7, samples_path=None,
    )
    model_args = ruler[ruler.index("--model_args") + 1]
    assert ruler[:2] == ["/py", str(suite.entry)]
    assert "max_model_len=32768" in model_args and "gpu_memory_utilization=0.700" in model_args
    assert "trust_remote_code=True" in model_args
    metadata = json.loads(ruler[ruler.index("--metadata") + 1])
    assert metadata["pretrained"] == str(model_dir) and metadata["num_samples"] == 100
    assert len(metadata["max_seq_lengths"]) == 1
    assert "--confirm_run_unsafe_code" in ruler
    assert ruler[ruler.index("--include_path") + 1] == str(suite.tasks_dir)

    hella = by_kind["hellaswag"]
    command = te.build_lm_eval_command(
        hella, eval_python=Path("/py"), suite=suite, model_dir=model_dir,
        out_dir=tmp_path / "out", max_model_len=32768, gpu_memory_utilization=0.7,
        samples_path=tmp_path / "samples.json",
    )
    assert command[command.index("--model") + 1] == "vllm"
    assert command[command.index("--batch_size") + 1] == "auto"
    assert command[command.index("--samples") + 1] == str(tmp_path / "samples.json")
    assert "--limit" not in command  # lm-eval refuses --limit together with --samples
    mmlu = te.build_lm_eval_command(
        by_kind["mmlu"], eval_python=Path("/py"), suite=suite, model_dir=model_dir,
        out_dir=tmp_path / "out", max_model_len=32768, gpu_memory_utilization=0.7, samples_path=None,
    )
    assert mmlu[mmlu.index("--num_fewshot") + 1] == "5"


def test_preflight_checks_the_gpu_and_vllm_extension():
    settings = make_settings(te.load_suite(BASE_SUITE))
    with_gpu = te.preflight_modules(settings, cuda=True)
    # vLLM's compiled extensions are loaded through the module that names them in every release
    # (vllm._C exists in 0.22 but not in 0.26).
    assert "vllm.platforms.cuda" in with_gpu and "vllm._C" not in with_gpu
    assert te.CUDA_CHECK in with_gpu and te.MANAGER_CHECK in with_gpu
    # lm-eval's vLLM backend is imported too, so a vLLM whose modules it cannot import stops the
    # run at startup.
    assert "lm_eval.models.vllm_causallms" in with_gpu
    assert te.CUDA_CHECK not in te.preflight_modules(settings)
    # The preflight must end before the NCCL timeout (10 minutes), since the other ranks wait for it.
    assert te.PREFLIGHT_TIMEOUT_SECONDS < 600


def test_preflight_of_a_suite_with_mbpp_alone_checks_code_eval(tmp_path):
    """MBPP needs evaluate and a working multiprocessing manager, as HumanEval does."""
    settings = make_settings(te.load_suite(write_suite(tmp_path, "benchmarks:\n  mbpp: {}\n")))
    modules = te.preflight_modules(settings)
    assert modules[-2:] == ["evaluate", te.MANAGER_CHECK]


def test_preflight_imports_rulers_modules_after_torch(tmp_path):
    """torch comes first, as in an evaluation, so RULER's imports meet the libstdc++ they meet there."""
    modules = te.preflight_modules(make_settings(te.load_suite(BASE_SUITE)), cuda=True)
    assert modules[0] == "torch"
    ruler = [modules.index(name) for name in ("wonderwords", "sqlite3", "nltk")]
    assert min(ruler) > modules.index("vllm.platforms.cuda")
    assert modules.index("sqlite3") < modules.index("nltk")
    without_ruler = te.load_suite(write_suite(tmp_path, "benchmarks:\n  mmlu: {}\n"))
    assert "sqlite3" not in te.preflight_modules(make_settings(without_ruler))


def test_lm_eval_must_be_the_pinned_release_from_the_harness_folder(tmp_path):
    harness = fake_lm_eval(tmp_path / "site", te.LM_EVAL_VERSION, files={"evaluator.py": ""})
    package = harness / "lm_eval"
    inside = {
        "modules": {"lm_eval": {"version": te.LM_EVAL_VERSION, "file": str(package / "__init__.py")}},
        "lm_eval_modules": {
            "lm_eval": [str(package / "__init__.py")], "lm_eval.evaluator": [str(package / "evaluator.py")],
        },
    }
    assert te.lm_eval_problems(inside, harness) == []
    assert inside["lm_eval_files_verified"] == 2
    # Without eval_harness_path, the environment's own install is checked the same way.
    assert te.lm_eval_problems(inside, None) == []
    # An environment's own lm-eval that is an editable install of another checkout.
    editable = {"modules": {"lm_eval": {"version": "0.4.12.dev0", "file": "/home/x/lm-evaluation-harness/lm_eval/__init__.py"}}}
    problems = te.lm_eval_problems(editable, None)
    assert len(problems) == 1 and "0.4.12.dev0" in problems[0] and te.LM_EVAL_VERSION in problems[0]
    assert len(te.lm_eval_problems(editable, harness)) == 2  # wrong release and wrong folder
    # The pinned release, but from another folder than eval_harness_path.
    elsewhere = {"modules": {"lm_eval": {"version": te.LM_EVAL_VERSION, "file": "/other/lm_eval/__init__.py"}}}
    assert te.lm_eval_problems(elsewhere, harness) == [
        f"lm_eval resolves to /other/lm_eval/__init__.py, not to eval_harness_path {harness}"
    ]
    # One module of the package comes from elsewhere, as when the import hook of an editable install
    # serves a module that the package lacks out of another checkout.
    leaked = json.loads(json.dumps(inside))
    leaked["lm_eval_modules"]["lm_eval.models"] = ["/home/x/lm-evaluation-harness/lm_eval/models/__init__.py"]
    problems = te.lm_eval_problems(leaked, harness)
    assert len(problems) == 1 and "1 lm_eval modules come from outside the lm_eval package" in problems[0]
    assert "lm_eval.models from /home/x/lm-evaluation-harness/lm_eval/models/__init__.py" in problems[0]
    # The pinned version from the right folder, but with a file changed in place.
    (package / "evaluator.py").write_text("patched = True\n")
    problems = te.lm_eval_problems(inside, harness, eval_python="/env/bin/python")
    assert len(problems) == 1 and "1 files differ from the release" in problems[0]
    assert "'/env/bin/python -m pip install --no-deps --no-compile --target <new folder> " in problems[0]
    # An lm_eval that did not import is reported by the import check, not twice.
    assert te.lm_eval_problems({"modules": {}}, harness) == []


def test_preflight_refuses_another_lm_eval(tmp_path, monkeypatch):
    """The real preflight subprocess, with lm_eval as the only module to import.

    The stand-in packages come first on PYTHONPATH, so the environment's own lm-eval is not loaded.
    """
    monkeypatch.setattr(te_preflight, "preflight_modules", lambda settings, cuda=False: ["lm_eval"])
    suite = te.load_suite(SMOKE_SUITE)
    pinned = fake_lm_eval(tmp_path / "pinned", te.LM_EVAL_VERSION)
    report = te.run_preflight(make_settings(suite, tmp_path, harness_path=pinned))
    assert report["modules"]["lm_eval"]["version"] == te.LM_EVAL_VERSION
    assert list(report["lm_eval_modules"]) == ["lm_eval"]
    assert report["lm_eval_files_verified"] == 1

    relabelled = fake_lm_eval(tmp_path / "relabelled", "0.4.9.1", dist_info=te.LM_EVAL_VERSION)
    with pytest.raises(RuntimeError, match=f"is version 0.4.9.1, but .* {re.escape(te.LM_EVAL_VERSION)}"):
        te.run_preflight(make_settings(suite, tmp_path, harness_path=relabelled))

    # The harness folder is first on the path but has no lm_eval, so another one is imported.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="not to eval_harness_path"):
        te.run_preflight(make_settings(suite, tmp_path, harness_path=empty, extra_pythonpath=str(pinned)))


def test_preflight_stops_the_run_when_rulers_imports_fail(tmp_path, monkeypatch):
    """A failed import stops the run, and the message names the module, its error and the preload.

    The stand-in nltk raises the error that the real one raises under an older system libstdc++.
    The test covers only the message, not the loading of libstdc++, which
    test_the_preload_beats_the_system_libstdcxx_that_torch_would_load covers.
    """
    monkeypatch.setattr(te_preflight, "preflight_modules", lambda settings, cuda=False: ["lm_eval", "sqlite3", "nltk"])
    pinned = fake_lm_eval(tmp_path / "pinned", te.LM_EVAL_VERSION)
    broken = tmp_path / "broken" / "nltk"
    broken.mkdir(parents=True)
    (broken / "__init__.py").write_text(
        "raise ImportError(\"/usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found\")\n"
    )
    settings = make_settings(te.load_suite(SMOKE_SUITE), tmp_path, harness_path=pinned,
                             extra_pythonpath=str(broken.parent))
    with pytest.raises(RuntimeError, match=r"cannot import nltk: ImportError.*CXXABI_1.3.15.*\n  libstdc\+\+: "):
        te.run_preflight(settings)


# The way lm-eval 0.4.13 and current checkouts set their version: from the installed metadata.
METADATA_VERSION_INIT = "from importlib.metadata import version\n__version__ = version('lm_eval')\n"


def test_preflight_refuses_a_checkout_that_pip_only_labelled(eval_env, tmp_path, monkeypatch):
    """The install command run over a source checkout instead of into a new folder.

    pip install --target leaves the checkout's lm_eval folder alone but adds the pinned release's
    metadata next to it. The checkout's code then reports the pinned version and comes from
    eval_harness_path, so only the file hashes show that it is not the release.
    """
    monkeypatch.setattr(te_preflight, "preflight_modules", lambda settings, cuda=False: ["lm_eval"])
    suite = te.load_suite(SMOKE_SUITE)
    release = fake_lm_eval(tmp_path / "release", te.LM_EVAL_VERSION, init=METADATA_VERSION_INIT,
                           files={"evaluator.py": ""})
    assert te.run_preflight(make_settings(suite, tmp_path, harness_path=release))["lm_eval_files_verified"] == 2

    checkout = tmp_path / "checkout"
    (checkout / "lm_eval").mkdir(parents=True)
    (checkout / "lm_eval" / "__init__.py").write_text(METADATA_VERSION_INIT + "LOCAL_CHANGE = True\n")
    (checkout / "lm_eval" / "evaluator.py").write_text("")
    metadata = f"lm_eval-{te.LM_EVAL_VERSION}.dist-info"
    shutil.copytree(release / metadata, checkout / metadata)
    # The checks of the folder names pass, on every rank and in the preflight.
    te.validate_eval_arguments(make_args(eval_env, eval_harness_path=checkout), raise_error)
    with pytest.raises(RuntimeError, match="1 files differ from the release, for example lm_eval/__init__.py") as refused:
        te.run_preflight(make_settings(suite, tmp_path, harness_path=checkout))
    assert "is version" not in str(refused.value)  # the checkout did report the pinned version


def test_preflight_refuses_lm_eval_modules_from_another_folder(tmp_path, monkeypatch):
    """Every lm_eval module that the preflight loads must come from the package in eval_harness_path.

    The stand-in's package path reaches into a second folder, the way the import hook of an
    editable install serves any module that the package lacks.
    """
    monkeypatch.setattr(te_preflight, "preflight_modules", lambda settings, cuda=False: ["lm_eval", "lm_eval.models"])
    suite = te.load_suite(SMOKE_SUITE)
    other = tmp_path / "checkout" / "lm_eval"
    (other / "models").mkdir(parents=True)
    (other / "models" / "__init__.py").write_text("")
    leaky = fake_lm_eval(
        tmp_path / "leaky", te.LM_EVAL_VERSION,
        init=f"__version__ = {te.LM_EVAL_VERSION!r}\n__path__.append({str(other)!r})\n",
    )
    expected = f"1 lm_eval modules come from outside .*lm_eval.models from {re.escape(str(other / 'models' / '__init__.py'))}"
    with pytest.raises(RuntimeError, match=expected):
        te.run_preflight(make_settings(suite, tmp_path, harness_path=leaky))


def test_prefetch_refuses_another_lm_eval(tmp_path, capsys):
    common = ["prefetch", "--suite", str(SMOKE_SUITE), "--eval-python", sys.executable,
              "--cache-dir", str(tmp_path / "cache"), "--tokenizer", str(tmp_path)]
    other = fake_lm_eval(tmp_path / "other", "0.4.9.1")
    assert te.main([*common, "--harness", str(other)]) == 1
    assert f"does not hold the metadata of lm-eval {te.LM_EVAL_VERSION}" in capsys.readouterr().out
    # Right metadata, but the package reports another version: the import check catches it.
    relabelled = fake_lm_eval(tmp_path / "relabelled", "0.4.9.1", dist_info=te.LM_EVAL_VERSION)
    assert te.main([*common, "--harness", str(relabelled)]) == 1
    assert "is version 0.4.9.1" in capsys.readouterr().out
    # A refused prefetch creates nothing in the cache folder.
    assert not (tmp_path / "cache").exists()


def test_the_fingerprint_covers_the_lm_eval_release(tmp_path, monkeypatch):
    """A retry reuses finished processes only under the same fingerprint, so a new pin redoes them."""
    settings = make_settings(te.load_suite(SMOKE_SUITE), tmp_path)
    plan = te.build_plan(settings.suite, 1)
    before = settings.fingerprint(plan)
    monkeypatch.setattr(te_settings, "LM_EVAL_VERSION", "0.4.14")
    assert settings.fingerprint(plan) != before


def test_process_timeouts_follow_the_plan():
    settings = make_settings(te.load_suite(BASE_SUITE), timeout_minutes=30.0)
    plan = te.build_plan(settings.suite, 64)
    startup = settings.suite.process_startup_seconds
    for procs in plan.processes:
        for process in procs:
            timeout = settings.process_timeout(process)
            assert process.cost + startup < timeout <= 30 * 60 * len(process.units)
            assert timeout <= 3 * (process.cost + startup) + 600
    # Every process of the 64-GPU plan is capped well below the 30 minutes per unit.
    assert max(settings.process_timeout(p) for procs in plan.processes for p in procs) < 35 * 60


def test_tokenizer_stop_ids(eval_env, tmp_path):
    eos, pad, bos = te.tokenizer_stop_ids(eval_env["tokenizer"] / "tokenizer_config.json")
    assert eos == [151643, 151645] and pad == 151643 and bos is None
    fixed2 = tmp_path / "fixed2.json"
    fixed2.write_text(json.dumps({
        "eos_token": "<|im_end|>", "pad_token": "<|im_end|>", "bos_token": "<|im_start|>",
        "added_tokens_decoder": {"0": {"content": "<|endoftext|>"}, "1": {"content": "<|im_start|>"},
                                 "2": {"content": "<|im_end|>"}},
    }))
    assert te.tokenizer_stop_ids(fixed2) == ([2, 0], 2, 1)


# ---------------------------------------------------------------------------
# Local task helpers and the wrapper
# ---------------------------------------------------------------------------

TASKS_DIR = REPO_ROOT / "eval" / "in_training" / "tasks"


def load_task_helper(name: str, monkeypatch, stubs: dict[str, dict]) -> types.ModuleType:
    """Loads eval/in_training/tasks/<name>.py with stand-ins for the lm-eval modules it imports.

    ``stubs`` maps a module name to the attributes of its stand-in. The parent packages get empty
    stand-ins, so no real lm-eval is needed or touched.
    """
    for module_name, attributes in stubs.items():
        parts = module_name.split(".")
        for depth in range(1, len(parts)):
            monkeypatch.setitem(sys.modules, ".".join(parts[:depth]), types.ModuleType(".".join(parts[:depth])))
        stub = types.ModuleType(module_name)
        for key, value in attributes.items():
            setattr(stub, key, value)
        monkeypatch.setitem(sys.modules, module_name, stub)
    spec = importlib.util.spec_from_file_location(f"test_{name}", TASKS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cruxeval_local(monkeypatch):
    return load_task_helper("cruxeval_local", monkeypatch, {
        "lm_eval.tasks.cruxeval.utils": {
            "pass_at_k": object(), "list_fewshot_samples_input": object(), "list_fewshot_samples_output": object(),
        },
    })


def run_program(program: str) -> bool:
    try:
        exec(compile(program, "<cruxeval>", "exec"), {})
    except Exception:
        return False
    return True


def test_cruxeval_output_predictions_keep_their_string_literals(cruxeval_local):
    doc = {"code": "def f(s):\n    return s + 'b=='", "input": "'a'", "output": "'ab=='"}
    # The generation as the model writes it: the tag, the assertion, and nothing after the stop string.
    program = cruxeval_local.build_predictions_output([[" \n[ANSWER]\nassert f('a') == 'ab=='\n"]], [doc])[0][0]
    assert program == f"{doc['code']}\nassert f('a') == 'ab=='"
    assert run_program(program)
    # A bare value, a value with a comment after it, and a wrong answer.
    bare, commented, wrong = cruxeval_local.build_predictions_output(
        [["'ab=='", "'ab==' # done", "'ab'"]], [doc]
    )[0]
    assert run_program(bare) and run_program(commented) and not run_program(wrong)


def test_cruxeval_input_predictions_keep_tuples_and_strings(cruxeval_local):
    doc = {"code": "def f(t, s):\n    return len(t) + len(s)", "input": "(1, 2), 'a#b'", "output": "5"}
    generations = ["[ANSWER]\nassert f((1, 2), 'a#b') == 5\n", "f((1, 2), 'a#b')", "assert 5 == f((1, 2), 'a#b')"]
    programs = cruxeval_local.build_predictions_input([generations], [doc])[0]
    assert programs[0] == f"{doc['code']}\nassert f((1, 2), 'a#b')==5"
    assert all(run_program(program) for program in programs)


def test_cruxeval_scorer_survives_degenerate_generations(cruxeval_local):
    doc = {"code": "def f(x):\n    return x", "input": "1", "output": "1"}
    for generation in ("(" * 300, "1\x00", "", "[ANSWER]"):
        for build in (cruxeval_local.build_predictions_output, cruxeval_local.build_predictions_input):
            program = build([[generation]], [doc])[0][0]
            assert program.startswith(doc["code"])


@pytest.fixture
def mbpp_local(monkeypatch):
    # mbpp_local imports nothing of lm-eval and loads evaluate only when it scores, so no stand-ins.
    return load_task_helper("mbpp_local", monkeypatch, {})


# The first problem of the MBPP test split (task_id 11), as the pinned revision of the dataset has it.
MBPP_DOC = {
    "task_id": 11,
    "text": "Write a python function to remove first and last occurrence of a given character from the string.",
    "code": (
        "def remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n"
        "            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n"
        "        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s "
    ),
    "test_list": [
        'assert remove_Occ("hello","l") == "heo"',
        'assert remove_Occ("abcda","a") == "bcd"',
        'assert remove_Occ("PHP","P") == "H"',
    ],
    "test_setup_code": "",
    "challenge_test_list": ['assert remove_Occ("hellolloll","l") == "helollol"', 'assert remove_Occ("","l") == ""'],
}
MBPP_ASSERTS = "\n".join(MBPP_DOC["test_list"])


def test_mbpp_prompt_of_one_problem(mbpp_local):
    assert mbpp_local.doc_to_text(MBPP_DOC) == (
        "You are an expert Python programmer, and here is your task: Write a python function to remove first and "
        "last occurrence of a given character from the string. Your code should pass these tests:\n\n"
        f"{MBPP_ASSERTS}\n[BEGIN]\n"
    )
    # A few-shot example ends with its solution and [DONE], which is also the stop string of the
    # generation. A test problem's target is the program that checks the generated solution.
    example = mbpp_local.list_fewshot_samples()[0]
    assert mbpp_local.doc_to_target(example) == example["code"] + "\n[DONE]"
    assert mbpp_local.doc_to_target(MBPP_DOC) == MBPP_ASSERTS


def test_mbpp_runs_the_setup_code_before_the_asserts(mbpp_local):
    """Task 367, the one test problem with setup code, builds the trees that its asserts check."""
    doc = dict(MBPP_DOC, test_setup_code="root = Node(1)", test_list=["assert depth(root) == 1"] * 3)
    target = mbpp_local.doc_to_target(doc)
    assert target == "root = Node(1)\n" + "\n".join(doc["test_list"])
    solution = "class Node:\n    def __init__(self, value):\n        self.value = value\ndef depth(node):\n    return 1"
    # code_eval runs the generated solution, a newline, and then the target.
    exec(compile(solution + "\n" + target, "<mbpp>", "exec"), {})


def test_mbpp_predictions_are_the_generations(mbpp_local):
    """Each response is run as it is; lm-eval has already cut it at [DONE]."""
    responses = [["def f():\r\n  return 1", "def f(): return 2"], ["x = 1\n"]]
    predictions = mbpp_local.build_predictions(responses, [MBPP_DOC, MBPP_DOC])
    assert predictions == responses and predictions[0] is not responses[0]


LM_EVAL_SITE = os.environ.get("ORBIT_TEST_LM_EVAL_SITE")


@pytest.mark.skipif(not LM_EVAL_SITE, reason="needs ORBIT_TEST_LM_EVAL_SITE, a folder with lm-eval 0.4.13 "
                                             "installed by pip --target")
def test_mbpp_prompt_and_examples_are_those_of_lm_eval_0_4_13(mbpp_local):
    """The copy in mbpp_local.py equals the stock mbpp task of the pinned lm-eval, read without importing it."""
    yaml = pytest.importorskip("yaml")
    jinja2 = pytest.importorskip("jinja2")
    site = Path(LM_EVAL_SITE)
    assert te.lm_eval_metadata_problem(site) is None, "not the pinned lm-eval release"
    stock_dir = site / "lm_eval" / "tasks" / "mbpp"

    def examples(path: Path) -> list:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "list_fewshot_samples")
        returned = next(node for node in ast.walk(function) if isinstance(node, ast.Return))
        return ast.literal_eval(returned.value)

    stock_examples = examples(stock_dir / "utils.py")
    assert [example["task_id"] for example in stock_examples] == [2, 3, 4]
    assert examples(TASKS_DIR / "mbpp_local.py") == stock_examples == mbpp_local.list_fewshot_samples()

    class TaskFileLoader(yaml.SafeLoader):
        """Reads a task file whose !function tags name helper functions, without importing them."""

    TaskFileLoader.add_constructor("!function", lambda loader, node: "!function " + loader.construct_scalar(node))
    stock = yaml.load((stock_dir / "mbpp.yaml").read_text(encoding="utf-8"), Loader=TaskFileLoader)
    ours = yaml.load((TASKS_DIR / "itr_mbpp.yaml").read_text(encoding="utf-8"), Loader=TaskFileLoader)
    # The stock templates, rendered the way lm-eval renders them (lm_eval/utils.py, apply_template).
    environment = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined,
                                     keep_trailing_newline=True)
    text = environment.from_string(stock["doc_to_text"])
    target = environment.from_string(stock["doc_to_target"])
    for doc in [*stock_examples, MBPP_DOC]:
        assert mbpp_local.doc_to_text(doc) == text.render(**doc)
        assert mbpp_local.doc_to_target(doc) == target.render(**doc)
    for key in ("dataset_path", "dataset_name", "test_split", "output_type", "unsafe_code", "target_delimiter",
                "num_fewshot"):
        assert ours[key] == stock[key], key
    assert ours["fewshot_config"]["sampler"] == stock["fewshot_config"]["sampler"] == "first_n"
    # The one generation setting that differs is the budget, which the stock task leaves at 256.
    assert "max_gen_toks" not in stock["generation_kwargs"]
    assert ours["generation_kwargs"] == {**stock["generation_kwargs"], "max_gen_toks": 512}


EVAL_PYTHON = os.environ.get("ORBIT_TEST_EVAL_PYTHON")
EVAL_CACHE = os.environ.get("ORBIT_TEST_EVAL_CACHE")


@pytest.mark.skipif(not (EVAL_PYTHON and EVAL_CACHE),
                    reason="needs ORBIT_TEST_EVAL_PYTHON and ORBIT_TEST_EVAL_CACHE, an evaluation environment "
                           "with evaluate and an offline cache that train_eval prefetch filled")
def test_mbpp_scores_programs_through_code_eval(tmp_path):
    """pass@1 from evaluate's real code_eval, in the offline environment of an evaluation process."""
    rank_cache = tmp_path / "rank"
    te.prepare_rank_cache(Path(EVAL_CACHE), rank_cache)
    env = te.build_eval_env(
        dict(os.environ), eval_python=Path(EVAL_PYTHON), gpu="", cache_dir=Path(EVAL_CACHE), harness_path=None,
        extra_pythonpath=str(TASKS_DIR), rank_cache=rank_cache,
    )
    # evaluate writes a file for every compute call under HF_HOME/metrics; the test keeps those out of
    # the shared cache.
    env["HF_METRICS_CACHE"] = str(tmp_path / "metrics")
    setup_doc = dict(MBPP_DOC, test_setup_code="root = Node(1)", test_list=["assert depth(root) == 1"] * 3)
    cases = {
        # The reference solution, with the \r\n line ends of the dataset.
        "correct": (MBPP_DOC, MBPP_DOC["code"]),
        "wrong": (MBPP_DOC, "def remove_Occ(s,ch):\n    return s"),
        "syntax_error": (MBPP_DOC, "def remove_Occ(s,ch)\n    return s"),
        "loops": (MBPP_DOC, "def remove_Occ(s,ch):\n    while True:\n        pass"),
        "needs_setup": (setup_doc, "class Node:\n    def __init__(self, v):\n        self.v = v\n"
                                   "def depth(node):\n    return 1"),
    }
    script = (
        "import json, sys, mbpp_local\n"
        "cases = json.loads(sys.argv[1])\n"
        "scores = {}\n"
        "for name, (doc, generation) in cases.items():\n"
        "    predictions = mbpp_local.build_predictions([[generation]], [doc])\n"
        "    scores[name] = mbpp_local.pass_at_k([mbpp_local.doc_to_target(doc)], predictions, k=[1])\n"
        "print(json.dumps(scores))\n"
    )
    result = subprocess.run([EVAL_PYTHON, "-c", script, json.dumps(cases)], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    scores = json.loads(result.stdout.strip().splitlines()[-1])
    assert scores == {
        "correct": {"pass@1": 1.0}, "wrong": {"pass@1": 0.0}, "syntax_error": {"pass@1": 0.0},
        "loops": {"pass@1": 0.0}, "needs_setup": {"pass@1": 1.0},
    }


STUB_LM_EVAL_MAIN = r'''
import json, os, sys

def cli_evaluate():
    # What the wrapper loaded of lm-eval before it handed over. Then whether it changed anything in
    # lm-eval's vLLM backend, where an older wrapper put its shims.
    loaded = sorted(name for name in sys.modules if name == "lm_eval" or name.startswith("lm_eval."))
    import lm_eval.models.vllm_causallms as backend

    with open(os.environ["STUB_OUT"], "w") as handle:
        json.dump({"argv": sys.argv[1:], "loaded": loaded, "changed": backend.changed_attributes()}, handle)
    if os.environ.get("STUB_RAISE"):
        raise ValueError("lm-eval failed")
    sys.exit(3)
'''

# A stand-in for the vLLM backend of lm-eval 0.4.13, with the names that a wrapper could patch and
# the 0.4.13 signature of modify_gen_kwargs. It remembers its own attributes once it has loaded,
# so the stand-in command line can tell whether anything replaced or added one.
STUB_VLLM_BACKEND = r'''
class LLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class VLLM:
    @staticmethod
    def modify_gen_kwargs(gen_kwargs, eos=None, default_max_gen_toks=256):
        return gen_kwargs, [], default_max_gen_toks


def get_tokenizer(*args, **kwargs):
    return None


def postprocess_generated_text(generation, stop, think_end_token):
    return generation


def changed_attributes():
    """The names in this module and in VLLM that no longer hold what this file defined."""
    now = {name: value for name, value in globals().items() if name not in ("_DEFINED", "_DEFINED_VLLM")}
    changed = sorted(name for name in now.keys() | _DEFINED.keys() if now.get(name) is not _DEFINED.get(name))
    methods = dict(vars(VLLM))
    changed += sorted(
        "VLLM." + name for name in methods.keys() | _DEFINED_VLLM.keys()
        if methods.get(name) is not _DEFINED_VLLM.get(name)
    )
    return changed


_DEFINED = dict(globals())
_DEFINED_VLLM = dict(vars(VLLM))
'''


@pytest.mark.parametrize("fail, rc", [(False, 3), (True, 1)])
def test_wrapper_runs_lm_eval_untouched_and_passes_the_exit_code_on(tmp_path, fail, rc):
    """run_lm_eval.py imports nothing of lm-eval besides its command line, and patches nothing.

    The stand-in package holds every lm-eval module that the wrapper or older shims touched,
    the vLLM backend included, so the result does not depend on the environment's own lm-eval.
    """
    package = tmp_path / "stub" / "lm_eval"
    (package / "models").mkdir(parents=True)
    (package / "__init__.py").write_text(f"__version__ = {te.LM_EVAL_VERSION!r}\n")
    (package / "__main__.py").write_text(STUB_LM_EVAL_MAIN)
    (package / "models" / "__init__.py").write_text("")
    (package / "models" / "vllm_causallms.py").write_text(STUB_VLLM_BACKEND)
    out = tmp_path / "out.json"
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "stub"), "STUB_OUT": str(out), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop(te.EVAL_PARENT_ENV, None)
    if fail:
        env["STUB_RAISE"] = "1"
    arguments = ["--model", "vllm", "--tasks", "itr_gsm8k,itr_humaneval", "--limit", "4"]
    entry = REPO_ROOT / "eval" / "in_training" / "run_lm_eval.py"
    result = subprocess.run([sys.executable, str(entry), *arguments], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == rc, result.stderr
    seen = json.loads(out.read_text())
    assert seen["argv"] == arguments
    assert seen["loaded"] == ["lm_eval", "lm_eval.__main__"]
    assert seen["changed"] == []


# A stand-in command line that records the libstdc++ files that the wrapper's process has loaded.
STUB_LM_EVAL_MAPS_MAIN = r'''
import json, os, sys

def cli_evaluate():
    with open("/proc/self/maps") as maps:
        fields = [line.split(None, 5) for line in maps]
    loaded = sorted({f[5].strip() for f in fields if len(f) == 6 and "/libstdc++.so" in f[5]})
    with open(os.environ["STUB_OUT"], "w") as handle:
        json.dump(loaded, handle)
    sys.exit(0)
'''


@pytest.mark.skipif(
    not Path("/proc/self/maps").exists() or not (Path(sys.prefix) / "lib" / "libstdc++.so.6").is_file(),
    reason="needs /proc and an interpreter whose environment ships libstdc++ (a conda environment)",
)
def test_the_wrapper_loads_the_environments_libstdcxx_unless_the_trainer_left_it_out(tmp_path):
    """run_lm_eval.py loads the environment's libstdc++ itself, as a second guard next to the preload.

    When the trainer decided against the preload, because another copy is newer, the wrapper must
    not load the older copy either.
    """
    package = tmp_path / "stub" / "lm_eval"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(STUB_LM_EVAL_MAPS_MAIN)
    out = tmp_path / "out.json"
    # flash_attn is hidden because the wrapper would otherwise import it, and with it torch, which
    # loads a libstdc++ of its own choosing.
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "stub"), "STUB_OUT": str(out),
           "PYTHONDONTWRITEBYTECODE": "1", "ORBIT_EVAL_HIDDEN_MODULES": "flash_attn"}
    for key in ("LD_PRELOAD", te.EVAL_PARENT_ENV, te.EVAL_NO_LIBSTDCXX_ENV):
        env.pop(key, None)
    entry = REPO_ROOT / "eval" / "in_training" / "run_lm_eval.py"

    def loaded(extra: dict[str, str]) -> list[str]:
        out.unlink(missing_ok=True)
        result = subprocess.run([sys.executable, str(entry)], env={**env, **extra},
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(out.read_text())

    own = str((Path(sys.prefix) / "lib" / "libstdc++.so.6").resolve())
    assert loaded({}) == [own]
    assert own not in loaded({te.EVAL_NO_LIBSTDCXX_ENV: "a newer copy elsewhere"})


# ---------------------------------------------------------------------------
# Export of a real (tiny) model
# ---------------------------------------------------------------------------


def tiny_ouro_config(vocab_size: int = 128, max_position_embeddings: int = 16384):
    from orbit.configuration_ouro import OuroConfig

    return OuroConfig(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=max_position_embeddings,
        rope_theta=1.0e6, total_ut_steps=4, early_exit_threshold=1.0,
    )


@pytest.mark.skipif(not os.environ.get("ORBIT_TEST_TOKENIZER_DIR"), reason="needs ORBIT_TEST_TOKENIZER_DIR")
def test_export_writes_a_loadable_hf_folder(tmp_path):
    from safetensors.torch import load_file

    from orbit.full_state import gather_full_state
    from orbit.modelling_ouro import OuroForCausalLM
    from orbit.train import OrbitLMTrainingModule

    tokenizer_dir = Path(os.environ["ORBIT_TEST_TOKENIZER_DIR"])
    tokenizer_config = tokenizer_dir / "tokenizer_config.json"
    vocab = json.loads(tokenizer_config.read_text()).get("vocab_size", 151936)
    config = tiny_ouro_config(vocab)
    config._attn_kernel_repo_path = "/torch29/only/build"
    model = OrbitLMTrainingModule(OuroForCausalLM(config), beta=0.1, exit_mode="learned",
                                 pad_token_id=None, lm_chunk_size=0)
    state = gather_full_state(model, True)
    out = tmp_path / "model"
    info = te.write_eval_export(
        state, model_config=te_evaluator._live_model_config(model), out_dir=out, tokenizer_source=tokenizer_dir,
        tokenizer_config=tokenizer_config, max_position_embeddings=32768 + te.RULER_PROMPT_MARGIN,
        metadata={"step": 7, "export_key": "key"},
    )
    saved = json.loads((out / "config.json").read_text())
    assert saved["max_position_embeddings"] == 32768 + te.RULER_PROMPT_MARGIN
    assert saved["rope_theta"] == 1.0e6
    assert saved["early_exit_threshold"] is None
    assert saved.get("_attn_kernel_repo_path") is None
    assert saved["architectures"] == ["OuroForCausalLM"]
    assert model.model.config.max_position_embeddings == 16384  # the live model is untouched
    generation = json.loads((out / "generation_config.json").read_text())
    assert generation["eos_token_id"] == info["eos_token_ids"]
    assert (out / "tokenizer.json").is_file() and (out / "modelling_ouro.py").is_file()
    tensors = load_file(str(out / "model.safetensors"))
    reference = OuroForCausalLM(config).state_dict()
    assert set(tensors) == set(reference)
    assert all(t.dtype == torch.bfloat16 for t in tensors.values() if t.is_floating_point())
    assert not out.with_name("model.partial").exists()
    marker = json.loads((out / "eval_export.json").read_text())
    assert marker["step"] == 7 and marker["export_key"] == "key"


def test_a_failed_export_leaves_no_folders(tmp_path, eval_env, monkeypatch):
    from orbit import export as merge

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(merge, "save_sharded_state_dict", full_disk)
    out = tmp_path / "model"
    out.mkdir()  # the export of an earlier step goes away too
    with pytest.raises(Exception):
        te.write_eval_export(
            {"model.embed_tokens.weight": torch.zeros(4, 4)}, model_config=tiny_ouro_config(), out_dir=out,
            tokenizer_source=eval_env["tokenizer"], tokenizer_config=eval_env["tokenizer"] / "tokenizer_config.json",
            max_position_embeddings=4096,
        )
    assert not out.exists() and not (tmp_path / "model.partial").exists()


# ---------------------------------------------------------------------------
# Triggers, attempts and restarts
# ---------------------------------------------------------------------------


def make_evaluator(tmp_path, *, every=10, at_start=False, suite=SMOKE_SUITE, **overrides) -> te.InTrainingEval:
    settings = make_settings(te.load_suite(suite), tmp_path, every=every, at_start=at_start, **overrides)
    evaluator = te.InTrainingEval(settings, rank=0, local_rank=0, world_size=1, distributed=False,
                                  device=None, run_dir=tmp_path / "run")
    evaluator.WRITE_RETRY_SECONDS = 0.0
    return evaluator


def write_fake_export(evaluator: te.InTrainingEval, step: int) -> None:
    evaluator.export_dir.mkdir(parents=True, exist_ok=True)
    te.write_json_atomic(evaluator.export_dir / "eval_export.json", {"step": step, "export_key": evaluator.export_key})


def test_due_steps_and_the_catch_up_at_start(tmp_path):
    evaluator = make_evaluator(tmp_path, every=10)
    assert [s for s in range(0, 31) if evaluator.due(s)] == [10, 20, 30]
    # A run resumed at an evaluated step catches up; a fresh run at step 0 does not.
    assert evaluator.due(20, at_start=True) and not evaluator.due(0, at_start=True)
    assert not evaluator.due(25, at_start=True)
    assert make_evaluator(tmp_path, every=10, at_start=True).due(0, at_start=True)


def test_the_last_step_is_evaluated_only_with_eval_at_end(tmp_path):
    # 4769 is the last step of the 32K stage, and eval_every 1000 does not divide it.
    plain = make_evaluator(tmp_path, every=1000)
    assert not plain.due(4769, last=True)
    at_end = make_evaluator(tmp_path, every=1000, at_end=True)
    assert at_end.due(4769, last=True) and not at_end.due(4769)
    assert [s for s in range(0, 4770) if at_end.due(s, last=(s == 4769))] == [1000, 2000, 3000, 4000, 4769]
    # A run resumed at its last step catches up on that evaluation before it finishes.
    assert at_end.due(4769, at_start=True, last=True)
    assert not plain.due(4769, at_start=True, last=True)


def test_eval_at_end_alone_turns_the_evaluation_on():
    assert te.eval_requested(argparse.Namespace(eval_every=0, eval_at_start=False, eval_at_end=True))
    assert not te.eval_requested(argparse.Namespace(eval_every=0, eval_at_start=False))


def test_the_export_path_is_the_same_for_every_step(tmp_path):
    """vLLM keys its compile cache by the model path, so one path keeps the cache valid."""
    evaluator = make_evaluator(tmp_path)
    assert evaluator.export_dir == tmp_path / "run" / "eval" / "model"
    assert evaluator.rank_cache == tmp_path / "run" / "eval" / "cache" / "rank_00000"


def test_a_retry_reuses_the_export_and_the_finished_processes(tmp_path):
    evaluator = make_evaluator(tmp_path)
    step_dir = evaluator.step_dir(10)
    assert evaluator._prepare_step_dir(10, step_dir) == (0, 0, 1)  # first attempt: nothing to reuse
    write_fake_export(evaluator, 10)
    done = step_dir / "procs" / "r00000_p00" / "done.json"
    te.write_json_atomic(done, {"status": "ok"})
    # The job died during attempt 1. Attempt 2 keeps the export and the finished process.
    assert evaluator._prepare_step_dir(10, step_dir) == (0, 1, 2)
    assert done.exists()


def test_an_export_of_another_step_or_other_weights_is_not_reused(tmp_path):
    evaluator = make_evaluator(tmp_path)
    write_fake_export(evaluator, 10)
    assert evaluator._prepare_step_dir(20, evaluator.step_dir(20)) == (0, 0, 1)
    other = make_evaluator(tmp_path, max_model_len=8192)
    assert other._prepare_step_dir(10, other.step_dir(10)) == (0, 0, 1)


def test_a_different_plan_discards_the_finished_processes(tmp_path):
    evaluator = make_evaluator(tmp_path)
    step_dir = evaluator.step_dir(10)
    evaluator._prepare_step_dir(10, step_dir)
    done = step_dir / "procs" / "r00000_p00" / "done.json"
    te.write_json_atomic(done, {"status": "ok"})
    other = make_evaluator(tmp_path, suite=BASE_SUITE)
    assert other._prepare_step_dir(10, step_dir)[0] == 0
    assert not done.exists()


def test_a_finished_step_is_skipped_and_unreported_scores_are_reported_once(tmp_path):
    evaluator = make_evaluator(tmp_path)
    logger = FakeClearMLLogger()
    evaluator.attach(model=None, clearml_logger=logger, tb_writer=None, stop_requested=lambda: False)
    step_dir = evaluator.step_dir(10)
    te.write_json_atomic(step_dir / "summary.json", {
        "step": 10, "status": "ok", "reported": False,
        "scores": {"benchmarks": {"mmlu": 0.5}, "ruler": {"4096": 0.9, "mean": 0.9}},
        "timings": {"wall_min": 7.0}, "failures": {"units_failed": 0},
    })
    # Rank 0 died after writing the summary and before reporting: the next attempt reports.
    assert evaluator._prepare_step_dir(10, step_dir) == (1, 0, 0)
    reported = {(t, s): (v, it) for t, s, v, it in logger.scalars}
    assert reported[("eval_mmlu", "raw")] == (0.5, 10)
    assert reported[("eval_ruler", "raw_4096")] == (0.9, 10)
    assert json.loads((step_dir / "summary.json").read_text())["reported"] is True
    logger.scalars.clear()
    assert evaluator._prepare_step_dir(10, step_dir) == (1, 0, 0)
    assert logger.scalars == []


def test_a_step_cut_off_by_crashes_twice_is_given_up(tmp_path):
    evaluator = make_evaluator(tmp_path)
    logger = FakeClearMLLogger()
    evaluator.attach(model=None, clearml_logger=logger, tb_writer=None, stop_requested=lambda: False)
    step_dir = evaluator.step_dir(10)
    write_fake_export(evaluator, 10)
    assert evaluator._prepare_step_dir(10, step_dir) == (0, 1, 1)
    assert evaluator._prepare_step_dir(10, step_dir) == (0, 1, 2)
    assert evaluator._prepare_step_dir(10, step_dir) == (1, 0, 0)
    summary = json.loads((step_dir / "summary.json").read_text())
    assert summary["status"] == "abandoned" and summary["reported"] is True
    assert ("eval_failures", "abandoned", 1.0, 10) in logger.scalars
    assert not evaluator.export_dir.exists()


def test_stopped_attempts_do_not_count_as_crashes(tmp_path):
    evaluator = make_evaluator(tmp_path)
    step_dir = evaluator.step_dir(10)
    for attempt in (1, 2, 3):
        assert evaluator._prepare_step_dir(10, step_dir) == (0, 0, attempt)
        evaluator._set_attempt_outcome(step_dir, attempt, "stopped")


# ---------------------------------------------------------------------------
# Whole evaluations with a stand-in for lm-eval
# ---------------------------------------------------------------------------

FAKE_LM_EVAL = r'''
import argparse, json, os, sys, time
time.sleep(float(os.environ.get("FAKE_LM_EVAL_SLEEP", "0")))
parser = argparse.ArgumentParser()
parser.add_argument("--tasks"); parser.add_argument("--output_path"); parser.add_argument("--samples")
parser.add_argument("--limit"); parser.add_argument("--metadata")
args, _ = parser.parse_known_args()
assert "RANK" not in os.environ and "MASTER_PORT" not in os.environ
samples = json.load(open(args.samples)) if args.samples else {}
lengths = json.loads(args.metadata)["max_seq_lengths"] if args.metadata else []
results, counts = {}, {}
for task in args.tasks.split(","):
    row = {"acc,none": 0.5, "acc_norm,none": 0.25, "exact_match,strict-match": 0.5,
           "exact_match,custom-extract": 0.5, "math_verify,none": 0.5, "pass@1,create_test": 0.5}
    for length in lengths:
        row[f"{length},none"] = 0.75
    results[task] = row
    # With --samples, "effective" is the size of the whole test set, as lm-eval releases before
    # 0.4.13 reported it. The scores must come out right even then.
    n = 10042 if task in samples else int(args.limit or 10)
    counts[task] = {"original": n, "effective": n}
out = os.path.join(args.output_path, "model")
os.makedirs(out, exist_ok=True)
json.dump({"results": results, "n-samples": counts}, open(os.path.join(out, "results_x.json"), "w"))
'''

SMALL_SUITE = "process_startup_seconds: 1\nbenchmarks:\n  mmlu: {limit: 2}\n  ruler: {lengths: [4096], num_samples: 2}\n"


def fake_export_recorder(exports: list):
    def fake_export(state, *, out_dir, metadata=None, **kwargs):
        exports.append(len(state))
        Path(out_dir).mkdir(parents=True)
        te.write_json_atomic(Path(out_dir) / "eval_export.json", dict(metadata or {}))
        return {"tensors": len(state)}

    return fake_export


def single_rank(tmp_path, monkeypatch, *, body=SMALL_SUITE, stop=lambda: False, **overrides):
    entry = tmp_path / "fake_lm_eval.py"
    entry.write_text(FAKE_LM_EVAL)
    suite = te.load_suite(write_suite(tmp_path, body, entry=entry))
    (tmp_path / "cache").mkdir(exist_ok=True)
    exports: list = []
    monkeypatch.setattr(te_evaluator, "write_eval_export", fake_export_recorder(exports))
    monkeypatch.setattr(te_evaluator, "_live_model_config", lambda model: object())
    settings = make_settings(suite, tmp_path, **overrides)
    evaluator = te.InTrainingEval(settings, rank=0, local_rank=0, world_size=1, distributed=False, device=None,
                                  run_dir=tmp_path / "run", poll_seconds=0.05)
    evaluator.WRITE_RETRY_SECONDS = 0.0
    logger = FakeClearMLLogger()
    evaluator.attach(model=torch.nn.Linear(4, 4), clearml_logger=logger, tb_writer=None,
                     stop_requested=stop)
    return evaluator, logger, exports


def test_a_stop_during_the_evaluation_is_redone_by_the_resumed_run(tmp_path, monkeypatch):
    stop = threading.Event()
    evaluator, logger, exports = single_rank(tmp_path, monkeypatch, stop=stop.is_set)
    monkeypatch.setenv("FAKE_LM_EVAL_SLEEP", "30")
    threading.Timer(1.0, stop.set).start()
    started = time.monotonic()
    evaluator.maybe_run(10)
    assert time.monotonic() - started < 20  # the sleeping process was killed at the stop
    step_dir = evaluator.step_dir(10)
    # Nothing marks the step as done, and no partial point lands on the curves.
    assert not (step_dir / "summary.json").exists()
    assert logger.scalars == []
    attempts = json.loads((step_dir / "attempts.json").read_text())["attempts"]
    assert [entry["outcome"] for entry in attempts] == ["stopped"]
    assert json.loads((step_dir / "attempt_001" / "release.json").read_text())["status"] == "stopped"
    assert evaluator.export_dir.is_dir()  # kept for the resumed run

    # The resumed run starts at step 10, catches up, and reuses the export.
    stop.clear()
    monkeypatch.setenv("FAKE_LM_EVAL_SLEEP", "0")
    evaluator.maybe_run(10, at_start=True)
    summary = json.loads((step_dir / "summary.json").read_text())
    assert summary["status"] == "ok" and summary["reported"] is True and summary["attempt"] == 2
    assert exports == [2]  # the Linear layer's two tensors, exported once
    assert ("eval_mmlu", "raw", 0.5, 10) in logger.scalars
    assert not evaluator.export_dir.exists()


def test_a_failed_export_and_marker_never_stop_training(tmp_path, monkeypatch):
    evaluator, logger, _ = single_rank(tmp_path, monkeypatch, total_timeout_minutes=0.02)
    evaluator.GRACE_SECONDS = 0.5
    evaluator.RELEASE_SECONDS = 0.5

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    original = te.write_json_atomic

    def write(path, payload):
        if Path(path).name == "export.json":
            full_disk()
        return original(path, payload)

    monkeypatch.setattr(te_evaluator, "write_eval_export", full_disk)
    monkeypatch.setattr(te_evaluator, "write_json_atomic", write)
    evaluator.maybe_run(10)  # must not raise
    summary = json.loads((evaluator.step_dir(10) / "summary.json").read_text())
    assert summary["status"] == "partial"
    assert {record["status"] for record in summary["processes"].values()} == {"export_failed"}
    assert not evaluator.export_dir.exists()
    assert not evaluator.export_dir.with_name("model.partial").exists()
    assert ("eval_failures", "processes_failed", 2.0, 10) in logger.scalars


# ---------------------------------------------------------------------------
# Evaluation processes that outlive their trainer
# ---------------------------------------------------------------------------

needs_proc = pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs /proc")


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"  # a zombie has exited


@needs_proc
def test_leftover_evaluation_processes_are_found_by_their_gpu_tag():
    tag = f"GPU-test-{os.getpid()}"
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    mine = subprocess.Popen(sleeper, env={**os.environ, te.EVAL_GPU_ENV: tag})
    other = subprocess.Popen(sleeper, env={**os.environ, te.EVAL_GPU_ENV: tag + "-other-gpu"})
    try:
        deadline = time.monotonic() + 10
        while te.find_eval_processes(tag) != [mine.pid] and time.monotonic() < deadline:
            time.sleep(0.05)
        assert te.find_eval_processes(tag) == [mine.pid]
        assert te.kill_eval_processes(tag) == [mine.pid]
        assert mine.wait(timeout=10) == -signal.SIGKILL
        assert other.poll() is None  # another GPU's process is left alone
        assert te.find_eval_processes("") == []
    finally:
        for process in (mine, other):
            process.kill()
            process.wait()


LEADER_WITH_ENGINE = r'''
import os, subprocess, sys, time
tag_key, out = sys.argv[1], sys.argv[2]
# The child stands in for vLLM's EngineCore: no visible tag, and a telling command line.
env = {key: value for key, value in os.environ.items() if key != tag_key}
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "run_lm_eval-engine"], env=env)
open(out + ".tmp", "w").write(str(child.pid))
os.replace(out + ".tmp", out)
time.sleep(60)
'''


@needs_proc
def test_an_untagged_engine_is_found_through_its_session(tmp_path):
    """vLLM's EngineCore overwrites the environment that /proc shows, so only its session finds it."""
    tag = f"GPU-session-{os.getpid()}"
    out = tmp_path / "child"
    leader = subprocess.Popen([sys.executable, "-c", LEADER_WITH_ENGINE, te.EVAL_GPU_ENV, str(out)],
                              env={**os.environ, te.EVAL_GPU_ENV: tag}, start_new_session=True)
    plain = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    child = None
    try:
        deadline = time.monotonic() + 10
        while not out.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        child = int(out.read_text())
        # While the tagged leader lives, its session leads to the untagged child.
        assert te.find_eval_sessions(tag) == sorted([leader.pid, child])
        # The leader dies, as when the wrapper is killed with the trainer. Now only the session
        # recorded by the trainer still finds the child.
        os.kill(leader.pid, signal.SIGKILL)
        leader.wait()
        assert te.find_eval_processes(tag) == [] and te.find_eval_sessions(tag) == []
        assert te.find_eval_sessions(tag, sessions=[leader.pid]) == [child]
        # A recorded session id that now belongs to an unrelated session is left alone.
        assert te.find_eval_sessions(tag, sessions=[plain.pid]) == []
        assert te.kill_eval_processes(tag, sessions=[leader.pid, plain.pid]) == [child]
        deadline = time.monotonic() + 10
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(child) and plain.poll() is None
    finally:
        for process in (leader, plain):
            process.kill()
            process.wait()
        if child is not None and _alive(child):
            os.kill(child, signal.SIGKILL)


def test_ranks_record_their_evaluation_sessions_per_host(tmp_path):
    evaluator = make_evaluator(tmp_path)
    evaluator._record_session(111)
    evaluator._record_session(222)
    assert evaluator._recorded_sessions() == [111, 222]
    record = evaluator.rank_cache / "eval_sessions.json"
    te.write_json_atomic(record, {**json.loads(record.read_text()), "host": "another-node"})
    assert evaluator._recorded_sessions() == []  # PIDs of another machine mean nothing here


WATCHDOG_TRAINER = r'''
import os, subprocess, sys, time
wrapper_dir, out = sys.argv[1], sys.argv[2]
code = (
    "import os, subprocess, sys, time\n"
    f"sys.path.insert(0, {wrapper_dir!r})\n"
    "import run_lm_eval\n"
    "run_lm_eval.start_parent_watchdog(0.1)\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
    f"open({out!r} + '.tmp', 'w').write(f'{{os.getpid()}} {{child.pid}}')\n"
    f"os.replace({out!r} + '.tmp', {out!r})\n"
    "time.sleep(120)\n"
)
env = dict(os.environ, ORBIT_EVAL_PARENT_PID=str(os.getpid()))
subprocess.Popen([sys.executable, "-c", code], env=env, start_new_session=True)
time.sleep(120)
'''


@needs_proc
def test_evaluation_processes_die_with_a_trainer_killed_by_sigkill(tmp_path):
    """The launcher's kill never reaches the evaluation's own session; run_lm_eval's watchdog does it."""
    out = tmp_path / "pids"
    trainer = subprocess.Popen([sys.executable, "-c", WATCHDOG_TRAINER, str(REPO_ROOT / "eval" / "in_training"), str(out)])
    try:
        deadline = time.monotonic() + 30
        while not out.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        wrapper, grandchild = (int(x) for x in out.read_text().split())
        assert _alive(wrapper) and _alive(grandchild)
        trainer.kill()
        trainer.wait()
        deadline = time.monotonic() + 10
        while (_alive(wrapper) or _alive(grandchild)) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(wrapper) and not _alive(grandchild)
    finally:
        trainer.kill()
        trainer.wait()


# ---------------------------------------------------------------------------
# The whole evaluation on two CPU ranks
# ---------------------------------------------------------------------------


def _distributed_worker(rank: int, world_size: int, root: str, init_file: str, mode: str) -> None:
    import torch.distributed as dist

    root_path = Path(root)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=60))
    exports: list = []
    te_evaluator.write_eval_export = fake_export_recorder(exports)
    te_evaluator._live_model_config = lambda model: object()
    if mode == "no_release" and rank == 0:
        original = te.write_json_atomic

        def full_disk(path, payload):
            if Path(path).name in ("summary.json", "release.json"):
                raise OSError(28, "No space left on device")
            return original(path, payload)

        te_evaluator.write_json_atomic = full_disk
    suite = te.load_suite(root_path / "suite.yaml")
    settings = make_settings(suite, root_path, total_timeout_minutes=2.0 if mode == "normal" else 0.05)
    evaluator = te.InTrainingEval(settings, rank=rank, local_rank=rank, world_size=world_size,
                                  distributed=True, device=None, run_dir=root_path / "run", poll_seconds=0.1)
    evaluator.WRITE_RETRY_SECONDS = 0.0
    if mode == "no_release":
        evaluator.GRACE_SECONDS = 1.0
        evaluator.RELEASE_SECONDS = 2.0
    logger = FakeClearMLLogger() if rank == 0 else None
    evaluator.attach(model=torch.nn.Linear(4, 4), clearml_logger=logger, tb_writer=None,
                     stop_requested=lambda: False)
    left = []
    for step in ((9, 10, 10) if mode == "normal" else (10,)):  # not due, due, already done
        evaluator.maybe_run(step)
        left.append(time.time())
    (root_path / f"rank{rank}.json").write_text(json.dumps({
        "scalars": logger.scalars if logger else [], "exports": exports, "left": left,
    }))
    dist.barrier()
    dist.destroy_process_group()


def run_two_ranks(tmp_path: Path, mode: str, body: str) -> list[dict]:
    import torch.multiprocessing as mp

    entry = tmp_path / "fake_lm_eval.py"
    entry.write_text(FAKE_LM_EVAL)
    write_suite(tmp_path, body, entry=entry)
    (tmp_path / "cache").mkdir(exist_ok=True)
    mp.spawn(_distributed_worker, args=(2, str(tmp_path), str(tmp_path / "init"), mode), nprocs=2, join=True)
    return [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]


def test_two_ranks_evaluate_report_and_skip_finished_steps(tmp_path):
    ranks = run_two_ranks(tmp_path, "normal", "process_startup_seconds: 1\nbenchmarks:\n  mmlu: {limit: 2}\n"
                          "  hellaswag: {shards: 2}\n  ruler: {lengths: [4096], num_samples: 2}\n")
    step_dir = tmp_path / "run" / "eval" / "step_00000010"
    summary = json.loads((step_dir / "summary.json").read_text())
    assert summary["status"] == "ok", summary
    assert summary["reported"] is True and summary["attempt"] == 1
    assert summary["scores"]["benchmarks"]["mmlu"] == pytest.approx(0.5)
    assert summary["scores"]["benchmarks"]["hellaswag"] == pytest.approx(0.25)
    # The shards' own documents count, not the 10042 that the stand-in reports for each of them.
    assert summary["scores"]["samples"]["hellaswag"] == 10042
    assert summary["scores"]["ruler"] == {"4096": 0.75, "mean": 0.75}
    assert summary["failures"]["units_failed"] == 0
    assert not (tmp_path / "run" / "eval" / "model").exists()  # the export is removed afterwards
    assert not (tmp_path / "run" / "eval" / "step_00000009").exists()
    attempt_dir = step_dir / "attempt_001"
    assert sorted(p.name for p in (attempt_dir / "ranks").iterdir()) == ["rank_00000.json", "rank_00001.json"]
    assert sorted(p.name for p in (attempt_dir / "heartbeat").iterdir()) == ["rank_00000", "rank_00001"]
    assert json.loads((attempt_dir / "release.json").read_text())["status"] == "ok"
    assert ranks[0]["exports"] == [2]  # one export: the third call found the summary and skipped
    reported = [(t, s, v, it) for t, s, v, it in ranks[0]["scalars"]]
    assert [(v, it) for t, s, v, it in reported if (t, s) == ("eval_mmlu", "raw")] == [(0.5, 10)]
    assert any((t, s) == ("eval_ruler", "raw_4096") for t, s, _, _ in reported)
    # Both ranks did work: the plan spreads the units over the two ranks.
    procs = sorted(p.name for p in (step_dir / "procs").iterdir())
    assert any(name.startswith("r00000") for name in procs) and any(name.startswith("r00001") for name in procs)


def test_ranks_resume_together_when_rank_0_cannot_write_the_summary(tmp_path):
    """With a full disk, rank 0 writes neither the summary nor the release file.

    The other rank waits for the release file until the shared release deadline, and rank 0
    waits for the same deadline, so both reach the final collective together. Before the fix,
    rank 0 went straight into the barrier and waited there far longer than the NCCL timeout.
    """
    ranks = run_two_ranks(tmp_path, "no_release", SMALL_SUITE)
    step_dir = tmp_path / "run" / "eval" / "step_00000010"
    assert not (step_dir / "summary.json").exists()
    assert not (step_dir / "attempt_001" / "release.json").exists()
    left = [rank["left"][-1] for rank in ranks]
    assert abs(left[0] - left[1]) < 2.0
