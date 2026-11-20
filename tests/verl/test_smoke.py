"""CPU tests of scripts/verl/smoke.sh, the short GPU check of both recipes.

Each test asks the script for its launcher arguments (``--print`` checks nothing and runs nothing),
composes the recipe with them the way the launcher does, and runs the launcher's checks that need no
GPU. The export and the teacher are fake dirs, as in test_launch.py and test_distill_launch.py, so
the tests need no model files. Run with the verl python and ``PYTHONPATH=<repo>/src``.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

from orbit_verl.distill import preflight
from orbit_verl.launch import config as lconfig
from orbit_verl.launch import env as lenv
from orbit_verl.launch import prepare as lprepare
from test_distill_launch import _teacher_dir
from test_launch import REPO, _compose, _export

pytestmark = pytest.mark.skipif(importlib.util.find_spec("verl") is None,
                                reason="composing a config needs verl's own config files")

SCRIPT = REPO / "scripts" / "verl" / "smoke.sh"
DATA = "/data/smoke"
# The overrides that name the run's inputs and outputs. A recipe composed with only these is the
# reference that the smoke's other overrides are compared with.
INPUT_KEYS = ("orbit.run_name", "orbit.prepare.data.name", "orbit.prepare.data.gold", "orbit.prepare.data.val",
              "orbit.prepare.data.pool", "distillation.teacher_models.teacher_model.model_path")
# What each smoke changes in its recipe, besides the inputs.
SHARED_CHANGES = {"data.train_batch_size", "data.max_response_length", "trainer.total_training_steps",
                  "trainer.save_freq", "trainer.test_freq", "actor_rollout_ref.rollout.agent.num_workers",
                  "reward.num_workers", "actor_rollout_ref.rollout.gpu_memory_utilization",
                  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
                  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
                  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu"}
CHANGES = {
    "rl": SHARED_CHANGES | {"actor_rollout_ref.rollout.n"},
    "distill": SHARED_CHANGES | {"orbit.prepare.data.limit", "orbit.distill.teacher_gpus_per_node",
                                 "orbit.env.ORBIT_THINK_BUDGET",
                                 "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens"},
}


def _run(*argv: str) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if key not in ("VERL_PYTHON", "ORBIT_VERL_RUNS")}
    return subprocess.run(["bash", str(SCRIPT), *argv], capture_output=True, text=True, env=env)


def _print(kind: str, tmp_path: Path, *extra: str) -> tuple[list[str], list[str], Path]:
    """The launcher options and the overrides of a smoke, and the fake export they name.

    ``extra`` goes to the script after "--", as hydra overrides of the user's own.
    """
    export = _export(tmp_path)
    argv = [kind, "--export", str(export), "--data", DATA, "--runs-root", str(tmp_path / "runs_root")]
    if kind == "distill":
        argv += ["--teacher", str(_teacher_dir(tmp_path))]
    proc = _run(*argv, "--print", *(("--", *extra) if extra else ()))
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    cut = lines.index("--")
    return lines[:cut], lines[cut + 1:], export


def _option(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def _compose_smoke(kind: str, tmp_path: Path, *, only_inputs: bool = False):
    """The recipe composed with the smoke's overrides (or only its input overrides), on the smoke's GPUs."""
    args, overrides, export = _print(kind, tmp_path)
    if only_inputs:
        overrides = [item for item in overrides if item.split("=", 1)[0] in INPUT_KEYS]
    gpus = int(_option(args, "--gpus-per-node"))
    cfg = _compose(_option(args, "--config"), tmp_path, *overrides, export=export,
                   model_name=_option(args, "--model-name"), runs=Path(_option(args, "--runs-root")),
                   mode=lconfig.mode_overrides(nnodes=1, gpus_per_node=gpus))
    return cfg, args, overrides, gpus


def _leaves(node, prefix: str = "") -> dict[str, object]:
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out.update(_leaves(value, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: node}


def _unresolved(cfg) -> dict[str, object]:
    from omegaconf import OmegaConf

    return _leaves(OmegaConf.to_container(cfg, resolve=False))


def _check_shared(cfg, kind: str) -> None:
    """What both smokes share: 2 steps of 4 prompts, 1,024-token responses, 4 validation rows per set."""
    assert (cfg.trainer.total_training_steps, cfg.trainer.save_freq, cfg.trainer.test_freq) == (2, 2, 2)
    assert cfg.trainer.val_before_train is True
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 4
    assert cfg.data.max_response_length == cfg.actor_rollout_ref.rollout.response_length == 1024
    assert cfg.orbit.run_name == f"smoke_{kind}" and cfg.orbit.prepare.data.name == f"smoke_{kind}__qwen3"
    assert list(cfg.orbit.prepare.data.gold) == [f"{DATA}/math_keep.jsonl:orbit/math:math:16",
                                                  f"{DATA}/code_keep.jsonl:orbit/code:code:16"]
    assert all(spec.startswith(f"{DATA}/") and spec.endswith(":4") for spec in cfg.orbit.prepare.data.val)
    assert lprepare._data_config_problem(cfg) is None
    assert lprepare.check_lengths(cfg) == []
    assert lprepare.check_settings(cfg, environ={}) == []
    assert preflight.check_think_budget(cfg) == []


def test_the_rl_smoke_is_the_rl_recipe_on_one_gpu(tmp_path):
    from orbit_verl.launch.train import validate

    cfg, args, _, gpus = _compose_smoke("rl", tmp_path)
    assert (_option(args, "--config"), gpus) == ("rl_sft_mix", 1)
    _check_shared(cfg, "rl")
    assert cfg.actor_rollout_ref.rollout.n == 4
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "rltt_exit"
    assert len(cfg.orbit.prepare.data.val) == 2
    assert lconfig.finish_distillation(cfg, nnodes=1, gpus_per_node=gpus) is None
    job = lenv.job_env(cfg, str(cfg.orbit.repo_root), base_env={})
    assert "ORBIT_THINK_BUDGET" not in job
    validate(cfg)


def test_the_distill_smoke_gives_the_student_and_the_teacher_one_gpu_each(tmp_path):
    from orbit_verl.launch.train import validate

    cfg, args, overrides, gpus = _compose_smoke("distill", tmp_path)
    assert (_option(args, "--config"), gpus) == ("distill_rl_sft_16k", 2)
    _check_shared(cfg, "distill")
    assert cfg.distillation.enabled and cfg.distillation.distillation_loss.use_task_rewards is True
    assert cfg.orbit.prepare.data.pool == f"{DATA}/prompts.jsonl" and cfg.orbit.prepare.data.limit == 32
    assert len(cfg.orbit.prepare.data.val) == 4
    setup = lconfig.finish_distillation(cfg, nnodes=1, gpus_per_node=gpus, user_overrides=overrides)
    assert setup.topology == ("shared", 1, 1, 1, 1)
    assert setup.max_model_len == {"teacher_model": 1088 + 1024 + 1}
    # The thinking budget leaves half of the response, so the forced close runs in the smoke too.
    assert cfg.orbit.env.ORBIT_THINK_BUDGET == "512"
    job = lenv.job_env(cfg, str(cfg.orbit.repo_root), base_env={})
    assert job["ORBIT_THINK_BUDGET"] == "512"
    assert preflight.check_think_budget(cfg) == []
    assert preflight.run_preflight(cfg) == ([], [])
    validate(cfg)


@pytest.mark.parametrize("kind", ["rl", "distill"])
def test_the_smoke_changes_only_the_scale_of_its_recipe(kind, tmp_path):
    smoke, _, _, _ = _compose_smoke(kind, tmp_path)
    recipe, _, _, _ = _compose_smoke(kind, tmp_path, only_inputs=True)
    a, b = _unresolved(smoke), _unresolved(recipe)
    assert {key for key in set(a) | set(b) if a.get(key) != b.get(key)} == CHANGES[kind]


def test_overrides_after_the_double_dash_come_last_and_win(tmp_path):
    longer = ("data.max_response_length=4096", "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192")
    args, overrides, export = _print("rl", tmp_path, *longer)
    assert overrides[-2:] == list(longer)
    cfg = _compose(_option(args, "--config"), tmp_path, *overrides, export=export,
                   model_name=_option(args, "--model-name"), runs=Path(_option(args, "--runs-root")),
                   mode=lconfig.mode_overrides(nnodes=1, gpus_per_node=int(_option(args, "--gpus-per-node"))))
    assert cfg.data.max_response_length == 4096
    assert cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == 8192


@pytest.mark.parametrize("argv,message", [
    (("distill", "--export", "/e", "--data", "/d", "--runs-root", "/r", "--print"), "needs --teacher"),
    (("both", "--export", "/e", "--data", "/d", "--runs-root", "/r", "--print"), "unknown smoke"),
    (("rl", "--data", "/d", "--runs-root", "/r", "--print"), "needs --export, --data and a runs root"),
    (("rl", "--export", "/e", "--data", "/d", "--runs-root", "/r"), "needs --python"),
    (("rl", "--export", "/e", "--data", "/d", "--runs-root", "/r", "--bogus"), "unknown argument"),
])
def test_the_script_refuses(argv, message):
    proc = _run(*argv)
    assert proc.returncode == 2 and message in proc.stderr, proc.stderr


def test_a_run_without_its_inputs_stops_before_anything_starts(tmp_path):
    proc = _run("rl", "--export", str(tmp_path / "no_export"), "--data", str(tmp_path), "--runs-root",
                str(tmp_path / "runs"), "--python", "/no/python")
    assert proc.returncode == 2 and "missing inputs" in proc.stderr
    assert str(tmp_path / "math500.jsonl") in proc.stderr and not (tmp_path / "runs").exists()
