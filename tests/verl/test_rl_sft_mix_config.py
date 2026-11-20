"""CPU tests of the RL recipe: configs/verl/rl_sft_mix.yaml and the arms of scripts/verl/run_rl.sh.

Each test asks the run script for the arguments of an arm (its --print mode checks nothing and
starts nothing), composes rl_sft_mix with them the way the launcher does, and compares the result
with the recipe. The tests that run the script further give it "echo" or a small stand-in as the
python, so the launcher command is printed instead of run. Run with the verl python and
``PYTHONPATH=<repo>/src``, like test_launch.py. Nothing starts Ray or touches a GPU.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orbit_verl.data import distill_data
from orbit_verl.launch import config as lconfig
from orbit_verl.launch import env as lenv
from orbit_verl.launch import prepare as lprepare
from test_launch import REPO, _compose, _export

pytestmark = pytest.mark.skipif(importlib.util.find_spec("verl") is None,
                                reason="composing a config needs verl's own config files")

SCRIPT = REPO / "scripts" / "verl" / "run_rl.sh"
CONFIG = "rl_sft_mix"
ARMS = ("smoke", "math", "code", "mix", "mix_ru")
REAL_ARMS = ("math", "code", "mix", "mix_ru")
# Neutral paths. The --print mode checks none of them, so none has to exist.
DATA = "/data/rl"
RUNS = "/runs/verl"
EXPORT = "/exports/sft_v5_step7000"
MODEL = "sft_v5_step7000"
REQUIRED = ["--export", EXPORT, "--data", DATA, "--runs-root", RUNS, "--python", "/x/py"]
# The kept rows of the Russian pass-rate run have a file name of their own here.
RU_ENV = {"RU_KEEP": "ru_keep_test.jsonl", "RU_COUNT": "800"}
# The variables that the script reads. The tests set each one explicitly.
SCRIPT_ENV = ("RECIPE_TAG", "SMOKE_STEPS", "RU_KEEP", "RU_COUNT", "VERL_PYTHON", "ORBIT_VERL_RUNS", "RANK")

GOLD = {
    "math": ["math_keep.jsonl:orbit/math:math"],
    "code": ["code_keep.jsonl:orbit/code:code"],
    "mix": ["math_keep.jsonl:orbit/math:math:3800", "code_keep.jsonl:orbit/code:code"],
    "mix_ru": ["math_keep.jsonl:orbit/math:math:3800", "code_keep.jsonl:orbit/code:code",
               "ru_keep_test.jsonl:orbit/math#ru_train:math_ru:800"],
}
GOLD["smoke"] = GOLD["mix"]
# The validation sets of every arm, without the data root, in the order of the dataset.
VAL = [
    "math500.jsonl:orbit/math#think:sft_think",
    "math500.jsonl:orbit/math#no_think:sft_no_think:200",
    "lcb_val78.jsonl:orbit/code#think:sft_think_code",
    "mgsm_ru.jsonl:orbit/math#ru:sft_think_ru",
    "gsm8k_en_paired250.jsonl:orbit/math#mgsm_en:sft_think",
    "chat_probe_ru.jsonl:orbit/none#chat_ru:sft_chat_think",
    "chat_probe_ru.jsonl:orbit/none#chat_ru2:sft_chat_think",
    "chat_probe_en.jsonl:orbit/none#chat_en:sft_chat_think",
]


def _script(*argv: str, required: bool = True, **env: str | None) -> subprocess.CompletedProcess:
    """Run run_rl.sh with the script's variables taken from ``env`` only; None leaves a variable unset.

    ``argv`` starts with the arm. The required options follow it unless ``required`` is false.
    """
    full = {key: value for key, value in os.environ.items() if key not in SCRIPT_ENV}
    full.update(RU_ENV)
    for key, value in env.items():
        if value is None:
            full.pop(key, None)
        else:
            full[key] = value
    args = [*argv[:1], *(REQUIRED if required else []), *argv[1:]]
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=full)


def _arm(arm: str, *options: str, extra: tuple[str, ...] = (), **env: str | None):
    """The launcher options and the hydra overrides of an arm, as the script prints them."""
    tail = ("--", *extra) if extra else ()
    proc = _script(arm, *options, "--print", *tail, **env)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    cut = lines.index("--")
    return lines[:cut], lines[cut + 1:]


def _option(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def _arm_cfg(tmp_path: Path, arm: str, *options: str, **env: str | None):
    """rl_sft_mix composed with the arm's overrides, a fake qwen3 export of 32K context and an 8-GPU node."""
    args, overrides = _arm(arm, *options, **env)
    assert _option(args, "--config") == CONFIG
    export = _export(tmp_path, "qwen3", max_positions=32768)
    return _compose(CONFIG, tmp_path, *overrides, export=export, model_name=_option(args, "--model-name"),
                    mode=lconfig.mode_overrides(nnodes=1, gpus_per_node=8))


def _leaves(node, prefix: str = "") -> dict[str, object]:
    """Every value of a plain config by its dotted key. A list counts as one value."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out.update(_leaves(value, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: node}


def _unresolved(cfg) -> dict[str, object]:
    """The config's values with interpolations left as written, so derived paths compare as their formula."""
    from omegaconf import OmegaConf

    return _leaves(OmegaConf.to_container(cfg, resolve=False))


def _differ(a: dict, b: dict) -> set[str]:
    return {key for key in set(a) | set(b) if a.get(key) != b.get(key)}


def test_the_config_is_an_experiment():
    assert CONFIG in lconfig.available_configs()


def test_gold_rows_and_validation_sets_have_no_default(tmp_path):
    export = _export(tmp_path, "qwen3", max_positions=32768)
    problem = lprepare._data_config_problem(_compose(CONFIG, tmp_path, export=export))
    assert problem is not None and "gold inputs" in problem.message
    cfg = _compose(CONFIG, tmp_path, 'orbit.prepare.data.gold=["/d/m.jsonl:orbit/math:math"]', export=export)
    problem = lprepare._data_config_problem(cfg)
    assert problem is not None and "validation set" in problem.message


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_has_the_recipe(arm, tmp_path):
    from orbit_verl.distill.preflight import check_think_budget
    from orbit_verl.launch.train import validate

    cfg = _arm_cfg(tmp_path, arm)
    smoke = arm == "smoke"
    actor = cfg.actor_rollout_ref.actor
    assert (actor.policy_loss.loss_mode, actor.loss_agg_mode, actor.entropy_coeff) == (
        "rltt_exit", "seq-mean-token-mean", 0.0)
    assert (actor.use_kl_loss, actor.kl_loss_type, actor.kl_loss_coef) == (True, "low_var_kl", 1e-3)
    assert actor.ppo_mini_batch_size == cfg.data.train_batch_size == (16 if smoke else 64)
    assert actor.ppo_epochs == 1
    optim = actor.optim
    # The warmup is a count of 10 steps; orbit_base's ratio would give 20 at the 200-step cap.
    assert (optim.lr, optim.lr_scheduler_type, optim.lr_warmup_steps) == (1e-6, "constant", 10)
    assert (optim.weight_decay, list(optim.betas), optim.clip_grad) == (0.1, [0.9, 0.99], 0.1)
    rollout = cfg.actor_rollout_ref.rollout
    assert (rollout.n, rollout.temperature, rollout.top_p, rollout.top_k) == (8, 0.9, 1.0, -1)
    # orbit_base validates greedily; this recipe validates like the SFT eval.
    val = rollout.val_kwargs
    assert (val.temperature, val.top_p, val.top_k, val.n) == (0.6, 0.95, 20, 1)
    assert rollout.engine_kwargs.vllm.kv_cache_dtype == "fp8"
    assert rollout.gpu_memory_utilization == 0.8
    correction = cfg.algorithm.rollout_correction
    assert (correction.rollout_is, correction.rollout_is_threshold, correction.rollout_rs, correction.bypass_mode) == (
        "token", 2.0, None, False)
    assert (cfg.algorithm.adv_estimator, cfg.algorithm.norm_adv_by_std_in_grpo, cfg.algorithm.use_kl_in_reward) == (
        "grpo", True, False)
    assert (cfg.data.max_prompt_length, cfg.data.max_response_length, cfg.data.seed) == (2048, 16384, 0)
    assert (actor.ppo_max_token_len_per_gpu, cfg.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu,
            rollout.log_prob_max_token_len_per_gpu) == (18432, 18432, 18432)
    assert (cfg.orbit.rltt.train_exit_gate, cfg.orbit.rltt.exit_drift_alarm) == (False, 0.05)
    kwargs = cfg.reward.custom_reward_function.reward_kwargs
    assert (kwargs.truncated_score, kwargs.code_threads) == (0.0, None)
    assert "loop_penalty_coef" not in kwargs
    trainer = cfg.trainer
    assert (trainer.total_training_steps, trainer.save_freq, trainer.test_freq) == (
        (2, 2, 2) if smoke else (200, 5, 10))
    assert trainer.val_before_train is True and trainer.max_actor_ckpt_to_keep is None
    assert trainer.rollout_data_dir == f"{cfg.orbit.run_dir}/rollouts"
    assert trainer.validation_data_dir == f"{cfg.orbit.run_dir}/val_generations"
    # ClearML stays off unless the run script is given a credentials file.
    assert list(trainer.logger) == ["console", "file"]

    # No teacher: distillation stays off, and the launcher has no GPUs to split.
    assert cfg.distillation.enabled is False
    assert lconfig.finish_distillation(cfg, nnodes=1, gpus_per_node=8) is None
    data = cfg.orbit.prepare.data
    assert lprepare.data_builder(cfg) == lprepare.DISTILL_BUILDER
    assert (data.pool, data.think_share, data.seed) == (None, 1.0, 0)
    # No thinking budget and no loop-tail mask: every answer closes or runs to the limit on its own.
    job = lenv.job_env(cfg, str(cfg.orbit.repo_root), base_env={})
    assert "ORBIT_THINK_BUDGET" not in job and "ORBIT_MASK_LOOP_TAIL" not in job
    assert job["ORBIT_RLTT_TRAIN_EXIT_GATE"] == "false"
    assert lprepare.check_settings(cfg, environ={}) == [] and check_think_budget(cfg) == []
    validate(cfg)


@pytest.mark.parametrize("arm", ARMS)
def test_the_dataset_specs_of_every_arm_are_accepted(arm, tmp_path):
    cfg = _arm_cfg(tmp_path, arm)
    assert lprepare._data_config_problem(cfg) is None


@pytest.mark.parametrize("arm", ARMS)
def test_the_gold_rows_and_the_validation_sets_of_each_arm(arm, tmp_path):
    data = _arm_cfg(tmp_path, arm).orbit.prepare.data
    assert list(data.gold) == [f"{DATA}/{spec}" for spec in GOLD[arm]]
    assert data.name == f"rl_{arm}__qwen3"
    val = [spec.rsplit(":200", 1)[0] for spec in VAL] if arm == "smoke" else VAL
    assert list(data.val) == [f"{DATA}/{spec}" + (":8" if arm == "smoke" else "") for spec in val]
    specs = [distill_data.parse_val_spec(spec) for spec in data.val]
    distill_data._check_val_sources(specs)


def test_a_russian_keep_given_as_a_path_is_used_as_it_is(tmp_path):
    data = _arm_cfg(tmp_path, "mix_ru", RU_KEEP="/elsewhere/ru.jsonl", RU_COUNT="all").orbit.prepare.data
    assert list(data.gold)[-1] == "/elsewhere/ru.jsonl:orbit/math#ru_train:math_ru"


def test_the_russian_keep_defaults_to_every_row_of_ru_math_keep(tmp_path):
    data = _arm_cfg(tmp_path, "mix_ru", RU_KEEP=None, RU_COUNT=None).orbit.prepare.data
    assert list(data.gold)[-1] == f"{DATA}/ru_math_keep.jsonl:orbit/math#ru_train:math_ru"


def test_the_arms_differ_only_in_their_rows_and_names(tmp_path):
    base = _unresolved(_arm_cfg(tmp_path, "math"))
    for arm in REAL_ARMS[1:]:
        other = _unresolved(_arm_cfg(tmp_path, arm))
        assert _differ(base, other) == {"orbit.run_name", "orbit.prepare.data.name",
                                        "orbit.prepare.data.gold"}, arm


def test_the_gpu_share_changes_only_the_vllm_share(tmp_path):
    # 0.8 fits a 141 GB GPU; on 80 GB GPUs the actor needs room for its 18K-token micro-batches.
    for arm in ARMS:
        default = _unresolved(_arm_cfg(tmp_path, arm))
        smaller = _unresolved(_arm_cfg(tmp_path, arm, "--gpu-mem", "0.7"))
        assert _differ(default, smaller) == {"actor_rollout_ref.rollout.gpu_memory_utilization"}, arm
        assert smaller["actor_rollout_ref.rollout.gpu_memory_utilization"] == 0.7


def test_the_smoke_runs_twice_under_one_name(tmp_path):
    first = _arm_cfg(tmp_path, "smoke", SMOKE_STEPS="2")
    second = _arm_cfg(tmp_path, "smoke", SMOKE_STEPS="3")
    assert _differ(_unresolved(first), _unresolved(second)) == {"trainer.total_training_steps"}
    assert (first.trainer.total_training_steps, second.trainer.total_training_steps) == (2, 3)
    assert first.orbit.run_name == "rl_smoke__lr1e6_fp8tis"

    # The first launch saved global_step_2 and logged its validation. The second launch may resume it, and it
    # skips the validation before its first step.
    ckpt = Path(first.trainer.default_local_dir)
    (ckpt / "global_step_2").mkdir(parents=True)
    (ckpt / lprepare.CHECKPOINT_TRACKER).write_text("2")
    run_dir = Path(first.orbit.run_dir)
    (run_dir / "config.yaml").write_text(lconfig.to_yaml(first))
    line = {"step": 2, "data": {"val-core/orbit/math#think/acc/mean@1": 0.75}}
    (run_dir / "metrics.20260101-120000.jsonl").write_text(json.dumps(line) + "\n")
    assert lprepare.check_resume(second) == []
    assert lprepare.resume_validation(second) is not None and second.trainer.val_before_train is False


def test_the_launcher_arguments_of_an_arm():
    args, overrides = _arm("math")
    assert args == ["--config", CONFIG, "--export", EXPORT, "--model-name", MODEL, "--runs-root", RUNS,
                    "--gpus-per-node", "8"]
    assert not any(arg.startswith("--force-prepare") for arg in [*args, *overrides])
    assert overrides[0] == "orbit.run_name=rl_math__lr1e6_fp8tis"
    args, _ = _arm("math", "--model-name", "other_dir", "--no-prepare")
    assert _option(args, "--model-name") == "other_dir" and args[-1] == "--no-prepare"


def test_the_runs_root_comes_from_the_environment():
    proc = _script("math", "--export", EXPORT, "--data", DATA, "--python", "/x/py", "--print", required=False,
                   ORBIT_VERL_RUNS="/env/runs")
    assert proc.returncode == 0, proc.stderr
    assert _option(proc.stdout.splitlines(), "--runs-root") == "/env/runs"


def test_clearml_logs_only_with_a_credentials_file():
    _, overrides = _arm("math")
    assert not [item for item in overrides if "clearml" in item]
    _, overrides = _arm("math", "--clearml-env", "/secrets/clearml.env")
    assert "orbit.clearml_env=/secrets/clearml.env" in overrides
    assert "trainer.logger=[console,file,clearml]" in overrides


def test_overrides_after_the_separator_need_a_recipe_tag_and_come_last():
    extra = ("actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=auto",
             "algorithm.rollout_correction.rollout_is=null")
    _, overrides = _arm("code", "--gpu-mem", "0.7", extra=extra, RECIPE_TAG="lr1e6_bf16")
    assert overrides[0] == "orbit.run_name=rl_code__lr1e6_bf16"
    assert overrides[-2:] == list(extra)


@pytest.mark.parametrize("argv,env,message,required", [
    (("math", "--print", "--force-prepare"), {}, "never passes --force-prepare", True),
    (("math", "--print", "--", "--force-prepare"), {"RECIPE_TAG": "x"}, "never passes --force-prepare", True),
    (("both", "--print"), {}, "unknown arm", True),
    (("math", "--print", "--", "actor_rollout_ref.actor.optim.lr=5e-7"), {}, "RECIPE_TAG", True),
    (("mix_ru", "--print"), {"RU_COUNT": "lots"}, "RU_COUNT", True),
    (("smoke", "--print"), {"SMOKE_STEPS": "0"}, "SMOKE_STEPS", True),
    (("math", "--print", "--dry-run"), {}, "only one of", True),
    (("math", "--prepare-only", "--print"), {}, "only one of", True),
    (("math", "--no-prepare", "--prepare-only"), {}, "only one of --no-prepare and --prepare-only", True),
    (("math", "--print", "--bogus"), {}, "unknown argument", True),
    (("math", "--print"), {}, "needs --export --data", False),
])
def test_the_script_refuses(argv, env, message, required):
    proc = _script(*argv, required=required, **env)
    assert proc.returncode == 2 and message in proc.stderr, proc.stderr


def _run_dirs(tmp_path: Path, with_inputs: bool = True) -> list[str]:
    """The path options of run_rl.sh on fresh dirs, with the inputs of the math arm when asked to."""
    data, runs, export = tmp_path / "data", tmp_path / "runs", tmp_path / "export"
    for path in (data, runs, export):
        path.mkdir()
    if with_inputs:
        for spec in [*GOLD["math"], *VAL]:
            (data / spec.split(":", 1)[0]).touch()
    return ["--export", str(export), "--data", str(data), "--runs-root", str(runs)]


def test_a_dry_run_runs_the_launchers_dry_run_and_builds_nothing(tmp_path):
    # The inputs need not exist for a dry run, but the dirs must: the script checks that it runs where they are.
    proc = _script("math", *_run_dirs(tmp_path, with_inputs=False), "--python", "echo", "--dry-run",
                   required=False)
    assert proc.returncode == 0 and "missing inputs" in proc.stderr, proc.stderr
    command = proc.stdout.splitlines()[-1].split(" ")
    assert command[0] == str(REPO / "scripts" / "verl" / "launch.py") and "--dry-run" in command
    assert not (tmp_path / "runs" / "models").exists() and not (tmp_path / "runs" / "runs").exists()


def test_a_run_starts_the_launcher_on_this_node_and_keeps_its_log(tmp_path):
    """"echo" stands in for the verl python, so the launcher command is printed and logged."""
    options = _run_dirs(tmp_path)
    proc = _script("math", *options, required=False, VERL_PYTHON="echo")
    assert proc.returncode == 0, proc.stderr
    assert "on this node, mode run" in proc.stdout
    command = proc.stdout.splitlines()[-1].split(" ")
    assert command[0] == str(REPO / "scripts" / "verl" / "launch.py")
    assert (_option(command, "--config"), _option(command, "--gpus-per-node")) == (CONFIG, "8")
    assert (_option(command, "--export"), _option(command, "--runs-root")) == (options[1], options[5])
    assert "orbit.run_name=rl_math__lr1e6_fp8tis" in command and "--no-prepare" not in command
    # The launcher's output goes to the run dir. A second launch keeps the first log under a date-time name,
    # and another node of a multi-node run writes a log of its own.
    run_dir = tmp_path / "runs" / "runs" / "rl_math__lr1e6_fp8tis"
    assert (run_dir / "launch.log").read_text().splitlines() == [" ".join(command)]
    assert _script("math", *options, required=False, VERL_PYTHON="echo").returncode == 0
    assert _script("math", *options, "--no-prepare", required=False, VERL_PYTHON="echo", RANK="1").returncode == 0
    logs = sorted(path.name for path in run_dir.iterdir())
    assert len(logs) == 3 and logs[0].startswith("launch.2") and logs[1:] == ["launch.log", "launch.node1.log"]
    assert "--no-prepare" in (run_dir / "launch.node1.log").read_text()


def test_prepare_only_builds_without_a_gpu_and_starts_nothing(tmp_path):
    proc = _script("math", *_run_dirs(tmp_path), "--python", "echo", "--prepare-only", required=False)
    assert proc.returncode == 0, proc.stderr
    command = proc.stdout.splitlines()[-1].split(" ")
    assert "--prepare-only" in command and "--gpus-per-node" not in command
    assert not (tmp_path / "runs" / "runs").exists()


def test_a_run_stops_without_its_inputs(tmp_path):
    proc = _script("math", *_run_dirs(tmp_path, with_inputs=False), "--python", "echo", required=False)
    assert proc.returncode == 2 and "missing inputs" in proc.stderr


def test_a_no_prepare_launch_needs_no_data_but_the_export(tmp_path):
    # A node of a multi-node run reads only the prepared dirs under the runs root, so --data need not exist there.
    options = _run_dirs(tmp_path, with_inputs=False)
    (tmp_path / "data").rmdir()
    proc = _script("math", *options, "--no-prepare", required=False, VERL_PYTHON="echo", RANK="1")
    assert proc.returncode == 0, proc.stderr
    assert "--no-prepare" in proc.stdout.splitlines()[-1].split(" ")
    # The prepared model dir links to the export's weights, so every node still needs the export.
    (tmp_path / "export").rmdir()
    proc = _script("math", *options, "--no-prepare", required=False, VERL_PYTHON="echo", RANK="1")
    assert proc.returncode == 2 and f"{tmp_path / 'export'} does not exist" in proc.stderr


def test_the_gate_stops_a_full_arm_until_the_smoke_passes(tmp_path):
    # The stand-in python runs the gate (python sandbox_gate.py ...) with the real python and only
    # prints the launcher command.
    python = tmp_path / "python"
    python.write_text(f'#!/bin/sh\ncase "$1" in *sandbox_gate.py) exec "{sys.executable}" "$@";; esac\necho "$@"\n')
    python.chmod(0o755)
    options = [*_run_dirs(tmp_path), "--python", str(python)]
    proc = _script("math", *options, "--gate", required=False)
    assert proc.returncode == 2 and "the sandbox gate did not pass" in proc.stderr, proc.stderr
    assert "rl_smoke__lr1e6_fp8tis does not exist" in proc.stderr
    # A dry run reports the gate and goes on, and without --gate there is no gate at all.
    proc = _script("math", *options, "--gate", "--dry-run", required=False)
    assert proc.returncode == 0 and "a run of math would stop here" in proc.stderr, proc.stderr
    assert _script("math", *options, required=False).returncode == 0
