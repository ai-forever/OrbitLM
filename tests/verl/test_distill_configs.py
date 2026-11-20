"""CPU tests of the distillation experiments: configs/verl/distill_sft.yaml, distill_rl_sft.yaml and
the 16K hybrid with a thinking budget, distill_rl_sft_16k.yaml.

Run with the verl python and ``PYTHONPATH=<repo>/src``, like test_launch.py: composing needs hydra
and verl's own config files. Each test composes a config the way the launcher does, applies the
launcher's distillation step (``finish_distillation``), and checks that verl's own config classes
and the launcher's pre-flight checks accept the result. Nothing starts Ray or touches a GPU. The
teacher and the student are fake model dirs that hold only a config.json and a tokenizer.json.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

from orbit_verl.data import distill_data, prompts
from orbit_verl.data.response_info import STOP_THINKING_TEXT
from orbit_verl.distill import preflight
from orbit_verl.distill import teacher as teacher_mod
from orbit_verl.launch import config as lconfig
from orbit_verl.launch import env as lenv
from orbit_verl.launch import prepare as lprepare
from test_distill_launch import STUDENT_EXTRA, _teacher_dir, _write_tokenizer
from test_launch import REPO, _compose, _export

pytestmark = pytest.mark.skipif(importlib.util.find_spec("verl") is None,
                                reason="composing a config needs verl's own config files")

CONFIGS = ("distill_sft", "distill_rl_sft")
# The 16K variant of distill_rl_sft. It changes the lengths, so the 8K recipe test below does not cover it.
HYBRID_16K = "distill_rl_sft_16k"
ALL_CONFIGS = (*CONFIGS, HYBRID_16K)
M500 = "/data/math500.jsonl"
VAL_SPECS = [f"{M500}:orbit/math#think:sft_think", f"{M500}:orbit/math#no_think:sft_no_think"]
# The validation override as scripts/verl/run_distill.sh writes it. The "#" of the tags is legal only
# inside quotes.
VAL_OVERRIDE = "orbit.prepare.data.val=[" + ",".join(f'"{spec}"' for spec in VAL_SPECS) + "]"


def _distill_cfg(name: str, tmp_path: Path, *extra: str, nnodes: int = 1, gpus: int = 8, teacher: Path | None = None):
    """``name`` composed with a qwen3 export, a teacher, a pool and the MATH-500 validation sets, then split."""
    teacher = teacher or _teacher_dir(tmp_path)
    # The SFT export this recipe is for has a 32K context. The override states it, so the launcher's
    # length check sees the context that the recipe was written for.
    user = [f"distillation.teacher_models.teacher_model.model_path={teacher}",
            "orbit.prepare.model.max_position_embeddings=32768",
            "orbit.prepare.data.pool=/data/pool.jsonl", VAL_OVERRIDE, *extra]
    cfg = _compose(name, tmp_path, *user, export=_export(tmp_path, "qwen3"),
                   mode=lconfig.mode_overrides(nnodes=nnodes, gpus_per_node=gpus))
    setup = lconfig.finish_distillation(cfg, nnodes=nnodes, gpus_per_node=gpus, user_overrides=user)
    return cfg, setup


def test_the_configs_are_experiments():
    assert set(ALL_CONFIGS) <= set(lconfig.available_configs())
    # The start export prepares as release_qwen3. An export prepared by hand with
    # --embedding-padding-idx none records the lineage unknown, and the recipes accept it too.
    for name in ALL_CONFIGS:
        fits = lconfig.compose_config(name, repo_root=Path("/repo")).orbit.fits
        assert list(fits.lineages) == ["release_qwen3", "unknown"]


def test_the_values_without_a_default_are_reported_before_anything_is_built(tmp_path):
    cfg = _compose("distill_sft", tmp_path, export=_export(tmp_path, "qwen3"))
    problem = lprepare._data_config_problem(cfg)
    assert problem is not None and "prompt pool" in problem.message
    for name in ALL_CONFIGS:
        cfg = _compose(name, tmp_path, "orbit.prepare.data.pool=/data/pool.jsonl", export=_export(tmp_path, "qwen3"))
        problem = lprepare._data_config_problem(cfg)
        assert problem is not None and "validation set" in problem.message
        teacher_problems = preflight.check_teacher_paths(cfg)
        assert len(teacher_problems) == 1 and "model_path is not set" in teacher_problems[0].message


@pytest.mark.parametrize("name", CONFIGS)
def test_the_recipe_composes_as_planned(name, tmp_path):
    cfg, setup = _distill_cfg(name, tmp_path)
    hybrid = name == "distill_rl_sft"
    loss = cfg.distillation.distillation_loss
    assert cfg.distillation.enabled and cfg.distillation.teacher_key == "data_source"
    assert (loss.loss_mode, loss.use_policy_gradient, loss.use_task_rewards) == ("k1", True, hybrid)
    assert (loss.distillation_loss_coef, loss.loss_max_clamp) == (1.0, 10.0)
    actor = cfg.actor_rollout_ref.actor
    assert (actor.policy_loss.loss_mode, actor.use_kl_loss, actor.loss_agg_mode) == ("vanilla", False, "token-mean")
    assert (actor.optim.lr, actor.optim.lr_scheduler_type, actor.optim.lr_warmup_steps_ratio) == (1e-6, "constant", 0.1)
    rollout = cfg.actor_rollout_ref.rollout
    assert (rollout.n, rollout.temperature, rollout.top_p, rollout.top_k) == (4, 1.0, 1.0, -1)
    assert rollout.engine_kwargs.vllm.kv_cache_dtype == "fp8"
    assert rollout.agent.agent_loop_manager_class == teacher_mod.MANAGER_CLASS_FQN
    assert (cfg.data.train_batch_size, cfg.data.max_prompt_length, cfg.data.max_response_length) == (132, 1088, 8192)
    assert actor.ppo_mini_batch_size == 132
    trainer = cfg.trainer
    assert (trainer.total_training_steps, trainer.save_freq, trainer.test_freq, trainer.val_before_train) == (
        200, 20, 20, True)
    assert trainer.rollout_data_dir == f"{cfg.orbit.run_dir}/rollouts"
    assert trainer.validation_data_dir == f"{cfg.orbit.run_dir}/val_generations"
    assert cfg.algorithm.adv_estimator == "grpo"
    assert cfg.reward.custom_reward_function.reward_kwargs.truncated_score == 0.0

    inference = cfg.distillation.teacher_models.teacher_model.inference
    assert (inference.tensor_model_parallel_size, inference.gpu_memory_utilization, inference.enforce_eager) == (
        1, 0.85, False)
    assert (inference.max_num_batched_tokens, inference.engine_kwargs.vllm.kv_cache_dtype) == (16384, "auto")
    # The launcher filled the teacher's context and gave it 2 of the node's 8 GPUs.
    assert inference.max_model_len == 1088 + 8192 + 1 and setup.max_model_len == {"teacher_model": 9281}
    assert setup.topology == ("shared", 1, 6, 1, 2)

    # The data side: the distill builder, a dataset name per variant, and the tagged validation sets.
    data = cfg.orbit.prepare.data
    assert lprepare.data_builder(cfg) == lprepare.DISTILL_BUILDER
    assert list(data.val) == VAL_SPECS
    assert data.name == ("distill_rl_mix__qwen3" if hybrid else "distill_mix__qwen3")
    assert lprepare._data_config_problem(cfg) is None
    command = lprepare.distill_command(cfg)
    assert "--pool /data/pool.jsonl" in command and (" --gold " in command) is hybrid
    assert lprepare.check_lengths(cfg) == []

    job = lenv.job_env(cfg, str(cfg.orbit.repo_root), base_env={})
    # The hybrid turns the loop-tail mask off, so that the GRPO penalty reaches the looping tokens.
    assert job["ORBIT_MASK_LOOP_TAIL"] == ("false" if hybrid else "true")
    assert job["ORBIT_NO_THINK_MAX_TOKENS"] == "4096"
    # Only the 16K hybrid has a thinking budget. In the 8K recipes a long thinking block runs to the length limit.
    assert prompts.THINK_BUDGET_ENV not in job and prompts.think_budget(job) is None
    assert teacher_mod.teacher_strip_prefix(cfg) == (151643,)


# The keys that the 16K hybrid must take over from the 8K hybrid unchanged: the teacher, the losses,
# GRPO, the batch, the sampling, the dataset, the validation, the reward and the schedule.
SHARED_WITH_THE_8K_HYBRID = (
    "distillation.enabled",
    "distillation.teacher_key",
    "distillation.distillation_loss",
    "distillation.teacher_models.teacher_model.model_path",
    "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size",
    "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization",
    "distillation.teacher_models.teacher_model.inference.engine_kwargs",
    "algorithm.adv_estimator",
    "algorithm.norm_adv_by_std_in_grpo",
    "data.train_batch_size",
    "data.max_prompt_length",
    "actor_rollout_ref.actor.ppo_mini_batch_size",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.actor.loss_agg_mode",
    "actor_rollout_ref.actor.policy_loss.loss_mode",
    "actor_rollout_ref.actor.optim.lr",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.temperature",
    "actor_rollout_ref.rollout.val_kwargs",
    "actor_rollout_ref.rollout.agent.agent_loop_manager_class",
    "orbit.distill",
    "orbit.prepare.data",
    "orbit.data_dir",
    "reward.custom_reward_function.reward_kwargs.truncated_score",
    "reward.custom_reward_function.reward_kwargs.code_threads",
    "trainer.total_training_steps",
    "trainer.save_freq",
    "trainer.test_freq",
    "trainer.val_before_train",
)


def _plain(cfg, key: str):
    """The value at ``key`` with interpolations resolved, as plain Python, so two configs compare by value."""
    from omegaconf import OmegaConf

    node = OmegaConf.select(cfg, key, throw_on_missing=True)
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def test_the_16k_hybrid_composes_as_planned(tmp_path):
    base, base_setup = _distill_cfg("distill_rl_sft", tmp_path)
    cfg, setup = _distill_cfg(HYBRID_16K, tmp_path)
    for key in SHARED_WITH_THE_8K_HYBRID:
        assert _plain(cfg, key) == _plain(base, key), key
    assert setup.topology == base_setup.topology == ("shared", 1, 6, 1, 2)
    # A run directory of its own, so the 16K run does not write over the 8K one. Both read the same dataset.
    assert cfg.orbit.run_name.startswith(f"{HYBRID_16K}__") and cfg.orbit.run_name != base.orbit.run_name
    assert cfg.trainer.experiment_name == cfg.orbit.run_name
    assert cfg.orbit.prepare.data.name == "distill_rl_mix__qwen3"
    assert lprepare.distill_command(cfg) == lprepare.distill_command(base)

    # The lengths: 16K responses, and every token budget holds a whole prompt and response.
    prompt, response = 1088, 16384
    assert (cfg.data.max_prompt_length, cfg.data.max_response_length) == (prompt, response)
    assert cfg.actor_rollout_ref.rollout.response_length == response
    assert cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu >= prompt + response
    assert cfg.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu >= prompt + response
    assert cfg.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu >= prompt + response
    inference = cfg.distillation.teacher_models.teacher_model.inference
    assert inference.max_model_len == prompt + response + 1 and setup.max_model_len == {"teacher_model": 17473}
    # One prefill step of the teacher holds a whole sequence, as in the 8K recipe.
    assert inference.max_num_batched_tokens >= inference.max_model_len >= prompt + response
    assert lprepare.check_lengths(cfg) == []

    # The thinking budget travels to every worker next to the 8K hybrid's two switches.
    job = lenv.job_env(cfg, str(cfg.orbit.repo_root), base_env={})
    assert job[prompts.THINK_BUDGET_ENV] == "14336" and prompts.think_budget(job) == 14336
    assert job["ORBIT_NO_THINK_MAX_TOKENS"] == "4096" and job["ORBIT_MASK_LOOP_TAIL"] == "false"
    # The budget stays below the response length by more than the whole stop-thinking text, counted
    # in characters. Every id of that text is at least one character, so the answer has room left.
    assert prompts.think_budget(job) + len(STOP_THINKING_TEXT) < response

    # The reward keeps forced answers at the cut-answer score.
    kwargs = cfg.reward.custom_reward_function.reward_kwargs
    assert kwargs.forced_score_scale is None and "closure_score" not in kwargs
    assert kwargs.truncated_score == 0.0


@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_verl_and_the_preflight_accept_the_composed_config(name, tmp_path):
    from verl.utils.config import omega_conf_to_dataclass

    from orbit_verl.launch.train import validate

    cfg, _ = _distill_cfg(name, tmp_path)
    _write_tokenizer(Path(cfg.actor_rollout_ref.model.path), extra=STUDENT_EXTRA)
    distill = omega_conf_to_dataclass(cfg.distillation)
    (teacher_cfg,) = distill.teacher_models.values()
    assert teacher_cfg.num_replicas == 2 and teacher_cfg.per_replica_world_size == 1
    validate(cfg)
    assert preflight.run_preflight(cfg) == ([], [])


@pytest.mark.parametrize("nnodes,extra,student", [
    (1, (), 6),
    (2, (), 12),
    (2, (f"{lconfig.TEACHER_NODES_KEY}=1",), 8),
    (1, (f"{lconfig.TEACHER_GPUS_KEY}=4",), 4),
])
def test_every_answer_of_a_step_divides_among_the_students_gpus(nnodes, extra, student, tmp_path):
    # verl's train_mini_batch asserts that the answers of a mini-batch divide evenly among the
    # student's data-parallel ranks. One mini-batch holds every answer of the step.
    cfg, setup = _distill_cfg("distill_sft", tmp_path, *extra, nnodes=nnodes)
    gpus = setup.topology.student_nnodes * setup.topology.student_gpus_per_node
    assert gpus == student
    answers = cfg.actor_rollout_ref.actor.ppo_mini_batch_size * cfg.actor_rollout_ref.rollout.n
    assert answers % gpus == 0


@pytest.mark.parametrize("teacher_gpus,nearest", [(1, 133), (3, 130)])
def test_layouts_whose_student_gpus_do_not_divide_the_batch_are_refused(teacher_gpus, nearest, tmp_path):
    # 1 or 3 teacher GPUs of an 8-GPU node leave 7 or 5 student GPUs, and 528 answers divide among
    # neither. The launcher reports this as a GPU problem with the nearest batch size that divides.
    cfg, setup = _distill_cfg("distill_sft", tmp_path, f"{lconfig.TEACHER_GPUS_KEY}={teacher_gpus}")
    assert setup.topology is None and f"Set data.train_batch_size={nearest}," in setup.topology_error
    cfg, setup = _distill_cfg("distill_sft", tmp_path, f"{lconfig.TEACHER_GPUS_KEY}={teacher_gpus}",
                              f"data.train_batch_size={nearest}")
    assert setup.topology is not None and setup.topology_error is None


def test_a_kl_arm_needs_the_task_loss_as_well(tmp_path):
    # verl drops the KL term with the task loss, so use_kl_loss alone is refused; the pair passes
    # with only the warning about the reference model's cost.
    cfg, _ = _distill_cfg("distill_sft", tmp_path, "actor_rollout_ref.actor.use_kl_loss=true")
    assert [p.commands for p in preflight.check_kl_loss(cfg)] == [
        ("distillation.distillation_loss.use_task_rewards=true",)]
    cfg, _ = _distill_cfg("distill_sft", tmp_path, "actor_rollout_ref.actor.use_kl_loss=true",
                          "distillation.distillation_loss.use_task_rewards=true")
    assert preflight.check_kl_loss(cfg) == [] and len(preflight.kl_loss_warnings(cfg)) == 1


def test_the_hybrid_placeholder_paths_stop_the_build(tmp_path):
    # An unset pool or gold list would read as empty and change the experiment without a word, so
    # the hybrid names paths that do not exist, and the dataset build refuses them.
    cfg = _compose("distill_rl_sft", tmp_path, VAL_OVERRIDE, export=_export(tmp_path, "qwen3"))
    assert lprepare._data_config_problem(cfg) is None
    data = cfg.orbit.prepare.data
    gold = [distill_data.parse_gold_spec(spec) for spec in data.gold]
    assert [(spec.data_source, spec.family) for spec in gold] == [("orbit/math", "math"), ("orbit/code", "code")]
    val = tmp_path / "math500.jsonl"
    val.write_text("")
    with pytest.raises(FileNotFoundError) as info:
        distill_data.convert(str(data.pool), str(tmp_path / "out"), tokenizer_dir=str(tmp_path), max_prompt_length=1024,
                             val=[f"{val}:orbit/math#think:sft_think"], gold=list(data.gold))
    for path in ("/path/to/prompts.jsonl", "/path/to/math_keep.jsonl", "/path/to/code_data/train.jsonl"):
        assert path in str(info.value)


def test_the_16k_thinking_budget_leaves_room_for_the_answer(tmp_path):
    # The budget is read from the composed orbit.env block, against the rollout's response length,
    # which follows data.max_response_length. 16,384 - 14,336 leaves 2,048 tokens for the stop-thinking
    # text and the answer after a forced close.
    cfg, _ = _distill_cfg("distill_rl_sft_16k", tmp_path)
    assert (cfg.orbit.env.ORBIT_THINK_BUDGET, cfg.actor_rollout_ref.rollout.response_length) == ("14336", 16384)
    assert preflight.check_think_budget(cfg) == []
    # A budget that leaves the answer less than 256 tokens is refused, with the largest budget that fits.
    key = "orbit.env.ORBIT_THINK_BUDGET"
    cfg, _ = _distill_cfg("distill_rl_sft_16k", tmp_path, f"{key}=16100")
    assert [p.commands for p in preflight.check_think_budget(cfg)] == [(f"{key}={16384 - 64 - 256}",)]


RUN_DISTILL = REPO / "scripts" / "verl" / "run_distill.sh"
# The input files of the recipe, by the names that run_distill.sh reads in --data.
DISTILL_INPUTS = ("prompts.jsonl", "math500.jsonl", "math_keep.jsonl", "code_train.jsonl", "code_val.jsonl",
                  "mgsm_ru.jsonl")
# The variables that the script reads besides its options. The tests set each one explicitly.
SCRIPT_ENV = ("VERL_PYTHON", "ORBIT_VERL_RUNS", "RANK")


def _run_distill(*argv: str, **env: str) -> subprocess.CompletedProcess:
    full = {key: value for key, value in os.environ.items() if key not in SCRIPT_ENV}
    full.update(env)
    return subprocess.run(["bash", str(RUN_DISTILL), *argv], capture_output=True, text=True, env=full)


def _distill_dirs(tmp_path: Path, with_inputs: bool = True) -> list[str]:
    """The path options of run_distill.sh, on fresh dirs that hold the inputs when asked to."""
    for name in ("data", "runs", "export", "teacher"):
        (tmp_path / name).mkdir()
    if with_inputs:
        for name in DISTILL_INPUTS:
            (tmp_path / "data" / name).touch()
    return ["--export", str(tmp_path / "export"), "--teacher", str(tmp_path / "teacher"), "--data",
            str(tmp_path / "data"), "--runs-root", str(tmp_path / "runs")]


def test_run_distill_prints_the_recipe_and_reads_the_math_keep_of_the_rl_arms(tmp_path):
    proc = _run_distill(*_distill_dirs(tmp_path), "--python", "/x/py", "--print", "--", "trainer.seed=1")
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    args, overrides = lines[:lines.index("--")], lines[lines.index("--") + 1:]
    assert args[args.index("--config") + 1] == HYBRID_16K
    assert args[args.index("--model-name") + 1] == "sft_v5_step7000"
    assert args[args.index("--gpus-per-node") + 1] == "8"
    data = tmp_path / "data"
    gold = f'orbit.prepare.data.gold=["{data}/math_keep.jsonl:orbit/math:math:6454",' \
           f'"{data}/code_train.jsonl:orbit/code:code:3300"]'
    assert gold in overrides and overrides[-1] == "trainer.seed=1"
    assert "orbit.run_name=distill_rl_16k" in overrides


def test_run_distill_runs_on_this_node_and_keeps_its_log(tmp_path):
    proc = _run_distill(*_distill_dirs(tmp_path), "--arm", "opd", VERL_PYTHON="echo")
    assert proc.returncode == 0, proc.stderr
    command = proc.stdout.splitlines()[-1].split(" ")
    assert command[0] == str(REPO / "scripts" / "verl" / "launch.py")
    assert command[command.index("--gpus-per-node") + 1] == "8"
    assert "distillation.distillation_loss.use_task_rewards=false" in command
    log = tmp_path / "runs" / "runs" / "distill_opd_16k" / "launch.log"
    assert log.read_text().splitlines() == [" ".join(command)]


def test_run_distill_prepares_once_for_several_nodes(tmp_path):
    options = [*_distill_dirs(tmp_path), "--python", "echo"]
    proc = _run_distill(*options, "--prepare-only")
    assert proc.returncode == 0, proc.stderr
    command = proc.stdout.splitlines()[-1].split(" ")
    assert "--prepare-only" in command and "--gpus-per-node" not in command
    # Every node then starts the same command with --no-prepare and writes a log of its own.
    proc = _run_distill(*options, "--no-prepare", RANK="1")
    assert proc.returncode == 0, proc.stderr
    assert "--no-prepare" in proc.stdout.splitlines()[-1].split(" ")
    assert (tmp_path / "runs" / "runs" / "distill_rl_16k" / "launch.node1.log").is_file()


@pytest.mark.parametrize("argv,message", [
    (("--print", "--dry-run"), "only one of"),
    (("--no-prepare", "--prepare-only"), "only one of --no-prepare and --prepare-only"),
    (("--print", "--bogus"), "unknown argument"),
])
def test_run_distill_refuses(argv, message, tmp_path):
    proc = _run_distill(*_distill_dirs(tmp_path), "--python", "/x/py", *argv)
    assert proc.returncode == 2 and message in proc.stderr, proc.stderr


def test_run_distill_stops_without_its_inputs(tmp_path):
    proc = _run_distill(*_distill_dirs(tmp_path, with_inputs=False), "--python", "echo")
    assert proc.returncode == 2 and "missing inputs" in proc.stderr


def test_a_no_prepare_launch_needs_no_data_files_but_the_teacher(tmp_path):
    # A node of a multi-node run reads only the prepared dataset under the runs root.
    options = [*_distill_dirs(tmp_path, with_inputs=False), "--python", "echo", "--no-prepare"]
    proc = _run_distill(*options, RANK="1")
    assert proc.returncode == 0, proc.stderr
    # Ray may place the teacher on any node, so every node needs it.
    (tmp_path / "teacher").rmdir()
    proc = _run_distill(*options, RANK="1")
    assert proc.returncode == 2 and f"missing inputs: {tmp_path / 'teacher'}" in proc.stderr
