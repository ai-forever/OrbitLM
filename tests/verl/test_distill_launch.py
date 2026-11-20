"""CPU tests of the launcher's side of on-policy distillation: the GPU split and the pre-flight checks.

Run with the verl python and ``PYTHONPATH=<repo>/src``, like test_launch.py. The split
(``launch.config.finish_distillation``) and the checks (``orbit_verl.distill.preflight``) are
tested on plain nested dicts and on tiny fake model dirs, which hold only a ``config.json`` and a
``tokenizer.json`` of the real shape: 151,643 made-up BPE tokens and the added tokens at Qwen3's
ids, plus our 12 code tokens for the student. Those tests need neither verl nor omegaconf. The
tests marked ``needs_verl`` compose real configs and run the launcher's dry run; nothing starts
Ray or touches a GPU.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orbit_verl import common
from orbit_verl.distill import preflight
from orbit_verl.distill import teacher as teacher_mod
from orbit_verl.launch import cli, cluster
from orbit_verl.launch import config as lconfig
from test_launch import CONFIG, DATA_NAME, NO_PLUGIN, SINGLE_MODE, _export, _set_env, _smoke_prepared

needs_verl = pytest.mark.skipif(importlib.util.find_spec("verl") is None,
                                reason="composing a config needs verl's own config files")

# Our tokenizer adds 12 code tokens after Qwen3's last id; the teacher has none of them.
STUDENT_EXTRA = 12
# Lengths of the dict configs below: the planned 1K prompts and 8K responses.
PROMPT, RESPONSE = 1024, 8192
# Prompts per step of the dict configs below, as in distill_sft.yaml.
BATCH = 132
# The thinking budget of the agent loop, as a config sets it.
THINK_BUDGET_KEY = "orbit.env.ORBIT_THINK_BUDGET"


@pytest.fixture
def process_state():
    """Restore os.environ and the working directory after a test that runs the launcher in-process."""
    saved, cwd = dict(os.environ), os.getcwd()
    yield
    os.environ.clear()
    os.environ.update(saved)
    os.chdir(cwd)


@pytest.fixture
def no_ray(monkeypatch):
    """Fail the test if anything tries to run a ray CLI command."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"a test tried to run {args[0] if args else kwargs}")
    monkeypatch.setattr(cluster.subprocess, "call", refuse)


# --------------------------------------------------------------------------- fake model dirs


def _write_tokenizer(model_dir: Path, *, extra: int = 0, changes: dict[int, str] | None = None,
                     merges: tuple[str, ...] = ("a b", "ab c")) -> None:
    """A tokenizer.json shaped like Qwen3's: BPE ids 0-151642, then added tokens up to 151668 (+ ``extra``).

    The special tokens sit at their Qwen3 ids. ``changes`` gives some ids another token, which is
    how a tokenizer of another family looks to the check.
    """
    specials = {index: token for token, index in preflight.SPECIAL_TOKENS.items()}
    tokens = {index: f"tok{index}" for index in range(151643)}
    tokens.update({index: specials.get(index, f"<|added_{index}|>")
                   for index in range(151643, preflight.QWEN3_TOKENS + extra)})
    tokens.update(changes or {})
    data = {
        "version": "1.0",
        "added_tokens": [{"id": index, "content": token, "special": True}
                         for index, token in tokens.items() if index >= 151643],
        "model": {"type": "BPE", "vocab": {token: index for index, token in tokens.items() if index < 151643},
                  "merges": list(merges)},
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")


def _teacher_dir(root: Path, *, name: str = "Qwen3-8B", vocab_size: int = preflight.QWEN3_VOCAB_SIZE,
                 max_positions: int = 40960, **tokenizer) -> Path:
    path = root / "teachers" / name
    _write_tokenizer(path, **tokenizer)
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "vocab_size": vocab_size, "max_position_embeddings": max_positions}))
    return path


def _student_dir(root: Path) -> Path:
    path = root / "student"
    _write_tokenizer(path, extra=STUDENT_EXTRA)
    return path


def _set(cfg: dict, dotted: str, value) -> dict:
    *parents, key = dotted.split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
    node[key] = value
    return cfg


def _cfg(student: Path, teacher, changes: dict | None = None) -> dict:
    """The keys of a composed distillation config that the launcher reads, as plain dicts.

    The teacher entry carries what verl's distillation.yaml gives it: the student's lengths and a
    null max_model_len. The batch is distill_sft's: 132 prompts x 4 answers in one mini-batch.
    """
    cfg = {
        "orbit": {},
        "data": {"max_prompt_length": PROMPT, "max_response_length": RESPONSE, "train_batch_size": BATCH},
        "trainer": {"nnodes": 1, "n_gpus_per_node": 8},
        "actor_rollout_ref": {
            "model": {"path": str(student), "tokenizer_path": None},
            "actor": {"use_kl_loss": False, "policy_loss": {"loss_mode": "vanilla"}, "ppo_mini_batch_size": BATCH},
            "rollout": {"temperature": 1.0, "prompt_length": PROMPT, "response_length": RESPONSE, "n": 4,
                        "agent": {"agent_loop_manager_class": teacher_mod.MANAGER_CLASS_FQN}},
        },
        "distillation": {
            "enabled": True, "nnodes": 0, "n_gpus_per_node": 8,
            "teacher_models": {"teacher_model": {
                "model_path": None if teacher is None else str(teacher),
                "inference": {"prompt_length": PROMPT, "response_length": RESPONSE, "max_model_len": None}}},
        },
    }
    for dotted, value in (changes or {}).items():
        _set(cfg, dotted, value)
    return cfg


def _messages(problems) -> str:
    return "\n".join(p.message for p in problems)


# --------------------------------------------------------------------------- the GPU split


def test_the_default_split_keeps_two_gpus_of_every_node_for_the_teacher():
    one = lconfig.split_distill_topology(nnodes=1, gpus_per_node=8)
    assert one == ("shared", 1, 6, 1, 2)
    assert one.overrides() == ["trainer.nnodes=1", "trainer.n_gpus_per_node=6", "distillation.nnodes=1",
                               "distillation.n_gpus_per_node=2"]
    assert one.describe() == "student 1 node(s) x 6 GPUs, teacher 1 node(s) x 2 GPUs (shared nodes)"
    # On several nodes both pools span every node.
    assert lconfig.split_distill_topology(nnodes=4, gpus_per_node=8, teacher_gpus_per_node=1) == ("shared", 4, 7, 4, 1)


def test_teacher_nodes_give_the_teacher_whole_nodes():
    split = lconfig.split_distill_topology(nnodes=3, gpus_per_node=8, teacher_gpus_per_node=2, teacher_nodes=1)
    # teacher_gpus_per_node does not apply: every node goes to one side whole.
    assert split == ("dedicated", 2, 8, 1, 8)
    assert lconfig.split_distill_topology(nnodes=2, gpus_per_node=8, teacher_nodes=0) == ("shared", 2, 6, 2, 2)


@pytest.mark.parametrize("nnodes,gpus,teacher_gpus,teacher_nodes,words", [
    (1, 8, 8, None, "leaves the student no GPU on a node with 8 GPU(s)"),
    (1, 2, None, None, "leaves the student no GPU on a node with 2 GPU(s)"),
    (1, 8, None, 1, "every node of this 1-node job"),
    (2, 8, None, 2, "Run on at least 3 nodes"),
    (1, 8, 0, None, "at least 1, not 0"),
    (1, 8, True, None, "not True"),
    (1, 8, "2", None, "not '2'"),
    (2, 8, None, -1, "at least 0, not -1"),
])
def test_splits_that_leave_the_student_nothing_are_refused(nnodes, gpus, teacher_gpus, teacher_nodes, words):
    with pytest.raises(lconfig.DistillTopologyError) as info:
        lconfig.split_distill_topology(nnodes=nnodes, gpus_per_node=gpus, teacher_gpus_per_node=teacher_gpus,
                                       teacher_nodes=teacher_nodes)
    assert words in str(info.value)


def test_finish_distillation_splits_the_job_and_fills_the_teacher_length(tmp_path):
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path))
    setup = lconfig.finish_distillation(cfg, nnodes=1, gpus_per_node=8)
    assert setup.topology == ("shared", 1, 6, 1, 2) and setup.topology_error is None
    assert (cfg["trainer"]["nnodes"], cfg["trainer"]["n_gpus_per_node"]) == (1, 6)
    assert (cfg["distillation"]["nnodes"], cfg["distillation"]["n_gpus_per_node"]) == (1, 2)
    need = PROMPT + RESPONSE + 1
    assert cfg["distillation"]["teacher_models"]["teacher_model"]["inference"]["max_model_len"] == need
    assert setup.max_model_len == {"teacher_model": need}
    assert setup.as_dict()["topology"]["student_gpus_per_node"] == 6

    # The orbit.distill keys choose the split; an explicit teacher length is kept as it is.
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path), {
        lconfig.TEACHER_GPUS_KEY: 4, "distillation.teacher_models.teacher_model.inference.max_model_len": 20000})
    setup = lconfig.finish_distillation(cfg, nnodes=2, gpus_per_node=8)
    assert setup.topology == ("shared", 2, 4, 2, 4) and setup.max_model_len == {}
    assert cfg["distillation"]["teacher_models"]["teacher_model"]["inference"]["max_model_len"] == 20000
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path), {lconfig.TEACHER_NODES_KEY: 1})
    assert lconfig.finish_distillation(cfg, nnodes=2, gpus_per_node=8).topology == ("dedicated", 1, 8, 1, 8)
    assert (cfg["trainer"]["nnodes"], cfg["distillation"]["n_gpus_per_node"]) == (1, 8)


def test_finish_distillation_does_nothing_when_distillation_is_off(tmp_path):
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path), {"distillation.enabled": False})
    assert lconfig.finish_distillation(cfg, nnodes=1, gpus_per_node=8) is None
    assert cfg["trainer"]["n_gpus_per_node"] == 8
    assert cfg["distillation"]["teacher_models"]["teacher_model"]["inference"]["max_model_len"] is None


def test_a_split_that_does_not_fit_is_returned_and_leaves_the_trainer_alone(tmp_path):
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path))
    setup = lconfig.finish_distillation(cfg, nnodes=1, gpus_per_node=1)
    assert setup.topology is None and "leaves the student no GPU" in setup.topology_error
    assert cfg["trainer"]["n_gpus_per_node"] == 8
    # The teacher length does not depend on the GPUs, so it is filled anyway.
    assert setup.max_model_len == {"teacher_model": PROMPT + RESPONSE + 1}


@pytest.mark.parametrize("teacher_gpus,student_gpus,nearest", [(1, 7, 133), (3, 5, 130)])
def test_a_split_whose_ranks_do_not_divide_the_batch_is_refused(teacher_gpus, student_gpus, nearest, tmp_path):
    # 132 prompts x 4 answers = 528, which divides among neither 7 nor 5 student GPUs. verl would
    # crash at the first update, so the split is refused with the nearest batch size that divides.
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path), {lconfig.TEACHER_GPUS_KEY: teacher_gpus})
    setup = lconfig.finish_distillation(cfg, nnodes=1, gpus_per_node=8)
    assert setup.topology is None
    message = setup.topology_error
    assert f"student 1 node(s) x {student_gpus} GPUs" in message and "= 528 answers" in message
    assert f"Set data.train_batch_size={nearest}, whose {nearest * 4} answers divide among {student_gpus}" in message
    assert (nearest * 4) % student_gpus == 0
    assert cfg["trainer"]["n_gpus_per_node"] == 8 and cfg["distillation"]["n_gpus_per_node"] == 8
    # The teacher length does not depend on the split, so it is filled anyway.
    assert setup.max_model_len == {"teacher_model": PROMPT + RESPONSE + 1}


def test_the_batch_check_counts_data_parallel_ranks_and_names_the_mini_batch_when_it_is_separate(tmp_path):
    student, teacher = _student_dir(tmp_path), _teacher_dir(tmp_path)
    # 5 prompts x 3 answers = 15 do not divide among 6 GPUs, but they do among 3 ranks of a
    # sequence parallel size of 2. Both the current key and the older key on the actor count.
    small = {"data.train_batch_size": 5, "actor_rollout_ref.actor.ppo_mini_batch_size": 5,
             "actor_rollout_ref.rollout.n": 3}
    for key in ("actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size",
                "actor_rollout_ref.actor.ulysses_sequence_parallel_size"):
        setup = lconfig.finish_distillation(_cfg(student, teacher, {**small, key: 2}), nnodes=1, gpus_per_node=8)
        assert setup.topology == ("shared", 1, 6, 1, 2) and setup.topology_error is None, key
    # Without sequence parallelism, 4 and 6 are equally near; the larger one is named.
    setup = lconfig.finish_distillation(_cfg(student, teacher, small), nnodes=1, gpus_per_node=8)
    assert "Set data.train_batch_size=6," in setup.topology_error
    # A mini-batch smaller than the batch is the value to change, so it is the one named.
    split = {**small, "data.train_batch_size": 10}
    setup = lconfig.finish_distillation(_cfg(student, teacher, split), nnodes=1, gpus_per_node=8)
    assert "Set actor_rollout_ref.actor.ppo_mini_batch_size=6," in setup.topology_error
    # A sequence parallel size that does not divide the student's GPUs leaves no whole rank count.
    bad_sp = {"actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size": 4}
    setup = lconfig.finish_distillation(_cfg(student, teacher, bad_sp), nnodes=1, gpus_per_node=8)
    assert "sequence parallel size is 4, which does not divide them" in setup.topology_error


@pytest.mark.parametrize("override", ["distillation.n_gpus_per_node=4", "+distillation.nnodes=1"])
def test_overrides_of_the_teacher_pool_are_refused_with_the_right_key(override, tmp_path):
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path))
    with pytest.raises(lconfig.DistillTopologyError) as info:
        lconfig.finish_distillation(cfg, nnodes=1, gpus_per_node=8, user_overrides=["x=1", override])
    assert lconfig.TEACHER_GPUS_KEY in str(info.value) and lconfig.TEACHER_NODES_KEY in str(info.value)
    # The same override on a run without distillation changes nothing the launcher owns.
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path), {"distillation.enabled": False})
    assert lconfig.finish_distillation(cfg, nnodes=1, gpus_per_node=8, user_overrides=[override]) is None


# --------------------------------------------------------------------------- pre-flight checks


def test_a_qwen3_teacher_passes_every_check(tmp_path):
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path))
    assert preflight.run_preflight(cfg, verl_check=False) == ([], [])


def test_the_teacher_entries_are_the_ones_verl_runs(tmp_path):
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path))
    assert list(preflight.teacher_entries(cfg)) == ["teacher_model"]
    # A second entry makes verl drop teacher_model, so its unset path is not reported.
    for name in ("think", "no_think"):
        _set(cfg, f"distillation.teacher_models.{name}", {
            "key": name, "model_path": str(_teacher_dir(tmp_path)),
            "inference": {"prompt_length": PROMPT, "response_length": RESPONSE, "max_model_len": None}})
    _set(cfg, "distillation.teacher_models.teacher_model.model_path", None)
    assert list(preflight.teacher_entries(cfg)) == ["think", "no_think"]
    assert preflight.run_preflight(cfg, verl_check=False) == ([], [])
    assert preflight.check_teacher_paths({"distillation": {"enabled": True, "teacher_models": {}}})[0].kind == "distill"


@pytest.mark.parametrize("vocab_config", [{"vocab_size": 248320}, {"text_config": {"vocab_size": 248320}}])
def test_a_teacher_with_another_vocabulary_is_refused(vocab_config, tmp_path):
    teacher = _teacher_dir(tmp_path)
    (teacher / "config.json").write_text(json.dumps({"max_position_embeddings": 262144, **vocab_config}))
    problems, _ = preflight.run_preflight(_cfg(_student_dir(tmp_path), teacher), verl_check=False)
    assert len(problems) == 1 and "vocab_size 248320" in problems[0].message
    assert "Qwen3-family teacher" in problems[0].message


def test_a_teacher_whose_ids_mean_other_tokens_is_refused(tmp_path):
    teacher = _teacher_dir(tmp_path, changes={1000: "Ġother", 151000: "Ġthing"})
    problems, _ = preflight.run_preflight(_cfg(_student_dir(tmp_path), teacher), verl_check=False)
    assert [p.kind for p in problems] == ["distill"]
    assert "disagree on 2 of the 151669 Qwen3 ids" in problems[0].message
    assert "id 1000: student 'tok1000', teacher 'Ġother'" in problems[0].message


def test_a_teacher_without_the_qwen3_special_ids_is_refused(tmp_path):
    teacher = _teacher_dir(tmp_path, changes={151667: "<|vision_pad|>"})
    problems, _ = preflight.check_teacher_tokenizers(_cfg(_student_dir(tmp_path), teacher))
    text = _messages(problems)
    assert "<think> should be id 151667, which is '<|vision_pad|>'" in text
    assert "disagree on 1 of the" in text


def test_the_students_extra_code_ids_and_a_missing_student_dir_are_not_problems(tmp_path):
    teacher = _teacher_dir(tmp_path)
    # The student has 151681 ids and the teacher 151669; only the shared ones are compared.
    assert preflight.check_teacher_tokenizers(_cfg(_student_dir(tmp_path), teacher)) == ([], [])
    # An unprepared student dir is the model checks' business; the teacher is still checked alone.
    bad = _teacher_dir(tmp_path, name="other", vocab_size=248320)
    problems, _ = preflight.check_teacher_tokenizers(_cfg(tmp_path / "not_built", bad))
    assert len(problems) == 1 and "vocab_size 248320" in problems[0].message
    # A student dir without a tokenizer.json cannot be compared, which is a problem.
    (tmp_path / "no_tokenizer").mkdir()
    problems, _ = preflight.check_teacher_tokenizers(_cfg(tmp_path / "no_tokenizer", teacher))
    assert len(problems) == 1 and "Cannot read the student's tokenizer" in problems[0].message


def test_other_merges_only_warn(tmp_path):
    teacher = _teacher_dir(tmp_path, merges=("a b", "b c"))
    problems, warnings = preflight.run_preflight(_cfg(_student_dir(tmp_path), teacher), verl_check=False)
    assert problems == [] and len(warnings) == 1 and "other BPE merges" in warnings[0]
    # Pairs and space-joined strings are the same merges.
    pairs = _teacher_dir(tmp_path, name="pairs", merges=(["a", "b"], ["ab", "c"]))
    assert preflight.check_teacher_tokenizers(_cfg(_student_dir(tmp_path), pairs)) == ([], [])


def test_hub_snapshot_finds_the_downloaded_snapshot(tmp_path):
    hub = tmp_path / "hub"
    repo = hub / "models--Qwen--Qwen3-8B"
    (repo / "snapshots" / "aaa").mkdir(parents=True)
    assert preflight.hub_snapshot("Qwen/Qwen3-8B", [str(hub)]) == str(repo / "snapshots" / "aaa")
    (repo / "snapshots" / "bbb").mkdir()
    assert preflight.hub_snapshot("Qwen/Qwen3-8B", [str(hub)]) is None   # two snapshots are ambiguous
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("bbb\n")
    assert preflight.hub_snapshot("Qwen/Qwen3-8B", [str(hub)]) == str(repo / "snapshots" / "bbb")
    assert preflight.hub_snapshot("/abs/Qwen3-8B", [str(hub)]) is None
    assert preflight.hub_snapshot("Qwen/Qwen3-14B", [str(hub)]) is None


def test_teacher_paths_must_be_absolute_local_dirs(tmp_path, monkeypatch):
    student = _student_dir(tmp_path)
    key = "distillation.teacher_models.teacher_model.model_path"
    hub = tmp_path / "hub"
    snapshot = hub / "models--Qwen--Qwen3-8B" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel_teacher").mkdir()
    cases = {
        None: f"{key}=/path/to/the/teacher",
        "???": f"{key}=/path/to/the/teacher",
        "Qwen/Qwen3-8B": f"{key}={snapshot}",
        "rel_teacher": f"{key}={tmp_path / 'rel_teacher'}",
        str(tmp_path / "gone"): f"{key}=/absolute/path/to/the/teacher",
    }
    for path, fix in cases.items():
        problems = preflight.check_teacher_paths(_cfg(student, path), hub_dirs=[str(hub)])
        assert len(problems) == 1 and problems[0].commands == (fix,), (path, problems)
    assert "under the runs root" in preflight.check_teacher_paths(_cfg(student, "Qwen/Qwen3-8B"), [str(hub)])[0].message


@pytest.mark.parametrize("value,ok", [(1.0, True), (1, True), (0.9, False), (0.6, False), (None, False)])
def test_the_student_must_sample_at_temperature_one(value, ok, tmp_path):
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path), {"actor_rollout_ref.rollout.temperature": value})
    problems = preflight.check_temperature(cfg)
    assert (problems == []) is ok
    if not ok:
        assert problems[0].commands == ("actor_rollout_ref.rollout.temperature=1.0",)
        assert "always scores at temperature 1.0" in problems[0].message


@pytest.mark.parametrize("mode", ["rltt_exit", "rltt_last", "rltt_progressive"])
def test_rltt_loss_modes_are_refused_under_distillation(mode, tmp_path):
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path),
               {"actor_rollout_ref.actor.policy_loss.loss_mode": mode})
    problems = preflight.check_loss_mode(cfg)
    assert len(problems) == 1 and "distillation_ppo_loss" in problems[0].message
    assert problems[0].commands == ("actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",)


def test_the_kl_loss_only_warns_when_the_task_loss_is_on(tmp_path):
    # verl's own default for use_task_rewards is true, so a config without the key counts as on.
    for task_rewards in ({}, {"distillation.distillation_loss.use_task_rewards": True}):
        cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path),
                   {"actor_rollout_ref.actor.use_kl_loss": True, **task_rewards})
        problems, warnings = preflight.run_preflight(cfg, verl_check=False)
        assert problems == [] and len(warnings) == 1 and "use_kl_loss is true" in warnings[0]


def test_the_kl_loss_is_refused_when_the_task_loss_is_off(tmp_path):
    # verl zeroes the whole task policy loss, the KL term included, when the task rewards are off.
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path), {
        "actor_rollout_ref.actor.use_kl_loss": True, "distillation.distillation_loss.use_task_rewards": False})
    problems, warnings = preflight.run_preflight(cfg, verl_check=False)
    assert warnings == [] and [p.kind for p in problems] == ["distill"]
    assert problems[0].commands == ("distillation.distillation_loss.use_task_rewards=true",)
    assert "GRPO advantage is 0 and only the KL is added" in problems[0].message
    # Without the KL, the task loss may stay off, as in distill_sft.
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path),
               {"distillation.distillation_loss.use_task_rewards": False})
    assert preflight.run_preflight(cfg, verl_check=False) == ([], [])


def _two_teachers(tmp_path: Path, keys: dict[str, str], changes: dict | None = None) -> dict:
    """A distill-builder config with one teacher entry per item of ``keys`` (entry name to its key)."""
    cfg = _cfg(_student_dir(tmp_path), None, {"orbit.prepare.data.builder": "distill", **(changes or {})})
    for name, key in keys.items():
        _set(cfg, f"distillation.teacher_models.{name}", {
            "key": key, "model_path": str(_teacher_dir(tmp_path)),
            "inference": {"prompt_length": PROMPT, "response_length": RESPONSE, "max_model_len": None}})
    return cfg


def test_two_teachers_of_a_distill_dataset_are_routed_by_the_thinking_mode(tmp_path):
    routed = {"distillation.teacher_key": "teacher_route"}
    cfg = _two_teachers(tmp_path, {"think": "think", "no_think": "no_think"}, routed)
    assert preflight.run_preflight(cfg, verl_check=False) == ([], [])

    # verl's default key, data_source, holds orbit/none and the like, which match no teacher.
    problems = preflight.check_teacher_routing(_two_teachers(tmp_path, {"think": "think", "no_think": "no_think"}))
    assert [p.kind for p in problems] == ["distill"]
    assert problems[0].commands == ("distillation.teacher_key=teacher_route",)

    # Keys that are not the two modes: entries named after a mode get it, the others the rest in order.
    problems = preflight.check_teacher_routing(_two_teachers(tmp_path, {"no_think": "nt", "big": "big"}, routed))
    assert len(problems) == 1 and "['think', 'no_think']" in problems[0].message
    assert problems[0].commands == ("distillation.teacher_models.no_think.key=no_think",
                                    "distillation.teacher_models.big.key=think")
    problems = preflight.check_teacher_routing(_two_teachers(tmp_path, {"a": "think", "b": "think"}, routed))
    assert problems[0].commands == ("distillation.teacher_models.a.key=think",
                                    "distillation.teacher_models.b.key=no_think")
    assert "Swap the values below" in problems[0].message


def test_teacher_routing_is_checked_only_for_several_teachers_and_the_distill_builder(tmp_path):
    keys = {"think": "a", "no_think": "b"}
    # prepare_data datasets have no teacher_route column; prepare.py refuses routing by it there.
    other = _two_teachers(tmp_path, keys, {"orbit.prepare.data.builder": None})
    assert preflight.check_teacher_routing(other) == []
    # A prompt mode of "mixed" selects the distill builder when no builder is named.
    mixed = _two_teachers(tmp_path, keys, {"orbit.prepare.data.builder": None,
                                           "orbit.prepare.data.prompt_mode": "mixed"})
    assert len(preflight.check_teacher_routing(mixed)) == 2
    # One teacher gets every row, whatever the key says.
    single = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path), {"orbit.prepare.data.builder": "distill"})
    assert preflight.check_teacher_routing(single) == []


def test_teacher_lengths_that_verl_or_vllm_would_refuse(tmp_path):
    student, need = _student_dir(tmp_path), PROMPT + RESPONSE + 1
    prefix = "distillation.teacher_models.teacher_model.inference"
    short = _cfg(student, _teacher_dir(tmp_path), {f"{prefix}.max_model_len": 4096})
    problems = preflight.check_teacher_lengths(short)
    assert len(problems) == 1 and problems[0].commands == (f"{prefix}.max_model_len={need}",)
    # A teacher trained at 8K cannot read a 1K prompt with an 8K response.
    small = _cfg(student, _teacher_dir(tmp_path, name="small", max_positions=8192))
    problems = preflight.check_teacher_lengths(small)
    assert len(problems) == 1 and "max_position_embeddings 8192" in problems[0].message
    assert problems[0].commands == (f"data.max_response_length={8192 - PROMPT - 1}",)
    # An entry that does not copy the student's lengths would get RolloutConfig's 512 and 512.
    copied = _cfg(student, _teacher_dir(tmp_path), {f"{prefix}.prompt_length": 512})
    assert "the student's rollout uses 1024 and 8192" in _messages(preflight.check_teacher_lengths(copied))
    bare = _cfg(student, _teacher_dir(tmp_path), {"distillation.teacher_models.teacher_model.inference": None})
    assert "inference is missing" in _messages(preflight.check_teacher_lengths(bare))


def test_a_thinking_budget_must_leave_room_for_the_answer(tmp_path):
    student, teacher = _student_dir(tmp_path), _teacher_dir(tmp_path)
    # No budget is the default. A null one reaches the agent loop as an empty string, which is no budget too.
    assert preflight.check_think_budget(_cfg(student, teacher)) == []
    assert preflight.check_think_budget(_cfg(student, teacher, {THINK_BUDGET_KEY: None})) == []
    # The largest budget leaves 64 ids for the stop-thinking text and 256 tokens for the answer. The
    # YAML may hold it as a string, as distill_rl_sft_16k does, or a hydra override may give a number.
    largest = RESPONSE - 64 - 256
    for value in (largest, str(largest)):
        assert preflight.check_think_budget(_cfg(student, teacher, {THINK_BUDGET_KEY: value})) == []
    problems = preflight.check_think_budget(_cfg(student, teacher, {THINK_BUDGET_KEY: str(largest + 1)}))
    assert [p.kind for p in problems] == ["config"]
    assert problems[0].commands == (f"{THINK_BUDGET_KEY}={largest}",)
    assert f"leaves {RESPONSE - largest - 1} of the {RESPONSE} tokens" in problems[0].message
    # The launcher runs the check for every config (cli.run_checks), so the distillation pre-flight
    # leaves it out and does not report it twice.
    problems, _ = preflight.run_preflight(_cfg(student, teacher, {THINK_BUDGET_KEY: str(largest + 1)}),
                                          verl_check=False)
    assert all(THINK_BUDGET_KEY not in " ".join(p.commands) for p in problems)


def test_the_thinking_budget_is_checked_against_the_rollouts_response_length(tmp_path):
    # The agent loop reads the rollout's response length, which only defaults to data.max_response_length.
    student, teacher = tmp_path / "student", tmp_path / "teacher"
    short = _cfg(student, teacher, {THINK_BUDGET_KEY: "7000", "actor_rollout_ref.rollout.response_length": 4096})
    assert [p.commands for p in preflight.check_think_budget(short)] == [(f"{THINK_BUDGET_KEY}={4096 - 64 - 256}",)]
    unset = _cfg(student, teacher, {THINK_BUDGET_KEY: "8000", "actor_rollout_ref.rollout.response_length": None})
    assert [p.commands for p in preflight.check_think_budget(unset)] == [(f"{THINK_BUDGET_KEY}={RESPONSE - 64 - 256}",)]
    # A response length below the stop-thinking ids and the answer leaves no budget, so the fix turns it off.
    tiny = _cfg(student, teacher, {THINK_BUDGET_KEY: "100", "actor_rollout_ref.rollout.response_length": 300})
    assert [p.commands for p in preflight.check_think_budget(tiny)] == [(f"{THINK_BUDGET_KEY}=null",)]


@pytest.mark.parametrize("value", ["14k", "1.5", "0", -1, True])
def test_a_thinking_budget_that_is_not_a_whole_number_of_tokens_is_refused(value, tmp_path):
    cfg = _cfg(tmp_path / "student", tmp_path / "teacher", {THINK_BUDGET_KEY: value})
    problems = preflight.check_think_budget(cfg)
    assert len(problems) == 1 and "not a positive whole number of tokens" in problems[0].message


def test_the_answer_room_is_the_agent_loops():
    # The agent loop imports verl, so its constant is read from the source file instead of importing it.
    source = Path(preflight.__file__).resolve().parents[1] / "data" / "agent_loop.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    values = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "MIN_ANSWER_TOKENS" for target in node.targets)]
    assert values == [preflight.MIN_ANSWER_TOKENS]


def test_the_teacher_prompt_fix_is_asked_for(tmp_path):
    cfg = _cfg(_student_dir(tmp_path), _teacher_dir(tmp_path), {teacher_mod.MANAGER_CLASS_KEY: None})
    problems, _ = preflight.run_preflight(cfg, verl_check=False)
    assert len(problems) == 1 and teacher_mod.MANAGER_CLASS_FQN in problems[0].message


def test_the_preflight_module_imports_without_verl_or_omegaconf():
    # A light CPU-only python may have neither, and the checks must still run in it.
    code = ("import sys; import orbit_verl.distill.preflight, orbit_verl.launch.config; "
            "print(sorted(m for m in ('verl', 'omegaconf', 'torch') if m in sys.modules))")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    assert out.strip() == "[]"


# --------------------------------------------------------------------------- composed configs and the launcher


def _distill_overrides(teacher: Path, *extra: str) -> list[str]:
    """The overrides that turn the single-mode RL config into a distillation run with the planned recipe.

    The batch of 6 prompts with 4 answers each gives 24 answers, which divide among the 6 and the 8
    student GPUs of the layouts below, as verl demands. Responses of 512 tokens keep the teacher's
    context at 2048 + 512 + 1 tokens.
    """
    return [*SINGLE_MODE, *NO_PLUGIN, "distillation.enabled=true", "data.train_batch_size=6",
            "actor_rollout_ref.rollout.n=4", "data.max_response_length=512",
            f"distillation.teacher_models.teacher_model.model_path={teacher}",
            "distillation.distillation_loss.loss_mode=k1",
            "distillation.distillation_loss.use_policy_gradient=true",
            "distillation.distillation_loss.use_task_rewards=false",
            "actor_rollout_ref.rollout.temperature=1.0",
            "actor_rollout_ref.actor.use_kl_loss=false",
            f"++{teacher_mod.MANAGER_CLASS_KEY}={teacher_mod.MANAGER_CLASS_FQN}", *extra]


def _prepared_release_model(tmp_path: Path) -> tuple[Path, Path]:
    """The single-mode prepared dirs for a release model, with a student tokenizer.json of the real shape."""
    runs, export, model = _smoke_prepared(tmp_path)
    model_dir = runs / "models" / model
    _write_tokenizer(model_dir, extra=STUDENT_EXTRA)
    # The dataset records the tokenizer it was filtered with, so it has to follow the new file.
    manifest_path = runs / "data" / DATA_NAME / common.DATA_MANIFEST
    manifest = common.read_manifest(manifest_path, common.DATA_MANIFEST_KIND)
    common.write_manifest(manifest_path, common.DATA_MANIFEST_KIND,
                          {**manifest, "tokenizer_sha256": common.tokenizer_sha256(model_dir)})
    return runs, export


def _launch(runs: Path, export: Path, teacher: Path, *extra: str, mode: str = "--dry-run",
            gpus: int | None = 8) -> list[str]:
    argv = ["--config", CONFIG, mode, "--runs-root", str(runs), "--export", str(export)]
    if gpus is not None:
        argv += ["--gpus-per-node", str(gpus)]
    return [*argv, *_distill_overrides(teacher, *extra)]


@needs_verl
def test_a_composed_config_gets_the_split_that_verl_accepts(tmp_path):
    from verl.utils.config import omega_conf_to_dataclass

    from orbit_verl.launch.train import validate

    teacher = _teacher_dir(tmp_path)
    export = _export(tmp_path, "qwen3")
    user = _distill_overrides(teacher)
    cfg = lconfig.compose_config(CONFIG, [*user, *lconfig.model_overrides(str(export), None, user)],
                                 repo_root=tmp_path, mode=lconfig.mode_overrides(nnodes=1, gpus_per_node=8))
    lconfig.finish_config(cfg, CONFIG, runs_root=str(tmp_path / "runs_root"), environ={})
    setup = lconfig.finish_distillation(cfg, nnodes=1, gpus_per_node=8, user_overrides=user)
    assert setup.topology == ("shared", 1, 6, 1, 2)
    assert (cfg.trainer.nnodes, cfg.trainer.n_gpus_per_node) == (1, 6)
    assert (cfg.distillation.nnodes, cfg.distillation.n_gpus_per_node) == (1, 2)
    need = cfg.data.max_prompt_length + cfg.data.max_response_length + 1
    assert cfg.distillation.teacher_models.teacher_model.inference.max_model_len == need == 2048 + 512 + 1
    # verl's own reading of the block: one teacher filling the 2-GPU pool, sized for a whole sequence.
    distill = omega_conf_to_dataclass(cfg.distillation)
    (teacher_cfg,) = distill.teacher_models.values()
    assert teacher_cfg.num_replicas * teacher_cfg.per_replica_world_size == 2
    assert (teacher_cfg.inference.prompt_length, teacher_cfg.inference.response_length) == (2048 + 512, 1)
    assert teacher_cfg.inference.max_model_len == need
    validate(cfg)
    assert preflight.run_preflight(cfg) == ([], [])


@needs_verl
def test_dry_run_prints_the_split_and_passes(tmp_path, process_state, no_ray, capsys):
    runs, export = _prepared_release_model(tmp_path)
    rc = cli.main(_launch(runs, export, _teacher_dir(tmp_path)))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "student 1 node(s) x 6 GPUs, teacher 1 node(s) x 2 GPUs (shared nodes)" in out
    assert "  trainer.n_gpus_per_node=6\n" in out and "  distillation.n_gpus_per_node=2\n" in out
    assert "distillation.teacher_models.teacher_model.inference.max_model_len=2561 (was null" in out
    assert "checks:      all passed" in out


@needs_verl
def test_dry_run_on_one_gpu_reports_the_split(tmp_path, process_state, no_ray, capsys):
    runs, export = _prepared_release_model(tmp_path)
    rc = cli.main(_launch(runs, export, _teacher_dir(tmp_path), gpus=1))
    out = capsys.readouterr().out
    assert rc == 2 and "PROBLEM (gpus)" in out and "leaves the student no GPU on a node with 1 GPU(s)" in out


@needs_verl
def test_prepare_only_leaves_the_split_to_the_training_launch(tmp_path, process_state, no_ray, monkeypatch, capsys):
    # A preparation may run on a machine whose GPU count says nothing about the training run's.
    runs, export = _prepared_release_model(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rc = cli.main(_launch(runs, export, _teacher_dir(tmp_path), mode="--prepare-only", gpus=None))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "PROBLEM (gpus)" not in out


@needs_verl
def test_dry_run_refuses_a_teacher_of_another_family_and_the_rl_settings(tmp_path, process_state, no_ray, capsys):
    runs, export = _prepared_release_model(tmp_path)
    teacher = _teacher_dir(tmp_path, name="Qwen3.5-9B", vocab_size=248320)
    rc = cli.main(_launch(runs, export, teacher, "actor_rollout_ref.rollout.temperature=0.9",
                          "actor_rollout_ref.actor.use_kl_loss=true"))
    out = capsys.readouterr().out
    assert rc == 2
    assert "PROBLEM (distill)" in out and "vocab_size 248320" in out
    assert "actor_rollout_ref.rollout.temperature is 0.9" in out
    assert "    actor_rollout_ref.rollout.temperature=1.0" in out
    # The task loss is off in these overrides, so the KL would never train and is refused.
    assert "actor_rollout_ref.actor.use_kl_loss is true, but distillation.distillation_loss.use_task_rewards" in out
    assert "    distillation.distillation_loss.use_task_rewards=true" in out


@needs_verl
def test_dry_run_refuses_a_batch_that_the_students_gpus_do_not_divide(tmp_path, process_state, no_ray, capsys):
    runs, export = _prepared_release_model(tmp_path)
    rc = cli.main(_launch(runs, export, _teacher_dir(tmp_path), f"+{lconfig.TEACHER_GPUS_KEY}=3"))
    out = capsys.readouterr().out
    # 6 prompts x 4 answers do not divide among the 5 student GPUs; 5 prompts make 20 answers, which do.
    assert rc == 2 and "PROBLEM (gpus)" in out and "Set data.train_batch_size=5," in out


@needs_verl
def test_dry_run_refuses_an_override_of_the_teacher_pool(tmp_path, process_state, no_ray, capsys):
    runs, export = _prepared_release_model(tmp_path)
    rc = cli.main(_launch(runs, export, _teacher_dir(tmp_path), "distillation.n_gpus_per_node=4"))
    out = capsys.readouterr().out
    assert rc == 2 and lconfig.TEACHER_GPUS_KEY in out


@needs_verl
def test_head_plan_gives_the_teacher_a_whole_node(tmp_path, process_state, no_ray, monkeypatch, capsys):
    runs, export = _prepared_release_model(tmp_path)
    _set_env(monkeypatch, WORLD_SIZE="2", RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
    rc = cli.main(_launch(runs, export, _teacher_dir(tmp_path), f"+{lconfig.TEACHER_NODES_KEY}=1"))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "mode:        head" in out
    # Ray still waits for every node and GPU of the job; verl splits them into the two pools.
    assert "for 2 nodes / 16 GPUs" in out
    assert "student 1 node(s) x 8 GPUs, teacher 1 node(s) x 8 GPUs (dedicated nodes)" in out
    assert "  trainer.nnodes=1\n" in out and "  distillation.nnodes=1\n" in out
