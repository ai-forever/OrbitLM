"""CPU tests of resuming a run: which checkpoints a launch may continue (``check_resume``) and whether
it validates again before its first step (``resume_validation``).

Run with the verl python and ``PYTHONPATH=<repo>/src``, like test_launch.py. The checkpoints and the
metrics files are written by hand in a temporary runs root; nothing starts Ray or touches a GPU.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from orbit_verl.launch import cli
from orbit_verl.launch import config as lconfig
from orbit_verl.launch import prepare as lprepare
# process_state and no_ray are fixtures of test_launch; importing them makes pytest offer them here too.
from test_launch import (CONFIG, NO_PLUGIN, SINGLE_MODE, _compose, _dry_run, _export, _smoke_prepared,  # noqa: F401
                         no_ray, process_state)

pytestmark = pytest.mark.skipif(importlib.util.find_spec("verl") is None,
                                reason="composing a config needs verl's own config files")

VAL_KEY = "val-core/orbit/math#think/acc/mean@1"


def _validating(tmp_path: Path, *overrides: str):
    """The RL recipe with a validation before training and every 10 steps, as it has them."""
    return _compose(CONFIG, tmp_path, *SINGLE_MODE, "trainer.val_before_train=true", "trainer.test_freq=10",
                    *overrides, export=_export(tmp_path))


def _checkpoint(cfg, step: int, *, tracked: bool = True) -> Path:
    """The checkpoint dir of ``step`` and, when ``tracked``, the tracker file that names it, as verl writes them."""
    ckpt = Path(cfg.trainer.default_local_dir)
    (ckpt / f"global_step_{step}" / "actor").mkdir(parents=True)
    if tracked:
        (ckpt / lprepare.CHECKPOINT_TRACKER).write_text(str(step))
    return ckpt


def _metrics(path: Path, rows: list[tuple[int, dict]], tail: str = "") -> None:
    """A metrics file in the format of verl's file logger, optionally ending in a broken line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"step": step, "data": data}) + "\n" for step, data in rows) + tail)


def test_a_run_from_scratch_validates_first(tmp_path):
    cfg = _validating(tmp_path)
    assert lprepare.resumed_step(cfg) is None
    assert lprepare.resume_validation(cfg) is None and cfg.trainer.val_before_train is True


def test_a_resume_whose_validation_is_logged_skips_it(tmp_path):
    # The first smoke launch logged step 0, trained steps 1-2, saved step 2 and validated it. The launcher
    # renamed its metrics file when the second launch started, so the validation is in the renamed file.
    cfg = _validating(tmp_path)
    _checkpoint(cfg, 2)
    run_dir = Path(cfg.orbit.run_dir)
    _metrics(run_dir / "metrics.20260101-120000.jsonl",
             [(0, {VAL_KEY: 0.8}), (1, {"actor/pg_loss": 0.1}), (2, {"actor/pg_loss": 0.1, VAL_KEY: 0.81})])
    note = lprepare.resume_validation(cfg)
    assert cfg.trainer.val_before_train is False
    assert note == "resuming from global_step_2, whose validation is already logged: trainer.val_before_train=false"


def test_a_checkpoint_saved_before_its_validation_was_logged_is_validated_again(tmp_path):
    # verl saves step 10 before it validates step 10 and logs the step only after the validation. A job
    # stopped during that validation leaves the checkpoint with no metrics line for step 10, and a half
    # written line may end the file.
    cfg = _validating(tmp_path)
    _checkpoint(cfg, 10)
    rows = [(0, {VAL_KEY: 0.8})] + [(step, {"actor/pg_loss": 0.1}) for step in range(1, 10)]
    _metrics(Path(cfg.orbit.run_dir) / "metrics.jsonl", rows, tail='{"step": 10, "data": {"val-co')
    note = lprepare.resume_validation(cfg)
    assert cfg.trainer.val_before_train is True
    assert note == ("resuming from global_step_10, whose validation is not logged: it is validated before "
                    "the first step")


def test_a_checkpoint_without_any_logged_validation_is_validated(tmp_path):
    # Step 15 is saved (save_freq 5) but is no validation step (test_freq 10). Only a logged validation of
    # the resumed step lets the launcher skip it, so this resume validates step 15.
    cfg = _validating(tmp_path)
    _checkpoint(cfg, 15)
    _metrics(Path(cfg.orbit.run_dir) / "metrics.jsonl",
             [(10, {VAL_KEY: 0.8}), *((step, {"actor/pg_loss": 0.1}) for step in range(11, 16))])
    assert lprepare.validated_steps(str(cfg.orbit.run_dir)) == {10}
    assert lprepare.resume_validation(cfg) is not None and cfg.trainer.val_before_train is True


def test_a_config_without_validation_before_training_is_left_alone(tmp_path):
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "trainer.val_before_train=false", export=_export(tmp_path))
    assert cfg.trainer.val_before_train is False
    _checkpoint(cfg, 10)
    assert lprepare.resume_validation(cfg) is None and cfg.trainer.val_before_train is False


@pytest.mark.parametrize("overrides,tracked", [
    (("trainer.resume_mode=disable",), True),
    ((), False),
])
def test_runs_that_verl_starts_from_scratch_validate_first(overrides, tracked, tmp_path):
    # resume_mode disable never loads a checkpoint, and without the tracker file verl starts from scratch.
    cfg = _validating(tmp_path, *overrides)
    _checkpoint(cfg, 10, tracked=tracked)
    _metrics(Path(cfg.orbit.run_dir) / "metrics.jsonl", [(10, {VAL_KEY: 0.8})])
    assert lprepare.resumed_step(cfg) is None
    assert lprepare.resume_validation(cfg) is None and cfg.trainer.val_before_train is True


def test_a_tracker_that_names_a_missing_checkpoint_is_a_start_from_scratch(tmp_path):
    cfg = _validating(tmp_path)
    ckpt = _checkpoint(cfg, 10)
    (ckpt / lprepare.CHECKPOINT_TRACKER).write_text("20")
    assert lprepare.resumed_step(cfg) is None


def test_resume_path_takes_the_step_from_the_path(tmp_path):
    cfg = _validating(tmp_path, "trainer.resume_mode=resume_path",
                      "trainer.resume_from_path=/runs/x/ckpt/global_step_20")
    assert lprepare.resumed_step(cfg) == 20
    _metrics(Path(cfg.orbit.run_dir) / "metrics.jsonl", [(20, {VAL_KEY: 0.8})])
    assert lprepare.resume_validation(cfg) is not None and cfg.trainer.val_before_train is False


def test_the_dry_run_reports_the_resume_decision(tmp_path, process_state, no_ray, capsys):
    runs, export, model = _smoke_prepared(tmp_path)
    overrides = [*SINGLE_MODE, *NO_PLUGIN, "trainer.val_before_train=true", "trainer.resume_mode=auto"]
    cfg = _compose(CONFIG, tmp_path, *overrides, export=export, runs=runs)
    _checkpoint(cfg, 4)
    _metrics(Path(cfg.orbit.run_dir) / "metrics.jsonl", [(4, {VAL_KEY: 0.5})])
    rc = cli.main(_dry_run(CONFIG, export, runs, *overrides))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "resume:      resuming from global_step_4, whose validation is already logged" in out


# --------------------------------------------------------------------------- check_resume


def _saved_run(tmp_path: Path, *overrides: str):
    """An RL run with a checkpoint and the saved config of its launch."""
    old = _compose(CONFIG, tmp_path, *SINGLE_MODE, *overrides, export=_export(tmp_path))
    _checkpoint(old, 20)
    (Path(old.orbit.run_dir) / "config.yaml").write_text(lconfig.to_yaml(old))
    return old


def test_the_recipe_keys_are_compared():
    assert set(lprepare._RESUME_KEYS) >= {
        "actor_rollout_ref.actor.optim.lr", "actor_rollout_ref.actor.kl_loss_coef",
        "actor_rollout_ref.actor.loss_agg_mode", "algorithm.rollout_correction",
        "actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype", "data.train_batch_size",
        "data.max_response_length", "reward.custom_reward_function.reward_kwargs.truncated_score"}


@pytest.mark.parametrize("override,change", [
    ("actor_rollout_ref.actor.optim.lr=5e-7", "actor_rollout_ref.actor.optim.lr: 1e-06 -> 5e-07"),
    ("actor_rollout_ref.actor.kl_loss_coef=0.01", "actor_rollout_ref.actor.kl_loss_coef: 0.001 -> 0.01"),
    ("actor_rollout_ref.actor.loss_agg_mode=token-mean",
     "actor_rollout_ref.actor.loss_agg_mode: seq-mean-token-mean -> token-mean"),
    ("algorithm.rollout_correction.rollout_is=null", "algorithm.rollout_correction: {"),
    ("algorithm.rollout_correction.rollout_is_threshold=3.0", "algorithm.rollout_correction: {"),
    ("actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=auto",
     "actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype: fp8 -> auto"),
    # rl_sft_mix runs 64 prompts per step with responses of up to 16K tokens.
    ("data.train_batch_size=32", "data.train_batch_size: 64 -> 32"),
    ("data.max_response_length=1024", "data.max_response_length: 16384 -> 1024"),
    ("reward.custom_reward_function.reward_kwargs.truncated_score=null",
     "reward.custom_reward_function.reward_kwargs.truncated_score: 0.0 -> None"),
])
def test_a_resume_with_another_recipe_is_refused(override, change, tmp_path):
    _saved_run(tmp_path)
    new = _compose(CONFIG, tmp_path, *SINGLE_MODE, override, export=_export(tmp_path))
    problems = lprepare.check_resume(new)
    assert [p.kind for p in problems] == ["resume"] and change in problems[0].message


def test_the_same_recipe_written_differently_resumes(tmp_path):
    # 0 and 0.0 are one reward, and the schedule length and the validation before training are no recipe
    # keys: the second smoke launch raises total_training_steps, and the launcher may turn the validation off.
    _saved_run(tmp_path)
    new = _compose(CONFIG, tmp_path, *SINGLE_MODE, "reward.custom_reward_function.reward_kwargs.truncated_score=0",
                   "trainer.total_training_steps=141", "trainer.val_before_train=true", export=_export(tmp_path))
    assert lprepare.check_resume(new) == []


def test_keys_that_an_older_launch_did_not_save_are_not_compared(tmp_path):
    old = _saved_run(tmp_path)
    saved = Path(old.orbit.run_dir) / "config.yaml"
    from omegaconf import OmegaConf

    plain = OmegaConf.to_container(OmegaConf.load(saved))
    del plain["algorithm"]["rollout_correction"]
    del plain["data"]["train_batch_size"]
    saved.write_text(OmegaConf.to_yaml(OmegaConf.create(plain)))
    new = _compose(CONFIG, tmp_path, *SINGLE_MODE, "algorithm.rollout_correction.rollout_is=null",
                   "data.train_batch_size=32", export=_export(tmp_path))
    assert lprepare.check_resume(new) == []
