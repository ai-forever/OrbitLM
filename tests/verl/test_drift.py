"""CPU tests for the exit drift alarm (``orbit_verl.rltt.drift``) and its place in verl's trainer.

The alarm itself is checked on plain metric dicts. Its wiring is checked on verl's real classes
without starting Ray: the trainer subclass on a stand-in for verl's trainer, the registry swap on
verl's own trainer registry, and the task runner on verl's own ``TaskRunnerV1`` with its ``run``
replaced by a function that records which trainer class it would build.
"""
from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from orbit_verl.rltt import drift
from orbit_verl.rltt.drift import ALARM_METRIC, EXIT_STEP_METRIC, ExitDriftAlarm


def _alarm(threshold=0.3, path=None):
    lines: list[str] = []
    return ExitDriftAlarm(threshold, None if path is None else str(path), emit=lines.append), lines


def _observe(alarm, step, value):
    metrics = {EXIT_STEP_METRIC: value, "critic/score/mean": 0.5}
    fired = alarm.observe(metrics, step)
    return fired, metrics


def test_the_alarm_fires_only_after_a_drop_larger_than_the_threshold():
    # The values are exact in binary floating point, so a drop of exactly the threshold is exact too.
    alarm, lines = _alarm(0.25)
    expected = [
        (1, 1.75, False),   # the first step is the baseline
        (2, 2.0, False),    # a rise never fires
        (3, 1.625, False),  # a drop of 0.125
        (4, 1.5, False),    # a drop of exactly the threshold does not fire
        (5, 1.375, True),   # a drop of 0.375
        (6, 0.25, True),
        (7, 1.625, False),  # back within the threshold: the alarm clears
    ]
    for step, value, fires in expected:
        fired, metrics = _observe(alarm, step, value)
        assert fired is fires, (step, value)
        assert metrics[ALARM_METRIC] == int(fires)
        assert metrics["critic/score/mean"] == 0.5   # the step's other metrics are left alone
    # One WARNING line per step while the alarm holds, naming the numbers and the config key.
    assert len(lines) == 2
    assert lines[0].startswith("[orbit_verl] WARNING step 5: exit drift alarm.")
    assert "is 1.375, 0.375 below its value of 1.750 at step 1" in lines[0]
    assert "orbit.rltt.exit_drift_alarm=0.25" in lines[0] and "Training continues." in lines[0]
    assert lines[1].startswith("[orbit_verl] WARNING step 6:")


def test_a_step_without_the_exit_metric_gets_no_alarm():
    # Stock GRPO (loss_mode=vanilla) has no exit pdf, so there is nothing to compare.
    alarm, lines = _alarm()
    metrics = {"critic/score/mean": 0.5}
    assert alarm.observe(metrics, 1) is None
    assert ALARM_METRIC not in metrics and alarm.baseline is None and lines == []


def test_the_first_step_value_survives_a_resume(tmp_path):
    path = tmp_path / drift.BASELINE_FILE
    first, _ = _alarm(path=path)
    _observe(first, 1, 1.8)
    _observe(first, 2, 1.7)
    assert json.loads(path.read_text()) == {"metric": EXIT_STEP_METRIC, "step": 1, "value": 1.8}

    # A relaunch resumes at step 81, where the gate has already drifted: the saved step-1 value counts.
    resumed, lines = _alarm(path=path)
    fired, _ = _observe(resumed, 81, 1.2)
    assert fired is True and resumed.baseline == (1, 1.8)
    assert "at step 1" in lines[0]

    # A new run with the same run dir starts at step 1 again and replaces the saved value.
    fresh, _ = _alarm(path=path)
    assert _observe(fresh, 1, 1.2)[0] is False
    assert json.loads(path.read_text())["value"] == 1.2


def test_a_resume_without_a_saved_value_compares_with_its_own_first_step(tmp_path):
    path = tmp_path / drift.BASELINE_FILE
    alarm, lines = _alarm(path=path)
    assert _observe(alarm, 41, 1.0)[0] is False
    assert alarm.baseline == (41, 1.0)
    assert len(lines) == 1 and "no saved first-step value" in lines[0] and "step 41" in lines[0]
    # A broken file is treated the same way.
    path.write_text("{not json")
    alarm, lines = _alarm(path=path)
    assert _observe(alarm, 61, 0.9)[0] is False and alarm.baseline == (61, 0.9)


def test_a_baseline_that_cannot_be_saved_does_not_stop_training(tmp_path):
    alarm, lines = _alarm(path=tmp_path / "missing_dir" / drift.BASELINE_FILE)
    assert _observe(alarm, 1, 1.8)[0] is False
    assert _observe(alarm, 2, 1.0)[0] is True
    assert "cannot save the baseline" in lines[0] and lines[1].startswith("[orbit_verl] WARNING step 2:")


@pytest.mark.parametrize("bad", [0, -0.1, True, "0.3", None, float("nan"), float("inf")])
def test_bad_thresholds_are_refused(bad):
    with pytest.raises(ValueError, match="orbit.rltt.exit_drift_alarm"):
        drift.check_threshold(bad)
    assert drift.check_threshold(1) == 1.0


def _run_config(tmp_path, threshold=0.3):
    return OmegaConf.create({
        "orbit": {"run_dir": str(tmp_path), "rltt": {"exit_drift_alarm": threshold}},
        "trainer": {"v1": {"trainer_mode": "sync"}},
    })


class _StockTrainer:
    """Stands in for verl's trainer: ``step`` puts the actor's metrics into the dict it is given."""

    def __init__(self, config):
        self.config = config
        self.global_steps = 0
        self.exit_steps = iter([1.8, 1.7, 1.2])

    def step(self, metrics, timing_raw):
        metrics[EXIT_STEP_METRIC] = next(self.exit_steps)
        return "batch"

    def fit(self):
        logged = []
        for step in (1, 2, 3):
            self.global_steps = step
            metrics = {}
            assert self.step(metrics, {}) == "batch"
            logged.append(metrics)
        return logged


def test_the_trainer_subclass_runs_the_alarm_after_every_step(tmp_path, capsys):
    trainer_cls = drift.with_exit_drift_alarm(_StockTrainer)
    assert issubclass(trainer_cls, _StockTrainer)
    assert trainer_cls.__name__ == "_StockTrainerWithExitDriftAlarm"
    logged = trainer_cls(config=_run_config(tmp_path)).fit()
    assert [m[ALARM_METRIC] for m in logged] == [0, 0, 1]
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("[orbit_verl] WARNING step 3:")
    assert json.loads((tmp_path / drift.BASELINE_FILE).read_text())["step"] == 1


def test_the_registry_hands_out_the_alarm_trainer_only_inside_the_block(tmp_path):
    from verl.trainer.ppo.v1 import PPOTrainerSync, get_trainer_cls

    from orbit_verl.launch.train import trainer_with_exit_drift_alarm

    with trainer_with_exit_drift_alarm(_run_config(tmp_path)) as alarmed:
        assert get_trainer_cls("sync") is alarmed
        assert issubclass(alarmed, PPOTrainerSync) and alarmed is not PPOTrainerSync
    assert get_trainer_cls("sync") is PPOTrainerSync

    # The stock class comes back after an error inside the block as well.
    with pytest.raises(RuntimeError):
        with trainer_with_exit_drift_alarm(_run_config(tmp_path)):
            raise RuntimeError("training failed")
    assert get_trainer_cls("sync") is PPOTrainerSync


def test_the_task_runner_is_verls_own_with_the_alarm_trainer(tmp_path, monkeypatch):
    import ray
    from verl.trainer.main_ppo import TaskRunnerV1
    from verl.trainer.ppo.v1 import PPOTrainerSync, get_trainer_cls

    from orbit_verl.launch.train import task_runner_class

    stock = TaskRunnerV1.__ray_metadata__.modified_class.__ray_actor_class__
    seen = []

    def stock_run(self, config):
        seen.append(get_trainer_cls(config.trainer.v1.trainer_mode))
        return "trained"

    monkeypatch.setattr(stock, "run", stock_run)
    runner_cls = task_runner_class()
    assert isinstance(runner_cls, ray.actor.ActorClass)
    plain = runner_cls.__ray_metadata__.modified_class.__ray_actor_class__
    assert issubclass(plain, stock) and plain is not stock

    # Calling the plain class runs verl's run in this process, the way the Ray actor would.
    assert plain().run(_run_config(tmp_path)) == "trained"
    assert len(seen) == 1 and issubclass(seen[0], PPOTrainerSync) and seen[0] is not PPOTrainerSync
    assert get_trainer_cls("sync") is PPOTrainerSync
