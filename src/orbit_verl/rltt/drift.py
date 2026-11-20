"""The exit drift alarm: a warning when the RLTT exit gate moves its weight toward early loops.

``actor/rltt/expected_exit_step`` is the mean exit loop of the exit gate over the response tokens
of a training step. When the gate shifts its probability toward loop 0, this number falls. In a long
RLTT run that trained the gate, that shift came before the responses grew to the length limit and
the reward fell, so it is an early sign worth a look.

The alarm compares the number of every training step with its value at the run's first step. When
it has dropped by more than ``orbit.rltt.exit_drift_alarm``, the step logs
``orbit/exit_drift_alarm = 1`` and prints one WARNING line; otherwise the step logs 0. The alarm
never stops training. It is there to make a person look at the run.

The first step's value is saved in ``<run_dir>/exit_drift_baseline.json``. A resumed run starts at
a later step, so without the file it would compare with a value that may already have drifted.

``with_exit_drift_alarm`` makes a subclass of verl's trainer class that runs the alarm after every
training step; ``orbit_verl.launch.train`` puts it in place. This module imports nothing from
verl, so the alarm itself can be tested without it.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Optional

EXIT_STEP_METRIC = "actor/rltt/expected_exit_step"
ALARM_METRIC = "orbit/exit_drift_alarm"
BASELINE_FILE = "exit_drift_baseline.json"
THRESHOLD_KEY = "orbit.rltt.exit_drift_alarm"


def check_threshold(value: Any) -> float:
    """Return the alarm threshold as a float, or raise ValueError when it is not a positive number.

    The threshold is a drop of the expected exit step, measured in loops. A bool is refused even
    though Python counts it as a number, because ``true`` in a config is a typo here, not a threshold.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{THRESHOLD_KEY} must be a positive number of loops, got {value!r}.")
    return float(value)


def _print_line(line: str) -> None:
    print(line, flush=True)


class ExitDriftAlarm:
    """Compares the expected exit step of each training step with the run's first step.

    ``baseline_path`` is where the first step's value is kept for a resumed run; None keeps it in
    memory only. ``emit`` receives every printed line, which lets a test collect them.
    """

    def __init__(
        self,
        threshold: Any,
        baseline_path: Optional[str] = None,
        emit: Callable[[str], None] = _print_line,
    ) -> None:
        self.threshold = check_threshold(threshold)
        self.baseline_path = baseline_path
        self.emit = emit
        # (step, value) of the step the comparison starts from; set by the first observed step.
        self.baseline: Optional[tuple[int, float]] = None

    def observe(self, metrics: dict, step: int) -> Optional[bool]:
        """Add the alarm metric for this step to ``metrics``, and warn while the alarm holds.

        Returns whether the alarm fires. A step without the expected exit step, such as a step of a
        ``vanilla`` GRPO run, gets no alarm metric, and the call returns None.
        """
        if EXIT_STEP_METRIC not in metrics:
            return None
        value = float(metrics[EXIT_STEP_METRIC])
        if self.baseline is None:
            self.baseline = self._start(step, value)
        base_step, base_value = self.baseline
        drop = base_value - value
        firing = drop > self.threshold
        metrics[ALARM_METRIC] = int(firing)
        if firing:
            self.emit(
                f"[orbit_verl] WARNING step {step}: exit drift alarm. {EXIT_STEP_METRIC} is {value:.3f}, "
                f"{drop:.3f} below its value of {base_value:.3f} at step {base_step}, which is more than "
                f"{THRESHOLD_KEY}={self.threshold:g}. The exit gate is moving its weight to earlier loops; "
                "check actor/rltt/exit_pdf_t* and the response lengths. Training continues."
            )
        return firing

    def _start(self, step: int, value: float) -> tuple[int, float]:
        """Choose the baseline at the first step that this process observes.

        Step 1 starts a new run, so its value is the baseline and replaces any saved one. A later
        first step means a run resumed from a checkpoint, and the saved value of its first step is
        used. Without a saved value, which happens for a run that started before the alarm
        existed, the current value is the best baseline available, and a line says so.
        """
        if step > 1:
            saved = self._load()
            if saved is not None:
                return saved
            self.emit(
                f"[orbit_verl] exit drift alarm: no saved first-step value in {self.baseline_path}, so the "
                f"alarm compares with step {step} ({EXIT_STEP_METRIC}={value:.3f})."
            )
        self._save(step, value)
        return step, value

    def _load(self) -> Optional[tuple[int, float]]:
        if self.baseline_path is None or not os.path.isfile(self.baseline_path):
            return None
        try:
            with open(self.baseline_path, encoding="utf-8") as handle:
                record = json.load(handle)
            return int(record["step"]), float(record["value"])
        except (OSError, ValueError, KeyError, TypeError):
            # A broken file is treated like a missing one: the caller reports that it has no baseline.
            return None

    def _save(self, step: int, value: float) -> None:
        if self.baseline_path is None:
            return
        record = {"metric": EXIT_STEP_METRIC, "step": int(step), "value": float(value)}
        tmp = f"{self.baseline_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(record, handle)
            # The rename makes the write atomic, so a crash never leaves half a file for the resume.
            os.replace(tmp, self.baseline_path)
        except OSError as exc:
            # The alarm must never stop training; without the file only a resumed run loses its baseline.
            self.emit(f"[orbit_verl] exit drift alarm: cannot save the baseline to {self.baseline_path}: {exc}")


def alarm_from_config(config: Any) -> ExitDriftAlarm:
    """The alarm of a composed run config: its threshold, and the baseline file in the run dir."""
    orbit = config.orbit
    return ExitDriftAlarm(orbit.rltt.exit_drift_alarm, os.path.join(str(orbit.run_dir), BASELINE_FILE))


def with_exit_drift_alarm(trainer_cls: type) -> type:
    """Return a subclass of a verl V1 trainer class that runs the exit drift alarm after every step.

    verl's ``fit`` calls ``step(metrics, timing_raw)`` once per training step. The actor's metrics,
    and with them the expected exit step, are in ``metrics`` when it returns, and ``fit`` logs the
    dict after the checkpoint save. The alarm adds its metric in between, so it is logged with the
    rest of the step's metrics.
    """

    class ExitDriftAlarmTrainer(trainer_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._orbit_exit_drift = alarm_from_config(self.config)

        def step(self, metrics, timing_raw):
            batch = super().step(metrics, timing_raw)
            self._orbit_exit_drift.observe(metrics, self.global_steps)
            return batch

    name = f"{trainer_cls.__name__}WithExitDriftAlarm"
    ExitDriftAlarmTrainer.__name__ = ExitDriftAlarmTrainer.__qualname__ = name
    return ExitDriftAlarmTrainer
