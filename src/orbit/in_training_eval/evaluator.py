"""The per-rank evaluator that train.py calls at every evaluated step, and the function that builds it."""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import socket
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist

from .environment import build_eval_env, build_lm_eval_command, compile_threads_per_rank, prepare_rank_cache
from .export import write_eval_export
from .files import (
    Heartbeat, HeartbeatWatcher, chat_records_path, find_results_file, read_json, wait_for_files, write_json_atomic,
)
from .harness import LM_EVAL_VERSION
from .options import eval_requested
from .plan import build_plan
from .preflight import CHAT_RENDER_CHECK, CUDA_CHECK, run_preflight
from .processes import GpuHandle, kill_eval_processes, run_eval_process
from .scoring import CHAT_PAUSE_WARNING_SECONDS, Scores, report_eval_scalars, score_process_folders
from .settings import EvalSettings

GIB = 1024 ** 3


class InTrainingEval:
    """Runs the evaluation at the configured steps. One instance lives on every rank.

    The class keeps a strict rule about collectives: every rank makes the same NCCL calls in the
    same order, and none is made while ranks may wait on each other for long. There are three:
    the decision at the start (one all_reduce), the weight gather for the export, and one
    all_reduce at the end, when every rank is done.

    In between, the ranks wait for each other on files in the shared run folder. Every such wait
    ends at the latest at a wall-clock deadline that all ranks agreed on in the first all_reduce.
    So even when something fails (a full disk, a dead rank, a hung process), all ranks give up at
    the same moment and reach the final all_reduce together, well within the NCCL timeout.

    On disk, every evaluated step has a folder run_dir/eval/step_XXXXXXXX with plan.json,
    attempts.json, the per-process results under procs/, one attempt_NNN folder per attempt
    (export.json, the rank markers, heartbeats and release.json) and, once it is done,
    summary.json.
    """

    # How long after the evaluation deadline rank 0 keeps waiting for the rank markers. A rank
    # that hits the deadline still kills its process (up to about a minute), waits for its GPU
    # memory (up to four minutes) and only then writes its marker.
    GRACE_SECONDS = 420.0
    # How much longer the other ranks wait for rank 0's release file. Rank 0 needs that time to
    # score, write the summary and report.
    RELEASE_SECONDS = 300.0
    # Once a stop (SIGTERM) is requested, waits for other ranks are cut to this many seconds.
    STOP_WAIT_SECONDS = 60.0
    QUIET_SECONDS = 60.0
    GPU_RECOVERY_SECONDS = 180.0
    GPU_RECHECK_SECONDS = 60.0
    GPU_MARGIN_BYTES = 2 * GIB
    HEARTBEAT_SECONDS = 15.0
    # A rank whose heartbeat has not changed for this long counts as dead. It is long compared
    # with how late a shared filesystem may show another node's writes (up to about a minute), so
    # a live rank is never taken for dead.
    HEARTBEAT_STALE_SECONDS = 600.0
    # A step whose evaluation was cut off by a crash this many times is given up. Otherwise an
    # evaluation that kills the job would run again first at every restart, and training could
    # never get past it.
    MAX_CRASHED_ATTEMPTS = 2
    WRITE_RETRY_SECONDS = 2.0

    def __init__(
        self,
        settings: EvalSettings,
        *,
        rank: int,
        local_rank: int,
        world_size: int,
        distributed: bool,
        device: Optional[torch.device],
        run_dir: Path,
        poll_seconds: float = 5.0,
        local_world_size: Optional[int] = None,
    ):
        self.settings = settings
        self.rank = int(rank)
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.distributed = bool(distributed)
        self.device = device
        self.run_dir = Path(run_dir)
        self.poll_seconds = float(poll_seconds)
        self.plan = build_plan(settings.suite, self.world_size)
        self.fingerprint = settings.fingerprint(self.plan)
        self.export_key = settings.export_key()
        self.gpu = GpuHandle(device)
        # The export of the step under evaluation always lives at this one path. vLLM keys its
        # compile cache by the model path among other things, so a path per step would make
        # every evaluation compile the model from scratch.
        self.export_dir = self.run_dir / "eval" / "model"
        # Compile caches per run and per rank. Two runs that share one eval_cache_dir would
        # otherwise write the same cache folders at the same time.
        self.rank_cache = self.run_dir / "eval" / "cache" / f"rank_{self.rank:05d}"
        if local_world_size is None:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE") or 1)
        self.compile_threads = compile_threads_per_rank(local_world_size)
        self.model: Any = None
        self.clearml_logger: Any = None
        self.tb_writer: Any = None
        self.stop_requested: Callable[[], bool] = lambda: False

    # -- wiring ----------------------------------------------------------------

    def attach(self, *, model: Any, clearml_logger: Any, tb_writer: Any,
               stop_requested: Callable[[], bool]) -> None:
        self.model = model
        self.clearml_logger = clearml_logger
        self.tb_writer = tb_writer
        self.stop_requested = stop_requested

    def log(self, message: str, *, all_ranks: bool = False) -> None:
        if all_ranks or self.rank == 0:
            print(f"in_training_eval rank={self.rank} {message}", flush=True)

    def describe_plan(self) -> str:
        busy = sum(1 for procs in self.plan.processes if procs)
        processes = sum(len(procs) for procs in self.plan.processes)
        return (
            f"suite={self.settings.suite.path} units={len(self.plan.units)} processes={processes} "
            f"busy_ranks={busy}/{self.world_size} estimated_wall={self.plan.makespan_seconds / 60.0:.1f}min "
            f"lm_eval={LM_EVAL_VERSION} weights={self.settings.weights} "
            f"max_model_len={self.settings.max_model_len}"
        )

    # -- triggers --------------------------------------------------------------

    def due(self, step: int, *, at_start: bool = False, last: bool = False) -> bool:
        every = self.settings.every
        on_cadence = every > 0 and step > 0 and step % every == 0
        # With eval_at_end, the last step of the run is evaluated even when eval_every does not
        # divide it. The trainer always saves a checkpoint at the last step, so that step, too, is
        # on disk before its evaluation starts.
        if last and self.settings.at_end and step > 0:
            on_cadence = True
        if at_start:
            # A run resumed at an evaluation step catches up on an evaluation that a crash, a
            # stop or an automatic restart interrupted. run() skips steps that have a summary.
            return self.settings.at_start or on_cadence
        return on_cadence

    def maybe_run(self, step: int, *, at_start: bool = False, last: bool = False) -> None:
        if self.due(int(step), at_start=at_start, last=last):
            self.run(int(step), at_start=at_start)

    # -- the evaluation ----------------------------------------------------------

    def step_dir(self, step: int) -> Path:
        return self.run_dir / "eval" / f"step_{step:08d}"

    def _agree(self, values: list[float]) -> list[float]:
        """Element-wise maximum over ranks.

        The values travel as float64, which holds the small integers here exactly and a Unix time
        to well under a millisecond.
        """
        if not self.distributed:
            return [float(value) for value in values]
        on_gpu = self.device is not None and self.device.type == "cuda"
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device if on_gpu else "cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return [float(value) for value in tensor.tolist()]

    def _write_quietly(self, path: Path, payload: Any, *, attempts: int = 3) -> bool:
        """Writes a JSON file and returns whether it worked, instead of raising.

        A few retries ride out a short hiccup of a shared filesystem. A full disk keeps failing,
        and the caller then takes the path that needs no file.
        """
        for attempt in range(attempts):
            try:
                write_json_atomic(path, payload)
                return True
            except Exception:
                if attempt + 1 == attempts:
                    self.log(f"cannot write {path}:\n{traceback.format_exc()}", all_ranks=True)
                else:
                    time.sleep(self.WRITE_RETRY_SECONDS)
        return False

    def _wait(self, paths: list[Path], *, deadline: float,
              given_up: Optional[Callable[[Path], bool]] = None,
              stop_window: Optional[float] = None) -> list[Path]:
        """wait_for_files that, once a stop is requested, waits at most ``stop_window`` more seconds."""
        should_stop = self.stop_requested if stop_window is not None else None
        missing = wait_for_files(paths, deadline=deadline, poll_seconds=self.poll_seconds,
                                 should_stop=should_stop, given_up=given_up)
        if missing and stop_window is not None and self.stop_requested():
            missing = wait_for_files(missing, deadline=min(deadline, time.monotonic() + stop_window),
                                     poll_seconds=self.poll_seconds, given_up=given_up)
        return missing

    def _sessions_path(self) -> Path:
        return self.rank_cache / "eval_sessions.json"

    def _recorded_sessions(self) -> list[int]:
        """Session ids of the evaluation processes this rank started on this host."""
        payload = read_json(self._sessions_path()) or {}
        if payload.get("host") != socket.gethostname():
            return []  # PIDs from another machine or container mean nothing here
        return [int(sid) for sid in payload.get("sessions", [])]

    def _record_session(self, sid: int) -> None:
        sessions = (self._recorded_sessions() + [int(sid)])[-32:]
        self._write_quietly(self._sessions_path(), {"host": socket.gethostname(), "sessions": sessions}, attempts=1)

    def kill_leftovers(self, when: str) -> list[int]:
        """Kills evaluation processes left on this rank's GPU by an earlier trainer; returns their PIDs.

        Normally run_lm_eval.py's watchdog has already killed them when that trainer died. This
        is the fallback for a watchdog that died too, for example when the wrapper process was
        killed together with the trainer and vLLM's EngineCore lived on.
        """
        killed = kill_eval_processes(self.gpu.uuid(), self._recorded_sessions())
        if killed:
            self.log(f"killed {len(killed)} evaluation processes left on this GPU ({when}): {killed}", all_ranks=True)
        return killed

    def _remove_export(self) -> None:
        shutil.rmtree(self.export_dir.with_name(self.export_dir.name + ".partial"), ignore_errors=True)
        shutil.rmtree(self.export_dir, ignore_errors=True)

    def _export_matches(self, step: int) -> bool:
        """Whether the export on disk was written for this step with the current settings."""
        info = read_json(self.export_dir / "eval_export.json") or {}
        return info.get("step") == step and info.get("export_key") == self.export_key

    def _dispose_export(self, step_dir: Path) -> None:
        """Deletes the export after the evaluation, or moves it into the step folder to keep it."""
        if not self.settings.keep_exports:
            self._remove_export()
            return
        try:
            if self.export_dir.is_dir():
                shutil.rmtree(step_dir / "model", ignore_errors=True)
                os.replace(self.export_dir, step_dir / "model")
        except OSError:
            self.log(f"cannot keep the export in {step_dir / 'model'}:\n{traceback.format_exc()}")

    def _set_attempt_outcome(self, step_dir: Path, attempt: int, outcome: str) -> None:
        attempts = (read_json(step_dir / "attempts.json") or {}).get("attempts", [])
        for entry in attempts:
            if entry.get("attempt") == attempt:
                entry["outcome"] = outcome
        self._write_quietly(step_dir / "attempts.json", {"attempts": attempts})

    def _report(self, step: int, *, scores: Scores, timings: dict[str, float], failures: dict[str, int]) -> None:
        from ..runtime import report_scalar

        try:
            report_eval_scalars(
                report_scalar, self.clearml_logger, self.tb_writer,
                scores=scores, step=step, weights=self.settings.weights, timings=timings, failures=failures,
            )
        except Exception:
            self.log(f"step={step} reporting failed:\n{traceback.format_exc()}")

    def _report_summary(self, step: int, step_dir: Path, summary: dict[str, Any]) -> None:
        """Reports the scalars stored in a summary and marks the summary as reported."""
        stored = summary.get("scores") or {}
        keys = ("benchmarks", "parts", "ruler", "samples", "incomplete", "diagnostics", "gates")
        scores = Scores(**{key: dict(stored.get(key) or {}) for key in keys})
        self._report(step, scores=scores, timings=summary.get("timings") or {}, failures=summary.get("failures") or {})
        self._write_quietly(step_dir / "summary.json", {**summary, "reported": True})

    def _prepare_step_dir(self, step: int, step_dir: Path) -> tuple[int, int, int]:
        """Rank 0 only: returns (skip, export_ready, attempt) and records the new attempt.

        A step that has a summary is done and is skipped. If its scores were never reported,
        because rank 0 died between writing the summary and reporting, they are reported now.
        A step whose evaluation was cut off by crashes MAX_CRASHED_ATTEMPTS times is given up.
        Otherwise a new attempt starts. It reuses the export and the finished processes of
        earlier attempts when they were made with the same suite, plan and weights.
        """
        summary = read_json(step_dir / "summary.json")
        if isinstance(summary, dict):
            if summary.get("reported") is False:
                self._report_summary(step, step_dir, summary)
            return 1, 0, 0
        step_dir.mkdir(parents=True, exist_ok=True)
        attempts = (read_json(step_dir / "attempts.json") or {}).get("attempts", [])
        crashed = [entry for entry in attempts if not entry.get("outcome")]
        if len(crashed) >= self.MAX_CRASHED_ATTEMPTS:
            self.log(f"step={step} the evaluation was cut off by a crash {len(crashed)} times; "
                     "it is given up so that training can go on")
            summary = {"step": step, "status": "abandoned", "attempts": attempts,
                       "failures": {"abandoned": 1}, "reported": False}
            if self._write_quietly(step_dir / "summary.json", summary):
                self._report_summary(step, step_dir, summary)
            self._remove_export()
            return 1, 0, 0
        plan_file = read_json(step_dir / "plan.json") or {}
        if plan_file.get("fingerprint") != self.fingerprint:
            # Results from a different suite, rank count, weights or lm-eval cannot be reused.
            shutil.rmtree(step_dir / "procs", ignore_errors=True)
            write_json_atomic(step_dir / "plan.json", {"fingerprint": self.fingerprint, **self.plan.to_dict()})
        attempt = len(attempts) + 1
        attempts.append({"attempt": attempt, "started": time.time(), "outcome": None})
        write_json_atomic(step_dir / "attempts.json", {"attempts": attempts})
        return 0, int(self._export_matches(step)), attempt

    def run(self, step: int, *, at_start: bool = False) -> None:
        """Evaluates one step. Every rank calls this at the same step.

        A failed evaluation never raises, and training resumes. There is one exception: when the
        evaluation processes on some rank did not give the GPU memory back, so that the trainer
        would run out of memory at its next step, all ranks raise the same error together. The
        checkpoint of the step is already on disk at that point.
        """
        started = time.monotonic()
        step_dir = self.step_dir(step)
        decision = (0, 0, 0)
        if self.rank == 0:
            try:
                decision = self._prepare_step_dir(step, step_dir)
            except Exception:
                self.log(f"step={step} cannot prepare {step_dir}, skipping:\n{traceback.format_exc()}")
                decision = (1, 0, 0)
        wanted_deadline = time.time() + self.settings.total_timeout_minutes * 60.0
        agreed = self._agree([float(bool(self.stop_requested())), *map(float, decision), wanted_deadline])
        stop, skip, export_ready, attempt = (int(value) for value in agreed[:4])
        if skip:
            self.log(f"step={step} already evaluated or not possible ({step_dir}), skipping")
            return
        if stop:
            self.log(f"step={step} stop requested, skipping the evaluation")
            return
        # The deadline is a wall-clock time that every rank agreed on, turned into this process's
        # monotonic clock. Every later wait derives from it, so ranks that give up do so together.
        deadline = time.monotonic() + (agreed[4] - time.time())
        attempt_dir = step_dir / f"attempt_{attempt:03d}"
        self.log(f"step={step} attempt={attempt} start: {self.describe_plan()}")
        heartbeat = Heartbeat(attempt_dir / "heartbeat" / f"rank_{self.rank:05d}", self.HEARTBEAT_SECONDS)
        heartbeat.start()
        try:
            gpu_state = self._evaluate(step, step_dir, attempt_dir, attempt=attempt, export_ready=bool(export_ready),
                                       deadline=deadline, started=started, strict_gpu=at_start)
        finally:
            heartbeat.stop()
        # The final collective. Every rank gets here within a poll or two of the release file
        # appearing, or at the shared release deadline when there is no release file.
        flagged = int(self._agree([float(self.rank + 1) if gpu_state.get("unusable") else 0.0])[0])
        if flagged:
            raise RuntimeError(
                f"In-training evaluation at step {step}: the evaluation processes on rank {flagged - 1} did not "
                "give the GPU memory back, and the trainer would run out of memory at its next step. All ranks "
                f"stop here. The checkpoint of step {step} is on disk, and {step_dir / 'summary.json'} lists "
                "the ranks whose GPU was not released."
            )
        self.log(f"step={step} done in {(time.monotonic() - started) / 60.0:.1f} min, training resumes")

    def _evaluate(self, step: int, step_dir: Path, attempt_dir: Path, *, attempt: int, export_ready: bool,
                  deadline: float, started: float, strict_gpu: bool) -> dict[str, Any]:
        """Everything between the first and the last collective of run(); returns the GPU state.

        It never raises. Rank 0 ends by writing release.json, which lets the other ranks go to
        the final collective. When rank 0 cannot write it, rank 0 waits until the shared release
        deadline instead, which is exactly when the other ranks stop waiting for the file.
        """
        marker_deadline = deadline + self.GRACE_SECONDS
        release_deadline = marker_deadline + self.RELEASE_SECONDS
        watcher = HeartbeatWatcher(attempt_dir / "heartbeat", self.HEARTBEAT_STALE_SECONDS)

        def rank0_dead(_path: Path) -> bool:
            return watcher.dead(0)

        export_seconds = 0.0
        export_ok = False
        records: list[dict[str, Any]] = []
        gpu_state: dict[str, Any] = {}
        try:
            export_seconds = self._export(step, attempt_dir, export_ready)
            # No collective from here until the end of run(): ranks may now wait long on each other.
            missing = self._wait([attempt_dir / "export.json"], deadline=deadline, given_up=rank0_dead, stop_window=0.0)
            export = None if missing else read_json(attempt_dir / "export.json")
            export_ok = bool(export and export.get("status") == "ok")
            records, gpu_state = self._run_rank_processes(step_dir, export_ok=export_ok, deadline=deadline,
                                                          strict_gpu=strict_gpu)
        except Exception:
            self.log(f"step={step} evaluation failed on this rank:\n{traceback.format_exc()}", all_ranks=True)
        self._write_quietly(attempt_dir / "ranks" / f"rank_{self.rank:05d}.json", {
            "rank": self.rank,
            "processes": records,
            "seconds": time.monotonic() - started,
            "stop": bool(self.stop_requested()),
            "gpu": gpu_state,
        })
        if self.rank == 0:
            released = False
            try:
                released = self._finish(step, step_dir, attempt_dir, attempt=attempt, marker_deadline=marker_deadline,
                                        started=started, export_seconds=export_seconds, export_ok=export_ok,
                                        watcher=watcher)
            except Exception:
                self.log(f"step={step} aggregation failed:\n{traceback.format_exc()}")
                released = self._close_after_failure(step, step_dir, attempt_dir, attempt)
            if not released:
                remaining = max(0.0, release_deadline - time.monotonic())
                self.log(f"step={step} the release file could not be written; waiting {remaining / 60.0:.1f} min "
                         "for the shared release deadline, when the other ranks stop waiting for it")
                time.sleep(remaining)
        else:
            missing = self._wait([attempt_dir / "release.json"], deadline=release_deadline, given_up=rank0_dead,
                                 stop_window=2.0 * self.STOP_WAIT_SECONDS)
            if missing:
                self.log(f"step={step} no release file from rank 0; resuming training anyway", all_ranks=True)
        return gpu_state

    def _export(self, step: int, attempt_dir: Path, export_ready: bool) -> float:
        """Gathers the weights (a collective) and writes the HF export on rank 0.

        It never raises. Whatever happens, rank 0 tries to leave export.json in the attempt folder
        with the status "ok" or "failed", because the other ranks wait for that file.
        """
        marker = attempt_dir / "export.json"
        if export_ready:
            if self.rank == 0:
                self.log(f"step={step} reusing the export of an earlier attempt")
                self._write_quietly(marker, {"status": "ok", "step": step, "reused": True, "path": str(self.export_dir)})
            return 0.0
        started = time.monotonic()
        state: Optional[dict[str, Any]] = None
        from ..full_state import gather_full_state

        try:
            # A collective under FSDP: every rank calls it, and only rank 0 gets the weights.
            state = gather_full_state(self.model, self.rank == 0)
        except Exception:
            self.log(f"step={step} weight gather failed:\n{traceback.format_exc()}", all_ranks=True)
        if self.rank == 0:
            try:
                if not state:
                    raise RuntimeError("No weights were gathered.")
                info = write_eval_export(
                    state,
                    model_config=_live_model_config(self.model),
                    out_dir=self.export_dir,
                    tokenizer_source=self.settings.tokenizer_source,
                    tokenizer_config=self.settings.tokenizer_config,
                    max_position_embeddings=self.settings.max_model_len,
                    metadata={"step": step, "export_key": self.export_key, **self._chat_export_metadata()},
                )
                payload = {"status": "ok", "step": step, "weights": self.settings.weights,
                           "seconds": time.monotonic() - started, "path": str(self.export_dir), **info}
            except Exception:
                error = traceback.format_exc()
                self.log(f"step={step} export failed:\n{error}")
                self._remove_export()
                payload = {"status": "failed", "step": step, "error": error}
            self._write_quietly(marker, payload)
        state = None
        gc.collect()
        return time.monotonic() - started

    def _chat_export_metadata(self) -> dict[str, Any]:
        """What a chat suite adds to eval_export.json: the template the prompts are rendered with."""
        if self.settings.suite.chat is None:
            return {}
        return {"chat_template_sha256": self.settings.chat_template_sha256}

    def _recover_gpu(self, baseline_free: int) -> bool:
        """Waits until the memory of a finished evaluation process is back; True when it is.

        vLLM's EngineCore can take a while to exit after lm-eval. If the memory is still missing
        after GPU_RECOVERY_SECONDS, every evaluation process left on this GPU is killed, and the
        memory gets one more short wait.
        """
        target = baseline_free - GIB
        if self.gpu.wait_until_free(target, timeout=self.GPU_RECOVERY_SECONDS):
            return True
        self.kill_leftovers("after the GPU memory did not come back")
        return self.gpu.wait_until_free(target, timeout=self.GPU_RECHECK_SECONDS)

    def _run_rank_processes(self, step_dir: Path, *, export_ok: bool, deadline: float,
                            strict_gpu: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Runs this rank's evaluation processes one after the other; returns (records, GPU state).

        The GPU state is empty when the memory came back after every process. Otherwise it says
        whether the trainer can still go on: it can when the free memory covers what the trainer
        reserved at its peak so far. At the start of a run (``strict_gpu``) that peak does not yet
        include a training step, so any missing memory counts as fatal.
        """
        processes = self.plan.processes[self.rank]
        if not processes:
            return [], {}
        if not export_ok:
            return [{"name": p.name, "status": "export_failed", "units": [u.name for u in p.units]} for p in processes], {}
        prepare_rank_cache(self.settings.cache_dir, self.rank_cache)
        gpu_uuid = self.gpu.uuid()
        if self.kill_leftovers("before this rank's evaluation"):
            time.sleep(10.0)  # their GPU memory is released when they have fully exited
        # The trainer's caching allocator holds most of the GPU between steps. Handing the cached
        # blocks back lets vLLM size its KV cache from the memory that is actually free.
        self.gpu.release_cached_memory()
        memory = self.gpu.memory()
        trainer_needs = self.gpu.trainer_needs()
        baseline_free = memory[0] if memory else 0
        utilization = self.settings.gpu_memory_utilization
        if memory is not None:
            free, total = memory
            utilization = min(utilization, (free - self.GPU_MARGIN_BYTES) / float(total))
        env = build_eval_env(
            dict(os.environ),
            eval_python=self.settings.eval_python,
            gpu=self.gpu.visible_id(),
            cache_dir=self.settings.cache_dir,
            rank_cache=self.rank_cache,
            harness_path=self.settings.harness_path,
            extra_pythonpath=self.settings.extra_pythonpath,
            parent_pid=os.getpid(),
            gpu_uuid=gpu_uuid,
            compile_threads=self.compile_threads,
        )
        records: list[dict[str, Any]] = []
        gpu_clean = True
        for process in processes:
            record = {"name": process.name, "units": [u.name for u in process.units]}
            proc_dir = step_dir / "procs" / process.name
            if (proc_dir / "done.json").exists() and find_results_file(proc_dir) is not None and (
                not process.is_chat or chat_records_path(proc_dir).exists()
            ):
                records.append({**record, "status": "ok", "reused": True})
                continue
            if self.stop_requested():
                records.append({**record, "status": "stopped"})
                continue
            if time.monotonic() >= deadline:
                records.append({**record, "status": "deadline"})
                continue
            if not gpu_clean:
                records.append({**record, "status": "gpu_busy"})
                continue
            if utilization < 0.2:
                records.append({**record, "status": "no_gpu_memory",
                                "free_gib": round(baseline_free / GIB, 1)})
                continue
            shutil.rmtree(proc_dir, ignore_errors=True)
            proc_dir.mkdir(parents=True)
            samples = process.samples
            samples_path = None
            if samples is not None:
                samples_path = proc_dir / "samples.json"
                write_json_atomic(samples_path, samples)
            command = build_lm_eval_command(
                process,
                eval_python=self.settings.eval_python,
                suite=self.settings.suite,
                model_dir=self.export_dir,
                out_dir=proc_dir / "lm_eval",
                max_model_len=self.settings.max_model_len,
                gpu_memory_utilization=utilization,
                samples_path=samples_path,
            )
            write_json_atomic(proc_dir / "command.json", command)
            outcome = self._run_subprocess(command, env, proc_dir, timeout=self.settings.process_timeout(process),
                                           deadline=deadline, chat=process.is_chat)
            if memory is not None and not self._recover_gpu(baseline_free):
                gpu_clean = False
                self.log(
                    f"GPU memory did not come back after {process.name}: free {self.gpu.memory()[0] / GIB:.1f} GiB, "
                    f"before {baseline_free / GIB:.1f} GiB; the remaining processes of this rank are skipped",
                    all_ranks=True,
                )
            if outcome["status"] == "ok":
                write_json_atomic(proc_dir / "done.json", outcome)
            records.append({**record, **outcome})
            self.log(f"{process.name} {outcome['status']} rc={outcome.get('rc')} "
                     f"{outcome['seconds'] / 60.0:.1f} min units={len(process.units)}", all_ranks=True)
        gpu_state: dict[str, Any] = {}
        if not gpu_clean:
            free_now = self.gpu.memory()[0]
            gpu_state = {
                "released": False,
                "free_gib": round(free_now / GIB, 2),
                "baseline_free_gib": round(baseline_free / GIB, 2),
                "trainer_needs_gib": round(trainer_needs / GIB, 2),
                "unusable": bool(strict_gpu or free_now < trainer_needs + self.GPU_MARGIN_BYTES),
            }
        return records, gpu_state

    def _run_subprocess(self, command: list[str], env: dict[str, str], proc_dir: Path, *,
                        timeout: float, deadline: float, chat: bool = False) -> dict[str, Any]:
        """Runs one lm-eval process with run_eval_process; the trainer's settings fill its knobs.

        The process's session id is recorded, so a trainer that restarts here after a crash can kill
        what is left of it.
        """
        return run_eval_process(
            command, env, proc_dir, timeout=timeout, deadline=deadline, poll_seconds=self.poll_seconds,
            quiet_seconds=self.QUIET_SECONDS, should_stop=self.stop_requested, on_start=self._record_session,
            chat=chat,
        )

    def _finish(self, step: int, step_dir: Path, attempt_dir: Path, *, attempt: int, marker_deadline: float,
                started: float, export_seconds: float, export_ok: bool, watcher: HeartbeatWatcher) -> bool:
        """Rank 0: waits for every rank, scores, writes the summary, releases the ranks, reports.

        Returns whether the release file was written.
        """
        markers = [attempt_dir / "ranks" / f"rank_{rank:05d}.json" for rank in range(self.world_size)]

        def rank_dead(path: Path) -> bool:
            return watcher.dead(int(path.stem.split("_")[1]))

        missing = self._wait(markers, deadline=marker_deadline, given_up=rank_dead, stop_window=self.STOP_WAIT_SECONDS)
        rank_records: dict[str, dict[str, Any]] = {}
        stop_seen = bool(self.stop_requested())
        gpu_ranks: list[int] = []
        for rank, marker in enumerate(markers):
            payload = read_json(marker) or {}
            stop_seen = stop_seen or bool(payload.get("stop"))
            if (payload.get("gpu") or {}).get("released") is False:
                gpu_ranks.append(rank)
            for record in payload.get("processes", []):
                rank_records[record["name"]] = record
        planned = [process for procs in self.plan.processes for process in procs]
        complete = all((rank_records.get(p.name) or {}).get("status") == "ok" for p in planned)
        if stop_seen and not complete:
            # A stop cut the evaluation short. Partial scores would put a misleading point on the
            # curves, so nothing is reported and no summary is written. A run resumed at this step
            # evaluates it again and reuses the export and the processes that did finish.
            self.log(f"step={step} stopped during the evaluation; nothing is reported, and a run resumed "
                     "at this step evaluates it again")
            self._set_attempt_outcome(step_dir, attempt, "stopped")
            return self._write_quietly(attempt_dir / "release.json", {"status": "stopped"})

        scores, chat_scores = score_process_folders(planned, step_dir / "procs", self.settings.suite.chat)
        failed_processes = sorted(name for name, record in rank_records.items() if record.get("status") != "ok")
        failed_processes += sorted(p.name for p in planned if p.name not in rank_records)
        units_failed = sum(len(units) for units in scores.incomplete.values())
        wall = time.monotonic() - started
        timings = {"wall_min": wall / 60.0, "export_min": export_seconds / 60.0}
        failures = {
            "units_failed": units_failed,
            "units_total": len(self.plan.units),
            "processes_failed": len(failed_processes),
            "ranks_missing": len(missing),
            "gpu_not_released": len(gpu_ranks),
        }
        if chat_scores is not None:
            # Chat units whose records are missing; their gates and diagnostics are withheld.
            failures["records_missing"] = len(chat_scores.missing_units)
        status = "ok" if not failed_processes and not units_failed and export_ok else "partial"
        summary = {
            "step": step,
            "status": status,
            "attempt": attempt,
            "reported": False,
            "weights": self.settings.weights,
            "lm_eval": LM_EVAL_VERSION,
            "suite": str(self.settings.suite.path),
            "fingerprint": self.fingerprint,
            "scores": scores.to_dict(),
            "timings": timings,
            "failures": failures,
            "failed_processes": failed_processes,
            "missing_ranks": [m.name for m in missing],
            "gpu_not_released_ranks": gpu_ranks,
            "processes": rank_records,
        }
        if chat_scores is not None:
            summary["chat_notes"] = chat_scores.notes
            summary["chat_template_sha256"] = self.settings.chat_template_sha256
        # The summary comes before the report and says "reported": false. If rank 0 dies in
        # between, the next attempt finds the summary and reports from it, so the scores are still
        # reported; only a crash in the moment between the report and the flag below can
        # report a step twice.
        summary_written = self._write_quietly(step_dir / "summary.json", summary)
        released = self._write_quietly(attempt_dir / "release.json", {"status": status})
        self._report(step, scores=scores, timings=timings, failures=failures)
        if summary_written:
            self._write_quietly(step_dir / "summary.json", {**summary, "reported": True})
        self._set_attempt_outcome(step_dir, attempt, "done")
        scores_line = " ".join(f"{k}={v:.4f}" for k, v in scores.benchmarks.items())
        ruler_line = " ".join(f"ruler_{k}={v:.4f}" for k, v in scores.ruler.items())
        self.log(f"step={step} scores {scores_line} {ruler_line} failed_units={units_failed} "
                 f"failed_processes={len(failed_processes)} wall={wall / 60.0:.1f}min")
        if chat_scores is not None:
            for note in chat_scores.notes:
                self.log(f"step={step} {note}")
            if wall > CHAT_PAUSE_WARNING_SECONDS:
                self.log(
                    f"step={step} the chat evaluation paused training for {wall / 60.0:.1f} min; above "
                    f"{CHAT_PAUSE_WARNING_SECONDS / 60.0:.0f} min, consider eval_every 2000 (eval_timing/wall_min "
                    "shows the trend)"
                )
        self._dispose_export(step_dir)
        return released

    def _close_after_failure(self, step: int, step_dir: Path, attempt_dir: Path, attempt: int) -> bool:
        """Rank 0, after _finish raised: marks the step as done without scores and releases the ranks."""
        try:
            self._write_quietly(step_dir / "summary.json",
                                {"step": step, "status": "aggregation_failed", "attempt": attempt, "reported": True})
            released = self._write_quietly(attempt_dir / "release.json", {"status": "aggregation_failed"})
            self._set_attempt_outcome(step_dir, attempt, "done")
            self._dispose_export(step_dir)
            return released
        except Exception:
            self.log(f"step={step} cleanup after the failed aggregation failed:\n{traceback.format_exc()}")
            return False


def _live_model_config(model: Any) -> Any:
    """The config of the OuroForCausalLM inside the trainer's wrappers."""
    from ..fsdp_utils import unwrap_model

    inner = unwrap_model(model)
    candidate = getattr(inner, "model", inner)
    config = getattr(candidate, "config", None)
    if config is None:
        raise RuntimeError(f"Cannot find the model config inside {type(model).__name__}.")
    return config


def build_in_training_eval(
    args: argparse.Namespace,
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    distributed: bool,
    device: Optional[torch.device],
    run_dir: Path,
) -> Optional[InTrainingEval]:
    """The evaluator for this rank, or None when the evaluation is off.

    Local rank 0 of every node checks the evaluation environment here, before the model is
    built: it imports the evaluation packages and runs a small CUDA matmul on its GPU with them.
    All ranks then agree in one all_reduce, and every rank raises when any node failed, so a
    broken environment on any node stops the run at startup.
    """
    if not eval_requested(args):
        return None
    settings = EvalSettings.from_args(args)
    evaluator = InTrainingEval(
        settings,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
        device=device,
        run_dir=run_dir,
    )
    # Before the model is built: evaluation processes that a crashed trainer left on this GPU
    # would otherwise hold most of its memory, and loading the model would run out of memory.
    if evaluator.kill_leftovers("at startup"):
        time.sleep(10.0)
    failure = ""
    if local_rank == 0:
        try:
            prepare_rank_cache(settings.cache_dir, evaluator.rank_cache)
            report = run_preflight(settings, gpu=evaluator.gpu.visible_id(), rank_cache=evaluator.rank_cache)
            modules = " ".join(
                f"{name}={info.get('version')}" for name, info in report["modules"].items()
                if "." not in name and not name.startswith("+")
            )
            gpu_name = (report["modules"].get(CUDA_CHECK) or {}).get("file")
            evaluator.log(f"preflight ok python={report['python']} {modules} gpu={gpu_name} "
                          f"lm_eval={report['modules']['lm_eval']['file']} "
                          f"lm_eval_modules={len(report['lm_eval_modules'])} "
                          f"lm_eval_files_verified={report.get('lm_eval_files_verified')} "
                          f"libstdc++={','.join(report.get('libstdcxx') or ['unknown'])}", all_ranks=True)
            evaluator.log(f"libstdc++: {report['libstdcxx_preload']}", all_ranks=True)
            if CHAT_RENDER_CHECK in report["modules"]:
                evaluator.log(f"chat render check ok: {report['modules'][CHAT_RENDER_CHECK]['file']}", all_ranks=True)
        except Exception as exc:
            failure = str(exc)
            evaluator.log(failure, all_ranks=True)
    failed = int(evaluator._agree([float(rank + 1) if failure else 0.0])[0])
    if failed:
        raise RuntimeError(failure or (
            f"in-training eval preflight failed on the node of global rank {failed - 1}; the log of that rank "
            "has the reason."
        ))
    if rank == 0:
        evaluator.log(f"every={settings.every} at_start={settings.at_start} at_end={settings.at_end} "
                      f"{evaluator.describe_plan()}")
        needed_minutes = (2.0 * evaluator.plan.makespan_seconds + 600.0) / 60.0
        if settings.total_timeout_minutes < needed_minutes:
            evaluator.log(
                f"WARNING eval_total_timeout_minutes={settings.total_timeout_minutes:g} is below twice the planned "
                f"pause plus 10 minutes ({needed_minutes:.0f} min); slow evaluations will be cut off"
            )
    return evaluator
