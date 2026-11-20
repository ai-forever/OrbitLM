"""Golden snapshot of what the base and smoke suites make the in-training evaluation do.

The chat evaluation of the SFT stage shares its code paths with the base protocol of the
pretraining stages. The base protocol must not move at all, because the curves of the released
runs were measured with it. This module computes, for base.yaml and smoke.yaml at several world
sizes, everything that decides what a base evaluation runs and reports: the units and their plan,
the command line of every process, the per-process timeouts, the fingerprint, the export key and
the reported scalar names of a fixed set of scores. It stores a sha256 digest of each part, so a
difference names the part that moved.

tests/train/test_train_eval_chat.py compares the digests with two files in
tests/train/fixtures/in_training_eval/:

- base_golden_reference.json holds the snapshot of the base protocol without the benchmarks in
  ADDED_SINCE_REFERENCE. It was written with the original trainer before the release cleanup and
  is never written again. The suites are compared with it after those benchmarks are left out, so
  the other benchmarks must still run and report exactly as the reference says. The fingerprint is
  not compared with this file, because it hashes the suite file among other inputs, and the added
  benchmarks changed that file. It is compared only with base_golden.json. A change to its other
  inputs, such as the pinned lm-eval version or the tokenizer path, therefore shows only there, and
  a rewrite of base_golden.json would accept such a change without notice.
- base_golden.json holds the suites as they are, the added benchmarks included. It is rewritten
  with this command when a suite changes on purpose, and the reference file shows that the change
  left the other benchmarks alone:

    PYTHONPATH=src python tests/train/base_golden.py --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "tests" / "train" / "fixtures" / "in_training_eval" / "base_golden.json"
REFERENCE_PATH = REPO_ROOT / "tests" / "train" / "fixtures" / "in_training_eval" / "base_golden_reference.json"
SUITES = ("base.yaml", "smoke.yaml")
WORLD_SIZES = (1, 8, 64)
# The benchmarks of base.yaml and smoke.yaml that the reference snapshot does not hold.
ADDED_SINCE_REFERENCE = ("mbpp",)
# The part of a snapshot that the comparison with the reference skips.
NOT_IN_REFERENCE = ("fingerprint",)


def _digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized(value: str) -> str:
    # Commands hold absolute paths of this checkout, which differ between machines.
    return str(value).replace(str(REPO_ROOT), "<repo>")


class _Recorder:
    """A ClearML logger stand-in that keeps every reported (title, series, value, step)."""

    def __init__(self):
        self.rows = []

    def report_scalar(self, title, series, value, iteration):
        self.rows.append([title, series, float(value), int(iteration)])


def snapshot(te, suite_name: str, world_size: int, without: tuple[str, ...] = ()) -> dict:
    """Digests of everything a base evaluation decides from the suite, for one world size.

    The benchmarks named in ``without`` are left out of the suite first.
    """
    from orbit.runtime import report_scalar

    suite = te.load_suite(REPO_ROOT / "eval" / "in_training" / "suites" / suite_name)
    suite = replace(suite, benchmarks={name: value for name, value in suite.benchmarks.items() if name not in without})
    plan = te.build_plan(suite, world_size)
    settings = te.EvalSettings(
        every=1000, at_start=False, suite=suite, eval_python=Path("/py"), harness_path=None,
        extra_pythonpath=None, max_model_len=33024, gpu_memory_utilization=0.75, timeout_minutes=30.0,
        total_timeout_minutes=45.0, weights="raw", keep_exports=False, cache_dir=Path("/cache"),
        tokenizer_source=Path("/tok"), tokenizer_config=Path("/tok/tokenizer_config.json"),
    )
    commands = {}
    timeouts = {}
    for processes in plan.processes:
        for process in processes:
            out = Path("/out") / process.name
            command = te.build_lm_eval_command(
                process, eval_python=Path("/py"), suite=suite, model_dir=Path("/model"), out_dir=out / "lm_eval",
                max_model_len=33024, gpu_memory_utilization=0.75,
                samples_path=out / "samples.json" if process.samples is not None else None,
            )
            commands[process.name] = [_normalized(part) for part in command]
            timeouts[process.name] = round(settings.process_timeout(process), 6)
    scores = te.Scores(
        benchmarks={name: 0.5 for name in suite.benchmarks if name != "ruler"},
        parts={"cruxeval_input": 0.25, "cruxeval_output": 0.75},
        ruler={"4096": 0.9, "mean": 0.8},
        samples={"mmlu": 14042},
        incomplete={},
    )
    recorder = _Recorder()
    te.report_eval_scalars(
        report_scalar, recorder, None, scores=scores, step=1000, weights="raw",
        timings={"wall_min": 1.0, "export_min": 0.1}, failures={"units_failed": 0, "processes_failed": 0},
    )
    return {
        "plan": _digest(plan.to_dict()),
        "commands": _digest(commands),
        "timeouts": _digest(timeouts),
        "fingerprint": settings.fingerprint(plan),
        "export_key": settings.export_key(),
        "reported": _digest(recorder.rows),
        "scores_dict": _digest(scores.to_dict()),
        "makespan_seconds": round(plan.makespan_seconds, 6),
        "processes": sum(len(procs) for procs in plan.processes),
    }


def all_snapshots(te, without: tuple[str, ...] = ()) -> dict:
    return {
        f"{suite}@{world_size}": snapshot(te, suite, world_size, without)
        for suite in SUITES
        for world_size in WORLD_SIZES
    }


def differing_parts(current: dict, golden: dict, skip: tuple[str, ...] = ()) -> list[str]:
    """The "suite@world_size:part" entries in which the current snapshots differ from a golden file."""
    return [
        f"{key}:{part}" for key in sorted(golden) for part in golden[key]
        if part not in skip and current.get(key, {}).get(part) != golden[key][part]
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="Write the golden file instead of comparing.")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from orbit import in_training_eval as te

    current = all_snapshots(te)
    differing: list[str] = []
    if args.write:
        GOLDEN_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {GOLDEN_PATH}")
    else:
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        differing = differing_parts(current, golden)
        print(f"{GOLDEN_PATH.name}: " + ("identical" if not differing else "differs: " + ", ".join(differing)))
    # The reference is compared on every call, --write included, so a rewrite of base_golden.json
    # shows at once whether the benchmarks of the reference moved.
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    moved = differing_parts(all_snapshots(te, without=ADDED_SINCE_REFERENCE), reference, skip=NOT_IN_REFERENCE)
    print(f"{REFERENCE_PATH.name} without {', '.join(ADDED_SINCE_REFERENCE)}: "
          + ("identical" if not moved else "differs: " + ", ".join(moved)))
    return 1 if moved or differing else 0


if __name__ == "__main__":
    sys.exit(main())
