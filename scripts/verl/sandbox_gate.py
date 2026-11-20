#!/usr/bin/env python3
"""Check that a finished RL smoke scored its code rows in training and validation without losing a prompt group.

    $PY scripts/verl/sandbox_gate.py <smoke run dir> [LOG LOG ...] [--val-steps 0,2,3]

Code rows need the code sandbox on the machine that computes the rewards. When the sandbox fails
there, verl's V1 trainer drops each code prompt with "Error in _run_prompt" in the launcher's output
and goes on with the other rows (src/orbit_verl/data/reward.py explains why). The launcher stops a
run whose machine fails the sandbox's self-check, but only a finished smoke shows that the whole path
works on that kind of machine: the code rewards were computed, some programs passed, and no prompt
group was lost. Run this gate on the smoke before the full runs on the same kind of machine.

The gate passes only when all of these hold:

1. Each launch log of the smoke (at least two: the first launch, and the launch that resumes it)
   comes from the launcher and has no line with "Error in _run_prompt" or "failed its self-check".
2. Every validation of the smoke (``--val-steps``, 0,2,3 by default) logged the metrics of every
   validation source of the run's dataset, the code sources included. Its dump holds every
   validation row, and no answer to a code problem ended in a sandbox error.
3. Every training step up to the last validation dumped exactly train_batch_size x n rollouts, so
   no prompt group was lost. Each of those steps has code rollouts, each code rollout has a score,
   and at least one code rollout passed its tests. The smoke's code rows should be problems that
   the start checkpoint solves sometimes (a pass-rate keep), so a smoke without a single pass means
   that the programs do not really run.

Without LOG arguments the gate reads the run dir's own logs: ``launch.log`` of the last launch and the
``launch.<date-time>.log`` files of the launches before it. scripts/verl/run_rl.sh writes them. A smoke
started some other way can pass the saved output of each launch as LOG files instead. The script only
reads files and uses only the standard library.

Exit codes: 0 the gate passed, 1 it failed, 2 the run dir does not exist.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# The run-dir readers of the watcher next to this script are reused, so both read verl's files alike.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import watch  # noqa: E402

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

# The line that verl logs for every lost prompt group, and the reward's message when the sandbox
# fails its self-check. The launcher's own refusal repeats the reward's message.
LOG_ERRORS = re.compile(r"Error in _run_prompt|failed its self-check")
# Every line of the launcher starts with this prefix, so a log without it is not the log of a launch.
LAUNCHER_MARK = "orbit-launch"
# The log of the last launch, and the logs that earlier launches left under a date-time name. The logs
# of other nodes (launch.node<rank>.log) do not match, because only node rank 0 trains and validates.
RUN_LOG = "launch.log"
EARLIER_RUN_LOG = re.compile(r"^launch\.\d{8}-\d{6}\.log$")


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()


def run_logs(run_dir: Path) -> list[Path]:
    """The run dir's launch logs, oldest first: the date-time named ones, then ``launch.log``."""
    earlier = sorted(path for path in run_dir.iterdir() if EARLIER_RUN_LOG.match(path.name))
    last = run_dir / RUN_LOG
    return [*earlier, last] if last.is_file() else earlier


def check_logs(paths: Sequence[Path]) -> list[str]:
    """The failures of rule 1: too few logs, a log that is not the launcher's, or an error line."""
    failures = []
    if len(set(paths)) < 2:
        failures.append(f"{len(set(paths))} launch log(s) found; the gate needs the logs of both smoke launches")
    for path in paths:
        try:
            lines = read_lines(path)
        except OSError as exc:
            failures.append(f"cannot read {path}: {exc}")
            continue
        if not any(LAUNCHER_MARK in line for line in lines):
            failures.append(f"{path} has no line of the launcher, so it is not the log of a smoke launch")
        errors = [line for line in lines if LOG_ERRORS.search(line)]
        if errors:
            failures.append(f"{path} has {len(errors)} line(s) with 'Error in _run_prompt' or 'failed its "
                            f"self-check', the first: {errors[0].strip()[:200]}")
    return failures


def val_sources(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Rows per validation data source, from a distill_data manifest (counts[split][domain][mode][source])."""
    sources: dict[str, int] = {}
    for modes in manifest.get("counts", {}).get("val", {}).values():
        for by_source in modes.values():
            for source, rows in by_source.items():
                sources[source] = sources.get(source, 0) + int(rows)
    return sources


def check_validation(run_dir: Path, steps: Sequence[int], sources: Mapping[str, int], val_n: int,
                     metrics: Mapping[int, Mapping[str, float]]) -> list[str]:
    """The failures of rule 2 at each validation step."""
    failures = []
    want_rows = sum(sources.values()) * val_n
    for step in steps:
        missing = sorted(source for source in sources
                         if f"val-core/{source}/acc/mean@{val_n}" not in metrics.get(step, {}))
        if missing:
            failures.append(f"step {step}: no validation metrics for {', '.join(missing)}")
        path = run_dir / "val_generations" / f"{step}.jsonl"
        if not path.is_file():
            failures.append(f"step {step}: {path} does not exist")
            continue
        rows = list(watch.read_jsonl(path))
        if len(rows) != want_rows:
            failures.append(f"step {step}: the validation dump has {len(rows)} rows, not {want_rows}")
        errors = sum(1 for row in rows if (watch._number(row.get("code_sandbox_error")) or 0) > 0)
        if errors:
            failures.append(f"step {step}: {errors} validation answer(s) ended in a sandbox error")
    return failures


def check_training(run_dir: Path, last_step: int, rows_per_step: int) -> list[str]:
    """The failures of rule 3 over training steps 1 to ``last_step``."""
    failures = []
    passed = 0
    for step in range(1, last_step + 1):
        path = run_dir / "rollouts" / f"{step}.jsonl"
        if not path.is_file():
            failures.append(f"step {step}: {path} does not exist")
            continue
        rows = list(watch.read_jsonl(path))
        if len(rows) != rows_per_step:
            failures.append(f"step {step}: {len(rows)} rollouts, not {rows_per_step}, so a prompt group was lost")
        code = [row for row in rows if watch.prompt_kind(str(row.get("input", ""))) == watch.CODE]
        scores = [watch._number(row.get("score")) for row in code]
        if not code:
            failures.append(f"step {step}: no code rollouts")
        elif None in scores:
            failures.append(f"step {step}: {scores.count(None)} of {len(code)} code rollouts have no score")
        passed += sum(1 for score in scores if score is not None and score > 0)
    if not passed:
        failures.append(f"no code rollout of steps 1-{last_step} passed its tests")
    return failures


def run_gate(run_dir: Path, logs: Sequence[Path], val_steps: Sequence[int]) -> tuple[bool, list[str]]:
    """Check the gate and return whether it passed, with one report line per rule."""
    report: list[str] = []
    try:
        config = (run_dir / "config.yaml").read_text(encoding="utf-8")
        values = {name: watch.yaml_scalar(config, keys) for name, keys in (
            ("batch", ("data", "train_batch_size")), ("n", ("actor_rollout_ref", "rollout", "n")),
            ("val_n", ("actor_rollout_ref", "rollout", "val_kwargs", "n")), ("data_dir", ("orbit", "data_dir")))}
        manifest = json.loads((Path(str(values["data_dir"])) / "manifest.json").read_text(encoding="utf-8"))
        batch, n, val_n = (int(str(values[name])) for name in ("batch", "n", "val_n"))
    except (OSError, ValueError) as exc:
        return False, [f"FAIL setup: cannot read the run's config.yaml or its dataset manifest: {exc}"]
    sources = val_sources(manifest)
    if not any(source.startswith(watch.CODE_SOURCE_PREFIX) for source in sources):
        return False, [f"FAIL setup: the dataset has no code validation source: {sorted(sources)}"]

    metrics = watch.read_metrics(run_dir)
    rules = (
        ("launch logs", check_logs(logs), f"{len(logs)} logs, no lost prompt group and no failed self-check"),
        ("validation", check_validation(run_dir, val_steps, sources, val_n, metrics),
         f"steps {', '.join(map(str, val_steps))}: {len(sources)} sources, {sum(sources.values()) * val_n} rows each"),
        ("training", check_training(run_dir, max(val_steps), batch * n),
         f"steps 1-{max(val_steps)}: {batch * n} rollouts each, code rollouts scored and some passed"),
    )
    for name, failures, summary in rules:
        report += [f"FAIL {name}: {failure}" for failure in failures] or [f"PASS {name}: {summary}"]
    return not any(line.startswith("FAIL") for line in report), report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="the smoke's run dir, <runs root>/runs/<run name>")
    parser.add_argument("logs", type=Path, nargs="*",
                        help="the logs of the smoke's launches (at least two); default: the run dir's launch logs")
    parser.add_argument("--val-steps", default="0,2,3",
                        help="the smoke's validation steps (default 0,2,3); training steps 1 to the last are checked")
    args = parser.parse_args(argv)
    if not args.run_dir.is_dir():
        print(f"sandbox gate: {args.run_dir} does not exist; run the two smoke launches first", file=sys.stderr)
        return EXIT_USAGE
    try:
        steps = sorted({int(item) for item in args.val_steps.split(",")})
    except ValueError:
        parser.error(f"--val-steps wants numbers separated by commas, not {args.val_steps!r}")
    logs = args.logs or run_logs(args.run_dir)
    passed, report = run_gate(args.run_dir, logs, steps)
    print("\n".join(report))
    print(f"sandbox gate: {'PASS' if passed else 'FAIL'} ({args.run_dir})", flush=True)
    return EXIT_PASS if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
