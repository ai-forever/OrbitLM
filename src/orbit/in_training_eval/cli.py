"""Command line of the in-training evaluation: plan and prefetch.

    python -m orbit.train_eval plan --suite S --world-size 64
    python -m orbit.train_eval prefetch --suite S --eval-python P --cache-dir C --tokenizer T [--harness H]

``plan`` prints the work units, their assignment to ranks and the estimated wall time. ``prefetch``
fills the offline cache (datasets, the code_eval metric, the RULER haystack, nltk data) by running
every task once with lm-eval's dummy model while the network is available.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .benchmarks import BENCHMARKS, NLTK_PUNKT_CHECK
from .environment import build_eval_env, libstdcxx_preload
from .harness import LM_EVAL_VERSION, harness_folder_problem
from .plan import build_plan, expand_units
from .preflight import lm_eval_problems, probe_eval_environment
from .suite import DEFAULT_SUITE_PATH, load_suite


def _cli_plan(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    plan = build_plan(suite, args.world_size)
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
        return 0
    per_benchmark: dict[str, float] = {}
    for unit in plan.units:
        per_benchmark[unit.benchmark] = per_benchmark.get(unit.benchmark, 0.0) + unit.cost
    total_cost = sum(per_benchmark.values())
    processes = sum(len(procs) for procs in plan.processes)
    print(f"suite {suite.path}")
    print(f"units {len(plan.units)}, processes {processes}, world size {plan.world_size}")
    for name, cost in per_benchmark.items():
        print(f"  {name:12s} {cost / 60.0:8.1f} GPU-min (estimate)")
    startup = processes * suite.process_startup_seconds
    print(f"work {total_cost / 3600.0:.2f} GPU-h + startup {startup / 3600.0:.2f} GPU-h")
    print(f"estimated wall time of one evaluation: {plan.makespan_seconds / 60.0:.1f} min "
          f"(the slowest rank; plus the export and the GPU hand-over)")
    return 0


def _cli_prefetch(args: argparse.Namespace) -> int:
    """Runs every task once with lm-eval's dummy model, online, so the offline cache has it all."""
    suite = load_suite(args.suite)
    cache_dir = Path(args.cache_dir).resolve()
    harness = Path(args.harness) if args.harness else None

    def prefetch_env() -> dict[str, str]:
        env = build_eval_env(
            dict(os.environ),
            eval_python=Path(args.eval_python),
            gpu="",
            cache_dir=cache_dir,
            # Scratch caches of the prefetch itself, so that nothing lands in the home folder.
            rank_cache=cache_dir / "ranks" / "prefetch",
            harness_path=harness,
            extra_pythonpath=args.extra_pythonpath,
            offline=False,
        )
        # The shared modules cache, not a rank's, so that later ranks find code_eval there.
        env["HF_MODULES_CACHE"] = str(cache_dir / "hf" / "modules")
        return env

    # The cache must hold what the pinned lm-eval's tasks load, so the prefetch refuses another
    # lm-eval in the same way as the trainer's startup check, before it creates any folder.
    problem = harness_folder_problem(harness, args.eval_python) if harness is not None else None
    if problem is None:
        report = probe_eval_environment(Path(args.eval_python), prefetch_env(), ["lm_eval"])
        problems = [f"cannot import {name}: {error}" for name, error in report["errors"].items()]
        problems += lm_eval_problems(report, harness, eval_python=args.eval_python)
        problem = "; ".join(problems) or None
    if problem is not None:
        print(f"prefetch FAILED: {problem}", flush=True)
        return 1
    print(f"prefetch lm_eval={report['modules']['lm_eval']['file']} "
          f"files_verified={report.get('lm_eval_files_verified')}", flush=True)
    # The prefetch runs RULER's prompt generator as well, so its log also says which libstdc++ the
    # evaluation processes get (see build_eval_env).
    print(f"prefetch libstdc++: {libstdcxx_preload(Path(args.eval_python), os.environ.get('LD_LIBRARY_PATH', ''))[1]}",
          flush=True)
    (cache_dir / "hf").mkdir(parents=True, exist_ok=True)
    (cache_dir / "nltk").mkdir(parents=True, exist_ok=True)
    # Built again now that the nltk folder exists, because build_eval_env points NLTK_DATA there
    # only when it exists, and the RULER tasks read the punkt data from it.
    env = prefetch_env()
    nltk_code = (
        "import nltk, sys; ok = all(nltk.download(n, download_dir=sys.argv[1], quiet=True) "
        "for n in ('punkt', 'punkt_tab')); sys.exit(0 if ok else 1)"
    )
    failures = []
    requested = [name for name in suite.benchmarks if not args.only or name in args.only.split(",")]
    # RULER's prompt generator and IFEval's checks both read nltk's punkt data from the cache.
    if any(name == "ruler" or NLTK_PUNKT_CHECK in BENCHMARKS[name].modules for name in requested):
        rc = subprocess.run([args.eval_python, "-c", nltk_code, str(cache_dir / "nltk")], env=env).returncode
        print(f"prefetch nltk punkt rc={rc}", flush=True)
        if rc != 0:
            failures.append("nltk")
    groups: dict[str, list[str]] = {}
    chat_tasks: set[str] = set()
    for unit in expand_units(suite):
        if unit.benchmark not in requested:
            continue
        key = unit.benchmark if unit.length is None else f"ruler@{unit.length}"
        for task in unit.tasks:
            if BENCHMARKS[unit.benchmark].chat_mode is not None:
                # Chat tasks serve several prompt modes; one download per task is enough.
                if task in chat_tasks:
                    continue
                chat_tasks.add(task)
            if task not in groups.setdefault(key, []):
                groups[key].append(task)
    lengths = suite.ruler_lengths
    out_root = cache_dir / "prefetch_runs"
    only = {name for name in (args.only or "").split(",") if name}
    for key, tasks in groups.items():
        if only and key.split("@")[0] not in only:
            continue
        if not tasks:
            continue
        if key.startswith("ruler@") and int(key.split("@")[1]) != min(lengths):
            continue  # the haystack is the same for every length
        command = [
            args.eval_python, str(suite.entry), "--model", "dummy", "--include_path", str(suite.tasks_dir),
            "--tasks", ",".join(tasks), "--limit", "1", "--output_path", str(out_root / key),
            "--confirm_run_unsafe_code",
        ]
        if key.startswith("ruler@"):
            metadata = {"max_seq_lengths": [min(lengths)], "num_samples": 1, "pretrained": str(args.tokenizer)}
            command += ["--metadata", json.dumps(metadata)]
        print(f"prefetch {key}: {len(tasks)} tasks", flush=True)
        rc = subprocess.run(command, env=env).returncode
        print(f"prefetch {key} rc={rc}", flush=True)
        if rc != 0:
            failures.append(key)
    shutil.rmtree(out_root, ignore_errors=True)
    print("prefetch " + ("FAILED: " + ", ".join(failures) if failures else "ok"), flush=True)
    return 1 if failures else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Print the work units and the estimated wall time.")
    plan.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)
    plan.add_argument("--world-size", type=int, required=True)
    plan.add_argument("--json", action="store_true")
    prefetch = sub.add_parser("prefetch", help="Fill the offline cache for a suite (needs network).")
    prefetch.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)
    prefetch.add_argument("--eval-python", required=True)
    prefetch.add_argument("--harness", default=None,
                          help=f"Folder with lm-eval {LM_EVAL_VERSION}, as eval_harness_path of the trainer.")
    prefetch.add_argument("--extra-pythonpath", default=None)
    prefetch.add_argument("--cache-dir", required=True)
    prefetch.add_argument("--tokenizer", required=True, help="Tokenizer folder, used to build RULER prompts.")
    prefetch.add_argument("--only", default=None, help="Comma-separated benchmarks to fetch (default: all).")
    args = parser.parse_args(argv)
    if args.command == "plan":
        return _cli_plan(args)
    return _cli_prefetch(args)
