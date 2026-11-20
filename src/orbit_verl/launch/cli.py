"""Command line of the launcher: ``scripts/verl/launch.py --config NAME --export PATH [options] [overrides]``.

``python -m orbit_verl.launch`` is the same command for a shell that has ``<repo>/src`` on its path.
One entry point serves all three ways to run:

- local: one node (``WORLD_SIZE`` unset or 1), started from a terminal or by a scheduler; verl's
  ``run_ppo`` starts a private Ray cluster in this process;
- head: node rank 0 of a multi-node launch; it starts the Ray head, waits for all nodes, trains, and
  stops Ray at the end;
- worker: any other node rank; it joins the head's cluster and leaves when the head is gone.

The steps run in a fixed order, and each step depends on the ones before it:
parse arguments, read the node topology, remove the distributed launcher's variables, compose the config,
derive the runs root, model dir name and run name, split the GPUs between the student and the
teacher when distillation is on, decide whether a resumed run validates before its first step
(``prepare.resume_validation``), set the node env (its caches live under the runs root), set the
job env (``os.environ`` and the Ray runtime env), load the ClearML dotenv into ``os.environ``, run
the pre-flight checks (and prepare the model dir and dataset in local mode), enter the run dir and
save the resolved config, then start the mode.

``--export PATH`` is the usual way to choose the model. In local mode it also builds a missing
model dir and dataset first. A launch with ``--no-prepare`` never prepares anything, and neither does
a multi-node launch, because every node would build the same dirs at once. Such launches need one
``--prepare-only`` run first. docs/verl/README.md has the naming rules of the runs root.

Exit codes: 0 success, 1 training error, 2 configuration or pre-flight error, 3 the Ray cluster
did not form.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Sequence

from orbit_verl import common
from orbit_verl.launch import cluster
from orbit_verl.launch import env as envmod
from orbit_verl.launch.config import (REPO_ROOT, DistillSetup, DistillTopologyError, compose_config, finish_config,
                                       finish_distillation, mode_overrides, model_overrides, put_runtime_env,
                                       to_yaml)
from orbit_verl.launch.prepare import Problem, resume_validation

EXIT_OK = 0
EXIT_TRAINING = 1
EXIT_CONFIG = 2
EXIT_CLUSTER = 3

# The thin entry script that people run; it puts <repo>/src on the path by itself.
LAUNCH_SCRIPT = REPO_ROOT / "scripts" / "verl" / "launch.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/verl/launch.py",
        description="Run OrbitLM GRPO/RLTT training with verl: locally, on one node, or on many nodes. "
                    "Arguments that are not options are hydra overrides, e.g. trainer.total_training_steps=10.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True,
                        help="Experiment name in configs/verl (e.g. rl_sft_mix) or a path to a YAML.")
    parser.add_argument("--export", default=None, metavar="PATH",
                        help="HF export dir of the checkpoint to train (config.json, *.safetensors, "
                             "modelling_ouro.py). In local mode a missing model dir and dataset are built first.")
    parser.add_argument("--model-name", default=None, metavar="NAME",
                        help="Name of the prepared model dir under <runs root>/models. Default with --export: "
                             "<export run dir>__<export step dir>.")
    parser.add_argument("--runs-root", default=None, metavar="PATH",
                        help="Root of all outputs and caches. Default: orbit.runs_root of the config, else "
                             f"${common.RUNS_ROOT_ENV}. There is no built-in default.")
    parser.add_argument("--gpus-per-node", type=int, default=None,
                        help="GPUs per node. Default: the count of CUDA_VISIBLE_DEVICES, else of `nvidia-smi -L`.")
    parser.add_argument("--port", type=int, default=cluster.DEFAULT_PORT, help="Ray GCS port on the head node.")
    parser.add_argument("--num-cpus", type=int, default=None,
                        help="CPUs that Ray may use on each node, for when Ray detects too few.")
    parser.add_argument("--join-timeout", type=float, default=900.0,
                        help="Seconds to wait for all nodes (rank 0) or for the head (other ranks).")
    parser.add_argument("--max-lifetime", type=float, default=0.0,
                        help="Optional hard stop in seconds for worker nodes; 0 (the default) means no limit. "
                             "A worker already leaves when the head's port stops answering. A limit applies "
                             "even while the head is healthy, and leaving kills the training actors on that "
                             "node, so a limit shorter than the run fails the run.")
    parser.add_argument("--prepare", action="store_true",
                        help="Build a missing model dir or dataset from orbit.prepare before training "
                             "(implied by --export in local mode).")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Like --prepare, then run every check that needs no GPU, and exit.")
    parser.add_argument("--force-prepare", action="store_true",
                        help="Also rebuild a model dir or dataset that exists but does not match the config, "
                             "for example a model dir built from another export. The old dir is replaced.")
    parser.add_argument("--no-prepare", action="store_true",
                        help="Never build anything; require the prepared dirs. Multi-node launches behave this "
                             "way anyway; run --prepare-only once before them.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compose, validate and check, print the plan, and start nothing.")
    return parser


def parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Split the command line into launcher options and hydra overrides.

    Hydra overrides never start with ``-`` (``key=value``, ``+key=value``, ``~key``), so anything
    unknown that starts with ``-`` is a mistyped option and is reported as such.
    """
    parser = build_parser()
    args, rest = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    overrides = [item for item in rest if item != "--"]
    unknown = [item for item in overrides if item.startswith("-")]
    if unknown:
        parser.error(f"unknown option(s): {' '.join(unknown)}")
    if args.no_prepare and (args.prepare or args.prepare_only or args.force_prepare):
        parser.error("--no-prepare cannot be combined with --prepare, --prepare-only or --force-prepare")
    # The same setting twice with two values would leave the reader guessing which one won.
    for option, value, key in (("--export", args.export, "orbit.prepare.model.export"),
                               ("--model-name", args.model_name, "orbit.prepare.model.name")):
        if value is not None and common.override_value(overrides, key) is not None:
            parser.error(f"{option} and the override {key}=... both set the model; pass only one of them")
    if args.model_name is not None:
        try:
            ok = common.safe_name(args.model_name) == args.model_name
        except ValueError:
            ok = False
        if not ok:
            parser.error(f"--model-name {args.model_name!r} must be one directory name made of letters, "
                         "digits, '.', '_' and '-'")
    return args, overrides


def count_visible_devices(value: str) -> int:
    """Number of GPUs that ``CUDA_VISIBLE_DEVICES=value`` exposes.

    CUDA stops reading the list at the first invalid entry, so ``0,-1,2`` exposes one GPU and an
    empty value exposes none.
    """
    count = 0
    for item in value.split(","):
        item = item.strip()
        if not item or item.startswith("-"):
            break
        count += 1
    return count


def detect_gpus(explicit: int | None, environ: Any = None) -> tuple[int, str]:
    """GPUs per node and where the number came from."""
    environ = os.environ if environ is None else environ
    if explicit is not None:
        return explicit, "--gpus-per-node"
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return count_visible_devices(visible), "CUDA_VISIBLE_DEVICES"
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return 0, "nvidia-smi -L (not available)"
    return sum(1 for line in out.splitlines() if line.startswith("GPU ")), "nvidia-smi -L"


@dataclass
class Plan:
    """What this process will do, computed before anything starts so ``--dry-run`` can print it."""

    topology: envmod.Topology
    gpus: int
    gpu_source: str
    port: int
    num_cpus: int | None
    head: str | None = None          # address of the Ray head (multi-node only)
    own_ip: str | None = None        # address this node advertises to Ray (multi-node only)
    overrides: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return self.topology.mode

    @property
    def address(self) -> str:
        return "local" if self.mode == "local" else f"{self.head}:{self.port}"

    @property
    def nnodes(self) -> int:
        """Nodes of the whole job, the value that the mode overrides give ``trainer.nnodes``."""
        return 1 if self.mode == "local" else self.topology.world_size

    def ray_steps(self, join_timeout: float, max_lifetime: float) -> list[str]:
        """Human-readable Ray actions of this node, in order."""
        nodes, gpus = self.topology.world_size, self.gpus
        if self.mode == "local":
            return ['verl run_ppo: ray.init(address="local") starts a private Ray cluster in this process',
                    "train; ray.shutdown() at the end"]
        stop = "$ " + shlex.join(cluster.ray_command(["stop", "--force"]))
        if self.mode == "head":
            start = cluster.head_start_args(ip=self.own_ip, port=self.port, gpus=gpus, num_cpus=self.num_cpus)
            return ["$ " + shlex.join(cluster.ray_command(start)),
                    f"wait up to {join_timeout:.0f} s for {nodes} nodes / {nodes * gpus} GPUs at {self.address}",
                    f"train with ray_kwargs.ray_init.address={self.address}",
                    stop]
        start = cluster.worker_start_args(head=self.head, port=self.port, ip=self.own_ip, gpus=gpus,
                                          num_cpus=self.num_cpus)
        limit = f" or after {max_lifetime:.0f} s" if max_lifetime > 0 else ""
        return [f"wait up to {join_timeout:.0f} s for the head port {self.address}",
                "$ " + shlex.join(cluster.ray_command(start)),
                f"stay until the head is gone (poll every {cluster.HEAD_POLL_SECONDS} s, leave after "
                f"{cluster.HEAD_MISSES_TO_LEAVE} misses{limit}); no training on this node",
                stop]


def make_plan(topology: envmod.Topology, gpus: int, gpu_source: str, args: argparse.Namespace) -> Plan:
    """Decide the mode overrides and the Ray addresses for this node."""
    plan = Plan(topology, gpus, gpu_source, args.port, args.num_cpus)
    if plan.mode == "local":
        plan.overrides = mode_overrides(nnodes=1, gpus_per_node=gpus, ray_address="local", num_cpus=args.num_cpus)
        return plan
    if plan.mode == "head":
        plan.own_ip = cluster.own_ip(None)
        plan.head = plan.own_ip
    else:
        if not topology.master_addr:
            raise ValueError(f"Node rank {topology.rank} of {topology.world_size} needs MASTER_ADDR to find "
                             "the Ray head, and it is not set.")
        plan.head = cluster.resolve_host(topology.master_addr)
        plan.own_ip = cluster.own_ip(plan.head)
    plan.overrides = mode_overrides(nnodes=topology.world_size, gpus_per_node=gpus, ray_address=plan.address)
    # Ray refuses num_cpus when the driver joins a running cluster, even if a user override set it.
    plan.overrides.append("ray_kwargs.ray_init.num_cpus=null")
    return plan


class _Logger:
    def __init__(self, topology: envmod.Topology):
        self.prefix = f"orbit-launch {topology.mode} {topology.rank}/{topology.world_size}"

    def __call__(self, msg: str) -> None:
        print(f"[{self.prefix} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _launcher_command(args: argparse.Namespace, overrides: Sequence[str], derived: dict[str, str],
                      flag: str) -> str:
    """The launcher command line for the same config, model and overrides with one extra flag.

    It runs the entry script under this python, with the runs root and, for ``--export``, the
    model dir name spelled out. Run on any machine that sees the runs root, it therefore prepares
    exactly the dirs that this launch looked for, also when this launch is one node of several.
    """
    cmd = [sys.executable, str(LAUNCH_SCRIPT), "--config", args.config]
    if args.export is not None:
        cmd += ["--export", os.path.abspath(os.path.expanduser(args.export))]
    if (args.export is not None or args.model_name is not None) and derived.get("model_name"):
        cmd += ["--model-name", derived["model_name"]]
    if derived.get("runs_root"):
        cmd += ["--runs-root", derived["runs_root"]]
    return shlex.join([*cmd, flag, *overrides])


def _prepare_policy(args: argparse.Namespace, never_prepare: bool) -> str:
    """What a real launch with these options would build, in words for the dry run."""
    if never_prepare:
        return "never (--no-prepare or a multi-node launch only uses prepared dirs)"
    if args.force_prepare:
        return "missing dirs, and dirs that do not match the config (--force-prepare)"
    if args.prepare or args.prepare_only or args.export is not None:
        return "missing dirs"
    return "nothing (pass --export or --prepare to build missing dirs)"


def _print_problems(problems: list[Problem], prepare_command: str, never_prepare: bool, log) -> None:
    for problem in problems:
        log(f"PROBLEM ({problem.kind}): {problem.message}")
        for command in problem.commands:
            print(f"    {command}", flush=True)
    if any(p.buildable for p in problems):
        if never_prepare:
            log("This launch never prepares (--no-prepare or several nodes). Build the missing dirs once, on a "
                "machine that sees the runs root, then start the launch again:")
        else:
            log("Or build every missing dir in one go with:")
        print(f"    {prepare_command}", flush=True)


def _print_distillation(distill: DistillSetup, prepare_only: bool) -> None:
    """The dry run's account of what ``finish_distillation`` changed after composing."""
    print("distillation (set after composing, so it replaces the trainer.* values above):")
    if prepare_only:
        # The machine that prepares may not be the one that trains, so its GPU count says nothing.
        print("  GPU split: decided when the run starts, from its own GPU count")
    elif distill.topology is not None:
        print(f"  {distill.topology.describe()}")
        for item in distill.topology.overrides():
            print(f"  {item}")
    else:
        print(f"  GPU split: none fits; {distill.topology_error}")
    for name, value in sorted(distill.max_model_len.items()):
        print(f"  distillation.teacher_models.{name}.inference.max_model_len={value} "
              "(was null; the student's prompt + response + 1)")


def _rotate_metrics(run_dir: str, log) -> None:
    """Keep the metrics of an earlier launch: verl's file logger truncates metrics.jsonl on start."""
    path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        kept = os.path.join(run_dir, f"metrics.{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
        os.replace(path, kept)
        log(f"kept the metrics of the previous launch as {kept}")


def _raise_keyboard_interrupt(signum, frame):  # noqa: ARG001 - signal handler signature
    raise KeyboardInterrupt(f"signal {signum}")


def run_checks(cfg: Any, *, args: argparse.Namespace, plan: Plan, extra: list[Problem],
               build: bool = False, log=print, distill: DistillSetup | None = None) -> list[Problem]:
    """The pre-flight checks, run on rank 0 or in local mode; prepares dirs when ``build``.

    The checks of ``orbit.fits`` come first. A model that the config does not fit, by lineage or
    by declared kind, is refused before any model dir or dataset is built for it. The missing dirs
    are then not reported, because building them would not help.

    ``--prepare-only`` also runs every check that needs no GPU, so a config that the training launch
    would refuse is refused already while it prepares. The GPU count, verl's own config validation
    and the code sandbox's self-check are left out: the preparing machine may have no GPU, the
    training run's GPU count is not known here, and the sandbox's protections depend on the machine
    that trains, not on this one.

    A distillation run (``distill`` is what ``finish_distillation`` derived) also gets the checks of
    ``orbit_verl.distill.preflight``. Their warnings are logged and do not stop the run. The split
    of the GPUs between the student and the teacher depends on the run's GPU count, so it is
    reported with that count and left out of ``--prepare-only`` as well.
    """
    from orbit_verl.launch import prepare
    from orbit_verl.launch.train import validate

    problems = list(extra)
    fits = prepare.check_fits(cfg)
    if fits:
        problems += fits
    elif build:
        problems += prepare.run_prepare(cfg, force=args.force_prepare, log=log)
        # A new manifest records the lineage that prepare_model detected from the weights,
        # which can differ from the prediction made from the vocabulary size.
        problems += prepare.check_lineage(cfg)
    else:
        problems += prepare.check_prepared(cfg)
    problems += prepare.check_settings(cfg)
    # The thinking budget applies to any run whose prompt mode opens a thinking block, RL runs
    # included, so its room check runs for every config and not only in the distillation pre-flight.
    from orbit_verl.distill.preflight import check_think_budget

    problems += check_think_budget(cfg)
    problems += prepare.check_resume(cfg)
    if bool(cfg.orbit.plugin):
        problems += prepare.check_plugin_import()
    if distill is not None:
        from orbit_verl.distill import preflight

        found, warnings = preflight.run_preflight(cfg)
        problems += found
        for warning in warnings:
            log(f"WARNING: {warning}")
    if args.prepare_only:
        return problems
    problems += prepare.check_code_sandbox(cfg)
    if plan.gpus < 1:
        problems.append(Problem("gpus", f"No GPUs found ({plan.gpu_source}). Pass --gpus-per-node N."))
    elif distill is not None and distill.topology_error:
        problems.append(Problem("gpus", distill.topology_error))
    if plan.gpus >= 1:
        try:
            validate(cfg)
        except Exception as exc:  # noqa: BLE001 - verl raises several exception types for bad configs
            problems.append(Problem("config", f"verl validate_config failed: {type(exc).__name__}: {exc}"))
    return problems


def _line_buffered_output() -> None:
    """Flush stdout and stderr at every newline, also when they are a file or a pipe.

    Ray prints the output of every worker through the driver's stdout. Python buffers a file or a
    pipe in 8 KB blocks, so the step lines reached launch.log, a redirected terminal log or a
    scheduler's job log only in bursts, often at the very end of a short run. Streams that cannot be
    reconfigured (pytest's capture, for example) are left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(line_buffering=True)
            except (ValueError, OSError):
                pass


def main(argv: Sequence[str] | None = None) -> int:
    # This has to run before the config is composed. Composing imports verl, and that import sets
    # TORCHINDUCTOR_CACHE_DIR to a directory in /tmp, which must not count as the user's choice.
    caller_cache_env = envmod.caller_set_node_env()
    args, overrides = parse_args(argv)
    _line_buffered_output()
    try:
        topology = envmod.read_topology()
    except ValueError as exc:
        print(f"[orbit-launch] cannot read the node topology: {exc}", flush=True)
        return EXIT_CONFIG
    log = _Logger(topology)

    # From here on, nothing that starts Ray may see the distributed launcher's variables.
    removed = envmod.remove_launcher_env()
    gpus, gpu_source = detect_gpus(args.gpus_per_node)
    try:
        plan = make_plan(topology, gpus, gpu_source, args)
    except ValueError as exc:
        log(f"ERROR: {exc}")
        return EXIT_CONFIG

    # A multi-node launch never builds anything: its nodes may not all see the export, and every
    # node would build the same dirs at once. --no-prepare asks a single node for the same.
    never_prepare = args.no_prepare or plan.mode != "local"
    if never_prepare and (args.prepare or args.prepare_only or args.force_prepare):
        log("ERROR: a multi-node launch never prepares. Run the same launch with --prepare-only on one "
            "machine first, then start every node without --prepare, --prepare-only or --force-prepare.")
        return EXIT_CONFIG

    chosen_model = model_overrides(args.export, args.model_name, overrides)
    try:
        cfg = compose_config(args.config, [*overrides, *chosen_model], mode=plan.overrides)
        derived = finish_config(cfg, args.config, runs_root=args.runs_root)
        # finish_config sees the chosen name as a config value; the reader wants to know the option.
        if args.model_name is not None:
            derived["model_name_source"] = "--model-name"
        elif args.export is not None and common.override_value(overrides, "orbit.prepare.model.name") is None:
            derived["model_name_source"] = "derived from --export"
        # The mode overrides gave the student every GPU of the job. A distillation run hands the
        # teacher its share here, before anything prints or saves the config.
        distill = finish_distillation(cfg, nnodes=plan.nnodes, gpus_per_node=plan.gpus, user_overrides=overrides)
        job_env = envmod.job_env(cfg, str(cfg.orbit.repo_root))
        put_runtime_env(cfg, job_env)
        run_dir = str(cfg.orbit.run_dir)
    except (common.RunsRootError, DistillTopologyError) as exc:
        log(f"ERROR: {exc}")
        return EXIT_CONFIG
    except Exception as exc:  # noqa: BLE001 - hydra and omegaconf raise many types; all are config errors
        log(f"ERROR: the config does not compose: {type(exc).__name__}: {exc}")
        return EXIT_CONFIG

    # A resumed run validates before its first step only when the step it resumes from has no
    # logged validation. Worker nodes do not train, so only rank 0 or a local run decides.
    resume_note = resume_validation(cfg) if plan.mode != "worker" else None

    if args.print_config:
        print(to_yaml(cfg), flush=True)
        return EXIT_OK

    # Tool functions that the checks call in-process find the root through this variable.
    os.environ[common.RUNS_ROOT_ENV] = derived["runs_root"]
    node_env = envmod.apply_node_env(derived["runs_root"], keep=caller_cache_env)
    os.environ.update(job_env)
    early: list[Problem] = []
    clearml_file = cfg.orbit.get("clearml_env")
    clearml_keys: list[str] = []
    if clearml_file:
        try:
            secrets = envmod.load_dotenv(str(clearml_file))
        except (OSError, ValueError) as exc:
            early.append(Problem("clearml", f"Cannot read orbit.clearml_env={clearml_file}: {exc}"))
        else:
            # The secrets go into this process's env only, never into the config or the Ray
            # runtime env. Ray processes that start after this point inherit them: the private
            # cluster in local mode, or `ray start` on every node, since every node loads the file.
            os.environ.update(secrets)
            clearml_keys = sorted(secrets)

    env_plan = {
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": sys.executable,
        "argv": list(sys.argv if argv is None else argv),
        "config": args.config,
        "export": args.export,
        **derived,
        "mode": plan.mode,
        "topology": topology._asdict(),
        "gpus_per_node": plan.gpus,
        "gpu_source": plan.gpu_source,
        "ray_address": plan.address,
        "overrides": list(overrides),
        "model_overrides": chosen_model,
        "mode_overrides": plan.overrides,
        "removed_launcher_vars": removed,
        "node_env": node_env,
        "job_env": envmod.redact(job_env),
        "clearml_env_file": str(clearml_file) if clearml_file else None,
        "clearml_keys": clearml_keys,
        "distillation": None if distill is None else distill.as_dict(),
        "resume_validation": resume_note,
    }

    # In local mode --export builds what is missing, so one command goes from a checkpoint to
    # training. A dry run only looks.
    build = not args.dry_run and not never_prepare and (
        args.prepare or args.prepare_only or args.force_prepare or args.export is not None)
    problems: list[Problem] = []
    if plan.mode != "worker":
        problems = run_checks(cfg, args=args, plan=plan, extra=early, build=build, log=log, distill=distill)
    prepare_command = _launcher_command(args, overrides, derived, "--prepare-only")

    if args.dry_run:
        from orbit_verl.launch.config import build_overrides, resolve_config_ref

        config_dir = resolve_config_ref(args.config)[0]
        print("=" * 30 + " orbit_verl.launch dry run " + "=" * 30, flush=True)
        print(f"mode:        {plan.mode}")
        print(f"topology:    node rank {topology.rank} of {topology.world_size}, "
              f"MASTER_ADDR={topology.master_addr or '-'}")
        print(f"gpus/node:   {plan.gpus} (from {plan.gpu_source})")
        print(f"runs root:   {derived['runs_root']} (from {derived['runs_root_source']})")
        print(f"model:       {derived['model_name'] or '-'} ({derived['model_name_source']}), "
              f"dir {cfg.orbit.model_dir if derived['model_name'] else '-'}")
        print(f"preset:      {derived['preset'] or '-'} (from {derived['preset_source']})")
        print(f"run name:    {derived['run_name']} ({derived['run_name_source']})")
        print(f"run dir:     {run_dir}")
        print(f"resume:      {resume_note or 'no checkpoint to resume, or no validation before training'}")
        print(f"prepare:     {_prepare_policy(args, never_prepare)}; a dry run builds nothing")
        print(f"removed launcher vars: {', '.join(removed) or 'none'}")
        print("overrides, in the order hydra applies them:")
        for item in build_overrides(str(cfg.orbit.repo_root), config_dir, [*overrides, *chosen_model],
                                    plan.overrides):
            print(f"  {item}")
        if distill is not None:
            _print_distillation(distill, args.prepare_only)
        print("ray plan:")
        for step in plan.ray_steps(args.join_timeout, args.max_lifetime):
            print(f"  {step}")
        print("node env (os.environ on this node only):")
        for key, value in sorted(node_env.items()):
            print(f"  {key}={value}")
        print("job env (os.environ and ray_kwargs.ray_init.runtime_env.env_vars):")
        for key, value in sorted(envmod.redact(job_env).items()):
            print(f"  {key}={value}")
        if clearml_file:
            print(f"clearml: {len(clearml_keys)} keys from {clearml_file} into os.environ only "
                  f"({', '.join(clearml_keys) or 'none'}; values hidden)")
        if plan.mode == "worker":
            print("checks:      skipped on worker nodes; rank 0 runs them")
        elif problems:
            print(f"checks:      {len(problems)} problem(s)", flush=True)
            _print_problems(problems, prepare_command, never_prepare, log)
        else:
            print("checks:      all passed")
        sys.stdout.flush()
        return EXIT_CONFIG if problems else EXIT_OK

    if problems:
        _print_problems(problems, prepare_command, never_prepare, log)
        return EXIT_CONFIG
    if args.prepare_only:
        log("model dir and dataset are ready" if plan.mode != "worker" else "nothing to prepare on worker nodes")
        return EXIT_OK

    # Relative paths that any library writes on its own now land in the run dir, not in the repo.
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)
    if plan.mode != "worker":
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as handle:
            handle.write(to_yaml(cfg))
        with open(os.path.join(run_dir, "env.json"), "w", encoding="utf-8") as handle:
            json.dump(env_plan, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _rotate_metrics(run_dir, log)
    log(f"mode={plan.mode} gpus/node={plan.gpus} run_dir={run_dir}")
    if resume_note:
        log(resume_note)
    if distill is not None and distill.topology is not None:
        log(f"distillation: {distill.topology.describe()}")

    # A stop request (SIGTERM, for example from a scheduler) unwinds like Ctrl-C, so the
    # finally blocks shut Ray down. signal() only works in the main thread; elsewhere the default
    # handler stays.
    #
    # SIGINT gets Python's Ctrl-C handler explicitly. A process started in the background by a
    # script (nohup, "&" in a non-interactive shell) inherits an ignored SIGINT, and Python then
    # installs no handler for it. Without this line, a "kill -INT" from a wrapper script did
    # nothing in that case, and the run went on training until the SIGTERM fallback.
    previous = {}
    for signum, handler in ((signal.SIGTERM, _raise_keyboard_interrupt), (signal.SIGINT, signal.default_int_handler)):
        try:
            previous[signum] = signal.signal(signum, handler)
        except ValueError:
            pass
    try:
        return _run_mode(cfg, plan, args, log)
    finally:
        for signum, handler in previous.items():
            # None means the old handler was installed outside Python and cannot be put back.
            if handler is not None:
                signal.signal(signum, handler)


def _run_mode(cfg: Any, plan: Plan, args: argparse.Namespace, log) -> int:
    if plan.mode == "worker":
        return cluster.run_worker_until_head_gone(
            head=plan.head, port=plan.port, gpus=plan.gpus, num_cpus=plan.num_cpus,
            join_timeout=args.join_timeout, max_lifetime=args.max_lifetime, log=log)

    from orbit_verl.launch.train import run_training

    try:
        if plan.mode == "head":
            if cluster.start_head(ip=plan.own_ip, port=plan.port, gpus=plan.gpus, num_cpus=plan.num_cpus,
                                  log=log) != 0:
                log("FAIL: ray start --head failed")
                return EXIT_CLUSTER
            formed = cluster.wait_for_cluster(plan.address, nodes=plan.topology.world_size,
                                              gpus_per_node=plan.gpus, timeout=args.join_timeout, log=log)
            log(f"cluster: {formed}")
            if not formed["ok"]:
                return EXIT_CLUSTER
        run_training(cfg)
        log("training finished")
        return EXIT_OK
    except KeyboardInterrupt as exc:
        log(f"interrupted ({str(exc) or 'SIGINT'}); shutting down")
        return EXIT_TRAINING
    except Exception:  # noqa: BLE001 - any failure inside verl is a training error
        traceback.print_exc()
        log("training failed")
        return EXIT_TRAINING
    finally:
        # The head's Ray processes outlive this process unless stopped. A failed `ray start` may
        # also leave some behind, so the stop runs in every head case.
        if plan.mode == "head":
            cluster.stop_ray(log=log)
