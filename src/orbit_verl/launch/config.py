"""Turn an experiment YAML from configs/verl into the verl config of one run.

The composition runs in this order:

1. The overrides are applied in this order: the verl config search path, ``orbit.repo_root``, the
   user's overrides, the model chosen with ``--export`` / ``--model-name``, and last the mode
   overrides (node count, GPUs per node, Ray address). The mode overrides come last so that a stray
   user override cannot contradict the real topology.
2. After composing, ``finish_config`` fills the values that the launcher derives: the runs root,
   the model dir name and the run name.
3. When ``distillation.enabled`` is true, ``finish_distillation`` splits the job's GPUs between
   the student and the teacher, and fills the teacher's ``max_model_len``.
4. Then the job env is copied into ``ray_kwargs.ray_init.runtime_env.env_vars``.

hydra and omegaconf are imported inside the functions, so the module itself imports in a python
that has neither, such as a light CPU-only python that only reads the helpers.
"""
from __future__ import annotations

import contextlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from orbit_verl import common

# src/orbit_verl/launch/config.py -> parents[3] is the repo root that holds this orbit_verl.
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "verl"

# Where hydra finds verl's own ppo_trainer.yaml, which orbit_base.yaml inherits.
VERL_SEARCHPATH = "pkg://verl.trainer.config"

# Unquoted hydra override values may only contain these characters safely. Anything else is
# passed as a quoted string, so paths with spaces or commas still work.
_PLAIN_VALUE = re.compile(r"[A-Za-z0-9_./:@+\-]+")


def hydra_value(value: Any) -> str:
    """Format a python value as the right-hand side of a hydra override."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    text = str(value)
    if _PLAIN_VALUE.fullmatch(text):
        return text
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


# YAML files in configs/verl that are not experiments: the shared base and verl's agent-loop registry.
_NOT_EXPERIMENTS = {"orbit_base", "agent_loops"}


def available_configs(config_dir: Path = CONFIG_DIR) -> list[str]:
    """Experiment names in ``config_dir``."""
    return sorted(p.stem for p in config_dir.glob("*.yaml") if p.stem not in _NOT_EXPERIMENTS)


def resolve_config_ref(config_ref: str) -> tuple[Path, str]:
    """Return ``(config_dir, config_name)`` for a ``--config`` value.

    A bare name such as ``rl_sft_mix`` means ``configs/verl/rl_sft_mix.yaml`` in this repo.
    Anything that looks like a path (it has a ``/`` or ends in ``.yaml``) is used as given, and its
    directory becomes the primary config directory.
    """
    ref = str(config_ref)
    if "/" in ref or ref.endswith((".yaml", ".yml")):
        path = Path(ref).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Config file {path} does not exist.")
        return path.parent, path.stem
    path = CONFIG_DIR / f"{ref}.yaml"
    if not path.is_file():
        names = ", ".join(available_configs()) or "none"
        raise FileNotFoundError(f"No config named {ref!r} in {CONFIG_DIR}. Available: {names}.")
    return CONFIG_DIR, ref


def base_overrides(repo_root: str | Path, config_dir: Path) -> list[str]:
    """The overrides that always come first: the verl search path and the repo root.

    An experiment YAML normally lists the verl search path itself. Passing it again makes a YAML
    without that block compose too. A YAML outside configs/verl also needs configs/verl on the
    search path, otherwise its ``orbit_base`` default cannot be found.
    """
    searchpath = [VERL_SEARCHPATH]
    if Path(config_dir).resolve() != CONFIG_DIR.resolve():
        searchpath.append(f"file://{CONFIG_DIR}")
    return [
        f"hydra.searchpath=[{','.join(searchpath)}]",
        f"orbit.repo_root={hydra_value(Path(repo_root).resolve())}",
    ]


def mode_overrides(*, nnodes: int, gpus_per_node: int, ray_address: str = "local",
                   num_cpus: int | None = None) -> list[str]:
    """The topology overrides that the launcher applies last.

    ``num_cpus`` is only meaningful when this process starts Ray itself (``address=local``). When it
    joins a running cluster, Ray refuses ``num_cpus``, and the CPU count goes to ``ray start``.
    """
    out = [
        f"trainer.nnodes={int(nnodes)}",
        f"trainer.n_gpus_per_node={int(gpus_per_node)}",
        f"ray_kwargs.ray_init.address={hydra_value(ray_address)}",
    ]
    if num_cpus is not None and ray_address == "local":
        out.append(f"ray_kwargs.ray_init.num_cpus={int(num_cpus)}")
    return out


def build_overrides(repo_root: str | Path, config_dir: Path, user_overrides: Iterable[str] = (),
                    mode: Sequence[str] = ()) -> list[str]:
    """All overrides in the order of the module docstring: base, then the user's, then the mode overrides."""
    return [*base_overrides(repo_root, config_dir), *list(user_overrides), *list(mode)]


def compose_config(config_ref: str, overrides: Iterable[str] = (), *, repo_root: str | Path | None = None,
                   mode: Sequence[str] = ()) -> Any:
    """Compose an experiment YAML into a verl config (an omegaconf DictConfig).

    ``overrides`` are the user's hydra overrides. ``mode`` holds the topology overrides from
    ``mode_overrides``; they are applied after the user's. ``repo_root`` defaults to the repo that
    holds this package; tests pass a fake one.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir, name = resolve_config_ref(config_ref)
    all_overrides = build_overrides(repo_root or REPO_ROOT, config_dir, overrides, mode)
    # An interactive session may have initialized hydra earlier; a second initialize would raise.
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None, job_name="orbit_verl"):
        return compose(config_name=name, overrides=all_overrides)


def model_overrides(export: str | None, model_name: str | None, user_overrides: Iterable[str] = ()) -> list[str]:
    """The overrides that ``--export`` and ``--model-name`` stand for.

    ``--export`` gives the export dir and, unless a name is given, the derived model dir name. A
    name from the experiment YAML is not kept for a new export: it belongs to the YAML's own
    export, and the run would then be refused for using a dir built from another checkpoint. A name
    that the user passes as a ``orbit.prepare.model.name`` override is kept.
    """
    out = []
    if export is not None:
        out.append(f"orbit.prepare.model.export={hydra_value(os.path.abspath(os.path.expanduser(export)))}")
    if model_name is not None:
        out.append(f"orbit.prepare.model.name={hydra_value(model_name)}")
    elif export is not None and common.override_value(user_overrides, "orbit.prepare.model.name") is None:
        out.append(f"orbit.prepare.model.name={hydra_value(common.model_name_for_export(export))}")
    return out


def _value(cfg: Any, dotted: str) -> Any:
    """``cfg.<dotted>``, or None when the key is absent, null or still the mandatory marker ``???``."""
    from omegaconf import OmegaConf

    parent, _, key = dotted.rpartition(".")
    node = OmegaConf.select(cfg, parent) if parent else cfg
    if node is None or key not in node or OmegaConf.is_missing(node, key):
        return None
    return node[key]


def finish_config(cfg: Any, config_ref: str, *, runs_root: str | None = None,
                  environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Fill the values that the launcher derives after composing; return them with their origin.

    - ``orbit.runs_root``: ``runs_root`` (the ``--runs-root`` option), else the config's value,
      else ``$ORBIT_VERL_RUNS``. Raises ``common.RunsRootError`` when none is set. There is no
      default, so a run never writes into a directory that nobody chose for it.
    - ``orbit.prepare.model.name``: when the config names an export but no dir name, the name
      derived from the export (``common.model_name_for_export``).
    - ``orbit.prepare.model.preset``: ``auto`` becomes the tokenizer family, ``qwen3`` for the
      release models. The export's vocabulary size decides, as prepare_model's own ``auto`` does. When
      the export cannot be read here (a node without access to it), the prepared dir's manifest
      decides. The dataset dir names end in this value, so a dataset is never shared between
      tokenizers.
    - ``orbit.run_name``: when the config gives none, ``<config>__<model name>``. Raises
      ValueError when no model is set, because the run then has no name.

    The values are assigned to the composed config. Every path that uses them (run dir, model dir,
    dataset dir, checkpoint dir) is an interpolation, which reads the new value.
    """
    root, root_source = common.resolve_runs_root_with_source(runs_root, _value(cfg, "orbit.runs_root"), environ)
    cfg.orbit.runs_root = root
    derived = {"runs_root": root, "runs_root_source": root_source}

    export, name = _value(cfg, "orbit.prepare.model.export"), _value(cfg, "orbit.prepare.model.name")
    derived["model_name_source"] = "config"
    if name is None and export is not None:
        name = common.model_name_for_export(str(export))
        cfg.orbit.prepare.model.name = name
        derived["model_name_source"] = "derived from the export"
    derived["model_name"] = "" if name is None else str(name)

    preset = _value(cfg, "orbit.prepare.model.preset")
    derived["preset_source"] = "config"
    if str(preset) == "auto" and export is not None and name is not None:
        from orbit_verl.launch.prepare import export_family, prepared_family

        family = export_family(str(export))
        derived["preset_source"] = "the export's vocabulary size"
        if family is None:
            family = prepared_family(str(cfg.orbit.model_dir))
            derived["preset_source"] = "the prepared model dir"
        if family is None:
            # Left at auto: the checks report the unreadable export, and prepare_model reports a
            # vocabulary that matches no preset.
            derived["preset_source"] = "nothing: the export is unreadable or its vocabulary matches no preset"
        else:
            cfg.orbit.prepare.model.preset = preset = family
    derived["preset"] = "" if preset is None else str(preset)

    run_name = _value(cfg, "orbit.run_name")
    derived["run_name_source"] = "config"
    if run_name is None:
        if name is None:
            raise ValueError("No model is set, so the run has no name. Pass the HF export dir of the "
                             "checkpoint to train with --export PATH.")
        run_name = common.default_run_name(config_ref, str(name))
        cfg.orbit.run_name = run_name
        derived["run_name_source"] = "<config>__<model name>"
    derived["run_name"] = str(run_name)
    return derived


# --------------------------------------------------------------------------- distillation

# How a distillation run shares the job's GPUs. These keys sit in the orbit block, which verl
# ignores, and finish_distillation turns them into verl's trainer.* and distillation.* keys.
# orbit_base.yaml does not declare them. A command line therefore sets them with hydra's "+"
# prefix, unless the experiment YAML declares a orbit.distill block.
TEACHER_GPUS_KEY = "orbit.distill.teacher_gpus_per_node"
TEACHER_NODES_KEY = "orbit.distill.teacher_nodes"
DEFAULT_TEACHER_GPUS_PER_NODE = 2
# verl's teacher pool. The launcher derives it from the two keys above and would replace an
# override of it without a word, so such an override is refused instead.
_TEACHER_POOL_KEYS = ("distillation.nnodes", "distillation.n_gpus_per_node")


class DistillTopologyError(ValueError):
    """The job's GPUs cannot be split between the student and the teacher as configured."""


class DistillTopology(NamedTuple):
    """How a distillation run splits the job's GPUs between the student and the teacher.

    verl runs the teacher in a resource pool of its own (``teacher_pool``) next to the student's,
    and refuses a run whose pools ask for more GPUs than the cluster has. The ``shared`` layout
    keeps a few GPUs of every node for the teacher. The ``dedicated`` layout gives the teacher
    whole nodes, and the student keeps whole nodes as well.
    """

    layout: str
    student_nnodes: int
    student_gpus_per_node: int
    teacher_nnodes: int
    teacher_gpus_per_node: int

    def overrides(self) -> list[str]:
        """The verl keys that this split sets, written as hydra overrides."""
        return [f"trainer.nnodes={self.student_nnodes}", f"trainer.n_gpus_per_node={self.student_gpus_per_node}",
                f"distillation.nnodes={self.teacher_nnodes}",
                f"distillation.n_gpus_per_node={self.teacher_gpus_per_node}"]

    def describe(self) -> str:
        return (f"student {self.student_nnodes} node(s) x {self.student_gpus_per_node} GPUs, teacher "
                f"{self.teacher_nnodes} node(s) x {self.teacher_gpus_per_node} GPUs ({self.layout} nodes)")


def _count(value: Any, key: str, minimum: int) -> int:
    # A YAML true is an int in Python, and a quoted number is a string. Both are typos here, and
    # reading them as one GPU, or not reading them at all, would give the teacher a wrong share.
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DistillTopologyError(f"{key} must be a whole number of at least {minimum}, not {value!r}.")
    return value


def split_distill_topology(*, nnodes: int, gpus_per_node: int, teacher_gpus_per_node: Any = None,
                           teacher_nodes: Any = None) -> DistillTopology:
    """Split a job of ``nnodes`` nodes with ``gpus_per_node`` GPUs each between the student and the teacher.

    - ``teacher_nodes`` above 0 gives the teacher that many whole nodes and the student the other
      nodes. ``teacher_gpus_per_node`` is not used then, because every node goes to one side whole.
    - Otherwise every node keeps ``teacher_gpus_per_node`` GPUs (2 when it is None) for the teacher
      and gives the rest to the student, so both pools span all nodes.

    Raises DistillTopologyError for a value that is not a count, and for a split that leaves the
    student without a node, or without a GPU on its nodes.
    """
    nodes = 0 if teacher_nodes is None else _count(teacher_nodes, TEACHER_NODES_KEY, 0)
    if nodes > 0:
        if nodes >= nnodes:
            raise DistillTopologyError(
                f"{TEACHER_NODES_KEY}={nodes} gives the teacher every node of this {nnodes}-node job, so the "
                f"student gets none. Run on at least {nodes + 1} nodes, or set {TEACHER_NODES_KEY}=0 to put the "
                f"teacher on {TEACHER_GPUS_KEY} GPUs of every node.")
        return DistillTopology("dedicated", nnodes - nodes, gpus_per_node, nodes, gpus_per_node)
    gpus = DEFAULT_TEACHER_GPUS_PER_NODE
    if teacher_gpus_per_node is not None:
        gpus = _count(teacher_gpus_per_node, TEACHER_GPUS_KEY, 1)
    if gpus >= gpus_per_node:
        raise DistillTopologyError(
            f"{TEACHER_GPUS_KEY}={gpus} leaves the student no GPU on a node with {gpus_per_node} GPU(s). verl "
            "runs the teacher on GPUs of its own, so every node needs at least one GPU more than the teacher "
            f"takes. Lower {TEACHER_GPUS_KEY}, or run on 2 or more nodes with {TEACHER_NODES_KEY}=1.")
    return DistillTopology("shared", nnodes, gpus_per_node - gpus, nnodes, gpus)


# The keys that decide how many answers one actor update holds, and how many ranks share them.
_TRAIN_BATCH_KEY = "data.train_batch_size"
_MINI_BATCH_KEY = "actor_rollout_ref.actor.ppo_mini_batch_size"
_ROLLOUT_N_KEY = "actor_rollout_ref.rollout.n"
# verl's FSDP engine reads the sequence parallel size from fsdp_config. The older key on the actor
# itself still wins when it is above 1, because FSDPActorConfig copies it over in __post_init__.
_LEGACY_ULYSSES_KEY = "actor_rollout_ref.actor.ulysses_sequence_parallel_size"
_ULYSSES_KEY = "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size"


def _sequence_parallel_size(cfg: Any) -> int:
    """The student's ulysses sequence parallel size, as verl's FSDP actor ends up using it (1 when unset)."""
    from orbit_verl.distill.preflight import select

    for key in (_LEGACY_ULYSSES_KEY, _ULYSSES_KEY):
        value = select(cfg, key)
        if value is not None and int(value) > 1:
            return int(value)
    return 1


def batch_split_problem(cfg: Any, topology: DistillTopology) -> str | None:
    """Why the answers of one actor mini-batch do not split evenly among the student's ranks, or None.

    verl's actor update asserts that the answers of a mini-batch (``ppo_mini_batch_size`` x
    ``rollout.n``) divide evenly among the student's data-parallel ranks (``train_mini_batch`` in
    verl/workers/engine_workers.py). Before that, the trainer pads the batch up to a multiple of both
    numbers, so the assert fires only at the first update, after a whole rollout and teacher pass.
    verl's own config check skips this test when ``use_dynamic_bsz`` is on, as it is in our configs.

    The data-parallel size is the student's GPU count divided by the ulysses sequence parallel
    size. The message names the batch size nearest to the current one that divides, and on a tie
    the larger one. It names ``data.train_batch_size`` when the mini-batch is the whole batch, as
    in orbit_base.yaml, and ``ppo_mini_batch_size`` otherwise. A config without these keys, such
    as a test's plain dict, is not checked.
    """
    from orbit_verl.distill.preflight import select

    mini_batch, n = select(cfg, _MINI_BATCH_KEY), select(cfg, _ROLLOUT_N_KEY)
    if mini_batch is None or n is None:
        return None
    mini_batch, n = int(mini_batch), int(n)
    gpus = topology.student_nnodes * topology.student_gpus_per_node
    sp = _sequence_parallel_size(cfg)
    if gpus % sp:
        return (f"The split gives the student {gpus} GPUs ({topology.describe()}), but its ulysses sequence "
                f"parallel size is {sp}, which does not divide them. Choose another split with "
                f"{TEACHER_GPUS_KEY} or {TEACHER_NODES_KEY}, or another sequence parallel size.")
    ranks = gpus // sp
    answers = mini_batch * n
    if answers % ranks == 0:
        return None
    # mini_batch x n divides by ranks exactly when mini_batch is a multiple of this step.
    step = ranks // math.gcd(n, ranks)
    lower = mini_batch // step * step
    candidates = [size for size in (lower, lower + step) if size >= 1]
    nearest = min(candidates, key=lambda size: (abs(size - mini_batch), -size))
    key = _TRAIN_BATCH_KEY if select(cfg, _TRAIN_BATCH_KEY) == mini_batch else _MINI_BATCH_KEY
    return (f"The split gives the student {ranks} data-parallel ranks ({topology.describe()}). One actor update "
            f"holds {_MINI_BATCH_KEY} {mini_batch} x {_ROLLOUT_N_KEY} {n} = {answers} answers, which do not divide "
            f"evenly among {ranks} ranks. verl would pad the batch and then fail an assert in train_mini_batch at "
            f"the first update, after a whole rollout and teacher pass. Set {key}={nearest}, whose {nearest * n} "
            f"answers divide among {ranks} ranks, or choose another split with {TEACHER_GPUS_KEY} or "
            f"{TEACHER_NODES_KEY}.")


@dataclass
class DistillSetup:
    """What ``finish_distillation`` derived. The launcher prints it and records it in env.json.

    ``topology_error`` takes the place of ``topology`` when the split does not fit the job, or when
    the actor's mini-batch does not divide among the student's ranks under that split. The launcher
    reports it together with the GPU count, which ``--prepare-only`` leaves to the job.
    ``max_model_len`` maps each teacher entry whose null value was filled to the value it got.
    """

    topology: DistillTopology | None = None
    topology_error: str | None = None
    max_model_len: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"topology": None if self.topology is None else self.topology._asdict(),
                "topology_error": self.topology_error, "teacher_max_model_len": dict(self.max_model_len)}


def _writable(cfg: Any):
    """A context in which new keys may be added to ``cfg``. A plain dict needs none."""
    try:
        from omegaconf import DictConfig, open_dict
    except ImportError:
        return contextlib.nullcontext()
    return open_dict(cfg) if isinstance(cfg, DictConfig) else contextlib.nullcontext()


def _put(cfg: Any, dotted: str, value: Any) -> None:
    *parents, key = dotted.split(".")
    node = cfg
    for part in parents:
        node = node[part]
    node[key] = value


def finish_distillation(cfg: Any, *, nnodes: int, gpus_per_node: int,
                        user_overrides: Iterable[str] = ()) -> DistillSetup | None:
    """Fill what a run with ``distillation.enabled`` needs from the launcher; None when distillation is off.

    1. Each teacher entry whose ``inference.max_model_len`` is null gets the student's prompt
       length + response length + 1, which is the room verl demands. With null, vLLM would take
       the teacher's ``max_position_embeddings`` (262,144 for the 2507 models) and reserve KV
       cache for one sequence of that length.
    2. The job's GPUs are split. The mode overrides gave the student every GPU of the job
       (``trainer.nnodes`` x ``trainer.n_gpus_per_node``). verl, however, always runs the teacher
       in a pool of its own when distillation is on (``_init_resource_pool_mgr`` in
       verl/trainer/ppo/v1/trainer_base.py), and refuses a run whose pools together ask for more
       GPUs than Ray has. ``orbit.distill.teacher_gpus_per_node`` or ``orbit.distill.teacher_nodes``
       decide the split (``split_distill_topology``). The launcher then writes ``trainer.nnodes``,
       ``trainer.n_gpus_per_node``, ``distillation.nnodes`` and ``distillation.n_gpus_per_node``.
    3. The answers of one actor update must divide evenly among the student's data-parallel ranks
       under that split (``batch_split_problem``), or verl fails at the first update.

    ``nnodes`` and ``gpus_per_node`` describe the whole job, as the mode overrides give them. A
    split that does not fit, or that the batch does not divide among, is returned in
    ``topology_error`` and leaves the trainer keys alone.

    Raises DistillTopologyError when ``user_overrides`` set ``distillation.nnodes`` or
    ``distillation.n_gpus_per_node``. The launcher would replace that value, and the person who
    typed it would silently not get the teacher they asked for.
    """
    from orbit_verl.distill.preflight import is_enabled, required_teacher_context, select, teacher_entries

    if not is_enabled(cfg):
        return None
    given = [key for key in _TEACHER_POOL_KEYS if common.override_value(user_overrides, key) is not None]
    if given:
        raise DistillTopologyError(
            f"{' and '.join(given)} set the teacher's GPUs directly, but the launcher derives them from the "
            f"job's topology. Set {TEACHER_GPUS_KEY}=N (N GPUs of every node) or {TEACHER_NODES_KEY}=N (N whole "
            "nodes) instead, with a leading + unless the experiment YAML has a orbit.distill block.")

    setup = DistillSetup()
    need = required_teacher_context(cfg)
    with _writable(cfg):
        for name, entry in teacher_entries(cfg).items():
            inference = select(entry, "inference")
            if inference is not None and select(inference, "max_model_len") is None:
                inference["max_model_len"] = need
                setup.max_model_len[name] = need
    try:
        topology = split_distill_topology(
            nnodes=nnodes, gpus_per_node=gpus_per_node, teacher_gpus_per_node=select(cfg, TEACHER_GPUS_KEY),
            teacher_nodes=select(cfg, TEACHER_NODES_KEY))
    except DistillTopologyError as exc:
        setup.topology_error = str(exc)
        return setup
    setup.topology_error = batch_split_problem(cfg, topology)
    if setup.topology_error is not None:
        return setup
    setup.topology = topology
    with _writable(cfg):
        _put(cfg, "trainer.nnodes", topology.student_nnodes)
        _put(cfg, "trainer.n_gpus_per_node", topology.student_gpus_per_node)
        _put(cfg, "distillation.nnodes", topology.teacher_nnodes)
        _put(cfg, "distillation.n_gpus_per_node", topology.teacher_gpus_per_node)
    return setup


def put_runtime_env(cfg: Any, env_vars: Mapping[str, str]) -> None:
    """Merge ``env_vars`` into ``ray_kwargs.ray_init.runtime_env.env_vars`` of ``cfg``.

    verl's config is in struct mode, which forbids new keys, so the write happens under
    ``open_dict``. Values already in the YAML are kept unless the job env sets the same name.
    """
    from omegaconf import open_dict

    bad = {key: type(value).__name__ for key, value in env_vars.items() if not isinstance(value, str)}
    if bad:
        raise TypeError(f"Ray runtime env values must be strings; got {bad}.")
    with open_dict(cfg):
        ray_init = cfg.ray_kwargs.ray_init
        if ray_init.get("runtime_env") is None:
            ray_init.runtime_env = {}
        if ray_init.runtime_env.get("env_vars") is None:
            ray_init.runtime_env.env_vars = {}
        for key, value in env_vars.items():
            # omegaconf would read "${...}" inside a value as an interpolation; the backslash keeps
            # it literal, and reading the key back returns the original string.
            ray_init.runtime_env.env_vars[key] = value.replace("${", "\\${")


def to_yaml(cfg: Any) -> str:
    """The config as YAML with interpolations resolved, so the saved file shows real paths."""
    from omegaconf import OmegaConf

    try:
        return OmegaConf.to_yaml(cfg, resolve=True)
    except Exception:  # noqa: BLE001 - an unresolvable value should not hide the rest of the config
        return OmegaConf.to_yaml(cfg, resolve=False)
