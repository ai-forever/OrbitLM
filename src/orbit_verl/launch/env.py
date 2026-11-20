"""Environment handling for the launcher: node topology, launcher cleanup, node env and job env.

The launcher deals with four groups of environment variables, and each group goes to a different
place:

- launcher variables (RANK, WORLD_SIZE, MASTER_ADDR, TORCHELASTIC_*, ...): read once to learn the
  node topology, then removed from ``os.environ`` before anything starts Ray;
- node env (cache directories and vLLM switches from ``orbit_verl.common.cache_env``): set in
  ``os.environ`` on every node, never in the Ray runtime env, because the paths contain the host name;
- job env (PYTHONPATH, the runs root, verl plugin switches, metrics file, RLTT knobs, ``orbit.env``):
  identical on all nodes, so it goes into ``os.environ`` and into
  ``ray_kwargs.ray_init.runtime_env.env_vars``;
- secrets (``CLEARML_*`` from ``orbit.clearml_env``): ``os.environ`` only, because verl prints the
  ``ray.init`` arguments and uploads the whole config to ClearML.

Only the standard library is used here, so the module also imports in a python without verl. The
inference package imports ``remove_launcher_env`` from here, so its name and behaviour stay stable.
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterable, Mapping, NamedTuple

from orbit_verl import common

# Variables that torchrun (torchelastic) and similar distributed launchers export for the process they start.
# Ray processes inherit the environment of whoever starts them. If these stay set, every Ray actor
# looks like a child of the elastic agent, and torch.distributed inside verl's workers waits for a
# TCPStore server that nobody starts. Check a change to this list with a 2-node launch: the nodes must
# form one Ray cluster and run NCCL across both. TORCHELASTIC_* variables are removed by prefix as well.
LAUNCHER_ENV_VARS = frozenset({
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
    "ROLE_RANK", "ROLE_NAME", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
})
LAUNCHER_ENV_PREFIXES = ("TORCHELASTIC_",)

# Value of VERL_USE_EXTERNAL_MODULES and VERL_ENGINE_VENDOR when the OrbitLM plugin is enabled.
PLUGIN_MODULE = "orbit_verl.plugin"
ENGINE_VENDOR = "orbit"

# Job env keys that a user may not set through orbit.env. Launcher variables would bring back the
# torch.distributed hang described above, and CLEARML_* values are secrets that must not reach the
# config or the Ray runtime env.
_FORBIDDEN_USER_PREFIXES = ("CLEARML_",) + LAUNCHER_ENV_PREFIXES

# Names whose values are hidden when the launcher prints or saves the env plan.
_SECRET_NAME = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL)", re.IGNORECASE)


class Topology(NamedTuple):
    """Node topology of this process, read from the distributed launcher's variables.

    It unpacks as ``(rank, world_size, master_addr)``. ``world_size`` is the number of nodes.
    """

    rank: int
    world_size: int
    master_addr: str | None

    @property
    def mode(self) -> str:
        """``local`` for one node (a terminal or a one-node job), else ``head`` or ``worker``."""
        if self.world_size <= 1:
            return "local"
        return "head" if self.rank == 0 else "worker"


def _int_var(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name}={value!r} is not an integer.") from exc


def read_topology(env: Mapping[str, str] | None = None) -> Topology:
    """Return the node rank, the node count and the head address from the launcher variables.

    The launcher runs as one process per node. RANK is therefore the node rank and WORLD_SIZE the
    number of nodes, as torchrun sets them with ``--nproc_per_node 1``. Without these variables
    (a terminal), the result is one node of rank 0.
    """
    env = os.environ if env is None else env
    rank = _int_var(env, "RANK", 0)
    world_size = _int_var(env, "WORLD_SIZE", 1)
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be at least 1, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK {rank} is outside 0..{world_size - 1} (WORLD_SIZE={world_size}).")
    master_addr = env.get("MASTER_ADDR") or None
    return Topology(rank, world_size, master_addr)


def is_launcher_var(name: str) -> bool:
    """True for a variable that must not reach Ray (see ``LAUNCHER_ENV_VARS``)."""
    return name in LAUNCHER_ENV_VARS or name.startswith(LAUNCHER_ENV_PREFIXES)


def strip_launcher_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of ``env`` without the distributed launcher's variables."""
    return {key: value for key, value in env.items() if not is_launcher_var(key)}


def remove_launcher_env(environ: Any = None) -> list[str]:
    """Delete the launcher variables from ``environ`` (default ``os.environ``) in place.

    Returns the removed names, so the launcher can report them without printing values.
    """
    environ = os.environ if environ is None else environ
    removed = sorted(key for key in list(environ) if is_launcher_var(key))
    for key in removed:
        del environ[key]
    return removed


def node_env(runs_root: str) -> dict[str, str]:
    """Cache directories and vLLM switches for this host, under the resolved runs root.

    This is ``common.cache_env``. It is a separate name so that the launcher code reads as the
    design describes it: node env goes to ``os.environ`` on each node and never into the job env.
    """
    return common.cache_env(runs_root)


def caller_set_node_env(environ: Mapping[str, str] | None = None) -> frozenset[str]:
    """The node env variables that already have a value, read before anything imports verl.

    The launcher calls this first thing and hands the result to ``apply_node_env``. Values set by
    the user before the launch can then be told apart from values that a library set on import.
    """
    environ = os.environ if environ is None else environ
    return frozenset(key for key in common.CACHE_ENV_KEYS if environ.get(key))


def apply_node_env(runs_root: str, environ: Any = None, keep: Iterable[str] | None = None) -> dict[str, str]:
    """Set the node env in ``environ`` and return what is in effect.

    A value that the user set before the launch wins, so a user can still point a cache somewhere
    else by exporting it first. ``keep`` names those variables (see ``caller_set_node_env``); every
    other variable gets the value under the runs root, even if something set it in the meantime.
    That matters for TORCHINDUCTOR_CACHE_DIR: ``import verl``, which composing the config does,
    makes torch set it to /tmp/torchinductor_<user>, and the Ray workers would inherit that value.

    Without ``keep``, every value that is already in ``environ`` is kept.
    """
    environ = os.environ if environ is None else environ
    wanted = node_env(runs_root)
    for key, value in wanted.items():
        if keep is None or key in keep:
            environ.setdefault(key, value)
        else:
            environ[key] = value
    return {key: environ[key] for key in wanted}


def _prepend_path(entry: str, current: str | None) -> str:
    """Put ``entry`` first in a PATH-style list and drop later duplicates of it."""
    parts = [entry] + [p for p in (current or "").split(os.pathsep) if p and p != entry]
    return os.pathsep.join(parts)


def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from a DictConfig or from nested dicts without importing omegaconf."""
    node = cfg
    for part in dotted.split("."):
        if node is None:
            return default
        try:
            node = node[part]
        except (KeyError, TypeError, AttributeError):
            try:
                node = getattr(node, part)
            except AttributeError:
                return default
    return default if node is None else node


def job_env(cfg: Any, repo_root: str, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The env that every process of the run needs, identical on all nodes, as plain strings.

    ``cfg`` is the composed config (only the ``orbit`` block is read). ``base_env`` supplies the
    current PYTHONPATH, which is kept after ``<repo_root>/src``; it defaults to ``os.environ``.
    Raises ValueError when ``orbit.env`` tries to set a launcher variable or a ClearML secret.
    """
    base_env = os.environ if base_env is None else base_env
    orbit = _cfg_get(cfg, "orbit")
    if orbit is None:
        raise ValueError("The config has no orbit block; it must inherit configs/verl/orbit_base.yaml.")

    train_exit_gate = _cfg_get(cfg, "orbit.rltt.train_exit_gate", False)
    if not isinstance(train_exit_gate, bool):
        # A string such as "no" would count as true in Python, and the two values train different
        # objectives, so only a real YAML boolean is accepted.
        raise ValueError(f"orbit.rltt.train_exit_gate must be true or false, got {train_exit_gate!r}.")

    src = os.path.join(os.path.abspath(repo_root), "src")
    env: dict[str, str] = {
        # Ray workers start fresh interpreters, and they import orbit_verl (agent loop, reward,
        # plugin) by name. They only find it if the repo's src/ is on their path.
        "PYTHONPATH": _prepend_path(src, base_env.get("PYTHONPATH")),
        # verl's "file" logger writes one JSON line per step to this path.
        "VERL_FILE_LOGGER_PATH": os.path.join(str(_cfg_get(cfg, "orbit.run_dir")), "metrics.jsonl"),
        # The RLTT knobs travel as env vars because verl's actor config cannot get new keys.
        "ORBIT_RLTT_PROGRESSIVE_ALPHA": str(_cfg_get(cfg, "orbit.rltt.progressive_alpha", 1.0)),
        "ORBIT_RLTT_CHUNK_SIZE": str(_cfg_get(cfg, "orbit.rltt.chunk_size", 512)),
        "ORBIT_RLTT_TRAIN_EXIT_GATE": "true" if train_exit_gate else "false",
    }
    runs_root = _cfg_get(cfg, "orbit.runs_root")
    if runs_root is not None:
        # Every process of the run sees the root the launcher resolved. Tool functions that fall
        # back to this variable for their caches then write under the same root on every node.
        env[common.RUNS_ROOT_ENV] = str(runs_root)
    if bool(_cfg_get(cfg, "orbit.plugin", True)):
        # verl imports the modules listed here at the end of `import verl` in every process, which
        # registers our engine; the vendor name then selects that engine over the stock one.
        env["VERL_USE_EXTERNAL_MODULES"] = PLUGIN_MODULE
        env["VERL_ENGINE_VENDOR"] = ENGINE_VENDOR

    user = _cfg_get(cfg, "orbit.env", {}) or {}
    for key in user:
        key = str(key)
        if key == common.RUNS_ROOT_ENV:
            # A second value here would split the run: the launcher's own outputs would go under
            # orbit.runs_root, and the Ray processes would use this one for their caches.
            raise ValueError(f"orbit.env may not set {key}. Set the runs root with --runs-root or "
                             "orbit.runs_root instead.")
        if key.startswith(_FORBIDDEN_USER_PREFIXES) or key in LAUNCHER_ENV_VARS:
            raise ValueError(
                f"orbit.env may not set {key}. Launcher variables break torch.distributed inside "
                "Ray, and CLEARML_* credentials belong in the dotenv file named by orbit.clearml_env."
            )
        value = user[key]
        env[key] = "" if value is None else str(value)
    return env


def load_dotenv(path: str | os.PathLike) -> dict[str, str]:
    """Parse a dotenv file into a dict without touching ``os.environ``.

    Supported syntax: ``KEY=value`` lines, an optional ``export`` prefix, blank lines, full-line
    ``#`` comments, single or double quotes around the value, and `` #`` comments after an unquoted
    value. Variable references such as ``$HOME`` are not expanded, so a secret containing ``$`` is
    read exactly as written.
    """
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, sep, value = line.partition("=")
            key = key.strip()
            if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"{path}:{lineno}: expected KEY=value, got {raw.rstrip()!r}")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                quote, value = value[0], value[1:-1]
                if quote == '"':
                    # Double quotes allow the two escapes that shells and python-dotenv also honor.
                    value = value.replace('\\"', '"').replace("\\n", "\n")
            else:
                # An unquoted value ends at the first " #", as in shell.
                value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
            values[key] = value
    return values


def is_secret_name(name: str) -> bool:
    """True for names that look like credentials; their values are never printed or saved."""
    return name.startswith("CLEARML_") or bool(_SECRET_NAME.search(name))


def redact(env: Mapping[str, str]) -> dict[str, str]:
    """Copy of ``env`` with credential-like values replaced by ``***``."""
    return {key: ("***" if is_secret_name(key) else value) for key, value in env.items()}
