"""Small helpers shared by the model tool, the data tool, the launcher and the run scripts.

Everything here uses only the standard library. A python without verl or a recent transformers can
therefore import it too. The inference package does, through ``orbit_verl.launch.env``.

What lives here:
- where runtime outputs go: the runs root (``resolve_runs_root``) and the per-host cache layout;
- the names that the launcher derives when the user does not give them (``model_name_for_export``,
  ``default_run_name``);
- reading single values from hydra overrides and experiment YAMLs without hydra;
- the manifest files that the prepare tools write and the launcher checks;
- the tokenizer fingerprint that ties a prepared dataset to the tokenizer it was filtered with.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import socket
from pathlib import Path
from typing import Any, Iterable, Mapping

# The environment variable that names the runs root when neither the command line nor the config
# does. Every output and cache goes under the runs root: prepared model dirs, datasets,
# checkpoints, logs and compiler caches.
RUNS_ROOT_ENV = "ORBIT_VERL_RUNS"

# File names of the manifests. The model tool writes MODEL_MANIFEST into the prepared model dir;
# the data tool writes DATA_MANIFEST next to train.parquet and val.parquet.
MODEL_MANIFEST = "orbit_prepare.json"
DATA_MANIFEST = "manifest.json"
MODEL_MANIFEST_KIND = "orbit_verl_model"
DATA_MANIFEST_KIND = "orbit_verl_data"
MANIFEST_VERSION = 1


class RunsRootError(ValueError):
    """No runs root was given: not on the command line, not in the config, not in the environment."""


def runs_root_help(sources: Iterable[str] = ("--runs-root PATH", "orbit.runs_root=PATH")) -> str:
    """The message shown when no runs root is set, listing the ways the calling tool accepts one.

    There is deliberately no default directory, so a run never writes its checkpoints and caches
    into a directory that nobody chose for it.
    """
    ways = ", ".join(sources)
    return (
        "No runs root is set. Every output of a run goes under it: prepared models, datasets, "
        "checkpoints, logs and caches. Pick a directory of your own and either export it once:\n"
        f"    export {RUNS_ROOT_ENV}=/path/to/your/verl_runs\n"
        f"or pass it to this command ({ways})."
    )


def resolve_runs_root_with_source(cli: str | os.PathLike | None = None,
                                  config: str | os.PathLike | None = None,
                                  environ: Mapping[str, str] | None = None) -> tuple[str, str]:
    """The runs root and where it came from, in the order command line > config > environment.

    ``cli`` is the value of a ``--runs-root`` option, ``config`` the config's ``orbit.runs_root``,
    and ``environ`` defaults to ``os.environ``. Empty values count as unset. The result is an
    absolute path with ``~`` expanded: the launcher changes into the run dir before training, so a
    relative root would point somewhere else afterwards. Raises ``RunsRootError`` when no source
    gives a value.
    """
    environ = os.environ if environ is None else environ
    for value, source in ((cli, "--runs-root"), (config, "orbit.runs_root"),
                          (environ.get(RUNS_ROOT_ENV), RUNS_ROOT_ENV)):
        if value is not None and str(value).strip():
            return os.path.abspath(os.path.expanduser(str(value).strip())), source
    raise RunsRootError(runs_root_help())


def resolve_runs_root(cli: str | os.PathLike | None = None, config: str | os.PathLike | None = None,
                      environ: Mapping[str, str] | None = None) -> str:
    """The runs root alone; see ``resolve_runs_root_with_source``."""
    return resolve_runs_root_with_source(cli, config, environ)[0]


def cache_env(runs_root: str, host: str | None = None) -> dict[str, str]:
    """Return the cache variables for one host, as plain strings.

    Every cache goes to ``<runs_root>/cache/<host>/...``, so a run keeps all of its outputs under
    the runs root and writes nothing into ``~/.cache``. Each host gets its own subdirectory because
    Ray workers on several nodes would otherwise compile into the same files on the shared
    filesystem at the same time.

    ``XDG_CACHE_HOME`` also moves the Hugging Face home, since ``HF_HOME`` defaults to
    ``$XDG_CACHE_HOME/huggingface``.

    ``CUTE_DSL_CACHE_DIR`` is the kernel cache of NVIDIA's CuTe DSL (the nvidia_cutlass_dsl
    package in the env). Without it, the vLLM engines of a run create
    ``/tmp/<user>/cutlass_python_cache``.

    ``VLLM_USE_FLASHINFER_SAMPLER=0`` is here too, although it is not a cache setting. vLLM's default
    FlashInfer sampler compiles its CUDA kernels with ``nvcc`` on first use. In an environment without
    ``nvcc`` every vLLM engine would then die at start-up with "Ninja build failed". With this variable
    vLLM samples with its own PyTorch code instead, which needs no compiler.

    ``VLLM_NO_USAGE_STATS=1`` stops vLLM from appending a usage record to
    ``~/.config/vllm/usage_stats.json`` and from trying to send it over the network. Every vLLM
    engine of a run would otherwise append to it.

    The Hugging Face offline switches are not set here, so the caller's environment decides. A run
    downloads nothing from the Hub, because the model, the teacher and every data file are local
    paths. On a machine without internet access, export ``HF_HUB_OFFLINE=1`` and
    ``TRANSFORMERS_OFFLINE=1`` before the launch, and every process of the run inherits them.
    """
    if not runs_root:
        raise RunsRootError(runs_root_help())
    host = host or socket.gethostname()
    base = os.path.join(str(runs_root), "cache", host)
    return {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_NO_USAGE_STATS": "1",
        "HF_MODULES_CACHE": os.path.join(base, "hf_modules"),
        "HF_DATASETS_CACHE": os.path.join(base, "hf_datasets"),
        "XDG_CACHE_HOME": os.path.join(base, "xdg"),
        "VLLM_CACHE_ROOT": os.path.join(base, "vllm"),
        "TRITON_CACHE_DIR": os.path.join(base, "triton"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(base, "inductor"),
        "FLASHINFER_WORKSPACE_BASE": os.path.join(base, "flashinfer"),
        "CUTE_DSL_CACHE_DIR": os.path.join(base, "cute_dsl"),
    }


def apply_cache_env(runs_root: str | None = None) -> dict[str, str]:
    """Set the cache variables in ``os.environ`` unless the caller already set them.

    Call this before importing transformers, datasets or vllm: they read some of these variables
    once, at import time. Returns the values that are in effect afterwards.

    Without ``runs_root`` the root comes from ``ORBIT_VERL_RUNS``. The launcher exports that
    variable as soon as it has resolved the root, so tool functions that it calls in-process find
    the same root. Raises ``RunsRootError`` when neither is set.
    """
    if not runs_root and RUNS_ROOT_ENV not in os.environ and all(key in os.environ for key in CACHE_ENV_KEYS):
        # Whoever set these variables already chose a root, and this call would change nothing.
        return {key: os.environ[key] for key in CACHE_ENV_KEYS}
    wanted = cache_env(runs_root or resolve_runs_root())
    for key, value in wanted.items():
        os.environ.setdefault(key, value)
    return {key: os.environ[key] for key in wanted}


# The names that cache_env sets; the values do not matter here.
CACHE_ENV_KEYS = tuple(cache_env("/", host="host"))


# Characters that a derived directory name keeps; every other run of characters becomes "_".
_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(text: str) -> str:
    """``text`` reduced to letters, digits, ``.``, ``_`` and ``-``, usable as one directory name.

    Leading dots and underscores are dropped, so the name can never be hidden, ``.`` or ``..``.
    """
    name = _NAME_UNSAFE.sub("_", str(text)).strip("._")
    if not name:
        raise ValueError(f"{text!r} leaves no usable characters for a directory name.")
    return name


def model_name_for_export(export: str | os.PathLike) -> str:
    """The prepared model dir name that ``--export`` gets when no ``--model-name`` is given.

    It is ``<parent dir name>__<export dir name>`` of the export, for example
    ``orbit_sft_v5__step_00007000`` for ``.../orbit_sft_v5/step_00007000``.
    The two last path parts name the checkpoint when exports are laid out per run and step, and the
    same export always gets the same name. Symlinks are resolved first, so a ``latest`` link is named
    after the step it points to. The run scripts and the smoke script always pass ``--model-name``,
    because the flat ``$ORBIT_EXPORTS`` folders of the README would otherwise take the exports
    folder's name as their first part.

    Two different exports can still end in the same two parts, for example a copy of a checkpoint
    in another folder. The launcher catches that case: the prepared dir records the export it was
    built from, and a dir built from another export is refused instead of reused.
    """
    path = Path(export).expanduser().resolve()
    parts = [part for part in (path.parent.name, path.name) if part]
    return safe_name("__".join(parts))


def config_stem(config: str) -> str:
    """The name of an experiment config: the name itself, or the file stem for a YAML path."""
    text = str(config)
    if "/" in text or text.endswith((".yaml", ".yml")):
        return Path(text).stem
    return text


def default_run_name(config: str, model_name: str) -> str:
    """The run name when none is given: ``<config>__<model name>``.

    Including the model keeps runs of one config on different checkpoints in different run dirs.
    Without that, the second run would resume from the first one's checkpoints.
    """
    return safe_name(f"{config_stem(config)}__{model_name}")


def override_value(overrides: Iterable[str], key: str) -> str | None:
    """The value of ``key=value`` in a list of hydra overrides, last one wins, quotes removed."""
    found = None
    for item in overrides:
        name, sep, value = str(item).lstrip("+").partition("=")
        if sep and name == key:
            found = value.strip().strip("'\"")
    return found


def yaml_orbit_value(path: str | os.PathLike, key: str) -> str | None:
    """Read a plain scalar ``orbit.<key>`` from an experiment YAML without a YAML parser.

    Only the experiment file itself is read, not the files it inherits. The callers need plain
    strings written in block style directly under ``orbit:`` (``run_name``, ``runs_root``). A
    value with an interpolation, ``???`` or ``null`` is not usable without hydra and gives None.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    inside = False
    for line in lines:
        if re.match(r"^orbit:\s*(#.*)?$", line):
            inside = True
            continue
        if inside and line and not line[0].isspace() and not line.startswith("#"):
            inside = False
        if inside:
            match = re.match(rf"^  {re.escape(key)}:\s*([^#\s][^#]*?)\s*(#.*)?$", line)
            if match:
                value = match.group(1).strip().strip("'\"")
                return None if "${" in value or value in ("???", "null", "~") else value
    return None


def sha256_file(path: str | os.PathLike) -> str:
    """Hex sha256 of a file's bytes. Symlinks are followed, so a linked file hashes like its target."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_sha256(tokenizer_dir: str | os.PathLike) -> str:
    """Fingerprint of the tokenizer in a directory, used to match a dataset to a model dir.

    The prepared model dir symlinks the export's tokenizer files, so comparing paths would call two
    identical tokenizers different. Hashing the file content treats them as equal. ``tokenizer.json``
    defines the whole tokenizer when it exists; otherwise ``vocab.json`` plus ``merges.txt`` do.
    """
    root = Path(tokenizer_dir)
    if (root / "tokenizer.json").exists():
        return sha256_file(root / "tokenizer.json")
    parts = [root / "vocab.json", root / "merges.txt"]
    if not all(p.exists() for p in parts):
        raise FileNotFoundError(
            f"No tokenizer.json, and no vocab.json + merges.txt, in {root}; cannot fingerprint it."
        )
    digest = hashlib.sha256()
    for part in parts:
        digest.update(sha256_file(part).encode())
    return digest.hexdigest()


def write_manifest(path: str | os.PathLike, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write a manifest as JSON and return what was written.

    The file is written to a temporary name and then renamed. A reader therefore never sees a
    half-written manifest, which matters because the launcher treats the manifest as the marker
    that a prepared directory is complete.
    """
    data = {
        "kind": kind,
        "version": MANIFEST_VERSION,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        **dict(payload),
    }
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return data


def read_manifest(path: str | os.PathLike, kind: str) -> dict[str, Any]:
    """Read a manifest and check that it is the expected kind and version."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Manifest {target} does not exist; the directory was not prepared.")
    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("kind") != kind:
        raise ValueError(f"{target} has kind {data.get('kind')!r}, expected {kind!r}.")
    if data.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"{target} has manifest version {data.get('version')!r}; this code reads version "
            f"{MANIFEST_VERSION}. Re-run the prepare tool with --force."
        )
    return data
