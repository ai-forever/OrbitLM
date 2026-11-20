"""Loads a trainer experiment config: one flat YAML mapping whose keys are the options of train.py.

The values become the defaults of train.parse_args, so a command-line option overrides the file.

Environment variables. A string value may name environment variables as ``${NAME}``, for example
``stage1_root: ${ORBIT_DATA}/ctx16k/streams``. They are expanded in expand_env_vars and nowhere
else. An unset variable stops the load with an error that names the variable and the key that
uses it, because a path with a missing part would point the run at the wrong place. README.md at
the repository root lists the variables of the release configs.

Relative paths. A relative value of a key in PATH_KEYS resolves against the folder of the config
file, so a config can point at files of the repository (``../../../src/orbit/...``).

This file is also copied into every HF export, because modelling_ouro.py reads the default FA3
kernel path from it. It therefore imports only the standard library, and PyYAML only when a
config is loaded.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


PATH_KEYS = {
    "save_dir",
    "resume_from",
    "init_from_hf",
    "model_config",
    "tokenizer_config",
    "attn_kernel_path",
    "clearml_env",
    "tensorboard_logdir",
    "tensorboard_root",
    # In-training evaluation (train_eval.py). eval_python is deliberately not here: resolving it
    # would follow a venv's python symlink out of the venv.
    "eval_suite",
    "eval_harness_path",
    "eval_cache_dir",
    "eval_tokenizer_source",
}

INT_KEYS = {
    "batch_size",
    "grad_accum",
    "max_steps",
    "epochs",
    "num_workers",
    "prefetch_factor",
    "predownload",
    "seed",
    "stage1_epoch_samples",
    "stage1_shuffle_seed",
    "stage1_shuffle_block_size",
    "stage1_num_canonical_nodes",
    "seq_len",
    "block_size",
    "total_ut_steps",
    "tokens_per_step",
    "token_budget",
    "log_every",
    "save_every",
    "lm_loss_chunk_size",
    "lr_warmup_steps",
    "eval_every",
    "eval_max_model_len",
}

FLOAT_KEYS = {
    "lr",
    "beta",
    "rope_theta",
    "weight_decay",
    "max_grad_norm",
    "loader_timeout",
    "lr_warmup_start_factor",
    "lr_warmup_start_lr",
    "lr_min",
    "lr_min_factor",
    "eval_gpu_memory_utilization",
    "eval_timeout_minutes",
    "eval_total_timeout_minutes",
}

# Matches ${NAME}. A bare $NAME is left alone on purpose, so that a dollar sign that is part of a
# value is never expanded by accident.
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def default_attn_kernel_path_from_env() -> Path | None:
    """Returns the local FA3 kernel build named by ORBIT_ATTN_KERNEL_PATH, or None.

    None means that the kernel is downloaded from the HF Hub. The trainer uses this value when
    the config sets no attn_kernel_path, and the model code uses it when its config carries no
    kernel path of its own.
    """
    raw = os.environ.get("ORBIT_ATTN_KERNEL_PATH")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def load_experiment_config(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    payload = _load_yaml_text(path.read_text(encoding="utf-8"), source=path)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(f"Experiment config must be a mapping: {path}")
    payload = expand_env_vars(payload, source=path)
    normalized = _normalize_config_values(payload, base_dir=path.parent)
    normalized["config"] = path
    return normalized


def _load_yaml_text(text: str, *, source: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(f"PyYAML is needed to read {source}.") from exc
    return yaml.safe_load(text)


def expand_env_vars(value: Any, *, source: Path, key: str = "") -> Any:
    """Replaces every ${NAME} in the string values of a loaded config, at any depth.

    This is the only place where the trainer expands environment variables in a config.
    """
    if isinstance(value, dict):
        return {
            name: expand_env_vars(item, source=source, key=f"{key}.{name}" if key else str(name))
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [
            expand_env_vars(item, source=source, key=f"{key}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    def substitute(match: re.Match) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if not resolved:
            raise KeyError(
                f"{source}: '{key}' uses the environment variable {name}, which is not set. "
                f"Set {name} before starting the run; README.md at the repository root lists the variables."
            )
        return resolved

    return _ENV_REFERENCE.sub(substitute, value)


def _normalize_config_values(payload: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    normalized = {str(key).replace("-", "_"): value for key, value in payload.items()}

    if "stage1_streams" in normalized:
        normalized["stage1_streams"] = normalize_stage1_streams(
            normalized["stage1_streams"]
        )
    if "stage1_dataset_proportions" in normalized:
        normalized["stage1_dataset_proportions"] = normalize_stage1_dataset_proportions(
            normalized["stage1_dataset_proportions"]
        )

    for key in PATH_KEYS:
        if key not in normalized:
            continue
        normalized[key] = _coerce_path(normalized[key], base_dir=base_dir)

    for key in INT_KEYS:
        if key in normalized and normalized[key] is not None:
            normalized[key] = int(normalized[key])
    for key in FLOAT_KEYS:
        if key in normalized and normalized[key] is not None:
            normalized[key] = float(normalized[key])

    if "attn_kernel_path" not in normalized:
        normalized["attn_kernel_path"] = default_attn_kernel_path_from_env()

    return normalized


def normalize_stage1_streams(value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [(str(name), str(path)) for name, path in value.items()]
    if isinstance(value, list):
        entries: list[tuple[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name") or item.get("key") or item.get("dataset")
                path = item.get("path") or item.get("folder") or item.get("local")
                if name is None or path is None:
                    raise ValueError(
                        "stage1_streams list entries must contain name and path."
                    )
                entries.append((str(name), str(path)))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                entries.append((str(item[0]), str(item[1])))
            else:
                raise TypeError(f"Unsupported stage1_streams entry: {item!r}")
        return entries
    raise TypeError("stage1_streams must be a mapping or a list.")


def normalize_stage1_dataset_proportions(value: Any) -> dict[str, dict[str, float]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("stage1_dataset_proportions must be a mapping.")
    normalized: dict[str, dict[str, float]] = {}
    for name, spec in value.items():
        if isinstance(spec, dict):
            if "proportion" not in spec:
                raise ValueError(f"Missing proportion for dataset {name!r}.")
            proportion = spec["proportion"]
        else:
            proportion = spec
        normalized[str(name)] = {"proportion": float(proportion)}
    return normalized


def _coerce_path(value: Any, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
