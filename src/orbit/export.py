"""Exports a trainer checkpoint to a HuggingFace model folder, on CPU.

It merges the FSDP shards of all rank files of one checkpoint step into full fp32 tensors and
writes them with config.json, the tokenizer and the model code, so the folder loads with
``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``. Every next stage starts
from such a folder through ``init_from_hf``. No GPU and no process group of the training job are
needed, and the world size of the checkpoint does not matter.

    python -m orbit.export <run dir | step dir | latest.txt> <output dir> \\
        --model-config src/orbit/ouro_1.4b_qwen3_config.json \\
        --tokenizer ${ORBIT_TOKENIZERS}/tokenizer_qwen3_orbit \\
        [--chat-export]

The architecture, rope_theta and max_position_embeddings come from the checkpoint's meta.json;
--model-config only fills keys that the checkpoint does not record. The vocabulary size and the
special-token ids come from the tokenizer folder, which must be the tokenizer the run trained with.
Pass --chat-export for SFT and later checkpoints: the export then ships <|im_end|> as eos and a
generation_config.json with the chat stop ids. Base (pretraining) exports keep <|endoftext|>.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

try:
    from safetensors.torch import save_file as save_safetensors_file
except Exception:  # pragma: no cover - optional dependency
    save_safetensors_file = None

try:
    from torch.distributed.tensor import DTensor as TorchDTensor
except Exception:  # pragma: no cover - optional dependency
    TorchDTensor = None

from .checkpointing import (
    read_checkpoint_metadata,
    resolve_checkpoint_dir_from_ref,
)
from .configuration_ouro import OuroConfig
from .modelling_ouro import OuroForCausalLM
from .runtime import TokenizerConfig, load_model_config


# The files that make an export folder loadable with trust_remote_code. modelling_ouro.py imports
# only configuration_ouro.py from this package, so an export needs no other module.
REMOTE_CODE_FILES = ("configuration_ouro.py", "modelling_ouro.py")
# Model config keys that hold a path on the training machine. Such a path means nothing on another
# machine, so neither config.json nor export_info.json of an export carries these keys.
MACHINE_LOCAL_CONFIG_KEYS = ("_attn_kernel_repo_path",)
_CPU_PROCESS_GROUP_FILE: Optional[Path] = None
_CPU_PROCESS_GROUP_CREATED = False
_SHARDED_TENSOR_OFFLINE_PATCHED = False


# ---------------------------------------------------------------------------
# Reading rank files without the training job's process group
# ---------------------------------------------------------------------------


def patch_sharded_tensor_unpickle_for_offline_merge() -> None:
    """Bypass ShardedTensor world-size validation while preserving local shards.

    torch.save() serializes rank-local ShardedTensor objects with process-group
    metadata. torch.load() normally refuses to unpickle them unless the current
    world size matches the save-time world size. For offline merge we only need
    each rank file's local shard tensors and shard metadata, so this patch keeps
    those fields and avoids distributed reconstruction.
    """
    global _SHARDED_TENSOR_OFFLINE_PATCHED
    if _SHARDED_TENSOR_OFFLINE_PATCHED:
        return
    try:
        from torch.distributed._shard.sharded_tensor import api as sharded_tensor_api
    except Exception:
        return

    sharded_tensor_cls = getattr(sharded_tensor_api, "ShardedTensor", None)
    if sharded_tensor_cls is None:
        return

    original_setstate = getattr(sharded_tensor_cls, "__setstate__", None)
    if getattr(original_setstate, "_orbit_offline_merge_patch", False):
        _SHARDED_TENSOR_OFFLINE_PATCHED = True
        return

    def _offline_setstate(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, (tuple, list)):
            if len(state) >= 1:
                self._local_shards = state[0]
            if len(state) >= 2:
                self._metadata = state[1]
            if len(state) >= 3:
                self._sharding_spec = state[2]
            if len(state) >= 5:
                self._init_rrefs = state[4]
        else:
            if original_setstate is not None:
                original_setstate(self, state)
                return
            raise RuntimeError(f"Unsupported ShardedTensor pickle state: {type(state).__name__}")

        # Attributes expected by ShardedTensor accessors in recent PyTorch.
        if not hasattr(self, "_local_shards"):
            self._local_shards = []
        if not hasattr(self, "_metadata"):
            self._metadata = None
        if not hasattr(self, "_sharding_spec"):
            self._sharding_spec = None
        self._process_group = None
        self._remote_shards = {}
        self._init_rrefs = bool(getattr(self, "_init_rrefs", False))

    _offline_setstate._orbit_offline_merge_patch = True  # type: ignore[attr-defined]
    sharded_tensor_cls.__setstate__ = _offline_setstate
    _SHARDED_TENSOR_OFFLINE_PATCHED = True


def ensure_cpu_process_group() -> None:
    """ShardedTensor unpickling requires a default process group, even offline."""
    global _CPU_PROCESS_GROUP_CREATED, _CPU_PROCESS_GROUP_FILE
    if not dist.is_available() or dist.is_initialized():
        return
    tmpdir = Path(tempfile.mkdtemp(prefix="orbit_cpu_merge_pg_"))
    init_file = tmpdir / "store"
    _CPU_PROCESS_GROUP_FILE = init_file
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    _CPU_PROCESS_GROUP_CREATED = True


def destroy_cpu_process_group() -> None:
    global _CPU_PROCESS_GROUP_CREATED
    if _CPU_PROCESS_GROUP_CREATED and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    _CPU_PROCESS_GROUP_CREATED = False
    if _CPU_PROCESS_GROUP_FILE is None:
        return
    try:
        shutil.rmtree(_CPU_PROCESS_GROUP_FILE.parent, ignore_errors=True)
    except Exception:
        pass


def load_checkpoint_payload(path: Path, *, mmap: bool) -> dict[str, Any]:
    patch_sharded_tensor_unpickle_for_offline_merge()
    ensure_cpu_process_group()
    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
    }
    if mmap:
        kwargs["mmap"] = True
    payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported checkpoint payload type at {path}: {type(payload).__name__}")
    return payload


def discover_rank_paths(checkpoint_ref: Path) -> list[Path]:
    checkpoint_dir = resolve_checkpoint_dir_from_ref(checkpoint_ref)
    paths = sorted(checkpoint_dir.glob("rank_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No rank_*.pt files under {checkpoint_dir}")
    return paths


def normalize_key(key: str) -> str:
    """Strips the training wrapper's "model." prefix, so the keys match an HF state dict."""
    if key.startswith("model.model.") or key.startswith("model.lm_head."):
        return key[len("model.") :]
    if key.startswith("module."):
        return key[len("module.") :]
    return key


def normalize_model_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise RuntimeError(f"Expected payload['model'] mapping, got {type(state).__name__}")
    return {normalize_key(str(key)): value for key, value in state.items()}


def build_reference_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"model", "optimizer", "dataloader_state", "rng_state"}
    }


# ---------------------------------------------------------------------------
# The model config of the export
# ---------------------------------------------------------------------------


def set_hf_config_metadata(config: OuroConfig, *, attn_impl: str) -> None:
    config.use_cache = True
    config._attn_implementation = str(attn_impl)
    setattr(config, "_attn_implementation_internal", str(attn_impl))
    config.architectures = ["OuroForCausalLM"]
    config.auto_map = {
        "AutoConfig": "configuration_ouro.OuroConfig",
        "AutoModelForCausalLM": "modelling_ouro.OuroForCausalLM",
    }


def build_export_config(
    *,
    checkpoint_ref: Path,
    model_config_path: Path,
    tokenizer_config_path: Path,
    attn_impl: str,
) -> OuroConfig:
    config_kwargs = load_model_config(model_config_path)
    metadata = read_checkpoint_metadata(checkpoint_ref)
    if isinstance(metadata, dict):
        overrides = metadata.get("model_config_overrides")
        if isinstance(overrides, dict):
            config_kwargs.update(dict(overrides))
    # The checkpoint records the local FA3 build that the training run used. That build is a path
    # on the training machine and loads only under the torch it was built for. The HF modelling
    # code prefers this path over ORBIT_ATTN_KERNEL_PATH, so the export leaves it out and the
    # loader picks its own kernel.
    for key in MACHINE_LOCAL_CONFIG_KEYS:
        config_kwargs.pop(key, None)

    tokenizer_cfg = TokenizerConfig(Path(tokenizer_config_path))
    config_kwargs["vocab_size"] = int(tokenizer_cfg.vocab_size)
    config_kwargs["pad_token_id"] = tokenizer_cfg.pad_id
    config_kwargs["bos_token_id"] = tokenizer_cfg.bos_id
    config_kwargs["eos_token_id"] = tokenizer_cfg.eos_id

    config_kwargs.setdefault("hidden_size", 2048)
    config_kwargs.setdefault("intermediate_size", 5632)
    config_kwargs.setdefault("num_hidden_layers", 24)
    config_kwargs.setdefault("num_attention_heads", 16)
    config_kwargs.setdefault("num_key_value_heads", int(config_kwargs["num_attention_heads"]))
    config_kwargs.setdefault("max_position_embeddings", 4096)
    config_kwargs.setdefault("attention_dropout", 0.0)
    config_kwargs.setdefault("rms_norm_eps", 1e-6)
    config_kwargs.setdefault("tie_word_embeddings", False)
    config_kwargs.setdefault("use_sliding_window", False)
    config_kwargs.setdefault("sliding_window", 4096)
    config_kwargs.setdefault("max_window_layers", int(config_kwargs["num_hidden_layers"]))
    config_kwargs.setdefault("early_exit_threshold", 1.0)
    config_kwargs.setdefault("rope_theta", 1.0e4)
    config_kwargs.setdefault("total_ut_steps", 4)
    config_kwargs.setdefault("use_exit_gate", True)
    config_kwargs["use_cache"] = True

    config = OuroConfig(**config_kwargs)
    set_hf_config_metadata(config, attn_impl=attn_impl)
    return config


def expected_model_shapes(config: OuroConfig) -> dict[str, tuple[int, ...]]:
    try:
        with torch.device("meta"):
            model = OuroForCausalLM(config)
        return {key: tuple(int(x) for x in tensor.shape) for key, tensor in model.state_dict().items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Merging the shards of all rank files
# ---------------------------------------------------------------------------


def is_dtensor(value: Any) -> bool:
    return TorchDTensor is not None and isinstance(value, TorchDTensor)


def is_sharded_tensor_like(value: Any) -> bool:
    return (
        callable(getattr(value, "local_shards", None))
        or hasattr(value, "_local_shards")
    )


def to_local_tensor(value: Any) -> torch.Tensor:
    if is_dtensor(value):
        value = value.to_local()
    if isinstance(value, torch.nn.Parameter):
        value = value.detach()
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor-like value, got {type(value).__name__}")
    return value.detach().cpu()


def dtensor_shard_dim(value: Any) -> Optional[int]:
    placements = getattr(value, "placements", None)
    if not placements:
        return None
    shard_dims: list[int] = []
    for placement in placements:
        name = type(placement).__name__.lower()
        if "shard" not in name:
            continue
        dim = getattr(placement, "dim", None)
        if dim is None:
            dim = getattr(placement, "_dim", None)
        if dim is None:
            raise RuntimeError(f"Could not infer DTensor shard dim from placement={placement!r}")
        shard_dims.append(int(dim))
    if not shard_dims:
        return None
    if len(shard_dims) > 1:
        raise RuntimeError(f"Only one DTensor sharding placement is supported, got {placements!r}")
    return shard_dims[0]


def sharded_tensor_global_shape(value: Any) -> tuple[int, ...]:
    raw_metadata = getattr(value, "_metadata", None)
    if raw_metadata is not None:
        size = getattr(raw_metadata, "size", None)
        if size is not None:
            return tuple(int(x) for x in size)
    metadata_fn = getattr(value, "metadata", None)
    if callable(metadata_fn):
        metadata = metadata_fn()
        size = getattr(metadata, "size", None)
        if size is not None:
            return tuple(int(x) for x in size)
    size_fn = getattr(value, "size", None)
    if callable(size_fn):
        return tuple(int(x) for x in size_fn())
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(x) for x in shape)
    raise RuntimeError(f"Could not infer global sharded tensor shape for {type(value).__name__}")


def place_sharded_tensor_value(
    accumulator: dict[str, Any],
    key: str,
    value: Any,
    expected_shape: tuple[int, ...],
) -> int:
    full_shape = expected_shape or sharded_tensor_global_shape(value)
    local_shards = getattr(value, "_local_shards", None)
    if local_shards is None:
        local_shards = value.local_shards()
    placed = 0
    for shard in local_shards:
        shard_tensor = getattr(shard, "tensor", shard).detach().cpu()
        metadata = getattr(shard, "metadata", None)
        if metadata is None:
            metadata = getattr(shard, "shard_metadata", None)
        offsets = getattr(metadata, "shard_offsets", None)
        sizes = getattr(metadata, "shard_sizes", None)
        if offsets is None or sizes is None:
            raise RuntimeError(
                f"ShardedTensor local shard metadata is missing offsets/sizes for key={key!r}."
            )
        if key not in accumulator:
            accumulator[key] = torch.empty(full_shape, dtype=shard_tensor.dtype)
        target = accumulator[key]
        slices = tuple(
            slice(int(offset), int(offset) + int(size))
            for offset, size in zip(offsets, sizes)
        )
        target[slices] = shard_tensor.reshape(tuple(int(x) for x in sizes))
        placed += 1
    return placed


def infer_concat_dim(
    tensors: list[torch.Tensor],
    expected_shape: tuple[int, ...],
) -> Optional[int]:
    if not expected_shape:
        return None
    for dim in range(len(expected_shape)):
        if any(tensor.ndim != len(expected_shape) for tensor in tensors):
            continue
        if any(
            any(int(tensor.shape[axis]) != int(expected_shape[axis]) for axis in range(len(expected_shape)) if axis != dim)
            for tensor in tensors
        ):
            continue
        if sum(int(tensor.shape[dim]) for tensor in tensors) == int(expected_shape[dim]):
            return dim
    return None


def merge_plain_or_dtensor_values(
    values: list[Any],
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    first = values[0]
    if is_dtensor(first):
        shard_dim = dtensor_shard_dim(first)
        tensors = [to_local_tensor(value) for value in values]
        if shard_dim is None:
            return tensors[0].contiguous()
        merged = torch.cat([tensor for tensor in tensors if tensor.numel() > 0], dim=shard_dim)
        if expected_shape:
            expected_numel = math.prod(expected_shape)
            if int(merged.numel()) == int(expected_numel) and tuple(merged.shape) != expected_shape:
                merged = merged.reshape(expected_shape)
            if tuple(merged.shape) != expected_shape:
                raise RuntimeError(f"DTensor merge shape mismatch: got={tuple(merged.shape)} expected={expected_shape}")
        return merged.contiguous()

    tensors = [to_local_tensor(value) for value in values]
    first_tensor = tensors[0]
    if expected_shape and tuple(first_tensor.shape) == expected_shape:
        return first_tensor.contiguous()
    if not expected_shape and all(tuple(tensor.shape) == tuple(first_tensor.shape) for tensor in tensors):
        return first_tensor.contiguous()

    concat_dim = infer_concat_dim(tensors, expected_shape)
    if concat_dim is not None:
        return torch.cat([tensor for tensor in tensors if tensor.numel() > 0], dim=concat_dim).contiguous()

    if expected_shape:
        expected_numel = math.prod(expected_shape)
        total_numel = sum(int(tensor.numel()) for tensor in tensors)
        if total_numel == expected_numel:
            return torch.cat([tensor.reshape(-1) for tensor in tensors if tensor.numel() > 0], dim=0).reshape(expected_shape).contiguous()

    raise RuntimeError(
        "Could not infer how to merge tensor shards: "
        f"first_shape={tuple(first_tensor.shape)} expected={expected_shape} "
        f"num_ranks={len(tensors)} total_numel={sum(int(t.numel()) for t in tensors)}"
    )


def merge_model_states_streaming(
    rank_paths: list[Path],
    *,
    expected_shapes: dict[str, tuple[int, ...]],
    mmap: bool,
    allow_missing_rank_keys: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merges the model shards of all rank files, reading one rank file at a time."""
    if not rank_paths:
        raise RuntimeError("No rank files selected for merge.")

    merged: dict[str, Any] = {}
    deferred: dict[str, list[torch.Tensor]] = {}
    completed_full_tensor_keys: set[str] = set()
    reference_payload: dict[str, Any] = {}
    reference_keys: Optional[list[str]] = None
    total_shards_placed = 0

    for rank_idx, path in enumerate(rank_paths):
        rank_started = time.monotonic()
        print(
            f"loading_rank_file={rank_idx + 1}/{len(rank_paths)} path={path}",
            flush=True,
        )
        payload = load_checkpoint_payload(path, mmap=mmap)
        load_time = time.monotonic() - rank_started
        if rank_idx == 0:
            reference_payload = build_reference_payload(payload)
        model_state = normalize_model_state(payload.get("model"))
        del payload

        if reference_keys is None:
            reference_keys = list(model_state.keys())
        elif not allow_missing_rank_keys:
            missing = [key for key in reference_keys if key not in model_state]
            if missing:
                raise RuntimeError(
                    f"Missing {len(missing)} model keys in rank file {path}: {missing[:5]}"
                )

        merge_started = time.monotonic()
        active_keys = reference_keys if reference_keys is not None else list(model_state.keys())
        for key in active_keys:
            if key not in model_state:
                continue
            value = model_state[key]
            expected_shape = expected_shapes.get(key, ())
            if is_sharded_tensor_like(value):
                total_shards_placed += place_sharded_tensor_value(
                    merged,
                    key,
                    value,
                    expected_shape,
                )
                continue

            if is_dtensor(value) or torch.is_tensor(value) or isinstance(value, torch.nn.Parameter):
                tensor = to_local_tensor(value).contiguous()
                if key in completed_full_tensor_keys:
                    continue
                if expected_shape and tuple(tensor.shape) == expected_shape:
                    merged[key] = tensor
                    completed_full_tensor_keys.add(key)
                    continue
                if not expected_shape and rank_idx == 0:
                    merged[key] = tensor
                    completed_full_tensor_keys.add(key)
                    continue
                deferred.setdefault(key, []).append(tensor)
                continue

            if rank_idx == 0:
                merged[key] = value

        merge_time = time.monotonic() - merge_started
        print(
            f"merged_rank_file={rank_idx + 1}/{len(rank_paths)} "
            f"load_time={load_time:.3f}s merge_time={merge_time:.3f}s "
            f"model_keys={len(model_state)} shards_placed={total_shards_placed}",
            flush=True,
        )
        del model_state
        gc.collect()

    if deferred:
        print(f"finalizing_deferred_tensor_keys={len(deferred)}", flush=True)
    for key, tensors in deferred.items():
        if key in completed_full_tensor_keys:
            continue
        expected_shape = expected_shapes.get(key, ())
        merged[key] = merge_plain_or_dtensor_values(tensors, expected_shape)

    return merged, reference_payload


# ---------------------------------------------------------------------------
# Writing the HF folder
# ---------------------------------------------------------------------------


def parse_size_to_bytes(value: str) -> int:
    text = str(value).strip().upper()
    units = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * multiplier)
    return int(text)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def build_shards(state_dict: dict[str, Any], max_shard_bytes: int) -> list[dict[str, torch.Tensor]]:
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_size = 0
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        size = tensor_nbytes(tensor)
        if current and current_size + size > max_shard_bytes:
            shards.append(current)
            current = {}
            current_size = 0
        current[key] = tensor
        current_size += size
    if current:
        shards.append(current)
    return shards


def save_sharded_state_dict(
    state_dict: dict[str, Any],
    output_dir: Path,
    *,
    safe_serialization: bool,
    max_shard_size: str,
) -> None:
    max_shard_bytes = parse_size_to_bytes(max_shard_size)
    shards = build_shards(state_dict, max_shard_bytes)
    total_size = sum(tensor_nbytes(tensor) for tensor in state_dict.values() if torch.is_tensor(tensor))
    if safe_serialization and save_safetensors_file is None:
        raise RuntimeError("safetensors is required for --safe-serialization. Use --no-safe-serialization.")

    if len(shards) == 1:
        filename = "model.safetensors" if safe_serialization else "pytorch_model.bin"
        path = output_dir / filename
        if safe_serialization:
            save_safetensors_file(shards[0], str(path), metadata={"format": "pt"})
        else:
            torch.save(shards[0], path)
        return

    weight_map: dict[str, str] = {}
    total = len(shards)
    for idx, shard in enumerate(shards, start=1):
        if safe_serialization:
            filename = f"model-{idx:05d}-of-{total:05d}.safetensors"
            save_safetensors_file(shard, str(output_dir / filename), metadata={"format": "pt"})
        else:
            filename = f"pytorch_model-{idx:05d}-of-{total:05d}.bin"
            torch.save(shard, output_dir / filename)
        for key in shard:
            weight_map[key] = filename
    index_name = "model.safetensors.index.json" if safe_serialization else "pytorch_model.bin.index.json"
    with (output_dir / index_name).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {"total_size": int(total_size)},
                "weight_map": weight_map,
            },
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )


def copy_remote_code_files(output_dir: Path) -> None:
    package_root = Path(__file__).resolve().parent
    for filename in REMOTE_CODE_FILES:
        shutil.copy2(package_root / filename, output_dir / filename)


def maybe_export_tokenizer(
    output_dir: Path,
    *,
    tokenizer_source: Optional[str],
    tokenizer_config_path: Optional[Path],
) -> None:
    """Saves the tokenizer into the export, and keeps the training tokenizer config next to it.

    train_tokenizer_config.json records the eos and pad tokens the run trained with, because a
    chat export changes the shipped tokenizer_config.json afterwards.
    """
    if tokenizer_config_path is not None and tokenizer_config_path.exists():
        shutil.copy2(tokenizer_config_path, output_dir / "train_tokenizer_config.json")
    if tokenizer_source is None:
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)


def source_checkpoint_name(checkpoint_ref: Path, step_dir: Path) -> str:
    """The source of an export as <run_name>/checkpoints/<step folder>.

    The run name is read from the reference as the caller wrote it, and symlinks are not followed.
    A run folder reached through a symlink is therefore recorded under the symlink's name, which
    lets a run be exported under another name than the folder it trained in.
    """
    ref = Path(os.path.abspath(checkpoint_ref))
    if ref.name.startswith("rank_"):
        run_dir = ref.parents[2]
    elif ref.name.startswith("step_"):
        run_dir = ref.parents[1]
    elif ref.name == "latest.txt":
        run_dir = ref.parent
    else:
        run_dir = ref
    return f"{run_dir.name}/checkpoints/{step_dir.name}"


def write_export_info(
    output_dir: Path,
    *,
    checkpoint_ref: Path,
    rank_paths: list[Path],
    reference_payload: dict[str, Any],
    state_dict: dict[str, Any],
    safe_serialization: bool,
    max_shard_size: str,
) -> None:
    """Writes export_info.json, which records where the weights came from and how they were merged.

    An export folder is meant to be copied to other machines and published. The file therefore
    names the source checkpoint as <run_name>/checkpoints/step_N, without the folders above the
    run, and the copied checkpoint metadata leaves out the config keys that hold local paths.
    """
    metadata = read_checkpoint_metadata(checkpoint_ref)
    overrides = metadata.get("model_config_overrides") if isinstance(metadata, dict) else None
    if isinstance(overrides, dict):
        for key in MACHINE_LOCAL_CONFIG_KEYS:
            overrides.pop(key, None)
    total_size = sum(tensor_nbytes(tensor) for tensor in state_dict.values() if torch.is_tensor(tensor))
    payload = {
        "source_checkpoint": source_checkpoint_name(checkpoint_ref, rank_paths[0].parent) if rank_paths else None,
        "source_rank_count": len(rank_paths),
        "source_world_size": reference_payload.get("world_size"),
        "source_parallelism": reference_payload.get("parallelism"),
        "fsdp_state_dict_api": reference_payload.get("fsdp_state_dict_api"),
        "training_state": reference_payload.get("training_state"),
        "checkpoint_metadata": metadata,
        "hf_safe_serialization": bool(safe_serialization),
        "hf_max_shard_size": str(max_shard_size),
        "merged_tensor_count": sum(1 for value in state_dict.values() if torch.is_tensor(value)),
        "merged_total_tensor_bytes": int(total_size),
    }
    with (output_dir / "export_info.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def export_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    model_config: Path,
    tokenizer_dir: Optional[Path],
    tokenizer_config: Optional[Path] = None,
    attn_impl: str = "eager",
    chat_export: bool = False,
    max_shard_size: str = "5GB",
    safe_serialization: bool = True,
    mmap: bool = True,
    allow_missing_rank_keys: bool = False,
) -> Path:
    """Writes the HF export of one checkpoint step and returns the output folder.

    tokenizer_config defaults to tokenizer_dir/tokenizer_config.json. A tokenizer_dir of None
    writes no tokenizer files, which only tests use.
    """
    if tokenizer_config is None:
        if tokenizer_dir is None:
            raise ValueError("export_checkpoint needs tokenizer_dir or tokenizer_config.")
        tokenizer_config = Path(tokenizer_dir) / "tokenizer_config.json"
    try:
        ensure_cpu_process_group()
        rank_paths = discover_rank_paths(checkpoint)
        print(f"rank_files={len(rank_paths)} checkpoint_dir={rank_paths[0].parent}", flush=True)
        config = build_export_config(
            checkpoint_ref=checkpoint,
            model_config_path=model_config,
            tokenizer_config_path=tokenizer_config,
            attn_impl=attn_impl,
        )
        expected_shapes = expected_model_shapes(config)
        print(f"expected_model_tensors={len(expected_shapes)}", flush=True)

        merged, reference_payload = merge_model_states_streaming(
            rank_paths,
            expected_shapes=expected_shapes,
            mmap=mmap,
            allow_missing_rank_keys=allow_missing_rank_keys,
        )
        gc.collect()
        print(f"merged_state_entries={len(merged)}", flush=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(output_dir)
        copy_remote_code_files(output_dir)
        maybe_export_tokenizer(
            output_dir,
            tokenizer_source=None if tokenizer_dir is None else str(tokenizer_dir),
            tokenizer_config_path=Path(tokenizer_config),
        )
        if chat_export:
            from .train_eval import ship_chat_export

            ship_chat_export(output_dir)
        save_sharded_state_dict(
            merged,
            output_dir,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )
        write_export_info(
            output_dir,
            checkpoint_ref=checkpoint,
            rank_paths=rank_paths,
            reference_payload=reference_payload,
            state_dict=merged,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )
        print(f"hf_cpu_merge_done={output_dir.resolve()}", flush=True)
        return output_dir
    finally:
        destroy_cpu_process_group()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a OrbitLM trainer checkpoint to a HuggingFace model folder (CPU only).",
        epilog="The module docstring of orbit.export describes the inputs.",
    )
    parser.add_argument("checkpoint", type=Path, help="A run folder (its latest step), a step folder, or latest.txt.")
    parser.add_argument("output_dir", type=Path, help="The HF folder to write.")
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="The trainer's model config, src/orbit/ouro_1.4b_qwen3_config.json.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="The tokenizer folder the run trained with; its tokenizer_config.json gives the vocabulary and the special ids.",
    )
    parser.add_argument(
        "--attn-impl",
        choices=["eager", "sdpa", "kernels-community/vllm-flash-attn3"],
        default="eager",
        help="The attention implementation recorded in the export's config object.",
    )
    parser.add_argument(
        "--chat-export",
        action="store_true",
        help="The checkpoint is a chat (SFT/RL) model: ship <|im_end|> as eos and write generation_config.json. "
             "Leave it off for base models, which end documents with <|endoftext|>.",
    )
    parser.add_argument("--max-shard-size", type=str, default="5GB")
    parser.add_argument("--safe-serialization", dest="safe_serialization", action="store_true")
    parser.add_argument("--no-safe-serialization", dest="safe_serialization", action="store_false")
    parser.set_defaults(safe_serialization=True)
    parser.add_argument("--mmap", dest="mmap", action="store_true")
    parser.add_argument("--no-mmap", dest="mmap", action="store_false")
    parser.set_defaults(mmap=True)
    parser.add_argument("--allow-missing-rank-keys", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    export_checkpoint(
        args.checkpoint,
        args.output_dir,
        model_config=args.model_config,
        tokenizer_dir=args.tokenizer,
        attn_impl=args.attn_impl,
        chat_export=bool(args.chat_export),
        max_shard_size=args.max_shard_size,
        safe_serialization=bool(args.safe_serialization),
        mmap=bool(args.mmap),
        allow_missing_rank_keys=bool(args.allow_missing_rank_keys),
    )


if __name__ == "__main__":
    main()
