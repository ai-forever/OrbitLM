"""Trainer checkpoints: one rank file per GPU, a meta.json, and a latest.txt pointer per run.

Layout of a run folder:

    <save_dir>/<run_name>/latest.txt                       name of the newest complete step folder
    <save_dir>/<run_name>/checkpoints/step_00001000/rank_0000.pt ...
    <save_dir>/<run_name>/checkpoints/step_00001000/meta.json

A rank file holds that rank's FSDP shard of the model and the optimizer (torch's distributed
state-dict API, recorded as "unified_v2"), the LR scheduler, the RNG state, the seeds, and the
position of the rank's data stream. Exact resume therefore needs the same world size. To start a
new stage from a checkpoint, export it to HF (orbit.export) and load it with init_from_hf.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .fsdp_utils import (
    fsdp_supports_unified_state_dict_api,
    get_fsdp_state_dicts,
    get_fsdp_use_orig_params,
    is_fsdp_model,
    set_fsdp_state_dicts,
    unwrap_model,
)
from .optional import StreamingDataset


CHECKPOINT_FORMAT = "orbit_train_fast_v1"
FSDP_STATE_DICT_API = "unified_v2"


def _dist_object_device() -> Optional[torch.device]:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return None


def _extract_decoder_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    module = unwrap_model(model)
    lm_model = getattr(module, "model", None)
    if lm_model is None:
        return None
    decoder = getattr(lm_model, "model", None)
    if decoder is None or not isinstance(decoder, torch.nn.Module):
        return None
    return decoder


def _extract_lm_model(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    module = unwrap_model(model)
    lm_model = getattr(module, "model", None)
    if lm_model is None or not isinstance(lm_model, torch.nn.Module):
        return None
    return lm_model


def build_model_config_overrides(model: torch.nn.Module) -> Optional[dict[str, Any]]:
    lm_model = _extract_lm_model(model)
    if lm_model is None:
        return None
    config = getattr(lm_model, "config", None)
    if config is None:
        return None
    if callable(getattr(config, "to_dict", None)):
        return copy.deepcopy(config.to_dict())
    return copy.deepcopy(dict(getattr(config, "__dict__", {})))


def build_trainer_config(model: torch.nn.Module) -> Optional[dict[str, Any]]:
    module = unwrap_model(model)
    beta = getattr(module, "beta", None)
    exit_mode = getattr(module, "exit_mode", None)
    lm_chunk_size = getattr(module, "lm_chunk_size", None)
    pad_token_id = getattr(module, "pad_token_id", None)
    if beta is None and exit_mode is None and lm_chunk_size is None and pad_token_id is None:
        return None
    return {
        "beta": None if beta is None else float(beta),
        "exit_mode": None if exit_mode is None else str(exit_mode),
        "lm_chunk_size": None if lm_chunk_size is None else int(lm_chunk_size),
        "pad_token_id": None if pad_token_id is None else int(pad_token_id),
    }


def build_checkpoint_metadata_extra(model: torch.nn.Module) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    model_config_overrides = build_model_config_overrides(model)
    if isinstance(model_config_overrides, dict) and model_config_overrides:
        extra["model_config_overrides"] = model_config_overrides

    trainer_config = build_trainer_config(model)
    if isinstance(trainer_config, dict) and trainer_config:
        extra["trainer_config"] = trainer_config

    fsdp_use_orig_params = get_fsdp_use_orig_params(model) if is_fsdp_model(model) else None
    if fsdp_use_orig_params is not None:
        extra["fsdp_use_orig_params"] = bool(fsdp_use_orig_params)
    return extra


def validate_trainer_config(saved: Optional[dict[str, Any]], model: torch.nn.Module) -> None:
    """Refuses to resume a checkpoint whose loss settings differ from the running trainer's."""
    if not isinstance(saved, dict):
        return
    current = build_trainer_config(model)
    if not isinstance(current, dict):
        return

    saved_beta = saved.get("beta")
    current_beta = current.get("beta")
    if saved_beta is not None and current_beta is not None:
        if not math.isclose(float(saved_beta), float(current_beta), rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError(
                f"Checkpoint beta mismatch: checkpoint={saved_beta} runtime={current_beta}"
            )

    saved_exit_mode = saved.get("exit_mode", "learned")
    current_exit_mode = current.get("exit_mode")
    if saved_exit_mode is not None and current_exit_mode is not None:
        if str(saved_exit_mode) != str(current_exit_mode):
            raise RuntimeError(
                "Checkpoint exit_mode mismatch: "
                f"checkpoint={saved_exit_mode} runtime={current_exit_mode}"
            )

    saved_lm_chunk_size = saved.get("lm_chunk_size")
    current_lm_chunk_size = current.get("lm_chunk_size")
    if saved_lm_chunk_size is not None and current_lm_chunk_size is not None:
        if int(saved_lm_chunk_size) != int(current_lm_chunk_size):
            raise RuntimeError(
                "Checkpoint lm_chunk_size mismatch: "
                f"checkpoint={saved_lm_chunk_size} runtime={current_lm_chunk_size}"
            )

    saved_pad_token_id = saved.get("pad_token_id")
    current_pad_token_id = current.get("pad_token_id")
    if saved_pad_token_id != current_pad_token_id:
        raise RuntimeError(
            "Checkpoint pad_token_id mismatch: "
            f"checkpoint={saved_pad_token_id} runtime={current_pad_token_id}"
        )


def infer_resume_model_config_overrides(
    checkpoint_ref: Path,
    *,
    rank: int,
) -> Optional[dict[str, Any]]:
    """The model config recorded in a checkpoint's meta.json, read on rank 0 and broadcast."""
    payload_hint: Optional[dict[str, Any]] = None
    if rank == 0:
        metadata = read_checkpoint_metadata(checkpoint_ref)
        hinted = metadata.get("model_config_overrides") if isinstance(metadata, dict) else None
        if isinstance(hinted, dict) and hinted:
            payload_hint = hinted
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        object_list = [payload_hint]
        torch.distributed.broadcast_object_list(
            object_list,
            src=0,
            device=_dist_object_device(),
        )
        payload_hint = object_list[0]
    if not isinstance(payload_hint, dict) or not payload_hint:
        return None
    return payload_hint


def refresh_model_runtime_state(model: torch.nn.Module) -> None:
    """Rebuilds the rotary frequency buffers from the model config after a state-dict load.

    inv_freq is a non-persistent buffer, so a checkpoint does not carry it. Recomputing it here
    makes a resumed run use exactly the frequencies of its config.
    """
    decoder = _extract_decoder_module(model)
    if decoder is None:
        return

    rotary = getattr(decoder, "rotary_emb", None)
    if rotary is None:
        return

    rope_init_fn = getattr(rotary, "rope_init_fn", None)
    config = getattr(rotary, "config", None)
    if rope_init_fn is None or config is None:
        return

    current_inv_freq = getattr(rotary, "inv_freq", None)
    current_original_inv_freq = getattr(rotary, "original_inv_freq", None)
    device = current_inv_freq.device if torch.is_tensor(current_inv_freq) else None
    inv_freq_dtype = current_inv_freq.dtype if torch.is_tensor(current_inv_freq) else None
    if torch.is_tensor(current_original_inv_freq):
        original_inv_freq = current_original_inv_freq
        if device is not None:
            if inv_freq_dtype is not None:
                inv_freq = original_inv_freq.to(device=device, dtype=inv_freq_dtype)
            else:
                inv_freq = original_inv_freq.to(device=device)
        elif inv_freq_dtype is not None:
            inv_freq = original_inv_freq.to(dtype=inv_freq_dtype)
        else:
            inv_freq = original_inv_freq
        _, attention_scaling = rope_init_fn(config, None)
    else:
        inv_freq, attention_scaling = rope_init_fn(config, None)
        original_inv_freq = inv_freq
        if device is not None:
            if inv_freq_dtype is not None:
                inv_freq = inv_freq.to(device=device, dtype=inv_freq_dtype)
            else:
                inv_freq = inv_freq.to(device=device)
        elif inv_freq_dtype is not None:
            inv_freq = inv_freq.to(dtype=inv_freq_dtype)

    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = original_inv_freq
    rotary.attention_scaling = attention_scaling
    original_max_seq_len = getattr(rotary, "original_max_seq_len", None)
    if original_max_seq_len is not None:
        rotary.max_seq_len_cached = original_max_seq_len


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    errors: list[str] = []
    if "python" in state:
        try:
            random.setstate(state["python"])
        except Exception as exc:
            errors.append(f"python={exc!r}")
    if "numpy" in state:
        try:
            np.random.set_state(state["numpy"])
        except Exception as exc:
            errors.append(f"numpy={exc!r}")
    if "torch" in state:
        try:
            torch.set_rng_state(state["torch"])
        except Exception as exc:
            errors.append(f"torch={exc!r}")
    if torch.cuda.is_available() and "torch_cuda" in state:
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception as exc:
            errors.append(f"torch_cuda={exc!r}")
    if errors:
        raise RuntimeError(f"Failed to restore RNG state components: {errors}")


def collect_stateful_dataset_nodes(root_dataset: Optional[Dataset]) -> list[tuple[str, Any]]:
    if root_dataset is None:
        return []

    nodes: list[tuple[str, Any]] = []
    visited: set[int] = set()

    def walk(obj: Any, path: str) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if callable(getattr(obj, "state_dict", None)) and callable(
            getattr(obj, "load_state_dict", None)
        ):
            nodes.append((path, obj))

        if isinstance(obj, ConcatDataset):
            for idx, child in enumerate(obj.datasets):
                walk(child, f"{path}.datasets[{idx}]")
            return

        child = getattr(obj, "dataset", None)
        if child is not None:
            walk(child, f"{path}.dataset")

        children = getattr(obj, "datasets", None)
        if children is not None:
            try:
                child_list = list(children)
            except Exception:
                child_list = []
            for idx, child_ds in enumerate(child_list):
                walk(child_ds, f"{path}.datasets[{idx}]")

    walk(root_dataset, "dataset")
    return nodes


def call_dataset_state_dict(ds_obj: Any, samples_seen: int) -> tuple[Any, dict]:
    fn = ds_obj.state_dict
    state_call: dict = {"mode": "unknown"}
    samples_seen = int(max(0, samples_seen))

    try:
        sig = inspect.signature(fn)
        params = sig.parameters
    except Exception:
        sig = None
        params = {}

    required_positional = []
    required_kwonly = []
    if sig is not None:
        for p in sig.parameters.values():
            if (
                p.default is inspect._empty
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ):
                required_positional.append(p)
            if p.default is inspect._empty and p.kind == inspect.Parameter.KEYWORD_ONLY:
                required_kwonly.append(p)

    can_call_no_args = (sig is None) or (not required_positional and not required_kwonly)
    if can_call_no_args:
        try:
            state = fn()
            state_call = {"mode": "no_args"}
            return state, state_call
        except TypeError:
            # Some datasets expose optional args but still reject no-arg call.
            pass

    kwargs = {}
    if "num_samples" in params:
        kwargs["num_samples"] = samples_seen
    if "from_beginning" in params:
        # For a StreamingDataset, samples_seen is already the absolute within-epoch
        # offset (wrapped modulo epoch_size in _dataset_state_dict_sample_hint), so set
        # it directly via from_beginning=True. from_beginning=False would instead do
        # sample_in_epoch = prior_resume_offset + num_samples, which double-counts the
        # pre-restart portion and eventually trips "Resuming further into the dataset".
        # Non-streaming datasets keep the additive behaviour.
        kwargs["from_beginning"] = bool(
            StreamingDataset is not None and isinstance(ds_obj, StreamingDataset)
        )

    if kwargs:
        try:
            state = fn(**kwargs)
            state_call = {"mode": "kwargs", "kwargs": dict(kwargs)}
            return state, state_call
        except TypeError:
            pass

    if required_positional:
        try:
            state = fn(samples_seen)
            state_call = {"mode": "positional_num_samples", "num_samples": samples_seen}
            return state, state_call
        except TypeError:
            pass

    raise RuntimeError(
        f"Could not call {ds_obj.__class__.__module__}.{ds_obj.__class__.__name__}.state_dict "
        "with supported signatures (no_args / kwargs / positional_num_samples)."
    )


def _streaming_epoch_size(ds_obj: Any) -> Optional[int]:
    """Best-effort GLOBAL samples-per-epoch for a MosaicML StreamingDataset."""
    for attr in ("epoch_size", "num_samples"):
        value = getattr(ds_obj, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    try:
        length = len(ds_obj)
    except Exception:
        return None
    return length if isinstance(length, int) and length > 0 else None


def _dataset_state_dict_sample_hint(
    ds_obj: Any,
    *,
    local_samples_seen: int,
    world_size: int,
) -> int:
    samples_seen = int(max(0, local_samples_seen))
    if StreamingDataset is not None and isinstance(ds_obj, StreamingDataset):
        # StreamingDataset.state_dict() expects the GLOBAL sample offset across ranks.
        global_seen = samples_seen * int(max(1, world_size))
        # Our counter is cumulative across epochs and restarts, but StreamingDataset
        # tracks a within-epoch offset and raises "Resuming further into the dataset"
        # once the value exceeds one epoch. Wrapping into the current epoch keeps resumes
        # valid past one epoch (upsampled streams) and after restarts in the middle of a
        # run, together with from_beginning=True in call_dataset_state_dict.
        epoch_size = _streaming_epoch_size(ds_obj)
        if epoch_size and epoch_size > 0:
            return global_seen % int(epoch_size)
        return global_seen
    return samples_seen


def build_local_dataloader_state(
    dataset: Optional[Dataset],
    *,
    samples_seen: int,
    rank: int,
    world_size: int,
    loader_active: bool,
    loader_config: Optional[dict[str, Any]] = None,
) -> dict:
    stateful_nodes = collect_stateful_dataset_nodes(dataset) if (loader_active and dataset is not None) else []
    state: dict[str, Any] = {
        "version": 2,
        "rank": int(rank),
        "loader_active": bool(loader_active),
        "samples_seen": int(max(0, samples_seen)),
        "stateful_node_count": int(len(stateful_nodes)),
        "capture_mode": "train_ouro_ddp_compatible_v1",
        "loader_config": dict(loader_config) if isinstance(loader_config, dict) else None,
        "datasets": [],
    }
    if not loader_active or dataset is None:
        return state

    for path, ds_obj in stateful_nodes:
        samples_seen_hint = _dataset_state_dict_sample_hint(
            ds_obj,
            local_samples_seen=int(state["samples_seen"]),
            world_size=world_size,
        )
        entry = {
            "path": path,
            "class": f"{ds_obj.__class__.__module__}.{ds_obj.__class__.__name__}",
            "samples_seen_hint": samples_seen_hint,
        }
        try:
            ds_state, state_call = call_dataset_state_dict(ds_obj, samples_seen_hint)
            entry["state"] = ds_state
            entry["state_call"] = state_call
        except Exception as exc:
            raise RuntimeError(
                f"Failed to capture dataloader state for {entry['class']} at {path}: {exc!r}"
            ) from exc
        state["datasets"].append(entry)
    return state


def select_rank_dataloader_state(
    dataloader_state: Optional[dict],
    *,
    rank: int,
    world_size: int,
) -> Optional[dict]:
    if not isinstance(dataloader_state, dict):
        return None
    states = dataloader_state.get("states")
    if not isinstance(states, list) or not states:
        return None
    if len(states) == world_size and 0 <= rank < len(states):
        return states[rank]
    if world_size == 1 and len(states) == 1 and isinstance(states[0], dict):
        return states[0]
    return None


def dataloader_configs_compatible(
    saved: Optional[dict[str, Any]],
    expected: dict[str, Any],
) -> bool:
    if saved == expected:
        return True
    if not isinstance(saved, dict):
        return False
    # Checkpoints whose loader config has no stream list and no proportions are compared on
    # the keys they do have.
    extra_expected_keys = {
        "stage1_streams",
        "stage1_dataset_proportions",
    }
    if set(expected).difference(saved).issubset(extra_expected_keys):
        return all(saved.get(key) == value for key, value in expected.items() if key in saved)
    return False


def restore_local_dataloader_state(
    dataset: Optional[Dataset],
    dataloader_state: Optional[dict],
    *,
    rank: int,
    world_size: int,
    loader_active: bool,
    expected_loader_config: Optional[dict[str, Any]] = None,
) -> bool:
    local_state = select_rank_dataloader_state(
        dataloader_state,
        rank=rank,
        world_size=world_size,
    )
    if local_state is None:
        return False
    if not loader_active:
        return True
    if dataset is None:
        return False
    if expected_loader_config is not None:
        saved_loader_config = local_state.get("loader_config")
        if saved_loader_config is not None and not dataloader_configs_compatible(
            saved_loader_config,
            expected_loader_config,
        ):
            raise RuntimeError(
                "Dataloader config mismatch for exact resume: "
                f"checkpoint={saved_loader_config} runtime={expected_loader_config}"
            )

    saved_entries = local_state.get("datasets", [])
    if not isinstance(saved_entries, list):
        return False
    expected_stateful_count = int(local_state.get("stateful_node_count", 0) or 0)
    if expected_stateful_count == 0 and len(saved_entries) == 0:
        return True
    if not saved_entries:
        return False

    current_nodes = collect_stateful_dataset_nodes(dataset)
    if not current_nodes:
        return expected_stateful_count == 0

    node_by_path = {path: ds_obj for path, ds_obj in current_nodes}
    load_plan: list[tuple[str, Any, Any]] = []
    for entry in saved_entries:
        if not isinstance(entry, dict):
            continue
        state_obj = entry.get("state")
        if state_obj is None:
            continue
        path = entry.get("path")
        ds_obj = node_by_path.get(path) if isinstance(path, str) else None
        if ds_obj is None:
            return False
        load_plan.append((path, ds_obj, state_obj))

    if not load_plan:
        return False

    for path, ds_obj, state_obj in load_plan:
        try:
            ds_obj.load_state_dict(state_obj)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to restore dataloader state for "
                f"{ds_obj.__class__.__module__}.{ds_obj.__class__.__name__} at {path}: {exc!r}"
            ) from exc
    return True


def checkpoint_dir_for_step(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{int(step):08d}"


def rank_checkpoint_path(checkpoint_dir: Path, rank: int) -> Path:
    return checkpoint_dir / f"rank_{int(rank):04d}.pt"


def latest_pointer_path(run_dir: Path) -> Path:
    return run_dir / "latest.txt"


def checkpoint_metadata_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "meta.json"


def write_latest_pointer(run_dir: Path, checkpoint_dir: Path) -> None:
    latest_path = latest_pointer_path(run_dir)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = latest_path.with_suffix(".txt.tmp")
    tmp_path.write_text(f"{checkpoint_dir.name}\n", encoding="utf-8")
    tmp_path.replace(latest_path)


def read_latest_pointer(run_dir: Path) -> Optional[Path]:
    latest_path = latest_pointer_path(run_dir)
    if not latest_path.exists():
        return None
    name = latest_path.read_text(encoding="utf-8").strip()
    if not name:
        return None
    candidate = run_dir / "checkpoints" / name
    return candidate if candidate.exists() else None


def resolve_checkpoint_dir_from_ref(checkpoint_ref: Path) -> Path:
    """A step folder from any checkpoint reference: a rank file, latest.txt, a step or a run folder."""
    checkpoint_ref = checkpoint_ref.resolve()
    if checkpoint_ref.is_file() and checkpoint_ref.name.startswith("rank_"):
        return checkpoint_ref.parent
    if checkpoint_ref.is_file() and checkpoint_ref.name == "latest.txt":
        latest_dir = read_latest_pointer(checkpoint_ref.parent)
        if latest_dir is None:
            raise FileNotFoundError(f"No latest checkpoint pointer under {checkpoint_ref.parent}")
        return latest_dir
    if checkpoint_ref.is_dir() and checkpoint_ref.name.startswith("step_"):
        return checkpoint_ref
    if checkpoint_ref.is_dir() and (checkpoint_ref / "checkpoints").exists():
        latest_dir = read_latest_pointer(checkpoint_ref)
        if latest_dir is None:
            raise FileNotFoundError(f"No latest checkpoint pointer under {checkpoint_ref}")
        return latest_dir
    raise FileNotFoundError(checkpoint_ref)


def read_checkpoint_metadata(checkpoint_ref: Path) -> Optional[dict[str, Any]]:
    try:
        checkpoint_dir = resolve_checkpoint_dir_from_ref(checkpoint_ref)
    except Exception:
        return None
    meta_path = checkpoint_metadata_path(checkpoint_dir)
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def resolve_run_dir_from_ref(checkpoint_ref: Path) -> Path:
    checkpoint_ref = checkpoint_ref.resolve()
    if checkpoint_ref.is_file() and checkpoint_ref.name.startswith("rank_"):
        return checkpoint_ref.parents[2]
    if checkpoint_ref.is_file() and checkpoint_ref.name == "latest.txt":
        return checkpoint_ref.parent
    if checkpoint_ref.is_dir() and checkpoint_ref.name.startswith("step_"):
        return checkpoint_ref.parents[1]
    if checkpoint_ref.is_dir() and (checkpoint_ref / "checkpoints").exists():
        return checkpoint_ref
    raise RuntimeError(f"Could not infer run dir from checkpoint reference: {checkpoint_ref}")


def resolve_rank_checkpoint_from_ref(checkpoint_ref: Path, rank: int) -> Path:
    rank_path = rank_checkpoint_path(resolve_checkpoint_dir_from_ref(checkpoint_ref), rank)
    checkpoint_ref = checkpoint_ref.resolve()
    if checkpoint_ref.is_file() and checkpoint_ref.name.startswith("rank_"):
        return checkpoint_ref
    if not rank_path.exists():
        raise FileNotFoundError(rank_path)
    return rank_path


def save_rank_checkpoint(
    checkpoint_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    rank: int,
    world_size: int,
    parallelism: str,
    training_state: dict[str, Any],
    rng_state: dict[str, Any],
    seed_state: Optional[dict[str, Any]],
    dataloader_state: Optional[dict[str, Any]],
    timings_out: Optional[dict[str, float]] = None,
) -> Path:
    overall_started = time.monotonic()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    fsdp_use_orig_params = None
    fsdp_state_dict_api = None
    state_export_started = time.monotonic()
    if is_fsdp_model(model):
        if not fsdp_supports_unified_state_dict_api():
            raise RuntimeError("Saving an FSDP checkpoint needs torch's distributed state-dict API.")
        fsdp_use_orig_params = get_fsdp_use_orig_params(model)
        model_state, optimizer_state = get_fsdp_state_dicts(model, optimizer)
        fsdp_state_dict_api = FSDP_STATE_DICT_API
    else:
        model_state = unwrap_model(model).state_dict()
        optimizer_state = optimizer.state_dict()
    state_export_time = time.monotonic() - state_export_started

    payload_build_started = time.monotonic()
    payload = {
        "format": CHECKPOINT_FORMAT,
        "parallelism": parallelism,
        "rank": int(rank),
        "world_size": int(world_size),
        "model": model_state,
        "optimizer": optimizer_state,
        "fsdp_state_dict_api": fsdp_state_dict_api,
        "fsdp_use_orig_params": fsdp_use_orig_params,
        "training_state": dict(training_state),
        "rng_state": rng_state,
        "seed_state": dict(seed_state) if isinstance(seed_state, dict) else None,
        "dataloader_state": dataloader_state,
        "model_config_overrides": build_model_config_overrides(model),
        "trainer_config": build_trainer_config(model),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    payload_build_time = time.monotonic() - payload_build_started

    rank_path = rank_checkpoint_path(checkpoint_dir, rank)
    tmp_path = rank_path.with_suffix(rank_path.suffix + ".tmp")
    serialize_started = time.monotonic()
    torch.save(payload, tmp_path)
    serialize_time = time.monotonic() - serialize_started
    replace_started = time.monotonic()
    tmp_path.replace(rank_path)
    replace_time = time.monotonic() - replace_started
    total_time = time.monotonic() - overall_started
    if timings_out is not None:
        timings_out.clear()
        timings_out.update(
            {
                "state_export_time_s": float(state_export_time),
                "payload_build_time_s": float(payload_build_time),
                "serialize_time_s": float(serialize_time),
                "replace_time_s": float(replace_time),
                "save_rank_total_time_s": float(total_time),
                "rank_checkpoint_bytes": float(rank_path.stat().st_size),
            }
        )
    return rank_path


def write_checkpoint_metadata(
    checkpoint_dir: Path,
    *,
    parallelism: str,
    world_size: int,
    training_state: dict[str, Any],
    extra: Optional[dict[str, Any]] = None,
) -> None:
    meta = {
        "format": CHECKPOINT_FORMAT,
        "parallelism": parallelism,
        "world_size": int(world_size),
        "training_state": dict(training_state),
    }
    if extra:
        meta.update(extra)
    meta_path = checkpoint_dir / "meta.json"
    tmp_path = meta_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True, sort_keys=True, indent=2)
    tmp_path.replace(meta_path)


def load_rank_checkpoint(
    checkpoint_ref: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    rank: int,
    world_size: int,
    timings_out: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Restores this rank's model, optimizer and scheduler state and returns the whole payload.

    The caller restores the data position and the RNG state from the returned payload.
    """
    overall_started = time.monotonic()
    checkpoint_path = resolve_rank_checkpoint_from_ref(checkpoint_ref, rank)
    read_started = time.monotonic()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_read_time = time.monotonic() - read_started
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"Unexpected checkpoint format for {checkpoint_path}: {payload.get('format')!r}"
        )
    saved_rank = payload.get("rank")
    if saved_rank is not None and int(saved_rank) != int(rank):
        raise RuntimeError(
            f"Checkpoint rank mismatch: checkpoint={saved_rank} runtime={rank}"
        )
    saved_world_size = int(payload.get("world_size") or 0)
    if saved_world_size != int(world_size):
        raise RuntimeError(
            f"Checkpoint world_size mismatch: checkpoint={saved_world_size} runtime={world_size}"
        )

    state_apply_started = time.monotonic()
    if is_fsdp_model(model):
        if payload.get("fsdp_state_dict_api") not in {"unified_v1", "unified_v2"}:
            raise RuntimeError(
                f"{checkpoint_path} was saved with the FSDP state-dict API "
                f"{payload.get('fsdp_state_dict_api')!r}; this trainer reads only 'unified_v1' and 'unified_v2'."
            )
        if not fsdp_supports_unified_state_dict_api():
            raise RuntimeError("Loading an FSDP checkpoint needs torch's distributed state-dict API.")
        set_fsdp_state_dicts(
            model,
            optimizer,
            model_state_dict=payload["model"],
            optim_state_dict=payload["optimizer"],
        )
    else:
        unwrap_model(model).load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
    refresh_model_runtime_state(model)
    state_apply_time = time.monotonic() - state_apply_started

    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    validate_trainer_config(payload.get("trainer_config"), model)
    total_time = time.monotonic() - overall_started
    if timings_out is not None:
        timings_out.clear()
        timings_out.update(
            {
                "checkpoint_read_time_s": float(checkpoint_read_time),
                "state_apply_time_s": float(state_apply_time),
                "resume_total_time_s": float(total_time),
            }
        )
    if int(rank) == 0:
        print(
            f"resume_state fsdp_state_dict_api={payload.get('fsdp_state_dict_api')} "
            f"world_size={saved_world_size}",
            flush=True,
        )
    return payload
