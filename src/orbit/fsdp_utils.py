"""FSDP wrapping of the training module and the distributed state-dict helpers of checkpoints.

Every decoder layer is its own FSDP unit. The trainer keeps fp32 master weights and lets FSDP's
mixed precision cast them to bf16 for forward and backward.
"""
from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist

from .modelling_ouro import OuroDecoderLayer

try:
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
except Exception:  # pragma: no cover - optional
    BackwardPrefetch = None
    FSDP = None
    MixedPrecision = None
    ShardingStrategy = None
    transformer_auto_wrap_policy = None

try:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions as DCPStateDictOptions,
        get_state_dict as dcp_get_state_dict,
        set_state_dict as dcp_set_state_dict,
    )
except Exception:  # pragma: no cover - optional
    DCPStateDictOptions = None
    dcp_get_state_dict = None
    dcp_set_state_dict = None


def distributed_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    current = module
    while FSDP is not None and isinstance(current, FSDP):
        current = current.module
    return current


def is_fsdp_model(module: torch.nn.Module) -> bool:
    return FSDP is not None and isinstance(module, FSDP)


def get_fsdp_use_orig_params(module: torch.nn.Module) -> Optional[bool]:
    if not is_fsdp_model(module):
        return None
    for attr_name in ("_use_orig_params", "_fsdp_use_orig_params"):
        value = getattr(module, attr_name, None)
        if value is not None:
            return bool(value)
    return None


def maybe_no_sync(module: torch.nn.Module, enabled: bool):
    if enabled and hasattr(module, "no_sync"):
        return module.no_sync()
    return nullcontext()


def fsdp_supports_unified_state_dict_api() -> bool:
    return (
        FSDP is not None
        and DCPStateDictOptions is not None
        and dcp_get_state_dict is not None
        and dcp_set_state_dict is not None
    )


def get_fsdp_state_dicts(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict, dict]:
    if not is_fsdp_model(module):
        raise RuntimeError("get_fsdp_state_dicts() requires an FSDP-wrapped module.")
    if not fsdp_supports_unified_state_dict_api():
        raise RuntimeError("Unified distributed state_dict API is unavailable.")
    options = DCPStateDictOptions(
        full_state_dict=False,
        cpu_offload=False,
    )
    return dcp_get_state_dict(module, optimizer, options=options)


def set_fsdp_state_dicts(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    model_state_dict: dict,
    optim_state_dict: dict,
) -> None:
    if not is_fsdp_model(module):
        raise RuntimeError("set_fsdp_state_dicts() requires an FSDP-wrapped module.")
    if not fsdp_supports_unified_state_dict_api():
        raise RuntimeError("Unified distributed state_dict API is unavailable.")
    options = DCPStateDictOptions(
        full_state_dict=False,
        cpu_offload=False,
    )
    result = dcp_set_state_dict(
        module,
        optimizer,
        model_state_dict=model_state_dict,
        optim_state_dict=optim_state_dict,
        options=options,
    )
    missing = list(getattr(result, "missing_keys", ()) or ())
    unexpected = list(getattr(result, "unexpected_keys", ()) or ())
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected FSDP load_state_dict result: missing_keys={missing}, unexpected_keys={unexpected}"
        )


def clip_grad_norm(module: torch.nn.Module, max_norm: float) -> Optional[torch.Tensor]:
    if max_norm <= 0:
        return None
    if is_fsdp_model(module):
        return module.clip_grad_norm_(max_norm)
    params = [p for p in unwrap_model(module).parameters() if p.grad is not None]
    if not params:
        return None
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


def resolve_fsdp_mixed_precision_dtype(
    name: Optional[str],
    *,
    default: torch.dtype,
) -> torch.dtype:
    normalized = str(name or "auto").strip().lower()
    if normalized == "auto":
        return default
    mapping = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported FSDP mixed-precision dtype {name!r}; "
            f"expected one of {sorted(mapping)} or 'auto'."
        )
    return mapping[normalized]


def build_parallel_model(
    module: torch.nn.Module,
    *,
    local_rank: int,
    bf16: bool,
    sync_module_states: bool,
    fsdp_sharding_strategy: str,
    fsdp_reduce_dtype: Optional[str] = None,
    fsdp_buffer_dtype: Optional[str] = None,
) -> torch.nn.Module:
    """Wraps the module in FSDP when more than one process runs; a single process trains it as is."""
    if not distributed_enabled() or dist.get_world_size() == 1:
        return module

    if FSDP is None or transformer_auto_wrap_policy is None or ShardingStrategy is None:
        raise RuntimeError("FSDP is not available in this torch build.")

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={OuroDecoderLayer},
    )
    sharding = {
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "full_shard": ShardingStrategy.FULL_SHARD,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[fsdp_sharding_strategy]
    mixed_precision = None
    if bf16 and MixedPrecision is not None:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=resolve_fsdp_mixed_precision_dtype(
                fsdp_reduce_dtype,
                default=torch.bfloat16,
            ),
            buffer_dtype=resolve_fsdp_mixed_precision_dtype(
                fsdp_buffer_dtype,
                default=torch.bfloat16,
            ),
        )

    # use_orig_params=True keeps the original parameters visible to the optimizer, which the
    # weight-decay groups need.
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "sharding_strategy": sharding,
        "mixed_precision": mixed_precision,
        "device_id": torch.device("cuda", local_rank),
        "sync_module_states": bool(sync_module_states),
        "use_orig_params": True,
        "limit_all_gathers": True,
    }
    if BackwardPrefetch is not None:
        fsdp_kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
    return FSDP(module, **fsdp_kwargs)
