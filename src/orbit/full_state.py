"""Gathers the full, unsharded model weights on rank 0 for the HF export of the in-training evaluation."""
from __future__ import annotations

from typing import Any

import torch

from .fsdp_utils import is_fsdp_model

try:  # torch build may lack FSDP
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )
except Exception:  # pragma: no cover
    FSDP = None
    FullStateDictConfig = None
    StateDictType = None


def _strip_training_wrapper_prefix(state: dict[str, Any]) -> dict[str, Any]:
    # The training module wraps OuroForCausalLM as .model, so its keys start with "model.model."
    # and "model.lm_head."; an HF state dict has them without the first "model.".
    keys = list(state.keys())
    if any(k.startswith("model.model.") or k.startswith("model.lm_head.") for k in keys):
        return {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in state.items()
        }
    return state


def gather_full_state(model: torch.nn.Module, is_rank0: bool) -> dict[str, torch.Tensor]:
    """Full model state on rank 0 (empty on other ranks), wrapper prefix stripped.

    Collective when the model is FSDP-wrapped: all ranks must call together.
    """
    if is_fsdp_model(model) and FSDP is not None:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state = dict(model.state_dict())
        if not is_rank0:
            return {}
    else:
        inner = getattr(model, "module", model)
        state = dict(inner.state_dict())
    return _strip_training_wrapper_prefix(state)
