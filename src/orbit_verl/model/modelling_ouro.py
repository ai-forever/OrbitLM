# OrbitLM (Ouro) model code used by the verl port. HF and vLLM load it as remote code from a
# prepared model dir (see orbit_verl.model.prepare_model).
#
# Source: the trainer's src/orbit/modelling_ouro.py as of the SFT runs. This file is that
# copy with only the changes listed below, so a diff against the source stays short. It must keep
# working on transformers 4.57 and 5.x. HF and vLLM copy it into their own module cache, so it
# imports nothing from verl or orbit_verl, and its only relative import is .configuration_ouro.
#
# Changes (docs/verl/DESIGN.md):
# P1  The `kernels` import is optional. HF refuses remote code whose imports are not installed, so a
#     required import would stop an env without `kernels` from loading the model at all. Without the
#     package only the FA3 hub-kernel attention fails, with an ImportError.
# P2  check_model_inputs is taken from merge_with_config_defaults when it exists, because
#     transformers 5 deprecated the old name.
# P3  create_causal_mask gets the keyword names of the installed transformers version.
# P4  UniversalTransformerCache.get_mask_sizes accepts the int query length of transformers 5.
# P5  The default RoPE frequencies are computed here. Transformers 5 has no "default" init
#     function, and the old fallback switched RoPE to "linear" and rewrote config.rope_scaling.
# P6  _tied_weights_keys is a dict, because save_pretrained on transformers 5 calls .keys() on it.
# P7  The token embedding takes padding_idx from config.embedding_padding_idx, which prepare_model
#     sets from the export's training code and weights. It never uses pad_token_id: verl overwrites
#     pad_token_id with the tokenizer's pad id (the Qwen3 eos 151643), and the release line was
#     trained with no frozen row at all.
# E1  The FA3 kernel path comes from ORBIT_ATTN_KERNEL_PATH directly instead of from
#     experiment_config, so a model dir needs only this file and configuration_ouro.py.
# E2  The two forward methods have no @auto_docstring. It printed "[ERROR] ... not documented"
#     lines into every Ray worker log.
import logging
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
try:
    from transformers.utils.generic import merge_with_config_defaults as check_model_inputs
except ImportError:  # transformers < 5
    from transformers.utils.generic import check_model_inputs
try:
    from transformers.integrations import use_kernel_forward_from_hub
except ImportError:  # pragma: no cover - depends on transformers version
    def use_kernel_forward_from_hub(*args, **kwargs):
        def decorator(obj):
            return obj

        return decorator

try:
    from kernels import get_kernel, get_local_kernel
except ImportError:  # P1: only the FA3 hub-kernel attention below needs `kernels`.
    get_kernel = None
    get_local_kernel = None

try:
    # Package-style import (when used inside a proper module)
    from .configuration_ouro import OuroConfig
except ImportError:  # pragma: no cover - fallback for script usage
    # Script/local usage
    from configuration_ouro import OuroConfig


logger = logging.getLogger(__name__)

# P3: transformers 5 renamed input_embeds to inputs_embeds and dropped cache_position.
_MASK_FN_TAKES_INPUTS_EMBEDS = "inputs_embeds" in inspect.signature(create_causal_mask).parameters

try:
    from torch.distributed.tensor import DTensor as TorchDTensor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TorchDTensor = None


def needs_universal_cache(
    cache: Optional[Cache], max_cache_size: Optional[int]
) -> bool:
    if cache is None:
        return True
    if isinstance(cache, UniversalTransformerCache):
        return False
    if not isinstance(cache, Cache):
        return False
    can_grow = getattr(cache, "layer_class_to_replicate", None) is not None
    if can_grow:
        # Dynamic caches can extend to any index, so let them be
        return False
    cache_layers = getattr(cache, "layers", [])
    if max_cache_size is not None and len(cache_layers) < max_cache_size:
        try:
            cached_tokens = cache.get_seq_length()
        except Exception:
            cached_tokens = 0
        if cached_tokens > 0:
            raise ValueError(
                "The provided cache cannot store all Universal Transformer iterations. Please "
                "instantiate Ouro.modeling_ouro.UniversalTransformerCache and pass it as past_key_values."
            )
        return True
    return False


class OuroMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Keep this path out-of-place for checkpoint/autograd safety.
        if self.config.hidden_act in ("silu", "swish"):
            gate = F.silu(gate)
        else:
            gate = self.act_fn(gate)

        gate = gate * up
        return self.down_proj(gate)


@dataclass
class OuroCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Output type for [`OuroForCausalLM`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss returned when `labels` are provided.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head.
        past_key_values (`Cache`, *optional*):
            Cache returned by the model when `use_cache=True`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states returned by the base model when requested.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights returned by the base model when requested.
        loop_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states produced at each Universal Transformer iteration.
        loop_gate_logits (`tuple(torch.FloatTensor)`, *optional*):
            Early-exit gate logits produced at each Universal Transformer iteration.
    """
    loop_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    loop_gate_logits: Optional[tuple[torch.FloatTensor, ...]] = None


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Memory-lean RoPE: avoid constructing full rotate_half(q/k) tensors.
    # Equivalent to:
    #   q_embed = q * cos + rotate_half(q) * sin
    # but built from half-chunks.
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    cos1, cos2 = cos.chunk(2, dim=-1)
    sin1, sin2 = sin.chunk(2, dim=-1)

    q_embed = torch.cat((q1 * cos1 - q2 * sin1, q2 * cos2 + q1 * sin2), dim=-1)
    k_embed = torch.cat((k1 * cos1 - k2 * sin1, k2 * cos2 + k1 * sin2), dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class UniversalTransformerCache(Cache):
    """Cache implementation that supports Ouro's multi-step Universal Transformer loops."""

    def __init__(self, max_cache_size: Optional[int] = None):
        # We intentionally don't call super().__init__ because the parent assumes static cache sizes.
        self.key_cache: list[Optional[torch.Tensor]] = []
        self.value_cache: list[Optional[torch.Tensor]] = []
        self.layers: list[Any] = []  # attribute expected by HF Cache utilities
        self._seen_tokens = 0
        self.max_cache_size = max_cache_size

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")

        if self.max_cache_size is not None and layer_idx >= self.max_cache_size:
            raise IndexError(
                f"Cache index {layer_idx} exceeds configured max_cache_size={self.max_cache_size}. "
                "Check total_ut_steps and num_hidden_layers."
            )

        # Expand cache storage so the requested index is available.
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        cached_key = self.key_cache[layer_idx]
        cached_value = self.value_cache[layer_idx]

        if cached_key is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            if (
                key_states.shape[0] != cached_key.shape[0]
                or key_states.shape[1] != cached_key.shape[1]
                or key_states.shape[3] != cached_key.shape[3]
            ):
                raise ValueError(
                    "Cached and incoming key/value tensors must match on batch, head, and head_dim dimensions."
                )
            assert cached_value is not None
            self.key_cache[layer_idx] = torch.cat([cached_key, key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([cached_value, value_states], dim=2)

        result_key = self.key_cache[layer_idx]
        result_value = self.value_cache[layer_idx]
        assert result_key is not None and result_value is not None

        # Track sequence length using the first populated cache entry.
        self._seen_tokens = result_key.shape[2]
        return result_key, result_value

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if layer_idx < 0 or len(self.key_cache) <= layer_idx:
            return 0
        cached = self.key_cache[layer_idx]
        if cached is None:
            return 0
        return cached.shape[2]

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: Optional[int] = 0
    ) -> tuple[int, int]:
        # (kv_length, kv_offset) used by create_causal_mask to size the attention
        # mask. The base Cache.get_mask_sizes assumes HF's per-layer CacheLayer
        # layout (self.layers), which this slot-based cache never populates, so it
        # returns a wrong kv_length -> at decode the mask collapses to length 1 and
        # left-padded cached K/V go unmasked, corrupting batched generation.
        # All UT/layer slots share the same length at mask-build time (before this
        # step's updates), so slot 0 is representative; full attention -> offset 0.
        # P4: transformers 5 passes the query length as an int. Version 4.57 passes cache_position.
        if isinstance(cache_position, int):
            query_length = cache_position
        else:
            query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(0)
        return past_seen_tokens + query_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for idx, (key_entry, value_entry) in enumerate(
            zip(self.key_cache, self.value_cache)
        ):
            if key_entry is None:
                continue
            assert value_entry is not None
            device = key_entry.device
            self.key_cache[idx] = key_entry.index_select(0, beam_idx.to(device))
            self.value_cache[idx] = value_entry.index_select(0, beam_idx.to(device))

    @property
    def is_compileable(self) -> bool:
        return False

    def clear(self) -> None:
        logger.debug("Clearing UniversalTransformerCache")
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


_KERNELS_FLASH_ATTN3 = None
_KERNELS_FLASH_ATTN3_SIG = None


def kernels_flash_attention_3_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    sliding_window: Optional[int] = None,
    **kwargs: Unpack[TransformersKwargs],
):
    del attention_mask, kwargs  # kernels backend handles causal masking internally
    global _KERNELS_FLASH_ATTN3, _KERNELS_FLASH_ATTN3_SIG
    if _KERNELS_FLASH_ATTN3 is None and get_kernel is None:
        raise ImportError(
            "attn_implementation='kernels-community/vllm-flash-attn3' needs the `kernels` package."
        )
    if _KERNELS_FLASH_ATTN3 is None:
        local_kernel_repo_path = getattr(module.config, "_attn_kernel_repo_path", None)
        if not local_kernel_repo_path:
            default_attn_kernel_path = os.environ.get("ORBIT_ATTN_KERNEL_PATH") or None
            if default_attn_kernel_path is not None:
                local_kernel_repo_path = str(default_attn_kernel_path)
        if local_kernel_repo_path:
            _KERNELS_FLASH_ATTN3 = get_local_kernel(
                Path(local_kernel_repo_path).expanduser().resolve(),
                "vllm_flash_attn3",
            )
        else:
            _KERNELS_FLASH_ATTN3 = get_kernel("kernels-community/vllm-flash-attn3")

    flash_attn_func = _KERNELS_FLASH_ATTN3.flash_attn_func
    if _KERNELS_FLASH_ATTN3_SIG is None:
        _KERNELS_FLASH_ATTN3_SIG = inspect.signature(flash_attn_func)
    supported_params = _KERNELS_FLASH_ATTN3_SIG.parameters
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # FA3 kernels accept only fp16/bf16 (and fp8) inputs. In an fp32 run, cast the
    # attention operands to bf16 for the kernel; softmax and the QK^T / P@V
    # accumulation stay fp32 inside FA3, and the output is cast back so the rest
    # of the model keeps running in fp32 (fp32 master weights are untouched).
    # For bf16/fp16 runs compute_dtype == orig_dtype, so the casts are no-ops.
    orig_dtype = query.dtype
    compute_dtype = orig_dtype
    if compute_dtype not in (torch.float16, torch.bfloat16):
        compute_dtype = torch.bfloat16

    q = query.transpose(1, 2).contiguous().to(compute_dtype)
    k = key_states.transpose(1, 2).contiguous().to(compute_dtype)
    v = value_states.transpose(1, 2).contiguous().to(compute_dtype)

    window_size = (-1, -1)
    if sliding_window is not None:
        window = int(sliding_window)
        window_size = (window, window)

    call_kwargs = {}
    if "dropout_p" in supported_params:
        call_kwargs["dropout_p"] = float(dropout)
    elif "dropout" in supported_params:
        call_kwargs["dropout"] = float(dropout)
    elif "dropout_prob" in supported_params:
        call_kwargs["dropout_prob"] = float(dropout)

    if "softmax_scale" in supported_params:
        call_kwargs["softmax_scale"] = float(scaling)
    elif "scale" in supported_params:
        call_kwargs["scale"] = float(scaling)

    if "causal" in supported_params:
        call_kwargs["causal"] = True
    if "window_size" in supported_params:
        call_kwargs["window_size"] = window_size
    else:
        if "window_size_left" in supported_params:
            call_kwargs["window_size_left"] = window_size[0]
        if "window_size_right" in supported_params:
            call_kwargs["window_size_right"] = window_size[1]
        if "window_left" in supported_params:
            call_kwargs["window_left"] = window_size[0]
        if "window_right" in supported_params:
            call_kwargs["window_right"] = window_size[1]

    try:
        result = flash_attn_func(q, k, v, **call_kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "kernels-community/vllm-flash-attn3 flash_attn_func signature mismatch: "
            f"signature={_KERNELS_FLASH_ATTN3_SIG}, kwargs={call_kwargs}, error={exc}"
        ) from exc
    if isinstance(result, tuple):
        attn_output = result[0]
    else:
        attn_output = result
    if attn_output.dtype != orig_dtype:
        attn_output = attn_output.to(orig_dtype)
    return attn_output, None


class OuroAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        current_ut: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                current_ut * self.config.num_hidden_layers + self.layer_idx,
                cache_kwargs,
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation == "kernels-community/vllm-flash-attn3":
            attention_interface = kernels_flash_attention_3_forward
        elif self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_kernel_forward_from_hub("RMSNorm")
class OuroRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        OuroRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # Prefer fused kernel when available: lower memory traffic and better throughput.
        rms_norm = getattr(F, "rms_norm", None)
        if rms_norm is not None:
            return rms_norm(hidden_states, self.weight.shape, self.weight, self.variance_epsilon)

        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        scale = torch.rsqrt(variance + self.variance_epsilon).to(dtype=input_dtype)
        return self.weight * (hidden_states * scale)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class OuroDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OuroConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OuroAttention(config=config, layer_idx=layer_idx)

        self.mlp = OuroMLP(config)
        self.input_layernorm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = OuroRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = OuroRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_2 = OuroRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class OuroPreTrainedModel(PreTrainedModel):
    config: OuroConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OuroDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": OuroDecoderLayer,
        "attentions": OuroAttention,
    }


class OuroRotaryEmbedding(nn.Module):
    def __init__(self, config: OuroConfig, device=None):
        super().__init__()
        # P5: plain RoPE is computed by compute_default_rope_parameters below on every
        # transformers version. Scaled RoPE types still come from ROPE_INIT_FUNCTIONS.
        # The config is only read here, never modified.
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_type = "default"
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        self.rope_type = rope_type
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        if self.rope_type == "default":
            self.rope_init_fn = self.compute_default_rope_parameters
        else:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
        # Same formula as transformers 4.57's _compute_default_rope_parameters.
        base = getattr(config, "rope_theta", None)
        if base is None:
            base = config.rope_parameters["rope_theta"]
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        exponents = torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim
        return 1.0 / (base**exponents), 1.0

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class OuroModel(OuroPreTrainedModel):
    def __init__(self, config: OuroConfig):
        super().__init__(config)
        # P7: the frozen embedding row is the one training froze, recorded by prepare_model.
        # None means that no row is frozen.
        self.padding_idx = config.embedding_padding_idx
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                OuroDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = OuroRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._collect_norm_diagnostics = False
        self._norm_diag_sums: dict[str, torch.Tensor] = {}
        self._norm_diag_counts: dict[str, int] = {}
        self.rotary_emb = OuroRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.total_ut_steps = getattr(self.config, "total_ut_steps", 4)
        self.use_exit_gate = bool(getattr(self.config, "use_exit_gate", True))
        self.embedding_injection_mode = str(
            getattr(self.config, "embedding_injection_mode", "none") or "none"
        )
        self.embedding_injection_s0_mode = str(
            getattr(
                self.config,
                "embedding_injection_s0_mode",
                "learned_zero",
            )
            or "learned_zero"
        )
        self.embedding_injection_projection_mode = str(
            getattr(
                self.config,
                "embedding_injection_projection_mode",
                "shared",
            )
            or "shared"
        )
        self.use_embedding_injection = self.embedding_injection_mode != "none"
        self.early_exit_gate = nn.Linear(config.hidden_size, 1)
        self.embedding_injection_proj = (
            nn.Linear(config.hidden_size * 2, config.hidden_size)
            if self.use_embedding_injection
            and self.embedding_injection_projection_mode == "shared"
            else None
        )
        if (
            self.use_embedding_injection
            and self.embedding_injection_projection_mode == "per_step"
        ):
            self.embedding_injection_step_projs = nn.ModuleList(
                [
                    nn.Linear(config.hidden_size * 2, config.hidden_size)
                    for _ in range(int(self.total_ut_steps))
                ]
            )
        else:
            self.embedding_injection_step_projs = nn.ModuleList()
        if (
            self.use_embedding_injection
            and self.embedding_injection_s0_mode == "learned_zero"
        ):
            self.embedding_injection_s0 = nn.Parameter(
                torch.zeros(config.hidden_size)
            )
        else:
            self.register_parameter("embedding_injection_s0", None)
        if not self.use_exit_gate:
            for param in self.early_exit_gate.parameters():
                param.requires_grad_(False)
        # Initialize weights and apply final processing
        self.post_init()
        if self.embedding_injection_s0 is not None:
            nn.init.zeros_(self.embedding_injection_s0)

    def set_norm_diagnostics(self, enabled: bool) -> None:
        self._collect_norm_diagnostics = bool(enabled)

    def reset_norm_diagnostics(self) -> None:
        self._norm_diag_sums = {}
        self._norm_diag_counts = {}

    def _record_norm_diag(self, name: str, value: torch.Tensor) -> None:
        value = value.detach()
        existing = self._norm_diag_sums.get(name)
        self._norm_diag_sums[name] = value if existing is None else existing + value
        self._norm_diag_counts[name] = self._norm_diag_counts.get(name, 0) + 1

    def _register_layer_grad_hook(
        self, tensor: torch.Tensor, current_ut: int, layer_idx: int
    ) -> None:
        if not tensor.requires_grad:
            return
        name = f"act_grad_norm/ut{current_ut}_step_layer_{layer_idx}"

        def _hook(grad: torch.Tensor) -> None:
            self._record_norm_diag(name, grad.detach().float().norm())

        tensor.register_hook(_hook)

    def pop_norm_diagnostics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, summed in self._norm_diag_sums.items():
            count = self._norm_diag_counts.get(name, 0)
            if count > 0:
                out[name] = float((summed / count).item())
        self.reset_norm_diagnostics()
        return out

    def _initial_embedding_injection_state(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if self.embedding_injection_s0_mode == "learned_zero":
            if self.embedding_injection_s0 is None:
                raise RuntimeError(
                    "embedding_injection_s0_mode='learned_zero' requires a learned s0 parameter."
                )
            s0 = self.embedding_injection_s0.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            return s0.view(1, 1, -1).expand_as(inputs_embeds)

        if self.embedding_injection_s0_mode == "trunc_normal":
            std = float(inputs_embeds.shape[-1]) ** -0.5
            s0 = torch.empty_like(inputs_embeds)
            nn.init.trunc_normal_(
                s0,
                mean=0.0,
                std=std,
                a=-3.0 * std,
                b=3.0 * std,
            )
            return s0

        raise RuntimeError(
            f"Unsupported embedding_injection_s0_mode={self.embedding_injection_s0_mode!r}"
        )

    def _embedding_injection_projection(self, current_ut: int) -> nn.Linear:
        if self.embedding_injection_projection_mode == "shared":
            if self.embedding_injection_proj is None:
                raise RuntimeError(
                    "embedding_injection_projection_mode='shared' requires a shared projection."
                )
            return self.embedding_injection_proj

        if self.embedding_injection_projection_mode == "per_step":
            if current_ut >= len(self.embedding_injection_step_projs):
                raise RuntimeError(
                    "embedding_injection_projection_mode='per_step' requires at least "
                    f"{current_ut + 1} projections, got {len(self.embedding_injection_step_projs)}."
                )
            return self.embedding_injection_step_projs[current_ut]

        raise RuntimeError(
            "Unsupported embedding_injection_projection_mode="
            f"{self.embedding_injection_projection_mode!r}"
        )

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        exit_at_step: Optional[int] = None,
        exit_threshold: Optional[float] = None,
        collect_loop_hidden_states: bool = True,
        collect_loop_gate_logits: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = self.config.use_cache

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing during training. "
                "Falling back to `use_cache=False`."
            )
            use_cache = False

        max_cache_size: Optional[int] = None
        if use_cache:
            total_ut_steps = getattr(self.config, "total_ut_steps", 1) or 1
            total_layers = getattr(self.config, "num_hidden_layers", None)
            if total_layers is not None:
                max_cache_size = total_layers * total_ut_steps

            if needs_universal_cache(past_key_values, max_cache_size):
                past_key_values = UniversalTransformerCache(max_cache_size)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            if (
                attention_mask is not None
                and not isinstance(attention_mask, dict)
                and attention_mask.dim() == 2
            ):
                # Derive positions from the 2D mask so left-padded batches get
                # correct RoPE positions (first real token -> 0). Without this,
                # batched generation with left padding produces garbage.
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 1)
                position_ids = position_ids[:, -inputs_embeds.shape[1]:]
            else:
                position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            if _MASK_FN_TAKES_INPUTS_EMBEDS:
                mask_kwargs["inputs_embeds"] = inputs_embeds
            else:
                mask_kwargs["input_embeds"] = inputs_embeds
                mask_kwargs["cache_position"] = cache_position
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds
        injection_embeddings = inputs_embeds
        recurrent_state = (
            self._initial_embedding_injection_state(inputs_embeds)
            if self.use_embedding_injection
            else None
        )
        collect_norm_diag = bool(getattr(self, "_collect_norm_diagnostics", False))
        if collect_norm_diag:
            self._record_norm_diag(
                "output_norm/token_embed_norm",
                inputs_embeds.detach().float().norm(dim=-1).mean(),
            )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states_list = []
        gate_list = []
        max_ut_steps = int(self.total_ut_steps)
        if exit_at_step is not None:
            max_ut_steps = min(max_ut_steps, max(0, int(exit_at_step)) + 1)
        threshold_remaining_prob = None
        threshold_cumulative_prob = None

        for current_ut in range(max_ut_steps):
            if self.use_embedding_injection:
                if recurrent_state is None:
                    raise RuntimeError(
                        "Embedding injection is enabled but its recurrent state is not initialized."
                    )
                projection = self._embedding_injection_projection(current_ut)
                hidden_states = projection(
                    torch.cat((injection_embeddings, recurrent_state), dim=-1)
                )
            for layer_idx, decoder_layer in enumerate(
                self.layers[: self.config.num_hidden_layers]
            ):
                layer_attention_mask = causal_mask_mapping[decoder_layer.attention_type]
                if self.gradient_checkpointing and self.training:
                    # Avoid manual outer checkpoint wrapper here: decoder_layer inherits
                    # GradientCheckpointingLayer and applies its own checkpointing path.
                    # Wrapping again leads to nested checkpointing and severe slowdown.
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=layer_attention_mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        use_cache=False,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        current_ut=current_ut,
                        **kwargs,
                    )
                else:
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=layer_attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        current_ut=current_ut,
                        **kwargs,
                    )

                if collect_norm_diag:
                    self._record_norm_diag(
                        f"output_norm/ut{current_ut}_step_layer_{layer_idx}",
                        hidden_states.detach().float().norm(dim=-1).mean(),
                    )
                    self._register_layer_grad_hook(
                        hidden_states, current_ut, layer_idx
                    )

            hidden_states = self.norm(hidden_states)
            if collect_norm_diag:
                self._record_norm_diag(
                    f"output_norm/ut{current_ut}_step_final",
                    hidden_states.detach().float().norm(dim=-1).mean(),
                )
            if self.use_embedding_injection:
                recurrent_state = hidden_states
            if collect_loop_hidden_states:
                hidden_states_list.append(hidden_states)
            if self.use_exit_gate and (
                collect_loop_gate_logits or exit_threshold is not None
            ):
                gate_logits = self.early_exit_gate(hidden_states)
                if collect_loop_gate_logits:
                    gate_list.append(gate_logits)
                if exit_threshold is not None:
                    lambdas = torch.sigmoid(gate_logits.squeeze(-1).float())
                    if threshold_remaining_prob is None:
                        threshold_remaining_prob = torch.ones_like(lambdas)
                    if current_ut < int(self.total_ut_steps) - 1:
                        step_prob = lambdas * threshold_remaining_prob
                        threshold_remaining_prob = threshold_remaining_prob * (
                            1.0 - lambdas
                        )
                    else:
                        step_prob = threshold_remaining_prob
                    threshold_cumulative_prob = (
                        step_prob
                        if threshold_cumulative_prob is None
                        else threshold_cumulative_prob + step_prob
                    )
                    threshold_value = exit_threshold
                    if isinstance(threshold_value, torch.Tensor):
                        threshold_value = threshold_value.to(
                            threshold_cumulative_prob.device
                        )
                    if bool(
                        (threshold_cumulative_prob >= threshold_value).all().item()
                    ):
                        break

        return (
            BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            ),
            hidden_states_list,
            gate_list,
        )


@auto_docstring
class OuroForCausalLM(OuroPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = OuroModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 分块大小配置
        self.chunk_size = getattr(config, "chunk_size", 2)  # 默认分块大小为2
        self.early_exit_step = getattr(config, "early_exit_step", None)
        self.early_exit_threshold = getattr(config, "early_exit_threshold", None)

        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        use_weighted_exit: Optional[bool] = False,  # 控制是否使用加权 early exit
        exit_at_step: Optional[int] = None,
        exit_threshold: Optional[float] = None,
        return_orbit_aux: bool = False,
        compute_logits: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> OuroCausalLMOutputWithPast:
        r"""
        Args:
            use_weighted_exit (`bool`, *optional*, defaults to `False`):
                Whether to use weighted early exit. If `True`, the logits from all UT steps will be
                averaged according to the exit probability distribution.
            exit_at_step (`int`, *optional*):
                Specifies which UT step to exit at. If set, the model will directly use the hidden states
                from this step to generate logits, ignoring other exit strategies.
            exit_threshold (`float`, *optional*):
                The cumulative probability threshold for early exit. When the cumulative exit probability
                reaches this threshold, the model will exit at that step.
            return_orbit_aux (`bool`, *optional*, defaults to `False`):
                Whether to include per-iteration hidden states and gate logits in the returned output.
            compute_logits (`bool`, *optional*, defaults to `True`):
                Whether to materialize logits inside the forward pass. Set this to `False` for custom
                training paths that consume hidden states directly and want to avoid the extra projection.
        Example:
        ```python
        >>> from transformers import AutoTokenizer, OuroForCausalLM
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        exit_at_step = (
            exit_at_step if exit_at_step is not None else self.early_exit_step
        )
        exit_threshold = (
            exit_threshold if exit_threshold is not None else self.early_exit_threshold
        )
        will_return_cache = bool(
            self.config.use_cache if use_cache is None else use_cache
        )
        model_exit_at_step = None
        model_exit_threshold = None
        needs_threshold_selection = (
            exit_threshold is not None and labels is None and compute_logits
        )
        needs_weighted_exit = bool(use_weighted_exit)
        collect_loop_aux = bool(
            return_orbit_aux
            or labels is not None
            or needs_weighted_exit
            or needs_threshold_selection
        )
        if labels is None and compute_logits and not use_weighted_exit:
            if exit_at_step is not None:
                model_exit_at_step = int(exit_at_step)
            elif exit_threshold is not None and not will_return_cache:
                # Dynamic q-exit with KV cache can leave deeper recurrent-step
                # cache entries missing for future tokens. Keep cached generation
                # semantically equivalent to the full-depth path.
                model_exit_threshold = exit_threshold

        outputs, hidden_states_list, gate_list = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            exit_at_step=model_exit_at_step,
            exit_threshold=model_exit_threshold,
            collect_loop_hidden_states=collect_loop_aux,
            collect_loop_gate_logits=collect_loop_aux,
            **kwargs,
        )
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        def _select_token_positions(tensor: torch.Tensor) -> torch.Tensor:
            if isinstance(slice_indices, slice):
                return tensor[:, slice_indices, ...]
            if isinstance(slice_indices, torch.Tensor):
                return tensor.index_select(1, slice_indices.to(tensor.device))
            raise TypeError(
                f"Unsupported index type for logits_to_keep: {type(slice_indices)}"
            )

        stacked_exit_pdf = None
        if gate_list:
            pdf_list = []
            total_ut_steps = int(
                getattr(
                    self.model,
                    "total_ut_steps",
                    getattr(self.config, "total_ut_steps", len(gate_list)),
                )
                or len(gate_list)
            )
            remaining_prob = torch.ones_like(
                gate_list[0].squeeze(-1),
                dtype=torch.float32,
            )
            for idx, gate_tensor in enumerate(gate_list):
                lambda_i = torch.sigmoid(gate_tensor.squeeze(-1).float())
                if idx < total_ut_steps - 1:
                    p_i = lambda_i * remaining_prob
                    remaining_prob = remaining_prob * (1.0 - lambda_i)
                else:
                    p_i = remaining_prob
                pdf_list.append(p_i)
            stacked_exit_pdf = torch.stack(pdf_list, dim=2)

        expected_logits_cache: Optional[torch.Tensor] = None

        def compute_expected_logits() -> Optional[torch.Tensor]:
            nonlocal expected_logits_cache
            if expected_logits_cache is not None:
                return expected_logits_cache
            if stacked_exit_pdf is None or not hidden_states_list:
                return None
            token_exit_pdf = _select_token_positions(stacked_exit_pdf)
            expected_logits = None
            for step_idx, hidden in enumerate(hidden_states_list):
                step_hidden = _select_token_positions(hidden)
                step_logits = self.lm_head(step_hidden)
                weight = (
                    token_exit_pdf[..., step_idx].unsqueeze(-1).to(step_logits.dtype)
                )
                expected_logits = (
                    step_logits * weight
                    if expected_logits is None
                    else expected_logits + step_logits * weight
                )
            expected_logits_cache = expected_logits
            return expected_logits_cache

        logits: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None

        if labels is not None:
            logits = compute_expected_logits()
            if logits is None:
                hidden_states = outputs.last_hidden_state
                logits = self.lm_head(_select_token_positions(hidden_states))
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        elif compute_logits:
            if stacked_exit_pdf is not None and hidden_states_list:
                if exit_at_step is not None and 0 <= exit_at_step < len(
                    hidden_states_list
                ):
                    selected_hidden = hidden_states_list[exit_at_step]
                    logits = self.lm_head(_select_token_positions(selected_hidden))
                elif exit_threshold is not None:
                    cumulative_probs = torch.cumsum(stacked_exit_pdf, dim=2)
                    threshold_value = exit_threshold
                    if isinstance(threshold_value, torch.Tensor):
                        threshold_value = threshold_value.to(cumulative_probs.device)
                    threshold_mask = cumulative_probs >= threshold_value
                    exit_steps = torch.argmax(threshold_mask.float(), dim=2)
                    last_step_idx = stacked_exit_pdf.shape[2] - 1
                    if last_step_idx >= 0:
                        never_exceeded = ~threshold_mask.any(dim=2)
                        exit_steps[never_exceeded] = last_step_idx
                    stacked_hidden = torch.stack(hidden_states_list, dim=2)
                    gather_index = (
                        exit_steps.unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand(-1, -1, 1, stacked_hidden.size(-1))
                    )
                    final_hidden_states = torch.gather(
                        stacked_hidden, 2, gather_index
                    ).squeeze(2)
                    logits = self.lm_head(_select_token_positions(final_hidden_states))
                elif use_weighted_exit:
                    logits = compute_expected_logits()

            if logits is None:
                hidden_states = outputs.last_hidden_state
                logits = self.lm_head(_select_token_positions(hidden_states))

        result = OuroCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loop_hidden_states=tuple(hidden_states_list) if return_orbit_aux else None,
            loop_gate_logits=tuple(gate_list) if return_orbit_aux else None,
        )

        return result


class OuroForSequenceClassification(
    GenericForSequenceClassification, OuroPreTrainedModel
):
    pass


class OuroForTokenClassification(GenericForTokenClassification, OuroPreTrainedModel):
    pass


class OuroForQuestionAnswering(GenericForQuestionAnswering, OuroPreTrainedModel):
    base_model_prefix = (
        "transformer"  # For BC, where `transformer` was used instead of `model`
    )


__all__ = [
    "OuroPreTrainedModel",
    "OuroModel",
    "OuroForCausalLM",
    "OuroForSequenceClassification",
    "OuroForTokenClassification",
    "OuroForQuestionAnswering",
    "UniversalTransformerCache",
]
