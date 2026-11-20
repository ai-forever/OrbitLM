# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# OrbitLM (Ouro) config class used by the verl port, next to modelling_ouro.py because HF loads
# both files as remote code from a prepared model dir.
# Source: the trainer's src/orbit/configuration_ouro.py as of the SFT runs. It already works on
# transformers 4.57 and 5.x, so it is copied unchanged apart from this comment and one addition:
# P7  The new field embedding_padding_idx names the token-embedding row that training kept at zero
#     and frozen. modelling_ouro.py builds the embedding from it instead of from pad_token_id
#     (docs/verl/DESIGN.md).
"""Ouro model configuration"""

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class OuroConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OuroModel`]. It is used to instantiate a
    Ouro model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Ouro-7B-beta [Qwen/Ouro-7B-beta](https://huggingface.co/Qwen/Ouro-7B-beta).
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Ouro model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`OuroModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
            additional layer afterwards will use SWA (Sliding Window Attention).
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        embedding_injection_mode (`str`, *optional*, defaults to `"none"`):
            Enables repeated input embedding injection into every recurrent step. Use `"concat"` to project
            `[input_embeddings, recurrent_state]` before each recurrent pass.
        embedding_injection_s0_mode (`str`, *optional*, defaults to `"learned_zero"`):
            Initial recurrent state for embedding injection. Use `"learned_zero"` for a learned vector initialized
            with zeros, or `"trunc_normal"` to sample `s0` from a truncated normal distribution on each forward.
        embedding_injection_projection_mode (`str`, *optional*, defaults to `"shared"`):
            Projection sharing for embedding injection. Use `"shared"` for one projection reused by every recurrent
            step, or `"per_step"` for a separate projection per recurrent step.
        embedding_padding_idx (`int`, *optional*, defaults to `None`):
            The row of the token embedding that training kept at zero and frozen. The model passes it to
            `nn.Embedding` as `padding_idx`, so that row gets no gradient. `None` means no row is frozen.
            `orbit_verl.model.prepare_model` sets it from the export's own training code and weights.
    ```python
    >>> from transformers import OuroModel, OuroConfig
    >>> # Initializing a Ouro style configuration
    >>> configuration = OuroConfig()
    >>> # Initializing a model from the Ouro-7B style configuration
    >>> model = OuroModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ouro"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Ouro`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        total_ut_steps=4,
        early_exit_threshold=1.0,
        use_exit_gate=True,
        embedding_injection_mode="none",
        embedding_injection_s0_mode="learned_zero",
        embedding_injection_projection_mode="shared",
        embedding_padding_idx=None,
        **kwargs,
    ):
        embedding_injection_mode = str(embedding_injection_mode or "none")
        embedding_injection_s0_mode = str(embedding_injection_s0_mode or "learned_zero")
        embedding_injection_projection_mode = str(
            embedding_injection_projection_mode or "shared"
        )
        valid_embedding_injection_modes = {"none", "concat"}
        valid_embedding_injection_s0_modes = {"learned_zero", "trunc_normal"}
        valid_embedding_injection_projection_modes = {"shared", "per_step"}
        if embedding_injection_mode not in valid_embedding_injection_modes:
            raise ValueError(
                "embedding_injection_mode must be one of "
                f"{sorted(valid_embedding_injection_modes)}, got {embedding_injection_mode!r}"
            )
        if embedding_injection_s0_mode not in valid_embedding_injection_s0_modes:
            raise ValueError(
                "embedding_injection_s0_mode must be one of "
                f"{sorted(valid_embedding_injection_s0_modes)}, got {embedding_injection_s0_mode!r}"
            )
        if embedding_injection_projection_mode not in valid_embedding_injection_projection_modes:
            raise ValueError(
                "embedding_injection_projection_mode must be one of "
                f"{sorted(valid_embedding_injection_projection_modes)}, "
                f"got {embedding_injection_projection_mode!r}"
            )

        # P7: a wrong value here silently freezes or unfreezes an embedding row during training,
        # so anything that is not a row index of this vocabulary is refused.
        if embedding_padding_idx is not None and (
            isinstance(embedding_padding_idx, bool)
            or not isinstance(embedding_padding_idx, int)
            or not 0 <= embedding_padding_idx < vocab_size
        ):
            raise ValueError(
                f"embedding_padding_idx must be None or a row index below vocab_size {vocab_size}, "
                f"got {embedding_padding_idx!r}. orbit_verl.model.prepare_model writes it into "
                "config.json; rebuild the model dir instead of editing the value by hand."
            )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.total_ut_steps = total_ut_steps
        self.early_exit_threshold = early_exit_threshold
        self.use_exit_gate = bool(use_exit_gate)
        self.embedding_injection_mode = embedding_injection_mode
        self.embedding_injection_s0_mode = embedding_injection_s0_mode
        self.embedding_injection_projection_mode = embedding_injection_projection_mode
        self.embedding_padding_idx = embedding_padding_idx
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["OuroConfig"]
