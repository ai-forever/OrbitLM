"""Per-loop token log-probs computed inside the OrbitLM forward, and the RLTT loop weights.

RLTT trains every loop of the looped model, not only the last one. The policy gradient uses
``sum_t w_t * log P^(t)(y)``, where ``P^(t)`` is the next-token distribution read out after loop
``t`` and ``w_t`` is a loop weight. With the ``exit`` scheme the weight is the per-token exit
probability from the model's early-exit gate. It enters the loss as a constant unless
``orbit.rltt.train_exit_gate`` is set, in which case the loss trains the gate as well (see
``loop_weights``).

verl's engine only ever asks the model for logits. ``install_rltt_forward`` therefore patches the
model class so that a forward call with ``rltt_labels`` returns, instead of logits:

- ``loop_log_probs``: the log-prob of each label token under every loop, ``[B, L, T]``;
- ``log_probs``: the same for the final loop only, ``[B, L]`` (verl's KL and old-log-prob logic
  are defined on the final loop);
- ``exit_pdf``: the exit distribution over loops built from the gate logits, ``[B, L, T]``;
- ``entropy``: the final loop's token entropy, only when asked for.

All of this has to happen inside the model's forward, because FSDP gathers the full weights only
while the module that owns them is running. The ``early_exit_gate`` sits in the root FSDP unit and
is called by the model as usual. The ``lm_head`` is in the root unit under FSDP1, but verl's FSDP2
setup shards it as a unit of its own, so its full weight exists only during a call of ``lm_head``
itself. The per-loop log-probs are therefore computed as one call of the ``lm_head`` module (see
``_per_loop_log_probs``), which works the same with FSDP1, FSDP2 and no FSDP.

The log-probs come from verl's ``FusedLinearForPPO``. It computes them chunk by chunk with a
custom backward, so a ``[tokens, vocab]`` logits tensor is never kept for backward. With four loops
and a 152k vocabulary, keeping those logits would cost several times the memory of the model.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers.utils import ModelOutput

CHUNK_SIZE_ENV = "ORBIT_RLTT_CHUNK_SIZE"
DEFAULT_CHUNK_SIZE = 512

# Loop-weight schemes of the RLTT loss. "last" is plain GRPO on the final loop, expressed through
# the RLTT code path; it exists so the RLTT path can be checked against stock GRPO.
LOOP_WEIGHT_SCHEMES = ("exit", "uniform", "progressive", "last")

# Attribute names stored on the patched model class.
_ORIGINAL_FORWARD_ATTR = "_orbit_rltt_original_forward"
_CHUNK_SIZE_ATTR = "_orbit_rltt_chunk_size"
_WRAPPER_MARK = "_orbit_rltt_wrapper"


@dataclass
class OrbitLMRLTTOutput(ModelOutput):
    """What the patched forward returns on the RLTT path. Shapes use ``L`` for the token axis."""

    log_probs: Optional[torch.Tensor] = None  # [B, L], final loop
    loop_log_probs: Optional[torch.Tensor] = None  # [B, L, T], differentiable
    exit_pdf: Optional[torch.Tensor] = None  # [B, L, T], differentiable through the gate
    entropy: Optional[torch.Tensor] = None  # [B, L], final loop, only when requested


def build_exit_pdf(gate_logits: list[torch.Tensor]) -> torch.Tensor | None:
    """The exit distribution over loops, ``[B, L, T]``, from the per-loop gate logits ``[B, L, 1]``.

    Loop ``t`` exits with probability ``lambda_t`` times the probability that no earlier loop exited,
    where ``lambda_t`` is the sigmoid of its gate logit. The last loop takes all the remaining mass.
    The rows are renormalised to sum to 1. This is the trainer's ``losses.build_exit_pdf``, kept
    identical so that RL weights the loops exactly as pretraining's exit loss does.
    """
    if not gate_logits:
        return None
    eps = 1e-6
    pdf_list = []
    remaining_prob = torch.ones_like(gate_logits[0].squeeze(-1), dtype=torch.float32)
    for idx, gate_tensor in enumerate(gate_logits):
        lambda_i = torch.sigmoid(gate_tensor.squeeze(-1).float()).clamp(min=eps, max=1.0 - eps)
        if idx < len(gate_logits) - 1:
            p_i = lambda_i * remaining_prob
            remaining_prob = remaining_prob * (1.0 - lambda_i)
        else:
            p_i = remaining_prob.clamp(min=0.0, max=1.0)
        pdf_list.append(p_i)
    stacked = torch.stack(pdf_list, dim=2)
    norm = stacked.sum(dim=2, keepdim=True).clamp_min(1e-12)
    return stacked / norm


def static_loop_weights(scheme: str, num_loops: int, *, alpha: float = 1.0, device=None,
                        dtype=torch.float32) -> torch.Tensor:
    """The loop weights of a scheme that does not depend on the token, as a vector ``[T]`` summing to 1.

    ``last`` puts all weight on the final loop, ``uniform`` spreads it evenly, and ``progressive``
    weighs loop ``t`` (counted from 1) by ``t ** alpha``.
    """
    loops = int(num_loops)
    if loops <= 0:
        raise ValueError("num_loops must be positive.")
    if scheme == "last":
        weights = torch.zeros(loops, device=device, dtype=dtype)
        weights[-1] = 1.0
    elif scheme == "uniform":
        weights = torch.full((loops,), 1.0 / loops, device=device, dtype=dtype)
    elif scheme == "progressive":
        weights = torch.arange(1, loops + 1, device=device, dtype=dtype).pow(float(alpha))
        weights = weights / weights.sum()
    else:
        raise ValueError(f"{scheme!r} has no static loop weights; use last, uniform or progressive.")
    return weights


def rltt_chunk_size() -> int:
    """Token chunk size for the per-loop log-prob computation, from ``ORBIT_RLTT_CHUNK_SIZE``."""
    raw = os.environ.get(CHUNK_SIZE_ENV, "").strip()
    if not raw:
        return DEFAULT_CHUNK_SIZE
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{CHUNK_SIZE_ENV}={raw!r} is not an integer.") from exc
    if value <= 0:
        raise ValueError(f"{CHUNK_SIZE_ENV} must be positive, got {value}.")
    return value


def loop_weights(
    scheme: str,
    exit_pdf: Optional[torch.Tensor],
    alpha: float = 1.0,
    *,
    num_loops: Optional[int] = None,
    train_exit_gate: bool = False,
) -> torch.Tensor:
    """Return the loop weights ``w`` for ``(loop_log_probs * w).sum(-1)``.

    For ``exit`` this is the exit pdf, shape ``[..., T]``. By default it is detached, so the weights
    are constants of the policy gradient, as in Eq. 4 of the RLTT paper, and the RL loss does not
    train the exit gate. With ``train_exit_gate=True`` the weights keep their gradient and the loss
    trains the gate too, as the RLTT reference code does. That gradient is
    ``-A * log P^(t)`` for loop ``t``. The first loop has the lowest log-prob, so every token with a
    negative advantage pushes the gate toward exiting at loop 0. Over a long run that can move most
    of the gate's mass to loop 0 (docs/verl/DESIGN.md).

    For ``uniform``, ``progressive`` and ``last`` the weights do not depend on the token, so a
    vector of shape ``[T]`` is returned and broadcasts against ``[..., T]`` (``static_loop_weights``). ``alpha``
    is the exponent of the progressive scheme. The number of loops is read from ``exit_pdf``, or
    from ``num_loops`` when the model has no exit gate and ``exit_pdf`` is None. These static
    weights never give the gate a gradient, so ``train_exit_gate`` does not apply to them.
    """
    if scheme not in LOOP_WEIGHT_SCHEMES:
        raise ValueError(f"Unknown loop-weight scheme {scheme!r}; expected one of {LOOP_WEIGHT_SCHEMES}.")
    if scheme == "exit":
        if exit_pdf is None:
            raise ValueError(
                "The 'exit' loop weights need the exit pdf, but the model returned no gate logits. "
                "Check that the model config has use_exit_gate=true."
            )
        return exit_pdf if train_exit_gate else exit_pdf.detach()
    if exit_pdf is not None:
        loops, device = int(exit_pdf.shape[-1]), exit_pdf.device
    elif num_loops is not None:
        loops, device = int(num_loops), None
    else:
        raise ValueError("loop_weights needs exit_pdf or num_loops to know the number of loops.")
    return static_loop_weights(scheme, loops, alpha=float(alpha), device=device, dtype=torch.float32)


_NO_ATTRIBUTE = object()


def stock_temperature(temperature: float, weight: torch.Tensor) -> float:
    """Return the temperature that verl's stock log-prob path effectively divides the logits by.

    verl casts the temperature to the dtype of the logits before dividing. Under bf16 autocast the
    logits are bf16, and 0.9 becomes 0.8984375 there. The old-log-prob and reference passes use
    that stock path, so the RLTT final loop has to divide by the same rounded number. With the
    exact 0.9 instead, the final-loop log-probs differed from the stock ones by 0.06 on average on
    the H200, and the KL term started at 0.004 instead of 0.

    The logits dtype is the autocast dtype when autocast is on, because the lm_head matmul is
    autocast. Otherwise it is the dtype of the lm_head weight.
    """
    device_type = weight.device.type
    if torch.is_autocast_enabled(device_type):
        logits_dtype = torch.get_autocast_dtype(device_type)
    else:
        logits_dtype = weight.dtype
    return float(torch.tensor(float(temperature), dtype=torch.float32).to(logits_dtype))


def _per_loop_log_probs(head, hidden_states, labels, temperature: float, chunk_size: int, need_entropy: bool):
    """Compute every loop's token log-probs as a single call of the ``lm_head`` module.

    Under FSDP2 the ``lm_head`` weight is only gathered while the module itself runs; read from
    outside, ``head.weight`` is the local shard. A call of the module also makes FSDP register the
    hooks that reduce the weight's gradient after backward. So the computation is installed as the
    module's forward for the duration of one call, and the original forward is restored afterwards.
    The hidden states go in as separate arguments, so FSDP sees each of them and nothing is copied.

    Returns ``(loop_log_probs [B, L, T], final_log_probs [B, L], final_entropy [B, L] or None)``.
    """
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    fused = FusedLinearForPPO(chunk_size=chunk_size)

    def per_loop_forward(*loop_hidden):
        weight = head.weight
        effective_temperature = stock_temperature(temperature, weight)
        per_loop, entropy = [], None
        for index, hidden in enumerate(loop_hidden):
            # The cast makes the matmul run in the dtype of the lm_head weight, which is what a
            # plain lm_head call does with the hidden states that come out of the final norm.
            log_probs, loop_entropy = fused(hidden.to(weight.dtype), weight, labels, effective_temperature)
            per_loop.append(log_probs)
            if need_entropy and index == len(loop_hidden) - 1:
                entropy = loop_entropy
        return torch.stack(per_loop, dim=-1), per_loop[-1], entropy

    previous = head.__dict__.get("forward", _NO_ATTRIBUTE)
    head.forward = per_loop_forward
    try:
        return head(*hidden_states)
    finally:
        if previous is _NO_ATTRIBUTE:
            del head.forward
        else:
            head.forward = previous


def _rltt_forward_body(self, original, args, kwargs, rltt_labels, rltt_temperature, rltt_need_entropy):
    """Run the original forward for all loops and turn each loop's hidden states into log-probs."""
    # exit_at_step pins the plain path to the final loop. Here every loop is needed, and the
    # original forward computes no logits at all, so the argument would only be misleading.
    kwargs.pop("exit_at_step", None)
    kwargs["return_orbit_aux"] = True
    kwargs["compute_logits"] = False
    kwargs["use_weighted_exit"] = False
    out = original(self, *args, **kwargs)

    hidden_states = out.loop_hidden_states
    gate_logits = out.loop_gate_logits
    if not hidden_states:
        raise RuntimeError(
            "The model returned no loop_hidden_states for return_orbit_aux=True. "
            "The RLTT path needs the OrbitLM (Ouro) forward contract described in docs/verl/DESIGN.md."
        )

    chunk_size = getattr(type(self), _CHUNK_SIZE_ATTR, None) or rltt_chunk_size()
    loop_log_probs, final_log_probs, entropy = _per_loop_log_probs(
        self.lm_head, tuple(hidden_states), rltt_labels, float(rltt_temperature), chunk_size, bool(rltt_need_entropy)
    )

    exit_pdf = None
    if gate_logits:
        exit_pdf = build_exit_pdf(list(gate_logits))

    return OrbitLMRLTTOutput(
        log_probs=final_log_probs,
        loop_log_probs=loop_log_probs,
        exit_pdf=exit_pdf,
        entropy=entropy,
    )


def install_rltt_forward(model, chunk_size: Optional[int] = None) -> None:
    """Patch ``type(model).forward`` so it can return per-loop log-probs. Safe to call repeatedly.

    The patched ``forward(self, *args, rltt_labels=None, rltt_temperature=1.0,
    rltt_need_entropy=False, **kwargs)`` behaves exactly like the original when ``rltt_labels`` is
    None. That is the path verl uses for rollout-side log-probs, the reference model and generation.
    With ``rltt_labels`` (int64 ``[B, L]``, the next token at each position) it returns a
    ``OrbitLMRLTTOutput``.

    The patch goes on the class, not on the instance, so every model of that class gets it. That
    includes the reference model that a second engine builds in the same process, and the subclass
    that FSDP2 generates when it wraps the module. The class keeps the original forward in an
    attribute, and the wrapper carries a marker. A second call, also for a subclass that inherits
    the patched forward, therefore changes nothing apart from ``chunk_size``.

    ``chunk_size`` overrides ``ORBIT_RLTT_CHUNK_SIZE`` for this class. When it is None the
    environment variable is read at every call, which lets the launcher set it per run.
    """
    cls = type(model)
    if chunk_size is not None:
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        setattr(cls, _CHUNK_SIZE_ATTR, int(chunk_size))
    if getattr(cls.forward, _WRAPPER_MARK, False):
        return

    original = cls.forward

    # functools.wraps keeps the original signature visible to inspect.signature. verl reads that
    # signature after building the model (it looks for a "cu_seqlens" parameter), and it must see
    # the same answer with and without this patch.
    @functools.wraps(original)
    def forward(self, *args, rltt_labels=None, rltt_temperature=1.0, rltt_need_entropy=False, **kwargs):
        if rltt_labels is None:
            return original(self, *args, **kwargs)
        return _rltt_forward_body(self, original, args, kwargs, rltt_labels, rltt_temperature, rltt_need_entropy)

    setattr(forward, _WRAPPER_MARK, True)
    setattr(cls, _ORIGINAL_FORWARD_ATTR, original)
    cls.forward = forward
