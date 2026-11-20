"""A frozen copy of the RLTT reference implementation, used only as the reference of the RLTT tests.

These functions are a GRPO/RLTT loss written outside verl as a port of the RLTT paper's reference
code (jonwill8/RLTT). The verl port in ``orbit_verl.rltt`` was checked against them
number for number, and ``test_rltt_forward.py`` and ``test_rltt_loss.py`` keep checking it. The code
is copied unchanged apart from this docstring and the imports, so that the comparison stays a
comparison with an independent implementation. Do not edit it to make a test pass.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint


def recompute_in_backward(fn, *tensors):
    """Run ``fn`` without keeping its activations; recompute them in backward.

    Chunking the LM head does NOT by itself bound memory. cross_entropy is
    ``nll_loss(log_softmax(x))`` and log_softmax's backward needs its own output,
    so autograd pins a ``[B, chunk, vocab]`` tensor per chunk until backward runs:
    rebinding ``logits`` each iteration frees nothing, and the total held is the
    same as computing the whole sequence at once. At vocab 49152 x 4 loops that is
    tens of GB (bs 8 x 4k context: ~6.4 GB per loop, ~26 GB total), which is memory
    the optimizer step and NCCL's collective buffers then cannot get.

    Recomputing costs one extra lm_head matmul + softmax per chunk -- cheap next to
    24 layers x 4 loops -- and leaves only one chunk's logits live at a time. Note
    the model-level ``gradient_checkpointing`` flag does not cover this: it wraps
    the transformer layers, not the head or the loss.

    Only the vocab-sized part belongs inside ``fn``; keep the small [B, chunk]
    weighting outside so its graph (and any gate gradients) stay intact.
    """
    if torch.is_grad_enabled():
        return checkpoint(fn, *tensors, use_reentrant=False)
    return fn(*tensors)


def build_exit_pdf(gate_list):
    if not gate_list:
        return None
    eps = 1e-6
    pdf_list = []
    remaining_prob = torch.ones_like(gate_list[0].squeeze(-1), dtype=torch.float32)
    for idx, gate_tensor in enumerate(gate_list):
        lambda_i = torch.sigmoid(gate_tensor.squeeze(-1).float()).clamp(min=eps, max=1.0 - eps)
        if idx < len(gate_list) - 1:
            p_i = lambda_i * remaining_prob
            remaining_prob = remaining_prob * (1.0 - lambda_i)
        else:
            p_i = remaining_prob.clamp(min=0.0, max=1.0)
        pdf_list.append(p_i)
    stacked = torch.stack(pdf_list, dim=2)
    norm = stacked.sum(dim=2, keepdim=True).clamp_min(1e-12)
    return stacked / norm


def iter_seq_chunks(seq_len: int, chunk_size: int):
    size = int(chunk_size)
    if size <= 0 or size >= int(seq_len):
        yield 0, int(seq_len)
        return
    for start in range(0, int(seq_len), size):
        end = min(int(seq_len), start + size)
        yield start, end


# ---------------------------------------------------------------------------
# Advantage
# ---------------------------------------------------------------------------

def group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    *,
    eps: float = 1e-6,
    scale_by_std: bool = True,
) -> torch.Tensor:
    """Standardized advantage within each group of ``group_size`` completions.

    ``rewards`` is 1-D of length ``num_prompts * group_size`` laid out group-major
    (all completions of prompt 0, then prompt 1, ...). Groups with zero variance
    (all completions equal) get advantage 0.
    """
    r = rewards.detach().float()
    if r.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got shape {tuple(r.shape)}.")
    if group_size <= 0 or r.numel() % group_size != 0:
        raise ValueError(
            f"len(rewards)={r.numel()} is not divisible by group_size={group_size}."
        )
    groups = r.view(-1, group_size)
    mean = groups.mean(dim=1, keepdim=True)
    adv = groups - mean
    if scale_by_std:
        std = groups.std(dim=1, keepdim=True)
        adv = adv / (std + eps)
    return adv.reshape(-1)


# ---------------------------------------------------------------------------
# Loop-weight schedules
# ---------------------------------------------------------------------------

VALID_SCHEMES = ("grpo", "exit", "progressive", "uniform", "learned")


class LearnedLoopWeights(torch.nn.Module):
    """Trainable per-loop weights (RLTT ``learned`` scheme), jointly optimized
    with the policy. Logits -> softmax(logits / temperature). Ported from the
    reference implementation (jonwill8/RLTT, rltt_algos.py)."""

    def __init__(
        self,
        num_loops: int,
        *,
        init_strategy: str = "uniform",
        init_alpha: float = 1.0,
        temperature: float = 1.0,
        device=None,
    ):
        super().__init__()
        self.num_loops = int(num_loops)
        self.temperature = float(temperature)
        if init_strategy == "progressive":
            t = torch.arange(1, num_loops + 1, dtype=torch.float32, device=device)
            target = t.pow(float(init_alpha))
            target = target / target.sum()
            init_logits = torch.log(target + 1e-8)
        else:  # "uniform" (default)
            init_logits = torch.zeros(num_loops, device=device)
        self.logits = torch.nn.Parameter(init_logits)

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits / self.temperature, dim=0)


def aggregate_multi_loop_log_probs(
    per_loop_log_probs: list[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """log P_RLTT(y) = sum_t omega_t * logP^(t)(y). ``per_loop_log_probs`` is a
    list of T tensors [B, L]; ``weights`` is [T] (static) or [B, L, T]
    (per-position, e.g. exit-pdf). Mirrors the reference function of the same name."""
    stacked = torch.stack(per_loop_log_probs, dim=-1)  # [B, L, T]
    if weights.dim() == 1:
        return (stacked * weights.view(1, 1, -1)).sum(dim=-1)
    if weights.dim() == 3:
        return (stacked * weights).sum(dim=-1)
    raise ValueError(f"Unexpected weights shape: {tuple(weights.shape)}")


def loop_weight_vector(
    scheme: str,
    num_loops: int,
    *,
    alpha: float = 1.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Per-loop weights omega_t as a 1-D tensor [T] summing to 1.

    Only used by the position-independent schemes; ``exit`` weighting is taken
    per position from the exit-pdf instead (see ``weighted_token_logprobs``).
    """
    T = int(num_loops)
    if T <= 0:
        raise ValueError("num_loops must be positive.")
    if scheme == "grpo":
        w = torch.zeros(T, device=device, dtype=dtype)
        w[-1] = 1.0
    elif scheme == "uniform":
        w = torch.full((T,), 1.0 / T, device=device, dtype=dtype)
    elif scheme == "progressive":
        idx = torch.arange(1, T + 1, device=device, dtype=dtype)
        w = idx.pow(float(alpha))
        w = w / w.sum()
    elif scheme == "exit":
        # exit weighting is per position; caller must pass exit_pdf.
        raise ValueError("scheme='exit' uses per-position weights, not a vector.")
    else:
        raise ValueError(f"Unknown loop-weight scheme '{scheme}'. Use {VALID_SCHEMES}.")
    return w


# ---------------------------------------------------------------------------
# Per-token log-probs
# ---------------------------------------------------------------------------

def token_logprobs(
    logits: torch.Tensor, targets: torch.Tensor, *, temperature: float = 1.0
) -> torch.Tensor:
    """log P(target) at each position under softmax(logits / temperature).

    The temperature MUST match the rollout sampling temperature so the training
    log-probs are on the same distribution that was sampled (reference divides
    logits by temperature before logprobs; verl standard). logits [..., V]."""
    if temperature != 1.0:
        logits = logits / float(temperature)
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1).long()).squeeze(-1)


def weighted_token_logprobs(
    hidden_states_list: list[torch.Tensor],
    gate_list: list[torch.Tensor],
    lm_head: torch.nn.Module,
    targets: torch.Tensor,
    *,
    scheme: str,
    alpha: float = 1.0,
    lm_chunk_size: int = 0,
    exit_pdf: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Combine per-loop next-token log-probs into one per-token log-prob [B, L].

    sum_t omega_t * logP^(t)(target). Chunked over the sequence so at most one
    [B, chunk, V] logits tensor per loop is materialized (the RLTT memory cost).
    Differentiable through ``hidden_states_list``, ``lm_head`` and, when
    ``weights`` come from ``LearnedLoopWeights``, the weight logits.

    ``weights`` (optional) overrides the scheme and is used directly: [T] static
    or [B, L, T] per-position. Required for ``scheme='learned'`` (pass
    ``learned_module()``); ``scheme='exit'`` builds [B, L, T] from the gate logits.
    """
    if not hidden_states_list:
        raise ValueError("hidden_states_list is empty.")
    T = len(hidden_states_list)
    B, L, _ = hidden_states_list[0].shape

    w3: Optional[torch.Tensor] = None   # [B, L, T] per-position weights
    wvec: Optional[torch.Tensor] = None  # [T] static weights
    if weights is not None:
        if weights.dim() == 3:
            w3 = weights
        elif weights.dim() == 1:
            wvec = weights
        else:
            raise ValueError(f"weights must be 1-D [T] or 3-D [B,L,T], got {tuple(weights.shape)}.")
    elif scheme == "exit":
        w3 = exit_pdf if exit_pdf is not None else build_exit_pdf(gate_list)
        if w3 is None:
            raise ValueError("scheme='exit' needs gate logits / exit_pdf.")
    elif scheme == "learned":
        raise ValueError("scheme='learned' requires passing weights=learned_module().")
    else:
        wvec = loop_weight_vector(
            scheme, T, alpha=alpha,
            device=hidden_states_list[0].device,
            dtype=torch.float32,
        )

    def chunk_logprobs(hs: torch.Tensor, tgt_chunk: torch.Tensor) -> torch.Tensor:
        return token_logprobs(lm_head(hs), tgt_chunk, temperature=temperature)

    wlp = hidden_states_list[0].new_zeros((B, L), dtype=torch.float32)
    for start, end in iter_seq_chunks(L, lm_chunk_size):
        acc = None
        tgt = targets[:, start:end]
        for t, hidden in enumerate(hidden_states_list):
            # [B, c]; the [B, c, V] log_softmax is recomputed in backward, not stored
            lp = recompute_in_backward(chunk_logprobs, hidden[:, start:end, :], tgt)
            w = w3[:, start:end, t] if w3 is not None else wvec[t]
            term = lp * w
            acc = term if acc is None else acc + term
        wlp[:, start:end] = acc
    return wlp


# ---------------------------------------------------------------------------
# KL penalty (aggregated-logprob forms, matching the reference)
# ---------------------------------------------------------------------------

KL_PENALTY_TYPES = ("kl", "abs", "mse", "low_var_kl")


def kl_penalty(
    policy_logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    kl_penalty_type: str = "low_var_kl",
) -> torch.Tensor:
    """Per-token KL penalty between the (aggregated) policy and reference
    log-probs. Mirrors the reference ``compute_rltt_kl_penalty`` forms."""
    if kl_penalty_type == "kl":
        return policy_logprob - ref_logprob
    if kl_penalty_type == "abs":
        return (policy_logprob - ref_logprob).abs()
    if kl_penalty_type == "mse":
        return 0.5 * (policy_logprob - ref_logprob).square()
    if kl_penalty_type == "low_var_kl":
        r = ref_logprob - policy_logprob
        kld = torch.expm1(r) - r  # exp(r) - 1 - r = ratio - kl - 1 (k3, >= 0)
        return torch.clamp(kld, min=-10.0, max=10.0)
    raise ValueError(f"Unknown kl_penalty_type '{kl_penalty_type}'. Use {KL_PENALTY_TYPES}.")


def compute_rltt_kl_penalty(
    per_loop_log_probs: list[torch.Tensor],
    per_loop_ref_log_probs: list[torch.Tensor],
    weights: torch.Tensor,
    kl_penalty_type: str = "low_var_kl",
) -> torch.Tensor:
    """Aggregate per-loop log-probs (policy and ref) with the same weights, then
    take the KL penalty. Name/signature mirror the reference implementation."""
    policy = aggregate_multi_loop_log_probs(per_loop_log_probs, weights)
    ref = aggregate_multi_loop_log_probs(per_loop_ref_log_probs, weights)
    return kl_penalty(policy, ref, kl_penalty_type)


# ---------------------------------------------------------------------------
# Policy loss
# ---------------------------------------------------------------------------

def token_entropy(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    """Per-token entropy H = -sum_v p_v log p_v under softmax(logits/temperature)."""
    if temperature != 1.0:
        logits = logits / float(temperature)
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def final_loop_token_logprobs(
    hidden_last: torch.Tensor,
    lm_head: torch.nn.Module,
    targets: torch.Tensor,
    *,
    lm_chunk_size: int = 0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """logP(target) under the FINAL loop only [B, L] -- used for the RLTT KL
    (reference Eq. 7: KL is on P^(T_max), not the weighted aggregation)."""
    B, L, _ = hidden_last.shape
    out = hidden_last.new_zeros((B, L), dtype=torch.float32)

    def chunk_logprobs(hs: torch.Tensor, tgt_chunk: torch.Tensor) -> torch.Tensor:
        return token_logprobs(lm_head(hs), tgt_chunk, temperature=temperature)

    for start, end in iter_seq_chunks(L, lm_chunk_size):
        out[:, start:end] = recompute_in_backward(
            chunk_logprobs, hidden_last[:, start:end, :], targets[:, start:end]
        )
    return out


def final_loop_token_entropy(
    hidden_last: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    lm_chunk_size: int = 0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token entropy of the FINAL loop's next-token distribution [B, L]
    (for the optional entropy bonus; reference uses the final-loop entropy)."""
    B, L, _ = hidden_last.shape
    out = hidden_last.new_zeros((B, L), dtype=torch.float32)

    def chunk_entropy(hs: torch.Tensor) -> torch.Tensor:
        return token_entropy(lm_head(hs), temperature=temperature)

    for start, end in iter_seq_chunks(L, lm_chunk_size):
        out[:, start:end] = recompute_in_backward(
            chunk_entropy, hidden_last[:, start:end, :]
        )
    return out


def grpo_rltt_loss(
    weighted_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    final_logprobs: Optional[torch.Tensor] = None,
    ref_final_logprobs: Optional[torch.Tensor] = None,
    kl_coef: float = 0.0,
    kl_penalty_type: str = "low_var_kl",
    entropy: Optional[torch.Tensor] = None,
    entropy_coef: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """RLTT/GRPO loss over completion tokens, matching the reference actor:

        L = -mean(A * sum_t w_t logP^(t))  - entropy_coef * mean(H)
            + kl_coef * mean(KL_final)

    - ``weighted_logprobs`` [B, L]: sum_t w_t logP^(t) under the policy (PG term).
    - ``advantages`` [B]: group-normalized advantage per sequence.
    - ``completion_mask`` [B, L]: 1 on tokens to train.
    - ``final_logprobs`` / ``ref_final_logprobs`` [B, L]: FINAL-loop logP under the
      policy and the frozen ref. The KL is on the final loop only (reference
      Eq. 7), NOT the weighted aggregation.
    - ``entropy`` [B, L] + ``entropy_coef``: optional final-loop entropy bonus
      (reference default entropy_coeff=0, i.e. off).
    """
    mask = completion_mask.float()
    denom = mask.sum().clamp(min=1.0)
    adv = advantages.detach().float().unsqueeze(1)  # [B, 1]

    pg_loss = (-(weighted_logprobs * adv) * mask).sum() / denom
    loss = pg_loss
    metrics = {
        "pg_loss": pg_loss.detach(),
        "adv_mean": advantages.detach().float().mean(),
        "adv_abs_mean": advantages.detach().float().abs().mean(),
        "tokens": denom.detach(),
    }

    if entropy is not None and entropy_coef != 0.0:
        ent = (entropy * mask).sum() / denom
        loss = loss - float(entropy_coef) * ent
        metrics["entropy"] = ent.detach()

    if kl_coef > 0.0 and final_logprobs is not None and ref_final_logprobs is not None:
        kl = kl_penalty(final_logprobs, ref_final_logprobs.detach(), kl_penalty_type)
        kl_loss = (kl * mask).sum() / denom
        loss = loss + float(kl_coef) * kl_loss
        metrics["kl"] = kl_loss.detach()
    return loss, metrics
