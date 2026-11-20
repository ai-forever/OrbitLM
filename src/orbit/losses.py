"""The OrbitLM training loss: the exit-weighted LM loss over all loops plus the entropy term.

The exit gate gives every token a probability of exiting after each loop (build_exit_pdf). The LM
loss is the cross entropy of each loop's prediction, weighted by that probability, and the exit
loss is minus beta times the entropy of the exit distribution, which keeps the gate from
collapsing onto one loop early in training.

src/orbit_verl/rltt/forward.py keeps an identical copy of build_exit_pdf, so a change to it here
must be made there too.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def recompute_in_backward(fn, *tensors):
    """Run ``fn`` without keeping its activations; recompute them in backward.

    Chunking the LM head does NOT by itself bound memory. cross_entropy is
    ``nll_loss(log_softmax(x))`` and log_softmax's backward needs its own output,
    so autograd pins a ``[B, chunk, vocab]`` tensor per chunk until backward runs:
    rebinding ``logits`` each iteration frees nothing, and the total held is the
    same as computing the whole sequence at once. Over 4 loops and a vocabulary of
    150K tokens that is tens of GB, which is memory the optimizer step and NCCL's
    collective buffers then cannot get.

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


def summarize_exit_distribution(
    stacked_exit_pdf: Optional[torch.Tensor], valid_mask: torch.Tensor
) -> Optional[torch.Tensor]:
    if stacked_exit_pdf is None:
        return None
    pdf = stacked_exit_pdf.detach().float()
    denom = valid_mask.sum().clamp(min=1).to(pdf.dtype)
    return (pdf * valid_mask.unsqueeze(-1)).sum(dim=(0, 1)) / denom


def summarize_threshold_exit_counts(
    stacked_exit_pdf: Optional[torch.Tensor],
    valid_mask: torch.Tensor,
    *,
    threshold: float,
) -> Optional[torch.Tensor]:
    if stacked_exit_pdf is None:
        return None
    pdf = stacked_exit_pdf.detach().float()
    cumulative_probs = torch.cumsum(pdf, dim=2)
    threshold_mask = cumulative_probs >= float(threshold)
    exit_steps = torch.argmax(threshold_mask.float(), dim=2)
    last_step_idx = int(pdf.shape[2]) - 1
    if last_step_idx >= 0:
        never_exceeded = ~threshold_mask.any(dim=2)
        exit_steps[never_exceeded] = last_step_idx
    counts = []
    for step_idx in range(int(pdf.shape[2])):
        counts.append(((exit_steps == step_idx) & valid_mask).sum(dtype=torch.float32))
    return torch.stack(counts, dim=0)


def compute_weighted_lm_and_exit_loss(
    hidden_states_list: list[torch.Tensor],
    stacked_exit_pdf: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_fn,
    lm_head: nn.Module,
    beta: float,
    need_per_ut_lm_losses: bool = False,
    *,
    lm_chunk_size: int,
):
    valid_f = valid_mask.to(dtype=torch.float32)
    token_count = valid_f.sum().clamp(min=1.0)
    seq_len = int(labels.shape[1])
    expected_loss_sum = labels.new_zeros((), dtype=torch.float32)
    per_ut_lm_losses = [] if need_per_ut_lm_losses else None

    def chunk_ce(hs: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
        logits = lm_head(hs)
        return loss_fn(logits.reshape(-1, logits.size(-1)), lb.reshape(-1)).view_as(lb).float()

    for step_idx, hidden in enumerate(hidden_states_list):
        ut_unweighted_sum = labels.new_zeros((), dtype=torch.float32) if per_ut_lm_losses is not None else None
        for start, end in iter_seq_chunks(seq_len, lm_chunk_size):
            vm = valid_f[:, start:end]
            ce = recompute_in_backward(chunk_ce, hidden[:, start:end, :], labels[:, start:end])
            weight = stacked_exit_pdf[:, start:end, step_idx].to(ce.dtype)
            expected_loss_sum = expected_loss_sum + (ce * weight * vm).sum()
            if ut_unweighted_sum is not None:
                ut_unweighted_sum = ut_unweighted_sum + (ce.detach() * vm).sum()
        if per_ut_lm_losses is not None and ut_unweighted_sum is not None:
            per_ut_lm_losses.append((ut_unweighted_sum / token_count).detach())

    lm_loss = expected_loss_sum / token_count
    pdf = stacked_exit_pdf.float().clamp_min(1e-9)
    entropy = -(pdf * pdf.log()).sum(dim=2)
    exit_entropy_mean = (entropy * valid_f).sum() / token_count
    exit_loss = -float(beta) * exit_entropy_mean
    return lm_loss, exit_loss, exit_entropy_mean, per_ut_lm_losses


def summarize_gate_probabilities(
    gate_list: list[torch.Tensor], valid_mask: torch.Tensor
) -> Optional[list[torch.Tensor]]:
    if not gate_list:
        return None
    valid_f = valid_mask.to(dtype=torch.float32)
    denom = valid_f.sum().clamp(min=1.0)
    lambda_means = []
    for gate_tensor in gate_list:
        lambdas = torch.sigmoid(gate_tensor.detach().squeeze(-1).float())
        lambda_means.append((lambdas * valid_f).sum() / denom)
    return lambda_means
