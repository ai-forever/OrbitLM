"""CPU tests for ``orbit_verl.rltt.loss``.

The RLTT loss is checked against three references:

- the frozen RLTT reference ``rltt_reference.grpo_rltt_loss`` for one prompt group, for every loop-weight scheme, with
  the KL on (loss value and gradients w.r.t. the per-loop log-probs and the gate logits);
- stock verl ``ppo_loss`` with ``loss_mode=vanilla``: at ratio 1, ``rltt_last`` must give the
  same gradients;
- verl's ``agg_loss`` applied by hand to every loss term, for ``token-mean`` and
  ``seq-mean-token-mean``, and the RLTT paper's per-response mean written out directly.

Micro-batches are built the way verl's V1 trainer hands them to the loss: jagged nested
``input_ids``, ``prompts``, ``responses``, ``response_mask``, ``advantages``, ``old_log_probs`` and
``ref_log_prob``, plus the non-tensor ``dp_size``, ``batch_num_tokens`` and ``global_batch_size``.
The model output is built by the engine helper ``build_rltt_model_output`` from leaf tensors, so
the gradients can be read on those leaves.

Run with ``-s`` to see the measured max-abs differences (lines starting with ``PARITY``).
"""
from __future__ import annotations

import functools

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, agg_loss, get_policy_loss_fn
from verl.trainer.ppo.core_algos import kl_penalty as verl_kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss

import rltt_reference as ALGO
from orbit_verl.rltt.engine import build_rltt_model_output
from orbit_verl.rltt.forward import OrbitLMRLTTOutput
from orbit_verl.rltt.loss import (
    RLTT_LOSS_MODES,
    STUB_MESSAGE,
    progressive_alpha,
    register_stub_policy_losses,
    rltt_loss,
)

# The reference module holds the exit pdf as well as the loss pieces.
LOSSES = ALGO

LOOPS = 4
KL_COEF = 0.1
REFERENCE_SCHEME = {"exit": "exit", "uniform": "uniform", "progressive": "progressive", "last": "grpo"}


def _nested(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _actor_config(
    loss_mode: str, *, use_kl_loss: bool = True, entropy_coeff: float = 0.0, loss_agg_mode: str = "token-mean"
) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=4,
        ppo_micro_batch_size_per_gpu=1,
        use_kl_loss=use_kl_loss,
        kl_loss_coef=KL_COEF,
        kl_loss_type="low_var_kl",
        loss_agg_mode=loss_agg_mode,
        entropy_coeff=entropy_coeff,
        policy_loss=PolicyLossConfig(loss_mode=loss_mode),
    )


class Batch:
    """One micro-batch in verl's no-padding layout, plus leaf tensors for the model output.

    Position ``k`` of the packed sequence holds the log-prob of token ``k + 1``. The response
    log-probs of sequence ``i`` therefore sit at ``start_i + P_i - 1 .. start_i + P_i + R_i - 2``,
    which is the slice ``no_padding_2_padding`` takes. ``index`` and ``mask`` describe that slice
    as a padded ``[B, R_max]`` gather, so the reference computations never call verl's slicing.
    """

    def __init__(self, prompt_lens, response_lens, advantages, *, seed=0, holes=(), dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        self.batch_size = len(prompt_lens)
        prompts = [torch.randint(0, 50, (p,), generator=g) for p in prompt_lens]
        if len(set(prompt_lens)) == 1:
            # A single prompt group: every completion continues the same prompt.
            prompts = [prompts[0].clone() for _ in prompt_lens]
        responses = [torch.randint(0, 50, (r,), generator=g) for r in response_lens]
        masks = [torch.ones(r, dtype=torch.int64) for r in response_lens]
        for row, col in holes:
            # A masked token inside a response, as the agent loop produces when it trims at a stop.
            masks[row][col] = 0

        self.nnz = sum(prompt_lens) + sum(response_lens)
        starts, offset = [], 0
        for p, r in zip(prompt_lens, response_lens):
            starts.append(offset)
            offset += p + r
        width = max(response_lens)
        self.index = torch.zeros(self.batch_size, width, dtype=torch.long)
        self.mask = torch.zeros(self.batch_size, width, dtype=dtype)
        for i, (start, p, r) in enumerate(zip(starts, prompt_lens, response_lens)):
            self.index[i, :r] = torch.arange(start + p - 1, start + p - 1 + r)
            self.mask[i, :r] = masks[i].to(dtype)

        # Per-loop log-probs and gate logits over the whole packed sequence, as model outputs.
        self.loop_values = -(torch.rand(self.nnz, LOOPS, generator=g, dtype=dtype) * 3.0 + 0.05)
        self.gate_values = torch.randn(self.nnz, LOOPS, generator=g, dtype=dtype) * 1.5
        self.entropy_values = torch.rand(self.nnz, generator=g, dtype=dtype) * 2.0

        final = self.loop_values[:, -1]
        self.ref_padded = (final[self.index] + 0.3 * torch.randn(self.batch_size, width, generator=g, dtype=dtype)) * (
            self.mask > 0
        )
        self.old_padded = final[self.index] * (self.mask > 0)
        self.seq_advantages = torch.as_tensor(advantages, dtype=dtype)

        input_ids = [torch.cat([p, r]) for p, r in zip(prompts, responses)]
        rows = lambda padded: _nested([padded[i, :r] for i, r in enumerate(response_lens)])  # noqa: E731
        self.td = TensorDict(
            {
                "input_ids": _nested(input_ids),
                "prompts": _nested(prompts),
                "responses": _nested(responses),
                "response_mask": _nested(masks),
                "advantages": _nested([torch.full((r,), float(a), dtype=dtype) for r, a in zip(response_lens, advantages)]),
                "old_log_probs": rows(self.old_padded),
                "ref_log_prob": rows(self.ref_padded),
            },
            batch_size=[self.batch_size],
        )
        tu.assign_non_tensor(
            self.td, dp_size=1, batch_num_tokens=int(self.mask.sum().item()), global_batch_size=self.batch_size
        )
        self.offsets = self.td["input_ids"].offsets()

    def leaves(self):
        """Fresh leaf copies of the per-loop log-probs and gate logits, both requiring grad."""
        return self.loop_values.clone().requires_grad_(True), self.gate_values.clone().requires_grad_(True)

    def model_output(self, loop_leaf, gate_leaf, scheme, *, alpha=1.0, entropy=None, train_exit_gate=False):
        """What ``OrbitLMRLTTEngine`` would hand to the loss, computed from the leaves."""
        gates = [gate_leaf[:, t].reshape(1, -1, 1) for t in range(LOOPS)]
        fake = OrbitLMRLTTOutput(
            # The real forward returns contiguous tensors; a column view would not be.
            log_probs=loop_leaf[:, -1].contiguous().unsqueeze(0),
            loop_log_probs=loop_leaf.unsqueeze(0),
            exit_pdf=LOSSES.build_exit_pdf(gates),
            entropy=None if entropy is None else entropy.unsqueeze(0),
        )
        return build_rltt_model_output(
            fake, unpad_fn=lambda t: t, offsets=self.offsets, scheme=scheme, alpha=alpha,
            train_exit_gate=train_exit_gate,
        )


def _reference_loss(batch: Batch, loop_leaf, gate_leaf, scheme, *, alpha=1.0, entropy_leaf=None, entropy_coef=0.0):
    """The reference loss on the same leaves, through reference functions only."""
    per_loop = [loop_leaf[:, t][batch.index] for t in range(LOOPS)]  # T x [B, R]
    if scheme == "exit":
        gates = [gate_leaf[:, t][batch.index].unsqueeze(-1) for t in range(LOOPS)]
        weights = LOSSES.build_exit_pdf(gates)
    else:
        weights = ALGO.loop_weight_vector(REFERENCE_SCHEME[scheme], LOOPS, alpha=alpha)
    weighted = ALGO.aggregate_multi_loop_log_probs(per_loop, weights)
    entropy = None if entropy_leaf is None else entropy_leaf[batch.index]
    return ALGO.grpo_rltt_loss(
        weighted,
        batch.seq_advantages,
        batch.mask,
        final_logprobs=per_loop[-1],
        ref_final_logprobs=batch.ref_padded,
        kl_coef=KL_COEF,
        kl_penalty_type="low_var_kl",
        entropy=entropy,
        entropy_coef=entropy_coef,
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max().item())


def _single_group_batch(seed=0) -> Batch:
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])
    advantages = ALGO.group_advantages(rewards, group_size=5).tolist()
    return Batch([6] * 5, [4, 2, 7, 3, 5], advantages, seed=seed, holes=[(2, 6), (0, 1)])


@pytest.mark.parametrize("scheme", ["exit", "uniform", "progressive", "last"])
def test_single_group_matches_reference_grpo_rltt_loss(scheme):
    batch = _single_group_batch()
    alpha = 1.5 if scheme == "progressive" else 1.0

    loop_leaf, gate_leaf = batch.leaves()
    # The reference loss trains the gate, so the comparison does too.
    model_output = batch.model_output(loop_leaf, gate_leaf, scheme, alpha=alpha, train_exit_gate=True)
    loss, _ = rltt_loss(_actor_config(f"rltt_{scheme}"), model_output, batch.td, scheme=scheme)
    loss.backward()

    ref_loop, ref_gate = batch.leaves()
    ref_loss, _ = _reference_loss(batch, ref_loop, ref_gate, scheme, alpha=alpha)
    ref_loss.backward()

    loss_diff = abs(loss.item() - ref_loss.item())
    loop_grad_diff = _max_abs(loop_leaf.grad, ref_loop.grad)
    print(f"PARITY reference scheme={scheme} loss={loss.item():.6f} |dloss|={loss_diff:.3e} "
          f"|dgrad_loop|={loop_grad_diff:.3e}", end="")
    assert loss_diff < 1e-6
    assert loop_grad_diff < 1e-9
    assert loop_leaf.grad.abs().sum() > 0
    if scheme == "exit":
        gate_grad_diff = _max_abs(gate_leaf.grad, ref_gate.grad)
        print(f" |dgrad_gate|={gate_grad_diff:.3e}")
        assert gate_grad_diff < 1e-7
        assert gate_leaf.grad.abs().sum() > 0
    else:
        print()
        # Static weights do not depend on the gate, so the gate must get no gradient at all.
        assert gate_leaf.grad is None or gate_leaf.grad.abs().sum() == 0


def test_gate_gets_a_gradient_only_with_train_exit_gate():
    """train_exit_gate=False is the paper's Eq. 4: the exit weights are constants of the loss.

    Detaching the weights changes nothing but the gate's gradient. The loss value and the
    gradients of the per-loop log-probs stay the same as when the gate is trained.
    """
    batch = Batch([3, 5, 2, 4], [4, 2, 6, 3], [1.0, -0.5, 0.75, -1.25], seed=6, holes=[(2, 5)])
    results = {}
    for train_exit_gate in (False, True):
        loop_leaf, gate_leaf = batch.leaves()
        output = batch.model_output(loop_leaf, gate_leaf, "exit", train_exit_gate=train_exit_gate)
        loss, metrics = rltt_loss(_actor_config("rltt_exit"), output, batch.td, scheme="exit")
        loss.backward()
        results[train_exit_gate] = (loss.item(), loop_leaf.grad, gate_leaf.grad, metrics)

    frozen_loss, frozen_loop_grad, frozen_gate_grad, frozen_metrics = results[False]
    trained_loss, trained_loop_grad, trained_gate_grad, trained_metrics = results[True]
    assert frozen_gate_grad is None
    assert trained_gate_grad is not None and trained_gate_grad.abs().sum() > 0
    assert frozen_loss == trained_loss
    assert torch.equal(frozen_loop_grad, trained_loop_grad)
    # The gate's metrics are logged either way.
    assert frozen_metrics["rltt/expected_exit_step"].aggregate() == trained_metrics["rltt/expected_exit_step"].aggregate()


def test_entropy_bonus_matches_reference():
    batch = _single_group_batch(seed=3)
    coef = 0.01
    loop_leaf, gate_leaf = batch.leaves()
    entropy_leaf = batch.entropy_values.clone().requires_grad_(True)
    model_output = batch.model_output(loop_leaf, gate_leaf, "exit", entropy=entropy_leaf)
    loss, metrics = rltt_loss(_actor_config("rltt_exit", entropy_coeff=coef), model_output, batch.td, scheme="exit")
    loss.backward()

    ref_loop, ref_gate = batch.leaves()
    ref_entropy = batch.entropy_values.clone().requires_grad_(True)
    ref_loss, _ = _reference_loss(batch, ref_loop, ref_gate, "exit", entropy_leaf=ref_entropy, entropy_coef=coef)
    ref_loss.backward()

    assert abs(loss.item() - ref_loss.item()) < 1e-6
    assert _max_abs(entropy_leaf.grad, ref_entropy.grad) < 1e-9
    assert "actor/entropy_loss" in metrics


def test_rltt_last_gradients_equal_stock_vanilla_ppo_loss():
    """At ratio 1 the clipped PPO objective has the REINFORCE gradient, so the two must agree."""
    batch = Batch([3, 5, 2, 4], [4, 2, 6, 3], [1.0, -0.5, 0.75, -1.25], seed=1, holes=[(2, 5)])

    stock_leaf, _ = batch.leaves()
    stock_output = {"log_probs": torch.nested.nested_tensor_from_jagged(stock_leaf[:, -1].contiguous(), batch.offsets)}
    stock_config = _actor_config("vanilla")
    stock_loss, stock_metrics = ppo_loss(stock_config, stock_output, batch.td)
    stock_loss.backward()

    rltt_leaf, gate_leaf = batch.leaves()
    rltt_config = _actor_config("rltt_last")
    loss, metrics = rltt_loss(rltt_config, batch.model_output(rltt_leaf, gate_leaf, "last"), batch.td, scheme="last")
    loss.backward()

    grad_diff = _max_abs(rltt_leaf.grad, stock_leaf.grad)
    kl_diff = abs(metrics["kl_loss"].aggregate() - stock_metrics["kl_loss"].aggregate())
    print(f"PARITY vanilla rltt_last |dgrad|={grad_diff:.3e} |dkl_loss|={kl_diff:.3e}")
    assert grad_diff < 1e-12
    assert kl_diff < 1e-12
    # Only the final loop is trained by rltt_last.
    assert rltt_leaf.grad[:, :-1].abs().sum() == 0
    assert rltt_leaf.grad[:, -1].abs().sum() > 0
    # The aggregation inputs are filled exactly as stock ppo_loss fills them.
    assert rltt_config.global_batch_info == stock_config.global_batch_info
    assert metrics["actor/pg_loss"].aggregation == stock_metrics["actor/pg_loss"].aggregation
    assert metrics["kl_loss"].aggregation == stock_metrics["kl_loss"].aggregation
    assert metrics["kl_coef"] == stock_metrics["kl_coef"]
    assert abs(metrics["actor/ppo_kl"].aggregate() - stock_metrics["actor/ppo_kl"].aggregate()) < 1e-6


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_verl_low_var_kl_equals_reference_k3(dtype):
    """verl clamps the log ratio to +-20 before exp; the reference k3 does not. Both clamp the result to +-10."""
    g = torch.Generator().manual_seed(7)
    log_ratio = torch.linspace(-20.0, 20.0, 4001, dtype=torch.float64)
    logprob = (-(torch.rand(log_ratio.shape, generator=g, dtype=torch.float64) * 5.0)).to(dtype)
    ref = (logprob.double() + log_ratio).to(dtype)

    policy_verl = logprob.clone().requires_grad_(True)
    policy_reference = logprob.clone().requires_grad_(True)
    verl_kl = verl_kl_penalty(policy_verl, ref, "low_var_kl")
    reference_kl = ALGO.kl_penalty(policy_reference, ref, "low_var_kl")
    verl_kl.sum().backward()
    reference_kl.sum().backward()

    value_diff = _max_abs(verl_kl, reference_kl)
    grad_diff = _max_abs(policy_verl.grad, policy_reference.grad)
    print(f"PARITY k3 dtype={dtype} |dkl|={value_diff:.3e} |dgrad|={grad_diff:.3e}")
    # verl computes exp(r) - r - 1 and the reference code computes expm1(r) - r. In float32 the two
    # round differently, by about one unit in the last place of values up to the clamp at 10.
    if dtype == torch.float64:
        assert value_diff < 1e-12
        assert grad_diff < 1e-12
    else:
        torch.testing.assert_close(verl_kl, reference_kl, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(policy_verl.grad, policy_reference.grad, rtol=1e-6, atol=1e-6)


def test_metrics_present_finite_and_well_formed():
    batch = Batch([3, 5, 2, 4], [4, 2, 6, 3], [1.0, -0.5, 0.75, -1.25], seed=2)
    loop_leaf, gate_leaf = batch.leaves()
    _, metrics = rltt_loss(_actor_config("rltt_exit"), batch.model_output(loop_leaf, gate_leaf, "exit"), batch.td,
                           scheme="exit")

    expected = {"actor/pg_loss", "kl_loss", "kl_coef", "rltt/expected_exit_step",
                "rltt/final_vs_old_logp_absdiff", "actor/ppo_kl"}
    expected |= {f"rltt/exit_pdf_t{t}" for t in range(LOOPS)}
    expected |= {f"rltt/loop_logp_t{t}" for t in range(LOOPS)}
    assert expected <= set(metrics)
    values = {k: (v.aggregate() if isinstance(v, Metric) else float(v)) for k, v in metrics.items()}
    assert all(torch.isfinite(torch.tensor(v)) for v in values.values()), values
    # ClearML needs a "/" in every new key; the stock keys without one are kept as verl names them.
    assert all("/" in key or key in {"kl_loss", "kl_coef"} for key in metrics)
    for key in expected - {"actor/pg_loss", "kl_loss", "kl_coef"}:
        assert metrics[key].aggregation == AggregationType.MEAN, key
    # batch_num_tokens is set, so the loss metrics are partial sums, as in stock ppo_loss.
    assert metrics["actor/pg_loss"].aggregation == AggregationType.SUM
    assert abs(sum(values[f"rltt/exit_pdf_t{t}"] for t in range(LOOPS)) - 1.0) < 1e-5
    assert 0.0 <= values["rltt/expected_exit_step"] <= LOOPS - 1
    # old_log_probs are the final-loop values of these same leaves.
    assert values["rltt/final_vs_old_logp_absdiff"] < 1e-9


def _loss_with_batch_info(loss_agg_mode: str, **batch_info) -> float:
    batch = Batch([3, 5, 2, 4], [4, 2, 6, 3], [1.0, -0.5, 0.75, -1.25], seed=4, holes=[(1, 1)])
    tu.assign_non_tensor(batch.td, **batch_info)
    config = _actor_config("rltt_exit", loss_agg_mode=loss_agg_mode)
    loop_leaf, gate_leaf = batch.leaves()
    loss, _ = rltt_loss(config, batch.model_output(loop_leaf, gate_leaf, "exit"), batch.td, scheme="exit")
    return loss.item()


@pytest.mark.parametrize("loss_agg_mode", ["token-mean", "seq-mean-token-mean"])
def test_every_term_is_aggregated_with_verls_agg_loss(loss_agg_mode):
    """Policy gradient, entropy and KL each go through agg_loss with the configured mode, as in ppo_loss.

    The expected loss is built from padded tensors with verl's own agg_loss and kl_penalty, so it
    depends on rltt_loss only through the model output. Two groups with different response lengths
    and masked tokens make the two modes give different numbers.
    """
    batch = Batch([3, 3, 3, 5, 5, 5], [4, 2, 7, 3, 6, 1], [1.0, -0.5, -0.5, 0.75, -1.25, 0.5],
                  seed=7, holes=[(2, 6), (4, 0)])
    entropy_coef = 0.02
    loop_leaf, gate_leaf = batch.leaves()
    entropy_leaf = batch.entropy_values.clone().requires_grad_(True)
    config = _actor_config("rltt_exit", entropy_coeff=entropy_coef, loss_agg_mode=loss_agg_mode)
    output = batch.model_output(loop_leaf, gate_leaf, "exit", entropy=entropy_leaf)
    loss, metrics = rltt_loss(config, output, batch.td, scheme="exit")
    loss.backward()

    ref_loop, ref_gate = batch.leaves()
    ref_entropy = batch.entropy_values.clone().requires_grad_(True)
    gates = [ref_gate[:, t].reshape(1, -1, 1) for t in range(LOOPS)]
    exit_pdf = LOSSES.build_exit_pdf(gates).squeeze(0).detach()
    weighted = (ref_loop * exit_pdf).sum(-1)[batch.index]
    final = ref_loop[:, -1][batch.index]
    mask = batch.mask.to(bool)
    advantages = batch.seq_advantages.unsqueeze(-1).expand_as(weighted)
    info = dict(dp_size=1, batch_num_tokens=int(batch.mask.sum().item()), global_batch_size=batch.batch_size,
                loss_scale_factor=None)
    pg = agg_loss(-advantages * weighted, mask, loss_agg_mode, **info)
    entropy = agg_loss(ref_entropy[batch.index], mask, loss_agg_mode, **info)
    kl = agg_loss(verl_kl_penalty(final, batch.ref_padded, "low_var_kl"), mask, loss_agg_mode, **info)
    expected = pg - entropy_coef * entropy + KL_COEF * kl
    expected.backward()

    print(f"PARITY agg_loss mode={loss_agg_mode} loss={loss.item():.6f} |dloss|={abs(loss.item() - expected.item()):.3e}")
    assert abs(loss.item() - expected.item()) < 1e-12
    assert _max_abs(loop_leaf.grad, ref_loop.grad) < 1e-12
    assert _max_abs(entropy_leaf.grad, ref_entropy.grad) < 1e-12
    assert abs(metrics["actor/pg_loss"].aggregate() - pg.item()) < 1e-12
    assert abs(metrics["kl_loss"].aggregate() - kl.item()) < 1e-12
    assert abs(metrics["actor/entropy_loss"].aggregate() - entropy.item()) < 1e-12

    other = "token-mean" if loss_agg_mode == "seq-mean-token-mean" else "seq-mean-token-mean"
    assert abs(agg_loss(-advantages * weighted, mask, other, **info).item() - pg.item()) > 1e-6


def test_seq_mean_token_mean_is_the_papers_per_response_mean():
    """RLTT paper, Eq. 6: the mean over groups of the mean over each group's responses of the token mean."""
    groups = [[0, 1, 2], [3, 4, 5]]
    batch = Batch([3, 3, 3, 5, 5, 5], [4, 2, 7, 3, 6, 1], [1.0, -0.5, -0.5, 0.75, -1.25, 0.5],
                  seed=8, holes=[(2, 6)])
    loop_leaf, gate_leaf = batch.leaves()
    config = _actor_config("rltt_exit", use_kl_loss=False, loss_agg_mode="seq-mean-token-mean")
    loss, _ = rltt_loss(config, batch.model_output(loop_leaf, gate_leaf, "exit"), batch.td, scheme="exit")

    gates = [gate_leaf[:, t].reshape(1, -1, 1) for t in range(LOOPS)]
    exit_pdf = LOSSES.build_exit_pdf(gates).squeeze(0).detach()
    weighted = (loop_leaf * exit_pdf).sum(-1)[batch.index].detach()
    per_response = [
        float((-batch.seq_advantages[i] * weighted[i] * batch.mask[i]).sum() / batch.mask[i].sum())
        for i in range(batch.batch_size)
    ]
    by_group = [sum(per_response[i] for i in group) / len(group) for group in groups]
    expected = sum(by_group) / len(by_group)
    assert abs(loss.item() - expected) < 1e-12


def test_global_batch_info_reaches_the_aggregation():
    """dp_size, batch_num_tokens and global_batch_size from the batch must all change the loss scale.

    On a real run every rank divides by the global token count and multiplies by dp_size, and FSDP
    averages the gradients over ranks. Doubling dp_size together with the token count must
    therefore leave the loss unchanged, and doubling only the token count must halve it.
    """
    tokens = 4 + 1 + 6 + 3  # response tokens of the batch above, minus the one masked token
    base = _loss_with_batch_info("token-mean", dp_size=1, batch_num_tokens=tokens, global_batch_size=4)
    same = _loss_with_batch_info("token-mean", dp_size=2, batch_num_tokens=2 * tokens, global_batch_size=4)
    half = _loss_with_batch_info("token-mean", dp_size=1, batch_num_tokens=2 * tokens, global_batch_size=4)
    assert abs(same - base) < 1e-12
    assert abs(half - base / 2) < 1e-12

    seq_base = _loss_with_batch_info("seq-mean-token-mean", dp_size=1, batch_num_tokens=tokens, global_batch_size=4)
    seq_half = _loss_with_batch_info("seq-mean-token-mean", dp_size=1, batch_num_tokens=tokens, global_batch_size=8)
    assert abs(seq_half - seq_base / 2) < 1e-12


def test_rltt_loss_refuses_stock_outputs_and_missing_ref():
    batch = Batch([3, 5], [4, 2], [1.0, -1.0], seed=5)
    loop_leaf, gate_leaf = batch.leaves()
    stock_output = {"log_probs": torch.nested.nested_tensor_from_jagged(loop_leaf[:, -1].contiguous(), batch.offsets)}
    with pytest.raises(RuntimeError, match="rltt_log_probs"):
        rltt_loss(_actor_config("rltt_exit"), stock_output, batch.td, scheme="exit")
    with pytest.raises(ValueError, match="scheme"):
        rltt_loss(_actor_config("rltt_exit"), batch.model_output(loop_leaf, gate_leaf, "exit"), batch.td,
                  scheme="learned")
    no_ref = batch.td.exclude("ref_log_prob")
    with pytest.raises(KeyError, match="ref_log_prob"):
        rltt_loss(_actor_config("rltt_exit"), batch.model_output(loop_leaf, gate_leaf, "exit"), no_ref, scheme="exit")
    # Without the KL the reference is not needed, and without old_log_probs the diagnostic is skipped.
    no_old = no_ref.exclude("old_log_probs")
    _, metrics = rltt_loss(_actor_config("rltt_exit", use_kl_loss=False),
                           batch.model_output(loop_leaf, gate_leaf, "exit"), no_old, scheme="exit")
    assert "kl_loss" not in metrics and "rltt/final_vs_old_logp_absdiff" not in metrics


def test_stub_registration_is_idempotent_and_stubs_raise():
    register_stub_policy_losses()
    first = {name: POLICY_LOSS_REGISTRY[name] for name in RLTT_LOSS_MODES}
    register_stub_policy_losses()
    for name in RLTT_LOSS_MODES:
        assert POLICY_LOSS_REGISTRY[name] is first[name]
        with pytest.raises(RuntimeError) as info:
            get_policy_loss_fn(name)(None, None, None, None)
        assert str(info.value) == STUB_MESSAGE.format(name=name)
        assert "VERL_ENGINE_VENDOR=orbit" in str(info.value)


def test_stub_registration_never_replaces_an_existing_entry():
    sentinel = functools.partial(lambda *a, **k: None)
    saved = POLICY_LOSS_REGISTRY.get("rltt_uniform")
    POLICY_LOSS_REGISTRY["rltt_uniform"] = sentinel
    try:
        register_stub_policy_losses()
        assert POLICY_LOSS_REGISTRY["rltt_uniform"] is sentinel
    finally:
        if saved is None:
            POLICY_LOSS_REGISTRY.pop("rltt_uniform", None)
        else:
            POLICY_LOSS_REGISTRY["rltt_uniform"] = saved


def test_progressive_alpha_from_env(monkeypatch):
    monkeypatch.delenv("ORBIT_RLTT_PROGRESSIVE_ALPHA", raising=False)
    assert progressive_alpha() == 1.0
    monkeypatch.setenv("ORBIT_RLTT_PROGRESSIVE_ALPHA", "2.5")
    assert progressive_alpha() == 2.5
    monkeypatch.setenv("ORBIT_RLTT_PROGRESSIVE_ALPHA", "two")
    with pytest.raises(ValueError, match="ORBIT_RLTT_PROGRESSIVE_ALPHA"):
        progressive_alpha()
