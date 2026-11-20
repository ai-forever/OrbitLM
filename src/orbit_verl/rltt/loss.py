"""The RLTT policy loss for verl, the RLTT loss-mode names, and the raising stubs.

The loss is the RLTT GRPO loss written on verl's tensors and aggregation:

    loss = agg(-A * rltt_log_probs)                                   policy gradient, no ratio
         + kl_loss_coef * agg(kl_penalty(final_log_probs, ref_log_prob))   if use_kl_loss
         - entropy_coeff * agg(final_entropy)                          if the entropy was computed

``rltt_log_probs`` is ``sum_t w_t * log P^(t)``, built by ``OrbitLMRLTTEngine``. With exit weights
``w`` is the exit pdf, detached unless ``orbit.rltt.train_exit_gate`` is set
(``orbit_verl.rltt.forward.loop_weights``).

``agg`` is verl's ``agg_loss`` with the actor's ``loss_agg_mode`` and the global batch info, for all
three terms, exactly as stock ``ppo_loss`` aggregates its policy, entropy and KL terms. The loss is
therefore normalized like the stock one across micro-batches and data-parallel ranks, whatever the
mode. ``orbit_base.yaml`` sets ``seq-mean-token-mean``: each response's token mean, then the mean
over all responses of the mini-batch. With equal group sizes that is Eq. 6 of the RLTT paper, the
mean over groups of the mean over each group's responses.

The loss is ratio-free on purpose. verl's ``old_log_probs`` are final-loop values, while the policy
gradient here uses a mixture over loops. A PPO ratio between the two would sit far from 1 even on
the first update, and clipping would then cut most of the gradient. With one optimizer step per
batch the data is on-policy, so REINFORCE is the correct estimator. This is also what the RLTT
reference implementation does.

The stubs are for runs where the plugin is loaded but our engine is not selected. Stock ``ppo_loss``
then looks the loss mode up in verl's ``POLICY_LOSS_REGISTRY``. Without an entry it fails with
"Unsupported loss mode", which does not say what is missing, so each RLTT name gets an entry that
raises with the fix in its message. The entries are deliberately not working losses: stock
``ppo_loss`` only ever sees final-loop log-probs, so a working loss under an RLTT name would
silently train plain final-loop GRPO.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, agg_loss, kl_penalty, register_policy_loss
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean
from verl.workers.utils.padding import no_padding_2_padding

from orbit_verl.rltt.forward import LOOP_WEIGHT_SCHEMES

# loss_mode (actor_rollout_ref.actor.policy_loss.loss_mode) -> loop-weight scheme.
RLTT_LOSS_MODES = {
    "rltt_exit": "exit",
    "rltt_uniform": "uniform",
    "rltt_progressive": "progressive",
    "rltt_last": "last",
}

PROGRESSIVE_ALPHA_ENV = "ORBIT_RLTT_PROGRESSIVE_ALPHA"
DEFAULT_PROGRESSIVE_ALPHA = 1.0

# The launcher writes orbit.rltt.train_exit_gate into this variable as "true" or "false".
TRAIN_EXIT_GATE_ENV = "ORBIT_RLTT_TRAIN_EXIT_GATE"
_TRUE_WORDS = ("1", "true", "yes", "on")
_FALSE_WORDS = ("0", "false", "no", "off")

STUB_MESSAGE = (
    "loss_mode {name} needs OrbitLMRLTTEngine: set VERL_ENGINE_VENDOR=orbit and import orbit_verl.plugin"
)


def progressive_alpha() -> float:
    """Exponent of the progressive loop weights ``(t+1)^alpha``, from ``ORBIT_RLTT_PROGRESSIVE_ALPHA``."""
    raw = os.environ.get(PROGRESSIVE_ALPHA_ENV, "").strip()
    if not raw:
        return DEFAULT_PROGRESSIVE_ALPHA
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{PROGRESSIVE_ALPHA_ENV}={raw!r} is not a number.") from exc


def trains_exit_gate() -> bool:
    """Whether the RLTT loss trains the exit gate, from ``ORBIT_RLTT_TRAIN_EXIT_GATE``.

    An unset variable means false, the paper's constant loop weights. Any value that is not a clear
    yes or no is refused, because guessing would silently pick one of two different objectives.
    """
    raw = os.environ.get(TRAIN_EXIT_GATE_ENV, "").strip().lower()
    if not raw or raw in _FALSE_WORDS:
        return False
    if raw in _TRUE_WORDS:
        return True
    raise ValueError(f"{TRAIN_EXIT_GATE_ENV}={raw!r} is not true or false.")


def _stub_policy_loss(name: str):
    def stub(*args: Any, **kwargs: Any):
        raise RuntimeError(STUB_MESSAGE.format(name=name))

    stub.__name__ = f"{name}_requires_orbit_engine"
    stub.__qualname__ = stub.__name__
    return stub


def register_stub_policy_losses() -> None:
    """Register a raising stub for every RLTT loss mode that verl's registry does not know yet.

    Names that are already registered are left alone. That makes the call idempotent, and it never
    replaces a real loss that someone registered under one of these names.
    """
    for name in RLTT_LOSS_MODES:
        if name not in POLICY_LOSS_REGISTRY:
            register_policy_loss(name)(_stub_policy_loss(name))


def _masked_means(columns: list[torch.Tensor], mask: torch.Tensor) -> list[float]:
    # Every metric is a masked mean over response tokens. They are stacked and moved to the host
    # in one transfer, because each separate .item() would be its own GPU synchronization.
    if not columns:
        return []
    values = torch.stack([masked_mean(column.detach().float(), mask) for column in columns])
    return values.tolist()


def rltt_loss(
    config,
    model_output: dict,
    data: TensorDict,
    dp_group=None,
    *,
    scheme: str,
) -> tuple[torch.Tensor, dict]:
    """RLTT loss for one micro-batch. Called by ``OrbitLMRLTTEngine`` in place of ``ppo_loss``.

    ``model_output`` holds jagged nested tensors over the packed sequence, produced by the engine:
    ``rltt_log_probs`` and ``log_probs`` (final loop) with gradients, ``loop_log_probs`` and
    ``exit_pdf`` detached for metrics, and ``entropy`` when verl asked for it. ``scheme`` names the
    loop weighting that the engine already applied; it is checked here so a mismatch between the
    configured loss mode and the engine fails instead of training the wrong objective.
    """
    if scheme not in LOOP_WEIGHT_SCHEMES:
        raise ValueError(f"Unknown RLTT scheme {scheme!r}; expected one of {LOOP_WEIGHT_SCHEMES}.")
    if "rltt_log_probs" not in model_output:
        raise RuntimeError(
            "model_output has no 'rltt_log_probs'. rltt_loss must run inside OrbitLMRLTTEngine's RLTT "
            "path (VERL_ENGINE_VENDOR=orbit and VERL_USE_EXTERNAL_MODULES=orbit_verl.plugin)."
        )

    # Slice every per-token output to the response tokens, padded to [B, R] (or [B, R, T]).
    rltt_log_prob = no_padding_2_padding(model_output["rltt_log_probs"], data)
    final_log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)
    loop_log_prob = model_output.get("loop_log_probs", None)
    if loop_log_prob is not None:
        loop_log_prob = no_padding_2_padding(loop_log_prob, data)
    exit_pdf = model_output.get("exit_pdf", None)
    if exit_pdf is not None:
        exit_pdf = no_padding_2_padding(exit_pdf, data)

    # Global batch info for the loss aggregation, filled exactly like stock ppo_loss does. Every
    # rank divides by a count over the whole mini-batch and multiplies by dp_size: by the token count
    # for token-mean, by the response count for seq-mean-token-mean. After FSDP averages the
    # gradients over ranks, the result is the mean over the whole mini-batch in either mode.
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # Same rule as stock ppo_loss: a loss normalized by global quantities is a partial sum per
    # micro-batch, so its metric must be summed over micro-batches to show the batch mean.
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    fields = ["response_mask", "advantages"]
    if config.use_kl_loss:
        if "ref_log_prob" not in data:
            raise KeyError(
                "use_kl_loss is set but the batch has no 'ref_log_prob'. The RLTT KL needs the "
                "reference model's final-loop log-probs."
            )
        fields.append("ref_log_prob")
    has_old = "old_log_probs" in data
    if has_old:
        fields.append("old_log_probs")
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    advantages = data["advantages"]
    loss_agg_mode = config.loss_agg_mode

    metrics: dict[str, Any] = {}

    # Policy gradient: REINFORCE on the loop-weighted log-prob, with no ratio and no clipping.
    pg_losses = -advantages * rltt_log_prob
    rollout_is_weights = data.get("rollout_is_weights", None)
    if rollout_is_weights is not None:
        # verl's rollout correction gives per-token importance weights between the vLLM sampler
        # and the trainer. Stock vanilla multiplies its per-token loss by them, and so do we.
        pg_losses = pg_losses * rollout_is_weights
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        policy_loss = policy_loss - config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    if config.use_kl_loss:
        # The KL uses the final loop only (RLTT paper, Eq. 7). The reference log-probs are final-loop
        # values too, so this is the same KL that stock GRPO would compute.
        kld = kl_penalty(logprob=final_log_prob, ref_logprob=data["ref_log_prob"], kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        policy_loss = policy_loss + kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    # Diagnostics. They are local masked means per micro-batch and are averaged over micro-batches,
    # like the policy-loss metrics of stock ppo_loss.
    names: list[str] = []
    columns: list[torch.Tensor] = []
    if exit_pdf is not None:
        loops = exit_pdf.shape[-1]
        for t in range(loops):
            names.append(f"rltt/exit_pdf_t{t}")
            columns.append(exit_pdf[..., t])
        steps = torch.arange(loops, device=exit_pdf.device, dtype=exit_pdf.dtype)
        names.append("rltt/expected_exit_step")
        columns.append((exit_pdf * steps).sum(-1))
    if loop_log_prob is not None:
        for t in range(loop_log_prob.shape[-1]):
            names.append(f"rltt/loop_logp_t{t}")
            columns.append(loop_log_prob[..., t])
    if has_old:
        old_log_prob = data["old_log_probs"]
        final_detached = final_log_prob.detach()
        # The final-loop log-probs of the current policy against the ones recorded before the
        # update. With one update per batch they agree up to numerics, so a large value means the
        # training forward and the old-log-prob forward disagree.
        names.append("rltt/final_vs_old_logp_absdiff")
        columns.append((final_detached - old_log_prob).abs())
        # Stock vanilla reports the same quantity under this name; keeping it lets GRPO and RLTT runs
        # share one dashboard panel.
        names.append("actor/ppo_kl")
        columns.append(old_log_prob - final_detached)
    for name, value in zip(names, _masked_means(columns, response_mask), strict=True):
        metrics[name] = Metric(value=value, aggregation=AggregationType.MEAN)

    return policy_loss, metrics
