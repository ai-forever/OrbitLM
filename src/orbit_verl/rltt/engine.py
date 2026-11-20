"""``OrbitLMRLTTEngine``: verl's FSDP language-model engine with the OrbitLM and RLTT additions.

verl builds the actor and the reference model through ``EngineRegistry``. This class is registered
for ``language_model`` on the ``fsdp`` and ``fsdp2`` backends under the vendor key ``orbit``, so
verl picks it only when ``VERL_ENGINE_VENDOR=orbit`` is set. The value model keeps verl's class.

It changes two things compared with ``FSDPEngineWithLMHead``:

1. For Ouro models every forward gets ``exit_at_step=T-1``. That pins the plain forward to the final
   loop, which is what vLLM samples from. Without it, an export whose config still has an exit
   threshold would read logits from a gate-selected loop. This applies to all passes, including
   the forward-only old-log-prob and reference passes.
2. When the actor's loss mode is one of the ``rltt_*`` modes, the training forward returns per-loop
   log-probs and the exit pdf instead of logits, and the loss becomes ``rltt_loss``. The flag that
   selects this path lives for one micro-batch and is cleared in a ``finally`` block. The exit pdf
   weights the loops as a constant unless ``ORBIT_RLTT_TRAIN_EXIT_GATE`` is true, which the
   launcher sets from ``orbit.rltt.train_exit_gate``.

The logic sits in module-level functions so it can be tested on the CPU without FSDP or a GPU. The
class methods only connect those functions to verl's call sequence:
``forward_step -> prepare_model_inputs -> model forward -> prepare_model_outputs -> loss``.
"""
from __future__ import annotations

import functools
from typing import Callable, Optional

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.utils.losses import ppo_loss

from orbit_verl.rltt.forward import install_rltt_forward, loop_weights
from orbit_verl.rltt.loss import RLTT_LOSS_MODES, progressive_alpha, rltt_loss, trains_exit_gate

ENGINE_VENDOR = "orbit"
_MODEL_TYPE = "language_model"
_BACKENDS = ("fsdp", "fsdp2")
_REGISTRY_KEY = ("cuda", ENGINE_VENDOR)


def ouro_loop_count(module) -> Optional[int]:
    """Number of loops (``total_ut_steps``) of an Ouro model, or None for any other model."""
    config = getattr(module, "config", None)
    if getattr(config, "model_type", None) != "ouro":
        return None
    loops = getattr(config, "total_ut_steps", None)
    if not loops or int(loops) <= 0:
        raise ValueError(f"Ouro model config has total_ut_steps={loops!r}; expected a positive integer.")
    return int(loops)


def _ppo_loss_config(loss_function):
    """Return the actor config if ``loss_function`` is verl's ``partial(ppo_loss, config=...)``."""
    if not isinstance(loss_function, functools.partial):
        return None
    func = loss_function.func
    # The identity check covers the normal case. The name check also accepts a partial that was
    # pickled and unpickled across a Ray call, which resolves to the same function by name anyway.
    same_function = func is ppo_loss or (
        getattr(func, "__module__", None) == ppo_loss.__module__
        and getattr(func, "__qualname__", None) == ppo_loss.__qualname__
    )
    if not same_function:
        return None
    if "config" in loss_function.keywords:
        return loss_function.keywords["config"]
    return loss_function.args[0] if loss_function.args else None


def swap_loss_function(loss_function, forward_only: bool) -> tuple[Optional[Callable], Optional[str]]:
    """Replace verl's ``ppo_loss`` partial by ``rltt_loss`` when the loss mode is an RLTT mode.

    Returns the loss function to use and the loop-weight scheme, or the unchanged function and None
    for every other case: no loss, a different loss, or a non-RLTT loss mode such as ``vanilla``.
    """
    config = _ppo_loss_config(loss_function)
    if config is None:
        return loss_function, None
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    scheme = RLTT_LOSS_MODES.get(loss_mode)
    if scheme is None:
        return loss_function, None
    if forward_only:
        # verl's log-prob passes call the engine with compute_loss=False, so no loss function
        # arrives here. A forward-only call that does carry the RLTT loss would otherwise reach the
        # raising stub through stock ppo_loss, whose message would wrongly blame a missing engine.
        raise RuntimeError(
            f"loss_mode {loss_mode}: OrbitLMRLTTEngine computes the RLTT loss only in training steps. "
            "Forward-only passes must run without a loss (compute_loss=False)."
        )
    return functools.partial(rltt_loss, config=config, scheme=scheme), scheme


def scalar_temperature(temperature) -> float:
    """Reduce verl's sampling temperature to one Python float.

    The RLTT forward divides each loop's logits by a single temperature. verl may pass a tensor
    with one value per sample; that is accepted only when all values are equal.
    """
    if isinstance(temperature, torch.Tensor):
        flat = temperature.detach().reshape(-1)
        if flat.numel() == 0:
            raise ValueError("The micro-batch carries an empty temperature tensor.")
        if not bool(torch.all(flat == flat[0])):
            raise ValueError(
                "The RLTT path supports one sampling temperature per micro-batch, but the samples "
                f"have different temperatures: {sorted(set(flat.tolist()))}."
            )
        return float(flat[0].item())
    return float(temperature)


def add_rltt_model_inputs(
    model_inputs: dict,
    output_args: dict,
    micro_batch: TensorDict,
    loop_count: int,
    scheme: Optional[str],
) -> dict:
    """Add the OrbitLM keyword arguments to the model inputs built by verl. Edits ``model_inputs`` in place.

    ``exit_at_step=T-1`` is always added. On the RLTT path (``scheme`` is not None) the function also
    adds the labels for the per-loop log-probs, the temperature as a Python float and the entropy
    flag. The labels are verl's own rolled packed input ids, so the per-loop log-probs line up with
    the stock ``log_probs`` token for token.
    """
    model_inputs["exit_at_step"] = int(loop_count) - 1
    if scheme is None:
        return model_inputs
    if "pad_size" not in output_args or "input_ids_rmpad_rolled" not in output_args:
        raise RuntimeError(
            "RLTT loss modes need packed inputs: set actor_rollout_ref.model.use_remove_padding=true."
        )
    if tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False):
        raise RuntimeError(
            "RLTT loss modes do not work with use_fused_kernels=true: the Ouro model returns per-loop "
            "outputs that verl's fused forward does not know. Set actor_rollout_ref.model.use_fused_kernels=false."
        )
    model_inputs["rltt_labels"] = output_args["input_ids_rmpad_rolled"].unsqueeze(0)
    # A Python float, not a tensor: FSDP casts floating-point tensor inputs of the root module to
    # the parameter dtype, and a bf16 temperature would shift every log-prob slightly.
    model_inputs["rltt_temperature"] = scalar_temperature(micro_batch["temperature"])
    model_inputs["rltt_need_entropy"] = bool(
        tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
    )
    return model_inputs


def build_rltt_model_output(
    output,
    unpad_fn: Callable[[torch.Tensor], torch.Tensor],
    offsets: torch.Tensor,
    scheme: str,
    alpha: float,
    *,
    train_exit_gate: bool = False,
) -> dict:
    """Turn the model's ``OrbitLMRLTTOutput`` into the ``model_output`` dict that ``rltt_loss`` reads.

    ``unpad_fn`` removes the packed-sequence padding that verl appended (and gathers the Ulysses
    shards when sequence parallelism is on). ``offsets`` are the sequence offsets of the
    micro-batch's nested ``input_ids``; every result is a jagged nested tensor over them, like the
    stock ``log_probs``.

    The loop weighting is applied before un-padding, so the per-loop values and the exit pdf can be
    detached before they are gathered. Only the weighted sum, the final loop and the entropy keep
    their gradients; the detached copies are there for the metrics. The weighted sum reaches the
    exit gate only with ``train_exit_gate=True`` (see ``loop_weights``).
    """
    if output.log_probs is None or output.loop_log_probs is None:
        raise RuntimeError("The model did not return RLTT outputs; was install_rltt_forward applied?")
    if output.log_probs.dim() != 2 or output.log_probs.shape[0] != 1:
        raise RuntimeError(
            f"Expected packed RLTT outputs of shape [1, tokens], got {tuple(output.log_probs.shape)}."
        )
    loop_log_probs = output.loop_log_probs.squeeze(0)  # [tokens, T]
    exit_pdf = output.exit_pdf.squeeze(0) if output.exit_pdf is not None else None
    weights = loop_weights(
        scheme, exit_pdf, alpha, num_loops=loop_log_probs.shape[-1], train_exit_gate=train_exit_gate
    )
    rltt_log_probs = (loop_log_probs * weights).sum(-1)

    def nested(values: torch.Tensor) -> torch.Tensor:
        return torch.nested.nested_tensor_from_jagged(unpad_fn(values), offsets)

    model_output = {
        "log_probs": nested(output.log_probs.squeeze(0)),
        "rltt_log_probs": nested(rltt_log_probs),
        "loop_log_probs": nested(loop_log_probs.detach()),
    }
    if exit_pdf is not None:
        model_output["exit_pdf"] = nested(exit_pdf.detach())
    if output.entropy is not None:
        model_output["entropy"] = nested(output.entropy.squeeze(0))
    return model_output


def _check_rltt_output_flags(micro_batch: TensorDict) -> None:
    """Refuse verl features whose extra outputs the RLTT path does not produce."""
    unsupported = [
        key
        for key in ("calculate_sum_pi_squared", "distillation_use_topk", "distillation_only")
        if tu.get_non_tensor_data(data=micro_batch, key=key, default=False)
    ]
    if unsupported:
        raise NotImplementedError(f"RLTT loss modes do not support {', '.join(unsupported)}.")


def _forget_previous_registration() -> None:
    """Drop this class's registry entries left by an earlier execution of this same module.

    ``EngineRegistry.register`` refuses a key that is already taken. A plain second import is a
    no-op and never gets here, but ``importlib.reload`` runs the module again. Only entries created
    by this module under this class name are removed, so a different engine that claims the same
    key still triggers verl's error. The registry has no public removal API, hence ``_engines``.
    """
    registry = EngineRegistry._engines.get(_MODEL_TYPE, {})
    for backend in _BACKENDS:
        entries = registry.get(backend, {})
        previous = entries.get(_REGISTRY_KEY)
        if (
            previous is not None
            and previous.__module__ == __name__
            and previous.__qualname__ == "OrbitLMRLTTEngine"
        ):
            del entries[_REGISTRY_KEY]


_forget_previous_registration()


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="cuda", vendor="orbit")
class OrbitLMRLTTEngine(FSDPEngineWithLMHead):
    """FSDP actor/ref engine for OrbitLM: final-loop pinning for all passes, RLTT for training."""

    # Set in _build_module: the number of loops of an Ouro model, or None for other models.
    _orbit_loop_count: Optional[int] = None
    # The loop-weight scheme of the micro-batch being processed, or None on the stock path.
    _rltt_scheme: Optional[str] = None

    def _build_module(self):
        module = super()._build_module()
        self._orbit_loop_count = ouro_loop_count(module)
        if self._orbit_loop_count is not None:
            install_rltt_forward(module)
        # One line per built model (actor and reference) in the run log. It shows that verl picked
        # this engine, which the metrics alone show only for the rltt_* loss modes.
        print(
            f"[orbit_verl] OrbitLMRLTTEngine built {type(module).__name__}: "
            f"loops={self._orbit_loop_count}, final loop pinned={self._orbit_loop_count is not None}",
            flush=True,
        )
        return module

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        loss_function, scheme = swap_loss_function(loss_function, forward_only)
        if scheme is not None and self._orbit_loop_count is None:
            raise RuntimeError(
                "RLTT loss modes need a OrbitLM (Ouro) model, but the actor's config has model_type "
                f"{getattr(self.model_config.hf_config, 'model_type', None)!r}."
            )
        self._rltt_scheme = scheme
        try:
            return super().forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
        finally:
            self._rltt_scheme = None

    def prepare_model_inputs(self, micro_batch: TensorDict):
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        if self._orbit_loop_count is not None:
            add_rltt_model_inputs(model_inputs, output_args, micro_batch, self._orbit_loop_count, self._rltt_scheme)
        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        if self._rltt_scheme is None:
            return super().prepare_model_outputs(
                output=output,
                output_args=output_args,
                micro_batch=micro_batch,
                logits_processor_func=logits_processor_func,
            )
        _check_rltt_output_flags(micro_batch)
        pad_size = output_args["pad_size"]
        return build_rltt_model_output(
            output,
            unpad_fn=lambda tensor: self._gather_and_unpad_packed(tensor, pad_size),
            offsets=micro_batch["input_ids"].offsets(),
            scheme=self._rltt_scheme,
            alpha=progressive_alpha(),
            train_exit_gate=trains_exit_gate(),
        )
