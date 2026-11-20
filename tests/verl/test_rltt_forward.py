"""CPU tests for ``orbit_verl.rltt.forward`` and the pure helpers of ``orbit_verl.rltt.engine``.

The forward wrapper is checked on a small fake model that follows the ``OuroForCausalLM`` forward
contract (``return_orbit_aux`` returns one hidden state and one gate logit per loop), and on the
ported tiny Ouro model. The reference for every number is the frozen RLTT reference implementation
in ``rltt_reference.py``: ``weighted_token_logprobs``, ``build_exit_pdf``, ``token_logprobs`` and
the final-loop entropy, all at temperature 0.9.

Both paths run on the same model and the same weights. The reference path calls the original
forward with ``return_orbit_aux=True`` and applies the reference functions; the tested path calls
the patched forward with ``rltt_labels``. Gradients are compared on the model parameters and on the
loop hidden states and gate logits (kept with ``retain_grad``), so the comparison covers the lm_head
weight, the hidden states and the gate.

verl's ``FusedLinearForPPO`` uses flash-attn's Triton cross-entropy when flash-attn imports. That
kernel needs CUDA tensors, so the tests switch it off and use the PyTorch fallback.
"""
from __future__ import annotations

import copy
import functools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from tensordict import TensorDict
from torch import nn
from transformers.utils import ModelOutput

import verl.utils.experimental.torch_functional as verl_torch_functional
from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss

import rltt_reference as ALGO
from orbit_verl.rltt.engine import (
    add_rltt_model_inputs,
    build_rltt_model_output,
    ouro_loop_count,
    scalar_temperature,
    swap_loss_function,
)
from orbit_verl.rltt.forward import (
    OrbitLMRLTTOutput,
    install_rltt_forward,
    loop_weights,
    rltt_chunk_size,
    stock_temperature,
)
from orbit_verl.rltt.loss import rltt_loss

# The reference module holds the exit pdf as well as the loss pieces.
LOSSES = ALGO

TEMPERATURE = 0.9
REFERENCE_SCHEME = {"exit": "exit", "uniform": "uniform", "progressive": "progressive", "last": "grpo"}


@pytest.fixture(autouse=True)
def _cpu_cross_entropy(monkeypatch):
    monkeypatch.setattr(verl_torch_functional, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)


@dataclass
class FakeOuroOutput(ModelOutput):
    logits: Optional[torch.Tensor] = None
    loop_hidden_states: Optional[tuple] = None
    loop_gate_logits: Optional[tuple] = None


class FakeBlock(nn.Module):
    """The fake model's shared layer. It is a class of its own so FSDP2 can wrap it like a decoder layer."""

    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, h):
        return torch.tanh(self.linear(h))


class FakeOuroForCausalLM(nn.Module):
    """A tiny stand-in for ``OuroForCausalLM`` that keeps its forward contract.

    Every loop applies the same block to the previous loop's output and then the final norm, as the
    real model does. ``early_exit_gate`` reads each loop's normed hidden state. With
    ``return_orbit_aux=True, compute_logits=False`` all loops run and no logits are computed. On
    the plain path ``exit_at_step`` limits the loops and the logits come from the last loop run.
    The keyword arguments of every call are recorded.
    """

    # verl's FSDP2 setup wraps the classes listed here, plus embed_tokens and lm_head, as units.
    _no_split_modules = ["FakeBlock"]

    def __init__(self, vocab: int = 37, hidden: int = 16, loops: int = 4, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(model_type="ouro", total_ut_steps=loops)
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.block = FakeBlock(hidden)
        self.norm = nn.LayerNorm(hidden)
        self.early_exit_gate = nn.Linear(hidden, 1)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.calls: list[dict] = []

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        use_cache=None,
        exit_at_step=None,
        return_orbit_aux=False,
        compute_logits=True,
        use_weighted_exit=False,
        **kwargs,
    ):
        self.calls.append(
            dict(
                exit_at_step=exit_at_step,
                return_orbit_aux=return_orbit_aux,
                compute_logits=compute_logits,
                use_weighted_exit=use_weighted_exit,
                position_ids=position_ids,
                use_cache=use_cache,
                extra=dict(kwargs),
            )
        )
        loops = self.config.total_ut_steps
        if compute_logits and exit_at_step is not None:
            loops = min(loops, int(exit_at_step) + 1)
        h = self.embed_tokens(input_ids)
        hidden, gates = [], []
        for _ in range(loops):
            h = self.norm(h + self.block(h))
            hidden.append(h)
            gates.append(self.early_exit_gate(h))
        return FakeOuroOutput(
            logits=self.lm_head(hidden[-1]) if compute_logits else None,
            loop_hidden_states=tuple(hidden) if return_orbit_aux else None,
            loop_gate_logits=tuple(gates) if return_orbit_aux else None,
        )


def _fake_model(**kwargs) -> FakeOuroForCausalLM:
    """A fake model whose class is a fresh subclass, created for this one test.

    install_rltt_forward patches the class. Giving every test its own class means each test starts
    from the unpatched forward and no patch leaks from one test into the next.
    """
    cls = type("FakeOuroUnderTest", (FakeOuroForCausalLM,), {})
    return cls(**kwargs)


def _inputs(batch: int, length: int, vocab: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab, (batch, length), generator=g)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    position_ids = torch.arange(length).unsqueeze(0).expand(batch, -1)
    coeff = torch.randn(batch, length, generator=g)
    return input_ids, labels, position_ids, coeff


def _grads(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


class LoopRecorder:
    """Keeps every loop's hidden state and gate logits, with gradients, through forward hooks.

    Both the fake and the ported model apply the final norm and the exit gate once per loop, and the
    norm output is exactly the loop hidden state the forward returns. Hooks on those two modules
    therefore see the loop tensors on both paths, including inside the RLTT wrapper.
    """

    def __init__(self, norm: nn.Module, gate: nn.Module):
        self.hidden: list[torch.Tensor] = []
        self.gates: list[torch.Tensor] = []
        norm.register_forward_hook(lambda module, args, output: self._keep(self.hidden, output))
        gate.register_forward_hook(lambda module, args, output: self._keep(self.gates, output))

    @staticmethod
    def _keep(store, output):
        if output.requires_grad:
            output.retain_grad()
        store.append(output)

    def reset(self):
        self.hidden, self.gates = [], []

    def grads(self):
        # A tensor that the loss does not depend on has no gradient. With the static loop weights
        # that is true of every gate output, on both paths alike.
        def copy(tensor):
            return None if tensor.grad is None else tensor.grad.clone()

        return [copy(h) for h in self.hidden], [copy(g) for g in self.gates]


def _reference_pass(model, recorder, input_ids, position_ids, labels, coeff, scheme, alpha):
    """The reference computation on the original forward's loop outputs."""
    model.zero_grad(set_to_none=True)
    recorder.reset()
    original = type(model)._orbit_rltt_original_forward
    out = original(model, input_ids=input_ids, position_ids=position_ids, use_cache=False,
                   return_orbit_aux=True, compute_logits=False)
    hidden, gates = list(out.loop_hidden_states), list(out.loop_gate_logits)
    assert all(a is b for a, b in zip(hidden, recorder.hidden)) and len(hidden) == len(recorder.hidden)
    exit_pdf = LOSSES.build_exit_pdf(gates)
    weighted = ALGO.weighted_token_logprobs(
        hidden, gates, model.lm_head, labels,
        scheme=REFERENCE_SCHEME[scheme], alpha=alpha,
        exit_pdf=exit_pdf if scheme == "exit" else None, temperature=TEMPERATURE,
    )
    per_loop = torch.stack(
        [ALGO.token_logprobs(model.lm_head(h), labels, temperature=TEMPERATURE) for h in hidden], dim=-1
    )
    entropy = ALGO.final_loop_token_entropy(hidden[-1], model.lm_head, temperature=TEMPERATURE)
    (weighted * coeff).sum().backward()
    hidden_grads, gate_grads = recorder.grads()
    return dict(
        weighted=weighted.detach(), per_loop=per_loop.detach(), exit_pdf=exit_pdf.detach(),
        entropy=entropy.detach(), params=_grads(model), hidden_grads=hidden_grads, gate_grads=gate_grads,
    )


def _rltt_pass(model, recorder, input_ids, position_ids, labels, coeff, scheme, alpha):
    """The patched forward plus the loop weighting that the engine applies.

    The reference reference trains the exit gate through the loss, so this pass does too.
    """
    model.zero_grad(set_to_none=True)
    recorder.reset()
    out = model(
        input_ids=input_ids, position_ids=position_ids, use_cache=False, rltt_labels=labels,
        rltt_temperature=TEMPERATURE, rltt_need_entropy=True,
    )
    weights = loop_weights(scheme, out.exit_pdf, alpha, train_exit_gate=True)
    weighted = (out.loop_log_probs * weights).sum(-1)
    (weighted * coeff).sum().backward()
    hidden_grads, gate_grads = recorder.grads()
    return dict(out=out, weighted=weighted.detach(), params=_grads(model), hidden_grads=hidden_grads,
                gate_grads=gate_grads)


def _max_abs(a, b) -> float:
    if a is None or b is None:
        # Both paths must agree on which tensors receive a gradient at all.
        assert a is None and b is None, "one path produced a gradient that the other did not"
        return 0.0
    return float((a - b).abs().max().item())


def _compare(reference, tested, scheme, tag):
    out = tested["out"]
    assert set(tested["params"]) == set(reference["params"]), (tested["params"].keys(), reference["params"].keys())
    assert len(tested["hidden_grads"]) == len(reference["hidden_grads"])
    # Every loop's hidden state feeds the loss on both paths. The gates do only for exit weights,
    # and even then not the last gate: the exit pdf gives the last loop the remaining probability.
    assert all(grad is not None for grad in tested["hidden_grads"])
    gate_has_grad = [grad is not None for grad in tested["gate_grads"]]
    if scheme == "exit":
        assert all(gate_has_grad[:-1]) and not gate_has_grad[-1], gate_has_grad
    else:
        assert not any(gate_has_grad), gate_has_grad
    diffs = {
        "per_loop": _max_abs(out.loop_log_probs.detach(), reference["per_loop"]),
        "weighted": _max_abs(tested["weighted"], reference["weighted"]),
        "exit_pdf": _max_abs(out.exit_pdf.detach(), reference["exit_pdf"]),
        "final": _max_abs(out.log_probs.detach(), reference["per_loop"][..., -1]),
        "entropy": _max_abs(out.entropy.detach().float(), reference["entropy"]),
        "grad_hidden": max(_max_abs(a, b) for a, b in zip(tested["hidden_grads"], reference["hidden_grads"])),
        "grad_gate": max(_max_abs(a, b) for a, b in zip(tested["gate_grads"], reference["gate_grads"])),
        "grad_lm_head": _max_abs(tested["params"]["lm_head.weight"], reference["params"]["lm_head.weight"]),
        # Parameter gradients sum over every token and every loop, so their float32 rounding grows
        # with their size. They are compared relative to the largest entry once that exceeds 1.
        "grad_params_rel": max(
            _max_abs(tested["params"][name], grad) / max(1.0, float(grad.abs().max()))
            for name, grad in reference["params"].items()
        ),
    }
    print(f"PARITY forward {tag} scheme={scheme} " + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()))
    for key, value in diffs.items():
        assert value < 2e-5, (key, value)
    return diffs


@pytest.mark.parametrize("scheme", ["exit", "uniform", "progressive", "last"])
@pytest.mark.parametrize("batch", [1, 2])
def test_fake_model_matches_reference(scheme, batch):
    model = _fake_model()
    install_rltt_forward(model)
    recorder = LoopRecorder(model.norm, model.early_exit_gate)
    input_ids, labels, position_ids, coeff = _inputs(batch, 11, vocab=37, seed=batch)
    alpha = 2.0 if scheme == "progressive" else 1.0

    reference = _reference_pass(model, recorder, input_ids, position_ids, labels, coeff, scheme, alpha)
    tested = _rltt_pass(model, recorder, input_ids, position_ids, labels, coeff, scheme, alpha)
    _compare(reference, tested, scheme, f"fake[B={batch}]")

    gate_params = ("early_exit_gate.weight", "early_exit_gate.bias")
    if scheme == "exit":
        assert all(tested["params"][name].abs().sum() > 0 for name in gate_params)
    else:
        # Static weights do not depend on the gate, so the gate gets no gradient on either path.
        assert all(name not in tested["params"] for name in gate_params)


def test_rltt_call_arguments_and_plain_path_untouched():
    model = _fake_model()
    original = type(model).forward
    install_rltt_forward(model)
    input_ids, labels, position_ids, _ = _inputs(1, 9, vocab=37)

    with torch.no_grad():
        plain = model(input_ids=input_ids, position_ids=position_ids, exit_at_step=1, use_cache=False, foo="bar")
        direct = original(model, input_ids=input_ids, position_ids=position_ids, exit_at_step=1, use_cache=False,
                          foo="bar")
    assert isinstance(plain, FakeOuroOutput)
    assert torch.equal(plain.logits, direct.logits)
    assert model.calls[-1]["exit_at_step"] == 1 and model.calls[-1]["return_orbit_aux"] is False
    assert model.calls[-1]["extra"] == {"foo": "bar"} and model.calls[-2]["extra"] == {"foo": "bar"}

    out = model(input_ids=input_ids, position_ids=position_ids, exit_at_step=3, use_cache=False, foo="bar",
                rltt_labels=labels, rltt_temperature=TEMPERATURE)
    call = model.calls[-1]
    assert isinstance(out, OrbitLMRLTTOutput)
    # exit_at_step is dropped, all loops run without logits, and no rltt_* argument leaks through.
    assert call["exit_at_step"] is None
    assert call["return_orbit_aux"] is True and call["compute_logits"] is False and call["use_weighted_exit"] is False
    assert call["use_cache"] is False and torch.equal(call["position_ids"], position_ids)
    assert call["extra"] == {"foo": "bar"}
    assert out.loop_log_probs.shape == (1, 9, 4) and out.log_probs.shape == (1, 9) and out.exit_pdf.shape == (1, 9, 4)
    assert out.entropy is None
    assert torch.allclose(out.exit_pdf.sum(-1), torch.ones(1, 9))


def test_install_is_idempotent_and_class_level():
    import inspect

    model = _fake_model()
    original = type(model).forward
    install_rltt_forward(model)
    patched = type(model).forward
    install_rltt_forward(model)
    install_rltt_forward(type(model)(seed=1))
    assert type(model).forward is patched
    assert type(model)._orbit_rltt_original_forward is original
    # The patch lives on the class, so a second instance has it without its own install call.
    other = type(model)(seed=2)
    assert type(other).forward is patched and "forward" not in vars(other)
    # A subclass inherits the patched forward; installing on it must not wrap it a second time.
    subclass = type("Sub", (type(model),), {})
    install_rltt_forward(subclass())
    assert subclass.forward is patched and "forward" not in vars(subclass)
    # inspect.signature still reports the original parameters (verl looks for "cu_seqlens" there).
    assert inspect.signature(patched) == inspect.signature(original)


def test_chunk_size_does_not_change_results(monkeypatch):
    model = _fake_model()
    install_rltt_forward(model)
    input_ids, labels, position_ids, _ = _inputs(1, 13, vocab=37)
    results = []
    for chunk in ("3", "512"):
        monkeypatch.setenv("ORBIT_RLTT_CHUNK_SIZE", chunk)
        with torch.no_grad():
            results.append(model(input_ids=input_ids, position_ids=position_ids, rltt_labels=labels).loop_log_probs)
    assert torch.allclose(results[0], results[1], atol=1e-6)


def test_chunk_size_env(monkeypatch):
    monkeypatch.delenv("ORBIT_RLTT_CHUNK_SIZE", raising=False)
    assert rltt_chunk_size() == 512
    monkeypatch.setenv("ORBIT_RLTT_CHUNK_SIZE", "1024")
    assert rltt_chunk_size() == 1024
    for bad in ("0", "-4", "big"):
        monkeypatch.setenv("ORBIT_RLTT_CHUNK_SIZE", bad)
        with pytest.raises(ValueError, match="ORBIT_RLTT_CHUNK_SIZE"):
            rltt_chunk_size()


def test_rltt_temperature_rounds_like_the_stock_path():
    # verl's stock log-prob path divides the logits by the temperature cast to the logits' dtype.
    # Under bf16 autocast 0.9 becomes 0.8984375, and the RLTT final loop must use the same number,
    # or it drifts from the old and reference log-probs that the stock path computes.
    model = _fake_model()
    install_rltt_forward(model)
    input_ids, labels, position_ids, _ = _inputs(1, 11, vocab=37)
    weight = model.lm_head.weight
    assert stock_temperature(TEMPERATURE, weight) == float(torch.tensor(TEMPERATURE, dtype=torch.float32))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert stock_temperature(TEMPERATURE, weight) == 0.8984375
        logits = model(input_ids=input_ids, position_ids=position_ids, exit_at_step=3).logits
        assert logits.dtype == torch.bfloat16
        temperature = torch.full(labels.shape, TEMPERATURE, dtype=torch.float32).unsqueeze(-1)
        scaled = (logits / temperature.to(logits.dtype)).float()
        stock = scaled.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        rltt = model(input_ids=input_ids, position_ids=position_ids, rltt_labels=labels,
                     rltt_temperature=TEMPERATURE).log_probs
    assert torch.allclose(rltt.float(), stock, atol=1e-6, rtol=0)


def test_loop_weights_schemes():
    pdf = torch.softmax(torch.randn(2, 5, 4), dim=-1).requires_grad_(True)
    # By default the exit weights are constants of the loss: the same values, cut from the gate.
    constant = loop_weights("exit", pdf)
    assert torch.equal(constant, pdf) and not constant.requires_grad
    assert loop_weights("exit", pdf, train_exit_gate=True) is pdf
    assert torch.allclose(loop_weights("last", pdf), torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(loop_weights("uniform", None, num_loops=4), torch.full((4,), 0.25))
    progressive = loop_weights("progressive", pdf, alpha=2.0)
    assert torch.allclose(progressive, torch.tensor([1.0, 4.0, 9.0, 16.0]) / 30.0)
    with pytest.raises(ValueError, match="use_exit_gate"):
        loop_weights("exit", None, num_loops=4)
    with pytest.raises(ValueError, match="Unknown"):
        loop_weights("learned", pdf)


@pytest.mark.parametrize("kind", ["fake", "ported"])
def test_fsdp2_sharded_lm_head_matches_plain_model(kind):
    """verl's FSDP2 setup shards lm_head as a unit of its own; the RLTT path must still be exact.

    Read from outside its own call, a sharded lm_head weight is a DTensor shard. The RLTT forward
    computes the log-probs as a call of lm_head, which makes FSDP2 gather the weight first and
    reduce its gradient afterwards. This runs verl's own ``apply_fsdp2`` on the model in a
    one-process gloo group on the CPU and compares outputs and gradients with an unsharded copy.
    With the ported model each decoder layer is an FSDP2 unit that runs four times per forward,
    so the test also checks that FSDP2 accumulates those layers' gradients over the loops.
    """
    import socket

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from verl.utils.fsdp_utils import apply_fsdp2

    if dist.is_initialized():
        pytest.skip("a torch.distributed process group already exists in this process")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        plain = _fake_model() if kind == "fake" else _ported_tiny_model()
        sharded = copy.deepcopy(plain)
        # The patch goes on before FSDP wraps the model, in the same order as the engine's _build_module.
        install_rltt_forward(plain)
        apply_fsdp2(sharded, {"mesh": init_device_mesh("cpu", (1,)), "reshard_after_forward": True}, {})
        assert type(sharded.lm_head.weight).__name__ == "DTensor"
        # fully_shard swaps the module's class for a generated subclass; it must inherit the patch.
        assert type(sharded) is not type(plain) and type(sharded).forward is type(plain).forward

        input_ids, labels, _, coeff = _inputs(1, 12, vocab=plain.lm_head.out_features, seed=5)
        results = []
        for model in (plain, sharded):
            out = model(input_ids=input_ids, use_cache=False, rltt_labels=labels, rltt_temperature=TEMPERATURE,
                        rltt_need_entropy=True)
            loss = ((out.loop_log_probs * out.exit_pdf).sum(-1) * coeff).sum() + 0.1 * out.entropy.sum()
            loss.backward()
            grads = {
                name: (p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad).detach()
                for name, p in model.named_parameters()
            }
            results.append((out, grads))
        (plain_out, plain_grads), (sharded_out, sharded_grads) = results
        diffs = {
            "loop_log_probs": _max_abs(plain_out.loop_log_probs.detach(), sharded_out.loop_log_probs.detach()),
            "exit_pdf": _max_abs(plain_out.exit_pdf.detach(), sharded_out.exit_pdf.detach()),
            "entropy": _max_abs(plain_out.entropy.detach(), sharded_out.entropy.detach()),
            "grads": max(_max_abs(plain_grads[name], sharded_grads[name]) for name in plain_grads),
        }
        print(f"PARITY forward fsdp2 {kind} " + " ".join(f"{k}={v:.2e}" for k, v in diffs.items()))
        assert set(plain_grads) == set(sharded_grads)
        assert plain_grads["lm_head.weight"].abs().sum() > 0
        for key, value in diffs.items():
            assert value < 1e-5, (key, value)
    finally:
        dist.destroy_process_group()


PORTED_LOOPS = 4
PORTED_VOCAB = 97


def _ported_tiny_model():
    """A tiny random model of the ported Ouro code, or a skip when the port is not importable."""
    modelling = pytest.importorskip("orbit_verl.model.modelling_ouro")
    configuration = pytest.importorskip("orbit_verl.model.configuration_ouro")
    # transformers 5 no longer gives configs default token ids, and the model reads pad_token_id,
    # so they are passed explicitly, as the real exports' config.json files do.
    config = configuration.OuroConfig(
        vocab_size=PORTED_VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, head_dim=8, max_position_embeddings=64, rope_theta=40000.0, rope_scaling=None,
        total_ut_steps=PORTED_LOOPS, early_exit_threshold=None, use_exit_gate=True,
        bos_token_id=0, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = modelling.OuroForCausalLM(config).float()
    # HF's init (std 0.02) leaves a tiny model's logits and gate outputs almost flat. Larger seeded
    # weights make the per-loop log-probs and the exit pdf differ between loops, which is what the
    # comparison needs in order to mean something.
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn(param.shape, generator=generator) * 0.1
            param.copy_(1.0 + noise if "norm" in name else noise)
    return model


def test_ported_tiny_model_matches_reference():
    model = _ported_tiny_model()
    loops = PORTED_LOOPS
    assert ouro_loop_count(model) == loops
    install_rltt_forward(model)
    recorder = LoopRecorder(model.model.norm, model.model.early_exit_gate)
    input_ids, labels, position_ids, coeff = _inputs(1, 10, vocab=PORTED_VOCAB, seed=3)

    for scheme in ("exit", "uniform", "progressive", "last"):
        alpha = 2.0 if scheme == "progressive" else 1.0
        reference = _reference_pass(model, recorder, input_ids, position_ids, labels, coeff, scheme, alpha)
        tested = _rltt_pass(model, recorder, input_ids, position_ids, labels, coeff, scheme, alpha)
        _compare(reference, tested, scheme, "ported")

    # The plain path with exit_at_step = T - 1 reads the final loop, the same as the RLTT log_probs.
    with torch.no_grad():
        plain = model(input_ids=input_ids, position_ids=position_ids, exit_at_step=loops - 1, use_cache=False)
        rltt = model(input_ids=input_ids, position_ids=position_ids, exit_at_step=loops - 1, use_cache=False,
                     rltt_labels=labels, rltt_temperature=TEMPERATURE)
    diff = _max_abs(ALGO.token_logprobs(plain.logits, labels, temperature=TEMPERATURE), rltt.log_probs)
    print(f"PARITY forward ported plain(exit_at_step=T-1) vs rltt final |d|={diff:.2e}")
    assert diff < 2e-5


# ---------------------------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------------------------


def _actor_config(loss_mode: str) -> ActorConfig:
    return ActorConfig(strategy="fsdp", rollout_n=4, ppo_micro_batch_size_per_gpu=1,
                       policy_loss=PolicyLossConfig(loss_mode=loss_mode))


def test_swap_loss_function():
    config = _actor_config("rltt_progressive")
    stock = functools.partial(ppo_loss, config=config)
    swapped, scheme = swap_loss_function(stock, forward_only=False)
    assert scheme == "progressive"
    assert swapped.func is rltt_loss and swapped.keywords == {"config": config, "scheme": "progressive"}

    with pytest.raises(RuntimeError, match="compute_loss=False"):
        swap_loss_function(stock, forward_only=True)

    for mode in ("vanilla", "gspo"):
        vanilla = functools.partial(ppo_loss, config=_actor_config(mode))
        assert swap_loss_function(vanilla, forward_only=False) == (vanilla, None)
        assert swap_loss_function(vanilla, forward_only=True) == (vanilla, None)
    assert swap_loss_function(None, forward_only=True) == (None, None)
    other = functools.partial(lambda **kw: None, config=config)
    assert swap_loss_function(other, forward_only=False) == (other, None)
    positional = functools.partial(ppo_loss, _actor_config("rltt_exit"))
    assert swap_loss_function(positional, forward_only=False)[1] == "exit"


def _packed_micro_batch(lengths, temperature=0.9, **non_tensor):
    rows = [torch.arange(n) + 10 * i for i, n in enumerate(lengths)]
    td = TensorDict({"input_ids": torch.nested.as_nested_tensor(rows, layout=torch.jagged)},
                    batch_size=[len(lengths)])
    tu.assign_non_tensor(td, temperature=temperature, **non_tensor)
    return td


def test_add_rltt_model_inputs():
    micro_batch = _packed_micro_batch([3, 4], calculate_entropy=True)
    rolled = torch.arange(9)
    output_args = {"pad_size": 2, "input_ids_rmpad_rolled": rolled, "temperature_rmpad": torch.ones(9)}

    plain = add_rltt_model_inputs({"input_ids": None}, dict(output_args), micro_batch, loop_count=4, scheme=None)
    assert plain == {"input_ids": None, "exit_at_step": 3}

    inputs = add_rltt_model_inputs({}, dict(output_args), micro_batch, loop_count=4, scheme="exit")
    assert inputs["exit_at_step"] == 3
    assert torch.equal(inputs["rltt_labels"], rolled.unsqueeze(0))
    assert isinstance(inputs["rltt_temperature"], float) and abs(inputs["rltt_temperature"] - 0.9) < 1e-12
    assert inputs["rltt_need_entropy"] is True

    with pytest.raises(RuntimeError, match="use_remove_padding"):
        add_rltt_model_inputs({}, {"temperature": torch.ones(2)}, micro_batch, loop_count=4, scheme="exit")
    fused = _packed_micro_batch([3, 4], use_fused_kernels=True)
    with pytest.raises(RuntimeError, match="use_fused_kernels"):
        add_rltt_model_inputs({}, dict(output_args), fused, loop_count=4, scheme="exit")
    # Without the flag the entropy is not requested.
    quiet = add_rltt_model_inputs({}, dict(output_args), _packed_micro_batch([3, 4]), loop_count=4, scheme="last")
    assert quiet["rltt_need_entropy"] is False


def test_scalar_temperature():
    assert scalar_temperature(0.7) == 0.7
    assert scalar_temperature(torch.full((3,), 0.6)) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="different temperatures"):
        scalar_temperature(torch.tensor([0.6, 1.0]))


def test_build_rltt_model_output_unpads_weights_and_detaches():
    lengths, pad, loops = [3, 5], 2, 4
    nnz = sum(lengths)
    offsets = _packed_micro_batch(lengths)["input_ids"].offsets()
    g = torch.Generator().manual_seed(0)
    loop_leaf = (-torch.rand(1, nnz + pad, loops, generator=g)).requires_grad_(True)
    gate_leaf = torch.randn(1, nnz + pad, loops, generator=g).requires_grad_(True)
    entropy_leaf = torch.rand(1, nnz + pad, generator=g).requires_grad_(True)
    exit_pdf = LOSSES.build_exit_pdf([gate_leaf[..., t : t + 1] for t in range(loops)])
    output = OrbitLMRLTTOutput(
        log_probs=loop_leaf[..., -1].contiguous(), loop_log_probs=loop_leaf, exit_pdf=exit_pdf, entropy=entropy_leaf
    )
    unpad_calls = []

    def unpad(tensor):
        unpad_calls.append(tuple(tensor.shape))
        return tensor[:-pad]

    result = build_rltt_model_output(output, unpad, offsets, scheme="exit", alpha=1.0, train_exit_gate=True)
    assert set(result) == {"log_probs", "rltt_log_probs", "loop_log_probs", "exit_pdf", "entropy"}
    assert all(value.is_nested for value in result.values())
    assert all(shape[0] == nnz + pad for shape in unpad_calls)
    expected = (loop_leaf[0, :nnz] * exit_pdf[0, :nnz]).sum(-1)
    assert torch.allclose(result["rltt_log_probs"].values(), expected)
    assert torch.equal(result["log_probs"].values(), loop_leaf[0, :nnz, -1].detach())
    assert result["loop_log_probs"].values().shape == (nnz, loops)
    assert not result["loop_log_probs"].requires_grad and not result["exit_pdf"].requires_grad
    assert result["rltt_log_probs"].requires_grad and result["entropy"].requires_grad
    # With train_exit_gate the exit weights carry the gradient to the gate logits; padded positions get none.
    result["rltt_log_probs"].values().sum().backward()
    assert gate_leaf.grad[0, :nnz].abs().sum() > 0 and gate_leaf.grad[0, nnz:].abs().sum() == 0

    # By default the same weighted sum reaches the per-loop log-probs but not the gate.
    loop_leaf.grad, gate_leaf.grad = None, None
    exit_pdf = LOSSES.build_exit_pdf([gate_leaf[..., t : t + 1] for t in range(loops)])
    output = OrbitLMRLTTOutput(log_probs=loop_leaf[..., -1].contiguous(), loop_log_probs=loop_leaf, exit_pdf=exit_pdf)
    default = build_rltt_model_output(output, unpad, offsets, scheme="exit", alpha=1.0)
    assert torch.allclose(default["rltt_log_probs"].values(), expected)
    default["rltt_log_probs"].values().sum().backward()
    assert gate_leaf.grad is None and loop_leaf.grad[0, :nnz].abs().sum() > 0

    static = build_rltt_model_output(
        OrbitLMRLTTOutput(log_probs=loop_leaf[..., -1].contiguous(), loop_log_probs=loop_leaf),
        unpad, offsets, scheme="uniform", alpha=1.0,
    )
    assert "exit_pdf" not in static and "entropy" not in static
    assert torch.allclose(static["rltt_log_probs"].values(), loop_leaf[0, :nnz].mean(-1))
    with pytest.raises(ValueError, match="use_exit_gate"):
        build_rltt_model_output(
            OrbitLMRLTTOutput(log_probs=loop_leaf[..., -1].contiguous(), loop_log_probs=loop_leaf),
            unpad, offsets, scheme="exit", alpha=1.0,
        )


def test_ouro_loop_count():
    assert ouro_loop_count(FakeOuroForCausalLM(loops=3)) == 3
    assert ouro_loop_count(SimpleNamespace(config=SimpleNamespace(model_type="qwen2"))) is None
    with pytest.raises(ValueError, match="total_ut_steps"):
        ouro_loop_count(SimpleNamespace(config=SimpleNamespace(model_type="ouro", total_ut_steps=0)))


@pytest.mark.parametrize("train_exit_gate", [False, True])
def test_rltt_path_end_to_end_through_the_loss(train_exit_gate):
    """Fake model -> patched forward -> build_rltt_model_output -> rltt_loss; the gate learns only when told to."""
    model = _fake_model()
    install_rltt_forward(model)
    lengths, prompt = [7, 9], 4
    micro_batch = _packed_micro_batch(lengths)
    ids = micro_batch["input_ids"].values()
    responses = [ids[o + prompt : o + n] for o, n in zip(micro_batch["input_ids"].offsets()[:-1].tolist(), lengths)]
    nested = lambda rows: torch.nested.as_nested_tensor(rows, layout=torch.jagged)  # noqa: E731
    data = TensorDict(
        {
            "input_ids": micro_batch["input_ids"],
            "prompts": nested([ids[o : o + prompt] for o in micro_batch["input_ids"].offsets()[:-1].tolist()]),
            "responses": nested(responses),
            "response_mask": nested([torch.ones(len(r), dtype=torch.int64) for r in responses]),
            "advantages": nested([torch.full((len(r),), a) for r, a in zip(responses, (1.0, -1.0))]),
            "ref_log_prob": nested([torch.full((len(r),), -2.0) for r in responses]),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=sum(len(r) for r in responses), global_batch_size=2)

    labels = torch.roll(ids, shifts=-1).unsqueeze(0)
    out = model(input_ids=ids.unsqueeze(0), rltt_labels=labels, rltt_temperature=TEMPERATURE)
    model_output = build_rltt_model_output(out, lambda t: t, data["input_ids"].offsets(), scheme="exit", alpha=1.0,
                                           train_exit_gate=train_exit_gate)
    config = ActorConfig(strategy="fsdp", rollout_n=2, ppo_micro_batch_size_per_gpu=1, use_kl_loss=True,
                         kl_loss_coef=0.01, policy_loss=PolicyLossConfig(loss_mode="rltt_exit"))
    loss, metrics = rltt_loss(config, model_output, data, scheme="exit")
    loss.backward()
    assert torch.isfinite(loss)
    gate_grad = model.early_exit_gate.weight.grad
    if train_exit_gate:
        assert gate_grad is not None and gate_grad.abs().sum() > 0
    else:
        # The KL and the entropy use the final loop only, so nothing else reaches the gate.
        assert gate_grad is None
    assert model.lm_head.weight.grad.abs().sum() > 0
    assert "rltt/exit_pdf_t3" in metrics and "kl_loss" in metrics


# ---------------------------------------------------------------------------------------------
# OrbitLMRLTTEngine on the CPU, without FSDP
# ---------------------------------------------------------------------------------------------


def _cpu_engine(model, monkeypatch, bucket=None):
    """A ``OrbitLMRLTTEngine`` that runs verl's per-micro-batch code on the CPU with a plain module.

    ``__new__`` skips the FSDP and optimizer setup. Only the attributes that ``forward_step``,
    ``prepare_model_inputs`` and ``prepare_model_outputs`` read are filled in. verl picks the
    device with its own detection, which reports CUDA on a GPU node even with no visible GPU, so the
    device helpers of the engine module are pointed at the CPU.
    """
    import verl.workers.engine.fsdp.transformer_impl as impl

    from orbit_verl.rltt.engine import OrbitLMRLTTEngine

    monkeypatch.setattr(impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(impl, "get_device_name", lambda: "cpu")
    # verl's stock log-prob path would use flash-attn's CUDA-only cross-entropy kernel.
    monkeypatch.setenv("VERL_DISABLE_FLASH_ATTN_CE", "1")
    monkeypatch.setattr(impl.FSDPEngineWithLMHead, "_build_module", lambda self: model)

    import verl.utils.torch_functional as verl_F

    engine = OrbitLMRLTTEngine.__new__(OrbitLMRLTTEngine)
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False, entropy_from_logits_with_chunking=False)
    engine.compute_entropy_from_logits = verl_F.entropy_from_logits
    engine.use_ulysses_sp = False
    engine.pad_to_length = bucket is not None
    engine.pad_to_length_bucket = bucket
    engine._autocast_dtype = torch.float32
    engine.get_data_parallel_group = lambda: None
    engine.module = engine._build_module()
    return engine


def _engine_micro_batch(prompt_len=4, response_lens=(5, 3, 6), vocab=37, seed=0, **non_tensor):
    g = torch.Generator().manual_seed(seed)
    nested = lambda rows: torch.nested.as_nested_tensor(rows, layout=torch.jagged)  # noqa: E731
    prompts = [torch.randint(0, vocab, (prompt_len,), generator=g) for _ in response_lens]
    responses = [torch.randint(0, vocab, (r,), generator=g) for r in response_lens]
    advantages = torch.randn(len(response_lens), generator=g)
    td = TensorDict(
        {
            "input_ids": nested([torch.cat([p, r]) for p, r in zip(prompts, responses)]),
            "position_ids": nested([torch.arange(prompt_len + r) for r in response_lens]),
            "prompts": nested(prompts),
            "responses": nested(responses),
            "response_mask": nested([torch.ones(r, dtype=torch.int64) for r in response_lens]),
            "advantages": nested([torch.full((r,), float(a)) for r, a in zip(response_lens, advantages)]),
            "old_log_probs": nested([-torch.rand(r, generator=g) for r in response_lens]),
            "ref_log_prob": nested([-torch.rand(r, generator=g) for r in response_lens]),
        },
        batch_size=[len(response_lens)],
    )
    tu.assign_non_tensor(
        td, temperature=TEMPERATURE, dp_size=1, batch_num_tokens=sum(response_lens),
        global_batch_size=len(response_lens), **non_tensor,
    )
    return td


def _rltt_config(loss_mode="rltt_exit"):
    return ActorConfig(strategy="fsdp", rollout_n=3, ppo_micro_batch_size_per_gpu=1, use_kl_loss=True,
                       kl_loss_coef=0.05, policy_loss=PolicyLossConfig(loss_mode=loss_mode))


@pytest.mark.parametrize("bucket", [None, 8])
def test_engine_forward_step_rltt_and_stock_paths(monkeypatch, bucket):
    monkeypatch.delenv("ORBIT_RLTT_TRAIN_EXIT_GATE", raising=False)
    model = _fake_model()
    engine = _cpu_engine(model, monkeypatch, bucket)
    assert engine._orbit_loop_count == 4
    assert getattr(type(model).forward, "_orbit_rltt_wrapper", False)
    micro_batch = _engine_micro_batch(calculate_entropy=True)

    # Training step with an RLTT loss mode: the RLTT forward and rltt_loss run.
    loss, output = engine.forward_step(micro_batch, functools.partial(ppo_loss, config=_rltt_config()), False)
    assert engine._rltt_scheme is None
    call = model.calls[-1]
    assert call["return_orbit_aux"] is True and call["exit_at_step"] is None
    assert call["extra"] == {}  # verl's own inputs only; the rltt_* arguments were consumed by the wrapper
    assert input_shape_is_packed(call)
    assert {"rltt/exit_pdf_t0", "rltt/expected_exit_step", "kl_loss", "actor/entropy_loss"} <= set(output["metrics"])
    assert {"log_probs", "rltt_log_probs", "loop_log_probs", "exit_pdf", "entropy"} == set(output["model_output"])
    model.zero_grad(set_to_none=True)
    loss.backward()
    # ORBIT_RLTT_TRAIN_EXIT_GATE is unset, so the gate gets no gradient (the next test sets it).
    assert model.early_exit_gate.weight.grad is None and model.lm_head.weight.grad.abs().sum() > 0

    # The same loss computed directly from the patched model, without the engine.
    ids = micro_batch["input_ids"].values()
    with torch.no_grad():
        out = model(input_ids=ids.unsqueeze(0), rltt_labels=torch.roll(ids, -1).unsqueeze(0),
                    rltt_temperature=TEMPERATURE, rltt_need_entropy=True)
    direct_output = build_rltt_model_output(out, lambda t: t, micro_batch["input_ids"].offsets(), "exit", 1.0)
    direct_loss, _ = rltt_loss(_rltt_config(), direct_output, micro_batch, scheme="exit")
    assert abs(loss.item() - direct_loss.item()) < 1e-6
    rltt_final = output["model_output"]["log_probs"].values()

    # Forward-only log-prob pass (old log-probs, reference): stock path, pinned to the final loop.
    with torch.no_grad():
        _, infer = engine.forward_step(micro_batch, None, True)
    call = model.calls[-1]
    assert call["exit_at_step"] == 3 and call["return_orbit_aux"] is False and call["compute_logits"] is True
    assert set(infer["model_output"]) == {"log_probs", "entropy"}
    diff = _max_abs(infer["model_output"]["log_probs"].values(), rltt_final)
    entropy_diff = _max_abs(infer["model_output"]["entropy"].values(), output["model_output"]["entropy"].values())
    print(f"PARITY engine bucket={bucket} stock vs RLTT final-loop |dlogp|={diff:.2e} |dentropy|={entropy_diff:.2e}")
    assert diff < 1e-5 and entropy_diff < 1e-5

    # Training step with vanilla GRPO: stock loss, final loop pinned.
    vanilla_loss, vanilla = engine.forward_step(micro_batch, functools.partial(ppo_loss, config=_rltt_config("vanilla")),
                                                False)
    assert model.calls[-1]["exit_at_step"] == 3 and model.calls[-1]["return_orbit_aux"] is False
    assert "actor/pg_clipfrac" in vanilla["metrics"] and torch.isfinite(vanilla_loss)


@pytest.mark.parametrize("value,trains", [(None, False), ("false", False), ("true", True)])
def test_engine_trains_the_exit_gate_only_when_the_env_says_so(monkeypatch, value, trains):
    """The launcher passes orbit.rltt.train_exit_gate to the engine as ORBIT_RLTT_TRAIN_EXIT_GATE."""
    if value is None:
        monkeypatch.delenv("ORBIT_RLTT_TRAIN_EXIT_GATE", raising=False)
    else:
        monkeypatch.setenv("ORBIT_RLTT_TRAIN_EXIT_GATE", value)
    model = _fake_model()
    engine = _cpu_engine(model, monkeypatch)
    micro_batch = _engine_micro_batch()
    loss, output = engine.forward_step(micro_batch, functools.partial(ppo_loss, config=_rltt_config()), False)
    model.zero_grad(set_to_none=True)
    loss.backward()
    gate_grad = model.early_exit_gate.weight.grad
    assert (gate_grad is not None and gate_grad.abs().sum() > 0) == trains
    # The gate's metrics are logged either way.
    assert {"rltt/exit_pdf_t0", "rltt/expected_exit_step"} <= set(output["metrics"])


def input_shape_is_packed(call) -> bool:
    # verl's remove-padding path hands the model one packed row: position_ids of shape [1, tokens].
    return call["position_ids"].dim() == 2 and call["position_ids"].shape[0] == 1


def test_engine_resets_the_rltt_flag_after_a_failure(monkeypatch):
    model = _fake_model()
    engine = _cpu_engine(model, monkeypatch)
    micro_batch = _engine_micro_batch(use_remove_padding=False)
    with pytest.raises(RuntimeError, match="use_remove_padding"):
        engine.forward_step(micro_batch, functools.partial(ppo_loss, config=_rltt_config()), False)
    assert engine._rltt_scheme is None
    with pytest.raises(RuntimeError, match="compute_loss=False"):
        engine.forward_step(_engine_micro_batch(), functools.partial(ppo_loss, config=_rltt_config()), True)
    assert engine._rltt_scheme is None


def test_engine_refuses_rltt_for_a_non_ouro_model(monkeypatch):
    model = _fake_model()
    model.config.model_type = "qwen2"
    engine = _cpu_engine(model, monkeypatch)
    engine.model_config = SimpleNamespace(hf_config=model.config)
    assert engine._orbit_loop_count is None
    with pytest.raises(RuntimeError, match="Ouro"):
        engine.forward_step(_engine_micro_batch(), functools.partial(ppo_loss, config=_rltt_config()), False)
