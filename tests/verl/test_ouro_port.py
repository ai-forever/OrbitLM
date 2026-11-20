"""CPU tests of the transformers-5 port of the OrbitLM model code (orbit_verl.model.modelling_ouro).

A tiny random model checks the properties that verl and the RLTT engine rely on: packed batches
give the same logits as padded ones, cached generation equals uncached, gradient checkpointing
does not change gradients, save/load round-trips, ``exit_at_step=T-1`` is the threshold-free
final loop, ``return_orbit_aux`` returns one normed hidden state and one gate logit per loop,
and the frozen embedding row comes from ``embedding_padding_idx`` and never from ``pad_token_id``.

Run with the verl python:
    PYTHONPATH=<repo>/src $PY -m pytest -q -p no:cacheprovider tests/verl/test_ouro_port.py
"""
from orbit_verl import common

# The cache variables must be set before transformers is imported, because transformers reads
# its module-cache location once, at import time.
common.apply_cache_env()

import ast  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from orbit_verl.model import MODEL_CODE_DIR  # noqa: E402
from orbit_verl.model import configuration_ouro, modelling_ouro  # noqa: E402

VOCAB = 128
LOOPS = 3
ATTN = ["eager", "sdpa"]
TF5 = int(transformers.__version__.split(".")[0]) >= 5


def make_config(attn: str, **overrides) -> configuration_ouro.OuroConfig:
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, max_position_embeddings=256,
        rope_theta=40000.0, rope_scaling=None, total_ut_steps=LOOPS, early_exit_threshold=None,
        use_exit_gate=True, bos_token_id=0, eos_token_id=2, pad_token_id=0, tie_word_embeddings=False,
    )
    kwargs.update(overrides)
    config = configuration_ouro.OuroConfig(**kwargs)
    config._attn_implementation = attn
    return config


def randomize(model: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """Give every parameter seeded random values that are large enough to spread the logits.

    HF's own init draws weights with std 0.02, which leaves the logits of a tiny model so close
    together that a greedy argmax can flip on rounding noise. That would make the generation test
    flaky, so the weights are redrawn with std 0.1, and norm weights stay near 1.
    Much larger weights are not better: the logits then grow so much that fp32 rounding alone
    exceeds the 1e-5 tolerance of the packed-versus-padded comparison.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn(param.shape, generator=generator) * 0.1
            param.copy_(1.0 + noise if "norm" in name else noise)
    return model


def build(attn: str, seed: int = 0, **overrides) -> modelling_ouro.OuroForCausalLM:
    return randomize(modelling_ouro.OuroForCausalLM(make_config(attn, **overrides)), seed).eval()


def token_ids(length: int, seed: int) -> torch.Tensor:
    # Ids start at 3 so the pad (0) and eos (2) tokens never appear inside a sequence.
    return torch.randint(3, VOCAB, (length,), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("attn", ATTN)
def test_padded_and_packed_logits_match_separate_sequences(attn):
    model = build(attn)
    a, b = token_ids(7, 1), token_ids(10, 2)
    last = LOOPS - 1
    with torch.no_grad():
        solo_a = model(input_ids=a[None], use_cache=False, exit_at_step=last).logits[0]
        solo_b = model(input_ids=b[None], use_cache=False, exit_at_step=last).logits[0]

        # Left padding (generation layout) and right padding (training layout) with a mask.
        for left in (True, False):
            batch = torch.zeros(2, 10, dtype=torch.long)
            mask = torch.zeros(2, 10, dtype=torch.long)
            span = slice(3, 10) if left else slice(0, 7)
            batch[0, span], mask[0, span] = a, 1
            batch[1], mask[1] = b, 1
            logits = model(input_ids=batch, attention_mask=mask, use_cache=False, exit_at_step=last).logits
            torch.testing.assert_close(logits[0, span], solo_a, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(logits[1], solo_b, rtol=1e-5, atol=1e-5)

        # One packed row without a mask and with restarting positions, as verl feeds the actor.
        packed = torch.cat([a, b])[None]
        positions = torch.cat([torch.arange(7), torch.arange(10)])[None]
        logits = model(input_ids=packed, position_ids=positions, attention_mask=None, use_cache=False,
                       exit_at_step=last).logits[0]
    torch.testing.assert_close(logits[:7], solo_a, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(logits[7:], solo_b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("attn", ATTN)
def test_cached_greedy_generation_matches_uncached(attn):
    model = build(attn)
    batch = torch.zeros(2, 10, dtype=torch.long)
    mask = torch.zeros(2, 10, dtype=torch.long)
    batch[0, 3:], mask[0, 3:] = token_ids(7, 1), 1
    batch[1], mask[1] = token_ids(10, 2), 1
    steps = 6
    with torch.no_grad():
        cached = model.generate(input_ids=batch, attention_mask=mask, max_new_tokens=steps, do_sample=False,
                                use_cache=True, pad_token_id=0, eos_token_id=None)[:, 10:]
        # The same greedy continuation, recomputed from scratch at every step without a cache.
        current, current_mask, manual = batch.clone(), mask.clone(), []
        for _ in range(steps):
            logits = model(input_ids=current, attention_mask=current_mask, use_cache=False).logits[:, -1]
            nxt = logits.argmax(-1, keepdim=True)
            manual.append(nxt)
            current = torch.cat([current, nxt], 1)
            current_mask = torch.cat([current_mask, torch.ones_like(nxt)], 1)
    assert torch.equal(cached, torch.cat(manual, 1))


@pytest.mark.parametrize("attn", ATTN)
def test_gradient_checkpointing_keeps_gradients(attn):
    x = token_ids(24, 3).view(2, 12)
    labels = token_ids(24, 4).view(2, 12)
    grads = []
    for checkpointing in (False, True):
        model = build(attn).train()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # The loss touches the final logits, every loop's hidden state and every gate logit, so
        # the per-loop path that the RLTT engine trains is covered as well.
        out = model(input_ids=x, use_cache=False, return_orbit_aux=True)
        loss = torch.nn.functional.cross_entropy(out.logits.reshape(-1, VOCAB), labels.reshape(-1))
        for hidden, gate in zip(out.loop_hidden_states, out.loop_gate_logits):
            loss = loss + model.lm_head(hidden).log_softmax(-1).mean() + gate.mean()
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    assert "model.early_exit_gate.weight" in grads[0]
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name], rtol=1e-5, atol=1e-6, msg=name)


@pytest.mark.parametrize("attn", ATTN)
def test_save_pretrained_round_trip(attn, tmp_path):
    model = build(attn, embedding_padding_idx=1)
    model.save_pretrained(tmp_path)
    # A verl checkpoint re-exported with save_pretrained must keep the frozen row.
    assert json.loads((tmp_path / "config.json").read_text())["embedding_padding_idx"] == 1
    again = modelling_ouro.OuroForCausalLM.from_pretrained(tmp_path, attn_implementation=attn).eval()
    assert again.config.embedding_padding_idx == 1
    assert again.model.embed_tokens.padding_idx == 1
    ids = token_ids(9, 5)[None]
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).logits
        actual = again(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # tie_word_embeddings is false, so the dict form of _tied_weights_keys (P6) must not tie them.
    assert again.lm_head.weight.data_ptr() != again.model.embed_tokens.weight.data_ptr()
    torch.testing.assert_close(again.lm_head.weight, model.lm_head.weight, rtol=0, atol=0)
    assert again.config.rope_theta == 40000.0


@pytest.mark.parametrize("attn", ATTN)
def test_exit_at_last_step_equals_threshold_free_forward(attn):
    plain = build(attn)
    with_threshold = build(attn, early_exit_threshold=1.0)
    ids = token_ids(11, 6)[None]
    with torch.no_grad():
        reference = plain(input_ids=ids, use_cache=False).logits
        fixed_last = plain(input_ids=ids, use_cache=False, exit_at_step=LOOPS - 1).logits
        # An export config still carrying early_exit_threshold=1.0 must give the same final-loop
        # logits once exit_at_step pins the last loop, which is what the RLTT engine passes.
        pinned = with_threshold(input_ids=ids, use_cache=False, exit_at_step=LOOPS - 1).logits
    torch.testing.assert_close(fixed_last, reference, rtol=0, atol=0)
    torch.testing.assert_close(pinned, reference, rtol=0, atol=0)


@pytest.mark.parametrize("attn", ATTN)
def test_orbit_aux_returns_normed_hidden_states_and_gates_per_loop(attn):
    model = build(attn)
    batch = torch.stack([token_ids(8, 7), token_ids(8, 8)])
    mask = torch.ones_like(batch)
    with torch.no_grad():
        out = model(input_ids=batch, attention_mask=mask, use_cache=False, return_orbit_aux=True,
                    compute_logits=False)
        assert out.logits is None
        assert len(out.loop_hidden_states) == LOOPS and len(out.loop_gate_logits) == LOOPS
        for hidden, gate in zip(out.loop_hidden_states, out.loop_gate_logits):
            assert hidden.shape == (2, 8, 64)
            assert gate.shape == (2, 8, 1)
        # Each loop's hidden state already went through the final norm: projecting it with
        # lm_head gives exactly the logits of exiting at that loop.
        for step, hidden in enumerate(out.loop_hidden_states):
            exit_logits = model(input_ids=batch, attention_mask=mask, use_cache=False, exit_at_step=step).logits
            torch.testing.assert_close(model.lm_head(hidden), exit_logits, rtol=1e-5, atol=1e-5)


def test_rope_is_the_default_formula_and_config_is_not_modified():
    config = make_config("sdpa")
    before = copy.deepcopy(config.to_dict())
    model = modelling_ouro.OuroForCausalLM(config)
    rotary = model.model.rotary_emb
    expected = 1.0 / (40000.0 ** (torch.arange(0, 16, 2).float() / 16))
    assert rotary.rope_type == "default"
    assert rotary.attention_scaling == 1.0
    torch.testing.assert_close(rotary.inv_freq, expected, rtol=0, atol=0)
    after = config.to_dict()
    for key in ("rope_scaling", "rope_parameters", "rope_theta"):
        assert after.get(key) == before.get(key), key


def test_embedding_padding_row_comes_from_embedding_padding_idx_only():
    """verl overwrites pad_token_id with the tokenizer's pad id, so it must never pick the frozen row."""
    # pad_token_id names a token that the model must keep training: the Qwen3 eos is an example.
    unfrozen = build("sdpa", pad_token_id=5)
    assert unfrozen.config.embedding_padding_idx is None
    assert unfrozen.model.embed_tokens.padding_idx is None
    frozen = build("sdpa", pad_token_id=5, embedding_padding_idx=1)
    assert frozen.model.embed_tokens.padding_idx == 1

    ids = torch.tensor([[1, 5, 7, 1, 9, 5]])
    for model, row_one_frozen in ((unfrozen, False), (frozen, True)):
        model.train()
        logits = model(input_ids=ids, use_cache=False).logits
        logits.float().log_softmax(-1).mean().backward()
        grad = model.model.embed_tokens.weight.grad
        # The pad row (5) is a used token in both models, so it always gets a gradient.
        assert grad[5].abs().sum() > 0
        assert grad[7].abs().sum() > 0
        if row_one_frozen:
            assert torch.count_nonzero(grad[1]) == 0
        else:
            assert grad[1].abs().sum() > 0


def test_config_without_embedding_padding_idx_freezes_nothing():
    """Configs written before the field existed load with no frozen row.

    That is why prepare_model's check refuses such model dirs instead of loading them.
    """
    legacy = make_config("sdpa").to_dict()
    legacy.pop("embedding_padding_idx", None)
    assert configuration_ouro.OuroConfig.from_dict(legacy).embedding_padding_idx is None


@pytest.mark.parametrize("bad", [VOCAB, -1, True, 2.0, "2"])
def test_config_rejects_bad_embedding_padding_idx(bad):
    with pytest.raises(ValueError, match="embedding_padding_idx"):
        make_config("sdpa", embedding_padding_idx=bad)


def test_fa3_kernel_attention_needs_kernels_package():
    if modelling_ouro.get_kernel is not None:
        pytest.skip("the kernels package is installed in this env")
    model = build("eager")
    q = torch.zeros(1, 4, 3, 16)
    with pytest.raises(ImportError, match="kernels"):
        modelling_ouro.kernels_flash_attention_3_forward(model.model.layers[0].self_attn, q, q, q, None, scaling=1.0)


def test_port_is_self_contained_remote_code():
    """The port is copied into HF's and vLLM's module cache, where only its sibling file exists."""
    tree = ast.parse((MODEL_CODE_DIR / "modelling_ouro.py").read_text(encoding="utf-8"))
    relative, absolute = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            (relative if node.level else absolute).add(node.module)
        elif isinstance(node, ast.Import):
            absolute.update(alias.name for alias in node.names)
    assert relative == {"configuration_ouro"}
    roots = {name.split(".")[0] for name in absolute}
    assert not roots & {"verl", "orbit_verl", "orbit", "experiment_config"}
    assert "@auto_docstring\n    def forward" not in (MODEL_CODE_DIR / "modelling_ouro.py").read_text()
    # P7: the model code never reads pad_token_id, because verl rewrites it from the tokenizer.
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "pad_token_id"]


def test_mask_keywords_follow_the_installed_transformers():
    assert modelling_ouro._MASK_FN_TAKES_INPUTS_EMBEDS is TF5
