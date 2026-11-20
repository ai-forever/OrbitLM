"""generate_hf.HFGenerate on a tiny random OrbitLM model on CPU.

It needs torch and a transformers version that the model code supports (the trainer environment), and
it is skipped elsewhere. It checks that exit_at_step reaches the model through generate, and that the
two-call budget forcing works with real generate output.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from inference import REPO_ROOT  # noqa: E402
from inference.budget import Budget, generate_responses  # noqa: E402
from inference.generate_hf import HFGenerate, transformers_problem  # noqa: E402
from orbit_verl.data.response_info import STOP_THINKING_TEXTS  # noqa: E402

if transformers_problem(transformers.__version__):
    pytest.skip(transformers_problem(transformers.__version__), allow_module_level=True)

VOCAB, END, CLOSE = 64, 63, 62
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": None}


def load_model_code():
    """OuroConfig and OuroForCausalLM of src/orbit, loaded without the trainer package.

    An HF export bundles these files as standalone model code. The orbit package's __init__ imports
    the whole trainer, which this test does not need, so the files are loaded under a package of their
    own whose __init__ never runs.
    """
    name = "_orbit_model_code"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(REPO_ROOT / "src" / "orbit")]
        sys.modules[name] = package
    configuration = importlib.import_module(f"{name}.configuration_ouro")
    modelling = importlib.import_module(f"{name}.modelling_ouro")
    return configuration.OuroConfig, modelling.OuroForCausalLM


@pytest.fixture(scope="module")
def model():
    OuroConfig, OuroForCausalLM = load_model_code()
    config = OuroConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                        total_ut_steps=4, early_exit_threshold=None, attn_implementation="eager")
    torch.manual_seed(0)
    model = OuroForCausalLM(config).eval()
    # A zero output row gives a logit of exactly 0, which greedy decoding never picks while another token
    # has a positive logit. So the model never ends its turn or closes its block by itself, and the test
    # knows that every first call runs into the budget.
    with torch.no_grad():
        model.lm_head.weight[END].zero_()
        model.lm_head.weight[CLOSE].zero_()
    return model


def make_generate(model, loops):
    return HFGenerate(model, loops=loops, end_ids=(END,), pad_id=0, sampling=GREEDY, device="cpu")


@pytest.mark.parametrize("loops", [1, 2, 4])
def test_the_first_token_comes_from_the_chosen_loop(model, loops):
    prompt = [1, 2, 3, 4, 5]
    [completion] = make_generate(model, loops)([prompt], [1])
    with torch.no_grad():
        logits = model(torch.tensor([prompt]), exit_at_step=loops - 1, use_cache=False).logits[0, -1]
    assert completion.token_ids == (int(logits.argmax()),)
    assert completion.finish_reason == "length"


def test_budget_forcing_with_hf_generate(model):
    budget = Budget(think_tokens=3, answer_tokens=2, close_id=CLOSE,
                    stop_ids_by_text={STOP_THINKING_TEXTS["en"]: (10, 11, CLOSE), STOP_THINKING_TEXTS["ru"]: (12, 13, CLOSE)})
    generate = make_generate(model, 4)
    responses = generate_responses([[1, 2, 3], [4, 5]], ["en", "ru"], generate, max_new_tokens=None,
                                   max_model_len=256, budget=budget)
    for response, inserted in zip(responses, [(10, 11, CLOSE), (12, 13, CLOSE)]):
        assert len(response.first.token_ids) == 3 and response.first.finish_reason == "length"
        assert response.forced_close and response.inserted_ids == inserted
        assert len(response.second.token_ids) == 2 and response.finish_reason == "length"


def test_sampling_runs(model):
    torch.manual_seed(1)
    generate = HFGenerate(model, loops=2, end_ids=(END,), pad_id=0,
                          sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20}, device="cpu")
    [completion] = generate([[1, 2, 3]], [4])
    assert len(completion.token_ids) <= 4
