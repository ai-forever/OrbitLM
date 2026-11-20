"""CPU checks of the orbit_vllm plugin, which brings the Ouro model back to vLLM releases after 0.26.

The registration tests replace vllm with a stand-in module that records what register() asks of
the model registry, so they run without vLLM. The RoPE tests import the model file and therefore
need a real vLLM of version 0.26 or newer; they skip in environments without one.
"""
from __future__ import annotations

import sys
import tomllib
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import orbit_vllm  # noqa: E402


class FakeRegistry:
    def __init__(self, architectures):
        self.architectures = set(architectures)
        self.registered = {}

    def get_supported_archs(self):
        return set(self.architectures)

    def register_model(self, architecture, target):
        self.registered[architecture] = target


@pytest.fixture
def fake_vllm(monkeypatch):
    def install(architectures):
        registry = FakeRegistry(architectures)
        module = types.ModuleType("vllm")
        module.ModelRegistry = registry
        monkeypatch.setitem(sys.modules, "vllm", module)
        return registry

    return install


def test_registers_ouro_when_vllm_has_none(fake_vllm):
    registry = fake_vllm({"Qwen2ForCausalLM"})
    orbit_vllm.register()
    assert registry.registered == {"OuroForCausalLM": "orbit_vllm.ouro:OuroForCausalLM"}


def test_keeps_the_built_in_model_of_vllm_0_26_and_older(fake_vllm):
    registry = fake_vllm({"OuroForCausalLM", "Qwen2ForCausalLM"})
    orbit_vllm.register()
    assert registry.registered == {}


def test_pyproject_declares_the_plugin_entry_point():
    """vLLM finds plugins only through this entry point group, in every process it starts."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["entry-points"]["vllm.general_plugins"] == {"orbit_ouro": "orbit_vllm:register"}


@pytest.fixture
def ouro():
    # The model file is vLLM 0.26's Ouro model, and it does not import on older vLLM.
    # The evaluation environment has vLLM 0.22, so these tests skip there.
    pytest.importorskip("vllm", minversion="0.26")
    from orbit_vllm import ouro as module

    return module


def test_rope_parameters_take_theta_from_either_config_form(ouro):
    new_form = types.SimpleNamespace(rope_parameters={"rope_type": "default", "rope_theta": 4.0e6})
    old_form = types.SimpleNamespace(rope_theta=4.0e6, rope_scaling=None)
    assert ouro.ouro_rope_parameters(new_form) == {"rope_type": "default", "rope_theta": 4.0e6}
    assert ouro.ouro_rope_parameters(old_form) == {"rope_type": "default", "rope_theta": 4.0e6}


def test_rope_parameters_refuse_a_config_without_theta(ouro):
    """A silently defaulted RoPE base still runs, so a missing base has to stop the load."""
    with pytest.raises(ValueError, match="rope_theta"):
        ouro.ouro_rope_parameters(types.SimpleNamespace(rope_parameters={"rope_type": "default"}))
