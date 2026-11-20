"""vLLM plugin that registers the Ouro architecture used by OrbitLM.

vLLM calls register() in every process it starts, through the "vllm.general_plugins"
entry point in pyproject.toml. Installing this package is therefore enough: no code has
to run before vllm.LLM or `vllm serve`.
"""

ARCHITECTURE = "OuroForCausalLM"


def register() -> None:
    """Registers OuroForCausalLM unless the installed vLLM already ships it.

    vLLM 0.26 and older have their own Ouro model, and that one is kept there.
    """
    from vllm import ModelRegistry

    if ARCHITECTURE in ModelRegistry.get_supported_archs():
        return
    ModelRegistry.register_model(ARCHITECTURE, "orbit_vllm.ouro:OuroForCausalLM")
