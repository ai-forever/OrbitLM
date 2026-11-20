"""CPU checks that the exported model code loads on transformers 4.56 and later, including 5.x.

Exports copy src/orbit/modelling_ouro.py and configuration_ouro.py, and HF runs them as remote
code. These tests pin the parts that differ between the major versions (the P1 to P6 comments in
the model code) and the rule that an export needs no other file of the orbit package.
"""
from __future__ import annotations

import ast
import copy
import subprocess
import sys
from pathlib import Path

import torch
import transformers

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from orbit import export as orbit_export  # noqa: E402
from orbit import modelling_ouro  # noqa: E402
from orbit.configuration_ouro import OuroConfig  # noqa: E402

MODEL_CODE = REPO_ROOT / "src" / "orbit" / "modelling_ouro.py"
TF5 = int(transformers.__version__.split(".")[0]) >= 5


def test_export_needs_only_the_two_model_files():
    """HF copies the export's code into its module cache, where only the sibling files exist."""
    assert orbit_export.REMOTE_CODE_FILES == ("configuration_ouro.py", "modelling_ouro.py")
    tree = ast.parse(MODEL_CODE.read_text(encoding="utf-8"))
    relative, absolute = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            (relative if node.level else absolute).add(node.module)
        elif isinstance(node, ast.Import):
            absolute.update(alias.name for alias in node.names)
    assert relative == {"configuration_ouro"}
    roots = {name.split(".")[0] for name in absolute}
    assert not roots & {"orbit", "orbit_verl", "experiment_config"}


def test_model_code_imports_without_the_kernels_package():
    """Transformers refuses remote code whose top-level imports are missing, so kernels stays optional.

    The import runs in a fresh interpreter in which `import kernels` fails, whether or not the
    package is installed here.
    """
    script = (
        "import sys; sys.modules['kernels'] = None; "
        f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); "
        "from orbit import modelling_ouro; "
        "print(modelling_ouro.get_kernel is None, modelling_ouro.get_local_kernel is None)"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["True", "True"]


def test_mask_keywords_follow_the_installed_transformers():
    assert modelling_ouro._MASK_FN_TAKES_INPUTS_EMBEDS is TF5


def test_default_rope_frequencies_and_untouched_config():
    """Transformers 5 has no "default" RoPE init function, so the model computes the frequencies.

    The result must equal the plain RoPE formula, and building the model must not rewrite
    rope_scaling, which the old fallback did.
    """
    config = OuroConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, head_dim=8, max_position_embeddings=64, rope_theta=4.0e6, rope_scaling=None,
        total_ut_steps=2, pad_token_id=0, eos_token_id=0,
    )
    scaling_before = copy.deepcopy(getattr(config, "rope_scaling", None))
    rotary = modelling_ouro.OuroRotaryEmbedding(config)
    expected = 1.0 / (4.0e6 ** (torch.arange(0, 8, 2, dtype=torch.float) / 8))
    assert rotary.rope_type == "default"
    assert torch.allclose(rotary.inv_freq, expected)
    assert getattr(config, "rope_scaling", None) == scaling_before
