"""Shared setup of the data_prep tests.

The build scripts read the three ORBIT_* roots when they are imported. The tests use only
synthetic data, so a root that the caller did not set points at a scratch path that no test
writes to.

The tokenizer fixture gives the tests the three release tokenizers. When ORBIT_TOKENIZERS
already holds them, they are used as they are. Otherwise prepare_tokenizers.py writes them into
a temporary directory from the Qwen3 tokenizers in the local Hugging Face cache, and the tests
that need them are skipped when the cache does not have those either.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOKENIZER_NAMES = ("tokenizer_qwen3_orbit", "tokenizer_qwen3_4b_qwen35_think", "tokenizer_qwen3_4b_qwen35_tools")

_CALLER_TOKENIZERS = os.environ.get("ORBIT_TOKENIZERS")
for _name in ("ORBIT_RAW", "ORBIT_DATA", "ORBIT_TOKENIZERS"):
    os.environ.setdefault(_name, os.path.join(tempfile.gettempdir(), "orbit-data-prep-tests-unused", _name.lower()))


def _complete(root):
    return all((Path(root) / name / "tokenizer_config.json").is_file() for name in TOKENIZER_NAMES)


@pytest.fixture(scope="session")
def tokenizer_root(tmp_path_factory):
    """Returns a directory that holds the three release tokenizer directories."""
    if _CALLER_TOKENIZERS and _complete(_CALLER_TOKENIZERS):
        return Path(_CALLER_TOKENIZERS)
    sys.path.insert(0, str(HERE / "tokenizer"))
    import prepare_tokenizers

    out = tmp_path_factory.mktemp("tokenizers")
    try:
        for name in TOKENIZER_NAMES:
            variant = prepare_tokenizers.VARIANTS[name]
            (out / name).mkdir()
            if variant.template is None:
                prepare_tokenizers.write_pretrain(variant, out / name, offline=True)
            else:
                prepare_tokenizers.write_sft(variant, out / name, offline=True)
            prepare_tokenizers.check_saved(out / name)
    except OSError as exc:
        pytest.skip(f"the Qwen3 base tokenizers are not in the local Hugging Face cache: {exc}")
    return out
