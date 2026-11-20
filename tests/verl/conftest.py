"""Shared fixtures of the verl CPU tests.

Most tests build what they need in a temporary directory. The tests that need the real Qwen3
tokenizer, or the chat template of an SFT export, read an SFT chat export dir from the environment
variable ``ORBIT_TEST_SFT_EXPORT`` and skip when it is unset. Use the export that the recipes start
from, SFT v5 at step 7000. Any other SFT chat export of the release line works too when its bundled
code builds the embedding without ``padding_idx``, because one test checks that the padding detection
reads it as the release line:

    ORBIT_TEST_SFT_EXPORT=$ORBIT_EXPORTS/sft_v5_step7000 python -m pytest -q tests/verl

Several test modules set up the verl caches at import, and that needs a runs root. When
``ORBIT_VERL_RUNS`` is unset, the tests get a fresh temporary one, so they never write into a real
runs root by accident.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

SFT_EXPORT_ENV = "ORBIT_TEST_SFT_EXPORT"

# conftest.py is imported before the test modules, so the default is in place when they call
# common.apply_cache_env() at import.
# The folder is removed when the test process exits.
if not os.environ.get("ORBIT_VERL_RUNS", "").strip():
    os.environ["ORBIT_VERL_RUNS"] = tempfile.mkdtemp(prefix="orbit_verl_test_runs_")
    atexit.register(shutil.rmtree, os.environ["ORBIT_VERL_RUNS"], ignore_errors=True)


def sft_export_dir() -> Path | None:
    """The SFT export named by ``ORBIT_TEST_SFT_EXPORT``, or None when it is unset or not a directory."""
    value = os.environ.get(SFT_EXPORT_ENV, "").strip()
    if not value or not Path(value).is_dir():
        return None
    return Path(value)


@pytest.fixture(scope="session")
def sft_export() -> Path:
    """The SFT chat export dir; the test is skipped without one."""
    path = sft_export_dir()
    if path is None:
        pytest.skip(f"set {SFT_EXPORT_ENV} to an SFT chat export dir to run this test")
    return path


@pytest.fixture(scope="session")
def qwen3_tokenizer(sft_export: Path):
    """The Qwen3 tokenizer of the SFT export, loaded once per test session."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(sft_export), trust_remote_code=True)
