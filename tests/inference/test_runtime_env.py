"""Environment fixes: each one applies only where it is needed and never overrides a value the user set."""
from __future__ import annotations

import pytest

from inference.runtime_env import HIDE_FLASH_ATTN_ENV, hide_flash_attn_if_broken, prepare_process, vllm_env_defaults


def importer(fails: bool):
    calls = []

    def import_module(name):
        calls.append(name)
        if fails:
            raise ImportError("undefined symbol: built for another torch")
        return object()

    return import_module, calls


def test_a_broken_flash_attn_is_hidden_and_the_engine_is_spawned():
    environ, modules = {}, {}
    import_module, calls = importer(fails=True)
    assert hide_flash_attn_if_broken(environ, modules, import_module, find_spec=lambda name: object())
    assert modules["flash_attn"] is None and calls == ["flash_attn"]
    assert environ[HIDE_FLASH_ATTN_ENV] == "1" and environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_the_engine_process_repeats_the_decision_without_importing():
    modules = {}
    import_module, calls = importer(fails=False)
    assert hide_flash_attn_if_broken({HIDE_FLASH_ATTN_ENV: "1"}, modules, import_module, find_spec=lambda name: object())
    assert modules == {"flash_attn": None} and calls == []


@pytest.mark.parametrize("installed, fails, preset", [(True, False, None), (False, False, None), (True, True, "0")])
def test_a_working_or_missing_flash_attn_is_left_alone(installed, fails, preset):
    environ = {} if preset is None else {HIDE_FLASH_ATTN_ENV: preset}
    modules = {}
    import_module, _ = importer(fails)
    find_spec = (lambda name: object()) if installed else (lambda name: None)
    assert not hide_flash_attn_if_broken(environ, modules, import_module, find_spec)
    assert modules == {} and environ[HIDE_FLASH_ATTN_ENV] == "0"
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in environ


def test_vllm_defaults():
    assert vllm_env_defaults({}, has_nvcc=True) == {"VLLM_USE_DEEP_GEMM": "0", "VLLM_USE_AOT_COMPILE": "0"}
    assert vllm_env_defaults({}, has_nvcc=False)["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert vllm_env_defaults({"VLLM_USE_AOT_COMPILE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "1"}, has_nvcc=False) == {
        "VLLM_USE_DEEP_GEMM": "0"}


def test_prepare_process_removes_the_launcher_variables():
    environ = {"RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "h", "MASTER_PORT": "1", "LOCAL_RANK": "0",
               "TORCHELASTIC_RUN_ID": "x", "PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0",
               "VLLM_USE_DEEP_GEMM": "1", "VLLM_USE_AOT_COMPILE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "1"}
    notes = prepare_process(for_vllm=True, environ=environ)
    assert environ == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0", "VLLM_USE_DEEP_GEMM": "1",
                       "VLLM_USE_AOT_COMPILE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "1"}
    assert len(notes) == 1 and "TORCHELASTIC_RUN_ID" in notes[0] and "MASTER_ADDR" in notes[0]
    environ = {"PATH": "/bin"}
    notes = prepare_process(for_vllm=False, environ=environ)
    assert environ == {"PATH": "/bin"} and notes == []


@pytest.mark.parametrize("preset", [None, "1"])
def test_the_hide_note_is_true_whether_the_check_or_the_user_set_the_variable(preset):
    # With preset None the import check below decides; with "1" the user has set the variable and no
    # import is tried. The note must not claim an import failure that was never checked.
    environ = {} if preset is None else {HIDE_FLASH_ATTN_ENV: preset}
    import_module, calls = importer(fails=True)
    assert hide_flash_attn_if_broken(environ, {}, import_module, find_spec=lambda name: object())
    assert calls == (["flash_attn"] if preset is None else [])
    notes = prepare_process(for_vllm=True, environ=environ)
    [note] = [note for note in notes if "flash_attn" in note]
    assert "hidden from vLLM" in note and "failed to import" in note and "set before the run" in note
