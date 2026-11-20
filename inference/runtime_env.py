"""Environment fixes that some setups need before generation starts.

Every fix here is either detected (it applies only where the problem exists) or a default that a value
the user already set overrides. The scripts print every fix that applied, so nothing changes silently.
inference/README.md lists them.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, MutableMapping, Optional

from orbit_verl.launch.env import remove_launcher_env

# "1" hides flash_attn, "0" leaves it alone, and an unset variable means "check whether it imports".
# The script that starts vLLM writes its decision here, so that the engine process vLLM spawns reads the
# decision instead of importing flash_attn a second time.
HIDE_FLASH_ATTN_ENV = "ORBIT_HIDE_FLASH_ATTN"


def hide_flash_attn_if_broken(environ: Optional[MutableMapping[str, str]] = None,
                              modules: Optional[MutableMapping[str, Any]] = None,
                              import_module: Callable[[str], Any] = importlib.import_module,
                              find_spec: Callable[[str], Any] = importlib.util.find_spec) -> bool:
    """Hide a flash_attn package that is installed but fails to import. Returns True when it was hidden.

    vLLM takes its rotary embedding kernel from flash_attn whenever it finds the package. A flash_attn
    built against another torch is found but fails on import, and that failure stops the engine. A None
    entry in sys.modules makes Python treat the package as absent, so vLLM uses its own kernel.

    The hide must also happen inside the engine process. That process re-runs the main script only when
    vLLM starts it with the "spawn" method, so the method is set to spawn whenever flash_attn is hidden.
    """
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    decision = environ.get(HIDE_FLASH_ATTN_ENV)
    if decision is None:
        decision = "0"
        if find_spec("flash_attn") is not None:
            try:
                import_module("flash_attn")
            except Exception:
                decision = "1"
        environ[HIDE_FLASH_ATTN_ENV] = decision
    if decision != "1":
        return False
    modules["flash_attn"] = None
    environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return True


def nvcc_available(environ: Optional[MutableMapping[str, str]] = None) -> bool:
    """True when the CUDA compiler is on PATH or in $CUDA_HOME/bin."""
    environ = os.environ if environ is None else environ
    if shutil.which("nvcc"):
        return True
    cuda_home = environ.get("CUDA_HOME")
    return bool(cuda_home) and (Path(cuda_home) / "bin" / "nvcc").is_file()


def vllm_env_defaults(environ: MutableMapping[str, str], has_nvcc: bool) -> dict[str, str]:
    """The vLLM variables to set, each only when the user has not set it already."""
    wanted = {
        # DeepGEMM serves only layers with FP8 weights, and OrbitLM weights are bf16, so it is never
        # needed. Some environments ship a DeepGEMM build that fails to load and stops the engine.
        "VLLM_USE_DEEP_GEMM": "0",
        # With torch 2.10 and later, vLLM reloads its AOT-compiled graph on every run after the first
        # and then mis-measures the free GPU memory, which leaves a tiny KV cache. The regular compile
        # cache still works without AOT.
        "VLLM_USE_AOT_COMPILE": "0",
    }
    if not has_nvcc:
        # The FlashInfer sampler compiles a kernel with nvcc when the engine starts. Without nvcc,
        # vLLM's PyTorch sampler draws from the same distribution.
        wanted["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return {key: value for key, value in wanted.items() if key not in environ}


def prepare_process(for_vllm: bool, environ: Optional[MutableMapping[str, str]] = None) -> list[str]:
    """Apply the fixes this process needs before the engine starts, and describe each one that applied.

    A distributed launcher such as torchrun exports RANK, WORLD_SIZE, MASTER_ADDR and TORCHELASTIC_*
    variables to the processes it starts. Generation runs as one process that starts its own workers,
    and with these variables set, vLLM's torch.distributed setup waits for peers that never come. They
    are removed for both engines, because HF generation has no use for them either.
    """
    environ = os.environ if environ is None else environ
    notes = []
    removed = remove_launcher_env(environ)
    if removed:
        notes.append(f"removed the distributed launcher's variables: {', '.join(removed)}")
    if for_vllm:
        if environ.get(HIDE_FLASH_ATTN_ENV) == "1":
            # The variable is 1 both when the import check failed and when the user set it. Only the
            # variable is visible here, so the note names both causes.
            notes.append(f"flash_attn is hidden from vLLM ({HIDE_FLASH_ATTN_ENV}=1: either it failed to import "
                         "here or the variable was set before the run)")
        settings = vllm_env_defaults(environ, nvcc_available(environ))
        environ.update(settings)
        if settings:
            notes.append("set " + " ".join(f"{key}={value}" for key, value in sorted(settings.items())))
    return notes
