"""CPU test for ``orbit_verl.plugin`` loaded the way verl loads it.

The checks run in a fresh Python process with ``VERL_USE_EXTERNAL_MODULES=orbit_verl.plugin``,
because the plugin is meant to be imported by ``import verl`` itself, and because the engine
registry and the loss registry are process-wide state that the other tests must not see changed.
One subprocess does all checks: ``import verl`` can take tens of seconds on a shared filesystem.

``VERL_ENGINE_DEVICE=cuda`` is set because the engine registry is keyed by device, and the test
must resolve the CUDA entries on a machine without a usable GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
MARKER = "PLUGIN_CHECK_RESULT "

CHECK_SCRIPT = r"""
import contextlib
import importlib
import io
import json
import os
import sys

import verl  # loads orbit_verl.plugin through VERL_USE_EXTERNAL_MODULES

result = {"loaded_by_verl": "orbit_verl.plugin" in sys.modules}

import torch
import torch.distributed

result["cuda_initialized"] = torch.cuda.is_initialized()
result["dist_initialized"] = torch.distributed.is_available() and torch.distributed.is_initialized()

from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, get_policy_loss_fn
from verl.workers.engine.base import EngineRegistry

import orbit_verl.plugin as plugin
import orbit_verl.rltt.engine as engine_module
from orbit_verl.rltt.loss import RLTT_LOSS_MODES, register_stub_policy_losses


def resolve():
    table = {}
    for vendor in (None, "orbit"):
        if vendor is None:
            os.environ.pop("VERL_ENGINE_VENDOR", None)
        else:
            os.environ["VERL_ENGINE_VENDOR"] = vendor
        for model_type in ("language_model", "value_model"):
            for backend in ("fsdp", "fsdp2"):
                cls = EngineRegistry.get_engine_cls(model_type, backend)
                table[f"{model_type}/{backend}/{vendor}"] = f"{cls.__module__}.{cls.__name__}"
    os.environ.pop("VERL_ENGINE_VENDOR", None)
    return table


result["registry"] = resolve()

stubs = {}
for name in RLTT_LOSS_MODES:
    try:
        get_policy_loss_fn(name)(None, None, None, None)
        stubs[name] = "did not raise"
    except RuntimeError as exc:
        stubs[name] = str(exc)
result["stubs"] = stubs
before = {name: id(POLICY_LOSS_REGISTRY[name]) for name in RLTT_LOSS_MODES}

# Importing the plugin again, reloading it and re-running the stub registration must all be harmless.
# The reloads run the module code again, so they also show whether that code prints anything.
printed = io.StringIO()
with contextlib.redirect_stdout(printed):
    importlib.import_module("orbit_verl.plugin")
    importlib.reload(plugin)
    register_stub_policy_losses()
result["stubs_unchanged"] = before == {name: id(POLICY_LOSS_REGISTRY[name]) for name in RLTT_LOSS_MODES}

# Re-executing the engine module replaces its own registry entry instead of tripping verl's assert.
with contextlib.redirect_stdout(printed):
    reloaded = importlib.reload(engine_module)
result["printed"] = printed.getvalue()
os.environ["VERL_ENGINE_VENDOR"] = "orbit"
result["reload_resolves_new_class"] = EngineRegistry.get_engine_cls("language_model", "fsdp") is reloaded.OrbitLMRLTTEngine
os.environ.pop("VERL_ENGINE_VENDOR", None)
result["registry_after_reload"] = resolve()
result["cuda_initialized_at_end"] = torch.cuda.is_initialized()

print("PLUGIN_CHECK_RESULT " + json.dumps(result, sort_keys=True))
"""


def _run_checks() -> dict:
    env = dict(os.environ)
    env.pop("VERL_ENGINE_VENDOR", None)
    env.update(
        {
            "VERL_USE_EXTERNAL_MODULES": "orbit_verl.plugin",
            "VERL_ENGINE_DEVICE": "cuda",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", CHECK_SCRIPT], env=env, capture_output=True, text=True, timeout=900
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith(MARKER)]
    assert proc.returncode == 0 and lines, f"exit {proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-5000:]}"
    return json.loads(lines[-1][len(MARKER):])


def test_plugin_registers_engine_and_stubs_in_a_verl_process():
    result = _run_checks()

    assert result["loaded_by_verl"]
    assert not result["cuda_initialized"] and not result["dist_initialized"]
    assert not result["cuda_initialized_at_end"]

    ours = "orbit_verl.rltt.engine.OrbitLMRLTTEngine"
    stock_lm = "verl.workers.engine.fsdp.transformer_impl.FSDPEngineWithLMHead"
    stock_value = "verl.workers.engine.fsdp.transformer_impl.FSDPEngineWithValueHead"
    for registry in (result["registry"], result["registry_after_reload"]):
        for backend in ("fsdp", "fsdp2"):
            assert registry[f"language_model/{backend}/orbit"] == ours
            assert registry[f"language_model/{backend}/None"] == stock_lm
            assert registry[f"value_model/{backend}/orbit"] == stock_value
            assert registry[f"value_model/{backend}/None"] == stock_value

    assert set(result["stubs"]) == {"rltt_exit", "rltt_uniform", "rltt_progressive", "rltt_last"}
    for name, message in result["stubs"].items():
        assert message == (
            f"loss_mode {name} needs OrbitLMRLTTEngine: set VERL_ENGINE_VENDOR=orbit and import orbit_verl.plugin"
        )
    assert result["stubs_unchanged"]
    assert result["reload_resolves_new_class"]
    assert result["printed"] == ""
