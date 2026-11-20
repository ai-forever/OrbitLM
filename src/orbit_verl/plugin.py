"""verl plugin for OrbitLM: registers ``OrbitLMRLTTEngine`` and the RLTT loss-mode stubs.

verl imports this module in every one of its processes when ``VERL_USE_EXTERNAL_MODULES`` contains
``orbit_verl.plugin``. The import happens at the end of ``import verl``. The launcher sets that
variable together with ``VERL_ENGINE_VENDOR=orbit``, which makes verl choose our engine.

Rules for this module, because it runs inside ``import verl`` in Ray actors, vLLM servers and
reward workers alike:

- no CUDA and no torch.distributed calls at import; some of those processes own no GPU;
- no printing; at most one debug log line;
- import it under this one name only. ``EngineRegistry.register`` refuses a key that is already
  taken, so loading the engine module under a second name would fail. Importing this module twice
  under this name is harmless, because Python caches the module and the stub registration skips
  names that are already registered.
"""
from __future__ import annotations

import logging

from orbit_verl.rltt import engine as _engine  # noqa: F401  (importing it registers the engine)
from orbit_verl.rltt.loss import register_stub_policy_losses

register_stub_policy_losses()

logging.getLogger(__name__).debug("orbit_verl.plugin: registered OrbitLMRLTTEngine and the rltt_* loss stubs")
