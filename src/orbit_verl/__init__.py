"""OrbitLM reinforcement learning on top of verl (GRPO and RLTT).

The design, the file layout and the interfaces between the parts are described in
``docs/verl/DESIGN.md``. Usage is in ``docs/verl/README.md``.

This file imports nothing. Ray workers import ``orbit_verl.plugin`` through
``VERL_USE_EXTERNAL_MODULES`` while ``verl`` itself is still being imported, so any import here
would run inside verl's own import and could create a cycle.
"""
