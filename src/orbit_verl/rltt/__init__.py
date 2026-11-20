"""RLTT (loop-weighted policy gradient) for OrbitLM under verl.

The modules here are:

- ``forward``: a wrapper around the Ouro model's forward that returns per-loop token log-probs and
  the exit pdf, plus the loop-weight schemes;
- ``loss``: the ratio-free RLTT loss, the loss-mode names and the raising stubs for verl's registry;
- ``engine``: ``OrbitLMRLTTEngine``, the verl FSDP engine subclass that runs the RLTT forward and loss;
- ``drift``: the exit drift alarm, which warns when the exit gate moves its weight to early loops
  during training, and the trainer subclass that runs it after every step.

``docs/verl/DESIGN.md`` summarises the RLTT loss and its choices.

This file stays free of imports. ``orbit_verl.plugin`` imports ``engine`` while ``verl`` itself is
still being imported, and an import here would run inside that and could create a cycle.
"""
