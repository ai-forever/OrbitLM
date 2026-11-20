"""The launcher of OrbitLM training on verl: ``scripts/verl/launch.py --config NAME --export PATH``.

``python -m orbit_verl.launch`` runs the same launcher when ``<repo>/src`` is on the path.

Modules:
- ``cli``: the command line and the order of the launch steps;
- ``config``: hydra composition of ``configs/verl/<experiment>.yaml``;
- ``env``: node topology, removal of the distributed launcher's variables, node env and job env;
- ``prepare``: pre-flight checks and building the prepared model dir and dataset;
- ``cluster``: forming one Ray cluster from the nodes of a multi-node job;
- ``train``: verl's own ``run_ppo`` on the composed config, with a task runner whose trainer runs
  the exit drift alarm.

This file imports nothing, so ``import orbit_verl.launch.env`` stays cheap and works in a python
that has no verl, such as the trainer env that the inference package runs in.
"""
