#!/usr/bin/env python3
"""Run OrbitLM RL training on verl from an HF export, without setting PYTHONPATH first.

    $PY scripts/verl/launch.py --config rl_sft_mix --export /path/to/hf/export <overrides>
    $PY scripts/verl/launch.py --config rl_sft_mix --export /path/to/hf/export --dry-run <overrides>
    $PY scripts/verl/launch.py --help

``$PY`` is the python of the verl env (``requirements/verl``).
The runs root comes from ``--runs-root`` or ``$ORBIT_VERL_RUNS``; the launcher refuses to start
without one. Everything else is ``orbit_verl.launch.cli``; this file only finds the package.
"""
import sys
from pathlib import Path

# scripts/verl/launch.py -> parents[2] is the repo root, whatever the current directory is.
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orbit_verl.launch.cli import main  # noqa: E402 - the path above has to be set first

if __name__ == "__main__":
    sys.exit(main())
