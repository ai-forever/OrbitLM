"""Generation with OrbitLM checkpoints: vLLM (recommended) and HF transformers (reference).

See README.md in this directory for the commands and the measured settings.

Importing the package puts two directories of this checkout at the front of sys.path. The SFT
evaluation renders its prompts in eval/in_training/itr_render.py, and the stop-thinking sentences of
budget forcing live in src/orbit_verl. Putting this checkout first makes sure that these files are
used even when another copy of orbit_verl is installed in the environment.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PATHS = (REPO_ROOT / "src", REPO_ROOT / "eval" / "in_training")

for _path in reversed(REPO_PATHS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
