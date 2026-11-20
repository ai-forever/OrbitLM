#!/bin/bash
# Builds the Python environment for OrbitLM RL and distillation on verl (verl 0.9.1, vLLM 0.26,
# transformers 5.10.4, torch 2.11 + CUDA 13).
#
#   bash requirements/verl/build_env.sh ENV_DIR [FLASH_ATTN_WHEEL]
#
#   ENV_DIR           where the virtual env is created; it must not exist yet.
#   FLASH_ATTN_WHEEL  a flash-attn 2.8.3 wheel built for torch 2.11 + CUDA 13 + Python 3.12. No official wheel
#                     exists for that combination, so it has to be built once on a machine with nvcc
#                     (pip wheel flash-attn==2.8.3 --no-build-isolation). Without it the env is built
#                     without flash-attn, and the model's flash_attention_2 path will not load.
#
# Needs python3.12 on PATH and a GPU node for the final checks. Takes about 10 minutes and 11 GB.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: build_env.sh ENV_DIR [FLASH_ATTN_WHEEL]" >&2; exit 2; }
ENV_DIR=$1
WHEEL=${2:-}
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ ! -e "$ENV_DIR" ] || { echo "$ENV_DIR exists; choose a new dir" >&2; exit 2; }

python3.12 -m venv "$ENV_DIR"
PY=$ENV_DIR/bin/python
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -c "$D/constraints.txt" -r "$D/requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu130
if [ -n "$WHEEL" ]; then
  "$PY" -m pip install --no-deps "$WHEEL"
fi
# The repository itself, as an editable package, so that orbit_verl imports without PYTHONPATH.
"$PY" -m pip install --no-deps -e "$D/../.."

"$PY" -m pip check || true
"$PY" - <<'PYEOF'
import importlib
import torch
for name in ["torch", "vllm", "verl", "transformers", "ray", "math_verify", "datasets", "tensordict", "transfer_queue",
             "orbit_verl"]:
    module = importlib.import_module(name)
    print(f"{name:14s} {getattr(module, '__version__', 'ok')}")
# vLLM 0.26 is the last release that still ships the Ouro model, which vLLM uses for OrbitLM.
from vllm.model_executor.models import ouro  # noqa: F401
print("vllm ouro ok")
try:
    from flash_attn import flash_attn_func
except ImportError:
    print("flash_attn missing: install a wheel built for this torch and CUDA")
else:
    if torch.cuda.is_available():
        q = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
        print("flash_attn_func ok", tuple(flash_attn_func(q, q, q, causal=True).shape))
PYEOF
