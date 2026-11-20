#!/usr/bin/env bash
# Runs the OrbitLM trainer on the GPUs of this machine.
#
# Usage:
#   scripts/train/run_local.sh <config.yaml> [number of GPUs] [trainer options...]
#
# The number of GPUs defaults to every GPU that nvidia-smi lists. Options after it go to the
# trainer and override the config, for example --max-steps 10 or --save-dir /tmp/runs.
# The config's ${VAR} paths need their environment variables (src/orbit/README.md).
# PYTHON picks the interpreter (default: python); it must have the packages of
# requirements/train.txt.
set -euo pipefail

if [ $# -lt 1 ]; then
  sed -n '2,11p' "$0"
  exit 2
fi
CONFIG=$1
shift
if [ $# -gt 0 ] && [[ $1 =~ ^[0-9]+$ ]]; then
  NGPU=$1
  shift
else
  NGPU=$(nvidia-smi --list-gpus | wc -l)
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node "$NGPU" \
  -m orbit.train --config "$CONFIG" "$@"
