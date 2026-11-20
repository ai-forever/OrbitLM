#!/bin/bash
# Runs the 16K on-policy distillation (configs/verl/distill_rl_sft_16k.yaml) on this node, which needs 8 GPUs:
# 6 GPUs for the student and 2 for the teacher.
#
#   bash scripts/verl/run_distill.sh [--arm main|opd] --export DIR --teacher DIR --data DIR [--runs-root DIR] \
#       [--python PY] [--model-name NAME] [--clearml-env FILE] [--no-prepare] \
#       [--print | --dry-run | --prepare-only] [-- OVERRIDE ...]
#
#   --arm main     (default) distillation from the teacher plus the GRPO task loss on the gold rows.
#   --arm opd      the same run with distillation.distillation_loss.use_task_rewards=false, so the gold rows get
#                  only the teacher's signal. Everything else is shared, so the two arms form an A/B.
#   --export       the HF chat export of the start checkpoint, $ORBIT_EXPORTS/sft_v5_step7000
#                  (SFT v5, step 7000).
#   --teacher      the absolute path of a local Qwen3-8B model dir.
#   --data         the dir that holds the prompt pool, the gold files and the validation files
#                  (docs/verl/README.md lists them).
#   --runs-root    the root of models, datasets, runs and caches. Default: $ORBIT_VERL_RUNS.
#   --python       the verl env's python. Default: $VERL_PYTHON.
#   --model-name   the prepared model dir's name under <runs root>/models. Default: sft_v5_step7000.
#   --clearml-env  a dotenv file with ClearML credentials. Without it, ClearML stays off.
#   --no-prepare   never build a model dir or dataset; the launch needs them prepared already.
#
#   no flag         runs the recipe on this node. The launcher builds a missing model dir or dataset first,
#                   and its output also goes to <runs root>/runs/<run name>/launch.log.
#   --print         prints the launcher's arguments and the overrides, one per line, and checks nothing.
#   --dry-run       composes and checks the run, prints the launcher's plan, and builds and starts nothing.
#   --prepare-only  builds the model dir and the dataset when they are missing, without a GPU, and starts
#                   nothing.
#
# Several nodes work as in scripts/verl/run_rl.sh: --prepare-only once, then the same command with
# --no-prepare once on every node, one process per node, with RANK set to the node rank, WORLD_SIZE to the
# number of nodes and MASTER_ADDR to the address of node 0 (torchrun --nproc_per_node 1 sets them so).
# Every node needs the runs root, the export and the teacher at the same paths. The prepared model dir links
# to the export's weights, and Ray may place the teacher on any node. A --no-prepare launch does not look for
# the files in --data. As in run_rl.sh, put --no-prepare before any "--".
#
# Overrides after "--" go to the launcher after the recipe's own, so they win.
#
# The recipe: 200 steps of 132 prompts x 4 answers up to 16K tokens, a thinking budget of 14,336 tokens,
# validation and a checkpoint every 10 steps. The gold rows are 6,454 math rows of the pass-rate keep and
# 3,300 code rows, next to the whole prompt pool. The math keep is the file that the RL arms read
# (math_keep.jsonl); scripts/verl/data/README.md says how each input is built.
set -euo pipefail

usage() {
  echo "usage: run_distill.sh [--arm main|opd] --export DIR --teacher DIR --data DIR [options]" \
       "[--print|--dry-run|--prepare-only] [-- OVERRIDE ...]" >&2
  exit 2
}
ARM=main MODE=run
EXPORT= TEACHER= DATA= CLEARML_ENV= NO_PREPARE=
RUNS=${ORBIT_VERL_RUNS:-}
PY=${VERL_PYTHON:-}
MODEL_NAME=sft_v5_step7000
EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    --arm) ARM=$2; shift 2 ;;
    --export) EXPORT=$2; shift 2 ;;
    --teacher) TEACHER=$2; shift 2 ;;
    --data) DATA=$2; shift 2 ;;
    --runs-root) RUNS=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --model-name) MODEL_NAME=$2; shift 2 ;;
    --clearml-env) CLEARML_ENV=$2; shift 2 ;;
    --no-prepare) NO_PREPARE=1; shift ;;
    --print|--dry-run|--prepare-only)
      [ $MODE = run ] || { echo "give only one of --print, --dry-run and --prepare-only" >&2; exit 2; }
      MODE=${1#--}; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
missing_args=()
required=("--export:$EXPORT" "--teacher:$TEACHER" "--data:$DATA" "--runs-root (or ORBIT_VERL_RUNS):$RUNS"
          "--python (or VERL_PYTHON):$PY")
for pair in "${required[@]}"; do
  [ -n "${pair#*:}" ] || missing_args+=("${pair%%:*}")
done
if [ ${#missing_args[@]} -gt 0 ]; then
  echo "run_distill.sh needs ${missing_args[*]}" >&2
  usage
fi
if [ -n "$NO_PREPARE" ] && [ "$MODE" = prepare-only ]; then
  echo "give only one of --no-prepare and --prepare-only" >&2
  exit 2
fi
case $ARM in
  main) RUN=distill_rl_16k; ARM_OVERRIDES=() ;;
  opd) RUN=distill_opd_16k; ARM_OVERRIDES=(distillation.distillation_loss.use_task_rewards=false) ;;
  *) echo "--arm must be main or opd, not $ARM" >&2; exit 2 ;;
esac
REPO=$(cd "$(dirname "$0")/../.." && pwd)

# The inputs, by the file names that docs/verl/README.md lists.
POOL=$DATA/prompts.jsonl
M500=$DATA/math500.jsonl
MATH_GOLD=$DATA/math_keep.jsonl
CODE_TRAIN=$DATA/code_train.jsonl
CODE_VAL=$DATA/code_val.jsonl
MGSM_RU=$DATA/mgsm_ru.jsonl

# The specs hold "#", which hydra accepts only inside quotes.
VAL="[\"$M500:orbit/math#think:sft_think\",\"$M500:orbit/math#no_think:sft_no_think\","
VAL+="\"$CODE_VAL:orbit/code#think:sft_think_code\",\"$MGSM_RU:orbit/math#ru:sft_think_ru\"]"
GOLD="[\"$MATH_GOLD:orbit/math:math:6454\",\"$CODE_TRAIN:orbit/code:code:3300\"]"

OVERRIDES=(
  distillation.teacher_models.teacher_model.model_path="$TEACHER"
  orbit.prepare.data.pool="$POOL" "orbit.prepare.data.val=$VAL" "orbit.prepare.data.gold=$GOLD"
  orbit.prepare.data.name=distill_rl_mix__qwen3 orbit.run_name="$RUN"
  ${ARM_OVERRIDES[@]+"${ARM_OVERRIDES[@]}"}
  trainer.total_training_steps=200 trainer.test_freq=10 trainer.save_freq=10
)
if [ -n "$CLEARML_ENV" ]; then
  OVERRIDES+=(orbit.clearml_env="$CLEARML_ENV" "trainer.logger=[console,file,clearml]")
fi
OVERRIDES+=(${EXTRA[@]+"${EXTRA[@]}"})

SHARED=(--config distill_rl_sft_16k --export "$EXPORT" --model-name "$MODEL_NAME" --runs-root "$RUNS")
LAUNCH=("${SHARED[@]}" --gpus-per-node 8)
[ -z "$NO_PREPARE" ] || LAUNCH+=(--no-prepare)
if [ "$MODE" = print ]; then
  printf '%s\n' "${LAUNCH[@]}" -- "${OVERRIDES[@]}"
  exit 0
fi

# A --no-prepare launch reads only the prepared dataset under the runs root, so it needs none of the data
# files. The teacher is always needed, because the run loads it.
missing=()
if [ -z "$NO_PREPARE" ]; then
  for file in "$POOL" "$M500" "$MATH_GOLD" "$CODE_TRAIN" "$CODE_VAL" "$MGSM_RU"; do
    [ -f "$file" ] || missing+=("$file")
  done
fi
[ -d "$TEACHER" ] || missing+=("$TEACHER")
if [ ${#missing[@]} -gt 0 ]; then
  echo "missing inputs: ${missing[*]}" >&2
  # A dry run builds nothing, so it may go on and check the rest.
  [ "$MODE" = dry-run ] || exit 2
fi

# The release runs and the two-node Ray check used the pure-python protobuf. A value set by the
# caller wins.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}

case $MODE in
  dry-run)
    # launch.py puts the repo's src on its path itself, and a PYTHONPATH from this shell could shadow it.
    exec env -u PYTHONPATH "$PY" "$REPO/scripts/verl/launch.py" "${LAUNCH[@]}" --dry-run "${OVERRIDES[@]}" ;;
  prepare-only)
    exec env -u PYTHONPATH CUDA_VISIBLE_DEVICES= \
      "$PY" "$REPO/scripts/verl/launch.py" "${SHARED[@]}" --prepare-only "${OVERRIDES[@]}" ;;
  run)
    # The run dir keeps the launcher's output. Each node of a multi-node run writes its own log there,
    # and the log of an earlier launch is renamed first.
    RUN_DIR=$RUNS/runs/$RUN
    NODE=${RANK:-0}
    LOG_NAME=launch
    [ "$NODE" = 0 ] || LOG_NAME=launch.node$NODE
    mkdir -p "$RUN_DIR"
    if [ -f "$RUN_DIR/$LOG_NAME.log" ]; then
      mv "$RUN_DIR/$LOG_NAME.log" "$RUN_DIR/$LOG_NAME.$(date +%Y%m%d-%H%M%S).log"
    fi
    # On a single node the launcher builds a missing model dir or dataset before it starts the GPUs.
    # tee ignores Ctrl-C and SIGTERM, so it keeps the launcher's shutdown lines and the launcher never
    # writes into a closed pipe while it stops Ray.
    set +e
    env -u PYTHONPATH "$PY" "$REPO/scripts/verl/launch.py" "${LAUNCH[@]}" "${OVERRIDES[@]}" 2>&1 \
      | { trap '' INT TERM; exec tee "$RUN_DIR/$LOG_NAME.log"; }
    exit "${PIPESTATUS[0]}" ;;
esac
