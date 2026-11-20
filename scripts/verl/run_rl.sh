#!/bin/bash
# Runs one arm of the RL recipe (configs/verl/rl_sft_mix.yaml) on this node, which needs 8 GPUs.
#
#   bash scripts/verl/run_rl.sh ARM --export DIR --data DIR [--runs-root DIR] [--python PY] [--gpu-mem U] \
#       [--model-name NAME] [--clearml-env FILE] [--gate] [--no-prepare] \
#       [--print | --dry-run | --prepare-only] [-- OVERRIDE ...]
#
#   ARM            math, code, mix, mix_ru, or smoke (the mix arm shrunk to 2 steps of 16 prompts on 8 GPUs).
#   --export       the HF chat export of the start checkpoint: the SFT export $ORBIT_EXPORTS/sft_v5_step7000,
#                  or the fp32 export of a distillation run (README.md).
#   --data         the dir that holds the gold and validation files (docs/verl/README.md lists them).
#   --runs-root    the root of models, datasets, runs and caches. Default: $ORBIT_VERL_RUNS.
#   --python       the verl env's python. Default: $VERL_PYTHON.
#   --gpu-mem      vLLM's share of each GPU. The config's 0.8 fits a 141 GB GPU; use 0.7 on 80 GB GPUs, so that
#                  the actor keeps room for its 18K-token micro-batches.
#   --model-name   the prepared model dir's name under <runs root>/models. Default: sft_v5_step7000.
#   --clearml-env  a dotenv file with ClearML credentials. Without it, ClearML stays off.
#   --gate         run the sandbox gate (below) on the smoke run first, and start a full arm only when it passes.
#   --no-prepare   never build a model dir or dataset; the launch needs them prepared already.
#
#   no flag         runs the arm on this node. The launcher builds a missing model dir or dataset first, and
#                   its output also goes to <runs root>/runs/<run name>/launch.log.
#   --print         prints the launcher's arguments and the overrides, one per line, and checks nothing.
#   --dry-run       composes and checks the arm, prints the launcher's plan, and builds and starts nothing.
#   --prepare-only  builds the arm's model dir and dataset when they are missing, without a GPU, and starts
#                   nothing.
#
# Several nodes. Run --prepare-only once on a machine that sees the runs root. Then start the same command
# with --no-prepare once on every node, one process per node: RANK is the node rank (0..N-1), WORLD_SIZE the
# number of nodes and MASTER_ADDR the address of node 0. Ray places the training processes on the GPUs
# itself, so a launcher must not start one process per GPU. With torchrun, node R runs
#
#   torchrun --nnodes N --node_rank R --nproc_per_node 1 --master_addr HOST --master_port 29500 \
#       --no-python bash scripts/verl/run_rl.sh ARM --no-prepare ...
#
# Node rank 0 runs the Ray head and trains, and the other nodes join it. The runs root must be on a
# filesystem that every node sees. Every node also needs the export at the same path, because the prepared
# model dir links to its weights. --data is read only while preparing, so a --no-prepare launch does not
# look for it. Put --no-prepare before any "--": the script passes everything after "--" to the launcher,
# and a --no-prepare there would not stop the script from checking the files in --data.
#
# Environment:
#   RU_KEEP, RU_COUNT  mix_ru only: the kept rows of the Russian pass-rate run (a file name in --data, or a
#                      path; default ru_math_keep.jsonl) and how many of them to sample (a number, or all;
#                      default all).
#   RECIPE_TAG         names the recipe in the run name. Overrides after "--" change the recipe, so they need
#                      a tag of their own; the launcher also refuses to resume a run whose recipe keys changed.
#   SMOKE_STEPS        the smoke arm's step count (default 2). See the sandbox gate below.
#
# The sandbox gate. Every arm validates on LCB, and verl drops a code prompt without stopping when the
# sandbox fails on the machine that computes the rewards. So, before the full arms run on a new kind of
# machine, the smoke arm runs there twice under one run name, and the gate checks both launches:
#
#   1. run_rl.sh smoke ...                  (SMOKE_STEPS=2: steps 1-2, validation at steps 0 and 2)
#   2. SMOKE_STEPS=3 run_rl.sh smoke ...    (resumes from step 2: step 3 and its validation, and no
#                                            validation again before step 3)
#   3. $VERL_PYTHON scripts/verl/sandbox_gate.py <runs root>/runs/rl_smoke__lr1e6_fp8tis
#   4. run_rl.sh math|code|mix|mix_ru --gate ...
#
# The second launch keeps the log of the first one as launch.<date-time>.log, so the gate finds the log of
# each launch in the run dir. It passes only with the logs of both launches, and it checks the validations
# at steps 0, 2 and 3.
#
# The arms differ only in their gold rows, their dataset name and their run name, and all of them share the
# validation sets, so their numbers compare across arms.
set -euo pipefail

usage() {
  echo "usage: run_rl.sh smoke|math|code|mix|mix_ru --export DIR --data DIR [options]" \
       "[--print|--dry-run|--prepare-only] [-- OVERRIDE ...]" >&2
  exit 2
}
# A prepared model dir may be read by other running launches, and --force-prepare would rebuild it under
# them. A dataset that does not match the config gets a new name instead.
refuse_force() {
  echo "run_rl.sh never passes --force-prepare: it would rebuild prepared dirs that other runs read" >&2
  exit 2
}
[ $# -ge 1 ] || usage
ARM=$1
shift
MODE=run
EXPORT= DATA= GPU_MEM= CLEARML_ENV= GATE= NO_PREPARE=
RUNS=${ORBIT_VERL_RUNS:-}
PY=${VERL_PYTHON:-}
MODEL_NAME=sft_v5_step7000
EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    --print|--dry-run|--prepare-only)
      [ $MODE = run ] || { echo "give only one of --print, --dry-run and --prepare-only" >&2; exit 2; }
      MODE=${1#--}; shift ;;
    --export) EXPORT=$2; shift 2 ;;
    --data) DATA=$2; shift 2 ;;
    --runs-root) RUNS=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --gpu-mem) GPU_MEM=$2; shift 2 ;;
    --model-name) MODEL_NAME=$2; shift 2 ;;
    --clearml-env) CLEARML_ENV=$2; shift 2 ;;
    --gate) GATE=1; shift ;;
    --no-prepare) NO_PREPARE=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    --force-prepare*) refuse_force ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
for arg in ${EXTRA[@]+"${EXTRA[@]}"}; do
  [ "${arg%%=*}" != --force-prepare ] || refuse_force
done
missing_args=()
[ -n "$EXPORT" ] || missing_args+=(--export)
[ -n "$DATA" ] || missing_args+=(--data)
[ -n "$RUNS" ] || missing_args+=("--runs-root (or ORBIT_VERL_RUNS)")
[ -n "$PY" ] || missing_args+=("--python (or VERL_PYTHON)")
if [ ${#missing_args[@]} -gt 0 ]; then
  echo "run_rl.sh needs ${missing_args[*]}" >&2
  usage
fi
if [ -n "$NO_PREPARE" ] && [ "$MODE" = prepare-only ]; then
  echo "give only one of --no-prepare and --prepare-only" >&2
  exit 2
fi
if [ ${#EXTRA[@]} -gt 0 ] && [ -z "${RECIPE_TAG:-}" ]; then
  echo "overrides after -- change the recipe: set RECIPE_TAG=<tag> to name it in the run name" >&2
  exit 2
fi
RECIPE_TAG=${RECIPE_TAG:-lr1e6_fp8tis}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

# The inputs, by the file names that docs/verl/README.md lists and scripts/verl/watch.py looks for.
MATH_KEEP=$DATA/math_keep.jsonl
CODE_KEEP=$DATA/code_keep.jsonl
M500=$DATA/math500.jsonl
LCB=$DATA/lcb_val78.jsonl
MGSM_RU=$DATA/mgsm_ru.jsonl
GSM_EN=$DATA/gsm8k_en_paired250.jsonl
CHAT_RU=$DATA/chat_probe_ru.jsonl
CHAT_EN=$DATA/chat_probe_en.jsonl

# The gold rows. The mixed arms sample 3,800 math rows with the dataset seed and keep every code row.
MATH_ALL="$MATH_KEEP:orbit/math:math"
MATH_HALF="$MATH_KEEP:orbit/math:math:3800"
CODE_ALL="$CODE_KEEP:orbit/code:code"
case $ARM in
  math) GOLD=("$MATH_ALL") ;;
  code) GOLD=("$CODE_ALL") ;;
  mix|smoke) GOLD=("$MATH_HALF" "$CODE_ALL") ;;
  mix_ru)
    RU_KEEP=${RU_KEEP:-ru_math_keep.jsonl}
    RU_COUNT=${RU_COUNT:-all}
    case $RU_KEEP in */*) RU_PATH=$RU_KEEP ;; *) RU_PATH=$DATA/$RU_KEEP ;; esac
    case $RU_COUNT in
      all) RU_SUFFIX= ;;
      *[!0-9]*|0*|'') echo "RU_COUNT must be a positive number of rows or all, not $RU_COUNT" >&2; exit 2 ;;
      *) RU_SUFFIX=:$RU_COUNT ;;
    esac
    # The Russian rows come after the math and code rows, so the seeded sampling picks the same math and
    # code rows as in mix, and mix_ru is mix plus the Russian rows.
    GOLD=("$MATH_HALF" "$CODE_ALL" "$RU_PATH:orbit/math#ru_train:math_ru$RU_SUFFIX") ;;
  *) echo "unknown arm: $ARM" >&2; usage ;;
esac

# The validation sets, the same for every arm. The Russian chat probe is listed twice under two tags, which
# gives two answers per prompt at n=1. The smoke arm keeps the first 8 rows of each set.
LIMIT=; NO_THINK_LIMIT=:200
if [ "$ARM" = smoke ]; then LIMIT=:8; NO_THINK_LIMIT=:8; fi
VAL=(
  "$M500:orbit/math#think:sft_think$LIMIT"
  "$M500:orbit/math#no_think:sft_no_think$NO_THINK_LIMIT"
  "$LCB:orbit/code#think:sft_think_code$LIMIT"
  "$MGSM_RU:orbit/math#ru:sft_think_ru$LIMIT"
  "$GSM_EN:orbit/math#mgsm_en:sft_think$LIMIT"
  "$CHAT_RU:orbit/none#chat_ru:sft_chat_think$LIMIT"
  "$CHAT_RU:orbit/none#chat_ru2:sft_chat_think$LIMIT"
  "$CHAT_EN:orbit/none#chat_en:sft_chat_think$LIMIT"
)

# A hydra list of quoted strings. The quotes are needed, because hydra refuses "#" outside them.
hydra_list() {
  local out= item
  for item in "$@"; do out+="${out:+,}\"$item\""; done
  printf '[%s]' "$out"
}

RUN=rl_${ARM}__${RECIPE_TAG}
OVERRIDES=(
  orbit.run_name="$RUN"
  orbit.prepare.data.name="rl_${ARM}__qwen3"
  "orbit.prepare.data.gold=$(hydra_list "${GOLD[@]}")"
  "orbit.prepare.data.val=$(hydra_list "${VAL[@]}")"
)
if [ "$ARM" = smoke ]; then
  SMOKE_STEPS=${SMOKE_STEPS:-2}
  case $SMOKE_STEPS in
    *[!0-9]*|''|0*) echo "SMOKE_STEPS must be a positive number, not $SMOKE_STEPS" >&2; exit 2 ;;
  esac
  # 16 prompts x 8 answers divide among 8 GPUs. The full response length stays, so the smoke tests the
  # memory of the real runs.
  OVERRIDES+=(data.train_batch_size=16 trainer.total_training_steps="$SMOKE_STEPS"
              trainer.save_freq=2 trainer.test_freq=2)
fi
if [ -n "$CLEARML_ENV" ]; then
  OVERRIDES+=(orbit.clearml_env="$CLEARML_ENV" "trainer.logger=[console,file,clearml]")
fi
if [ -n "$GPU_MEM" ]; then
  OVERRIDES+=(actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM")
fi
OVERRIDES+=(${EXTRA[@]+"${EXTRA[@]}"})

SHARED=(--config rl_sft_mix --export "$EXPORT" --model-name "$MODEL_NAME" --runs-root "$RUNS")
LAUNCH=("${SHARED[@]}" --gpus-per-node 8)
[ -z "$NO_PREPARE" ] || LAUNCH+=(--no-prepare)

if [ "$MODE" = print ]; then
  printf '%s\n' "${LAUNCH[@]}" -- "${OVERRIDES[@]}"
  exit 0
fi

# A --no-prepare launch reads only the prepared model dir and dataset under the runs root, so it needs no
# --data. It still needs the export, because the prepared model dir links to the export's weights.
CHECK_DIRS=("$RUNS")
[ -n "$NO_PREPARE" ] || CHECK_DIRS+=("$DATA")
CHECK_DIRS+=("$EXPORT")
for dir in "${CHECK_DIRS[@]}"; do
  [ -d "$dir" ] || { echo "$dir does not exist" >&2; exit 2; }
done
missing=()
if [ -z "$NO_PREPARE" ]; then
  for spec in "${GOLD[@]}" "${VAL[@]}"; do
    file=${spec%%:*}
    [ -f "$file" ] || missing+=("$file")
  done
fi
if [ ${#missing[@]} -gt 0 ]; then
  echo "missing inputs: ${missing[*]}" >&2
  # A dry run builds nothing, so it may go on and check the rest.
  [ "$MODE" = dry-run ] || exit 2
fi

if [ -n "$GATE" ] && [ "$ARM" != smoke ] && [ "$MODE" != prepare-only ]; then
  # The sandbox does not depend on the recipe, so every arm reads the smoke of the base recipe.
  GATE_RUN=$RUNS/runs/rl_smoke__lr1e6_fp8tis
  if ! "$PY" "$REPO/scripts/verl/sandbox_gate.py" "$GATE_RUN"; then
    if [ "$MODE" = run ]; then
      echo "the sandbox gate did not pass, so $ARM does not start (see scripts/verl/sandbox_gate.py)" >&2
      exit 2
    fi
    echo "the sandbox gate did not pass; a run of $ARM would stop here" >&2
  fi
fi

echo "arm $ARM, run $RUN, on this node, mode $MODE"
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
    # The run dir keeps the launcher's output. Each node of a multi-node run writes its own log there.
    # The log of an earlier launch is renamed first, so the sandbox gate finds one log per launch.
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
