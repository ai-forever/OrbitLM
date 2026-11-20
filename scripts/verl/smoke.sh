#!/bin/bash
# A short GPU check of the two recipes without the full data: 2 steps of 4 prompts with responses of at most
# 1,024 tokens, on the real configs with small overrides. It checks the whole path (data build, vLLM rollouts
# with the fp8 KV cache, rewards and the code sandbox, the RLTT or distillation update, validation, the
# checkpoint save). Its numbers mean nothing.
#
#   bash scripts/verl/smoke.sh rl      --export DIR --data DIR [options]   # rl_sft_mix on 1 GPU
#   bash scripts/verl/smoke.sh distill --export DIR --data DIR --teacher DIR [options]
#                                                                         # distill_rl_sft_16k on 2 GPUs:
#                                                                         # 1 student + 1 teacher
# Options:
#   --export DIR      the HF chat export of the start checkpoint ($ORBIT_EXPORTS/sft_v5_step7000).
#   --data DIR        the dir with the input files (docs/verl/README.md, "Smoke test", lists them).
#   --teacher DIR     distill only: a local Qwen3-8B model dir.
#   --runs-root DIR   default: $ORBIT_VERL_RUNS.
#   --python PY       the verl env's python; default: $VERL_PYTHON.
#   --print           print the launcher arguments, one per line, and run nothing. The tests use it.
#   --dry-run         compose and check everything, build nothing, start nothing.
#   -- OVERRIDE ...   hydra overrides for the launcher. They come after the smoke's own overrides, so
#                     they win. A longer data.max_response_length, for example, gives GRPO groups whose
#                     answers differ in reward. Raise the three token budgets with it
#                     (actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
#                     actor_rollout_ref.ref.log_prob_max_token_len_per_gpu and
#                     actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu); the launcher refuses a
#                     budget below the prompt plus the response.
#
# Run it again after deleting <runs root>/runs/smoke_<kind>, because a relaunch with the same run name
# resumes from the saved step 2 and ends at once.
set -euo pipefail

usage() {
  echo "usage: smoke.sh rl|distill --export DIR --data DIR [--teacher DIR] [--runs-root DIR] [--python PY]" \
       "[--print|--dry-run] [-- OVERRIDE ...]" >&2
  exit 2
}
[ $# -ge 1 ] || usage
KIND=$1
shift
EXPORT= DATA= TEACHER= MODE=run
EXTRA=()
RUNS=${ORBIT_VERL_RUNS:-}
PY=${VERL_PYTHON:-}
while [ $# -gt 0 ]; do
  case $1 in
    --export) EXPORT=$2; shift 2 ;;
    --data) DATA=$2; shift 2 ;;
    --teacher) TEACHER=$2; shift 2 ;;
    --runs-root) RUNS=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --print) MODE=print; shift ;;
    --dry-run) MODE=dry-run; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[ -n "$EXPORT" ] && [ -n "$DATA" ] && [ -n "$RUNS" ] || { echo "smoke.sh needs --export, --data and a runs root" >&2; usage; }
[ "$MODE" = print ] || [ -n "$PY" ] || { echo "smoke.sh needs --python or VERL_PYTHON" >&2; usage; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)

MATH_KEEP=$DATA/math_keep.jsonl
CODE_KEEP=$DATA/code_keep.jsonl
M500=$DATA/math500.jsonl
LCB=$DATA/lcb_val78.jsonl
POOL=$DATA/prompts.jsonl
MGSM_RU=$DATA/mgsm_ru.jsonl
# 16 rows of each gold file are sampled after filtering, and 4 rows of each validation set are kept.
GOLD="[\"$MATH_KEEP:orbit/math:math:16\",\"$CODE_KEEP:orbit/code:code:16\"]"

# The settings both smokes share: 2 steps of 4 prompts, responses of at most 1,024 tokens, validation
# before training and after step 2, a checkpoint at step 2, and fewer agent-loop and reward workers.
SHARED=(
  "orbit.run_name=smoke_$KIND" "orbit.prepare.data.name=smoke_${KIND}__qwen3" "orbit.prepare.data.gold=$GOLD"
  data.train_batch_size=4 data.max_response_length=1024
  trainer.total_training_steps=2 trainer.save_freq=2 trainer.test_freq=2 trainer.val_before_train=true
  actor_rollout_ref.rollout.agent.num_workers=2 reward.num_workers=2
  # vLLM sleeps while the actor trains, but on one GPU the actor's fp32 weights and optimizer stay
  # resident, so vLLM gets a smaller share than in the 8-GPU recipes.
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
)
case $KIND in
  rl)
    CONFIG=rl_sft_mix; GPUS=1; FILES=("$MATH_KEEP" "$CODE_KEEP" "$M500" "$LCB")
    VAL="[\"$M500:orbit/math#think:sft_think:4\",\"$LCB:orbit/code#think:sft_think_code:4\"]"
    OVERRIDES=("${SHARED[@]}" "orbit.prepare.data.val=$VAL" actor_rollout_ref.rollout.n=4
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192) ;;
  distill)
    [ -n "$TEACHER" ] || { echo "the distill smoke needs --teacher" >&2; usage; }
    CONFIG=distill_rl_sft_16k; GPUS=2; FILES=("$MATH_KEEP" "$CODE_KEEP" "$M500" "$LCB" "$POOL" "$MGSM_RU")
    VAL="[\"$M500:orbit/math#think:sft_think:4\",\"$M500:orbit/math#no_think:sft_no_think:4\","
    VAL+="\"$LCB:orbit/code#think:sft_think_code:4\",\"$MGSM_RU:orbit/math#ru:sft_think_ru:4\"]"
    # A thinking budget of 512 of the 1,024 response tokens, so the forced close runs in the smoke too.
    # The train rows are capped at 32. The build still reads and length-filters the whole pool, which
    # takes a minute or two on the CPU.
    OVERRIDES=("${SHARED[@]}" "orbit.prepare.data.val=$VAL" "orbit.prepare.data.pool=$POOL"
      orbit.prepare.data.limit=32 orbit.distill.teacher_gpus_per_node=1
      "orbit.env.ORBIT_THINK_BUDGET='512'"
      "distillation.teacher_models.teacher_model.model_path=$TEACHER"
      distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=4096
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096) ;;
  *) echo "unknown smoke: $KIND" >&2; usage ;;
esac
OVERRIDES+=(${EXTRA[@]+"${EXTRA[@]}"})
ARGS=(--config "$CONFIG" --export "$EXPORT" --model-name sft_v5_step7000 --runs-root "$RUNS")

if [ "$MODE" = print ]; then
  printf '%s\n' "${ARGS[@]}" --gpus-per-node "$GPUS" -- "${OVERRIDES[@]}"
  exit 0
fi
missing=()
for file in "${FILES[@]}"; do [ -f "$file" ] || missing+=("$file"); done
[ -z "$TEACHER" ] || [ -d "$TEACHER" ] || missing+=("$TEACHER")
[ -d "$EXPORT" ] || missing+=("$EXPORT")
[ ${#missing[@]} -eq 0 ] || { echo "missing inputs: ${missing[*]}" >&2; exit 2; }

export ORBIT_VERL_RUNS=$RUNS
DRY=(); [ "$MODE" = dry-run ] && DRY=(--dry-run)
# The model dir and the dataset are built on the CPU first, so the GPUs start only on prepared inputs.
[ "$MODE" = dry-run ] || CUDA_VISIBLE_DEVICES= "$PY" "$REPO/scripts/verl/launch.py" "${ARGS[@]}" --prepare-only \
  "${OVERRIDES[@]}"
exec "$PY" "$REPO/scripts/verl/launch.py" ${DRY[@]+"${DRY[@]}"} "${ARGS[@]}" --gpus-per-node "$GPUS" "${OVERRIDES[@]}"
