#!/usr/bin/env bash
# Measures, for one checkpoint, how many of n sampled answers to each prompt are correct. The keep set
# (prompts with 1..n-1 correct answers) is the gold data of the RL runs, because a prompt that every
# sample solves, or that none solves, gives GRPO no gradient.
#
# The measurement uses the RL run's own generation and reward path. Each shard of prompts is one
# val-only launch of configs/verl/rl_sft_mix.yaml, switched to the single-mode dataset builder
# (prepare_data) with the prompt mode of --kind. The validation sampling (val_kwargs) is set to the
# rollout sampling of the same composed config: temperature, top_p, top_k and n, with sampling on. The
# rollout n is the k of the pass rate. No checkpoint is resumed; the weights are the HF export's.
#
#   bash passrate.sh --kind math|code|math_ru --export <HF export> --model-name <name> --tag <tag> \
#       --src <jsonl> [options] [-- <hydra overrides>]
#
# Kinds:
#   math     prompt mode sft_think, the orbit/math reward (the last \boxed{} after </think>)
#   code     prompt mode sft_think_code, the orbit/code reward (the last ```python block after </think>
#            runs against the row's tests in the sandbox; 1 only when every test passes)
#   math_ru  prompt mode sft_think_ru (the Russian instruction), the orbit/math reward
# Every reward gives 0 to a response cut by the length limit and to one that never closes its thinking
# block, so a sample counts as correct when its score is 1.
#
# Required:
#   --kind K, --export DIR, --model-name NAME, --tag TAG
#   --src FILE         rows to sample from (instruction, response, source, ...). A new tag needs it. A tag
#                      whose subset exists reuses that subset, and a --src given again must be the file the
#                      subset was drawn from, so that one tag never mixes two datasets.
#   --python PY        the verl env's python (default: $VERL_PYTHON)
#   --runs-root DIR    root of model dirs, datasets, runs and caches (default: $ORBIT_VERL_RUNS)
# Options:
#   --work-dir DIR     where subsets and results go (default: <runs root>/passrate)
#   --count N          random rows of --src to measure (default: all of them)
#   --seed S           seed of that sample (default 0)
#   --sources S...     keep only rows whose source field is one of these
#   --exclude FILE...  skip rows whose unique_id is in these jsonl files (earlier subsets or results)
#   --omr-prior FILE   the OpenMathReasoning prior that omr_join.py writes; needed by the two filters below
#   --drop-omr-unmatched / --drop-omr-prior0
#                      skip OMR rows (rows whose source field is "OMR", the label of nvidia/OpenMathReasoning
#                      rows) without a pass_rate_72b_tir, or with a prior of 0 (make_subset.py says why)
#   --gpus G           GPUs of this node (default: nvidia-smi count)
#   --n N              samples per prompt (default: actor_rollout_ref.rollout.n of the config)
#   --kv fp8|auto      vLLM KV cache dtype (default fp8, as in the RL rollouts)
#   --gpu-mem U        vLLM share of each GPU (default 0.85; actor and optimizer are offloaded to the CPU)
#   --shard-size M     prompts per val-only launch (default 512; 200 for math_ru). A shard that is
#                      interrupted is generated again from its start, so smaller shards lose less work when
#                      a run is interrupted.
#   --shard-from I / --shard-to J
#                      measure only shards I..J (numbers of sNNN), e.g. to split one tag over several
#                      machines that share the runs root. Run the same command once with
#                      --prepare-only first, so the machines do not build the model dir, the subset
#                      and its shards at the same time. Such a run skips the final merge, which
#                      --merge-only then does for all shards.
#   --merge-only       generate nothing; merge every shard of the tag that has passrate.jsonl
#   --workers W        agent-loop and reward workers (default 8; use 2-4 on a machine with few CPUs)
#   --code-threads T   code only: sandbox threads per reward worker (default 4), or "config" to keep the
#                      reward's own default of max(2, CPUs // 8). The CPU note below explains the default.
#   --keep-min K / --keep-max K
#                      keep rule for results/<tag>/keep.jsonl (default 1 .. n-1)
#   --run-prefix X     run dirs are <X>__<tag>__sNNN (default passrate, passrate_code or passrate_ru)
#   --echo             also print the launcher output, not only into the run's launch.log
#   --prepare-only     build the model dir, the subset, its shards and the datasets of the selected
#                      shards on the CPU, print what the datasets dropped, and stop
#   --dry-run          print the launcher plan of the first selected shard and stop
#   --skip-sandbox-check
#                      code only: do not run the sandbox self-check before the first generating launch
# Everything after -- goes to the launcher as hydra overrides.
#
# Prompt length. The math and math_ru datasets are built with data.max_prompt_length=1024, the limit
# that the recipes' math keeps were measured with. The code datasets keep rl_sft_mix's 2048 tokens,
# the limit of the recipes' code keep and of RL itself. An override after -- replaces either limit.
#
# CPU note (code). Each reward worker runs the programs in a pool of code_threads threads, and each
# test case is one sandbox run with a limit of 6 CPU seconds. The sandbox also kills a run after 14
# wall seconds (2 * 6 + 2). On a node with more busy processes than CPUs, a correct but slow program
# can therefore hit the wall limit and count as a timeout. The default of 8 workers x 4 threads runs
# at most 32 programs at once and leaves the other CPUs to the 8 vLLM engines and Ray. On a machine with
# fewer CPUs, --code-threads config sizes the pool from the CPU quota. The thread count changes only how many programs run at the same time, not how they are judged.
#
# Outputs: <runs root>/runs/<prefix>__<tag>__sNNN/{launch.log,val_generations,passrate.jsonl,
# passrate_summary.json} per shard; <work dir>/<kind>/subsets/<tag>.jsonl and its shards; and
# <work dir>/<kind>/results/<tag>/{passrate.jsonl,keep.jsonl,summary.json}. A relaunch with the same
# tag skips the shards that already have passrate.jsonl.
set -euo pipefail

P=$(cd "$(dirname "$0")" && pwd)
# scripts/verl/data/passrate -> the repo root is four levels up.
REPO=$(cd "$P/../../../.." && pwd)
CONFIG=rl_sft_mix

KIND= EXPORT= NAME= TAG= SRC= COUNT= SEED=0 GPUS= N= KV=fp8 GMEM=0.85 SHARD= WORKERS=8 CODE_THREADS=4
SFROM=0 STO=999999 MERGE_ONLY= KEEP_MIN=1 KEEP_MAX= PREFIX= ECHO= DRY= PREP_ONLY= SKIP_SANDBOX= RANGE=
PY=${VERL_PYTHON:-} RUNS_ROOT=${ORBIT_VERL_RUNS:-} WORK= PRIOR=
FILTERS=() SOURCES=() EXCLUDE=() EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    --kind) KIND=$2; shift 2;;
    --export) EXPORT=$2; shift 2;;
    --model-name) NAME=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    --src) SRC=$2; shift 2;;
    --python) PY=$2; shift 2;;
    --runs-root) RUNS_ROOT=$2; shift 2;;
    --work-dir) WORK=$2; shift 2;;
    --count) COUNT=$2; shift 2;;
    --seed) SEED=$2; shift 2;;
    --sources) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do SOURCES+=("$1"); shift; done;;
    --exclude) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do EXCLUDE+=("$1"); shift; done;;
    --omr-prior) PRIOR=$2; shift 2;;
    --drop-omr-unmatched|--drop-omr-prior0) FILTERS+=("$1"); shift;;
    --gpus) GPUS=$2; shift 2;;
    --n) N=$2; shift 2;;
    --kv) KV=$2; shift 2;;
    --gpu-mem) GMEM=$2; shift 2;;
    --shard-size) SHARD=$2; shift 2;;
    --workers) WORKERS=$2; shift 2;;
    --code-threads) CODE_THREADS=$2; shift 2;;
    --shard-from) SFROM=$2; RANGE=1; shift 2;;
    --shard-to) STO=$2; RANGE=1; shift 2;;
    --merge-only) MERGE_ONLY=1; shift;;
    --keep-min) KEEP_MIN=$2; shift 2;;
    --keep-max) KEEP_MAX=$2; shift 2;;
    --run-prefix) PREFIX=$2; shift 2;;
    --echo) ECHO=--echo; shift;;
    --prepare-only) PREP_ONLY=1; shift;;
    --dry-run) DRY=1; shift;;
    --skip-sandbox-check) SKIP_SANDBOX=1; shift;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "passrate.sh: unknown argument $1 (see the header of $0)" >&2; exit 2;;
  esac
done
die() { echo "passrate.sh: $*" >&2; exit 2; }
[ -n "$KIND" ] && [ -n "$EXPORT" ] && [ -n "$NAME" ] && [ -n "$TAG" ] || die "--kind, --export, --model-name and --tag are required"
[ -n "$PY" ] || die "pass --python with the verl env's python, or set VERL_PYTHON"
[ -n "$RUNS_ROOT" ] || die "pass --runs-root, or set ORBIT_VERL_RUNS"
# Each kind fixes its prompt mode, the reward its rows must route to, its aggregator and its defaults.
# PROMPT_LIMIT is empty when the kind keeps the config's data.max_prompt_length (see "Prompt length").
case $KIND in
  math)    MODE=sft_think;      DSRC_WANT=orbit/math; AGG=passrate_agg.py;      SHARD=${SHARD:-512}; PREFIX=${PREFIX:-passrate};      PROMPT_LIMIT=1024;;
  code)    MODE=sft_think_code; DSRC_WANT=orbit/code; AGG=passrate_agg_code.py; SHARD=${SHARD:-512}; PREFIX=${PREFIX:-passrate_code}; PROMPT_LIMIT=;;
  math_ru) MODE=sft_think_ru;   DSRC_WANT=orbit/math; AGG=passrate_agg.py;      SHARD=${SHARD:-200}; PREFIX=${PREFIX:-passrate_ru};   PROMPT_LIMIT=1024;;
  *) die "--kind must be math, code or math_ru";;
esac
case $KV in fp8|auto) ;; *) die "--kv must be fp8 or auto";; esac
case $CODE_THREADS in config|[1-9]|[1-9][0-9]) ;; *) die "--code-threads must be 1-99 or config";; esac
if [ ${#FILTERS[@]} -gt 0 ] && [ -z "$PRIOR" ]; then die "${FILTERS[*]} needs --omr-prior <prior jsonl>"; fi
# Subsets and results are data, so they go under the runs root by default, never into the repo.
WORK=${WORK:-$RUNS_ROOT/passrate}
SUBSETS=$WORK/$KIND/subsets
RESULTS=$WORK/$KIND/results
# A merge needs no GPU, and a CPU node may have no nvidia-smi at all.
[ -z "$GPUS" ] && GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)

# Every cache, the model dir, the datasets and the runs live under the runs root, never in the repo.
export ORBIT_VERL_RUNS=$RUNS_ROOT
# The release runs and the two-node Ray check used the pure-python protobuf. A value set by the
# caller wins.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}
log() { echo "[passrate-$KIND $(date +%H:%M:%S)] $*"; }

# 1. The model dir, prepared as the launcher prepares it, so the RL runs can share it. A dir that exists
# is reused. The recipes' start export prepares on its own (prepare_model's docstring says why). An
# export that bundles padding_idx=151643 with a non-zero row is refused here; prepare its dir by hand
# first with --embedding-padding-idx none (docs/verl/README.md).
MD=$RUNS_ROOT/models/$NAME
if [ ! -f "$MD/orbit_prepare.json" ] && [ -z "$MERGE_ONLY" ]; then
  log "preparing model dir $MD from $EXPORT"
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$REPO/src "$PY" -m orbit_verl.model.prepare_model --export "$EXPORT" \
    --out-dir "$MD" --runs-root "$RUNS_ROOT" --preset qwen3 --check
fi

# 2. The subset and its shards. An existing subset of the same tag is reused as it is.
SUB=$SUBSETS/$TAG.jsonl
if [ ! -s "$SUB" ]; then
  [ -z "$MERGE_ONLY" ] || die "--merge-only: tag $TAG has no subset $SUB"
  [ -n "$SRC" ] || die "tag $TAG has no subset yet; pass --src <jsonl>"
  args=(sample --src "$SRC" --out "$SUB" --seed "$SEED" ${FILTERS[@]+"${FILTERS[@]}"})
  [ -n "$PRIOR" ] && args+=(--prior "$PRIOR")
  [ -n "$COUNT" ] && args+=(--count "$COUNT")
  [ ${#EXCLUDE[@]} -gt 0 ] && args+=(--exclude "${EXCLUDE[@]}")
  [ ${#SOURCES[@]} -gt 0 ] && args+=(--sources "${SOURCES[@]}")
  "$PY" -I "$P/make_subset.py" "${args[@]}"
else
  if [ -n "$SRC" ]; then
    "$PY" -I - "$SUB.meta.json" "$SRC" <<'EOF' || exit 2
import json, os, sys
recorded = json.load(open(sys.argv[1], encoding="utf-8"))["src"]
if os.path.abspath(sys.argv[2]) != recorded:
    sys.exit(f"passrate.sh: the subset of this tag was drawn from {recorded}, not from {os.path.abspath(sys.argv[2])}; "
             "use another --tag for another source")
EOF
  fi
  log "reusing subset $SUB ($(wc -l < "$SUB") rows)"
fi
"$PY" -I "$P/make_subset.py" shard --subset "$SUB" --out-dir "$SUBSETS/$TAG.shards" --shard-size "$SHARD"

# 3. The rollout sampling, read from the composed config with the user's overrides applied. The data
# source comes from the same config and must be the reward of this kind: a math reward on code
# problems, for example, would score every sample 0.
BASE_OVERRIDES=(orbit.prompt_mode=$MODE orbit.prepare.data.builder=prepare_data
                orbit.prepare.data.gold=null orbit.prepare.data.val=null)
[ -n "$PROMPT_LIMIT" ] && BASE_OVERRIDES+=(data.max_prompt_length=$PROMPT_LIMIT)
read -r TEMP TOPP TOPK RN MAXRESP MAXPROMPT DSRC CFG_THREADS < <(PYTHONPATH=$REPO/src "$PY" - "$CONFIG" \
    "${BASE_OVERRIDES[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" <<'EOF'
import sys
from omegaconf import OmegaConf
from orbit_verl.data.prompts import get_mode
from orbit_verl.launch.config import compose_config
cfg = compose_config(sys.argv[1], sys.argv[2:])
r = cfg.actor_rollout_ref.rollout
mode = str(cfg.orbit.prompt_mode)
source = OmegaConf.select(cfg, "orbit.prepare.data.data_source") or get_mode(mode).default_data_source
threads = OmegaConf.select(cfg, "reward.custom_reward_function.reward_kwargs.code_threads")
print(r.temperature, r.top_p, r.top_k, r.n, OmegaConf.select(cfg, "data.max_response_length"),
      OmegaConf.select(cfg, "data.max_prompt_length"), source, threads)
EOF
)
[ "$DSRC" = "$DSRC_WANT" ] || die "prompt mode $MODE routes to data source $DSRC, not $DSRC_WANT"
N=${N:-$RN}
THREADS_ARG=()
if [ "$KIND" = code ] && [ "$CODE_THREADS" != config ]; then
  THREADS_ARG=(reward.custom_reward_function.reward_kwargs.code_threads=$CODE_THREADS)
fi
log "rollout sampling: temperature $TEMP top_p $TOPP top_k $TOPK n $RN, response limit $MAXRESP; prompt mode $MODE" \
  "(prompt limit $MAXPROMPT); measuring n=$N, kv $KV, $GPUS GPU(s)"
if [ "$KIND" = code ]; then
  log "reward: $WORKERS workers x $([ "$CODE_THREADS" = config ] && echo "the config's $CFG_THREADS" || echo "$CODE_THREADS") sandbox threads"
fi

# Code only: every row's tests are parsed with the reward's own parser before a shard goes to the GPUs.
# The reward raises on a malformed row, and the launch would then stop in the middle of generation.
check_rows() {
  PYTHONPATH=$REPO/src "$PY" - "$1" <<'EOF'
import json, sys
from orbit_verl.data.reward import parse_code_ground_truth
bad = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
                raise ValueError("the row has no instruction text")
            parse_code_ground_truth(str(row.get("response")))
        except ValueError as exc:
            bad.append(f"  line {number} ({row.get('unique_id')}): {str(exc)[:300]}")
if bad:
    print(f"passrate.sh: {len(bad)} row(s) of {sys.argv[1]} are not code problems the reward can judge:",
          *bad[:20], sep="\n", file=sys.stderr)
    sys.exit(2)
EOF
}

# Code only: the sandbox needs user namespaces, Landlock and seccomp on this machine. Each reward worker
# checks them itself, but only when the first response is scored, which is long after start-up. A
# machine without them is found here in a second instead, before any GPU time is spent.
SANDBOX_CHECKED=
sandbox_check() {
  if [ "$KIND" != code ] || [ -n "$SKIP_SANDBOX" ] || [ -n "$SANDBOX_CHECKED" ]; then return 0; fi
  PYTHONPATH=$REPO/src "$PY" - "$WORKERS" "$CODE_THREADS" <<'EOF'
import json, sys
from orbit_verl.code_sandbox import available_cpus, self_check
report = self_check()
cpus = available_cpus()
failed = sorted(name for name, value in report.items() if value is not True)
workers, threads = int(sys.argv[1]), sys.argv[2]
per_worker = max(2, cpus // 8) if threads == "config" else int(threads)
print(f"[passrate-code] sandbox self-check {'FAILED: ' + ', '.join(failed) if failed else 'passed'}; "
      f"{cpus} CPUs available; up to {workers * per_worker} programs at once ({workers} x {per_worker})")
if workers * per_worker > cpus:
    print(f"[passrate-code] WARNING: more sandbox threads than CPUs; slow but correct programs may hit "
          f"the wall-clock limit and count as timeouts. Lower --code-threads or --workers.")
if failed:
    print(json.dumps(report), file=sys.stderr)
    sys.exit(2)
EOF
  SANDBOX_CHECKED=1
}

# The launcher arguments of one shard. verl's train loader drops the last partial batch and refuses
# to start without a full one, so the train batch must not exceed the rows that the dataset kept.
make_args() {
  local shard=$1 run=$2 tb=$3
  ARGS=(--config "$CONFIG" --export "$EXPORT" --model-name "$NAME" --runs-root "$RUNS_ROOT"
    "${BASE_OVERRIDES[@]}"
    orbit.run_name=$run data.train_batch_size=$tb
    orbit.prepare.data.src="$shard" orbit.prepare.data.val_src="$shard"
    orbit.prepare.data.name=${run}__${MODE}__qwen3
    trainer.val_before_train=True trainer.val_only=True trainer.resume_mode=disable
    'trainer.logger=[console,file]'
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMP actor_rollout_ref.rollout.val_kwargs.top_p=$TOPP
    actor_rollout_ref.rollout.val_kwargs.top_k=$TOPK actor_rollout_ref.rollout.val_kwargs.n=$N
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.gpu_memory_utilization=$GMEM
    # vLLM logs its KV cache size and, every 10 s, its throughput, and the aggregators read those lines.
    # verl sets VLLM_LOGGING_LEVEL=WARN for every Ray process, and orbit.env overrides it.
    actor_rollout_ref.rollout.disable_log_stats=False ++orbit.env.VLLM_LOGGING_LEVEL=INFO
    actor_rollout_ref.rollout.agent.num_workers=$WORKERS reward.num_workers=$WORKERS
    # Nothing trains, so the actor weights and the optimizer wait on the CPU, the reference model is
    # not built at all, and vLLM gets the GPU memory they would take.
    actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.use_kl_loss=False
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=$KV
    ${THREADS_ARG[@]+"${THREADS_ARG[@]}"}
    ${EXTRA[@]+"${EXTRA[@]}"})
}

# Prints the train batch for a dataset dir: the given batch, or fewer when the dataset kept fewer rows.
batch_for() {
  "$PY" -I -c 'import json, sys; m = json.load(open(sys.argv[1])); print(min(int(sys.argv[2]), m["rows"]["train"]))' "$1" "$2"
}

# One line on what prepare_data kept. Every prompt longer than data.max_prompt_length is dropped here,
# exactly as the RL run itself drops it. The manifest adds up the drops of the train and the val
# split. Both splits are this same shard, so each count is halved to count every dropped row once.
data_info() {
  "$PY" -I -c 'import json, sys; m = json.load(open(sys.argv[1])); same = m.get("src") == m.get("val_src"); dropped = {k: v // 2 if same else v for k, v in m["dropped"].items()}; print("%s of %s rows kept, dropped %s, prompt tokens %s" % (m["rows"]["val"], sys.argv[2], dropped, m["prompt_tokens"]))' "$1" "$2"
}

RUN_DIRS=()
for shard in "$SUBSETS/$TAG.shards"/s*.jsonl; do
  sid=$(basename "$shard" .jsonl)
  RUN=${PREFIX}__${TAG}__${sid}
  RD=$RUNS_ROOT/runs/$RUN
  DATA=$RUNS_ROOT/data/${RUN}__${MODE}__qwen3
  if [ -n "$MERGE_ONLY" ]; then
    [ -f "$RD/passrate.jsonl" ] && RUN_DIRS+=("$RD")
    continue
  fi
  num=$((10#${sid#s}))
  { [ "$num" -lt "$SFROM" ] || [ "$num" -gt "$STO" ]; } && continue
  RUN_DIRS+=("$RD")
  if [ -f "$RD/passrate.jsonl" ]; then log "$RUN done already"; continue; fi
  [ "$KIND" = code ] && check_rows "$shard"
  rows=$(wc -l < "$shard")
  TB=$(( rows < 64 ? rows : 64 ))
  [ -f "$DATA/manifest.json" ] && TB=$(batch_for "$DATA/manifest.json" "$TB")
  make_args "$shard" "$RUN" "$TB"
  if [ -n "$DRY" ]; then
    set +e
    CUDA_VISIBLE_DEVICES= "$PY" "$REPO/scripts/verl/launch.py" --dry-run --gpus-per-node "$GPUS" "${ARGS[@]}"
    status=$?
    set -e
    if [ "$status" -ne 0 ] && [ ! -f "$DATA/manifest.json" ]; then
      log "the launcher reports problems (exit $status). Before --prepare-only or the first real launch, a" \
        "missing dataset dir is one of them and is expected."
    fi
    exit "$status"
  fi
  # The dataset is built on the CPU before any GPU starts.
  if [ ! -f "$DATA/manifest.json" ]; then
    CUDA_VISIBLE_DEVICES= "$PY" "$REPO/scripts/verl/launch.py" --prepare-only "${ARGS[@]}"
    TB=$(batch_for "$DATA/manifest.json" "$TB")
    make_args "$shard" "$RUN" "$TB"
  fi
  log "$RUN: $(data_info "$DATA/manifest.json" "$rows")"
  [ -n "$PREP_ONLY" ] && continue
  mkdir -p "$RD"
  if [ -d "$RD/val_generations" ] && ls "$RD/val_generations"/*.jsonl >/dev/null 2>&1; then
    log "$RUN has generations from an earlier launch; aggregating them instead of generating again"
  else
    sandbox_check
    log "$RUN: $rows rows x $N samples on $GPUS GPU(s); log $RD/launch.log"
    # A failed earlier launch keeps its log under another name, so the timing of this one stays clean.
    [ -f "$RD/launch.log" ] && mv "$RD/launch.log" "$RD/launch.$(date +%Y%m%d-%H%M%S).log"
    set +e
    "$PY" "$REPO/scripts/verl/launch.py" --gpus-per-node "$GPUS" "${ARGS[@]}" 2>&1 | "$PY" -I "$P/tslog.py" "$RD/launch.log" $ECHO
    status=${PIPESTATUS[0]}
    set -e
    [ "$status" -eq 0 ] || { log "$RUN failed with exit code $status; see $RD/launch.log"; exit "$status"; }
  fi
  # Exit code 3 means some prompts of the shard could not be matched to the subset. The shard's other
  # rows are still valid, so the run goes on with the next shard instead of stopping the measurement.
  AGG_ARGS=(shard --run-dir "$RD" --subset "$shard" --tokenizer "$MD")
  if [ "$KIND" = code ]; then AGG_ARGS+=(--data-manifest "$DATA/manifest.json"); else AGG_ARGS+=(--prompt-mode "$MODE"); fi
  set +e
  HF_MODULES_CACHE=$RUNS_ROOT/cache/$(hostname)/hf_modules "$PY" -I "$P/$AGG" "${AGG_ARGS[@]}"
  agg_status=$?
  set -e
  if [ "$agg_status" -eq 3 ]; then
    log "WARNING: $RUN has prompts that did not match the subset; see its passrate_summary.json"
  elif [ "$agg_status" -ne 0 ]; then
    log "aggregation of $RUN failed with exit code $agg_status"; exit "$agg_status"
  fi
done

if [ -n "$PREP_ONLY" ]; then
  log "datasets of the selected shards are ready; nothing was generated"
  exit 0
fi

# A run that measures only a range of shards must not write results/<tag> from its own shards alone,
# because runs on other ranges would overwrite each other. The final merge is then a separate
# --merge-only call, which collects every finished shard of the tag.
if [ -n "$RANGE" ] && [ -z "$MERGE_ONLY" ]; then
  log "shards $SFROM..$STO done; run the same command with --merge-only to collect every shard of $TAG"
  exit 0
fi

[ ${#RUN_DIRS[@]} -gt 0 ] || { log "no finished shards to merge"; exit 0; }
merge=(merge --subset "$SUB" --out-dir "$RESULTS/$TAG" --keep-min "$KEEP_MIN")
[ -n "$KEEP_MAX" ] && merge+=(--keep-max "$KEEP_MAX")
"$PY" -I "$P/$AGG" "${merge[@]}" "${RUN_DIRS[@]}"
log "results in $RESULTS/$TAG (passrate.jsonl, keep.jsonl, summary.json)"
