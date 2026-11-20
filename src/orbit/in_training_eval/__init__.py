"""In-training evaluation for the OrbitLM trainer.

Every ``eval_every`` optimizer steps, training pauses on all ranks. Each rank evaluates its share
of a benchmark suite on its own GPU, rank 0 collects the scores and reports them to TensorBoard,
and to ClearML when it is enabled, at iteration = global_step, and then training resumes.
eval/in_training/README.md describes the configuration keys, the suites, the setup and the on-disk
layout.

The evaluation runs lm-eval on vLLM in subprocesses started with a separate interpreter
(``eval_python``). The training environment has its own torch and no lm-eval, while the evaluation
environment has its own torch, vLLM and lm-eval. That is why this package never imports lm_eval or
vllm, and why everything it needs from that environment goes through command lines and files.
The lm-eval is the public release pinned by LM_EVAL_VERSION. The startup check refuses any other
release, and also a copy of the pinned one whose files differ from what pip installed.

A suite with a ``chat:`` section evaluates SFT checkpoints the way they are used: through the chat
template, in think and no_think mode, with sampled answers and the gates of the think/no_think switch.
Suites without that section run the base few-shot protocol of the pretraining stages.

The modules, in the order in which an evaluation uses them:

    benchmarks   the benchmark registry: lm-eval tasks, metrics and cost estimates
    suite        suite files (eval/in_training/suites/*.yaml)
    plan         work units, their assignment to ranks, and the lm-eval processes
    options      the trainer's eval_* options and their startup checks
    harness      the pinned lm-eval release and the checks of its install folder
    preflight    the startup check of the evaluation environment
    settings     the evaluation settings of a run and the hashes that decide reuse
    export       the HF export of the live model
    environment  the environment and the command line of one evaluation process
    processes    the GPU of a rank, and running and killing evaluation processes
    files        JSON files, the file barrier and heartbeats in the shared run folder
    scoring      scores, chat diagnostics, gates and the report
    evaluator    InTrainingEval, which runs all of the above at an evaluated step
    cli          the plan and prefetch commands

src/orbit/train_eval.py is the import surface that train.py and orbit.export (for --chat-export)
use, and the entry point of ``python -m orbit.train_eval``.
"""
from __future__ import annotations

from .benchmarks import (
    BENCHMARKS, CHAT_MODEL, CHAT_TASK_DOCS, NLTK_PUNKT_CHECK, RULER_PROMPT_MARGIN, RULER_TASKS, Benchmark,
)
from .cli import main
from .environment import (
    EVAL_GPU_ENV, EVAL_NO_LIBSTDCXX_ENV, EVAL_PARENT_ENV, build_eval_env, build_lm_eval_command,
    compile_threads_per_rank, describe_libstdcxx_versions, environment_libstdcxx, libstdcxx_preload,
    libstdcxx_versions, prepare_rank_cache,
)
from .evaluator import InTrainingEval, build_in_training_eval
from .export import set_shipping_specials, ship_chat_export, tokenizer_stop_ids, write_eval_export
from .files import (
    Heartbeat, HeartbeatWatcher, process_outputs_ready, read_json, wait_for_files, write_json_atomic,
)
from .harness import (
    LM_EVAL_VERSION, WONDERWORDS_VERSION, harness_folder_problem, lm_eval_install_command, lm_eval_metadata_problem,
    verify_lm_eval_files,
)
from .options import add_eval_arguments, eval_requested, validate_eval_arguments
from .plan import (
    CHAT_TAIL_SECONDS_PER_TOKEN, CHAT_TIMEOUT_SECONDS_PER_TOKEN, EvalProcess, Plan, Unit, assign_units, build_plan,
    expand_units, process_key, signature_budget,
)
from .preflight import (
    CHAT_RENDER_CHECK, CUDA_CHECK, MANAGER_CHECK, PREFLIGHT_TIMEOUT_SECONDS, lm_eval_problems, preflight_modules,
    probe_eval_environment, run_preflight,
)
from .processes import find_eval_processes, find_eval_sessions, kill_eval_processes, run_eval_process
from .scoring import (
    ChatScores, Scores, clopper_pearson, report_eval_scalars, score_process_folders, score_processes, score_records,
    smallest_n_that_can_pass,
)
from .settings import EvalSettings
from .suite import (
    CHAT_PROMPT_ROOM, DEFAULT_DETECTORS, DEFAULT_SUITE_PATH, BenchmarkSettings, ChatSettings, Suite, load_suite,
)
