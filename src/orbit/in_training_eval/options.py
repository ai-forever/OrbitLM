"""The trainer's eval_* options: the argparse definitions and the startup checks that train.py calls."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Optional

from .benchmarks import RULER_PROMPT_MARGIN
from .harness import LM_EVAL_VERSION, harness_folder_problem
from .plan import expand_units, signature_budget
from .suite import CHAT_PROMPT_ROOM, DEFAULT_SUITE_PATH, Suite, load_itr_render, load_suite

# The weights an evaluation can export. The trainer keeps no other copy of the weights than the model
# as trained, so "raw" is the only kind; it also names the series of every reported score.
WEIGHT_KINDS = ("raw",)


def add_eval_arguments(parser: argparse.ArgumentParser, d: Callable[..., Any]) -> None:
    """Adds the in-training evaluation options; ``d`` reads the experiment-config default.

    Every default keeps the evaluation off, so a config without these keys trains with no
    evaluation pauses at all.
    """
    group = parser.add_argument_group("in-training evaluation (eval/in_training/README.md)")
    group.add_argument("--eval-every", type=int, default=d("eval_every", 0),
                       help="Evaluate every N optimizer steps; 0 switches the evaluation off.")
    group.add_argument("--eval-at-start", dest="eval_at_start", action="store_true",
                       help="Also evaluate once before the first training step.")
    group.add_argument("--no-eval-at-start", dest="eval_at_start", action="store_false")
    parser.set_defaults(eval_at_start=bool(d("eval_at_start", False)))
    group.add_argument("--eval-at-end", dest="eval_at_end", action="store_true",
                       help="Also evaluate the last step of the run when eval_every does not divide it.")
    group.add_argument("--no-eval-at-end", dest="eval_at_end", action="store_false")
    parser.set_defaults(eval_at_end=bool(d("eval_at_end", False)))
    group.add_argument("--eval-suite", type=Path, default=d("eval_suite", DEFAULT_SUITE_PATH))
    group.add_argument("--eval-python", type=Path, default=d("eval_python"),
                       help="Interpreter of the environment with lm-eval and vLLM.")
    group.add_argument("--eval-harness-path", type=Path, default=d("eval_harness_path"),
                       help=f"Folder with lm-eval {LM_EVAL_VERSION} installed by 'pip install --target' into "
                            "a new, empty folder, put first on PYTHONPATH of the evaluation processes. "
                            "Default: the lm-eval of eval_python's environment.")
    group.add_argument("--eval-extra-pythonpath", type=str, default=d("eval_extra_pythonpath"),
                       help="More PYTHONPATH entries for the evaluation processes (os.pathsep-separated).")
    group.add_argument("--eval-max-model-len", type=int, default=d("eval_max_model_len"),
                       help="Context window of the evaluated model; default max(seq_len, longest RULER "
                            f"length + {RULER_PROMPT_MARGIN}).")
    group.add_argument("--eval-gpu-memory-utilization", type=float,
                       default=d("eval_gpu_memory_utilization", 0.75))
    group.add_argument("--eval-timeout-minutes", type=float, default=d("eval_timeout_minutes", 30.0),
                       help="Upper cap per work unit. A subprocess gets the smaller of this times its "
                            "unit count and three times its planned time plus 10 minutes.")
    group.add_argument("--eval-total-timeout-minutes", type=float,
                       default=d("eval_total_timeout_minutes", 45.0),
                       help="Cap on the evaluation work of one pause; the pause can last up to about "
                            "12 minutes longer while the ranks clean up and line up.")
    group.add_argument("--eval-weights", choices=list(WEIGHT_KINDS), default=d("eval_weights", "raw"))
    group.add_argument("--eval-keep-exports", dest="eval_keep_exports", action="store_true",
                       help="Keep the HF export of every evaluated step (about 3.7 GB each for 1.4B).")
    group.add_argument("--no-eval-keep-exports", dest="eval_keep_exports", action="store_false")
    parser.set_defaults(eval_keep_exports=bool(d("eval_keep_exports", False)))
    group.add_argument("--eval-cache-dir", type=Path, default=d("eval_cache_dir"),
                       help="Offline cache of datasets and metrics; also holds the per-rank compile caches.")
    group.add_argument("--eval-tokenizer-source", type=Path, default=d("eval_tokenizer_source"),
                       help="Tokenizer folder for the export; default the folder of tokenizer_config.")


def eval_requested(args: argparse.Namespace) -> bool:
    return (int(getattr(args, "eval_every", 0) or 0) > 0 or bool(getattr(args, "eval_at_start", False))
            or bool(getattr(args, "eval_at_end", False)))


def _has_tokenizer_files(folder: Path) -> bool:
    return any((folder / name).is_file() for name in ("tokenizer.json", "vocab.json", "tokenizer.model"))


def validate_eval_arguments(args: argparse.Namespace, error: Callable[[str], Any]) -> None:
    """Checks the evaluation settings at startup, so a bad setting stops the run before step 1.

    ``error`` is parser.error. Nothing is checked or changed when the evaluation is off.
    """
    if args.eval_every is None:
        args.eval_every = 0
    if int(args.eval_every) < 0:
        error("--eval-every must be 0 (off) or a positive number of steps.")
    if not eval_requested(args):
        return
    if int(args.eval_every) > 0 and int(args.eval_every) % int(args.save_every) != 0:
        # The evaluation runs after the checkpoint of its step, so a crash or a stop during the
        # long pause loses nothing. That only holds when every evaluated step is a checkpoint step.
        error(
            f"--eval-every ({args.eval_every}) must be a multiple of --save-every ({args.save_every}), "
            "so that every evaluated step has a checkpoint."
        )
    if args.eval_python is None:
        error("The in-training evaluation needs eval_python, the interpreter of the lm-eval + vLLM environment.")
    if not Path(args.eval_python).is_file():
        error(f"eval_python {args.eval_python} does not exist.")
    if args.eval_harness_path is not None:
        problem = harness_folder_problem(Path(args.eval_harness_path), args.eval_python)
        if problem:
            error(problem)
    if args.eval_cache_dir is None:
        error("The in-training evaluation needs eval_cache_dir, the offline cache filled by 'train_eval prefetch'.")
    if not Path(args.eval_cache_dir).is_dir():
        error(f"eval_cache_dir {args.eval_cache_dir} is not a folder.")
    if not 0.05 <= float(args.eval_gpu_memory_utilization) <= 0.95:
        error("--eval-gpu-memory-utilization must be between 0.05 and 0.95.")
    if float(args.eval_timeout_minutes) <= 0 or float(args.eval_total_timeout_minutes) <= 0:
        error("--eval-timeout-minutes and --eval-total-timeout-minutes must be positive.")
    tokenizer_source = args.eval_tokenizer_source
    if tokenizer_source is None:
        tokenizer_source = Path(args.tokenizer_config).expanduser().resolve().parent
    if not _has_tokenizer_files(Path(tokenizer_source)):
        # tokenizer_config may name a config file that is kept outside a tokenizer folder. The export
        # copies the tokenizer files from this folder, so it must hold the run's real tokenizer.
        error(
            f"No tokenizer files in {tokenizer_source}. Set eval_tokenizer_source to the folder of "
            "the tokenizer this run trains with."
        )
    args.eval_tokenizer_source = Path(tokenizer_source).expanduser().resolve()
    try:
        suite = load_suite(args.eval_suite)
    except ValueError as exc:
        error(str(exc))
        return
    longest_ruler = max(suite.ruler_lengths, default=0)
    needed = longest_ruler + RULER_PROMPT_MARGIN if longest_ruler else 0
    if args.eval_max_model_len is None:
        args.eval_max_model_len = max(int(args.block_size), needed)
    if int(args.eval_max_model_len) < needed:
        error(
            f"eval_max_model_len {args.eval_max_model_len} is below the longest RULER length "
            f"{longest_ruler} of the suite {suite.path} plus {RULER_PROMPT_MARGIN} tokens for the "
            "answer prefix that lm-eval appends to RULER prompts."
        )
    if suite.chat is not None:
        problem = chat_suite_problem(suite, int(args.eval_max_model_len), Path(args.eval_tokenizer_source))
        if problem:
            error(problem)


def chat_suite_problem(suite: Suite, max_model_len: int, tokenizer_source: Path) -> Optional[str]:
    """Why a chat suite cannot run with this window and tokenizer folder, or None when it can.

    Every chat unit's budget plus CHAT_PROMPT_ROOM must fit the window, and the tokenizer folder that
    the export copies must carry a chat template, because the prompts are rendered with it.
    """
    try:
        units = expand_units(suite)
    except ValueError as exc:
        return f"{suite.path}: {exc}"
    for unit in units:
        budget = signature_budget(unit.signature)
        if budget + CHAT_PROMPT_ROOM > max_model_len:
            return (
                f"{suite.path}: {unit.benchmark} generates up to {budget} tokens, which leaves less than "
                f"{CHAT_PROMPT_ROOM} tokens for the prompt in the window of {max_model_len} (eval_max_model_len). "
                "lm-eval would cut the start of the prompt; lower the budget or raise the window."
            )
    if load_itr_render().template_text(tokenizer_source) is None:
        return (
            f"{tokenizer_source} has no chat template (chat_template.jinja or a chat_template entry in "
            "tokenizer_config.json). A chat suite renders its prompts with the template of the exported "
            "tokenizer; set eval_tokenizer_source to the tokenizer folder of the SFT run."
        )
    return None
