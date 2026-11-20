"""Batch generation with vLLM, the recommended way to run OrbitLM checkpoints.

    python inference/generate_vllm.py MODEL_DIR --prompts prompts.jsonl --output answers.jsonl \\
        --speculative --budget-forcing

The defaults are the recommended serving settings: an FP8 KV cache on the FLASH_ATTN backend (FA3 on
Hopper GPUs), all loops of the model, think mode, and sampling at temperature 0.6, top_p 0.95, top_k 20.
--speculative adds self-speculative decoding with a one-loop draft of the same checkpoint, and
--budget-forcing closes the thinking block at a token budget. README.md in this folder has the
commands.

vLLM's Ouro model always runs all total_ut_steps loops and answers from the last one; it has no
early exit. --loops N sets total_ut_steps to N through hf_overrides.

It runs in the verl environment (requirements/verl/), which has vLLM 0.26, or with a newer vLLM.
vLLM removed its Ouro model after 0.26, and the orbit_vllm plugin brings it back in any environment
where the repository is installed with `pip install -e . --no-deps` (checked with vLLM 0.34).
Without --output, standard output carries the records and nothing else; everything else goes to
standard error.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

if __package__ in (None, ""):
    # Run as a file: put the repo root on sys.path, so that the inference package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import cli, runtime_env  # noqa: E402
from inference.budget import Completion, GenerateFn  # noqa: E402
from inference.make_draft import DRAFT_LOOPS, DraftDirError, default_draft_dir, make_draft  # noqa: E402
from inference.prompts import PromptError  # noqa: E402

if __name__ in ("__main__", "__mp_main__"):
    # vLLM starts its engine in a new process, which runs this file again under the name __mp_main__.
    # The check has to run in both processes, and in both before vLLM is imported.
    runtime_env.hide_flash_attn_if_broken()

# The measured self-speculative setting is four draft tokens per step. With it, each verification step
# of the full model produced 3.8 to 4.1 tokens on average.
DEFAULT_SPECULATIVE_TOKENS = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_shared_args(parser)
    engine = parser.add_argument_group("vLLM engine")
    engine.add_argument("--kv-cache-dtype", choices=("auto", "fp8"), default="fp8",
                        help="fp8 (default) doubles the KV capacity; auto keeps bf16")
    engine.add_argument("--attention-backend", default="FLASH_ATTN",
                        help="vLLM attention backend (default FLASH_ATTN, which runs FA3 on Hopper); auto lets vLLM choose")
    engine.add_argument("--speculative", action="store_true",
                        help="self-speculative decoding: the same checkpoint with one loop drafts tokens for the full model")
    engine.add_argument("--draft-dir", type=Path,
                        help="the one-loop draft dir; created when missing, reused when it is a draft of the same model "
                             "(default: a dir under the system temp dir)")
    engine.add_argument("--num-speculative-tokens", type=int,
                        help=f"draft tokens per step (default {DEFAULT_SPECULATIVE_TOKENS})")
    engine.add_argument("--tensor-parallel-size", type=int, default=1)
    engine.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser


def check_engine_args(args: argparse.Namespace) -> None:
    """Checks of the vLLM options that need nothing but the arguments. Raises cli.UsageError."""
    if args.tensor_parallel_size < 1:
        raise cli.UsageError("--tensor-parallel-size must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise cli.UsageError("--gpu-memory-utilization must be in (0, 1]")
    if not args.speculative:
        if args.draft_dir is not None or args.num_speculative_tokens is not None:
            raise cli.UsageError("--draft-dir and --num-speculative-tokens take effect only with --speculative")
        return
    if args.num_speculative_tokens is None:
        args.num_speculative_tokens = DEFAULT_SPECULATIVE_TOKENS
    if args.num_speculative_tokens < 1:
        raise cli.UsageError("--num-speculative-tokens must be positive")


def check_speculative_loops(args: argparse.Namespace) -> None:
    """Refuse a speculative run whose target runs no more loops than the one-loop draft."""
    if args.speculative and args.loops <= DRAFT_LOOPS:
        raise cli.UsageError(f"--speculative needs a target of at least {DRAFT_LOOPS + 1} loops, "
                             f"because the draft runs {DRAFT_LOOPS}")


def engine_kwargs(args: argparse.Namespace, draft_dir: Optional[Path] = None) -> dict[str, Any]:
    """The arguments of vllm.LLM for this run."""
    kwargs: dict[str, Any] = {
        "model": str(args.model),
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
    }
    if args.attention_backend != "auto":
        kwargs["attention_backend"] = args.attention_backend
    if args.loops != args.model_loops:
        kwargs["hf_overrides"] = {"total_ut_steps": args.loops}
    if draft_dir is not None:
        # The draft proposes its most likely tokens at every temperature. The full model then accepts
        # or replaces each proposed token by rejection sampling, so the output follows the full
        # model's own distribution, and the draft's sampling method changes only how many tokens are
        # accepted. The speed numbers in README.md were measured with this greedy draft.
        speculative: dict[str, Any] = {
            "method": "draft_model",
            "model": str(draft_dir),
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_sample_method": "greedy",
        }
        if args.attention_backend != "auto":
            # The draft picks its own attention backend unless it is told, so it gets the target's.
            speculative["attention_backend"] = args.attention_backend
        kwargs["speculative_config"] = speculative
        # vLLM schedules a draft_model run asynchronously only when asked to, and the measured speed-up
        # was with asynchronous scheduling on.
        kwargs["async_scheduling"] = True
    return kwargs


def sampling_kwargs(args: argparse.Namespace, end_ids: Sequence[int]) -> dict[str, Any]:
    """The arguments of vllm.SamplingParams that every request shares, that is all but max_tokens."""
    settings = cli.sampling_settings(args)
    kwargs: dict[str, Any] = {
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "stop_token_ids": list(end_ids),
        # The records are decoded from the token ids, so vLLM's own detokenizer is not needed.
        "detokenize": False,
    }
    if settings["top_k"] is not None:
        kwargs["top_k"] = settings["top_k"]
    return kwargs


def start_engine(args: argparse.Namespace, tokenizer: Any, end_ids: Sequence[int]) -> GenerateFn:
    draft_dir = None
    if args.speculative:
        draft_dir = make_draft(args.model, args.draft_dir or default_draft_dir(args.model))
        cli.log(f"draft model: {draft_dir} (total_ut_steps=1, files linked to the model dir)")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kwargs = engine_kwargs(args, draft_dir)
    cli.log(f"vLLM engine: { {key: value for key, value in kwargs.items() if key != 'model'} }")
    llm = LLM(**kwargs)
    shared = sampling_kwargs(args, end_ids)

    def generate(prompt_ids: list[list[int]], limits: list[int]) -> list[Completion]:
        outputs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids],
                               [SamplingParams(max_tokens=limit, **shared) for limit in limits], use_tqdm=True)
        return [Completion(tuple(int(t) for t in output.outputs[0].token_ids), str(output.outputs[0].finish_reason))
                for output in outputs]

    return generate


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cli.check_shared_args(args)
        check_engine_args(args)
        notes = cli.resolve_model_args(args, cli.read_model_config(args.model))
        check_speculative_loops(args)
    except cli.UsageError as exc:
        parser.error(str(exc))
    for note in notes + runtime_env.prepare_process(for_vllm=True):
        cli.log(note)
    try:
        return cli.run(args, start_engine)
    except (cli.UsageError, PromptError, DraftDirError) as exc:
        cli.log(f"error: {exc}")
        return 2


def exit_hard(entry: Callable[[], int]) -> None:
    """Run entry and end the process with os._exit, whatever happens.

    vLLM can hang in its shutdown, also after an exception, and keep holding the GPU. os._exit skips
    that shutdown. The output file is closed by then, and the standard streams are flushed here first.
    """
    code = 1
    try:
        code = entry()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    exit_hard(main)
