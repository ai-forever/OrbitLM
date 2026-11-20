"""The command line and the run that generate_vllm.py and generate_hf.py share.

Each generator adds its engine's options and passes a start_engine function to run(). Everything else
is the same for both engines: argument checks, prompt rendering, budget forcing and the output records.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TextIO

from inference import prompts
from inference.budget import (
    DEFAULT_ANSWER_BUDGET,
    DEFAULT_THINK_BUDGET,
    GenerateFn,
    Response,
    generate_responses,
    make_budget,
    split_response,
)
import itr_render  # noqa: E402  (on sys.path through the inference package)
from orbit_verl.data.response_info import think_token_ids

DEFAULT_MAX_NEW_TOKENS = 16384
# The context of the SFT and RL checkpoints, used when a config.json does not name one.
DEFAULT_CONTEXT = 32768
# The recommended sampling of the SFT and RL models in think mode. Pass --temperature 0 for greedy checks.
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20

StartEngine = Callable[[argparse.Namespace, Any, Sequence[int]], GenerateFn]


class UsageError(ValueError):
    """The arguments do not make a valid run."""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", type=Path, help="HF export dir of a OrbitLM checkpoint (a chat export for think and no_think)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompts", type=Path, help="a .jsonl file ('prompt' or 'messages', optional 'id' and 'lang') or a .txt file with one prompt per line")
    source.add_argument("--prompt", action="append", help="a prompt given on the command line; may be repeated")
    parser.add_argument("--output", type=Path,
                        help="output .jsonl, one record per prompt (default: standard output, which then carries "
                             "nothing but the records)")

    prompt = parser.add_argument_group("prompt format")
    prompt.add_argument("--mode", choices=prompts.MODES, default="think",
                        help="think: open thinking block (default); no_think: empty block; raw: plain text without chat format")
    prompt.add_argument("--no-lead-eos", action="store_true",
                        help="do not start a think or no_think prompt with <|endoftext|>, which the SFT evaluation "
                             "puts there; raw prompts never get it")
    prompt.add_argument("--lang", choices=("auto",) + prompts.LANGUAGES, default="auto",
                        help="prompt language for the stop-thinking sentence; auto takes the file's 'lang' field or detects it")

    generation = parser.add_argument_group("generation")
    generation.add_argument("--loops", type=int, help="loops to run, from 1 to the model's total_ut_steps (default: all of them)")
    generation.add_argument("--max-new-tokens", type=int, help=f"token limit without budget forcing (default {DEFAULT_MAX_NEW_TOKENS})")
    generation.add_argument("--max-model-len", type=int, help="context length (default: max_position_embeddings of config.json)")
    generation.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="0 is greedy")
    generation.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    generation.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="0 turns top-k off")
    generation.add_argument("--seed", type=int, default=0)

    budget = parser.add_argument_group("budget forcing (think mode)")
    budget.add_argument("--budget-forcing", action="store_true",
                        help="close the thinking block by force at --think-budget and answer in --answer-budget more tokens")
    budget.add_argument("--think-budget", type=int, help=f"tokens of thinking before the forced close (default {DEFAULT_THINK_BUDGET})")
    budget.add_argument("--answer-budget", type=int, help=f"tokens for the answer after the budget (default {DEFAULT_ANSWER_BUDGET})")


def check_shared_args(args: argparse.Namespace) -> None:
    """Checks that need nothing but the arguments. Raises UsageError."""
    if args.temperature < 0:
        raise UsageError("--temperature must not be negative")
    if not 0.0 < args.top_p <= 1.0:
        raise UsageError("--top-p must be in (0, 1]")
    if args.top_k < 0:
        raise UsageError("--top-k must not be negative")
    if args.loops is not None and args.loops < 1:
        raise UsageError("--loops starts at 1")
    for name in ("max_new_tokens", "max_model_len", "think_budget", "answer_budget"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise UsageError(f"--{name.replace('_', '-')} must be positive")
    if args.budget_forcing:
        if args.mode != "think":
            raise UsageError("--budget-forcing needs --mode think, because only a thinking block can be closed by force")
        if args.max_new_tokens is not None:
            raise UsageError("with --budget-forcing the limits are --think-budget and --answer-budget, not --max-new-tokens")
    elif args.think_budget is not None or args.answer_budget is not None:
        raise UsageError("--think-budget and --answer-budget take effect only with --budget-forcing")


def read_model_config(model_dir: Path) -> dict[str, Any]:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise UsageError(f"{model_dir} has no config.json, so it is not an HF export")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_args(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    """Fill in the defaults that depend on the model's config.json, check them, and return notes to print.

    It sets args.model_loops (the model's total_ut_steps), args.loops, args.max_model_len,
    args.max_new_tokens and the two budgets. Raises UsageError.
    """
    notes = []
    args.model_loops = int(config.get("total_ut_steps") or 1)
    if args.loops is None:
        args.loops = args.model_loops
    if not 1 <= args.loops <= args.model_loops:
        raise UsageError(f"--loops must be from 1 to the model's total_ut_steps, {args.model_loops}")
    if args.loops == 1 and args.model_loops > 1:
        notes.append("note: a looped model run with one loop answers poorly; two loops is the useful minimum")
    if args.max_model_len is None:
        args.max_model_len = int(config.get("max_position_embeddings") or DEFAULT_CONTEXT)
    if args.budget_forcing:
        args.think_budget = args.think_budget or DEFAULT_THINK_BUDGET
        args.answer_budget = args.answer_budget or DEFAULT_ANSWER_BUDGET
        if args.think_budget + args.answer_budget >= args.max_model_len:
            raise UsageError(
                f"--think-budget {args.think_budget} plus --answer-budget {args.answer_budget} leaves no room for "
                f"the prompt in a context of {args.max_model_len}"
            )
    elif args.max_new_tokens is None:
        args.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    return notes


def sampling_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Temperature, top_p and top_k as both engines need them: greedy at temperature 0, top_k None when off."""
    if args.temperature == 0:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": None}
    return {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k or None}


def build_record(item: prompts.PromptItem, lang: str, prompt_ids: Sequence[int], response: Response,
                 args: argparse.Namespace, tokenizer: Any, think_ids: Optional[tuple[int, int]],
                 end_ids: Sequence[int]) -> dict[str, Any]:
    """The output record of one prompt.

    thinking and answer leave out the end tokens and the inserted stop-thinking sentence. text is the whole
    response with special tokens, including that sentence, which is what parity checks compare.
    """
    parts = split_response(response, args.mode, think_ids, end_ids)

    def decode(ids: Sequence[int]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=False)

    return {
        "id": item.id,
        "lang": lang,
        "mode": args.mode,
        "loops": args.loops,
        "prompt": item.user_text if len(item.messages) == 1 else list(item.messages),
        "prompt_tokens": len(prompt_ids),
        "thinking": decode(parts.thinking_ids).strip(),
        "answer": decode(parts.answer_ids).strip(),
        "think": parts.think,
        "forced_close": response.forced_close,
        "continued_after_budget": response.continued,
        "finish_reason": response.finish_reason,
        "completion_tokens": response.generated_tokens,
        "text": decode(response.token_ids),
    }


def check_prompt_lengths(items: Sequence[prompts.PromptItem], prompt_ids: Sequence[Sequence[int]],
                         max_model_len: int) -> None:
    """Refuse a prompt that leaves no room in the context for a single new token. Raises PromptError."""
    for item, ids in zip(items, prompt_ids):
        if len(ids) >= max_model_len:
            raise prompts.PromptError(
                f"prompt {item.id} has {len(ids)} tokens, which fills the context of {max_model_len}"
            )


def reserve_stdout() -> TextIO:
    """Keep standard output for the records alone, and return the stream to write them to.

    vLLM's logger writes to standard output, and so do the engine processes it starts and some other
    libraries. File descriptor 1 is therefore pointed at standard error, which this process and every
    process it starts from now on inherit. The records go to a copy of the original descriptor, so a
    plain "> answers.jsonl" receives clean JSONL.
    """
    sys.stdout.flush()
    records = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    return records


def write_records(records: Sequence[dict[str, Any]], output: Optional[Path], stream: Optional[TextIO] = None) -> None:
    """Write the records to the output file, or to stream when there is no output file."""
    lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    if output is None:
        stream = sys.stdout if stream is None else stream
        stream.write(lines)
        stream.flush()
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(lines, encoding="utf-8")
    log(f"wrote {len(records)} records to {output}")


def summary_lines(records: Sequence[dict[str, Any]]) -> list[str]:
    def counts(key: str) -> str:
        return ", ".join(f"{value} {count}" for value, count in sorted(collections.Counter(str(r[key]) for r in records).items()))

    tokens = sum(record["completion_tokens"] for record in records)
    return [
        f"prompts {len(records)}, generated tokens {tokens} (mean {tokens / max(1, len(records)):.0f})",
        f"finish: {counts('finish_reason')}; think: {counts('think')}; language: {counts('lang')}",
        f"forced close {sum(r['forced_close'] for r in records)}, continued after the budget "
        f"{sum(r['continued_after_budget'] for r in records)}",
    ]


def load_tokenizer(model_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)


def run(args: argparse.Namespace, start_engine: StartEngine, tokenizer_loader: Callable[[Path], Any] = load_tokenizer,
        records_stream: Optional[TextIO] = None) -> int:
    """Render the prompts, start the engine, generate, and write one record per prompt.

    The prompts are rendered and checked, their length included, before the engine starts. So a bad
    prompt file fails in seconds instead of after the engine has loaded. args must have passed
    check_shared_args and resolve_model_args.

    Without --output the records go to standard output, which reserve_stdout keeps free of everything
    else. Tests pass records_stream instead, so that the process's real standard output is left alone.
    """
    if args.output is None and records_stream is None:
        records_stream = reserve_stdout()
    tokenizer = tokenizer_loader(args.model)
    items = prompts.load_prompts(args.prompts, args.prompt)
    end_ids = prompts.end_token_ids(tokenizer)
    lead_eos = not args.no_lead_eos
    contract = None if args.mode == "raw" else itr_render.expected_ids(tokenizer)
    prompt_ids = [prompts.render_ids(tokenizer, item, args.mode, lead_eos, contract) for item in items]
    check_prompt_lengths(items, prompt_ids, args.max_model_len)
    langs = [prompts.prompt_language(item, args.lang) for item in items]
    think_ids = think_token_ids(tokenizer)
    if args.mode == "think" and think_ids is None:
        raise UsageError("think mode needs a tokenizer with single-token <think> and </think>")
    budget = make_budget(tokenizer, args.think_budget, args.answer_budget) if args.budget_forcing else None
    limits = (f"think budget {args.think_budget} + answer budget {args.answer_budget}" if budget
              else f"max new tokens {args.max_new_tokens}")
    log(f"{len(items)} prompts, mode {args.mode}, loops {args.loops} of {args.model_loops}, {limits}, "
        f"context {args.max_model_len}, sampling {sampling_settings(args)}")

    generate = start_engine(args, tokenizer, end_ids)
    responses = generate_responses(prompt_ids, langs, generate, max_new_tokens=args.max_new_tokens,
                                   max_model_len=args.max_model_len, budget=budget)
    records = [build_record(item, lang, ids, response, args, tokenizer, think_ids, end_ids)
               for item, lang, ids, response in zip(items, langs, prompt_ids, responses)]
    write_records(records, args.output, records_stream)
    for line in summary_lines(records):
        log(line)
    return 0
