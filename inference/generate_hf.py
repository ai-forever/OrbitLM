"""Reference generation with HF transformers on one GPU, for debugging and parity checks against vLLM.

    python inference/generate_hf.py MODEL_DIR --prompt "What is 17 times 23?" --temperature 0

It renders the prompts, picks the loops and forces the thinking budget exactly as generate_vllm.py
does, but it generates one prompt at a time with the model code that the export bundles. With
--temperature 0 its output follows the greedy output of generate_vllm.py --kv-cache-dtype auto
(bf16 KV cache, no --speculative) token for token until the two engines meet a near-tie. In bf16 the
two best logits are often equal or one rounding step apart, and the engines' different attention
kernels may then pick different tokens. Both choices are correct greedy outputs.

The model code that this repository puts into an export works with transformers 4.56 and later, up to
5.23, the newest version checked. Exports made before that code changed stop working at 5.2; copy
src/orbit/modelling_ouro.py over the export's file to run them on a newer version. The script runs in
the trainer environment (requirements/train.txt) or in any environment within that range, and it
refuses other versions instead of failing somewhere inside generate.
"""
from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Optional, Sequence

if __package__ in (None, ""):
    # Run as a file: put the repo root on sys.path, so that the inference package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import cli, runtime_env  # noqa: E402
from inference.budget import Completion, GenerateFn  # noqa: E402
from inference.prompts import PromptError  # noqa: E402
from orbit_verl.data.response_info import classify_finish  # noqa: E402

# transformers 4.55 lacks a cache setter the model code needs, and 5.2 renamed an argument of
# create_causal_mask that the model code passes.
TRANSFORMERS_MIN = (4, 56)
TRANSFORMERS_BELOW = (5, 24)


def transformers_problem(version: str) -> Optional[str]:
    """Why this transformers version cannot run the model code, or None when it can."""
    from packaging.version import Version

    release = Version(version).release[:2]
    if TRANSFORMERS_MIN <= release < TRANSFORMERS_BELOW:
        return None
    return (f"transformers {version} cannot run the OrbitLM model code; it needs >= "
            f"{'.'.join(map(str, TRANSFORMERS_MIN))} and < {'.'.join(map(str, TRANSFORMERS_BELOW))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_shared_args(parser)
    engine = parser.add_argument_group("HF transformers")
    engine.add_argument("--device", default="cuda", help="torch device (default cuda)")
    engine.add_argument("--attn", choices=("sdpa", "eager"),
                        help="attention implementation (default: the one in the export's config, normally sdpa)")
    return parser


def generate_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    """The sampling arguments of model.generate for cli.sampling_settings.

    Greedy decoding also clears temperature, top_p and top_k, because an export's generation_config may
    set them, and transformers warns about every one of them that is set while sampling is off.
    """
    if settings["temperature"] == 0:
        return {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    return {"do_sample": True, "temperature": settings["temperature"], "top_p": settings["top_p"],
            "top_k": settings["top_k"] or 0}


class HFGenerate:
    """A GenerateFn over model.generate that runs one prompt at a time.

    exit_at_step picks the loop: the model runs loops 0 to exit_at_step and answers from the last one,
    which is what vLLM does with total_ut_steps set to the same number of loops. One prompt at a time
    needs no padding, so an output never depends on which other prompts are in the run. This path is
    for checks, so its speed does not matter.
    """

    def __init__(self, model: Any, *, loops: int, end_ids: Sequence[int], pad_id: int,
                 sampling: dict[str, Any], device: str):
        self.model = model
        self.loops = loops
        self.end_ids = tuple(end_ids)
        self.pad_id = pad_id
        self.sampling = generate_kwargs(sampling)
        self.device = device

    def __call__(self, prompt_ids: list[list[int]], limits: list[int]) -> list[Completion]:
        import torch

        completions = []
        for ids, limit in zip(prompt_ids, limits):
            input_ids = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=limit,
                    eos_token_id=list(self.end_ids),
                    pad_token_id=self.pad_id,
                    exit_at_step=self.loops - 1,
                    **self.sampling,
                )
            new = [int(token) for token in output[0, input_ids.shape[1]:].tolist()]
            finish = classify_finish(new, stopped_at_string=False, max_new_tokens=limit, end_token_ids=self.end_ids)
            completions.append(Completion(tuple(new), "length" if finish == "length" else "stop"))
        return completions


def start_engine(args: argparse.Namespace, tokenizer: Any, end_ids: Sequence[int]) -> GenerateFn:
    import torch
    from transformers import AutoModelForCausalLM

    # The exports carry early_exit_threshold=1.0, and with it the trained exit gate would pick the loop.
    # It is turned off, so that exit_at_step alone picks the loop. The config field early_exit_step must
    # not be passed here, because from_pretrained crashes on it.
    kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "trust_remote_code": True, "early_exit_threshold": None}
    if args.attn:
        kwargs["attn_implementation"] = args.attn
    model = AutoModelForCausalLM.from_pretrained(str(args.model), **kwargs)
    attention = getattr(model.config, "_attn_implementation", None)
    if str(args.device).startswith("cuda") and attention == "sdpa":
        # PyTorch's cuDNN attention kernel fails on long contexts in some CUDA environments ("No valid
        # execution plans built"). Without it, SDPA runs its flash or memory-efficient kernel, which
        # computes the same attention.
        torch.backends.cuda.enable_cudnn_sdp(False)
        cli.log("attention: sdpa without the cuDNN kernel")
    else:
        cli.log(f"attention: {attention}")
    model.to(args.device).eval()
    torch.manual_seed(args.seed)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end_ids[-1]
    return HFGenerate(model, loops=args.loops, end_ids=end_ids, pad_id=int(pad_id),
                      sampling=cli.sampling_settings(args), device=args.device)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        problem = transformers_problem(metadata.version("transformers"))
        if problem:
            raise cli.UsageError(problem)
        cli.check_shared_args(args)
        notes = cli.resolve_model_args(args, cli.read_model_config(args.model))
    except cli.UsageError as exc:
        parser.error(str(exc))
    for note in notes + runtime_env.prepare_process(for_vllm=False):
        cli.log(note)
    try:
        return cli.run(args, start_engine)
    except (cli.UsageError, PromptError) as exc:
        cli.log(f"error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
