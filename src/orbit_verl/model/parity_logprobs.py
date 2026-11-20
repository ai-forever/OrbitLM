"""Compare the ported HF model's log-probs with vLLM's on the same completion tokens.

verl samples completions with vLLM (in-tree Ouro, fixed 4 loops, bf16) and trains the HF model
(the ported ``modelling_ouro.py``, final loop). Both must assign nearly the same log-prob to every
sampled token, or the rollout and the training forward disagree about the policy. This tool
measures that on a handful of prompts. It is run by hand on a GPU, in three phases, each in its
own process, so vLLM's GPU memory is gone before the HF model loads:

    $PY -m orbit_verl.model.parity_logprobs --phase vllm    --model-dir DIR --data train.parquet --out-dir OUT
    $PY -m orbit_verl.model.parity_logprobs --phase hf      --model-dir DIR --out-dir OUT
    $PY -m orbit_verl.model.parity_logprobs --phase compare --out-dir OUT

- ``vllm``: greedy generation from the first ``--n`` prompts. Writes ``vllm.json`` with the prompt
  ids, the completion ids, vLLM's log-prob of each completion token, and the text.
- ``hf``: recomputes the log-probs of exactly those completion tokens with the ported model on the
  final loop (``exit_at_step=T-1``), once as a right-padded batch and once as a single packed row
  with ``attention_mask=None`` and position ids restarting per sequence, which is how verl feeds
  the actor. Writes ``hf.json``.
- ``compare``: writes ``report.json`` and exits 1 when the mean absolute log-prob difference is
  above 0.05 or when the HF argmax matches the vLLM token at fewer than 95% of the positions.

Prompts come from a verl parquet made by ``orbit_verl.data.prepare_data`` (``--data``), tokenized
the way the rollout tokenizes them, or from a jsonl file of ``{"text": ...}`` rows
(``--prompts-jsonl``), tokenized as raw text without special tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from orbit_verl import common

# Pass thresholds of the compare phase (docs/verl/DESIGN.md).
MAX_MEAN_ABS_DLOGP = 0.05
MIN_GREEDY_AGREEMENT = 0.95

VLLM_FILE = "vllm.json"
HF_FILE = "hf.json"
REPORT_FILE = "report.json"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run the earlier phase first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_prompt_ids(tokenizer, *, data: str | None, prompts_jsonl: str | None, n: int) -> list[dict]:
    """Return the first ``n`` prompts as ``{"index", "source", "prompt_ids"}`` dicts."""
    prompts: list[dict] = []
    if prompts_jsonl:
        with open(prompts_jsonl, encoding="utf-8") as handle:
            for line in handle:
                if len(prompts) >= n:
                    break
                if not line.strip():
                    continue
                text = json.loads(line)["text"]
                ids = tokenizer.encode(text, add_special_tokens=False)
                prompts.append({"index": len(prompts), "source": "jsonl", "prompt_ids": ids})
        return prompts

    import pyarrow.parquet as pq

    # The prompts module defines how each prompt mode is tokenized. Using it here means the
    # parity run sees exactly the token ids the rollout sends to vLLM.
    from orbit_verl.data.prompts import PROMPT_MODES, prompt_token_ids

    rows = pq.read_table(data).slice(0, n).to_pylist()
    for row in rows:
        extra = row.get("extra_info") or {}
        mode = PROMPT_MODES[extra["prompt_mode"]]
        ids = prompt_token_ids(tokenizer, row["prompt"])
        prompts.append({"index": int(extra.get("index", len(prompts))), "source": mode.name, "prompt_ids": list(ids)})
    return prompts


def run_vllm(args: argparse.Namespace) -> int:
    common.apply_cache_env()
    from transformers import AutoTokenizer

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    prompts = load_prompt_ids(tokenizer, data=args.data, prompts_jsonl=args.prompts_jsonl, n=args.n)
    if not prompts:
        raise ValueError("No prompts were loaded.")
    max_model_len = int(json.loads((Path(model_dir) / "config.json").read_text())["max_position_embeddings"])
    too_long = [p["index"] for p in prompts if len(p["prompt_ids"]) + args.max_new_tokens > max_model_len]
    if too_long:
        raise ValueError(f"Prompts {too_long} plus {args.max_new_tokens} new tokens exceed max_model_len {max_model_len}.")

    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        seed=0,
    )
    # Greedy decoding, and logprobs=0 returns the log-prob of the sampled token only. vLLM's
    # default logprobs_mode reports raw model log-probs, which is what the HF phase computes.
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, logprobs=0)
    outputs = llm.generate([{"prompt_token_ids": p["prompt_ids"]} for p in prompts], params)

    records = []
    for prompt, output in zip(prompts, outputs):
        completion = output.outputs[0]
        ids = [int(t) for t in completion.token_ids]
        logprobs = [float(step[token].logprob) for token, step in zip(ids, completion.logprobs or [])]
        if len(logprobs) != len(ids):
            raise RuntimeError(f"vLLM returned {len(logprobs)} log-probs for {len(ids)} tokens.")
        records.append({
            **prompt,
            "completion_ids": ids,
            "completion_logprobs": logprobs,
            "finish_reason": completion.finish_reason,
            "text": completion.text,
        })
    out = Path(args.out_dir) / VLLM_FILE
    _write_json(out, {
        "model_dir": model_dir,
        "vllm_version": vllm.__version__,
        "enforce_eager": bool(args.enforce_eager),
        "max_new_tokens": args.max_new_tokens,
        "prompts": records,
    })
    print(f"[parity] vllm: {len(records)} prompts, {sum(len(r['completion_ids']) for r in records)} tokens -> {out}")
    return 0


def completion_logprobs(logits, start: int, completion_ids: list[int]):
    """Log-probs and argmax of the completion tokens of one sequence, at temperature 1.0.

    ``logits`` is ``[L, V]`` for the whole sequence. The token at position ``p`` is predicted by
    the logits at ``p - 1``, so a completion that starts at ``start`` uses rows
    ``start - 1 .. start + C - 2``.
    """
    import torch

    count = len(completion_ids)
    if count == 0:
        return [], []
    rows = logits[start - 1 : start - 1 + count].float().log_softmax(-1)
    target = torch.tensor(completion_ids, device=rows.device).unsqueeze(-1)
    return rows.gather(-1, target).squeeze(-1).tolist(), rows.argmax(-1).tolist()


def run_hf(args: argparse.Namespace) -> int:
    common.apply_cache_env()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    data = _read_json(Path(args.out_dir) / VLLM_FILE)
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    # transformers 5 renamed torch_dtype to dtype; 4.57 still expects the old name.
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, attn_implementation=args.attn, **{dtype_key: dtype}
    )
    model.to(device).eval()
    last = int(model.config.total_ut_steps) - 1
    # transformers 5 configs have no pad_token_id attribute unless the export sets one.
    pad_id = getattr(model.config, "pad_token_id", None) or 0

    prompts = data["prompts"]
    sequences = [p["prompt_ids"] + p["completion_ids"] for p in prompts]
    longest = max(len(s) for s in sequences)

    # Right-padded batch with an attention mask: the layout of a plain HF forward.
    input_ids = torch.full((len(sequences), longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), longest), dtype=torch.long)
    for row, seq in enumerate(sequences):
        input_ids[row, : len(seq)] = torch.tensor(seq)
        attention_mask[row, : len(seq)] = 1
    with torch.no_grad():
        padded_logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
            exit_at_step=last,
        ).logits

    # One packed row without a mask: the layout verl uses with use_remove_padding=True.
    packed_ids = torch.tensor([t for seq in sequences for t in seq], dtype=torch.long).unsqueeze(0)
    position_ids = torch.cat([torch.arange(len(seq)) for seq in sequences]).unsqueeze(0)
    with torch.no_grad():
        packed_logits = model(
            input_ids=packed_ids.to(device),
            position_ids=position_ids.to(device),
            attention_mask=None,
            use_cache=False,
            exit_at_step=last,
        ).logits[0]

    records = []
    offset = 0
    for row, prompt in enumerate(prompts):
        start = len(prompt["prompt_ids"])
        padded_lp, argmax = completion_logprobs(padded_logits[row], start, prompt["completion_ids"])
        packed_lp, _ = completion_logprobs(packed_logits[offset : offset + len(sequences[row])], start, prompt["completion_ids"])
        offset += len(sequences[row])
        records.append({
            "index": prompt["index"],
            "logprobs_padded": padded_lp,
            "logprobs_packed": packed_lp,
            "argmax_padded": argmax,
        })
    out = Path(args.out_dir) / HF_FILE
    _write_json(out, {
        "model_dir": model_dir,
        "device": args.device,
        "dtype": args.dtype,
        "attn": args.attn,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "exit_at_step": last,
        "prompts": records,
    })
    print(f"[parity] hf: {len(records)} prompts ({args.device}, {args.dtype}, {args.attn}) -> {out}")
    return 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def run_compare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    vllm_data = _read_json(out_dir / VLLM_FILE)
    hf_data = _read_json(out_dir / HF_FILE)
    if len(vllm_data["prompts"]) != len(hf_data["prompts"]):
        raise ValueError("vllm.json and hf.json have different prompt counts; re-run the hf phase.")

    diffs, packed_diffs, packed_vs_padded, agree = [], [], [], []
    per_prompt = []
    for vp, hp in zip(vllm_data["prompts"], hf_data["prompts"]):
        if vp["index"] != hp["index"] or len(vp["completion_logprobs"]) != len(hp["logprobs_padded"]):
            raise ValueError(f"Prompt {vp['index']} does not line up between vllm.json and hf.json.")
        d = [abs(h - v) for h, v in zip(hp["logprobs_padded"], vp["completion_logprobs"])]
        dk = [abs(h - v) for h, v in zip(hp["logprobs_packed"], vp["completion_logprobs"])]
        pp = [abs(a - b) for a, b in zip(hp["logprobs_packed"], hp["logprobs_padded"])]
        ok = [int(a == t) for a, t in zip(hp["argmax_padded"], vp["completion_ids"])]
        diffs += d
        packed_diffs += dk
        packed_vs_padded += pp
        agree += ok
        per_prompt.append({
            "index": vp["index"],
            "tokens": len(d),
            "mean_abs_dlogp": _mean(d),
            "max_abs_dlogp": max(d, default=0.0),
            "greedy_agreement": _mean(ok),
            "finish_reason": vp.get("finish_reason"),
            "text": vp.get("text", ""),
        })

    tokens = len(diffs)
    mean_abs = _mean(diffs)
    agreement = _mean(agree)
    passed = tokens > 0 and mean_abs <= MAX_MEAN_ABS_DLOGP and agreement >= MIN_GREEDY_AGREEMENT
    report = {
        "passed": passed,
        "prompts": len(per_prompt),
        "tokens": tokens,
        "mean_abs_dlogp": mean_abs,
        "max_abs_dlogp": max(diffs, default=0.0),
        "mean_abs_dlogp_packed": _mean(packed_diffs),
        "max_abs_dlogp_packed": max(packed_diffs, default=0.0),
        "max_abs_packed_vs_padded": max(packed_vs_padded, default=0.0),
        "greedy_agreement": agreement,
        "thresholds": {"max_mean_abs_dlogp": MAX_MEAN_ABS_DLOGP, "min_greedy_agreement": MIN_GREEDY_AGREEMENT},
        "vllm": {k: vllm_data.get(k) for k in ("model_dir", "vllm_version", "enforce_eager", "max_new_tokens")},
        "hf": {k: hf_data.get(k) for k in ("model_dir", "device", "dtype", "attn", "transformers_version")},
        "per_prompt": per_prompt,
    }
    _write_json(out_dir / REPORT_FILE, report)
    for item in per_prompt:
        print(f"[parity] prompt {item['index']}: tokens {item['tokens']} mean|d| {item['mean_abs_dlogp']:.4f} "
              f"max|d| {item['max_abs_dlogp']:.4f} agree {item['greedy_agreement']:.3f} text {item['text'][:60]!r}")
    print(f"[parity] tokens {tokens} mean|dlogp| {mean_abs:.4f} max|dlogp| {report['max_abs_dlogp']:.4f} "
          f"packed-vs-padded max {report['max_abs_packed_vs_padded']:.2e} greedy agreement {agreement:.3f} "
          f"-> {'PASS' if passed else 'FAIL'} ({out_dir / REPORT_FILE})")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orbit_verl.model.parity_logprobs",
        description="HF (ported) vs vLLM completion log-prob parity, one phase per process.",
    )
    parser.add_argument("--phase", required=True, choices=["vllm", "hf", "compare"])
    parser.add_argument("--model-dir", help="prepared model dir (vllm and hf phases)")
    parser.add_argument("--out-dir", required=True, help="where vllm.json, hf.json and report.json go")
    parser.add_argument("--runs-root", help=f"root of the compiler and model-code caches "
                        f"(default: ${common.RUNS_ROOT_ENV})")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", help="verl parquet from orbit_verl.data.prepare_data (vllm phase)")
    source.add_argument("--prompts-jsonl", help='jsonl with {"text": ...} raw prompts (vllm phase)')
    parser.add_argument("--n", type=int, default=8, help="number of prompts (default %(default)s)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--enforce-eager", action="store_true", help="vLLM without CUDA graphs")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="hf phase")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="hf phase")
    parser.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "sdpa"], help="hf phase")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.phase in ("vllm", "hf") and not args.model_dir:
        parser.error("--model-dir is required for the vllm and hf phases")
    if args.phase == "vllm" and not (args.data or args.prompts_jsonl):
        parser.error("the vllm phase needs --data or --prompts-jsonl")
    if args.phase != "compare":
        # Only the phases that load a model need the caches; compare reads two JSON files.
        try:
            runs_root = common.resolve_runs_root(args.runs_root)
        except common.RunsRootError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        # The phases set the caches again without a root; they then find this one.
        os.environ[common.RUNS_ROOT_ENV] = runs_root
        # Set the cache variables before the phase imports transformers or vllm.
        common.apply_cache_env(runs_root)
    return {"vllm": run_vllm, "hf": run_hf, "compare": run_compare}[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
