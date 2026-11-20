"""Turns verl val-only generations into per-prompt pass rates, and merges the shards of one measurement.

This is the aggregator of the math kinds of passrate.sh (prompt modes sft_think and sft_think_ru).
passrate_agg_code.py is its counterpart for code problems.

Commands:

  shard  reads one run's val_generations/<step>.jsonl (written by verl's validation: input, output,
         gts, score and the reward's extra keys acc, answer_correct, truncated, think_unclosed),
         groups the samples by prompt, and writes <run dir>/passrate.jsonl and passrate_summary.json.
         It also reads the timestamped launch.log that passrate.sh writes, to split the wall time
         into start-up and generation, and the vLLM stats lines for the KV cache use.
  merge  concatenates the per-shard results of one tag into results/<tag>/passrate.jsonl, writes
         summary.json, and writes keep.jsonl: the source rows (the input format plus the pass-rate
         fields) whose number of correct samples lies within [keep_min, keep_max].

Prompts are matched to the subset by their question text. verl decodes the prompt with
skip_special_tokens=True, so the dumped input is "user\\n<question>\\n<instruction>\\nassistant\\n<think>\\n"
for the sft_think mode. The question is cut out of that frame and looked up in the subset. The
sft_think_ru mode has the Russian instruction in its frame instead, so ``--prompt-mode`` picks the
frame. "<think>" stays in the decoded text because it is not a special token of the Qwen3 tokenizer.
Response lengths are counted by re-encoding the decoded output with the model's tokenizer. That
drops the end-of-turn token, so a finished response counts one token less than vLLM generated.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import statistics
import sys
import unicodedata

PREFIX = "user\n"
# The decoded prompt frame after the question, per prompt mode. Each is the template of
# configs/verl/prompts/<mode>.yaml as verl decodes it: the special tokens are dropped.
SUFFIXES = {
    "sft_think": "\nPlease reason step by step, and put your final answer within \\boxed{}.\nassistant\n<think>\n",
    "sft_think_ru": "\nРассуждай пошагово и запиши окончательный ответ в \\boxed{}.\nassistant\n<think>\n",
}


def question_of(text: str, suffix: str = SUFFIXES["sft_think"]) -> str | None:
    """The question inside a decoded prompt of the given frame, or None when the frame does not fit."""
    if text.startswith(PREFIX) and text.endswith(suffix):
        return text[len(PREFIX) : -len(suffix)]
    return None


def histogram(rows: list[dict], n: int) -> dict:
    counts = collections.Counter(r["n_correct"] for r in rows)
    total = len(rows)
    if not total:
        return {"prompts": 0}
    return {
        "prompts": total,
        "k_hist": [counts.get(k, 0) for k in range(n + 1)],
        "zero_frac": round(counts.get(0, 0) / total, 4),
        "all_frac": round(counts.get(n, 0) / total, 4),
        "mixed_frac": round(1 - (counts.get(0, 0) + counts.get(n, 0)) / total, 4),
        "mean_pass_rate": round(sum(r["pass_rate"] for r in rows) / total, 4),
        "mean_response_len": round(sum(r["mean_response_len"] for r in rows) / total, 1),
        "truncated_frac": round(sum(r["truncated_frac"] for r in rows) / total, 4),
        "think_unclosed_frac": round(sum(r["think_unclosed_frac"] for r in rows) / total, 4),
    }


def parse_log(path: str) -> dict:
    """Wall-time split and progress from a launch.log whose lines start with '<epoch seconds>\\t'.

    verl 0.9.1 validates with its V1 trainer. It logs "all initialize finished, ready to fit" when
    start-up is over, then the validation requests go to vLLM, and the replay buffer logs a
    "pending: P, running: R, finished: F, failure: X" line about once a minute, counted in prompts
    (whole groups of n samples). The generations are dumped right after the last group finishes,
    so generation time is taken from the end of start-up to the "Dumped generations to" line.
    With VLLM_LOGGING_LEVEL=INFO, vLLM also logs its KV cache size and a throughput line every 10 s.
    """
    out: dict = {}
    if not os.path.isfile(path):
        return out
    first = last = ready = dumped = None
    progress = []
    stats = []
    kv_tokens = []
    stat_re = re.compile(r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
                         r"GPU KV cache usage: ([\d.]+)%")
    kv_re = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
    prog_re = re.compile(r"pending: (\d+), running: (\d+), finished: (\d+), failure: (\d+)")
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stamp, _, text = line.partition("\t")
            try:
                t = float(stamp)
            except ValueError:
                continue
            first = t if first is None else first
            last = t
            if ready is None and "all initialize finished, ready to fit" in text:
                ready = t
            elif "Dumped generations to" in text:
                dumped = t
            m = prog_re.search(text)
            if m:
                progress.append((t, *(int(x) for x in m.groups())))
            m = stat_re.search(text)
            if m:
                stats.append((t, float(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))))
            m = kv_re.search(text)
            if m:
                kv_tokens.append(int(m.group(1).replace(",", "")))
    out["wall_s"] = round(last - first, 1) if first is not None else None
    if ready is not None:
        out["init_s"] = round(ready - first, 1)
    if ready is not None and dumped is not None:
        out["generation_s"] = round(dumped - ready, 1)
        out["after_generation_s"] = round(last - dumped, 1)
    if progress:
        out["progress"] = [[round(t - (ready or first)), p, r, f, x] for t, p, r, f, x in progress]
    if kv_tokens:
        out["vllm_kv_cache_tokens_per_engine"] = kv_tokens
    if stats:
        # Only stats lines with waiting requests count as steady state: the engine then had all
        # the work it could take, as it has for most of a large shard.
        steady = [x for x in stats if x[3] > 0]
        out["vllm_stats_lines"] = len(stats)
        out["gen_tok_s_per_engine_median_all"] = round(statistics.median(x[1] for x in stats), 1)
        if steady:
            out["steady_gen_tok_s_per_engine_median"] = round(statistics.median(x[1] for x in steady), 1)
            out["steady_running_median"] = statistics.median(x[2] for x in steady)
            out["steady_kv_usage_pct_median"] = statistics.median(x[4] for x in steady)
            out["steady_lines"] = len(steady)
    return out


def shard_cmd(args: argparse.Namespace) -> int:
    subset = {}
    with open(args.subset, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                # The tokenizer round trip applies Unicode NFC (for example U+2329 becomes U+3008), so the
                # decoded question can differ from the source text. Keys are stored in NFC as well.
                subset[row["instruction"]] = row
                subset.setdefault(unicodedata.normalize("NFC", row["instruction"]), row)
    files = sorted(glob.glob(os.path.join(args.run_dir, "val_generations", "*.jsonl")))
    if len(files) != 1:
        print(f"passrate_agg: expected one val_generations file in {args.run_dir}, found {files}", file=sys.stderr)
        return 2
    groups: dict[str, list[dict]] = collections.OrderedDict()
    unframed = 0
    with open(files[0], encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question = question_of(row["input"], SUFFIXES[args.prompt_mode])
            if question is None:
                unframed += 1
                question = row["input"]
            groups.setdefault(question, []).append(row)

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    questions = list(groups)
    lengths: dict[str, list[int]] = {}
    if tokenizer is not None:
        # Outputs are encoded in chunks and only their lengths are kept. A shard of 16K samples at
        # 10K tokens each would otherwise hold 160M token ids in Python lists at once.
        batch = [s["output"] for q in questions for s in groups[q]]
        flat: list[int] = []
        for start in range(0, len(batch), 256):
            ids = tokenizer(batch[start : start + 256], add_special_tokens=False)["input_ids"]
            flat += [len(x) for x in ids]
        pos = 0
        for q in questions:
            lengths[q] = flat[pos : pos + len(groups[q])]
            pos += len(groups[q])

    out_rows, unmatched = [], 0
    total_tokens = 0
    for q in questions:
        samples = groups[q]
        src = subset.get(q) or subset.get(unicodedata.normalize("NFC", q))
        if src is None:
            unmatched += 1
        n = len(samples)
        correct = sum(float(s["score"]) >= 1.0 for s in samples)
        lens = lengths.get(q) or [len(s["output"]) for s in samples]
        total_tokens += sum(lens)
        out_rows.append({
            "unique_id": src["unique_id"] if src else None,
            "prompt_sha1": hashlib.sha1(q.encode()).hexdigest(),
            "source": src.get("source") if src else None,
            "n": n,
            "n_correct": int(correct),
            "pass_rate": round(correct / n, 4),
            "answer_correct_frac": round(sum(float(s.get("answer_correct", 0)) for s in samples) / n, 4),
            "mean_response_len": round(sum(lens) / n, 1),
            "max_response_len": max(lens),
            "truncated_frac": round(sum(float(s.get("truncated", 0)) for s in samples) / n, 4),
            "think_unclosed_frac": round(sum(float(s.get("think_unclosed", 0)) for s in samples) / n, 4),
            "len_unit": "tokens" if tokenizer is not None else "chars",
        })
    ns = collections.Counter(r["n"] for r in out_rows)
    n_main = ns.most_common(1)[0][0] if ns else 0
    missing = len(subset) - (len(out_rows) - unmatched)
    timing = parse_log(os.path.join(args.run_dir, "launch.log"))
    summary = {
        "run_dir": args.run_dir,
        "val_generations": files[0],
        "samples": sum(r["n"] for r in out_rows),
        "prompts": len(out_rows),
        "n_per_prompt": dict(ns),
        "subset_rows": len(subset),
        "subset_rows_not_measured": missing,
        "unmatched_prompts": unmatched,
        "unframed_samples": unframed,
        "response_tokens_total": total_tokens,
        **histogram(out_rows, n_main),
        "by_source": {s: histogram([r for r in out_rows if r["source"] == s], n_main)
                      for s in sorted({str(r["source"]) for r in out_rows})},
        "timing": timing,
    }
    if timing.get("generation_s"):
        summary["gen_tok_s_node"] = round(total_tokens / timing["generation_s"], 1)
        summary["prompts_per_hour_generation"] = round(len(out_rows) / timing["generation_s"] * 3600, 1)
    if timing.get("wall_s"):
        summary["prompts_per_hour_wall"] = round(len(out_rows) / timing["wall_s"] * 3600, 1)
    with open(os.path.join(args.run_dir, "passrate.jsonl"), "w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(args.run_dir, "passrate_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_source"}, indent=1))
    return 0 if unmatched == 0 else 3


def merge_cmd(args: argparse.Namespace) -> int:
    rows, summaries = [], []
    for run_dir in args.run_dirs:
        with open(os.path.join(run_dir, "passrate.jsonl"), encoding="utf-8") as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
        with open(os.path.join(run_dir, "passrate_summary.json"), encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    subset = {}
    with open(args.subset, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                subset[row["unique_id"]] = row
    n = collections.Counter(r["n"] for r in rows).most_common(1)[0][0]
    keep_max = n - 1 if args.keep_max is None else args.keep_max
    os.makedirs(args.out_dir, exist_ok=True)
    kept = 0
    with open(os.path.join(args.out_dir, "passrate.jsonl"), "w", encoding="utf-8") as all_out, \
            open(os.path.join(args.out_dir, "keep.jsonl"), "w", encoding="utf-8") as keep_out:
        for r in rows:
            all_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["unique_id"] in subset and args.keep_min <= r["n_correct"] <= keep_max:
                src = dict(subset[r["unique_id"]])
                src.update({k: r[k] for k in ("n", "n_correct", "pass_rate", "mean_response_len", "truncated_frac")})
                keep_out.write(json.dumps(src, ensure_ascii=False) + "\n")
                kept += 1
    gen_s = sum(s["timing"].get("generation_s") or 0 for s in summaries)
    wall_s = sum(s["timing"].get("wall_s") or 0 for s in summaries)
    tokens = sum(s["response_tokens_total"] for s in summaries)
    summary = {
        "shards": len(summaries),
        "subset_rows": len(subset),
        "keep_rule": f"{args.keep_min} <= n_correct <= {keep_max} of n={n}",
        "kept": kept,
        **histogram(rows, n),
        "by_source": {s: histogram([r for r in rows if r["source"] == s], n) for s in sorted({str(r["source"]) for r in rows})},
        "generation_s": round(gen_s, 1),
        "wall_s": round(wall_s, 1),
        "response_tokens_total": tokens,
        "gen_tok_s_node": round(tokens / gen_s, 1) if gen_s else None,
        "prompts_per_hour_wall": round(len(rows) / wall_s * 3600, 1) if wall_s else None,
    }
    summary["kept_frac"] = round(kept / len(rows), 4) if rows else None
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_source"}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="passrate_agg.py")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--subset", required=True)
    s.add_argument("--tokenizer", default=None, help="model dir; without it lengths are in chars")
    s.add_argument("--prompt-mode", default="sft_think", choices=sorted(SUFFIXES),
                   help="the prompt mode of the run, which fixes the frame around each question")
    s.set_defaults(func=shard_cmd)
    m = sub.add_parser("merge")
    m.add_argument("--subset", required=True)
    m.add_argument("--out-dir", required=True)
    m.add_argument("--keep-min", type=int, default=1)
    m.add_argument("--keep-max", type=int, default=None, help="default n - 1")
    m.add_argument("run_dirs", nargs="+")
    m.set_defaults(func=merge_cmd)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
