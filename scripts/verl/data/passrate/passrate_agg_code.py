"""Turns verl val-only generations of code problems into per-problem pass rates, and merges the shards.

This is the code twin of passrate_agg.py. ``passrate.sh --kind code`` calls it the same way.

Commands:

  shard  reads one run's val_generations/<step>.jsonl (written by verl's validation: input, output,
         gts, score, uid and the reward's extra keys acc, answer_correct, truncated, think_unclosed
         and the code_* keys of orbit_verl.code_sandbox.REWARD_INFO_KEYS), groups the samples by
         problem, matches every problem to its subset row, and writes <run dir>/passrate.jsonl and
         passrate_summary.json. It also reads the timestamped launch.log that passrate.sh writes,
         to split the wall time into start-up and generation, and the vLLM stats lines.
  merge  concatenates the per-shard results of one tag into results/<tag>/passrate.jsonl,
         writes summary.json, and writes keep.jsonl: the source rows (code_data format plus the
         pass-rate fields) whose number of correct samples lies within [keep_min, keep_max].

What counts as correct. The orbit/code reward (orbit_verl.data.reward) is binary. It is 1 when the
last ```python block after </think> passes every test of the problem in the sandbox, and 0 otherwise.
A response cut by the length limit scores 0 (truncated_score), and so does a response that never
closes its thinking block. A sample is therefore correct when its score is 1. There is no partial
credit: answer_correct is the same check without the truncation rule, and the code_* keys only say
why a program failed.

How a problem finds its subset row, without knowing the prompt template. verl decodes the prompt
with skip_special_tokens=True, so the dumped input is the rendered prompt minus its special tokens,
with the tokenizer's Unicode NFC normalisation applied. The statement text of the subset row is
therefore a substring of that input once both are in NFC and the special-token strings are removed
from the statement. The match uses two independent facts:

1. gts is the row's ground truth, which prepare_data copies unchanged from the row's "response"
   (the tests as JSON). Rows are indexed by a hash of it, so the lookup is one dict access.
2. The candidate's statement must be contained in the decoded input.

A problem matches when exactly one row passes both. Rows that are exact copies of each other (the
same statement and the same tests) count as one, the first in file order, because prepare_data
also keeps only the first copy. When the dump has no gts, the statement alone decides, through an
index of each statement's first characters. When several rows fit because one statement contains
another, the prompt frame that the other problems of the shard share picks the row whose statement
is exactly what the frame leaves. A problem that still fits several rows, or none, or whose
statement fits a row with other tests, is refused: its row gets unique_id null, the summary lists
it, and the exit code is 3.

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
import sys
import unicodedata

# The launch log is read exactly as the math aggregator reads it, so both tools report the same timing.
# passrate.sh runs the aggregators with python -I, which leaves this file's directory off sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from passrate_agg import parse_log  # noqa: E402

# The reward is 0 or 1. The tolerance only absorbs float rounding of the dumped score.
SOLVED = 1.0 - 1e-6
# Statements at least this long are found through an index of their first characters. Shorter ones
# are few, and each is searched in the input directly.
ANCHOR = 48
# The reward keys that mean the program tried to fool the judge rather than solve the problem. A
# non-zero share of any of them is worth reading the responses for.
HACK_KEYS = ("code_shadowed", "code_verdict_tampered", "code_non_plain_result", "code_early_exit", "code_no_verdict")
# How many refused problems and unmeasured rows the shard summary lists by name.
LIST_LIMIT = 2000


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def truth_text(value) -> str:
    """The ground truth as prepare_data stores it: the row's response field passed through str()."""
    return value if isinstance(value, str) else str(value)


def statement_key(text: str, special: tuple[str, ...]) -> str:
    """The part of a statement that must appear verbatim in the decoded prompt.

    The tokenizer turns a special-token string inside the statement into that special token, and
    decoding with skip_special_tokens=True then drops it. Those strings are removed here for the
    same reason. Leading and trailing whitespace is stripped, which keeps the key a substring.
    """
    for token in special:
        text = text.replace(token, "")
    return unicodedata.normalize("NFC", text).strip()


class SubsetIndex:
    """The subset rows of one shard, indexed by ground truth and by the start of their statement."""

    def __init__(self, rows: list[dict], special: tuple[str, ...]):
        self.rows = rows
        self.keys = [statement_key(str(r.get("instruction", "")), special) for r in rows]
        self.truths = [sha1(truth_text(r.get("response", ""))) for r in rows]
        self.by_truth: dict[str, list[int]] = collections.defaultdict(list)
        self.by_anchor: dict[str, list[int]] = collections.defaultdict(list)
        self.short: list[int] = []
        # Two rows with the same stripped statement and the same tests are one problem to
        # prepare_data, which keeps the first copy. Later copies point at that first row.
        first: dict[tuple[str, str], int] = {}
        self.copy_of: dict[int, int] = {}
        for pos, (row, key, truth) in enumerate(zip(rows, self.keys, self.truths)):
            ident = (str(row.get("instruction", "")).strip(), truth)
            if ident in first:
                self.copy_of[pos] = first[ident]
                continue
            first[ident] = pos
            if not key:
                # An empty statement is dropped by prepare_data, and an empty key would match any input.
                continue
            self.by_truth[truth].append(pos)
            if len(key) >= ANCHOR:
                self.by_anchor[key[:ANCHOR]].append(pos)
            else:
                self.short.append(pos)

    def by_text(self, text: str) -> list[int]:
        """Rows whose statement is contained in ``text``, found through the anchor index."""
        found = {pos for pos in self.short if self.keys[pos] in text}
        anchors = self.by_anchor
        for start in range(len(text) - ANCHOR + 1):
            hit = anchors.get(text[start : start + ANCHOR])
            if hit:
                found.update(pos for pos in hit if text.startswith(self.keys[pos], start))
        return sorted(found)

    def match(self, text: str, truth: str | None) -> tuple[int | None, str, list[int]]:
        """Returns (row position or None, how it matched or why not, the candidate positions)."""
        if truth is not None:
            hits = [pos for pos in self.by_truth.get(truth, []) if self.keys[pos] in text]
            if len(hits) == 1:
                return hits[0], "truth+text", hits
            if len(hits) > 1:
                return None, "ambiguous", hits
            # Nothing with these tests contains the statement. Look by text alone to tell a row
            # whose tests changed apart from a prompt that is not in the subset at all.
            text_hits = self.by_text(text)
            return None, ("truth_mismatch" if text_hits else "no_match"), text_hits
        hits = self.by_text(text)
        if len(hits) == 1:
            return hits[0], "text", hits
        return None, ("ambiguous" if hits else "no_match"), hits


def frame_of(text: str, key: str) -> tuple[str, str]:
    """The prompt text around the statement: what the template put before and after it."""
    start = text.find(key)
    return text[:start], text[start + len(key) :]


def resolve_by_frame(text: str, candidates: list[int], keys: list[str], frame: tuple[str, str]) -> int | None:
    """The one candidate whose statement is exactly what ``frame`` leaves of ``text``, or None.

    Several rows fit a prompt when one statement contains another, as "Print hello." is inside
    "Read a line. Print hello.". The prompt of a row is the template's text before the statement,
    the statement, and the template's text after it. Once the other problems of the shard have
    shown that text, cutting it off the prompt leaves exactly one statement, which picks the row.
    """
    before, after = frame
    if not (text.startswith(before) and text.endswith(after)) or len(before) + len(after) > len(text):
        return None
    middle = text[len(before) : len(text) - len(after)].strip()
    exact = [pos for pos in candidates if keys[pos] == middle]
    return exact[0] if len(exact) == 1 else None


def code_block_after_think(output: str) -> bool:
    """True when the response closed its thinking block and wrote a fenced block after it.

    The reward reads only the text after the last </think>. Without a fence there, it runs that
    whole text as a program, which almost always fails. The share of such responses tells a model
    that does not answer in the expected format apart from one whose programs are wrong.
    """
    head, sep, tail = output.rpartition("</think>")
    return bool(sep) and "```" in tail


def fmean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


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
        "code_block_frac": round(sum(r.get("code_block_frac", 0.0) for r in rows) / total, 4),
    }


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_groups(path: str) -> tuple[dict, int]:
    """Groups the samples of one dump by problem. Returns ({(input, truth hash): samples}, gts-less samples).

    The key holds the tests as well as the prompt. Two rows whose statements decode to the same text
    but whose tests differ were two prompts to verl, and they stay two problems here. Each sample
    keeps only the fields that the pass rate needs: the tests travel in every sample of the dump,
    and a shard can hold gigabytes of them.
    """
    groups: dict[tuple[str, str | None], list[dict]] = collections.OrderedDict()
    no_truth = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            gts = row.pop("gts", None)
            truth = None if gts is None else sha1(truth_text(gts))
            no_truth += truth is None
            text = unicodedata.normalize("NFC", row.pop("input"))
            sample = {k: v for k, v in row.items() if k in ("output", "score", "answer_correct", "truncated",
                                                            "think_unclosed") or k.startswith("code_")}
            groups.setdefault((text, truth), []).append(sample)
    return groups, no_truth


def special_strings(tokenizer) -> tuple[str, ...]:
    """Every special-token string of the tokenizer, longest first, which decoding with skip_special_tokens drops."""
    if tokenizer is None:
        return ()
    names = set(getattr(tokenizer, "all_special_tokens", []) or [])
    for token in (getattr(tokenizer, "added_tokens_decoder", None) or {}).values():
        if getattr(token, "special", False):
            names.add(str(token.content))
    return tuple(sorted((n for n in names if n), key=len, reverse=True))


def shard_cmd(args: argparse.Namespace) -> int:
    subset_rows = read_jsonl(args.subset)
    files = sorted(glob.glob(os.path.join(args.run_dir, "val_generations", "*.jsonl")))
    if len(files) != 1:
        print(f"passrate_agg_code: expected one val_generations file in {args.run_dir}, found {files}", file=sys.stderr)
        return 2
    groups, no_truth = read_groups(files[0])

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    index = SubsetIndex(subset_rows, special_strings(tokenizer))
    keys = list(groups)
    lengths: dict[tuple, list[int]] = {}
    if tokenizer is not None:
        # Outputs are encoded in chunks and only their lengths are kept. A shard of 4K samples at
        # 10K tokens each would otherwise hold 40M token ids in Python lists at once.
        batch = [s["output"] for k in keys for s in groups[k]]
        flat: list[int] = []
        for start in range(0, len(batch), 256):
            ids = tokenizer(batch[start : start + 256], add_special_tokens=False)["input_ids"]
            flat += [len(x) for x in ids]
        pos = 0
        for k in keys:
            lengths[k] = flat[pos : pos + len(groups[k])]
            pos += len(groups[k])

    # The first pass matches every problem on its own. The prompt frame that the matched problems
    # share then settles the problems that several rows fit (resolve_by_frame explains how).
    matches = {k: index.match(*k) for k in keys}
    frames: collections.Counter = collections.Counter(
        frame_of(k[0], index.keys[pos]) for k, (pos, _, _) in matches.items() if pos is not None)
    if frames:
        common = frames.most_common(1)[0][0]
        for k, (pos, how, candidates) in matches.items():
            if how == "ambiguous":
                picked = resolve_by_frame(k[0], candidates, index.keys, common)
                if picked is not None:
                    matches[k] = (picked, "frame", candidates)
                    frames[common] += 1

    out_rows = []
    refused: list[dict] = []
    methods: collections.Counter = collections.Counter()
    matched_positions: collections.Counter = collections.Counter()
    reason_totals: collections.Counter = collections.Counter()
    total_tokens = 0
    cpu_total = 0.0
    for k in keys:
        text, truth = k
        samples = groups[k]
        pos, how, candidates = matches[k]
        methods[how] += 1
        src = subset_rows[pos] if pos is not None else None
        if pos is None:
            refused.append({"prompt_sha1": sha1(text), "reason": how,
                            "candidates": [subset_rows[c].get("unique_id") for c in candidates][:10]})
        else:
            matched_positions[pos] += 1
        n = len(samples)
        solved = [float(s.get("score") or 0.0) >= SOLVED for s in samples]
        correct = sum(solved)
        lens = lengths.get(k) or [len(s["output"]) for s in samples]
        total_tokens += sum(lens)
        correct_lens = [length for length, ok in zip(lens, solved) if ok]
        # Every code_<reason> key is 1 for the reason the program failed, so its mean is that
        # reason's share. Only the reasons that occur are written, to keep the rows short.
        # code_passed, code_shadowed and code_cpu_s are not failure reasons, so they are left out.
        reasons = {}
        for key in sorted({key for s in samples for key in s if key.startswith("code_")}):
            if key in ("code_cpu_s", "code_passed", "code_shadowed"):
                continue
            share = fmean([float(s.get(key) or 0.0) for s in samples])
            if share:
                reasons[key[len("code_"):]] = share
                reason_totals[key[len("code_"):]] += sum(float(s.get(key) or 0.0) for s in samples)
        cpu = [float(s.get("code_cpu_s") or 0.0) for s in samples]
        cpu_total += sum(cpu)
        out_rows.append({
            "unique_id": src.get("unique_id") if src else None,
            "prompt_sha1": sha1(text),
            "source": src.get("source") if src else None,
            "id": src.get("id") if src else None,
            "difficulty": src.get("difficulty") if src else None,
            "n": n,
            "n_correct": int(correct),
            "pass_rate": round(correct / n, 4),
            "answer_correct_frac": fmean([float(s.get("answer_correct") or 0.0) for s in samples]),
            "mean_response_len": round(sum(lens) / n, 1),
            "max_response_len": max(lens),
            "mean_response_len_correct": round(sum(correct_lens) / len(correct_lens), 1) if correct_lens else None,
            "truncated_frac": fmean([float(s.get("truncated") or 0.0) for s in samples]),
            "think_unclosed_frac": fmean([float(s.get("think_unclosed") or 0.0) for s in samples]),
            "code_block_frac": fmean([float(code_block_after_think(s["output"])) for s in samples]),
            "code_cpu_s_mean": round(sum(cpu) / n, 3),
            "fail_reasons": reasons,
            "hack_signal_frac": fmean([float(any(float(s.get(h) or 0.0) for h in HACK_KEYS)) for s in samples]),
            "match": how,
            "len_unit": "tokens" if tokenizer is not None else "chars",
        })

    ns = collections.Counter(r["n"] for r in out_rows)
    n_main = ns.most_common(1)[0][0] if ns else 0
    copies = set(index.copy_of)
    not_measured = [r.get("unique_id") for p, r in enumerate(subset_rows) if p not in matched_positions]
    twice = [subset_rows[p].get("unique_id") for p, c in matched_positions.items() if c > 1]
    samples_total = sum(r["n"] for r in out_rows)
    timing = parse_log(os.path.join(args.run_dir, "launch.log"))
    summary = {
        "run_dir": args.run_dir,
        "val_generations": files[0],
        "samples": samples_total,
        "prompts": len(out_rows),
        "n_per_prompt": dict(ns),
        "subset_rows": len(subset_rows),
        "subset_rows_not_measured": len(not_measured),
        # prepare_data drops statements whose prompt is longer than data.max_prompt_length, empty
        # ones and repeated ones before verl sees them. The data manifest below counts each kind.
        "not_measured_ids": not_measured[:LIST_LIMIT],
        "subset_copies": len(copies),
        "unmatched_prompts": len(refused),
        "match_methods": dict(methods),
        "refused": refused[:LIST_LIMIT],
        "rows_matched_twice": twice[:LIST_LIMIT],
        "samples_without_gts": no_truth,
        # Each distinct (before, after) pair is one shape of prompt around the statement. One pair
        # is expected: the prompt template of the run. More mean that some statements carry
        # leading or trailing text of their own, or that the shard mixes templates.
        "frames": {"distinct": len(frames),
                   "top": [{"before": b[-300:], "after": a[:400], "prompts": c} for (b, a), c in frames.most_common(3)]},
        "response_tokens_total": total_tokens,
        "fail_reasons_frac": {k: round(v / samples_total, 4) for k, v in sorted(reason_totals.items())} if samples_total else {},
        "code_cpu_s_total": round(cpu_total, 1),
        **histogram(out_rows, n_main),
        "by_source": {s: histogram([r for r in out_rows if r["source"] == s], n_main)
                      for s in sorted({str(r["source"]) for r in out_rows})},
        "timing": timing,
    }
    if args.data_manifest and os.path.isfile(args.data_manifest):
        with open(args.data_manifest, encoding="utf-8") as handle:
            manifest = json.load(handle)
        summary["data_manifest"] = {k: manifest.get(k) for k in ("rows", "dropped", "prompt_tokens", "max_prompt_length",
                                                                 "prompt_mode", "data_source")}
        # The manifest adds up the drops of the train and the val split. passrate.sh builds both
        # splits from the same shard, so half of each count is the number of problems dropped.
        if manifest.get("src") and manifest.get("src") == manifest.get("val_src"):
            summary["data_manifest"]["dropped_problems"] = {k: v // 2 for k, v in (manifest.get("dropped") or {}).items()}
    if timing.get("generation_s"):
        summary["gen_tok_s_node"] = round(total_tokens / timing["generation_s"], 1)
        summary["prompts_per_hour_generation"] = round(len(out_rows) / timing["generation_s"] * 3600, 1)
    if timing.get("wall_s"):
        summary["prompts_per_hour_wall"] = round(len(out_rows) / timing["wall_s"] * 3600, 1)
    with open(os.path.join(args.run_dir, "passrate.jsonl"), "w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(args.run_dir, "passrate_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)
    brief = {k: v for k, v in summary.items() if k not in ("by_source", "not_measured_ids", "refused", "frames")}
    brief["frames_distinct"] = summary["frames"]["distinct"]
    print(json.dumps(brief, indent=1, ensure_ascii=False))
    return 0 if not refused and not twice else 3


def merge_cmd(args: argparse.Namespace) -> int:
    rows, summaries = [], []
    for run_dir in args.run_dirs:
        rows += read_jsonl(os.path.join(run_dir, "passrate.jsonl"))
        with open(os.path.join(run_dir, "passrate_summary.json"), encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    if not rows:
        print("passrate_agg_code: the run dirs hold no measured problems", file=sys.stderr)
        return 2
    subset = {}
    for row in read_jsonl(args.subset):
        subset[row["unique_id"]] = row
    n = collections.Counter(r["n"] for r in rows).most_common(1)[0][0]
    keep_max = n - 1 if args.keep_max is None else args.keep_max
    os.makedirs(args.out_dir, exist_ok=True)
    kept = 0
    seen: set = set()
    repeated = 0
    with open(os.path.join(args.out_dir, "passrate.jsonl"), "w", encoding="utf-8") as all_out, \
            open(os.path.join(args.out_dir, "keep.jsonl"), "w", encoding="utf-8") as keep_out:
        for r in rows:
            all_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            uid = r["unique_id"]
            if uid in seen:
                # A row measured in two run dirs would otherwise appear twice in keep.jsonl.
                repeated += 1
                continue
            if uid is not None:
                seen.add(uid)
            if uid in subset and args.keep_min <= r["n_correct"] <= keep_max:
                src = dict(subset[uid])
                src.update({k: r[k] for k in ("n", "n_correct", "pass_rate", "mean_response_len", "truncated_frac")})
                keep_out.write(json.dumps(src, ensure_ascii=False) + "\n")
                kept += 1
    gen_s = sum(s["timing"].get("generation_s") or 0 for s in summaries)
    wall_s = sum(s["timing"].get("wall_s") or 0 for s in summaries)
    tokens = sum(s["response_tokens_total"] for s in summaries)
    summary = {
        "shards": len(summaries),
        "subset_rows": len(subset),
        "subset_rows_not_measured": len(set(subset) - seen),
        "unmatched_prompts": sum(1 for r in rows if r["unique_id"] is None),
        "repeated_rows": repeated,
        "keep_rule": f"{args.keep_min} <= n_correct <= {keep_max} of n={n}",
        "kept": kept,
        **histogram(rows, n),
        "by_source": {s: histogram([r for r in rows if r["source"] == s], n) for s in sorted({str(r["source"]) for r in rows})},
        "generation_s": round(gen_s, 1),
        "wall_s": round(wall_s, 1),
        "response_tokens_total": tokens,
        "code_cpu_s_total": round(sum(s.get("code_cpu_s_total") or 0 for s in summaries), 1),
        "gen_tok_s_node": round(tokens / gen_s, 1) if gen_s else None,
        "prompts_per_hour_wall": round(len(rows) / wall_s * 3600, 1) if wall_s else None,
    }
    summary["kept_frac"] = round(kept / len(rows), 4) if rows else None
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_source"}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="passrate_agg_code.py")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--subset", required=True)
    s.add_argument("--tokenizer", default=None, help="model dir; without it lengths are in chars")
    s.add_argument("--data-manifest", default=None, help="manifest.json of the shard's dataset dir, for its drop counts")
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
