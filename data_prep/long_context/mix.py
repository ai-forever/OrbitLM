"""Mixture calculator: prints how many tokens and passes each stream of a stage gets, and writes
the data block of the stage yaml.

The shares come from the stage's MIX (stage_16k.py or stage_32k.py), or from a JSON file given
with --mix. Every sample of a stage has the same length, so a stream's sample share is also its
token share. The trainer renormalizes the shares, so they do not have to sum to exactly 1.
A mix can be changed without rebuilding any data as long as no stream needs more passes than
--max-passes.

Usage: mix.py --stage 32k [--mix shares.json] [--budget 20e9] [--max-passes 2] [--out data.yaml]
"""
import argparse
import json
import os

import paths
import stages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages.add_stage_argument(ap)
    ap.add_argument("--mix", default="", help="a JSON file of {stream: share}; default: the stage's MIX")
    ap.add_argument("--budget", type=float, default=None, help="training tokens of the stage (default: BUDGET_TOKENS)")
    ap.add_argument("--max-passes", type=float, default=None, help="warn above this (default: MAX_PASSES)")
    ap.add_argument("--out", default=None, help="the yaml data block to write (default: MIX_YAML)")
    a = ap.parse_args()
    stage = stages.get(a.stage)
    budget = stage.BUDGET_TOKENS if a.budget is None else a.budget
    max_passes = stage.MAX_PASSES if a.max_passes is None else a.max_passes
    out = a.out or stage.MIX_YAML
    if a.mix:
        with open(a.mix) as f:
            mix = json.load(f)
    else:
        mix = stage.MIX
    total = sum(mix.values())
    streams_root = os.path.join(stage.ROOT, "streams")
    # The model trains on SEQ_LEN tokens of every sample; the last token of a block is only a label.
    sample_tokens = stage.SEQ_LEN

    rows, problems, proportions = [], [], {}
    for name, share in mix.items():
        index_path = os.path.join(streams_root, name, "index.json")
        if not os.path.exists(index_path):
            problems.append(f"{name}: stream not built yet (no {index_path})")
            continue
        indexes = [d for d, _, files in os.walk(os.path.join(streams_root, name)) if "index.json" in files]
        if len(indexes) != 1:
            problems.append(f"{name}: {len(indexes)} index.json files; the trainer would split its share between them")
        with open(index_path) as f:
            samples = sum(s["samples"] for s in json.load(f)["shards"])
        pool = samples * sample_tokens
        planned = share / total * budget
        passes = planned / pool if pool else float("inf")
        rows.append((name, share / total, pool, planned, passes))
        proportions[name] = round(share / total, 6)
        if passes > max_passes:
            problems.append(f"{name}: {passes:.1f} passes (> {max_passes}); lower its share or add data")

    width = max([len(r[0]) for r in rows] + [6])
    print(f"{'stream':{width}s} {'share':>7s} {'pool, B':>9s} {'planned, B':>11s} {'passes':>7s}")
    for name, share, pool, planned, passes in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:{width}s} {share * 100:6.2f}% {pool / 1e9:9.2f} {planned / 1e9:11.2f} {passes:7.2f}")
    if stage.LONG_STREAMS:
        long_share = sum(r[1] for r in rows if r[0] in stage.LONG_STREAMS)
        print(f"long share {long_share * 100:.1f}%")
    tokens_per_step = stage.SAMPLES_PER_GPU * sample_tokens * stage.GPUS
    print(f"budget {budget / 1e9:.0f}B tokens = {budget / tokens_per_step:.0f} steps at "
          f"{stage.SAMPLES_PER_GPU} x {stage.SEQ_LEN // 1024}K x {stage.GPUS} GPUs")
    for p in problems:
        print("WARNING:", p)

    # mosaic_random is the trainer's only mixing mode. The block still names it, so that a config
    # made from this block says how its streams are mixed. stage1_root is written in the
    # ${ORBIT_DATA} form of the configs, so the block carries no path of this machine.
    lines = [f"stage1_root: {paths.config_path(streams_root)}", "stage1_mixing_mode: mosaic_random",
             "stage1_token_dtype: uint32", "stage1_streams:"]
    lines += [f"  {name}: {name}" for name in proportions]
    lines += ["stage1_dataset_proportions:"]
    lines += [f"  {name}: {value}" for name, value in proportions.items()]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
