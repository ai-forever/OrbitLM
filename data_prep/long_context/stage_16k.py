"""The 16K context-extension stage: how each of its streams is packed, and the mix it trains on.

The tools take this module by name (--stage 16k): pack.py reads the block length and the root,
run_tasks.py (its stage_tasks function) turns BUILD_TASKS and STREAMS into its task list, and
mix.py turns MIX into the data block of the stage yaml. The document stores that STREAMS packs
are written by the build_*.py scripts. data_prep/README.md gives the order in which to run
the builders.
"""
import os

import paths

NAME = "16k"
SEQ_LEN = 16384
BLOCK_TOKENS = SEQ_LEN + 1
ROOT = paths.CTX16K_ROOT
STORE_ROOT = os.path.join(ROOT, "work", "docs")
# The 16K build packed store files in sorted order and 4K runs in source order.
SHUFFLE_SEED = None

E = paths.TOKENIZED_4K_EN
RU = paths.TOKENIZED_4K_RU
CHAT = "--allow-think --max-doc-tokens 16384"
# The 4K pretraining streams were cleaned when they were tokenized, and the 16K build repacked
# them with only the length limits (see pack.py).
REPACK = "--length-filter-only"

# name: (nparts, pack.py arguments that pick the source and its filters). The part counts are the
# ones the streams were built with; packing from a store depends on them, because every part packs
# its own share of the store files and deduplicates only inside itself.
STREAMS = {
    # Pretraining text, repacked from the 4K shards.
    "nemotron_cc": (4, f"--src {E}/nemotron-cc --sample-tokens 150e9 {REPACK}"),
    "ru_web_le8k": (3, f"--src {RU}/web --max-doc-tokens 8192 {REPACK}"),
    "cc_math": (2, f"--src {E}/Nemotron-CC-Math-v1 {REPACK}"),
    "opc": (1, f"--src {E}/opc-annealing-corpus {REPACK}"),
    "ru_wiki": (1, f"--src {RU}/wikipedia --src {RU}/wikiquote {REPACK}"),
    "ru_books": (1, f"--src {RU}/books {REPACK}"),
    "cascade_sft": (1, f"--src {E}/Nemotron-Cascade-SFT-Stage-1_merged --max-doc-tokens 16384 {REPACK}"),
    # Long documents. Truncating at 128K tokens keeps one huge document from filling dozens of
    # blocks in a row.
    "olmocr_science": (16, "--store olmocr_science --truncate-doc-tokens 131072"),
    "olmocr_health": (16, "--store olmocr_health --truncate-doc-tokens 131072"),
    "olmocr_literature": (16, "--store olmocr_literature --truncate-doc-tokens 131072"),
    "arxiv": (16, "--store arxiv --truncate-doc-tokens 131072"),
    "finepdfs_ru": (16, "--store finepdfs_ru --truncate-doc-tokens 131072"),
    "code_repos": (16, "--store code_repos"),
    # Instructions and reasoning. A conversation longer than one block is dropped rather than cut.
    "acereason": (16, f"--store acereason {CHAT}"),
    "openthoughts3": (16, f"--store openthoughts3 {CHAT}"),
    "nemotron_if": (6, f"--store nemotron_if {CHAT}"),
    "smoltalk2_en": (1, f"--store smoltalk2_en {CHAT}"),
    "twix": (4, f"--store twix {CHAT}"),
    "ru_sft": (2, f"--store ru_sft {CHAT}"),
    "grandmaster": (1, f"--store grandmaster {CHAT}"),
    "oasst2_ru": (1, f"--store oasst2_ru {CHAT}"),
    # Benchmark train splits and benchmark-style open sets, in the evaluation format.
    "bench_train": (2, "--store bench_train --max-doc-tokens 16384"),
    "bench_math": (7, "--store bench_math --max-doc-tokens 16384"),
    "bench_math_dm": (13, "--store bench_math_dm --max-doc-tokens 16384"),
    "bench_mcq": (3, "--store bench_mcq --max-doc-tokens 16384"),
}

# The 16K build has no BUILD_TASKS: its builders were run step by step, in the order that
# data_prep/README.md describes.
BUILD_TASKS = []
STREAM_NEEDS = {}

# Token shares of the mix. Every sample is BLOCK_TOKENS long, so sample shares are token shares.
# The trainer renormalizes the shares, so they do not have to sum to exactly 1.
MIX = {
    "nemotron_cc": 0.29, "ru_web_le8k": 0.17, "cc_math": 0.06, "opc": 0.0075, "ru_wiki": 0.0015,
    "olmocr_science": 0.14, "olmocr_health": 0.04, "olmocr_literature": 0.014, "ru_books": 0.054,
    "finepdfs_ru": 0.035, "code_repos": 0.04, "arxiv": 0.0175,
    # Instructions stay at 10%. The English pools came out smaller than planned (Nemotron-IF
    # 0.97B, SmolTalk2 0.08B), so their shares are cut to keep them under about 3 passes, and
    # the difference goes to AceReason, whose pool grows to about 11B with 4 answers per question.
    "acereason": 0.056, "openthoughts3": 0.015, "cascade_sft": 0.005,
    "nemotron_if": 0.013, "smoltalk2_en": 0.001,
    "twix": 0.0065, "ru_sft": 0.002, "grandmaster": 0.001, "oasst2_ru": 0.00001,
    # Benchmark train splits and same-type open sets: 3% in total. bench_math gets about 3.3
    # passes and the templated DeepMind arithmetic under one pass. bench_train and bench_mcq are
    # small stores, so at these shares mix.py warns that they get more passes than MAX_PASSES.
    # The shares are those of the release run; lower them to repeat these stores less.
    "bench_train": 0.0015, "bench_math": 0.015, "bench_mcq": 0.0045, "bench_math_dm": 0.009,
}
# mix.py reports the share of these streams separately. The 16K stage has no long-only streams.
LONG_STREAMS = ()
BUDGET_TOKENS = 200e9
MAX_PASSES = 4.0
SAMPLES_PER_GPU = 4  # the micro-batch of the 16K run, for the step count mix.py prints
GPUS = 64
MIX_YAML = os.path.join(ROOT, "manifests", "stage16k_data.yaml")

# run_tasks.py keeps its logs and its finished-part markers here.
RUN_LOG_DIR = os.path.join(ROOT, "logs", "build16")
PREFLIGHT_PATHS = (STORE_ROOT, E, RU)
MIN_FREE_TB = 3
