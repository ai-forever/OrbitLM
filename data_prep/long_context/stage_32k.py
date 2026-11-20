"""The 32K context-extension stage: how each of its streams is packed, and the mix it trains on.

The tools take this module by name (--stage 32k): pack.py reads the block length and the root,
run_tasks.py (its stage_tasks function) turns BUILD_TASKS and STREAMS into its task list, and
mix.py turns MIX into the data block of the stage yaml.

Most 32K streams are packed again from the document stores and 4K shards of the 16K build. Only
prolong_code has a store of its own, written by build_prolong_code.py.
"""
import os

import paths

NAME = "32k"
SEQ_LEN = 32768
BLOCK_TOKENS = SEQ_LEN + 1
ROOT = paths.CTX32K_ROOT
STORE_ROOT = os.path.join(paths.CTX16K_ROOT, "work", "docs")
# Store files and 4K runs are visited in this seeded random order, so a stream that is capped
# with --target-tokens is a random sample of its pool.
SHUFFLE_SEED = 32768

E = paths.TOKENIZED_4K_EN
RU = paths.TOKENIZED_4K_RU
# With document masking, attention never crosses an EOS, so a block filled with short documents
# trains no position past its longest document. Long streams therefore keep only documents of
# at least 16,384 tokens, which are the only ones that teach the 16K-32K range.
LONG = "--min-doc-tokens 16384 --truncate-doc-tokens 131072"
CHAT = "--allow-think --max-doc-tokens 32768"

# name: (nparts, pack.py arguments that pick the source and its filters).
# The caps (--target-tokens) are about 1.5 times what the mix below plans for a 20B-token stage,
# so the shares can still be raised a little without a rebuild. Every stream with a large enough
# store is read at most once. bench_mcq (QASC only) is the exception: it is small and gets many
# passes at its share, which mix.py reports.
STREAMS = {
    # Long, 40% of the stage.
    "prolong_code": (4, f"--store prolong_code --store-root {ROOT}/work/docs {LONG}"),
    "code_repos_long": (2, f"--store code_repos {LONG} --target-tokens 1.6e9"),
    "olmocr_science_long": (8, f"--store olmocr_science {LONG} --target-tokens 4.2e9"),
    "olmocr_health_long": (2, f"--store olmocr_health {LONG} --target-tokens 1.0e9"),
    "olmocr_literature_long": (2, f"--store olmocr_literature {LONG} --target-tokens 2.0e9"),
    "arxiv_long": (2, f"--store arxiv {LONG} --target-tokens 1.2e9"),
    "finepdfs_ru_long": (2, f"--store finepdfs_ru {LONG} --target-tokens 3.0e9"),
    "ru_books_long": (2, f"--src {RU}/books {LONG} --target-tokens 1.6e9"),
    # Short, 60%: the 16K stage's other streams rebuilt as 32K blocks, plus the STEM streams of the
    # 4K stage. nemotron_cc skips the 818 runs that the 16K stage sampled, so this replay is web
    # text the model has not seen.
    "nemotron_cc": (4, f"--src {E}/nemotron-cc --exclude-sample-tokens 150e9 --exclude-sample-seed 16384 "
                       f"--target-tokens 5.2e9"),
    "ru_web_le8k": (3, f"--src {RU}/web --max-doc-tokens 8192 --target-tokens 3.1e9"),
    "cc_math": (16, f"--src {E}/Nemotron-CC-Math-v1 --target-tokens 2.1e9"),
    "opc": (1, f"--src {E}/opc-annealing-corpus --target-tokens 0.14e9"),
    "ru_wiki": (1, f"--src {RU}/wikipedia --src {RU}/wikiquote --target-tokens 0.03e9"),
    # A 4K run holds one raw file, and one raw file of these sources holds more tokens than a part
    # needs, so each part reads about one run. The part counts below are therefore what sets how
    # many raw files a stream samples: all 12 Cascade files (code, general, science and math
    # domains), and enough files of the other sources to cover their subsets.
    "cascade_sft": (12, f"--src {E}/Nemotron-Cascade-SFT-Stage-1_merged {CHAT} --target-tokens 2.7e9"),
    # STEM streams of the 4K stage that the 16K stage left out (see the mix below).
    "nemotron_specialized": (32, f"--src {E}/Nemotron-Pretraining-Specialized-v1 --target-tokens 2.7e9"),
    "nemotron_synth_code": (8, f"--src {E}/Nemotron-Post-Training_merged {CHAT} --target-tokens 1.1e9"),
    "megamath": (8, f"--src {E}/MegaMath-high-quality --target-tokens 0.65e9"),
    "acereason": (2, f"--store acereason {CHAT} --target-tokens 1.05e9"),
    "openthoughts3": (1, f"--store openthoughts3 {CHAT} --target-tokens 0.28e9"),
    "nemotron_if": (1, f"--store nemotron_if {CHAT} --target-tokens 0.25e9"),
    "smoltalk2_en": (1, f"--store smoltalk2_en {CHAT}"),
    "twix": (1, f"--store twix {CHAT} --target-tokens 0.12e9"),
    "ru_sft": (1, f"--store ru_sft {CHAT} --target-tokens 0.05e9"),
    "grandmaster": (1, f"--store grandmaster {CHAT} --target-tokens 0.03e9"),
    "oasst2_ru": (1, f"--store oasst2_ru {CHAT}"),
    "bench_train": (1, "--store bench_train --max-doc-tokens 32768"),
    "bench_math": (1, "--store bench_math --max-doc-tokens 32768 --target-tokens 0.28e9"),
    "bench_mcq": (1, "--store bench_mcq --max-doc-tokens 32768 --target-tokens 0.1e9"),
    "bench_math_dm": (1, "--store bench_math_dm --max-doc-tokens 32768 --target-tokens 0.17e9"),
}

# Builders that write the stores of this stage, in the task format that run_tasks.py describes.
# run_tasks.py runs them before the pack tasks that STREAM_NEEDS ties to them.
BUILD_TASKS = [
    {"name": "build_prolong_code", "nparts": 8,
     "cmd": "{py} build_prolong_code.py --part {part} --nparts {nparts} --target-tokens 4.5e9"},
]
STREAM_NEEDS = {"prolong_code": ["build_prolong_code"]}

# The 16K stage cut the dense STEM and reasoning data of the 4K stage's final mix (CC-Math,
# MegaMath, Cascade-SFT, Nemotron-Pretraining-Specialized and synthetic code: about 47% of its
# tokens) to about 17%, and MATH fell about 3 points against the 4K checkpoint. This mix brings
# STEM back to 35% of the stage, in the 4K stage's proportions plus the reasoning SFT sets, and
# gives long documents 40% instead of ProLong's 60% (arXiv 2410.02660, section 3.2): the 16K
# stage reached near-perfect passkey retrieval with about 34% long documents. General web text
# and chat take the remaining 25%. Every sample is BLOCK_TOKENS long, so sample shares are token
# shares, and the trainer renormalizes them.
LONG_TOTAL = 0.40
# Relative weights of the long streams; they are scaled to LONG_TOTAL below.
LONG_SHARES = {
    "prolong_code": 0.15, "code_repos_long": 0.05,
    "olmocr_science_long": 0.14, "olmocr_health_long": 0.03, "olmocr_literature_long": 0.05, "arxiv_long": 0.03,
    "finepdfs_ru_long": 0.10, "ru_books_long": 0.05,
}
# 35%: the 4K stage's STEM streams in its proportions (30%), plus reasoning SFT and math sets (5%).
STEM_SHARES = {
    "cc_math": 0.067, "megamath": 0.021, "nemotron_synth_code": 0.035, "cascade_sft": 0.089,
    "nemotron_specialized": 0.089,
    "acereason": 0.030, "openthoughts3": 0.008, "bench_math": 0.008, "bench_math_dm": 0.003,
}
# 25%: English and Russian web text, code, instruction following and chats.
GENERAL_SHARES = {
    "nemotron_cc": 0.1375, "ru_web_le8k": 0.085, "opc": 0.004, "ru_wiki": 0.001,
    "nemotron_if": 0.008, "smoltalk2_en": 0.001, "twix": 0.006, "ru_sft": 0.002, "grandmaster": 0.001,
    "oasst2_ru": 0.00001, "bench_train": 0.0015, "bench_mcq": 0.003,
}
_long_sum = sum(LONG_SHARES.values())
MIX = dict({name: share * LONG_TOTAL / _long_sum for name, share in LONG_SHARES.items()},
           **STEM_SHARES, **GENERAL_SHARES)
LONG_STREAMS = tuple(LONG_SHARES)
BUDGET_TOKENS = 20e9
MAX_PASSES = 2.0
SAMPLES_PER_GPU = 2  # the micro-batch of the 32K run, for the step count mix.py prints
GPUS = 64
MIX_YAML = os.path.join(ROOT, "manifests", "stage32k_data.yaml")

# run_tasks.py keeps its logs and its finished-part markers here.
RUN_LOG_DIR = os.path.join(ROOT, "logs", "build32")
PREFLIGHT_PATHS = (STORE_ROOT, paths.PROLONG_TEXT, E, RU)
MIN_FREE_TB = 1
