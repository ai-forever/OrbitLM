"""The benchmark registry: the lm-eval tasks of every suite benchmark, its metric and its cost.

Base benchmarks follow the paper's few-shot protocol. Chat benchmarks evaluate SFT checkpoints through
the chat template, in one prompt mode each. eval/in_training/README.md describes both protocols.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


# The document counts are those of the full test sets, taken from real lm-eval results. They
# set the cost weights that balance the work across ranks, and they bound the document ranges of
# shards. The scores are weighted by the number of documents each process really evaluated (see
# score_processes).

MATH_SUBJECTS = {
    "algebra": 1187,
    "counting_and_prob": 474,
    "geometry": 479,
    "intermediate_algebra": 903,
    "num_theory": 540,
    "prealgebra": 871,
    "precalc": 546,
}

MMLU_SUBJECTS = {
    "abstract_algebra": 100, "anatomy": 135, "astronomy": 152, "business_ethics": 100,
    "clinical_knowledge": 265, "college_biology": 144, "college_chemistry": 100,
    "college_computer_science": 100, "college_mathematics": 100, "college_medicine": 173,
    "college_physics": 102, "computer_security": 100, "conceptual_physics": 235,
    "econometrics": 114, "electrical_engineering": 145, "elementary_mathematics": 378,
    "formal_logic": 126, "global_facts": 100, "high_school_biology": 310,
    "high_school_chemistry": 203, "high_school_computer_science": 100,
    "high_school_european_history": 165, "high_school_geography": 198,
    "high_school_government_and_politics": 193, "high_school_macroeconomics": 390,
    "high_school_mathematics": 270, "high_school_microeconomics": 238, "high_school_physics": 151,
    "high_school_psychology": 545, "high_school_statistics": 216, "high_school_us_history": 204,
    "high_school_world_history": 237, "human_aging": 223, "human_sexuality": 131,
    "international_law": 121, "jurisprudence": 108, "logical_fallacies": 163,
    "machine_learning": 112, "management": 103, "marketing": 234, "medical_genetics": 100,
    "miscellaneous": 783, "moral_disputes": 346, "moral_scenarios": 895, "nutrition": 306,
    "philosophy": 311, "prehistory": 324, "professional_accounting": 282,
    "professional_law": 1534, "professional_medicine": 272, "professional_psychology": 612,
    "public_relations": 110, "security_studies": 245, "sociology": 201,
    "us_foreign_policy": 100, "virology": 166, "world_religions": 171,
}

MMLU_PRO_CATEGORIES = {
    "biology": 717, "business": 789, "chemistry": 1132, "computer_science": 410,
    "economics": 844, "engineering": 969, "health": 818, "history": 381, "law": 1101,
    "math": 1351, "other": 924, "philosophy": 499, "physics": 1299, "psychology": 798,
}

# The eight RULER needle-in-a-haystack tasks, as local copies whose sample count per length is
# configurable (eval/in_training/tasks/ruler_local.py). Their metric is reported per length.
RULER_TASKS = (
    "itr_niah_single_1",
    "itr_niah_single_2",
    "itr_niah_single_3",
    "itr_niah_multikey_1",
    "itr_niah_multikey_2",
    "itr_niah_multikey_3",
    "itr_niah_multivalue",
    "itr_niah_multiquery",
)
# The local RULER task files define a metric for each of these lengths.
RULER_LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)
# RULER fills a prompt up to its length minus the 128 generated tokens, and lm-eval then appends
# the answer prefix ("The special magic number for ... is", about 20 tokens) on top. The model
# window must leave room for that prefix. Otherwise lm-eval cuts the start of the longest
# prompts, which is where the task instruction sits.
RULER_PROMPT_MARGIN = 256


@dataclass(frozen=True)
class Benchmark:
    """One benchmark of the suite: its lm-eval tasks, the metric we report and a cost estimate.

    ``subtasks`` lists (lm-eval task, number of documents). Each subtask becomes at least one work
    unit, and ``shardable`` subtasks can be split further into document ranges.
    ``lm_eval_args`` are command-line arguments that every process running this benchmark needs;
    processes are only shared by units whose arguments are identical, because lm-eval applies
    them to every task of the process. ``sec_per_doc`` is a rough H100 cost estimate used only to
    balance the ranks. ``parts`` are extra scores over subsets of the subtasks.

    ``own_process`` keeps the units of other benchmarks out of this benchmark's processes, even when
    their arguments are identical. A benchmark whose scoring can fail its whole lm-eval process then
    loses only its own units, and not those of the benchmarks that would have shared the process.

    The remaining fields are used only by the chat benchmarks of SFT suites. ``chat_mode`` is the
    prompt mode (think, no_think or no_prefill), and ``report_as`` is the name in the reported titles.
    ``scored`` says whether the benchmark has an accuracy to report or only feeds the gates.
    ``modules`` are imports that the startup check makes for it in the evaluation environment.
    """

    name: str
    metric: str
    subtasks: tuple[tuple[str, int], ...]
    sec_per_doc: float
    lm_eval_args: tuple[str, ...] = ()
    shardable: bool = False
    parts: tuple[tuple[str, tuple[str, ...]], ...] = ()
    own_process: bool = False
    chat_mode: Optional[str] = None
    report_as: Optional[str] = None
    scored: bool = True
    modules: tuple[str, ...] = ()


BENCHMARKS: dict[str, Benchmark] = {
    "math": Benchmark(
        name="math",
        metric="math_verify,none",
        subtasks=tuple((f"itr_minerva_{name}", docs) for name, docs in MATH_SUBJECTS.items()),
        # Measured on one H100 with vLLM 0.22: the 1187 algebra problems generated in 71 s. The
        # harder subjects write longer solutions, hence the margin.
        sec_per_doc=0.08,
    ),
    "gsm8k": Benchmark(
        name="gsm8k",
        metric="exact_match,strict-match",
        subtasks=(("itr_gsm8k", 1319),),
        sec_per_doc=0.06,
        shardable=True,
    ),
    "mmlu": Benchmark(
        name="mmlu",
        metric="acc,none",
        subtasks=tuple((f"mmlu_{name}", docs) for name, docs in MMLU_SUBJECTS.items()),
        sec_per_doc=0.11,
        # The stock MMLU tasks carry no shot count, so it comes from the command line.
        lm_eval_args=("--num_fewshot", "5"),
    ),
    "mmlu_pro": Benchmark(
        name="mmlu_pro",
        metric="exact_match,custom-extract",
        subtasks=tuple((f"mmlu_pro_{name}", docs) for name, docs in MMLU_PRO_CATEGORIES.items()),
        sec_per_doc=0.09,
        # The stock MMLU-Pro tasks of the pinned lm-eval (task version 3.1; base.yaml says how it
        # differs from the version 2.1 behind earlier numbers). They generate up to 2048 tokens.
        # 512 is what the earlier OrbitLM evaluations used, and it is enough for the
        # "the answer is (X)" format.
        lm_eval_args=("--gen_kwargs", "max_gen_toks=512"),
    ),
    "hellaswag": Benchmark(
        name="hellaswag",
        metric="acc_norm,none",
        subtasks=(("itr_hellaswag", 10042),),
        sec_per_doc=0.12,
        shardable=True,
    ),
    "winogrande": Benchmark(
        name="winogrande",
        metric="acc,none",
        subtasks=(("itr_winogrande", 1267),),
        sec_per_doc=0.03,
        shardable=True,
    ),
    "cruxeval": Benchmark(
        name="cruxeval",
        metric="pass@1,create_test",
        subtasks=(("itr_cruxeval_output", 800), ("itr_cruxeval_input", 800)),
        sec_per_doc=0.12,
        shardable=True,
        parts=(
            ("cruxeval_input", ("itr_cruxeval_input",)),
            ("cruxeval_output", ("itr_cruxeval_output",)),
        ),
    ),
    "humaneval": Benchmark(
        name="humaneval",
        metric="pass@1,create_test",
        subtasks=(("itr_humaneval", 164),),
        sec_per_doc=0.4,
    ),
    "mbpp": Benchmark(
        name="mbpp",
        metric="pass@1,create_test",
        subtasks=(("itr_mbpp", 500),),
        # Measured on one H100 with vLLM 0.22, from the start of the generation to the results
        # file: 0.124 s per problem for the final 32K export and 0.128 s for Qwen3-1.7B-Base.
        # About four fifths of that is code_eval, which runs the programs after the generation, one
        # problem at a time, while vLLM still holds the GPU. HumanEval of the same 32K export took
        # 7% longer in the last evaluation pause of the 32K stage than on that H100. The margin
        # covers this and checkpoints that write longer programs.
        sec_per_doc=0.15,
        # code_eval runs every program through a multiprocessing manager. When the manager cannot
        # start, scoring raises and ends the lm-eval process, which takes every task of that process
        # with it. In a process of its own, only MBPP is lost.
        own_process=True,
    ),
    "ruler": Benchmark(
        name="ruler",
        metric="<length>,none",
        subtasks=tuple((name, 0) for name in RULER_TASKS),
        sec_per_doc=0.0,
    ),
}


# ---------------------------------------------------------------------------
# Chat benchmarks of SFT suites
# ---------------------------------------------------------------------------
#
# These evaluate SFT checkpoints through the chat template, 0-shot, in the prompt modes of the
# think/no_think switch. Each task file (eval/in_training/tasks/itr_chat_*.yaml) serves every mode; the
# mode, the sampling and the budget come from the suite's chat section and are part of the unit's
# signature, so units of different modes never share a process. eval/in_training/README.md
# explains the protocol.

# How a mode appears in the reported series names.
CHAT_SERIES_MODE = {"think": "think", "no_think": "nothink", "no_prefill": "open"}
# The lm-eval model that run_lm_eval.py registers for chat processes (eval/in_training/itr_chat.py).
CHAT_MODEL = "vllm_itr_chat"

# Document counts of the chat tasks. IFEval has all 541 prompts, MMLU-Redux 30 questions per subject,
# and the chat probe 200 English and 60 Russian OASST2 validation prompts plus 100 trivial ones. The
# task files pin the dataset revisions that these counts belong to.
CHAT_TASK_DOCS = {
    "itr_chat_math500": 500,
    "itr_chat_gsm8k": 1319,
    "itr_chat_ifeval": 541,
    "itr_chat_mgsm_ru": 250,
    "itr_chat_mmlu_redux": 1710,
    "itr_chat_probe": 360,
}
MATH_MODULES = ("sympy", "math_verify")
# lm-eval's IFEval needs immutabledict, langdetect and nltk's punkt_tab data. Its helper module tries to
# download punkt_tab when it is missing, which fails quietly offline, so the startup check looks for the
# data itself ("+nltk_punkt_tab") before it imports the module.
NLTK_PUNKT_CHECK = "+nltk_punkt_tab"
IFEVAL_MODULES = (
    "immutabledict", "langdetect", "nltk", NLTK_PUNKT_CHECK, "lm_eval.tasks.ifeval.instructions_util",
)


def _chat_benchmark(name: str, mode: str, report_as: str, task: str, sec_per_doc: float, *,
                    metric: str = "acc,none", **fields: Any) -> Benchmark:
    return Benchmark(
        name=name, metric=metric, subtasks=((task, CHAT_TASK_DOCS[task]),), sec_per_doc=sec_per_doc,
        shardable=True, chat_mode=mode, report_as=report_as, **fields,
    )


def _register_chat_benchmarks() -> dict[str, Benchmark]:
    # sec_per_doc is an assumed H100 time per response, computed from assumed think lengths rather
    # than measured. It only balances the work across the ranks. The tail of the longest response is
    # added to each unit from its budget (chat_unit_tail).
    ifeval = dict(metric="prompt_level_strict_acc,none", modules=IFEVAL_MODULES)
    entries = [
        _chat_benchmark("math500_think", "think", "math500", "itr_chat_math500", 5.0, modules=MATH_MODULES),
        _chat_benchmark("gsm8k_think", "think", "gsm8k", "itr_chat_gsm8k", 1.3, modules=MATH_MODULES),
        _chat_benchmark("ifeval_think", "think", "ifeval", "itr_chat_ifeval", 1.4, **ifeval),
        _chat_benchmark("mgsm_ru_think", "think", "mgsm_ru", "itr_chat_mgsm_ru", 1.6, modules=MATH_MODULES),
        _chat_benchmark("chatprobe_think", "think", "chat", "itr_chat_probe", 1.5, scored=False),
        _chat_benchmark("math500_nothink", "no_think", "math500", "itr_chat_math500", 0.25, modules=MATH_MODULES),
        _chat_benchmark("gsm8k_nothink", "no_think", "gsm8k", "itr_chat_gsm8k", 0.08, modules=MATH_MODULES),
        _chat_benchmark("ifeval_nothink", "no_think", "ifeval", "itr_chat_ifeval", 0.15, **ifeval),
        _chat_benchmark("mgsm_ru_nothink", "no_think", "mgsm_ru", "itr_chat_mgsm_ru", 0.1, modules=MATH_MODULES),
        _chat_benchmark("mmlu_redux_nothink", "no_think", "mmlu_redux", "itr_chat_mmlu_redux", 0.1),
        _chat_benchmark("chatprobe_nothink", "no_think", "chat", "itr_chat_probe", 0.2, scored=False),
        # Gate G3: the prompts of the think benchmarks and the chat probe without the think prefill. One
        # unit per task, all in one process; 64 generated tokens each.
        Benchmark(
            name="think_open", metric="", sec_per_doc=0.03, chat_mode="no_prefill", report_as="np", scored=False,
            subtasks=tuple((task, CHAT_TASK_DOCS[task]) for task in (
                "itr_chat_math500", "itr_chat_gsm8k", "itr_chat_ifeval", "itr_chat_mgsm_ru", "itr_chat_probe")),
            modules=MATH_MODULES + IFEVAL_MODULES,
        ),
    ]
    return {entry.name: entry for entry in entries}


BENCHMARKS.update(_register_chat_benchmarks())


def ruler_seconds_per_sample(length: int) -> float:
    """Estimated H100 seconds for one RULER sample of the given length, including its build.

    An H200 measured about 20K prefill tokens per second at 4K-16K; an H100 is taken as 1.3x
    slower, and the second factor grows the cost with the quadratic attention part.
    """
    return 1.3 * float(length) / 20000.0 * (1.0 + float(length) / 65536.0) + 0.05
