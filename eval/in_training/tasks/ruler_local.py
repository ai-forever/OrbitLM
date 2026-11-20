"""RULER needle-in-a-haystack tasks with a configurable number of samples per length.

lm-eval's RULER tasks build 500 samples per length (niah_utils.py hard-codes it), which is too
many for an evaluation that pauses training. These functions call the same generator with the
same settings per task, and take the sample count from the task metadata
(--metadata '{"num_samples": 100, "max_seq_lengths": [4096]}'). The scoring functions are the
stock ones.
"""
from lm_eval.tasks.ruler.common_utils import (
    DEFAULT_SEQ_LENGTHS,
    aggregate_metrics,
    get_tokenizer,
    process_results,
)
from lm_eval.tasks.ruler.niah_utils import TEMPLATE, download_dataset
from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack

# The itr_niah_*.yaml task files name these functions with !function, so they are the interface of
# this module. Listing them here also tells linters that the two scoring functions imported from
# lm-eval are used.
__all__ = [
    "aggregate_metrics",
    "process_results",
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
]


# The generator settings of the eight stock niah tasks (lm_eval/tasks/ruler/niah_utils.py).
NIAH_SETTINGS = {
    "single_1": dict(type_haystack="repeat", type_needle_k="words", type_needle_v="numbers"),
    "single_2": dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers"),
    "single_3": dict(type_haystack="essay", type_needle_k="words", type_needle_v="uuids"),
    "multikey_1": dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_k=4),
    "multikey_2": dict(type_haystack="needle", type_needle_k="words", type_needle_v="numbers"),
    "multikey_3": dict(type_haystack="needle", type_needle_k="uuids", type_needle_v="uuids"),
    "multivalue": dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_v=4),
    "multiquery": dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_q=4),
}


def _build(name, kwargs):
    settings = NIAH_SETTINGS[name]
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    num_samples = int(kwargs.pop("num_samples", 500))
    # get_tokenizer is cached on its arguments, so only hashable metadata may reach it.
    tokenizer_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, type(None)))}
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack=settings["type_haystack"]),
            max_seq_length=seq,
            template=TEMPLATE,
            num_samples=num_samples,
            TOKENIZER=get_tokenizer(**tokenizer_kwargs),
            **settings,
        )
        for seq in seq_lengths
    )


def niah_single_1(**kwargs):
    return _build("single_1", kwargs)


def niah_single_2(**kwargs):
    return _build("single_2", kwargs)


def niah_single_3(**kwargs):
    return _build("single_3", kwargs)


def niah_multikey_1(**kwargs):
    return _build("multikey_1", kwargs)


def niah_multikey_2(**kwargs):
    return _build("multikey_2", kwargs)


def niah_multikey_3(**kwargs):
    return _build("multikey_3", kwargs)


def niah_multivalue(**kwargs):
    return _build("multivalue", kwargs)


def niah_multiquery(**kwargs):
    return _build("multiquery", kwargs)
