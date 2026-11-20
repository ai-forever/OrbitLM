"""Loop detection on response token ids, and the loop-tail mask for RL and distillation.

A loop is a response that keeps repeating itself until the length limit cuts it. The SFT chat
evaluation of this repo (``_is_loop`` in ``src/orbit/in_training_eval/scoring.py``, its thresholds
in ``DEFAULT_DETECTORS`` of ``src/orbit/in_training_eval/suite.py``, ``rep_ngram`` and
``token_facts`` in ``eval/in_training/tasks/chat_local.py``) calls a response a loop when it has
at least 512 generated ids and its last 1,024 ids repeat themselves. "Repeat themselves" means rep16 >= 0.5: at least half
of the 16-grams in that tail repeat an earlier 16-gram of the tail. This module repeats that detector
with the same thresholds, so the rollouts of RL and distillation count exactly the responses that the
evaluation counts.

The evaluation also counts a response as a loop when vLLM's repetition detection stopped it. Our
rollouts never turn repetition detection on, so that case cannot happen here, and only the tail test
is ported.

Two things are added that the evaluation does not need:

1. ``loop_start`` finds where the repeating text begins. The tail test only says that the last 1,024
   ids repeat, but a loop can begin long before them, or only in their second half.
2. ``mask_loop_tail`` turns the response mask off for the loop's tail when the run sets
   ``ORBIT_MASK_LOOP_TAIL``. A teacher that scores a loop sees the repeated text in its own context, so it is likely to
   predict each further copy with high confidence. The distillation advantage could then reward
   the student for continuing the loop. The mask keeps the repeated tail out of the distillation
   loss and out of a task-reward loss. It begins one n-gram into the first repeated copy
   (``tail_mask_start``), so the tokens where the student starts to repeat itself, which is where
   the teacher pushes against the repeat, stay in the loss. The mask is off unless
   ``ORBIT_MASK_LOOP_TAIL`` turns it on, so RL runs are unchanged.

This module imports only the standard library, so the reward workers and the tests can use it
without verl.
"""
from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping, Sequence

# The detector thresholds of the evaluation: DEFAULT_DETECTORS in src/orbit/in_training_eval/suite.py
# ("loop_min_tokens": 512, "loop_rep16": 0.5), and the 1,024-id tail and 16-grams of chat_local.py.
LOOP_NGRAM = 16
LOOP_TAIL_TOKENS = 1024
LOOP_MIN_TOKENS = 512
LOOP_REP16 = 0.5

# The run config switches the loop-tail mask on through this variable. A config sets it under
# orbit.env, and the launcher copies that block into the env of every Ray process, the agent-loop
# workers included.
MASK_LOOP_TAIL_ENV = "ORBIT_MASK_LOOP_TAIL"
_TRUE_WORDS = ("1", "true", "yes", "on")
_FALSE_WORDS = ("0", "false", "no", "off")


def _ngrams(ids: Sequence[int], n: int) -> list[tuple[int, ...]]:
    """Every n-gram of ``ids`` in order, as tuples, built the way the evaluation builds them.

    Tuples of ids compare and hash by value, so equal n-grams are found exactly, with no hash
    collisions to worry about. Building them for a 16,384-id response takes a millisecond or two.
    """
    return [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]


def rep_ngram(ids: Sequence[int], n: int = LOOP_NGRAM) -> float:
    """The share of n-grams in ``ids`` that repeat an earlier n-gram: 1 - distinct / all.

    It is 0 for text without repeats and close to 1 for a loop. This is the evaluation's
    ``rep_ngram`` with the same edge case: fewer than n + 1 ids give 0.
    """
    ids = [int(token) for token in ids]
    if len(ids) < n + 1:
        return 0.0
    grams = _ngrams(ids, n)
    return 1.0 - len(set(grams)) / float(len(grams))


def tail_rep16(ids: Sequence[int]) -> float:
    """rep16 over the last 1,024 ids, rounded to four decimals.

    The evaluation stores this value rounded (``rep16_tail`` in its records) and compares the
    rounded value with the threshold, so the same rounding is applied here.
    """
    return round(rep_ngram(list(ids)[-LOOP_TAIL_TOKENS:], LOOP_NGRAM), 4)


def is_loop(ids: Sequence[int], min_tokens: float = LOOP_MIN_TOKENS, threshold: float = LOOP_REP16) -> bool:
    """True when the response has at least ``min_tokens`` ids and its tail rep16 is at least ``threshold``.

    The defaults are the evaluation's. With them a response shorter than 512 ids is never called a
    loop, however much it repeats itself.
    """
    return len(ids) >= min_tokens and tail_rep16(ids) >= threshold


def loop_start(ids: Sequence[int], n: int = LOOP_NGRAM) -> int:
    """The index of the first id of the repeating text at the end of ``ids``, or -1 when there is none.

    Definition. Every n-gram position gets +1 when its n-gram occurs at least twice in the whole
    response, and -1 when it occurs only once. The result is the start of the suffix with the
    largest positive sum of these marks; among equal sums the longest suffix wins. A suffix with a
    positive sum holds more repeated n-grams than unique ones. That is close to the detector's own
    criterion, which asks that at least half of the tail's n-grams repeat.

    Why this point. Take a response that is ordinary text up to index s and from there repeats a
    block of p ids at least twice. Every n-gram from s on occurs again one period earlier or later,
    so each one adds +1. The n-gram that starts at s - 1 is normally unique, because if it matched
    the n-gram one period later, the repetition would already start at s - 1. The ordinary text
    before it is mostly unique as well. So the sum is largest exactly at s, the first id of the
    first copy of the block. The first copy is included on purpose, because it is the text that the
    loop goes on to repeat. The loop-tail mask starts later, one n-gram after the first repeated
    copy begins (``tail_mask_start``).

    Why it is robust. A loop with a changing counter (step 1, step 2, ...) has a few unique
    n-grams in every period, but as long as the repeated ones outnumber them the region keeps a
    positive sum and is found whole. A 16-gram that the ordinary text happens to repeat, such as
    an equation written twice, adds a few +1 marks among many -1s and cannot pull the start back. A
    loop that began long before the last 1,024 ids is found from its real start, not from the
    start of the tail window.

    When ``is_loop`` is true the result is never -1: at least half of the tail's n-grams then
    repeat, so the sum over the tail is positive. The result is at most ``len(ids) - n``, so the
    repeating text it marks is never empty.

    The cost is O(n * len(ids)) for building the n-grams, which is linear for the fixed n of 16,
    plus one pass over them.
    """
    ids = [int(token) for token in ids]
    if len(ids) < n + 1:
        return -1
    grams = _ngrams(ids, n)
    counts = Counter(grams)
    best, best_sum, running = -1, 0, 0
    for position in range(len(grams) - 1, -1, -1):
        running += 1 if counts[grams[position]] > 1 else -1
        # The scan runs from the end towards the start, so ">=" moves a tie to the earlier
        # position, which is the longer suffix.
        if running > 0 and running >= best_sum:
            best, best_sum = position, running
    return best


def detect_loop(ids: Sequence[int]) -> tuple[bool, int]:
    """Whether the response loops by the evaluation's detector, and where the loop starts (-1 if not)."""
    ids = [int(token) for token in ids]
    if not is_loop(ids):
        return False, -1
    return True, loop_start(ids)


def tail_mask_start(ids: Sequence[int], start: int, n: int = LOOP_NGRAM) -> int:
    """Where the loop-tail mask begins for a loop that ``loop_start`` found at ``start``, or -1.

    The mask does not begin at ``start`` itself. Let r be the first index from ``start`` on whose
    n-gram already occurred earlier in the response; that is where the first repeated copy of the
    loop begins, r = s + p for a block of p ids repeated from s. The mask begins at r + n.

    The tokens before r are the original block, which the response wrote once like any other text.
    The n tokens from r on are where the student starts to repeat that block. The teacher pushes
    against a repeat exactly there, before the repeated text dominates its context, so those
    tokens stay in the loss. From r + n on, the teacher has seen the whole first repeated n-gram
    and tends to predict each further copy with confidence, so that part is masked.

    A negative ``start`` means no loop and gives -1. When ``start`` comes from ``loop_start``, an r
    always exists, because the repeated n-grams after ``start`` occur at least twice there.
    """
    if start < 0:
        return -1
    ids = [int(token) for token in ids]
    seen = set(_ngrams(ids[:start + n - 1], n))
    for position in range(start, len(ids) - n + 1):
        gram = tuple(ids[position:position + n])
        if gram in seen:
            return position + n
        seen.add(gram)
    return -1


def mask_loop_tail(response_mask: Sequence[int], start: int, end: int | None = None) -> list[int]:
    """A copy of ``response_mask`` with every position from ``start`` up to ``end`` set to 0.

    ``end`` defaults to the end of the mask. A response whose thinking block was closed by force
    passes the index of the first inserted id, so only the looping tail of its thinking is masked and
    the answer written after the forced close stays in the loss.

    A negative ``start`` means no loop and returns the mask unchanged. The mask keeps its length,
    because verl pairs it position by position with the response ids, the log-probs and the
    teacher's log-probs.

    When the loop starts at the very first id, the first id stays unmasked. verl's distillation
    loss takes the min and the max of the per-token loss over the unmasked ids of each micro-batch
    (``compute_distillation_loss_range``), and torch refuses that on an empty selection. With
    dynamic batch sizes, a micro-batch can hold a single long response, so an all-zero mask could
    crash the update step. One trained token of a loop does no harm.
    """
    mask = [int(value) for value in response_mask]
    if start < 0:
        return mask
    cut = max(int(start), 1)
    stop = len(mask) if end is None else min(int(end), len(mask))
    if cut >= stop:
        return mask
    return mask[:cut] + [0] * (stop - cut) + mask[stop:]


def mask_loop_tail_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the loop-tail mask is on, from ``ORBIT_MASK_LOOP_TAIL`` (default off).

    An unset or empty variable means off, so RL runs that never set it are unchanged. Any value
    that is not a clear yes or no is refused, because guessing would silently pick one of two
    different training objectives.
    """
    environ = os.environ if environ is None else environ
    raw = str(environ.get(MASK_LOOP_TAIL_ENV, "")).strip().lower()
    if not raw or raw in _FALSE_WORDS:
        return False
    if raw in _TRUE_WORDS:
        return True
    raise ValueError(f"{MASK_LOOP_TAIL_ENV}={raw!r} is not true or false.")
