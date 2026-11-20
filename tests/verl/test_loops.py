"""CPU tests for orbit_verl.data.loops: the loop detector against the SFT evaluation's, where a loop
starts, the loop-tail mask and its switch.

The module imports only the standard library, so these tests run anywhere with PYTHONPATH=<repo>/src.
They also run with the rest of tests/verl under the verl python.
"""
from __future__ import annotations

import random

import pytest

from orbit_verl.data import loops


# The evaluation's detector, copied with only the line breaks changed. ``rep_ngram`` is from
# eval/in_training/tasks/chat_local.py, ``_is_loop`` is from src/orbit/in_training_eval/scoring.py,
# the thresholds are DEFAULT_DETECTORS of src/orbit/in_training_eval/suite.py, and the rep16_tail
# record field is computed as in ``token_facts``. They are copied rather than imported because the
# trainer's evaluation package needs the trainer env, and these tests run in the verl env. The tests
# below check that the port gives the same answers.
EVAL_DEFAULT_DETECTORS = {"loop_min_tokens": 512.0, "loop_rep16": 0.5, "short_think_tokens": 32.0}


def eval_rep_ngram(ids, n=16):
    if len(ids) < n + 1:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / float(len(grams))


def eval_is_loop(record, detectors):
    if record.get("finish") == "repetition":
        return True
    return (
        record.get("n_gen", 0) >= detectors["loop_min_tokens"]
        and record.get("rep16_tail", 0.0) >= detectors["loop_rep16"]
    )


def eval_record(ids, finish="length"):
    return {"n_gen": len(ids), "finish": finish, "rep16_tail": round(eval_rep_ngram(ids[-1024:]), 4)}


def noise(rng, count, low=1000, high=60000):
    """Random ids from a wide range, so that no 16-gram of them repeats by chance."""
    return [rng.randrange(low, high) for _ in range(count)]


def sample_sequences():
    """Responses of every shape the detector meets: clean, late loops, long loops, short, borderline."""
    rng = random.Random(0)
    clean = noise(rng, 16000)
    cases = {
        "empty": [],
        "one_id": [5],
        "seventeen_ids": list(range(17)),
        "clean_16k": clean,
        "short_loop": [1, 2, 3] * 100,
        "loop_at_511": ([1, 2, 3, 4, 5] * 200)[:511],
        "loop_at_512": ([1, 2, 3, 4, 5] * 200)[:512],
        "eval_late_loop": clean[:14500] + [7, 8, 9, 10] * 20 + noise(rng, 3) + list(range(500, 600)) * 13,
        "long_loop": noise(rng, 2000) + (list(range(300, 350)) * 200)[:6000],
        "half_window": noise(rng, 9000) + (list(range(300, 350)) * 20)[:700],
    }
    # Responses near the threshold: a loop that fills a growing part of the tail.
    for loop_len in (300, 450, 500, 520, 540, 600, 800):
        cases[f"tail_{loop_len}"] = noise(rng, 4000) + (list(range(100, 140)) * 40)[:loop_len]
    # Random mixtures, to cover what the hand-made cases miss.
    for index in range(40):
        pieces, target = [], rng.randrange(400, 3000)
        while sum(len(piece) for piece in pieces) < target:
            if rng.random() < 0.5:
                pieces.append(noise(rng, rng.randrange(1, 400)))
            else:
                period = noise(rng, rng.randrange(1, 80), 100, 400)
                pieces.append((period * 100)[: rng.randrange(1, 900)])
        cases[f"random_{index}"] = [token for piece in pieces for token in piece]
    return cases


SEQUENCES = sample_sequences()


@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_the_port_agrees_with_the_evaluation(name):
    ids = SEQUENCES[name]
    record = eval_record(ids)
    assert loops.tail_rep16(ids) == record["rep16_tail"]
    assert loops.rep_ngram(ids) == eval_rep_ngram(ids)
    assert loops.is_loop(ids) == eval_is_loop(record, EVAL_DEFAULT_DETECTORS)
    looped, start = loops.detect_loop(ids)
    assert looped == loops.is_loop(ids)
    if looped:
        # A detected loop always has a start, and the masked part is never empty.
        assert 0 <= start <= len(ids) - loops.LOOP_NGRAM
        # The mask of a detected loop always has a start too: one n-gram or more after the loop's
        # start, and never past the end of the response.
        assert start + loops.LOOP_NGRAM <= loops.tail_mask_start(ids, start) <= len(ids)
    else:
        assert start == -1


def test_thresholds_are_the_evaluation_defaults():
    assert loops.LOOP_MIN_TOKENS == EVAL_DEFAULT_DETECTORS["loop_min_tokens"]
    assert loops.LOOP_REP16 == EVAL_DEFAULT_DETECTORS["loop_rep16"]
    assert (loops.LOOP_NGRAM, loops.LOOP_TAIL_TOKENS) == (16, 1024)


def test_the_evaluation_cases():
    """The cases of the evaluation's own detector test."""
    clean, late = SEQUENCES["clean_16k"], SEQUENCES["eval_late_loop"]
    assert loops.tail_rep16(clean) == 0.0 and loops.detect_loop(clean) == (False, -1)
    assert loops.tail_rep16(late) >= 0.5 and loops.is_loop(late)
    # Too short to call, even though the ids repeat.
    assert not loops.is_loop(late[-300:])


def test_short_responses_are_never_loops():
    assert loops.detect_loop([]) == (False, -1)
    assert loops.detect_loop([1, 2, 3] * 170) == (False, -1)  # 510 ids
    assert loops.detect_loop([1, 2, 3] * 171)[0]  # 513 ids
    assert loops.loop_start([]) == -1 and loops.loop_start(list(range(16))) == -1
    # Without the length gate the repeating text is still found, from its first id.
    assert loops.loop_start([1, 2, 3] * 100) == 0


def test_start_of_an_exact_loop():
    rng = random.Random(1)
    prefix = noise(rng, 3000)
    period = list(range(100, 137))
    ids = prefix + (period * 200)[:2000]
    # The start is the first id of the first copy of the repeated block.
    assert loops.detect_loop(ids) == (True, 3000)


def test_start_of_the_evaluation_late_loop():
    """The late loop of the evaluation's test is a short loop, three random ids and a long loop.

    The three random ids make 18 unique 16-grams, fewer than the 65 repeated 16-grams of the short
    loop before them, so the whole degenerate stretch counts as one loop from its first id.
    """
    assert loops.detect_loop(SEQUENCES["eval_late_loop"]) == (True, 14500)


def test_start_of_a_loop_that_began_before_the_tail_window():
    """The tail window only decides that the response loops. The start comes from the whole response,
    so a loop of 6,000 ids is found from its first id and not from the start of the last 1,024."""
    looped, start = loops.detect_loop(SEQUENCES["long_loop"])
    assert looped and start == 2000 < len(SEQUENCES["long_loop"]) - loops.LOOP_TAIL_TOKENS


def test_start_of_a_loop_that_fills_only_part_of_the_window():
    assert loops.detect_loop(SEQUENCES["half_window"]) == (True, 9000)


def test_start_of_a_loop_with_a_counter():
    """A loop that changes one id per period (step 1, step 2, ...) still repeats most of its 16-grams."""
    rng = random.Random(2)
    prefix = noise(rng, 3000)
    block = list(range(200, 260))
    ids = prefix + [token for k in range(40) for token in block + [70000 + k]]
    assert loops.tail_rep16(ids) > 0.5
    assert loops.detect_loop(ids) == (True, 3000)


def test_an_early_repeat_does_not_pull_the_start_back():
    """Ordinary text that writes the same 40 ids twice, such as an equation, stays outside the loop,
    even when the loop repeats those ids as well."""
    rng = random.Random(3)
    equation = list(range(400, 440))
    text = noise(rng, 200) + equation + noise(rng, 100) + equation + noise(rng, 3000)
    ids = text + ((equation + list(range(500, 530))) * 50)[:2000]
    assert loops.detect_loop(ids) == (True, len(text))


def test_a_loop_in_the_middle_is_not_a_loop():
    """A response that repeated itself for a while and then went on with new text is not a loop,
    because the tail test looks only at the end."""
    rng = random.Random(4)
    ids = noise(rng, 2000) + (list(range(100, 150)) * 30)[:1500] + noise(rng, 1200)
    assert loops.detect_loop(ids) == (False, -1)


def test_ids_in_a_tuple():
    ids = SEQUENCES["half_window"]
    assert loops.detect_loop(tuple(ids)) == loops.detect_loop(ids) == (True, 9000)


def test_mask_loop_tail_places_the_zeros():
    mask = [1] * 10
    assert loops.mask_loop_tail(mask, 6) == [1] * 6 + [0] * 4
    assert mask == [1] * 10  # the input is not changed
    assert loops.mask_loop_tail(mask, -1) == mask
    # A start at or past the end leaves nothing to mask.
    assert loops.mask_loop_tail(mask, 10) == mask
    # A loop from the first id keeps that id, so no micro-batch is left without a trained token.
    assert loops.mask_loop_tail(mask, 0) == [1] + [0] * 9
    # Zeros that were there before, such as a token that trimming left with mask 0, stay zeros.
    assert loops.mask_loop_tail([1, 0, 1, 1, 1], 3) == [1, 0, 1, 0, 0]
    assert loops.mask_loop_tail([], 0) == []


def test_mask_loop_tail_with_an_end():
    """A forced response passes the index of its first inserted id as ``end``. Only the positions from
    ``start`` up to that index are masked, and everything from ``end`` on keeps its value."""
    mask = [1] * 10
    assert loops.mask_loop_tail(mask, 3, 7) == [1] * 3 + [0] * 4 + [1] * 3
    assert mask == [1] * 10  # the input is not changed
    # Without an end the zeros run to the end of the mask, as before.
    assert loops.mask_loop_tail(mask, 3, None) == loops.mask_loop_tail(mask, 3) == [1] * 3 + [0] * 7
    # An end past the mask is cut to its length.
    assert loops.mask_loop_tail(mask, 6, 50) == [1] * 6 + [0] * 4
    # A loop from the first id still keeps that id.
    assert loops.mask_loop_tail(mask, 0, 4) == [1] + [0] * 3 + [1] * 6
    assert loops.mask_loop_tail(mask, 0, 1) == mask
    # Nothing is masked when the start is negative or not before the end.
    assert loops.mask_loop_tail(mask, -1, 5) == mask
    assert loops.mask_loop_tail(mask, 5, 5) == mask
    assert loops.mask_loop_tail(mask, 7, 5) == mask
    # The zeros of the inserted ids after the end stay zeros, and the answer after them stays 1.
    assert loops.mask_loop_tail([1, 1, 1, 1, 0, 0, 1, 1], 2, 4) == [1, 1, 0, 0, 0, 0, 1, 1]


def test_mask_loop_tail_of_a_loop_that_stops_at_an_end():
    """The loop of a forced response's thinking is masked from its tail mask start to the end of the
    thinking. The ids after the thinking keep their mask."""
    thinking = SEQUENCES["long_loop"]
    looped, start = loops.detect_loop(thinking)
    cut = loops.tail_mask_start(thinking, start)
    mask = [1] * len(thinking) + [0] * 5 + [1] * 100
    masked = loops.mask_loop_tail(mask, cut, end=len(thinking))
    assert looped and len(masked) == len(mask)
    assert masked == [1] * cut + [0] * (len(thinking) - cut) + [0] * 5 + [1] * 100


@pytest.mark.parametrize("period", [37, 13])
def test_the_tail_mask_starts_one_ngram_into_the_first_repeated_copy(period):
    """For a block of p ids repeated from s, the first repeated copy starts at s + p. Its first
    16-gram stays in the loss, so the mask begins at s + p + 16. A period shorter than 16 ids works
    the same way."""
    rng = random.Random(5)
    prefix = noise(rng, 3000)
    ids = prefix + (list(range(100, 100 + period)) * 200)[:2000]
    looped, start = loops.detect_loop(ids)
    assert looped and start == len(prefix)
    assert loops.tail_mask_start(ids, start) == len(prefix) + period + loops.LOOP_NGRAM


def test_the_tail_mask_start_without_a_loop_and_after_an_early_repeat():
    assert loops.tail_mask_start(SEQUENCES["clean_16k"], -1) == -1
    # The loop's first copy opens with an equation that the ordinary text already wrote. Its first
    # 16-gram is then already a repeat, so the mask begins one n-gram after the loop's start.
    rng = random.Random(3)
    equation = list(range(400, 440))
    text = noise(rng, 200) + equation + noise(rng, 100) + equation + noise(rng, 3000)
    ids = text + ((equation + list(range(500, 530))) * 50)[:2000]
    assert loops.tail_mask_start(ids, len(text)) == len(text) + loops.LOOP_NGRAM


def test_mask_loop_tail_on_a_detected_loop():
    ids = SEQUENCES["long_loop"]
    looped, start = loops.detect_loop(ids)
    cut = loops.tail_mask_start(ids, start)
    mask = loops.mask_loop_tail([1] * len(ids), cut)
    # long_loop repeats a block of 50 ids from index 2000.
    assert looped and (start, cut) == (2000, 2000 + 50 + loops.LOOP_NGRAM)
    assert len(mask) == len(ids)
    assert mask[:cut] == [1] * cut and set(mask[cut:]) == {0}


@pytest.mark.parametrize(
    "value, expected",
    [(None, False), ("", False), ("0", False), ("false", False), ("No", False), ("off", False),
     ("1", True), ("true", True), ("True", True), (" yes ", True), ("ON", True)],
)
def test_the_switch(value, expected):
    environ = {} if value is None else {loops.MASK_LOOP_TAIL_ENV: value}
    assert loops.mask_loop_tail_enabled(environ) is expected


def test_the_switch_refuses_unclear_values():
    with pytest.raises(ValueError, match=loops.MASK_LOOP_TAIL_ENV):
        loops.mask_loop_tail_enabled({loops.MASK_LOOP_TAIL_ENV: "ture"})


def test_the_switch_is_off_without_the_variable(monkeypatch):
    monkeypatch.delenv(loops.MASK_LOOP_TAIL_ENV, raising=False)
    assert loops.mask_loop_tail_enabled() is False
    monkeypatch.setenv(loops.MASK_LOOP_TAIL_ENV, "true")
    assert loops.mask_loop_tail_enabled() is True
