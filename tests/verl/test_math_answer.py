"""CPU tests for orbit_verl.data.math_answer, the MATH check behind the orbit/math reward.

The module is a port of the RLTT reference's answer parsing, and the frozen pass-rate files and
every RL run were scored with it. These cases pin its behaviour, so a change to it fails here
before it changes a reward. The cases that need ``math_verify`` skip where it is not installed;
the verl env pins it. The tests call the functions in the main thread, where math_verify's
``signal.alarm`` timeouts work, as they do in the reward's process pool.
"""
from __future__ import annotations

import pytest

from orbit_verl.data import math_answer as ma

needs_math_verify = pytest.mark.skipif(not ma.MATH_VERIFY_AVAILABLE, reason="math_verify is not installed")


def test_the_last_boxed_answer_is_extracted_with_matching_braces():
    assert ma.extract_boxed_answer(r"first \boxed{7} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert ma.extract_boxed_answer(r"\boxed{ 42 }") == "42"
    assert ma.extract_boxed_answer(r"\boxed{\sqrt{x^{2}}}") == r"\sqrt{x^{2}}"
    assert ma.extract_boxed_answer("no box") is None
    assert ma.extract_boxed_answer("") is None
    # A box that never closes gives nothing, and the earlier closed box does not count either.
    assert ma.extract_boxed_answer(r"\boxed{1} and \boxed{2") is None


@pytest.mark.parametrize("raw,normalized", [
    (r"\dfrac{1}{2}", "(1)/(2)"),
    (r"\frac12", "(1)/(2)"),
    (r"-\frac{3}{4}", "(-3)/(4)"),
    ("3/4", "(3)/(4)"),
    (r"\text{apples}", "apples"),
    (r"50\%", "50"),
    (r"5^\circ", "5"),
    ("(C)", "c"),
    (r"\left( 1, 2 \right)", "(1,2)"),
    ("$x + 1$", "x+1"),
    (None, ""),
])
def test_normalize_answer(raw, normalized):
    assert ma.normalize_answer(raw) == normalized


def test_normalize_for_math_verify_keeps_the_structure():
    assert ma.normalize_for_math_verify(r"\dfrac{1}{2}") == r"\frac{1}{2}"
    assert ma.normalize_for_math_verify(r"\text{ cm}") == " cm"
    assert ma.normalize_for_math_verify(r"\frac12") == r"\frac{1}{2}"
    assert ma.normalize_for_math_verify(None) == ""


def test_equivalence_without_math_verify(monkeypatch):
    """The string and numeric stages alone, as on a python without math_verify."""
    monkeypatch.setattr(ma, "MATH_VERIFY_AVAILABLE", False)
    assert ma.check_math_answer(r"\frac{1}{2}", r"\dfrac12")
    assert ma.check_math_answer("1,000", "1000")
    assert ma.check_math_answer("42.0", "42")
    assert ma.check_math_answer("(C)", "C")
    assert not ma.check_math_answer("3", "4")
    assert not ma.check_math_answer("", "4")
    assert not ma.check_math_answer("4", "  ")
    assert not ma.check_math_answer(None, "4")
    # Symbolic equivalence needs math_verify, so 0.5 and 1/2 differ without it.
    assert not ma.check_math_answer("0.5", r"\frac{1}{2}")
    assert ma.try_math_verify("0.5", r"\frac{1}{2}") is False


@needs_math_verify
def test_equivalence_with_math_verify():
    assert ma.check_math_answer("0.5", r"\frac{1}{2}")
    # Bare LaTeX gets $ delimiters before parsing; without them math_verify extracts nothing.
    assert ma.try_math_verify(r"\sqrt3-\sqrt2", r"\sqrt{3}-\sqrt{2}")
    assert ma.try_math_verify(r"$\frac{2}{4}$", r"\frac{1}{2}")
    assert not ma.check_math_answer("0.4", r"\frac{1}{2}")


def test_math_reward_reads_the_last_box_of_the_completion():
    assert ma.math_reward(r"so \boxed{\frac12}", r"\dfrac{1}{2}") == 1.0
    assert ma.math_reward(r"answer \boxed{7}", "7") == 1.0
    assert ma.math_reward(r"first \boxed{3}, finally \boxed{7}", "7") == 1.0
    assert ma.math_reward(r"answer \boxed{8}", "7") == 0.0
    assert ma.math_reward("no box here", "7") == 0.0
    # A gold solution with a box is reduced to its boxed answer.
    assert ma.math_reward(r"answer \boxed{7}", r"The sum is $\boxed{7}$.") == 1.0
    # An empty gold answer never matches.
    assert ma.math_reward(r"\boxed{7}", "") == 0.0
    assert ma.math_reward(r"\boxed{7}", r"\boxed{}") == 0.0
