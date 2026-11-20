"""MATH answer extraction and equivalence, the check behind the ``orbit/math`` reward.

This is a faithful port of the RLTT reference implementation (jonwill8/RLTT,
``math_utils/answer_parsing.py``), so the verifier matches the paper's. The behaviour must not
change: the frozen pass-rate files and every RL run so far were scored with exactly these rules.

``check_math_answer`` tries four stages in order and accepts the first match:

1. ``math_verify`` on the raw strings;
2. ``math_verify`` on lightly normalised strings (``normalize_for_math_verify``);
3. an exact match of the fully normalised strings (``normalize_answer``);
4. a numeric match of the normalised strings.

``math_verify`` is optional. Without it only the last two stages run, so symbolic equivalence such
as ``0.5`` against ``\\frac{1}{2}`` is lost. The verl env pins it (``requirements/verl``).

``math_verify`` enforces its timeouts with ``signal.alarm``, which works only in the main thread of
a process. ``orbit_verl.data.reward`` therefore runs ``math_reward`` in a process pool. Called from
another thread, ``try_math_verify`` catches the resulting error and returns False.

At module level this file imports the standard library and, when it is installed, ``math_verify``
(which loads sympy).
"""
from __future__ import annotations

import re
from typing import Optional

try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig
    from math_verify import parse as math_parse
    from math_verify import verify as _math_verify

    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    math_parse = None
    _math_verify = None
    LatexExtractionConfig = None
    ExprExtractionConfig = None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Return the content of the last ``\\boxed{...}`` in ``text``, matching braces, or None."""
    if not text:
        return None
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    brace_count = 1
    pos = start
    while pos < len(text) and brace_count > 0:
        if text[pos] == "{":
            brace_count += 1
        elif text[pos] == "}":
            brace_count -= 1
        pos += 1
    if brace_count == 0:
        return text[start : pos - 1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalise LaTeX aggressively into a lowercase string without whitespace, for the string match."""
    if answer is None:
        return ""
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()
    answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\textbf\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\textit\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
    for cmd in ("\\!", "\\,", "\\;", "\\:"):
        answer = answer.replace(cmd, "")
    answer = answer.replace("\\ ", " ").replace("\\quad", " ").replace("\\qquad", " ")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\$", "")
    answer = answer.replace("\\%", "")
    answer = re.sub(r"(\d)\s*%", r"\1", answer)
    answer = re.sub(r"\^\\circ", "", answer)
    answer = answer.replace("\\circ", "")
    answer = re.sub(r"\\mbox\{[^}]*\}", "", answer)
    answer = re.sub(r"\\frac([^{])([^{])", r"\\frac{\1}{\2}", answer)

    def replace_frac(match: re.Match) -> str:
        prefix = match.group(1) if match.group(1) else ""
        numer, denom = match.group(2), match.group(3)
        return f"(-{numer})/({denom})" if prefix == "-" else f"({numer})/({denom})"

    frac_pattern = r"(-)?\\frac\{([^{}]*)\}\{([^{}]*)\}"
    for _ in range(5):
        new_answer = re.sub(frac_pattern, replace_frac, answer)
        if new_answer == answer:
            break
        answer = new_answer
    answer = re.sub(r"(?<!\()(-?\d+)/(\d+)(?!\))", r"(\1)/(\2)", answer)
    answer = answer.replace("$", "")
    answer = re.sub(r"^\(([A-E])\)$", r"\1", answer)
    answer = re.sub(r"\s+", "", answer)
    return answer.lower()


def normalize_for_math_verify(answer: str) -> str:
    """Normalise LaTeX lightly, keeping the structure that ``math_verify`` needs to parse it."""
    if answer is None:
        return ""
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()
    answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\textbf\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\textit\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\$", "").replace("\\%", "")
    answer = re.sub(r"(\d)\s*%", r"\1", answer)
    answer = re.sub(r"\\frac([^{])([^{])", r"\\frac{\1}{\2}", answer)
    return answer


def try_math_verify(pred: str, gold: str) -> bool:
    """Whether ``math_verify`` finds ``pred`` and ``gold`` mathematically equal; False without the library."""
    if not MATH_VERIFY_AVAILABLE:
        return False
    if pred is None or gold is None:
        return False
    pred, gold = str(pred), str(gold)
    if not pred.strip() or not gold.strip():
        return False
    try:
        # The LaTeX extractor needs delimiters. Bare LaTeX such as "\sqrt3-\sqrt2" parses to an
        # empty list without them, and verify would then reject a correct answer. Inputs that are
        # already delimited with $, \( or \[ are passed as they are.
        pred_latex = pred if pred.lstrip().startswith(("$", "\\(", "\\[")) else f"${pred}$"
        gold_latex = gold if gold.lstrip().startswith(("$", "\\(", "\\[")) else f"${gold}$"
        cfg = [LatexExtractionConfig(), ExprExtractionConfig()]
        pred_parsed = math_parse(pred_latex, extraction_config=cfg)
        gold_parsed = math_parse(gold_latex, extraction_config=cfg)
        # math_verify expects the gold answer first.
        return bool(_math_verify(gold_parsed, pred_parsed))
    except Exception:
        return False


def check_math_answer(pred: str, gold: str) -> bool:
    """Whether ``pred`` equals ``gold`` by any of the four stages of the module docstring."""
    if pred is None or gold is None:
        return False
    pred, gold = str(pred), str(gold)
    if not pred.strip() or not gold.strip():
        return False
    if try_math_verify(pred, gold):
        return True
    if try_math_verify(normalize_for_math_verify(pred), normalize_for_math_verify(gold)):
        return True
    pred_norm, gold_norm = normalize_answer(pred), normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    try:
        if abs(float(pred_norm.replace(",", "")) - float(gold_norm.replace(",", ""))) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    return False


def math_reward(completion: str, gold: str) -> float:
    """1.0 when the last boxed answer of ``completion`` equals the gold answer, else 0.0.

    ``gold`` may be the bare answer or a solution that contains ``\\boxed{}``. A completion without
    a boxed answer, or an empty gold answer, scores 0.
    """
    pred = extract_boxed_answer(completion)
    gold_ans = extract_boxed_answer(gold) if "\\boxed" in str(gold) else str(gold)
    if pred is None or not str(gold_ans).strip():
        return 0.0
    return 1.0 if check_math_answer(pred, gold_ans) else 0.0
