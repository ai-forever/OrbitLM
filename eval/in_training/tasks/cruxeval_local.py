"""Scoring and few-shot functions of the in-training CRUXEval tasks.

The few-shot examples and the execution of the predicted programs (pass_at_k) are those of the
stock lm-eval cruxeval task, imported from the installed lm-eval. The step that turns a generation
into the program to execute is defined here instead, because the stock one loses correct answers.
It strips the quotes from every predicted value, so a predicted string 'ab' turns into the name
ab, and the program fails even though the answer is right. Fed the true answers of all 800
problems, the stock step passes 427 of them on CRUXEval-O and 784 on CRUXEval-I.

The functions below read the generation with Python's ast module and keep every literal exactly
as the model wrote it, so the true answers pass all 800 problems of both tasks. They are the
scorer that produced the earlier CRUXEval numbers of OrbitLM, so those numbers stay comparable.
Upstream lm-eval PR #3770 fixes the same quote-stripping bug. Remove this
copy only after a pinned release with that fix passes the true-answers check on both tasks, 800
of 800 problems each.
"""
import ast

from lm_eval.tasks.cruxeval.utils import (
    list_fewshot_samples_input,
    list_fewshot_samples_output,
    pass_at_k,
)

# The task files reach these names through this module.
__all__ = [
    "build_predictions_input",
    "build_predictions_output",
    "list_fewshot_samples_input",
    "list_fewshot_samples_output",
    "pass_at_k",
]

# The errors with which ast.parse rejects a generation. A degenerate generation with very deep
# nesting makes CPython's parser raise MemoryError or RecursionError instead of SyntaxError, and a
# null byte raises ValueError. Left uncaught, any of them would end the whole task run. Each of
# them only means that this way of reading the generation does not apply, so the caller tries the
# next one.
PARSE_ERRORS = (SyntaxError, ValueError, MemoryError, RecursionError)


def _parse_single_statement(source):
    """The one statement that source consists of, or None when it is not exactly one statement."""
    try:
        tree = ast.parse(source)
    except PARSE_ERRORS:
        return None
    if len(tree.body) != 1:
        return None
    return tree.body[0]


def _source_of(node, source):
    """The text of node as it stands in source, so that a string keeps the quotes it was written with."""
    segment = ast.get_source_segment(source, node)
    if segment is not None:
        return segment
    return ast.unparse(node)


def extract_assert_rhs(source):
    """The right-hand side of a single 'assert <left> == <right>' statement, or None for anything else.

    Parsing with ast rather than splitting at "==" keeps an "==" inside a string literal intact.
    """
    statement = _parse_single_statement(source)
    if not isinstance(statement, ast.Assert):
        return None
    comparison = statement.test
    if not isinstance(comparison, ast.Compare):
        return None
    if len(comparison.ops) != 1 or len(comparison.comparators) != 1:
        return None
    if not isinstance(comparison.ops[0], ast.Eq):
        return None
    return _source_of(comparison.comparators[0], source)


def extract_bare_expression(source):
    """The text of source when it is a single Python expression, such as "abc" or [1, 'x'], or None."""
    statement = _parse_single_statement(source)
    if not isinstance(statement, ast.Expr):
        return None
    return _source_of(statement.value, source)


def extract_call_args(source, func_name="f"):
    """The argument text of the call to func_name in source, or None when there is no such call.

    It accepts the shapes in which a model answers an input prediction: a full assertion
    (assert f(1, 2) == 3), an assertion with the call on the right (assert 3 == f(1, 2)), and a
    bare call (f(1, 2)). The arguments are cut out of the original text, so a tuple argument keeps
    its own parentheses and a string keeps its quotes.
    """
    statement = _parse_single_statement(source)
    call = None
    if isinstance(statement, ast.Assert):
        test = statement.test
        if isinstance(test, ast.Call):
            call = test
        elif (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
        ):
            if isinstance(test.left, ast.Call):
                call = test.left
            elif isinstance(test.comparators[0], ast.Call):
                call = test.comparators[0]
    elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        call = statement.value
    if call is None:
        return None
    if not (isinstance(call.func, ast.Name) and call.func.id == func_name):
        return None
    parts = [_source_of(arg, source) for arg in call.args]
    parts += [f"{kw.arg}={_source_of(kw.value, source)}" for kw in call.keywords if kw.arg is not None]
    return ", ".join(parts)


def _answer_text(generation):
    """The generation without the [ANSWER] tag in front and without anything after [/ANSWER].

    Models write the tag in both task variants. The closing tag is usually gone already, because
    it is the stop string of the tasks.
    """
    if "[ANSWER]" in generation:
        generation = generation.split("[ANSWER]", 1)[1].strip()
        if "[/ANSWER]" in generation:
            generation = generation.split("[/ANSWER]", 1)[0].strip()
    return generation


def _first_line(text):
    """The first line of text that is not blank, or an empty string."""
    return next((line for line in text.split("\n") if line.strip()), "")


def extract_code_output(doc, generation):
    """The program that checks an output prediction: the function, then assert f(input) == prediction.

    The prediction is the right-hand side of the generated assertion, or else the first non-blank
    line read as an expression. Splitting at "==" and cutting a "#" comment is the last resort
    for malformed answers, because both break when the characters sit inside a string literal.
    """
    generation = _answer_text(generation)
    prediction = extract_assert_rhs(generation)
    if prediction is None:
        first_line = _first_line(generation)
        prediction = extract_bare_expression(first_line)
        if prediction is None and "==" in first_line:
            prediction = first_line.split("==", 1)[1].split("#", 1)[0].strip()
        elif prediction is None:
            prediction = first_line.split("#", 1)[0].strip()
    return f"{doc['code']}\nassert f({doc['input']}) == {prediction}"


def extract_code_input(doc, generation):
    """The program that checks an input prediction: the function, then assert f(prediction) == output.

    The prediction is read, in this order, from the whole generation as one statement with a call
    to f, from its first non-blank line as such a statement, and from that line as a bare
    argument expression (a model may answer just 16). Cutting the line at "#" and "==" is the
    last resort for malformed answers, because both break when the characters sit inside a string
    literal.
    """
    generation = _answer_text(generation)
    first_line = _first_line(generation)
    prediction = extract_call_args(generation)
    if prediction is None:
        prediction = extract_call_args(first_line)
    if prediction is None:
        prediction = extract_bare_expression(first_line)
    if prediction is None:
        text = first_line.split("#", 1)[0].strip()
        if "==" in text:
            text = text.split("==", 1)[0].strip()
        if text.startswith("assert "):
            text = text[len("assert "):].strip()
        if text.startswith("f(") and text.endswith(")"):
            # A slice rather than strip("()"), so that a tuple argument such as (1, 2) keeps its
            # own parentheses.
            text = text[2:-1].strip()
        prediction = text
    return f"{doc['code']}\nassert f({prediction})=={doc['output']}"


def build_predictions_output(resps, docs):
    """The filter of itr_cruxeval_output: one program to execute per generation."""
    return [[extract_code_output(doc, r) for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_input(resps, docs):
    """The filter of itr_cruxeval_input: one program to execute per generation."""
    return [[extract_code_input(doc, r) for r in resp] for resp, doc in zip(resps, docs)]
