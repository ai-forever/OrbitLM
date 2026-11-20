"""verl reward function for OrbitLM: the MATH check and a sandboxed code check, routed by ``data_source``.

The reward is chosen by the part of ``data_source`` before an optional "#" tag
(``prompts.reward_of``), so "orbit/math#no_think" gets the MATH check. verl groups the validation
metrics by the whole ``data_source``, which is how validation sets in different prompt modes are
reported separately. Rows of ``orbit/none`` have no gold answer, as most distillation prompts.
They score 0 and run no check, but their diagnostics are reported like those of any other row.

verl loads this module as ``pkg://orbit_verl.data.reward`` and calls ``compute_score`` once per
response, from the event loop of a reward worker. Imports are absolute so that the module also
works when ``custom_reward_function.path`` names the file instead. verl then loads it as a
standalone file, outside its package, where relative imports would fail.

Why MATH runs in a separate process: ``math_verify`` enforces its timeouts with ``signal.alarm``,
and Python allows signal handlers only in a process's main thread. verl runs reward code in worker
threads. There, ``math_answer.try_math_verify`` catches the resulting error and returns False, so
MATH rewards would silently lose symbolic equivalence (``0.5`` would no longer match
``\\frac{1}{2}``). Each task in the process pool below runs in the main thread of its own
process, where the alarm works. verl's own ``utils/reward_score/math_verify.py`` does the same.

What a code reward does (``data_source`` orbit/code): it takes the last ```python block of the
checked text (``code_sandbox.extract_code``) and runs it against the row's tests with
``code_sandbox.execute``, in a sandbox on this machine. The reward is 1 only when every test passes.
The row's ground truth is a JSON string with the tests and the judge settings, and
``parse_code_ground_truth`` describes its format.

Why code runs in threads and not in processes: ``execute`` spends almost all of its time waiting
for the sandbox's child process, and its own work is a few milliseconds of CPU per run. Threads are
therefore enough. They also need none of the care that the MATH pool takes with stuck workers,
because the sandbox enforces its own time limits. Each reward worker process creates its own pool.
By default the pool has ``max(2, available_cpus() // 8)`` threads. orbit_base.yaml runs 8 reward
workers, so together they run about one sandbox per CPU of the machine's quota.

Why the sandbox checks itself first: its protections depend on kernel features of the machine (user
namespaces, Landlock, seccomp). A model trained against a sandbox that does not hold could learn to
read the expected answers instead of solving the problems. When a process creates its pool, it runs
``check_sandbox`` once. If any protection is missing, the code reward raises RuntimeError.

That error alone does not stop a run. verl's V1 trainer catches any error of a reward call, logs
"Error in _run_prompt", and leaves that prompt's group out of the batch. The run stops only when a
training batch has no group left. A failing sandbox would therefore drop every code row from
training and validation while the run goes on with the other rows. This is why the launcher runs
``check_sandbox`` before training whenever the dataset has code rows, and stops with exit code 2
when it fails (``orbit_verl.launch.prepare.check_code_sandbox``).

Two rules come on top of the checks. They use what the agent loop recorded about the response
(see ``orbit_verl.data.response_info``):

1. A response cut by the length limit gets ``truncated_score`` (0 by default), whatever its answer.
   Without this rule, a response that keeps writing after its answer earns as much as one that
   stops, so RL teaches the model not to stop.
2. Only the text after the last ``</think>`` is checked. A response that ends inside a thinking
   block scores 0, and its code is not run. The MATH check takes the last ``\\boxed{}`` anywhere,
   which would credit an answer written inside the thinking block. Code has the same problem: a
   sketch inside the thinking block would otherwise count as the answer.

Under a thinking budget (``ORBIT_THINK_BUDGET``) the agent loop closes a training response's
thinking block by force and lets the model answer. Such a response scores like a response cut by
the length limit, unless ``forced_score_scale`` says otherwise: without the budget it would have run
on past the budget, and a positive reward for it would teach the model to think past the budget.
``compute_score`` explains the rule.

The reward rules are summarised in ``docs/verl/DESIGN.md``.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import multiprocessing
import threading
import weakref
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any

from orbit_verl.code_sandbox import Limits, available_cpus, execute, extract_code, reward_info, self_check
from orbit_verl.data.math_answer import math_reward
from orbit_verl.data.prompts import DATA_SOURCES, reward_of
from orbit_verl.data.response_info import ANSWER_KEY, FINISH_KEY, FORCED_KEY, LOOP_KEY, THINK_KEY

logger = logging.getLogger(__name__)

# A MATH check normally takes milliseconds. math_verify gives itself 5 s per parse and per verify,
# and the check calls it twice, so 30 s is reached only by a check that is truly stuck.
MATH_TIMEOUT_S = 30.0
MATH_POOL_WORKERS = 2

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()
# One semaphore per event loop. It keeps at most MATH_POOL_WORKERS checks in the pool at a time,
# so the timeout measures how long a check runs and not how long it waited in the queue.
_loop_slots: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _math_reward_in_subprocess(solution_str: str, ground_truth: str) -> float:
    """Run in a pool process: the MATH reward, in that process's main thread."""
    return float(math_reward(solution_str, ground_truth))


def _get_pool() -> ProcessPoolExecutor:
    """Return the shared MATH pool, creating it on first use or after it was discarded."""
    global _pool
    with _pool_lock:
        if _pool is None:
            # "spawn" starts each worker from a fresh interpreter. A forked worker would inherit
            # the reward worker's threads and locks, which is unsafe inside a Ray actor.
            _pool = ProcessPoolExecutor(
                max_workers=MATH_POOL_WORKERS, mp_context=multiprocessing.get_context("spawn")
            )
        return _pool


def _discard_pool(pool: ProcessPoolExecutor) -> None:
    """Stop a pool whose worker hung or died. The next MATH check creates a fresh pool.

    ``shutdown`` alone would wait for the stuck check to finish, which may be never, so the worker
    processes are terminated. The process list is read before ``shutdown``, because the pool
    forgets its processes during shutdown.
    """
    global _pool
    with _pool_lock:
        if _pool is pool:
            _pool = None
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)


def _slots_for(loop: asyncio.AbstractEventLoop) -> asyncio.Semaphore:
    slots = _loop_slots.get(loop)
    if slots is None:
        slots = asyncio.Semaphore(MATH_POOL_WORKERS)
        _loop_slots[loop] = slots
    return slots


async def _math_score(solution_str: str, ground_truth: str) -> float:
    """MATH reward computed in the process pool, with a timeout and recovery from a broken pool."""
    loop = asyncio.get_running_loop()
    async with _slots_for(loop):
        # The loop runs at most twice. When another check's timeout kills the pool, this check
        # fails with BrokenProcessPool through no fault of its own, so it gets one retry on a
        # fresh pool. A check that breaks the pool twice is scored 0.
        for attempt in range(2):
            pool = _get_pool()
            try:
                future = loop.run_in_executor(pool, _math_reward_in_subprocess, solution_str, ground_truth)
            except (BrokenProcessPool, RuntimeError):
                # The pool broke or was shut down between _get_pool and submit.
                _discard_pool(pool)
                continue
            try:
                return float(await asyncio.wait_for(future, MATH_TIMEOUT_S))
            except asyncio.TimeoutError:
                logger.warning(
                    "MATH reward timed out after %.0f s; scoring 0 and restarting the pool.", MATH_TIMEOUT_S
                )
                _discard_pool(pool)
                return 0.0
            except BrokenProcessPool:
                _discard_pool(pool)
                if attempt == 0:
                    continue
                logger.warning("MATH reward broke the process pool twice; scoring 0.")
                return 0.0
            except Exception as exc:  # noqa: BLE001
                # Any other error is raised by the reward code itself. One bad completion must not
                # stop a training run, so it scores 0, and the warning keeps the problem visible.
                logger.warning("MATH reward raised %s: %s; scoring 0.", type(exc).__name__, exc)
                return 0.0
    return 0.0


# orbit_base.yaml runs this many reward workers (reward.num_workers), and each worker process has
# its own sandbox pool. The default pool size divides the machine's CPUs by this number, so that all
# workers together run about one sandbox per CPU. reward_kwargs.code_threads overrides the default.
CODE_REWARD_WORKERS = 8
# The smallest default pool. On a machine with fewer than 16 CPUs the division gives 1 thread, and a
# second thread still lets one program start while another one runs.
MIN_CODE_THREADS = 2
# The float tolerance of a stdin row that does not set its own. LiveCodeBench and AtCoder rows need
# it: their expected fractional answers have more digits than a correct program prints, and the
# last digits differ.
CODE_FLOAT_TOLERANCE = 1e-6

_code_pool: ThreadPoolExecutor | None = None
_code_pool_lock = threading.Lock()

# The keys that a code ground truth may have, per kind. Anything else is refused, because an unknown
# key is most likely a typo of a setting, which would otherwise silently fall back to its default.
_CODE_TRUTH_KEYS = {
    "stdin": frozenset({"kind", "tests", "judge_imports", "float_tolerance", "time_s", "memory_mb"}),
    "assert": frozenset({"kind", "tests", "judge_imports", "entry_point", "context", "cases", "time_s", "memory_mb"}),
}


@dataclass(frozen=True)
class CodeTests:
    """One code row's tests and judge settings, read from its ground truth.

    The fields are the arguments that ``code_sandbox.execute`` takes for this row. ``tests`` is a
    list of ``{"input", "output"}`` cases for the stdin kind and Python test code for the assert kind.
    """

    kind: str
    tests: list[dict[str, str]] | str
    limits: Limits
    judge_imports: bool = True
    float_tolerance: float | None = None
    entry_point: str | None = None
    context: str | None = None


def _is_number(value: Any) -> bool:
    """True for a finite int or float.

    JSON true and false arrive as Python bools, and a bool is also an int, so bools are excluded.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_code_ground_truth(ground_truth: str) -> CodeTests:
    """Read a code row's ground truth, a JSON object with the tests and the judge settings.

    A stdin row runs the program once per case, with the case's input on stdin, and compares the
    program's output with the expected one::

        {"kind": "stdin", "tests": [{"input": str, "output": str}, ...], "judge_imports": bool,
         "float_tolerance": float | null, "time_s": number | null, "memory_mb": int | null}

    An assert row runs the program once, followed by Python test code in MBPP or HumanEval style::

        {"kind": "assert", "tests": str, "entry_point": str | null, "context": str | null,
         "judge_imports": bool, "cases": int | null, "time_s": number | null, "memory_mb": int | null}

    Only ``kind`` and ``tests`` are required. ``judge_imports`` defaults to true. The judges of
    LiveCodeBench and LeetCode add their imports and a higher recursion limit to every program,
    stdin programs included, so correct solutions may rely on them. A missing ``float_tolerance``
    means 1e-6, and an explicit null compares numbers exactly. ``time_s`` (CPU seconds) and
    ``memory_mb`` replace the defaults of ``Limits``, which are the agreed reward limits, and null
    keeps the default. ``cases`` is the number of test cases in an assert row. They all share one
    run, so the run gets ``time_s`` seconds per case, up to the cap of ``Limits.for_cases``.
    ``entry_point`` names the function that a HumanEval-style ``check(candidate)`` tests.
    ``context`` is trusted code, such as the HumanEval prompt, whose definitions the tests use
    instead of the program's.

    Anything else raises ValueError: invalid JSON, an unknown kind or key, a missing or empty test
    set, or a value of the wrong type. A misconfigured row raises instead of scoring 0, like an
    unknown ``data_source``, because a 0 would make it look like a hard problem. verl's V1 trainer
    catches the error and leaves that prompt's group out, with "Error in _run_prompt" in the log
    (see the module docstring). Extra keys inside a stdin case, such as LiveCodeBench's
    ``testtype``, are ignored, because they cannot change how the case is judged.
    """
    text = str(ground_truth)
    preview = repr(text if len(text) <= 200 else text[:200] + "...")
    try:
        spec = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"Code ground truth is not valid JSON ({exc}): {preview}") from None
    if not isinstance(spec, dict):
        raise ValueError(f"Code ground truth must be a JSON object: {preview}")
    kind = spec.get("kind")
    if not isinstance(kind, str) or kind not in _CODE_TRUTH_KEYS:
        raise ValueError(f"Code ground truth has kind {kind!r}; known kinds: {sorted(_CODE_TRUTH_KEYS)}: {preview}")
    unknown = sorted(set(spec) - _CODE_TRUTH_KEYS[kind])
    if unknown:
        raise ValueError(
            f"Code ground truth of kind {kind!r} has unknown keys {unknown}; "
            f"allowed keys: {sorted(_CODE_TRUTH_KEYS[kind])}: {preview}"
        )

    def invalid(problem: str) -> ValueError:
        return ValueError(f"Code ground truth of kind {kind!r} {problem}: {preview}")

    judge_imports = spec.get("judge_imports", True)
    if not isinstance(judge_imports, bool):
        raise invalid("needs true or false for judge_imports")
    overrides: dict[str, Any] = {}
    if spec.get("time_s") is not None:
        if not _is_number(spec["time_s"]) or spec["time_s"] <= 0:
            raise invalid("needs a positive number for time_s")
        overrides["time_s"] = float(spec["time_s"])
    if spec.get("memory_mb") is not None:
        if not isinstance(spec["memory_mb"], int) or isinstance(spec["memory_mb"], bool) or spec["memory_mb"] < 1:
            raise invalid("needs a positive integer for memory_mb")
        overrides["memory_mb"] = spec["memory_mb"]
    limits = Limits(**overrides)
    tests = spec.get("tests")

    if kind == "stdin":
        if not isinstance(tests, list) or not tests:
            raise invalid("needs a non-empty list of test cases")
        for case in tests:
            if not isinstance(case, dict) or not all(isinstance(case.get(key), str) for key in ("input", "output")):
                raise invalid("needs every test case to be an object with string input and output")
        tolerance = spec.get("float_tolerance", CODE_FLOAT_TOLERANCE)
        if tolerance is not None and (not _is_number(tolerance) or tolerance < 0):
            raise invalid("needs a non-negative number or null for float_tolerance")
        return CodeTests(
            kind=kind,
            tests=[{"input": case["input"], "output": case["output"]} for case in tests],
            limits=limits,
            judge_imports=judge_imports,
            float_tolerance=None if tolerance is None else float(tolerance),
        )

    if not isinstance(tests, str) or not tests.strip():
        raise invalid("needs its tests as non-empty Python source")
    entry_point = spec.get("entry_point")
    if entry_point is not None and not (isinstance(entry_point, str) and entry_point.isidentifier()):
        raise invalid("needs a Python identifier or null for entry_point")
    context = spec.get("context")
    if context is not None and not isinstance(context, str):
        raise invalid("needs a string or null for context")
    cases = spec.get("cases")
    if cases is not None:
        if not isinstance(cases, int) or isinstance(cases, bool) or cases < 1:
            raise invalid("needs a positive integer or null for cases")
        limits = limits.for_cases(cases)
    return CodeTests(
        kind=kind,
        tests=tests,
        limits=limits,
        judge_imports=judge_imports,
        entry_point=entry_point,
        context=context,
    )


def _code_pool_size(threads: Any) -> int:
    """Threads of the sandbox pool: ``code_threads`` from reward_kwargs, else a share of the machine's CPUs."""
    if threads is None:
        return max(MIN_CODE_THREADS, available_cpus() // CODE_REWARD_WORKERS)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError(f"reward_kwargs code_threads must be a positive integer or null, not {threads!r}.")
    return threads


def check_sandbox() -> None:
    """Run the sandbox's self-check and raise RuntimeError when any protection is missing here.

    The reward workers call it before their first code reward, and the launcher calls it before
    training (``orbit_verl.launch.prepare.check_code_sandbox``).
    """
    report = self_check()
    failed = sorted(name for name, value in report.items() if value is not True)
    if failed:
        raise RuntimeError(
            f"The code sandbox failed its self-check in this process: {', '.join(failed)} "
            f"(full report: {report}). Code rewards are refused, because a program in a sandbox that "
            "does not hold could earn its reward by reading files instead of solving the problem. "
            "See src/orbit_verl/code_sandbox/README.md."
        )


def _get_code_pool(threads: Any) -> ThreadPoolExecutor:
    """Return this process's sandbox pool, creating it on the first code reward.

    The sandbox's self-check runs once, right before the pool is created. It blocks the event loop
    for about as long as one sandbox run, and only once per process. While the check fails, no pool
    exists, so every later code reward runs the check again and raises again.

    ``threads`` counts only when the pool is created. reward_kwargs are the same for every call of
    a run, so a later call never asks for another size.
    """
    global _code_pool
    with _code_pool_lock:
        if _code_pool is None:
            size = _code_pool_size(threads)
            check_sandbox()
            _code_pool = ThreadPoolExecutor(max_workers=size, thread_name_prefix="orbit-code-reward")
            logger.info("Code sandbox self-check passed; code rewards run in %d threads.", size)
        return _code_pool


async def _code_score(text: str, tests: CodeTests, threads: Any) -> tuple[float, dict[str, float]]:
    """Run the program in ``text`` against ``tests`` in the sandbox pool.

    Returns the reward (1.0 when every test passed, else 0.0) and the code metrics of the run.
    """
    pool = _get_code_pool(threads)
    run = functools.partial(
        execute,
        extract_code(text),
        tests.tests,
        tests.kind,
        tests.limits,
        entry_point=tests.entry_point,
        test_context=tests.context,
        judge_imports=tests.judge_imports,
        float_tolerance=tests.float_tolerance,
    )
    try:
        result = await asyncio.get_running_loop().run_in_executor(pool, run)
    except Exception as exc:  # noqa: BLE001
        # execute reports every failure of the program in its result, so an exception comes from
        # the executor itself, for example when /tmp is full. One bad response must not stop a
        # training run, so it scores 0. The warning keeps the problem visible, and the metrics
        # count it as a sandbox error.
        logger.warning("Code reward raised %s: %s; scoring 0.", type(exc).__name__, exc)
        return 0.0, reward_info({"passed": False, "reason": "sandbox_error"})
    return float(bool(result["passed"])), reward_info(result)


# The keyword arguments that verl itself passes to a custom reward function, next to the ones of
# reward.custom_reward_function.reward_kwargs. Anything else is refused, so a misspelled reward kwarg
# fails the run instead of silently leaving its rule off.
VERL_REWARD_KWARGS = frozenset({"reward_router_address", "reward_model_tokenizer"})


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    truncated_score: float | None = 0.0,
    code_threads: int | None = None,
    forced_score_scale: float | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Score one response.

    For a code row the answer is the program in the response, and it matches when it passes every
    test of the row. ``ground_truth`` then holds the tests (``parse_code_ground_truth``).

    A response that the agent loop marked as cut by the length limit gets ``truncated_score``
    instead. ``None`` turns that rule off, and the response is scored on its text like any other.
    Reward shaping values come from ``reward.custom_reward_function.reward_kwargs`` in the config,
    and so does ``code_threads``, the size of this process's sandbox pool. ``None`` sizes the pool
    from the machine's CPUs (see the module docstring).

    Only the text after the last ``</think>`` is checked, and a response that ends inside a
    thinking block scores 0 without running any code. A response without the agent loop's fields,
    such as one from a test, is checked on its whole text.

    The reward is chosen by the part of ``data_source`` before an optional "#" tag. A
    ``orbit/none`` row has no gold answer: it scores 0 whatever its text, the truncation rule does
    not apply to it, and no check or program runs.

    The returned keys never vary. verl takes the key list from the first response of a batch and
    looks every key up in every other response, so a key missing from one response fails the step.
    The keys are ``score`` (the reward), ``acc`` (the same value, because verl's validation reports
    ``acc`` as its main metric), ``answer_correct`` (whether the checked text has the right answer,
    before the truncation rule), and ``truncated``,
    ``think_unclosed``, ``looped`` and ``think_forced`` (0 or 1). ``looped`` is 1 when the agent
    loop marked the response as caught in a repetition loop, and ``think_forced`` when it closed
    the thinking block by force. The
    keys of ``code_sandbox.REWARD_INFO_KEYS`` follow: ``code_passed``, ``code_shadowed``,
    ``code_cpu_s`` and one ``code_<reason>`` per way a program can fail. A batch may mix math and
    code rows, so every response returns them, with zeros when it ran no program. All keys but
    ``score`` and ``acc`` are diagnostics for the logs.

    A response whose thinking block the agent loop closed by force at the thinking budget (the
    ``orbit_forced`` field) scores as if the length limit had cut it, ``truncated_score``, unless
    ``forced_score_scale`` is set: then it scores its correctness times that scale. A scale above 0
    rewards answers that needed the forced close, so it also rewards thinking past the budget.

    The key ``think_forced`` reports the forced responses. Under std-normalised GRPO
    (``algorithm.norm_adv_by_std_in_grpo``) the size of the scale hardly matters. A
    ``forced_score_scale`` above 0 gives correct forced answers the full advantage in any group that
    has no correct answer which closed by itself, so a small scale does not damp the reward for
    thinking past the budget on the hardest prompts.

    An unknown ``data_source`` raises ValueError, and so does a code row whose ground truth is
    malformed. A silent 0 reward would look like a very hard dataset rather than a misconfigured one.
    The same goes for a keyword argument that neither the config nor verl defines.
    """
    unknown = sorted(set(kwargs) - VERL_REWARD_KWARGS)
    if unknown:
        raise ValueError(f"compute_score got unknown keyword arguments {unknown}; check reward_kwargs for a typo.")
    reward_name = reward_of(data_source)
    if reward_name is None:
        raise ValueError(
            f"Unknown data_source {data_source!r}; known sources: {sorted(DATA_SOURCES)}, each optionally "
            "followed by a '#' tag."
        )
    # A code row's tests are read first, so a malformed row raises even when this response would
    # score 0 without running its program.
    tests = parse_code_ground_truth(str(ground_truth)) if reward_name == "code" else None
    info = extra_info or {}
    truncated = info.get(FINISH_KEY) == "length"
    forced = bool(info.get(FORCED_KEY))
    think = info.get(THINK_KEY) or "none"
    answer = info.get(ANSWER_KEY)
    text = str(answer) if think == "closed" and answer is not None else str(solution_str)

    code_info = reward_info(None)
    if reward_name == "none" or think == "unclosed":
        # A row without a gold answer has nothing to check. A response that ends inside its
        # thinking block has no answer to check.
        correct = 0.0
    elif tests is not None:
        correct, code_info = await _code_score(text, tests, code_threads)
    else:
        correct = await _math_score(text, str(ground_truth))

    if reward_name == "none":
        # A row without a gold answer has no reward at all. Giving it truncated_score instead
        # would turn the truncation rule into a length penalty on prompts that have no task.
        outcome_score = 0.0
    elif truncated and truncated_score is not None:
        outcome_score = float(truncated_score)
    elif forced:
        if forced_score_scale is not None:
            outcome_score = correct * float(forced_score_scale)
        else:
            outcome_score = float(truncated_score) if truncated_score is not None else correct
    else:
        outcome_score = correct

    return {
        "score": outcome_score,
        "acc": outcome_score,
        "answer_correct": correct,
        "truncated": float(truncated),
        "think_unclosed": float(think == "unclosed"),
        "looped": float(bool(info.get(LOOP_KEY))),
        "think_forced": float(forced),
        **code_info,
    }
