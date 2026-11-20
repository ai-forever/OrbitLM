"""Scores: lm-eval results per benchmark, the diagnostics and gates of chat suites, and the report."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .benchmarks import BENCHMARKS, Benchmark, CHAT_SERIES_MODE
from .files import chat_records_path, find_results_file, read_json
from .plan import EvalProcess
from .suite import ChatSettings


@dataclass
class Scores:
    """Aggregated scores of one evaluation, ready to report."""

    benchmarks: dict[str, float] = field(default_factory=dict)
    parts: dict[str, float] = field(default_factory=dict)
    ruler: dict[str, float] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=dict)
    incomplete: dict[str, list[str]] = field(default_factory=dict)
    # Chat suites only: per (benchmark, mode) diagnostics, keyed "<bench>_<mode>", and the gates.
    diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)
    gates: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "benchmarks": self.benchmarks,
            "parts": self.parts,
            "ruler": self.ruler,
            "samples": self.samples,
            "incomplete": self.incomplete,
        }
        # Written only when present, so the summary of a base suite has no chat keys.
        if self.diagnostics:
            payload["diagnostics"] = self.diagnostics
        if self.gates:
            payload["gates"] = self.gates
        return payload


def _metric_value(row: dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        # RULER writes -1 for a length that the process did not build.
        return None
    return float(value)


def _effective_samples(results: dict[str, Any], task: str) -> Optional[int]:
    entry = (results.get("n-samples") or {}).get(task)
    if isinstance(entry, dict):
        value = entry.get("effective", entry.get("original"))
    else:
        value = entry
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return int(value)


def _weighted_mean(records: list[tuple[int, float]]) -> Optional[float]:
    total = sum(n for n, _ in records)
    if total <= 0:
        return None
    return sum(n * value for n, value in records) / float(total)


def score_processes(processes: list[tuple[EvalProcess, Optional[dict[str, Any]]]]) -> Scores:
    """Computes the benchmark scores from the lm-eval results of every process.

    Each benchmark score is the mean over its subtasks weighted by the number of documents that
    were evaluated. A benchmark is reported only when every one of its units produced its metric,
    because a mean over part of the subjects would silently move the curve. RULER is judged per
    length instead: a length is reported when its eight tasks all finished, and the mean over
    lengths only when every length did.

    Several units in one process can share a task (two shards of HellaSwag on the same rank); the
    process then has one results row for it, which is counted once.

    For a sharded task the document count comes from the process's own document list, which is
    exact by construction. lm-eval's "n-samples" is not used there: the pinned release reports the
    documents it scored, but older releases reported the size of the whole test set for every
    shard, and the weights must not depend on which of the two behaviours runs.
    """
    records: dict[str, list[tuple[str, int, float]]] = {}
    ruler_records: dict[int, list[tuple[str, int, float]]] = {}
    incomplete: dict[str, list[str]] = {}
    ruler_failed_lengths: set[int] = set()
    for process, results in processes:
        counted: set[tuple[str, Optional[int]]] = set()
        documents = process.samples or {}
        for unit in process.units:
            benchmark = BENCHMARKS[unit.benchmark]
            if not benchmark.scored:
                # The chat probe and the no-prefill probe feed the gates (score_records) and have no
                # accuracy to report.
                continue
            metric = f"{unit.length},none" if unit.length is not None else benchmark.metric
            for task in unit.tasks:
                row = (results or {}).get("results", {}).get(task) if results else None
                value = _metric_value(row, metric) if isinstance(row, dict) else None
                if task in documents:
                    samples = len(documents[task]) or None
                else:
                    samples = _effective_samples(results, task) if results else None
                if value is None or samples is None:
                    incomplete.setdefault(unit.benchmark, []).append(unit.name)
                    if unit.length is not None:
                        ruler_failed_lengths.add(unit.length)
                    continue
                key = (task, unit.length)
                if key in counted:
                    continue
                counted.add(key)
                if unit.length is not None:
                    ruler_records.setdefault(unit.length, []).append((task, samples, value))
                else:
                    records.setdefault(unit.benchmark, []).append((task, samples, value))

    scores = Scores(incomplete={name: sorted(set(units)) for name, units in incomplete.items()})
    for name, rows in records.items():
        if name in incomplete:
            continue
        mean = _weighted_mean([(n, v) for _, n, v in rows])
        if mean is None:
            continue
        scores.benchmarks[name] = mean
        scores.samples[name] = sum(n for _, n, _ in rows)
        for part, part_tasks in BENCHMARKS[name].parts:
            part_mean = _weighted_mean([(n, v) for task, n, v in rows if task in part_tasks])
            if part_mean is not None:
                scores.parts[part] = part_mean
    per_length: dict[int, float] = {}
    for length, rows in sorted(ruler_records.items()):
        if length in ruler_failed_lengths:
            continue
        mean = _weighted_mean([(n, v) for _, n, v in rows])
        if mean is not None:
            per_length[length] = mean
    for length, value in per_length.items():
        scores.ruler[str(length)] = value
    if per_length:
        scores.samples["ruler"] = sum(n for length in per_length for _, n, _ in ruler_records[length])
        if not ruler_failed_lengths:
            scores.ruler["mean"] = sum(per_length.values()) / len(per_length)
    return scores


# ---------------------------------------------------------------------------
# Records and gates of the chat evaluation
# ---------------------------------------------------------------------------
#
# Every chat process leaves one record per response (itr_chat.py writes them): the token facts of the
# generated ids, the finish reason, the answer's language and reflection markers, and the correctness.
# Rank 0 turns them into per-benchmark diagnostics and into the gates of the think/no_think switch
# (eval/in_training/README.md). The thresholds of the detectors are applied here, so
# a recalibration needs no change in the evaluation processes.

# A chat pause longer than this gets a suggestion in the log to evaluate less often. At eval_every 1000
# and about 8 s per SFT step (an estimate), 15 minutes is about 10% of the training time.
CHAT_PAUSE_WARNING_SECONDS = 15 * 60.0
# The two-sided confidence level of every gate bound.
GATE_ALPHA = 0.05


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed term by term over the shorter of the two tails."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q, log_n = math.log(p), math.log1p(-p), math.lgamma(n + 1)

    def term(i: int) -> float:
        return math.exp(log_n - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q)

    if k + 1 <= n - k:
        return min(1.0, sum(term(i) for i in range(k + 1)))
    return max(0.0, 1.0 - sum(term(i) for i in range(k + 1, n + 1)))


def _solve_decreasing(function: Callable[[float], float], target: float) -> float:
    """The p in [0, 1] where a decreasing function of p crosses target, by bisection."""
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        if function(middle) > target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def clopper_pearson(successes: int, n: int, alpha: float = GATE_ALPHA) -> tuple[float, float]:
    """The exact two-sided 1 - alpha interval of a binomial proportion (Clopper and Pearson, 1934).

    The upper bound is the p at which seeing at most ``successes`` has probability alpha / 2, and the
    lower bound the p at which seeing at least that many has probability alpha / 2. Both come from the
    binomial distribution itself, by bisection, so no statistics package is needed in the trainer.
    """
    if n <= 0:
        return 0.0, 1.0
    lower = 0.0 if successes <= 0 else _solve_decreasing(lambda p: _binomial_cdf(successes - 1, n, p), 1.0 - alpha / 2.0)
    upper = 1.0 if successes >= n else _solve_decreasing(lambda p: _binomial_cdf(successes, n, p), alpha / 2.0)
    return lower, upper


def smallest_n_that_can_pass(threshold: float, alpha: float = GATE_ALPHA) -> int:
    """The smallest sample whose lower bound can reach threshold at all, that is with every trial a success.

    With n successes out of n the lower bound is (alpha / 2) ** (1 / n). Below this size a success-rate
    gate cannot pass even at 100%, so it is reported as "insufficient", not as a failure.
    """
    if threshold <= 0.0:
        return 1
    if threshold >= 1.0:
        raise ValueError("a lower bound never reaches 1")
    return int(math.ceil(math.log(alpha / 2.0) / math.log(threshold)))


def read_chat_records(proc_dir: Path) -> Optional[list[dict[str, Any]]]:
    """The records of one chat process, or None when they are missing or unreadable."""
    try:
        text = chat_records_path(proc_dir).read_text(encoding="utf-8")
    except OSError:
        return None
    rows = []
    for line in text.splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                return None
    return rows


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / float(len(values)) if values else None


def _quantile(values: list[float], q: float) -> Optional[float]:
    """The q-quantile by the nearest-rank rule, so a median of token counts is itself a token count."""
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))])


def _is_loop(record: dict[str, Any], detectors: dict[str, float]) -> bool:
    """A loop: vLLM's repetition detection stopped it, or its tail repeats itself.

    The tail is the last 1,024 generated ids, so a loop that starts late in a long response is still
    seen; rep16 >= 0.5 there means that at least half of those 16-grams repeat earlier ones.
    """
    if record.get("finish") == "repetition":
        return True
    return record.get("n_gen", 0) >= detectors["loop_min_tokens"] and record.get("rep16_tail", 0.0) >= detectors["loop_rep16"]


def _is_loop_w4k(record: dict[str, Any], detectors: dict[str, float]) -> bool:
    return record.get("n_gen", 0) >= detectors["loop_min_tokens"] and record.get("rep16_w4k", 0.0) >= detectors["loop_rep16"]


def _is_short_think(record: dict[str, Any], detectors: dict[str, float]) -> bool:
    return bool(record.get("closed")) and (record.get("reasoning_len") or 0) < detectors["short_think_tokens"]


def _strict_correct(record: dict[str, Any]) -> float:
    """Correctness that counts a response cut by the budget or by repetition detection as wrong."""
    return float(record.get("correct") or 0.0) if record.get("finish") == "stop" else 0.0


def chat_diagnostics(rows: list[dict[str, Any]], benchmark: Benchmark, detectors: dict[str, float]) -> dict[str, float]:
    """The per-benchmark diagnostics of one chat benchmark in one mode (the eval_chat_* titles)."""
    mode = benchmark.chat_mode
    loops = [_is_loop(r, detectors) for r in rows]
    capped = [r.get("finish") == "length" for r in rows]
    values: dict[str, Optional[float]] = {
        "n": float(len(rows)),
        "trunc": _mean([float(c) for c in capped]),
        "trunc_noloop": _mean([float(c and not lp) for c, lp in zip(capped, loops)]),
        "loop": _mean([float(lp) for lp in loops]),
        "loop_w4k": _mean([float(_is_loop_w4k(r, detectors)) for r in rows]),
        "rep_stop": _mean([float(r.get("finish") == "repetition") for r in rows]),
        "format_fail": _mean([float(not r.get("format_ok")) for r in rows]),
        "len_p50": _quantile([r.get("n_gen", 0) for r in rows], 0.5),
        "len_p90": _quantile([r.get("n_gen", 0) for r in rows], 0.9),
        "answer_p50": _quantile([r.get("answer_len", 0) for r in rows if r.get("has_answer")], 0.5),
    }
    if mode in ("think", "no_prefill"):
        values["unclosed"] = _mean([float(not r.get("closed")) for r in rows])
        values["think_p50"] = _quantile([r.get("reasoning_len") or 0 for r in rows if r.get("closed")], 0.5)
    if mode == "no_think":
        values["markers"] = _mean([float(r.get("markers") or 0) for r in rows])
        values["opener"] = _mean([float(bool(r.get("opener"))) for r in rows])
        values["tags"] = _mean([float((r.get("n_open") or 0) + (r.get("n_close") or 0) > 0) for r in rows])
    if mode == "no_prefill":
        values["opens"] = _mean([float(bool(r.get("np_ok"))) for r in rows])
    determined = [r for r in rows if r.get("lang") and r.get("lang_expected")]
    values["lang_match"] = _mean([float(r["lang"] == r["lang_expected"]) for r in determined])
    if benchmark.scored:
        scored = [r for r in rows if r.get("correct") is not None]
        values["acc_strict"] = _mean([_strict_correct(r) for r in scored])
        if mode == "think":
            values["acc_closed"] = _mean([float(r["correct"]) for r in scored if r.get("closed")])
    return {key: float(value) for key, value in values.items() if value is not None}


def _failure_gate(failures: int, n: int, threshold: Optional[float]) -> dict[str, float]:
    _, upper = clopper_pearson(failures, n)
    gate = {"rate": failures / float(n), "ub95": upper, "n": float(n), "count": float(failures)}
    if threshold is not None:
        gate["pass"] = float(upper <= threshold)
    return gate


def _success_gate(successes: int, n: int, threshold: Optional[float]) -> dict[str, float]:
    lower, _ = clopper_pearson(successes, n)
    gate = {"rate": successes / float(n), "lb95": lower, "n": float(n), "count": float(successes)}
    if threshold is not None:
        gate["pass"] = float(lower >= threshold)
    return gate


@dataclass
class ChatScores:
    """What rank 0 derives from the records of one chat evaluation."""

    diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)
    gates: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_units: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def score_records(processes: list[tuple[EvalProcess, Optional[list[dict[str, Any]]]]],
                  chat: ChatSettings) -> ChatScores:
    """Turns the records of every chat process into diagnostics and gates.

    A benchmark counts only when every one of its units left records and their number is exactly the
    planned number of responses; anything else would move a rate without saying so. A gate is computed
    only when every benchmark that feeds it counts.
    """
    result = ChatScores()
    expected: dict[str, int] = {}
    found: dict[str, list[dict[str, Any]]] = {}
    withheld: set[str] = set()
    for process, records in processes:
        units = [unit for unit in process.units if BENCHMARKS[unit.benchmark].chat_mode is not None]
        for unit in units:
            expected[unit.benchmark] = expected.get(unit.benchmark, 0) + unit.docs
        if not units:
            continue
        if records is None:
            result.missing_units += [unit.name for unit in units]
            withheld.update(unit.benchmark for unit in units)
            continue
        owner = {task: unit.benchmark for unit in units for task in unit.tasks}
        for record in records:
            benchmark = owner.get(record.get("task"))
            if benchmark is not None:
                found.setdefault(benchmark, []).append(record)
    complete: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(expected):
        rows = found.get(name, [])
        if name in withheld:
            continue
        if len(rows) != expected[name]:
            result.notes.append(f"{name}: {len(rows)} records instead of {expected[name]}; its diagnostics and gates are withheld")
            continue
        complete[name] = rows

    detectors = chat.detectors
    for name, rows in complete.items():
        benchmark = BENCHMARKS[name]
        if rows:
            result.diagnostics[f"{benchmark.report_as}_{CHAT_SERIES_MODE[benchmark.chat_mode]}"] = chat_diagnostics(
                rows, benchmark, detectors)

    def pool(modes: tuple[str, ...]) -> Optional[list[dict[str, Any]]]:
        names = [name for name in expected if BENCHMARKS[name].chat_mode in modes]
        if not names or any(name not in complete for name in names):
            return None
        return [record for name in names for record in complete[name]]

    thresholds = chat.gates
    think = pool(("think",))
    no_think = pool(("no_think",))
    if think:
        result.gates["g2"] = _failure_gate(
            sum(_is_short_think(r, detectors) or bool(r.get("stop_inside")) for r in think), len(think), thresholds.get("g2_max"))
        cells: dict[str, list[float]] = {}
        for record in think:
            cell = f"{record.get('lang_expected') or 'xx'}_{record.get('domain') or 'other'}"
            cells.setdefault(cell, []).append(float(_is_short_think(record, detectors) or bool(record.get("stop_inside"))))
        result.gates["g2_cells"] = {cell: _mean(flags) for cell, flags in sorted(cells.items())}
    opened = pool(("no_prefill",))
    if opened:
        result.gates["g3"] = _success_gate(sum(bool(r.get("np_ok")) for r in opened), len(opened), thresholds.get("g3_min"))
    if think or no_think:
        g4: dict[str, float] = {}
        passes = []
        for label, rows in (("think", think), ("nothink", no_think)):
            if not rows:
                continue
            gate = _success_gate(sum(bool(r.get("format_ok")) for r in rows), len(rows), thresholds.get("g4_min"))
            g4.update({f"{label}_{key}": value for key, value in gate.items() if key != "pass"})
            if "pass" in gate:
                passes.append(gate["pass"])
        if passes:
            g4["pass"] = float(all(passes))
        result.gates["g4"] = g4
    if "math500_think" in complete:
        rows = complete["math500_think"]
        gate = _failure_gate(sum(_is_loop(r, detectors) for r in rows), len(rows), thresholds.get("g5_max"))
        gate["loop_or_trunc_rate"] = _mean([float(_is_loop(r, detectors) or r.get("finish") == "length") for r in rows])
        if think:
            gate["pooled_rate"] = _mean([float(_is_loop(r, detectors)) for r in think])
        result.gates["g5"] = gate
    if no_think:
        gate = _failure_gate(
            sum((r.get("n_open") or 0) + (r.get("n_close") or 0) > 0 for r in no_think), len(no_think),
            thresholds.get("g6_tags_max"))
        checks = [gate["pass"]] if "pass" in gate else []
        math_rows = complete.get("math500_nothink")
        if math_rows:
            gate["markers_mean"] = _mean([float(r.get("markers") or 0) for r in math_rows])
            gate["len_p50"] = _quantile([r.get("n_gen", 0) for r in math_rows], 0.5)
            if "g6_markers_max" in thresholds:
                checks.append(float(gate["markers_mean"] <= thresholds["g6_markers_max"]))
            if "g6_len_p50_max" in thresholds:
                checks.append(float(gate["len_p50"] <= thresholds["g6_len_p50_max"]))
        gate.pop("pass", None)
        if checks:
            gate["pass"] = float(all(checks))
        result.gates["g6"] = gate
    g7: dict[str, float] = {}
    g7_checks = []
    for name, key in (("math500_nothink", "g7_math500_min"), ("gsm8k_nothink", "g7_gsm8k_min")):
        rows = [r for r in complete.get(name, []) if r.get("correct") is not None]
        if rows:
            gate = _success_gate(int(sum(_strict_correct(r) for r in rows)), len(rows), thresholds.get(key))
            label = name.replace("_nothink", "")
            g7.update({f"{label}_rate": gate["rate"], f"{label}_lb95": gate["lb95"]})
            if "pass" in gate:
                g7_checks.append(gate["pass"])
    if g7:
        if g7_checks:
            g7["pass"] = float(all(g7_checks))
        result.gates["g7"] = g7
    languages = pool(("think", "no_think"))
    if languages:
        russian = [r for r in languages if r.get("lang_expected") == "ru"]
        determined = [r for r in russian if r.get("lang")]
        english = [r for r in languages if r.get("lang_expected") == "en" and r.get("lang")]
        if determined:
            gate = _success_gate(sum(r["lang"] == "ru" for r in determined), len(determined), thresholds.get("g11_min"))
        else:
            gate = {"n": 0.0}
        gate["n_ru_prompts"] = float(len(russian))
        if english:
            gate["en_rate"] = _mean([float(r["lang"] == "en") for r in english])
        if "g11_min" in thresholds and len(determined) < smallest_n_that_can_pass(thresholds["g11_min"]):
            # Even all-Russian answers could not lift the lower bound to the threshold.
            gate.pop("pass", None)
            gate["insufficient"] = 1.0
        result.gates["g11"] = gate
    return result


def score_process_folders(planned: list[EvalProcess], procs_root: Path,
                          chat: Optional[ChatSettings]) -> tuple[Scores, Optional[ChatScores]]:
    """Scores the processes of a plan from their folders under procs_root (procs/<process name>).

    For a chat suite the records are scored too, and their diagnostics and gates join the scores.
    """
    processes: list[tuple[EvalProcess, Optional[dict[str, Any]]]] = []
    for process in planned:
        results_file = find_results_file(procs_root / process.name)
        processes.append((process, read_json(results_file) if results_file else None))
    scores = score_processes(processes)
    if chat is None:
        return scores, None
    chat_scores = score_records(
        [(process, read_chat_records(procs_root / process.name)) for process in planned if process.is_chat], chat,
    )
    scores.diagnostics = chat_scores.diagnostics
    scores.gates = chat_scores.gates
    return scores, chat_scores


def report_eval_scalars(
    report: Callable[..., Any],
    clearml_logger: Any,
    tb_writer: Any,
    *,
    scores: Scores,
    step: int,
    weights: str,
    timings: dict[str, float],
    failures: dict[str, int],
) -> None:
    """Reports the scores to TensorBoard (and ClearML when it is enabled) under fixed titles.

    Each benchmark is the title ``eval_<benchmark>`` with the weights kind ('raw') as the series.
    RULER has one series per length plus the mean over lengths. ``report`` is runtime.report_scalar,
    which writes to TensorBoard, and to ClearML when it is enabled, at iteration = step.

    Chat benchmarks use their own titles, so that a comparison of an SFT task with a base task never
    puts a 0-shot chat line on the plot of a few-shot base line: the accuracy is ``eval_chat_<bench>``
    with ``<mode>_<weights>`` as the series, a diagnostic is ``eval_chat_<metric>`` with
    ``<bench>_<mode>_<weights>``, and a gate is ``eval_gate_<gate>`` with ``<statistic>_<weights>``.
    """
    for name, value in scores.benchmarks.items():
        benchmark = BENCHMARKS.get(name)
        if benchmark is not None and benchmark.chat_mode is not None:
            series = f"{CHAT_SERIES_MODE[benchmark.chat_mode]}_{weights}"
            report(clearml_logger, tb_writer, f"eval_chat_{benchmark.report_as}", series, value, step)
        else:
            report(clearml_logger, tb_writer, f"eval_{name}", weights, value, step)
    for name, value in scores.parts.items():
        report(clearml_logger, tb_writer, f"eval_{name}", weights, value, step)
    for length, value in scores.ruler.items():
        report(clearml_logger, tb_writer, "eval_ruler", f"{weights}_{length}", value, step)
    for key, metrics in scores.diagnostics.items():
        for metric, value in metrics.items():
            report(clearml_logger, tb_writer, f"eval_chat_{metric}", f"{key}_{weights}", value, step)
    for gate, statistics in scores.gates.items():
        for name, value in statistics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                report(clearml_logger, tb_writer, f"eval_gate_{gate}", f"{name}_{weights}", value, step)
    for name, value in timings.items():
        report(clearml_logger, tb_writer, "eval_timing", name, value, step)
    for name, value in failures.items():
        report(clearml_logger, tb_writer, "eval_failures", name, value, step)
    for sink in (clearml_logger, tb_writer):
        flush = getattr(sink, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception as exc:  # reporting must never stop training
                print(f"in_training_eval WARNING: flush of {type(sink).__name__} failed: {exc!r}", flush=True)
