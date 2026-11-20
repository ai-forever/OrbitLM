"""CPU tests of scripts/verl/watch.py: reading a run dir, the facts of dumped answers, the stop rules.

The script uses only the standard library and two modules of orbit_verl that do the same, so these
tests run anywhere with PYTHONPATH=<repo>/src. Every run dir here is synthetic. Its metrics.jsonl
lines have the shape of verl's file logger, {"step": N, "data": {...}}. Its dump files have the
shape of verl's ``_write_generations``: input, output, gts, score, step and uid, plus the reward's
keys in validation dumps. The last test writes a dump with verl's own writer when verl imports.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verl" / "watch.py"


def _load_script():
    """Import the script as a module. dataclasses look the module up in sys.modules, so it goes there first."""
    spec = importlib.util.spec_from_file_location("watch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


watch = _load_script()

LIMIT = 8192
# The SFT prompts as verl's dump decodes them: special tokens such as <|im_start|> are dropped, and
# <think> and </think> stay, because the Qwen3 tokenizer does not mark them as special.
THINK_PROMPT = ("user\nWhat is 2+2?\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
                "assistant\n<think>\n")
NO_THINK_PROMPT = ("user\nWhat is 2+2?\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
                   "assistant\n<think>\n\n</think>\n\n")
CHAT_PROMPT = "user\nWrite a haiku about rain.\nassistant\n<think>\n\n</think>\n\n"
CONFIG_YAML = f"""orbit:
  run_name: r
data:
  tokenizer: null
  max_prompt_length: 1024
  max_response_length: {LIMIT}
actor_rollout_ref:
  rollout:
    response_length: {LIMIT}
trainer:
  logger:
  - console
  - file
"""


def prose(words: int, seed: int = 0) -> str:
    """Text of distinct made-up words, so that no 16 units ever repeat."""
    rng = random.Random(seed)
    return " ".join(f"w{rng.randrange(10**9)}" for _ in range(words))


def loop_text(seed: int = 0) -> str:
    """A response that starts normally and then repeats one sentence until the limit, as a loop does."""
    return prose(200, seed) + " So the answer is 4, and we check it again." * 120


def answer(prompt: str, output: str, step: int = 1, **extra) -> dict:
    """One dumped answer with verl's keys."""
    return {"input": prompt, "output": output, "gts": "", "score": 0.0, "step": step, "uid": "u_0_0", **extra}


def think_answer(chars: int, closed: bool = True, loops: bool = False, seed: int = 0) -> str:
    body = loop_text(seed) if loops else ("x" * chars)
    return body + ("</think>\n\nThe answer is 4." if closed else "")


def forced_answer(thinking: str, after: str = "The answer is \\boxed{4}.") -> str:
    """A think answer whose thinking the agent loop closed at the thinking budget, as the dump decodes it."""
    return thinking + watch.STOP_THINKING_TEXT + after


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_row(step: int, *, kl: float = 0.30, grad: float = 0.07, clip: float = 0.05, longest: int = LIMIT,
              mean: float = 3000.0) -> dict:
    return {"step": step, "data": {
        "actor/distillation/loss": kl, "actor/grad_norm": grad, "response_length/mean": mean,
        "response_length/max": longest, "response_length/clip_ratio": clip, "critic/score/mean": 0.0,
    }}


def val_row(step: int, source: str, *, acc: float = 0.7, truncated: float = 0.1, unclosed: float = 0.1,
            looped: float = 0.02) -> dict:
    return {"step": step, "data": {
        f"val-core/{source}/acc/mean@1": acc, f"val-aux/{source}/truncated/mean@1": truncated,
        f"val-aux/{source}/think_unclosed/mean@1": unclosed, f"val-aux/{source}/looped/mean@1": looped,
        f"val-aux/{source}/answer_correct/mean@1": acc,
    }}


def rollout_rows(step: int, *, think_chars: int = 1000, think_loops: int = 0, rows_per_mode: int = 10) -> list[dict]:
    """One step's dump: ``rows_per_mode`` think and no-think answers, ``think_loops`` of the think answers looping."""
    rows = []
    for i in range(rows_per_mode):
        loops = i < think_loops
        rows.append(answer(THINK_PROMPT, think_answer(think_chars, closed=not loops, loops=loops, seed=i), step))
        rows.append(answer(NO_THINK_PROMPT, "The answer is \\boxed{4}.", step))
    return rows


def make_run(tmp_path: Path, *, steps: int = 0, metrics=None, val=None, rollouts=None, answers=None,
             config: str | None = CONFIG_YAML) -> Path:
    """A run dir with config.yaml, metrics.jsonl and the two dump dirs, built from the given rows.

    ``steps`` adds that many healthy training steps unless ``metrics`` gives the training rows.
    ``rollouts`` maps a step to its dumped answers; by default every training step gets healthy ones.
    """
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (run / "config.yaml").write_text(config, encoding="utf-8")
    training = metrics if metrics is not None else [train_row(step) for step in range(1, steps + 1)]
    write_jsonl(run / "metrics.jsonl", [*(val or []), *training])
    if rollouts is None:
        rollouts = {step: rollout_rows(step) for step in range(1, steps + 1)}
    for step, rows in rollouts.items():
        write_jsonl(run / "rollouts" / f"{step}.jsonl", rows)
    for step, rows in (answers or {}).items():
        write_jsonl(run / "val_generations" / f"{step}.jsonl", rows)
    return run


def rule(report, words: str):
    (found,) = [result for result in report.rules if words in result.rule]
    return found


# --------------------------------------------------------------------------- facts of one answer


def test_the_mode_comes_from_the_end_of_the_decoded_prompt():
    assert watch.answer_mode(answer(THINK_PROMPT, "")) == "think"
    assert watch.answer_mode(answer(NO_THINK_PROMPT, "")) == "no_think"
    assert watch.answer_mode(answer(CHAT_PROMPT, "")) == "no_think"
    assert watch.answer_mode(answer("Question: 2+2?\nAnswer:", "")) == "unknown"
    # A dump that records the mode is believed over the prompt text.
    assert watch.answer_mode(answer("Question:", "", mode="think")) == "think"


def test_the_thinking_block_is_read_like_split_think():
    assert watch.think_block("reasoning</think>\n\nanswer", starts_inside=True) == (False, len("reasoning"))
    assert watch.think_block("reasoning that never ends", starts_inside=True) == (True, 25)
    # A <think> after the last </think> opens a block that is never closed.
    assert watch.think_block("a</think>b<think>c", starts_inside=True) == (True, 18)
    # A no-think answer has no block unless it opens one.
    assert watch.think_block("plain answer", starts_inside=False) == (False, None)
    assert watch.think_block("plain <think> again", starts_inside=False) == (True, 19)


def test_the_text_loop_estimate_finds_loops_and_only_loops():
    assert watch.text_loops(loop_text())
    assert not watch.text_loops(prose(3000))
    # The evaluation needs at least 512 tokens; a short repetition is not a loop.
    assert not watch.text_loops(" the same words again" * 100)
    assert not watch.text_loops("")


def test_flags_in_the_row_win_over_the_text():
    row = answer(THINK_PROMPT, "short</think>ok", truncated=1.0, looped=1.0, think_unclosed=0.0, score=0.0)
    fact = watch.row_facts(row)
    assert (fact.mode, fact.truncated, fact.looped, fact.looped_exact) == ("think", True, True, True)
    assert fact.unclosed is False
    plain = watch.row_facts(answer(THINK_PROMPT, loop_text()))
    assert (plain.truncated, plain.looped, plain.looped_exact, plain.unclosed) == (None, True, False, True)
    assert plain.think_chars == plain.chars
    assert watch.row_facts(answer(NO_THINK_PROMPT, "4")).think_chars is None


def test_groups_sum_and_summarise_answers():
    group = watch.Group()
    for row in rollout_rows(1, think_loops=2, rows_per_mode=4):
        group.add(watch.row_facts(row))
    summary = group.summary()
    assert summary["rows"] == 8 and summary["looped"] == pytest.approx(2 / 8)
    assert summary["unclosed"] == pytest.approx(2 / 8) and summary["looped_estimated"]
    # The training dump says nothing about truncation, so the share is unknown rather than 0.
    assert summary["truncated"] is None
    merged = group.merged(group).summary()
    assert merged["rows"] == 16 and merged["looped"] == summary["looped"]


def test_a_forced_answer_is_split_at_the_stop_thinking_sentence():
    # The splitter relies on the shape of the agent loop's text: a blank line, the sentence, then the close.
    assert watch.STOP_THINKING_TEXT == "\n\n" + watch.STOP_SENTENCE + "\n</think>\n\n"
    assert watch.STOP_SENTENCE.startswith("Considering the limited time by the user")
    assert watch.forced_split(forced_answer("x" * 500)) == ("x" * 500, "The answer is \\boxed{4}.")
    assert watch.forced_split(think_answer(500)) is None
    # A thinking that ends with a line break of its own keeps it; only the inserted blank line is cut off.
    assert watch.forced_split("a\n" + watch.STOP_THINKING_TEXT + "b") == ("a\n", "b")


def test_a_forced_answer_counts_only_its_own_thinking():
    text = forced_answer("x" * 500)
    fact = watch.row_facts(answer(THINK_PROMPT, text))
    assert (fact.mode, fact.forced, fact.unclosed, fact.think_chars) == ("think", True, False, 500)
    assert fact.chars == len(text)
    # An answer that closed its thinking by itself is not forced.
    assert watch.row_facts(answer(THINK_PROMPT, think_answer(500))).forced is False
    # The agent loop forces only modes that open a thinking block, so a no-think answer is never forced.
    assert watch.row_facts(answer(NO_THINK_PROMPT, text)).forced is False
    # The reward's think_forced flag wins over the text, as the other flags do.
    flagged = watch.row_facts(answer(THINK_PROMPT, text, think_forced=0.0))
    assert flagged.forced is False and flagged.think_chars == text.rfind("</think>")


def test_a_loop_before_a_forced_close_is_found_behind_a_long_answer():
    # The answer after the forced close is long enough to dilute the tail that the detector reads.
    text = forced_answer(loop_text(), prose(800, seed=7))
    assert not watch.text_loops(text)
    fact = watch.row_facts(answer(THINK_PROMPT, text))
    assert (fact.forced, fact.looped, fact.looped_exact) == (True, True, False)
    # An answer that loops after the forced close counts as well, as it does in the agent loop.
    assert watch.row_facts(answer(THINK_PROMPT, forced_answer(prose(300), loop_text(seed=3)))).looped
    assert not watch.row_facts(answer(THINK_PROMPT, forced_answer(prose(1500), prose(300, seed=1)))).looped


# --------------------------------------------------------------------------- reading the run dir


def test_yaml_scalar_reads_direct_children_only():
    text = CONFIG_YAML + "other:\n  data:\n    max_response_length: 1\n"
    assert watch.yaml_scalar(text, ("data", "max_response_length")) == str(LIMIT)
    assert watch.yaml_scalar(text, ("actor_rollout_ref", "rollout", "response_length")) == str(LIMIT)
    nested = "data:\n  inner:\n    max_response_length: 1\n  max_response_length: 2\n"
    assert watch.yaml_scalar(nested, ("data", "max_response_length")) == "2"
    assert watch.yaml_scalar(text, ("data", "missing")) is None
    assert watch.yaml_scalar(text, ("trainer", "logger")) is None


def test_the_response_limit_comes_from_the_option_then_config_yaml(tmp_path):
    run = make_run(tmp_path, steps=1)
    assert watch.response_limit(run, None) == (LIMIT, "config.yaml")
    assert watch.response_limit(run, 4096) == (4096, "--max-response-length")
    (run / "config.yaml").unlink()
    limit, source = watch.response_limit(run, None)
    assert limit is None and "--max-response-length" in source


def test_metrics_of_earlier_launches_are_merged_and_a_torn_line_is_skipped(tmp_path):
    run = tmp_path / "run"
    write_jsonl(run / "metrics.20260101-120000.jsonl", [train_row(1, kl=0.5), train_row(2, kl=0.4)])
    write_jsonl(run / "metrics.jsonl", [val_row(2, "orbit/math#think"), train_row(3, kl=0.3)])
    with open(run / "metrics.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"step": 4, "data": {"actor/distillation/lo')
    metrics = watch.read_metrics(run)
    assert sorted(metrics) == [1, 2, 3]
    # Step 2 holds the training metrics of the first launch and the validation of the resumed one.
    assert metrics[2]["actor/distillation/loss"] == 0.4
    assert metrics[2]["val-aux/orbit/math#think/truncated/mean@1"] == 0.1


def test_the_training_cut_share_is_exact_once_the_limit_is_known():
    metrics = {1: train_row(1, clip=0.25)["data"], 2: train_row(2, clip=0.002, longest=5000)["data"]}
    assert watch.train_truncated(metrics, LIMIT) == [(1, 0.25), (2, 0.0)]
    assert watch.train_truncated(metrics, None) == []


def test_only_complete_windows_count():
    points = [(step, float(step)) for step in range(1, 13)]
    assert watch.window_means(points, 5) == [(5, 3.0), (10, 8.0)]
    assert watch.window_end(1, 5) == 5 and watch.window_end(6, 5) == 10
    assert watch.rolling_mean([(1, 1.0), (2, 3.0), (3, 5.0)], 2) == [(2, 2.0), (3, 4.0)]


def test_validation_metrics_are_grouped_by_source_and_key(tmp_path):
    metrics = {0: {**val_row(0, "orbit/math#think")["data"], **val_row(0, "orbit/math#no_think", acc=0.5)["data"]},
               20: val_row(20, "orbit/math#think", acc=0.75)["data"]}
    found = watch.validation_metrics(metrics)
    assert found["orbit/math#think"]["acc"] == [(0, 0.7), (20, 0.75)]
    assert found["orbit/math#no_think"]["acc"] == [(0, 0.5)]
    assert set(found["orbit/math#think"]) == {"acc", "truncated", "think_unclosed", "looped"}


# --------------------------------------------------------------------------- the stop rules


def test_a_healthy_run_fires_no_rule(tmp_path, capsys):
    sources = ("orbit/math#think", "orbit/math#no_think")
    val = [val_row(step, source) for step in (0, 20, 40) for source in sources]
    answers = {step: rollout_rows(step) for step in (0, 20, 40)}
    run = make_run(tmp_path, steps=40, val=val, answers=answers)
    report = watch.build_report(run)
    # Rules 1-5 have data and stay quiet; rule 6 has no Russian prompts to read in this run.
    assert [result.status for result in report.rules] == ["ok"] * 5 + ["n/a"], report.rules
    assert len(report.steps) == 40 and report.steps[0]["truncated"] == 0.05
    assert report.steps[0]["modes"]["think"]["rows"] == 10
    assert {entry["source"] for entry in report.validation} == set(sources)
    assert watch.main([str(run)]) == watch.EXIT_OK
    out = capsys.readouterr().out
    assert "no stop rule fired" in out and "orbit/math#no_think" in out


def test_a_doubled_cut_share_fires_rule_1_with_a_floor(tmp_path):
    val = [val_row(0, "orbit/math#think", truncated=0.10), val_row(20, "orbit/math#think", truncated=0.12),
           val_row(40, "orbit/math#think", truncated=0.25)]
    result = rule(watch.build_report(make_run(tmp_path, steps=40, val=val)), "truncated")
    assert (result.status, result.step) == ("FIRED", 40) and "val orbit/math#think" in result.detail
    # From 0 to 0.01 is more than twice as much, but below the floor of 0.02.
    val = [val_row(0, "orbit/math#no_think", truncated=0.0), val_row(20, "orbit/math#no_think", truncated=0.01)]
    assert rule(watch.build_report(make_run(tmp_path / "b", steps=20, val=val)), "truncated").status == "ok"


def test_the_training_cut_share_fires_rule_1_by_windows(tmp_path):
    rows = [train_row(step, clip=0.05 if step <= 10 else 0.12) for step in range(1, 16)]
    result = rule(watch.build_report(make_run(tmp_path, metrics=rows, rollouts={})), "truncated")
    assert (result.status, result.step) == ("FIRED", 15) and "train all rows" in result.detail
    # Without the limit the training share is unknown, and with no validation the rule cannot run.
    result = rule(watch.build_report(make_run(tmp_path / "b", metrics=rows, rollouts={}, config=None)), "truncated")
    assert result.status == "n/a" and "response limit" in result.detail


def test_loops_rising_three_checks_fire_rule_2(tmp_path):
    shares = (0.01, 0.02, 0.03, 0.045)
    val = [val_row(20 * i, "orbit/math#think", looped=share) for i, share in enumerate(shares)]
    result = rule(watch.build_report(make_run(tmp_path, steps=60, val=val)), "looped")
    assert (result.status, result.step) == ("FIRED", 60)
    shares = (0.01, 0.02, 0.015, 0.03)
    val = [val_row(20 * i, "orbit/math#think", looped=share) for i, share in enumerate(shares)]
    assert rule(watch.build_report(make_run(tmp_path / "b", steps=60, val=val)), "looped").status == "ok"


def test_loops_in_the_training_rollouts_fire_rule_2_by_windows(tmp_path):
    # Windows of 5 steps; the loops per step of the think answers grow from window to window.
    loops = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    loops.update({step: 1 for step in range(6, 11)})
    loops.update({step: 2 for step in range(11, 16)})
    loops.update({step: 4 for step in range(16, 21)})
    rollouts = {step: rollout_rows(step, think_loops=count) for step, count in loops.items()}
    result = rule(watch.build_report(make_run(tmp_path, steps=20, rollouts=rollouts)), "looped")
    assert (result.status, result.step) == ("FIRED", 20) and "train think" in result.detail


def test_loops_before_forced_closes_fire_rule_2(tmp_path):
    # More forced answers loop in their thinking from window to window. Their answers are long enough
    # to hide the loop from a detector that would read the whole text.
    per_window = {5: 0, 10: 1, 15: 2, 20: 4}
    rollouts = {}
    for step in range(1, 21):
        rows = rollout_rows(step)
        for i in range(per_window[watch.window_end(step, 5)]):
            # The think answers sit at the even positions of a step's rows.
            rows[2 * i] = answer(THINK_PROMPT, forced_answer(loop_text(seed=i), prose(800, seed=100 + i)), step)
        rollouts[step] = rows
    result = rule(watch.build_report(make_run(tmp_path, steps=20, rollouts=rollouts)), "looped")
    assert (result.status, result.step) == ("FIRED", 20) and "train think" in result.detail


def test_forced_answers_get_their_own_column(tmp_path, capsys):
    rollouts = {}
    for step in range(1, 6):
        rows = rollout_rows(step)
        # Two of the ten think answers of every step reach the thinking budget.
        rows[0] = answer(THINK_PROMPT, forced_answer("x" * 1000), step)
        rows[2] = answer(THINK_PROMPT, forced_answer("x" * 1000), step)
        rollouts[step] = rows
    run = make_run(tmp_path, steps=5, rollouts=rollouts)
    modes = watch.build_report(run).steps[0]["modes"]
    assert modes["think"]["forced"] == pytest.approx(0.2) and modes["no_think"]["forced"] == 0.0
    # The forced answers are closed, and their thinking stops at the inserted sentence.
    assert modes["think"]["unclosed"] == 0.0 and modes["think"]["think_chars"] == pytest.approx(1000)
    capsys.readouterr()
    assert watch.main([str(run)]) == watch.EXIT_OK
    out = capsys.readouterr().out
    assert "  loop forced |" in out and " 0.200 |" in out


def test_a_grad_norm_burst_of_three_steps_fires_rule_3(tmp_path):
    rows = [train_row(step, grad=0.07) for step in range(1, 21)]
    rows += [train_row(21, grad=1.0), train_row(22, grad=1.0), train_row(23, grad=0.07)]
    result = rule(watch.build_report(make_run(tmp_path, metrics=rows, rollouts={})), "grad_norm")
    assert result.status == "ok"
    rows += [train_row(step, grad=0.9) for step in (24, 25, 26)]
    result = rule(watch.build_report(make_run(tmp_path / "b", metrics=rows, rollouts={})), "grad_norm")
    assert (result.status, result.step) == ("FIRED", 26) and "steps 24-26" in result.detail
    # With fewer than 20 steps the median of the steps so far stands in, and says so.
    rows = [train_row(step, grad=0.07) for step in range(1, 7)] + [train_row(s, grad=1.0) for s in (7, 8, 9)]
    result = rule(watch.build_report(make_run(tmp_path / "c", metrics=rows, rollouts={})), "grad_norm")
    assert result.status == "FIRED" and "median of only 9 steps" in result.detail


def test_the_high_grad_norm_of_the_first_steps_keeps_rule_3_quiet(tmp_path):
    # The start of a healthy 16K distillation run: high norms that decay within the first 5 steps.
    start = [12.66, 11.18, 9.58, 6.20, 3.95, 2.24, 1.81, 1.57, 1.41, 1.33]
    rows = [train_row(step, grad=grad) for step, grad in enumerate(start, 1)]
    rows += [train_row(step, grad=1.0) for step in range(11, 26)]
    result = rule(watch.build_report(make_run(tmp_path, metrics=rows, rollouts={})), "grad_norm")
    assert result.status == "ok", result


def test_longer_think_answers_fire_rule_4(tmp_path):
    def run_with(chars_late: int, where: Path) -> Path:
        rollouts = {step: rollout_rows(step, think_chars=1000 if step <= 5 else chars_late) for step in range(1, 11)}
        return make_run(where, steps=10, rollouts=rollouts)

    result = rule(watch.build_report(run_with(1200, tmp_path / "a")), "think answer length")
    assert result.status == "ok"
    result = rule(watch.build_report(run_with(1400, tmp_path / "b")), "think answer length")
    assert (result.status, result.step) == ("FIRED", 10) and "train think" in result.detail


def test_a_rising_teacher_kl_fires_rule_5(tmp_path):
    rows = [train_row(step, kl=0.30 + 0.01 * step) for step in range(1, 31)]
    result = rule(watch.build_report(make_run(tmp_path, metrics=rows, rollouts={})), "teacher KL")
    assert result.status == "FIRED" and "actor/distillation/loss" in result.detail
    # A noisy but flat KL does not rise 20 smoothed steps in a row.
    rng = random.Random(1)
    rows = [train_row(step, kl=0.3 + rng.uniform(-0.02, 0.02)) for step in range(1, 61)]
    assert rule(watch.build_report(make_run(tmp_path / "b", metrics=rows, rollouts={})), "teacher KL").status == "ok"
    # Without a distillation loss in the metrics the rule cannot run.
    rows = [{"step": step, "data": {"actor/grad_norm": 0.1}} for step in range(1, 30)]
    assert rule(watch.build_report(make_run(tmp_path / "c", metrics=rows, rollouts={})), "teacher KL").status == "n/a"


def test_exit_codes_json_and_follow(tmp_path, capsys):
    assert watch.main([str(tmp_path / "missing")]) == watch.EXIT_USAGE
    rows = [train_row(step, kl=0.30 + 0.01 * step) for step in range(1, 31)]
    run = make_run(tmp_path, metrics=rows, rollouts={})
    capsys.readouterr()
    assert watch.main([str(run), "--json"]) == watch.EXIT_STOP
    report = json.loads(capsys.readouterr().out)
    assert report["fired"] == ["teacher KL rising 20 steps in a row (mean over 5 steps)"]
    assert report["limit"] == LIMIT and len(report["steps"]) == 30
    # --follow returns as soon as a rule fires, so it never sleeps here.
    assert watch.main([str(run), "--follow", "3600", "--last", "3"]) == watch.EXIT_STOP
    assert "STOP RULE FIRED" in capsys.readouterr().out
    # --kl-smooth 1 checks the raw values, which also rise here.
    assert watch.main([str(run), "--kl-smooth", "1", "--json"]) == watch.EXIT_STOP


def test_validation_answers_give_exact_flags_and_lengths(tmp_path):
    answers = {0: [answer(THINK_PROMPT, "x" * 100 + "</think>4", truncated=0.0, looped=0.0, think_unclosed=0.0),
                   answer(THINK_PROMPT, "y" * 300, truncated=1.0, looped=1.0, think_unclosed=1.0)]}
    report = watch.build_report(make_run(tmp_path, steps=1, answers=answers))
    (entry,) = report.validation_answers
    assert (entry["mode"], entry["rows"], entry["truncated"], entry["looped"]) == ("think", 2, 0.5, 0.5)
    assert entry["looped_estimated"] is False and entry["think_chars"] == pytest.approx(200)


def test_a_forced_validation_answer_is_reported(tmp_path):
    # The agent loop never forces validation rows, so the reward's think_forced flag should be 0 there.
    answers = {0: [answer(THINK_PROMPT, forced_answer("x" * 100), think_forced=1.0),
                   answer(THINK_PROMPT, "y" * 100 + "</think>4", think_forced=0.0)]}
    report = watch.build_report(make_run(tmp_path, steps=1, answers=answers))
    (entry,) = report.validation_answers
    assert entry["forced"] == 0.5
    assert any("never forces validation rows" in note for note in report.notes)
    clean = {0: [answer(THINK_PROMPT, "y" * 100 + "</think>4", think_forced=0.0)]}
    report = watch.build_report(make_run(tmp_path / "b", steps=1, answers=clean))
    assert not any("validation rows" in note for note in report.notes)


def test_a_dump_written_by_verls_own_writer_is_read(tmp_path):
    trainer_base = pytest.importorskip("verl.trainer.ppo.v1.trainer_base")
    extra = {"truncated": [0.0, 1.0], "looped": [0.0, 1.0], "think_unclosed": [0.0, 1.0], "acc": [1.0, 0.0],
             "uid": ["a_0_0", "b_0_0"]}
    trainer_base.PPOTrainer._write_generations(
        [THINK_PROMPT, NO_THINK_PROMPT], ["r</think>4", "z" * 50], ["4", "4"], [1.0, 0.0], extra,
        str(tmp_path / "val_generations"), 20)
    steps = watch.read_dump_dir(tmp_path / "val_generations")
    assert sorted(steps) == [20]
    assert steps[20]["think"].summary()["truncated"] == 0.0
    assert steps[20]["no_think"].summary()["looped"] == 1.0



def test_the_russian_stop_sentence_is_recognised():
    """Russian rows get the Russian stop-thinking text, and the watcher splits them at it as well."""
    from orbit_verl.data.response_info import STOP_THINKING_TEXT_RU

    text = "Рассуждаю о задаче" + STOP_THINKING_TEXT_RU + "Ответ: 3."
    assert watch.forced_split(text) == ("Рассуждаю о задаче", "Ответ: 3.")



# --------------------------------------------------------------------------- the Russian guard (rule 6)

RU_THINK_PROMPT = ("user\nВ классе 12 учеников. Сколько учеников останется, если 3 уйдут?\n"
                   "Please reason step by step, and put your final answer within \\boxed{}.\nassistant\n<think>\n")
RU_ANSWER = "Ответ: в классе останется девять учеников, потому что двенадцать минус три равно девяти. $\\boxed{9}$"
EN_ANSWER = "The answer is that nine students remain in the class, since twelve minus three is nine. $\\boxed{9}$"


def ru_rows(step: int, *, english: int = 0, rows: int = 10) -> list[dict]:
    """``rows`` think answers to a Russian prompt, the first ``english`` of them answering in English."""
    thinking = "Считаем учеников: было двенадцать, ушли трое, значит осталось девять. "
    return [answer(RU_THINK_PROMPT, thinking + "</think>\n\n" + (EN_ANSWER if i < english else RU_ANSWER), step)
            for i in range(rows)]


def test_the_cyrillic_share_reads_the_prose_only():
    # The Latin letters of a formula would otherwise lower the share of every Russian math answer.
    assert watch.cyrillic_share("Для вычисления интеграла $$\\int_1^\\infty \\frac{dx}{x^2(e^x+1)}$$ сделаем замену") == 1.0
    assert watch.cyrillic_share("$$\\int_1^\\infty \\frac{dx}{x^2}$$") is None
    assert watch.cyrillic_share("The answer is nine students in the class.") == 0.0


def test_a_russian_prompt_is_recognised_before_the_english_instruction():
    assert watch.russian_prompt(RU_THINK_PROMPT)
    assert not watch.russian_prompt(THINK_PROMPT)


def test_a_russian_answer_reports_the_language_of_its_thinking_and_answer():
    fact = watch.row_facts(ru_rows(1)[0])
    assert fact.russian and fact.ru_think == 1.0 and fact.ru_answer == 1.0
    fact = watch.row_facts(ru_rows(1, english=1)[0])
    assert fact.ru_think == 1.0 and fact.ru_answer == 0.0
    english = watch.row_facts(answer(THINK_PROMPT, think_answer(500)))
    assert not english.russian and english.ru_think is None and english.ru_answer is None


def test_a_forced_russian_answer_is_read_after_the_russian_stop_sentence():
    from orbit_verl.data.response_info import STOP_THINKING_TEXT_RU

    text = "Считаем учеников: было двенадцать, ушли трое." + STOP_THINKING_TEXT_RU + RU_ANSWER
    fact = watch.row_facts(answer(RU_THINK_PROMPT, text))
    assert fact.forced and fact.ru_think == 1.0 and fact.ru_answer == 1.0


def test_answers_switching_to_english_fire_rule_6(tmp_path, capsys):
    rollouts = {step: rollout_rows(step) + ru_rows(step, english=0 if step <= 5 else 6) for step in range(1, 11)}
    run = make_run(tmp_path, steps=10, rollouts=rollouts)
    report = watch.build_report(run)
    fired = rule(report, "Cyrillic share")
    assert fired.status == "FIRED" and fired.step == 10, fired
    assert report.steps[0]["modes"]["all"]["ru_rows"] == 10
    watch.main([str(run)])
    out = capsys.readouterr().out
    assert "math ru n" in out and "chat ru n" in out


def test_steady_russian_answers_keep_rule_6_quiet(tmp_path):
    rollouts = {step: rollout_rows(step) + ru_rows(step, english=1) for step in range(1, 11)}
    report = watch.build_report(make_run(tmp_path, steps=10, rollouts=rollouts))
    assert rule(report, "Cyrillic share").status == "ok"


RU_CHAT_PROMPT = "user\nНапиши короткое стихотворение про осенний дождь в большом городе.\nassistant\n<think>\n"
RU_CODE_PROMPT = ("user\nДан массив из n целых чисел. Выведите сумму всех его элементов.\n\n"
                  "Write Python code to solve the problem. Please place the solution code in the following format:\n"
                  "```python\n# Your solution code here\n```\nassistant\n<think>\n")
RU_MATH_RU_PROMPT = ("user\nВ классе 12 учеников. Сколько учеников останется, если 3 уйдут?\n"
                     "Рассуждай пошагово и запиши окончательный ответ в \\boxed{}.\nassistant\n<think>\n")


def kind_rows(prompt: str, step: int, *, english: int = 0, rows: int = 10) -> list[dict]:
    """``rows`` Russian think answers to ``prompt``, the first ``english`` of them thinking and answering in English."""
    ru = "Считаем учеников: было двенадцать, ушли трое, значит осталось девять. </think>\n\n" + RU_ANSWER
    en = "Counting the students: twelve were there and three left, so nine remain. </think>\n\n" + EN_ANSWER
    return [answer(prompt, en if i < english else ru, step) for i in range(rows)]


def test_the_kind_of_a_prompt_comes_from_its_instruction():
    assert watch.prompt_kind(THINK_PROMPT) == watch.prompt_kind(RU_THINK_PROMPT) == "math"
    assert watch.prompt_kind(RU_MATH_RU_PROMPT) == "math"
    assert watch.prompt_kind(RU_CODE_PROMPT) == "code"
    assert watch.prompt_kind(CHAT_PROMPT) == watch.prompt_kind(RU_CHAT_PROMPT) == "chat"


def test_a_russian_question_is_recognised_before_any_instruction():
    # The English code instruction alone has more Latin letters than this question has Cyrillic ones.
    assert watch.russian_prompt(RU_CODE_PROMPT)
    assert watch.russian_prompt(RU_MATH_RU_PROMPT)
    assert watch.russian_prompt(RU_CHAT_PROMPT)
    assert not watch.russian_prompt(CHAT_PROMPT)


def test_chat_answers_switching_to_english_fire_rule_6_while_math_holds(tmp_path):
    # 30 Russian math answers and 3 Russian chat answers per step. When the chat answers turn English,
    # the mean over both stays at 30/33 = 0.91 of its start, so only the chat series can catch it.
    rollouts = {step: rollout_rows(step) + kind_rows(RU_THINK_PROMPT, step, rows=30)
                + kind_rows(RU_CHAT_PROMPT, step, english=0 if step <= 5 else 3, rows=3) for step in range(1, 11)}
    report = watch.build_report(make_run(tmp_path, steps=10, rollouts=rollouts))
    fired = rule(report, "Cyrillic share")
    assert fired.status == "FIRED" and fired.step == 10, fired
    # The detail of a fired rule lists only the series that fired.
    assert "kind:chat" in fired.detail and "kind:math" not in fired.detail


def test_code_answers_switching_to_english_keep_rule_6_quiet(tmp_path):
    rollouts = {step: rollout_rows(step) + kind_rows(RU_THINK_PROMPT, step) + kind_rows(RU_CHAT_PROMPT, step)
                + kind_rows(RU_CODE_PROMPT, step, english=0 if step <= 5 else 10) for step in range(1, 11)}
    report = watch.build_report(make_run(tmp_path, steps=10, rollouts=rollouts))
    assert rule(report, "Cyrillic share").status == "ok"
    assert report.steps[0]["modes"]["kind:code"]["ru_rows"] == 10


def test_kind_groups_stay_out_of_the_validation_answers(tmp_path):
    answers = {0: [answer(THINK_PROMPT, think_answer(500)), answer(NO_THINK_PROMPT, "The answer is \\boxed{4}.")]}
    report = watch.build_report(make_run(tmp_path, steps=1, answers=answers))
    assert {entry["mode"] for entry in report.validation_answers} == {"think", "no_think"}


# --------------------------------------------------------------------------- RL mode (--rl), rules W1-W8

# Real decoded prompts, copied from the step-0 validation dump of a 16K distillation run.
REAL_MGSM_PROMPT = ("user\nВ вазе 4 розы. В вазе георгинов на 7 больше, чем роз. Сколько всего цветов в вазе?\n"
                    "Рассуждай пошагово и запиши окончательный ответ в \\boxed{}.\nassistant\n<think>\n")
REAL_MATH_PROMPT = ("user\nEvaluate $\\log_264$.\nPlease reason step by step, and put your final answer within "
                    "\\boxed{}.\nassistant\n<think>\n")
# The chat probe renders sft_chat_think, which adds no instruction. These Russian prompts would be
# misread if the template stayed, because "user", "assistant" and "think" outweigh their Cyrillic.
RU_CHAT_QUESTIONS = ("Привет!", "Кто такой Унабомбер?", "Как установить TensorFlow?")
EN_CHAT_QUESTIONS = ("Hi!", "How do I install TensorFlow?", "I want you to act as a Linux terminal.")


def chat_prompt(question: str) -> str:
    """A chat probe prompt as verl's dump decodes it."""
    return f"user\n{question}\nassistant\n<think>\n"


def rl_train_row(step: int, *, kl: float = 0.01, grad: float = 0.1, clip: float = 0.1, mean: float = 8000.0,
                 exit_step: float = 2.6, gap: float = 0.3) -> dict:
    """One training step of an RL run, with the RLTT loss's per-loop log-probs (loop 3 is the final one)."""
    loops = {f"actor/rltt/loop_logp_t{t}": -0.5 - gap / (t + 1) for t in range(3)}
    return {"step": step, "data": {
        "actor/kl_loss": kl, "actor/grad_norm": grad, "response_length/mean": mean, "response_length/max": LIMIT,
        "response_length/clip_ratio": clip, "critic/score/mean": 0.5, "actor/rltt/expected_exit_step": exit_step,
        "actor/rltt/loop_logp_t3": -0.5, **loops}}


def rl_val_row(step: int, source: str, *, acc: float = 0.8, unclosed: float = 0.05) -> dict:
    return {"step": step, "data": {f"val-core/{source}/acc/mean@1": acc,
                                   f"val-aux/{source}/think_unclosed/mean@1": unclosed}}


def fires_of(report, rule: str) -> list[tuple[str, int]]:
    return [(fire.level, fire.step) for fire in report.fires if fire.rule == rule]


def test_the_decoded_template_is_cut_off_before_the_language_is_read():
    for question in RU_CHAT_QUESTIONS:
        prompt = chat_prompt(question)
        assert watch.question_text(prompt) == question
        assert watch.russian_prompt(prompt) and watch.russian_prompt(prompt + "\n</think>\n\n")
    # With the template left in, its Latin letters decide: "Привет!" ("Hi!") reads as a quarter Cyrillic.
    assert watch.cyrillic_share(chat_prompt("Привет!"), minimum=10) < 0.5
    for question in EN_CHAT_QUESTIONS:
        assert not watch.russian_prompt(chat_prompt(question))
    assert watch.question_text(REAL_MGSM_PROMPT).startswith("В вазе 4 розы.")
    assert watch.question_text(REAL_MGSM_PROMPT).endswith("цветов в вазе?")
    assert watch.russian_prompt(REAL_MGSM_PROMPT) and not watch.russian_prompt(REAL_MATH_PROMPT)
    assert watch.question_text(REAL_MATH_PROMPT) == "Evaluate $\\log_264$."


def write_val_files(directory: Path) -> Path:
    """Frozen validation files with the names the watcher looks for, holding the questions of the prompts above."""
    write_jsonl(directory / "mgsm_ru.jsonl", [{"instruction": watch.question_text(REAL_MGSM_PROMPT), "response": "18"}])
    write_jsonl(directory / "math500.jsonl", [{"instruction": "Evaluate $\\log_264$.", "response": "6"}])
    write_jsonl(directory / "chat_probe_ru.jsonl", [{"instruction": q} for q in RU_CHAT_QUESTIONS])
    write_jsonl(directory / "chat_probe_en.jsonl", [{"instruction": q} for q in EN_CHAT_QUESTIONS])
    return directory


def test_validation_rows_are_matched_to_their_set_by_the_exact_question(tmp_path):
    files = watch.val_files(str(write_val_files(tmp_path / "data")), [])
    assert sorted(files) == ["chat_en", "chat_ru", "math500", "mgsm_ru"]
    index = watch.load_val_index(files)
    assert watch.val_role(REAL_MGSM_PROMPT, index) == ("mgsm_ru", True)
    assert watch.val_role(REAL_MATH_PROMPT, index) == ("math500", True)
    assert watch.val_role(REAL_MATH_PROMPT.replace("<think>\n", "<think>\n\n</think>\n\n"), index) == ("math500", True)
    for question in RU_CHAT_QUESTIONS:
        assert watch.val_role(chat_prompt(question), index) == ("chat_ru", True)
    for question in EN_CHAT_QUESTIONS:
        assert watch.val_role(chat_prompt(question), index) == ("chat_en", True)
    # A row that is in no file falls back to its template.
    assert watch.val_role(chat_prompt("Расскажи о себе."), index) == ("chat_ru", False)
    # Both copies of the Russian probe (#chat_ru and #chat_ru2) land in one role, two samples per prompt.
    ru_answer = "Думаю, как ответить на приветствие вежливо.</think>\n\nЗдравствуйте! Чем я могу вам помочь сегодня?"
    rows = [answer(chat_prompt(q), ru_answer, 0) for q in RU_CHAT_QUESTIONS] * 2
    rows += [answer(chat_prompt(q), "Thinking about it.</think>\n\nSure.", 0) for q in EN_CHAT_QUESTIONS]
    write_jsonl(tmp_path / "val_generations" / "0.jsonl", rows)
    roles, unmatched = watch.read_val_roles(tmp_path / "val_generations", index)
    assert roles[0]["chat_ru"].rows == 6 and roles[0]["chat_en"].rows == 3 and unmatched == {0: 0}
    summary = roles[0]["chat_ru"].summary()
    assert summary["ru_rows"] == 6 and summary["ru_think"] == 1.0 and summary["ru_answer"] == 1.0


def test_w1_reads_a_two_eval_mean_and_stops_only_on_a_confirmed_large_drop(tmp_path):
    def report_for(accs, source, where):
        val = [rl_val_row(10 * i, source, acc=acc) for i, acc in enumerate(accs)]
        rows = [rl_train_row(step) for step in range(1, 10 * (len(accs) - 1) + 1)]
        return watch.build_rl_report(make_run(where, metrics=rows, val=val, rollouts={}))

    # One low eval on a plateau is noise: its 2-eval mean stays within 0.03 of the best.
    assert fires_of(report_for([0.79, 0.82, 0.84, 0.85, 0.80, 0.85, 0.84], "orbit/math#think", tmp_path / "a"),
                    "W1") == []
    collapse = [0.79, 0.84, 0.85, 0.84, 0.78, 0.70, 0.60]
    assert fires_of(report_for(collapse, "orbit/math#think", tmp_path / "b"), "W1") == [("notify", 50),
                                                                                         ("hard-stop", 60)]
    # LCB-78 has no drop rule, and an untagged MATH-500 think source is watched like the tagged one.
    assert fires_of(report_for(collapse, "orbit/code#think", tmp_path / "c"), "W1") == []
    assert fires_of(report_for(collapse, "orbit/math", tmp_path / "d"), "W1")[-1] == ("hard-stop", 60)


def test_grad_kl_and_loop_gap_rules(tmp_path):
    # The first steps start high, as they do in a real run, and do not count.
    rows = [rl_train_row(step, grad=1.0 if step <= 3 else 0.1) for step in range(1, 21)]
    rows += [rl_train_row(step, grad=0.25, gap=0.5) for step in range(21, 24)]
    report = watch.build_rl_report(make_run(tmp_path / "a", metrics=rows, rollouts={}))
    assert fires_of(report, "W2") == [("notify", 23)] and fires_of(report, "W8") == [("notify", 23)]
    # Two steps above the bar are not enough.
    report = watch.build_rl_report(make_run(tmp_path / "b", metrics=rows[:-1], rollouts={}))
    assert fires_of(report, "W2") == [] and fires_of(report, "W8") == []
    # The KL clock: the 5-step mean passes 0.045 (notify) and then 0.06 (hard stop).
    rows = [rl_train_row(step, kl=0.002 * step) for step in range(1, 41)]
    report = watch.build_rl_report(make_run(tmp_path / "c", metrics=rows, rollouts={}))
    assert fires_of(report, "W3") == [("notify", 25), ("hard-stop", 33)]
    report = watch.build_rl_report(make_run(tmp_path / "d", metrics=rows, rollouts={}), watch.RLThresholds(kl_hard=0))
    assert fires_of(report, "W3") == [("notify", 25)]


def test_the_truncation_rule_ignores_spikes_and_only_notifies(tmp_path):
    # Spikes of single batches, like steps 10 and 13 of a real MATH run, do not move a 5-step mean enough.
    clips = {step: 0.07 for step in range(1, 41)}
    clips.update({10: 0.145, 13: 0.145, 25: 0.2})
    rows = [rl_train_row(step, clip=clip) for step, clip in sorted(clips.items())]
    assert fires_of(watch.build_rl_report(make_run(tmp_path / "a", metrics=rows, rollouts={})), "W4b") == []
    # A sustained rise fires once three 5-step means in a row are high, and only as a notify.
    rows = [rl_train_row(step, clip=0.07 if step <= 20 else 0.25) for step in range(1, 31)]
    report = watch.build_rl_report(make_run(tmp_path / "b", metrics=rows, rollouts={}))
    # Steps 22, 23 and 24 are the first three 5-step means at or above 0.14 (2 x 0.07).
    assert fires_of(report, "W4b") == [("notify", 24)]


def test_think_unclosed_is_judged_against_step_0_and_lcb_has_no_floor(tmp_path):
    def report_for(source, series, where):
        val = [rl_val_row(10 * i, source, unclosed=value) for i, value in enumerate(series)]
        rows = [rl_train_row(step) for step in range(1, 10 * (len(series) - 1) + 1)]
        return watch.build_rl_report(make_run(where, metrics=rows, val=val, rollouts={}))

    # LCB starts with most answers unclosed (0.684 at step 0 of a 16K run), which is no reason to stop.
    assert fires_of(report_for("orbit/code#think", [0.684, 0.724, 0.605, 0.645], tmp_path / "a"), "W4d") == []
    # MATH-500 think: one high validation notifies, the second in a row stops.
    assert fires_of(report_for("orbit/math#think", [0.05, 0.55, 0.30], tmp_path / "b"), "W4d") == [("notify", 10)]
    assert fires_of(report_for("orbit/math#think", [0.05, 0.55, 0.60], tmp_path / "c"), "W4d") == [
        ("notify", 10), ("hard-stop", 20)]
    # Below the floor of 0.5 nothing fires, except on LCB, which has only the relative bar.
    assert fires_of(report_for("orbit/math#think", [0.10, 0.30, 0.30], tmp_path / "d"), "W4d") == []
    assert fires_of(report_for("orbit/code#think", [0.10, 0.30, 0.30], tmp_path / "e"), "W4d")[-1] == (
        "hard-stop", 20)


def test_the_russian_math_rule_reads_the_gap_to_the_paired_english_problems(tmp_path):
    def report_for(pairs, where):
        val = [row for i, (ru, en) in enumerate(pairs)
               for row in (rl_val_row(10 * i, "orbit/math#ru", acc=ru), rl_val_row(10 * i, "orbit/math#mgsm_en", acc=en))]
        rows = [rl_train_row(step) for step in range(1, 10 * (len(pairs) - 1) + 1)]
        return watch.build_rl_report(make_run(where, metrics=rows, val=val, rollouts={}))

    # Both languages move together: a raw MGSM-ru dip that GSM8K-en shares is not a Russian problem.
    assert fires_of(report_for([(0.64, 0.80), (0.52, 0.68), (0.70, 0.86)], tmp_path / "a"), "W6d") == []
    assert fires_of(report_for([(0.64, 0.80), (0.65, 0.88), (0.64, 0.90)], tmp_path / "b"), "W6d") == [
        ("notify", 10), ("confirmed", 20)]


RU_THINKING = "Нужно ответить по-русски и подробно объяснить, как это сделать. "
EN_THINKING = "I should answer this in English and explain how to do it in detail. "
RU_REPLY = "Вот подробный ответ на ваш вопрос, с примерами и пояснениями."
EN_REPLY = "Here is a detailed answer to your question, with examples and explanations."


def probe_rows(step: int, english: int, *, en_chars: int = 0) -> list[dict]:
    """Both copies of the Russian probe (10 prompts each), ``english`` rows per copy in English, plus the English probe."""
    rows = []
    for _ in range(2):
        for i in range(10):
            thinking, reply = (EN_THINKING, EN_REPLY) if i < english else (RU_THINKING, RU_REPLY)
            rows.append(answer(chat_prompt(f"Вопрос номер {i}: как дела?"), thinking + "</think>\n\n" + reply, step,
                               looped=0.0, think_unclosed=0.0))
    for i in range(10):
        rows.append(answer(chat_prompt(f"Question {i}: how are you?"),
                           EN_THINKING + "</think>\n\n" + (EN_REPLY + "x" * en_chars), step, looped=0.0, think_unclosed=0.0))
    rows.append(answer(REAL_MGSM_PROMPT, RU_THINKING + "</think>\n\n" + "Ответ: \\boxed{11}.", step,
                       looped=0.0, think_unclosed=0.0))
    return rows


def test_russian_chat_drift_confirmed_by_two_rules_is_a_hard_stop(tmp_path):
    answers = {0: probe_rows(0, 0), 10: probe_rows(10, 3), 20: probe_rows(20, 3)}
    rows = [rl_train_row(step) for step in range(1, 21)]
    report = watch.build_rl_report(make_run(tmp_path, metrics=rows, answers=answers, rollouts={}))
    assert fires_of(report, "W6b") == [("notify", 10), ("confirmed", 20)]
    assert fires_of(report, "W6c") == [("notify", 10), ("confirmed", 20)]
    assert fires_of(report, "W6") == [("hard-stop", 20)] and fires_of(report, "W6a") == []
    # A probe that keeps its share of English rows from step 0 on fires nothing.
    answers = {0: probe_rows(0, 1), 10: probe_rows(10, 1), 20: probe_rows(20, 1)}
    report = watch.build_rl_report(make_run(tmp_path / "b", metrics=rows, answers=answers, rollouts={}))
    assert [fire for fire in report.fires if fire.rule.startswith("W6")] == []


def test_a_chat_probe_whose_answers_shrink_notifies(tmp_path):
    answers = {0: probe_rows(0, 0, en_chars=4000), 10: probe_rows(10, 0, en_chars=500)}
    rows = [rl_train_row(step) for step in range(1, 11)]
    report = watch.build_rl_report(make_run(tmp_path, metrics=rows, answers=answers, rollouts={}))
    assert ("notify", 10) in fires_of(report, "W7")
    assert all(fire.series == "val chat_en mean response chars" for fire in report.fires if fire.rule == "W7")


def test_cyrillic_words_in_english_training_answers_fire_w6e(tmp_path):
    def step_rows(step: int, spliced: int) -> list[dict]:
        clean = "We add the numbers and get four, so the answer is 4.</think>\n\n\\boxed{4}"
        mixed = "We add the numbers, то есть get four, so the answer is 4.</think>\n\n\\boxed{4}"
        return [answer(THINK_PROMPT, mixed if i < spliced else clean, step) for i in range(50)]

    rollouts = {step: step_rows(step, 1 if step <= 5 else 4) for step in range(1, 16)}
    rows = [rl_train_row(step) for step in range(1, 16)]
    report = watch.build_rl_report(make_run(tmp_path, metrics=rows, rollouts=rollouts))
    assert fires_of(report, "W6e") == [("notify", 10), ("confirmed", 15)]
    assert "frag 0.080" in [text for step, order, text in report.lines if step == 15 and order == 0][0]


def test_rl_status_lines_and_fires_are_printed_once_while_following(tmp_path, capsys, monkeypatch):
    rows = [rl_train_row(step, kl=0.002 * step) for step in range(1, 41)]
    val = [rl_val_row(0, "orbit/math#think"), rl_val_row(10, "orbit/math#think", acc=0.82)]
    # At step 30 the KL rule W3 has notified (at step 25) but has not reached its hard stop (step 33).
    run = make_run(tmp_path, metrics=rows[:30], val=val, rollouts={})
    sleeps = []

    def fake_sleep(seconds):
        # The run trains on to step 40 between the two checks.
        sleeps.append(seconds)
        write_jsonl(run / "metrics.jsonl", [*val, *rows])

    monkeypatch.setattr(watch.time, "sleep", fake_sleep)
    # The hard stop ends the follow loop with its exit code, which is how a wrapper learns to stop the run.
    assert watch.main([str(run), "--rl", "--follow", "60", "--last", "0"]) == watch.EXIT_HARD
    assert sleeps == [60]
    out = capsys.readouterr().out
    assert out.count("FIRE W3 notify at step 25") == 1 and out.count("FIRE W3 hard-stop at step 33") == 1
    assert out.count("step   30 score") == 1 and out.count("step   40 score") == 1
    assert out.count("val    10 acc/uncl math#think 0.820/0.050") == 1
    # One check without --follow ends with the first fires and the exit code of a hard stop.
    assert watch.main([str(run), "--rl"]) == watch.EXIT_HARD
    assert "first notify: step 25 (W3); first hard stop: step 33 (W3)" in capsys.readouterr().out


def test_following_a_run_without_a_hard_stop_ends_only_with_ctrl_c(tmp_path, capsys, monkeypatch):
    rows = [rl_train_row(step, kl=0.002 * step) for step in range(1, 31)]
    run = make_run(tmp_path, metrics=rows, rollouts={})
    sleeps = []

    def fake_sleep(seconds):
        # A notify alone keeps the loop going, so the third sleep ends it, as Ctrl-C would.
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(watch.time, "sleep", fake_sleep)
    assert watch.main([str(run), "--rl", "--follow", "60"]) == watch.EXIT_OK
    assert capsys.readouterr().out.count("FIRE W3 notify at step 25") == 1
