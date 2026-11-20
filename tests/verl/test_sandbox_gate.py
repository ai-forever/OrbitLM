"""CPU tests of scripts/verl/sandbox_gate.py, the check of an RL smoke before the full runs on the same machines.

Every run dir here is synthetic and has the shape of a finished smoke: metrics files of two launches
(the second resumed at step 2), verl's rollout and validation dumps, the dataset manifest that
config.yaml points at, and the logs of the two launches as scripts/verl/run_rl.sh leaves them. Each failure case breaks one thing that a
failing sandbox or a lost prompt group would break in a real run. The script uses only the standard
library, so these tests run anywhere with PYTHONPATH=<repo>/src.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verl" / "sandbox_gate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("sandbox_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_script()

# The prompts as verl's dumps decode them. The instructions are those of sft_think and sft_think_code.
MATH_PROMPT = ("user\nWhat is 2+2?\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
               "assistant\n<think>\n")
CODE_PROMPT = ("user\nRead a and b and print their sum.\n\nWrite Python code to solve the problem. Present the "
               "code in ```python\nYour code\n```\nat the end.\nassistant\n<think>\n")
# The validation sources of the synthetic dataset and their rows, in the manifest's layout.
COUNTS = {"math": {"think": {"orbit/math#think": 2}}, "code": {"think": {"orbit/code#think": 2}},
          "chat": {"think": {"orbit/none#chat_ru": 1, "orbit/none#chat_ru2": 1}}}
VAL_ROWS = 6
BATCH, N = 2, 2
LAUNCH_LINE = "[orbit-launch local 0/1 12:00:00] mode=local gpus/node=8 run_dir=/runs/rl_smoke\n"


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _val_metrics(step: int) -> dict:
    sources = [source for modes in COUNTS.values() for by_source in modes.values() for source in by_source]
    return {"step": step, "data": {f"val-core/{source}/acc/mean@1": 0.5 for source in sources}}


def _rollouts(step: int) -> list[dict]:
    """One math and one code prompt, two answers each. One code answer passes its tests."""
    rows = []
    for prompt, scores in ((MATH_PROMPT, (1.0, 0.0)), (CODE_PROMPT, (1.0, 0.0))):
        rows += [{"input": prompt, "output": "...", "gts": "x", "score": score, "step": step, "uid": prompt[:9]}
                 for score in scores]
    return rows


def _val_rows(step: int) -> list[dict]:
    rows = [{"input": MATH_PROMPT, "output": "4", "score": 1.0, "step": step, "code_sandbox_error": 0.0}] * 2
    rows += [{"input": CODE_PROMPT, "output": "```python\n```", "score": 0.0, "step": step,
              "code_sandbox_error": 0.0}] * 2
    rows += [{"input": "user\nПривет!\nassistant\n<think>\n", "output": "Привет", "score": 0.0, "step": step,
              "code_sandbox_error": 0.0}] * 2
    return rows


@pytest.fixture
def smoke(tmp_path):
    """A finished smoke: the first launch ran steps 1-2 and validated at 0 and 2, the second resumed to 3.

    config.yaml quotes the key n, as omegaconf writes it, because YAML 1.1 would read a bare n as false.
    """
    data = tmp_path / "data" / "rl_smoke__qwen3"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text(json.dumps({"counts": {"val": COUNTS}, "rows": {"val": VAL_ROWS}}))
    run = tmp_path / "runs" / "rl_smoke"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        f"data:\n  train_batch_size: {BATCH}\n  max_response_length: 16384\nactor_rollout_ref:\n  rollout:\n"
        f"    'n': {N}\n    val_kwargs:\n      temperature: 0.6\n      'n': 1\norbit:\n  data_dir: {data}\n")
    _jsonl(run / "metrics.20260101-120000.jsonl",
           [_val_metrics(0), {"step": 1, "data": {"actor/kl_loss": 0.0}},
            {"step": 2, "data": {"actor/kl_loss": 0.0}}, _val_metrics(2)])
    _jsonl(run / "metrics.jsonl", [{"step": 3, "data": {"actor/kl_loss": 0.0}}, _val_metrics(3)])
    for step in (1, 2, 3):
        _jsonl(run / "rollouts" / f"{step}.jsonl", _rollouts(step))
    for step in (0, 2, 3):
        _jsonl(run / "val_generations" / f"{step}.jsonl", _val_rows(step))
    # The first launch's log was renamed when the second launch started.
    logs = [run / "launch.20260101-120000.log", run / "launch.log"]
    for log in logs:
        log.write_text("arm smoke, run rl_smoke, on this node, mode run\n" + LAUNCH_LINE + "step:1 ...\n")
    return run, logs


def _run(run: Path, logs, capsys) -> tuple[int, str]:
    code = gate.main([str(run), *map(str, logs)])
    return code, capsys.readouterr().out


def test_a_clean_smoke_passes(smoke, capsys):
    run, logs = smoke
    code, out = _run(run, logs, capsys)
    assert code == gate.EXIT_PASS, out
    assert [line.split(":")[0] for line in out.splitlines()[:3]] == ["PASS launch logs", "PASS validation",
                                                                     "PASS training"]
    assert "sandbox gate: PASS" in out


def test_without_log_arguments_the_gate_reads_the_logs_in_the_run_dir(smoke, capsys):
    run, logs = smoke
    # The log of another node is not one of the launches that trained.
    (run / "launch.node1.log").write_text("ERROR Error in _run_prompt for uid=3f2a\n")
    assert gate.run_logs(run) == logs
    code, out = _run(run, [], capsys)
    assert code == gate.EXIT_PASS and "PASS launch logs: 2 logs" in out, out
    # With only the last launch's log the gate cannot see the first launch.
    logs[0].unlink()
    code, out = _run(run, [], capsys)
    assert code == gate.EXIT_FAIL and "1 launch log(s) found" in out, out


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def _drop_val_row(run: Path, step: int, kind_prompt: str) -> None:
    path = run / "val_generations" / f"{step}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.remove(next(row for row in rows if row["input"] == kind_prompt))
    _jsonl(path, rows)


def _drop_metric(run: Path, key: str) -> None:
    path = run / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["data"].pop(key, None)
    _jsonl(path, rows)


def _rewrite_rollouts(run: Path, step: int, change) -> None:
    path = run / "rollouts" / f"{step}.jsonl"
    _jsonl(path, change([json.loads(line) for line in path.read_text().splitlines()]))


@pytest.mark.parametrize("breakage,rule,message", [
    # verl's line for a prompt group whose reward raised, as agent_loop_tq logs it.
    (lambda run, logs: _append(logs[1], "ERROR Error in _run_prompt for uid=3f2a\n"), "launch logs", "_run_prompt"),
    # The launcher's own refusal when the machine fails the self-check.
    (lambda run, logs: _append(logs[0], "PROBLEM (sandbox): RuntimeError: The code sandbox failed its self-check "
                                        "in this process: proc_blocked\n"), "launch logs", "self-check"),
    (lambda run, logs: logs.pop(), "launch logs", "both smoke launches"),
    (lambda run, logs: logs[1].write_text(""), "launch logs", "no line of the launcher"),
    # LCB lost at the validation after the resume: its metrics are missing, as when every group failed.
    (lambda run, logs: _drop_metric(run, "val-core/orbit/code#think/acc/mean@1"), "validation",
     "step 3: no validation metrics for orbit/code#think"),
    # One LCB group lost at step 2: the metrics exist, but the dump is short.
    (lambda run, logs: _drop_val_row(run, 2, CODE_PROMPT), "validation", "step 2: the validation dump has 5 rows"),
    (lambda run, logs: (run / "val_generations" / "3.jsonl").unlink(), "validation", "does not exist"),
    (lambda run, logs: _jsonl(run / "val_generations" / "0.jsonl",
                              [dict(row, code_sandbox_error=float(row["input"] == CODE_PROMPT))
                               for row in _val_rows(0)]), "validation", "2 validation answer(s) ended in a sandbox error"),
    # A code group lost in training, which leaves the math group alone in the batch.
    (lambda run, logs: _rewrite_rollouts(run, 2, lambda rows: [r for r in rows if r["input"] == MATH_PROMPT]),
     "training", "step 2: 2 rollouts, not 4"),
    (lambda run, logs: _rewrite_rollouts(run, 1, lambda rows: [dict(r, score=None) if r["input"] == CODE_PROMPT
                                                               else r for r in rows]),
     "training", "2 of 2 code rollouts have no score"),
    # Programs that never pass, as when the sandbox runs nothing: every code score is 0 in every step.
    (lambda run, logs: [_rewrite_rollouts(run, step, lambda rows: [dict(r, score=0.0) if r["input"] == CODE_PROMPT
                                                                   else r for r in rows]) for step in (1, 2, 3)],
     "training", "no code rollout of steps 1-3 passed"),
    (lambda run, logs: _rewrite_rollouts(run, 3, lambda rows: [dict(r, input=MATH_PROMPT) for r in rows]),
     "training", "step 3: no code rollouts"),
])
def test_the_gate_fails(smoke, capsys, breakage, rule, message):
    run, logs = smoke
    breakage(run, logs)
    code, out = _run(run, logs, capsys)
    assert code == gate.EXIT_FAIL, out
    failed = [line for line in out.splitlines() if line.startswith("FAIL")]
    assert failed and all(line.startswith(f"FAIL {rule}:") for line in failed), out
    assert message in out and "sandbox gate: FAIL" in out


def test_the_gate_needs_the_run_dir_and_a_code_validation_source(smoke, capsys, tmp_path):
    run, logs = smoke
    assert gate.main([str(tmp_path / "runs" / "missing"), *map(str, logs)]) == gate.EXIT_USAGE
    manifest = tmp_path / "data" / "rl_smoke__qwen3" / "manifest.json"
    manifest.write_text(json.dumps({"counts": {"val": {"math": COUNTS["math"]}}}))
    code, out = _run(run, logs, capsys)
    assert code == gate.EXIT_FAIL and "no code validation source" in out
