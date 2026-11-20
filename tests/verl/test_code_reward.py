"""CPU tests for the code reward of orbit_verl.data.reward: routing, the ground-truth format, the
sandbox pool, the key set, and the truncation and thinking rules.

Most tests replace the sandbox with a stand-in, so they run wherever the reward module imports. The
tests marked ``needs_sandbox`` start real sandboxes, which need Linux, so they are skipped
elsewhere. Run all of them on the machine that will run the rewards, with the verl python and
PYTHONPATH=<repo>/src.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile

import pytest

from orbit_verl import common

# The caches go under the runs root of the person running the tests, or into a temporary dir.
common.apply_cache_env(os.environ.get("ORBIT_VERL_RUNS") or tempfile.mkdtemp(prefix="orbit_verl_tests_"))

from orbit_verl.code_sandbox import REWARD_INFO_KEYS, Limits  # noqa: E402
from orbit_verl.data import reward  # noqa: E402
from orbit_verl.data.prompts import DATA_SOURCES, PROMPT_MODES, repo_root  # noqa: E402
from orbit_verl.data.response_info import ANSWER_KEY, FINISH_KEY, THINK_KEY  # noqa: E402

needs_sandbox = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the sandbox needs Linux (Landlock, seccomp, user namespaces)"
)

KEYS = {"score", "acc", "answer_correct", "truncated", "think_unclosed", "looped", "think_forced", *REWARD_INFO_KEYS}

ADD_PROGRAM = "a, b = map(int, input().split())\nprint(a + b)\n"
SUB_PROGRAM = "a, b = map(int, input().split())\nprint(a - b)\n"
STDIN_TRUTH = json.dumps(
    {"kind": "stdin", "tests": [{"input": "1 2\n", "output": "3\n"}, {"input": "10 -4\n", "output": "6\n"}]}
)
ADD_FUNCTION = "def add(a, b):\n    return a + b\n"
ASSERT_TRUTH = json.dumps(
    {"kind": "assert", "tests": "assert add(1, 2) == 3\nassert add(-1, 1) == 0\n", "entry_point": None,
     "context": None, "judge_imports": True, "cases": 2}
)


def score(*args, **kwargs):
    return asyncio.run(reward.compute_score(*args, **kwargs))


def loop_info(finish="eos", think="none", answer=None):
    """The extra_info entries the agent loop adds to every response."""
    return {FINISH_KEY: finish, THINK_KEY: think, ANSWER_KEY: answer}


def code_score(program, truth=STDIN_TRUTH, finish="eos", **kwargs):
    """Score a think-mode response that closes its thinking block and then gives ``program``.

    The thinking block holds a wrong sketch, so a test passes only if the reward ran the code after
    </think>. The agent loop's answer field holds the text after </think>, as the loop decodes it.
    """
    after = f"\nHere is the solution.\n```python\n{program}```\n"
    full = f"Maybe subtraction?\n```python\n{SUB_PROGRAM}```\n</think>{after}"
    return score("orbit/code", full, truth, extra_info=loop_info(finish, "closed", after), **kwargs)


class FakeSandbox:
    """Stands in for the sandbox: records every call and passes exactly ``ADD_PROGRAM``."""

    def __init__(self):
        self.calls: list[dict] = []
        self.checks = 0
        self.report = {"ran": True, "proc_blocked": True, "socket_blocked": True, "env_empty": True}
        self.error: Exception | None = None

    def self_check(self):
        self.checks += 1
        return dict(self.report)

    def execute(self, program, tests, kind="stdin", limits=None, **kwargs):
        self.calls.append({"program": program, "tests": tests, "kind": kind, "limits": limits, **kwargs})
        if self.error is not None:
            raise self.error
        passed = program == ADD_PROGRAM
        return {"passed": passed, "reason": "passed" if passed else "wrong_answer", "policy_violations": [],
                "runtime": 0.01, "stdout_tail": "", "tests_passed": int(passed), "tests_total": 1,
                "cpu_time": 0.25, "detail": "", "shadowed_names": []}


def _drop_code_pool():
    pool, reward._code_pool = reward._code_pool, None
    if pool is not None:
        pool.shutdown(wait=True)


@pytest.fixture(autouse=True)
def fresh_pools():
    """Each test starts and ends without pools, so the self-check and the pool size are decided anew."""
    _drop_code_pool()
    yield
    _drop_code_pool()
    if reward._pool is not None:
        reward._discard_pool(reward._pool)


@pytest.fixture
def fake(monkeypatch):
    sandbox = FakeSandbox()
    monkeypatch.setattr(reward, "self_check", sandbox.self_check)
    monkeypatch.setattr(reward, "execute", sandbox.execute)
    # available_cpus reads Linux-only interfaces. A fixed count makes the default pool size the same everywhere.
    monkeypatch.setattr(reward, "available_cpus", lambda: 16)
    return sandbox


# --------------------------------------------------------------------------- routing and the format


def test_code_rows_route_to_the_code_reward(fake):
    assert DATA_SOURCES["orbit/code"] == "code"
    assert PROMPT_MODES["sft_think_code"].default_data_source == "orbit/code"
    result = code_score(ADD_PROGRAM)
    assert result["score"] == result["acc"] == result["answer_correct"] == result["code_passed"] == 1.0
    assert result["code_cpu_s"] == 0.25
    # One run, on the program after </think>, with the stdin defaults that LiveCodeBench data needs.
    [call] = fake.calls
    assert call["program"] == ADD_PROGRAM and call["kind"] == "stdin"
    assert call["tests"] == json.loads(STDIN_TRUTH)["tests"]
    assert call["limits"] == Limits()
    assert call["judge_imports"] is True and call["float_tolerance"] == 1e-6
    assert call["entry_point"] is None and call["test_context"] is None


def test_wrong_program_scores_zero(fake):
    result = code_score(SUB_PROGRAM)
    assert result["score"] == result["answer_correct"] == result["code_passed"] == 0.0
    assert result["code_wrong_answer"] == 1.0


def test_stdin_settings_reach_execute(fake):
    truth = json.loads(STDIN_TRUTH)
    truth.update(judge_imports=False, float_tolerance=None, time_s=2.5, memory_mb=512)
    code_score(ADD_PROGRAM, json.dumps(truth))
    call = fake.calls[-1]
    assert call["judge_imports"] is False
    # An explicit null compares numbers exactly; only a missing key gets the default tolerance.
    assert call["float_tolerance"] is None
    assert call["limits"] == Limits(time_s=2.5, memory_mb=512)


def test_assert_kind_reaches_execute(fake):
    code_score(ADD_FUNCTION, ASSERT_TRUTH)
    call = fake.calls[-1]
    assert call["kind"] == "assert" and call["program"] == ADD_FUNCTION
    assert call["tests"] == json.loads(ASSERT_TRUTH)["tests"]
    # The assert cases share one run, so the run gets the per-case budget for each of them.
    assert call["limits"] == Limits().for_cases(2)
    assert call["judge_imports"] is True and call["float_tolerance"] is None
    humaneval = {"kind": "assert", "tests": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                 "entry_point": "add", "context": "def helper():\n    return 1\n"}
    code_score(ADD_FUNCTION, json.dumps(humaneval))
    call = fake.calls[-1]
    assert call["entry_point"] == "add" and call["test_context"] == humaneval["context"]
    assert call["limits"] == Limits()


GOOD_CASES = [{"input": "1\n", "output": "1\n"}]
MALFORMED = [
    "not json",
    "[]",
    json.dumps({"tests": GOOD_CASES}),
    json.dumps({"kind": "docker", "tests": GOOD_CASES}),
    json.dumps({"kind": ["stdin"], "tests": GOOD_CASES}),
    json.dumps({"kind": "stdin", "tests": []}),
    json.dumps({"kind": "stdin", "tests": "assert True"}),
    json.dumps({"kind": "stdin", "tests": [{"input": 1, "output": "1"}]}),
    json.dumps({"kind": "stdin", "tests": [{"input": "1"}]}),
    json.dumps({"kind": "stdin", "tests": GOOD_CASES, "judge_imports": "yes"}),
    json.dumps({"kind": "stdin", "tests": GOOD_CASES, "float_tolerance": -1}),
    json.dumps({"kind": "stdin", "tests": GOOD_CASES, "time_s": 0}),
    json.dumps({"kind": "stdin", "tests": GOOD_CASES, "memory_mb": 1.5}),
    # A typo of a setting is refused rather than silently replaced by the default.
    json.dumps({"kind": "stdin", "tests": GOOD_CASES, "float_tol": 1e-6}),
    json.dumps({"kind": "stdin", "tests": GOOD_CASES, "entry_point": "f"}),
    json.dumps({"kind": "assert", "tests": ""}),
    json.dumps({"kind": "assert", "tests": ["assert f() == 1"]}),
    json.dumps({"kind": "assert", "tests": "assert f() == 1", "cases": 0}),
    json.dumps({"kind": "assert", "tests": "assert f() == 1", "entry_point": "not a name"}),
    json.dumps({"kind": "assert", "tests": "assert f() == 1", "float_tolerance": 1e-6}),
]


@pytest.mark.parametrize("truth", MALFORMED)
def test_malformed_ground_truth_raises(fake, truth):
    with pytest.raises(ValueError, match="Code ground truth"):
        code_score(ADD_PROGRAM, truth)
    # The row fails even when the response would score 0 without running anything.
    with pytest.raises(ValueError, match="Code ground truth"):
        score("orbit/code", "<think>no end", truth, extra_info=loop_info(think="unclosed"))
    assert fake.calls == [] and reward._code_pool is None


# --------------------------------------------------------------------------- the rules and the key set


def test_unclosed_thinking_scores_zero_and_runs_nothing(fake):
    text = f"```python\n{ADD_PROGRAM}```"
    result = score("orbit/code", text, STDIN_TRUTH, extra_info=loop_info(think="unclosed"))
    assert result["score"] == result["answer_correct"] == 0.0 and result["think_unclosed"] == 1.0
    assert all(result[key] == 0.0 for key in REWARD_INFO_KEYS)
    # Nothing ran, so the sandbox was neither checked nor given a pool.
    assert fake.calls == [] and fake.checks == 0 and reward._code_pool is None


def test_truncated_code_response_gets_truncated_score(fake):
    result = code_score(ADD_PROGRAM, finish="length")
    # The program is right, but the response ran into the length limit.
    assert result["score"] == result["acc"] == 0.0
    assert result["answer_correct"] == result["code_passed"] == 1.0 and result["truncated"] == 1.0
    assert code_score(ADD_PROGRAM, finish="length", truncated_score=-0.5)["score"] == -0.5
    # None turns the rule off: the response is scored on its program.
    assert code_score(ADD_PROGRAM, finish="length", truncated_score=None)["score"] == 1.0


def test_whole_text_is_checked_without_the_agent_loop_fields(fake):
    # The last python block of the whole text is the one after the thinking block.
    text = f"<think>```python\n{SUB_PROGRAM}```</think>\n```python\n{ADD_PROGRAM}```"
    assert score("orbit/code", text, STDIN_TRUTH)["score"] == 1.0
    assert fake.calls[-1]["program"] == ADD_PROGRAM


def test_key_set_is_the_same_for_every_data_source(fake):
    cases = [
        ("orbit/math", "\\boxed{3}", "3"),
        ("orbit/code", f"```python\n{ADD_PROGRAM}```", STDIN_TRUTH),
        ("orbit/code", "no code at all", STDIN_TRUTH),
    ]
    infos = [{}, loop_info(), loop_info("length"), loop_info(think="unclosed"), loop_info(think="closed", answer="")]
    for case in cases:
        for info in infos:
            result = score(*case, extra_info=dict(info))
            assert set(result) == KEYS
            assert result["score"] == result["acc"]
            assert all(isinstance(value, float) for value in result.values())
            if not case[0].endswith("code"):
                assert all(result[key] == 0.0 for key in REWARD_INFO_KEYS)


def test_exception_in_execute_scores_zero(fake, caplog):
    fake.error = OSError("No space left on device")
    with caplog.at_level(logging.WARNING, logger=reward.__name__):
        result = code_score(ADD_PROGRAM)
    assert result["score"] == result["answer_correct"] == 0.0
    assert result["code_sandbox_error"] == 1.0 and result["code_passed"] == 0.0
    assert "No space left on device" in caplog.text
    # The pool survives, and the next response is scored normally.
    fake.error = None
    assert code_score(ADD_PROGRAM)["score"] == 1.0


# --------------------------------------------------------------------------- the sandbox pool


def test_failed_self_check_raises_and_creates_no_pool(fake):
    fake.report.update(proc_blocked=False, env_empty=False)
    for _ in range(2):
        # Every code reward runs the check again while it fails, so no code row is ever scored.
        with pytest.raises(RuntimeError, match="env_empty, proc_blocked"):
            code_score(ADD_PROGRAM)
    assert fake.checks == 2 and fake.calls == [] and reward._code_pool is None


def test_self_check_runs_once_per_process(fake):
    for _ in range(3):
        assert code_score(ADD_PROGRAM)["score"] == 1.0
    assert fake.checks == 1 and len(fake.calls) == 3


def test_code_threads_sets_the_pool_size(fake, monkeypatch):
    code_score(ADD_PROGRAM, code_threads=3)
    assert reward._code_pool._max_workers == 3
    # The size is decided when the pool is created; later calls reuse that pool.
    code_score(ADD_PROGRAM, code_threads=5)
    assert reward._code_pool._max_workers == 3
    # Without code_threads, each of the 8 reward workers takes an eighth of the CPUs, but at least 2 threads.
    for cpus, size in ((40, 5), (12, 2)):
        _drop_code_pool()
        monkeypatch.setattr(reward, "available_cpus", lambda cpus=cpus: cpus)
        code_score(ADD_PROGRAM)
        assert reward._code_pool._max_workers == size


@pytest.mark.parametrize("threads", [0, -1, "4", True, 2.0])
def test_bad_code_threads_raise(fake, threads):
    with pytest.raises(ValueError, match="code_threads"):
        code_score(ADD_PROGRAM, code_threads=threads)
    assert reward._code_pool is None


def test_code_threads_reach_compute_score_through_verl(fake):
    """verl binds reward_kwargs to compute_score, so code_threads set in the config must reach the pool."""
    from omegaconf import OmegaConf
    from verl.trainer.ppo.reward import get_custom_reward_fn

    for recipe in ("rl_sft_mix.yaml", "distill_rl_sft.yaml"):
        recipe_cfg = OmegaConf.load(repo_root() / "configs" / "verl" / recipe)
        assert recipe_cfg.reward.custom_reward_function.reward_kwargs.code_threads is None, recipe
    config = OmegaConf.load(repo_root() / "configs" / "verl" / "orbit_base.yaml")
    config.reward.custom_reward_function.reward_kwargs.code_threads = 3
    fn = get_custom_reward_fn(config)
    after = f"```python\n{ADD_PROGRAM}```"
    result = asyncio.run(fn(data_source="orbit/code", solution_str=after, ground_truth=STDIN_TRUTH,
                            extra_info=loop_info(think="closed", answer=after)))
    assert result["score"] == 1.0 and reward._code_pool._max_workers == 3


# --------------------------------------------------------------------------- the launcher's sandbox check


def _parquet(path, sources):
    """A dataset file with only the column that the launcher's check reads, plus one other."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"data_source": sources, "prompt": ["x"] * len(sources)}), path)
    return str(path)


def _data_cfg(train, val):
    from omegaconf import OmegaConf

    return OmegaConf.create({"data": {"train_files": train, "val_files": val}})


def test_the_launcher_checks_the_sandbox_only_for_a_dataset_with_code_rows(fake, tmp_path):
    from orbit_verl.launch import prepare

    math = _parquet(tmp_path / "train.parquet", ["orbit/math", "orbit/math#ru_train"])
    chat = _parquet(tmp_path / "val_chat.parquet", ["orbit/none#chat_ru", "orbit/math#think"])
    lcb = _parquet(tmp_path / "val_lcb.parquet", ["orbit/math#think", "orbit/code#think"])
    assert prepare.check_code_sandbox(_data_cfg(math, chat)) == [] and fake.checks == 0
    # Validation on LCB alone needs the sandbox, as in the math arm of the RL recipe.
    assert prepare.check_code_sandbox(_data_cfg(math, [chat, lcb])) == [] and fake.checks == 1
    # A missing file is left to the dataset check, and a file that is not parquet to verl's own loader.
    broken = tmp_path / "broken.parquet"
    broken.write_text("not parquet")
    for train in (str(tmp_path / "gone.parquet"), str(broken)):
        assert prepare.check_code_sandbox(_data_cfg(train, chat)) == [] and fake.checks == 1


def test_a_failed_sandbox_stops_the_launch_before_training(fake, tmp_path):
    from orbit_verl.launch import prepare

    fake.report.update(proc_blocked=False)
    train = _parquet(tmp_path / "train.parquet", ["orbit/math", "orbit/code"])
    problems = prepare.check_code_sandbox(_data_cfg(train, train))
    assert [p.kind for p in problems] == ["sandbox"]
    assert "failed its self-check" in problems[0].message and "proc_blocked" in problems[0].message
    # The launcher's check creates no pool, so each reward worker still runs its own check.
    assert fake.checks == 1 and reward._code_pool is None
    # A check that cannot run at all stops the launch too.
    fake.report = None
    problems = prepare.check_code_sandbox(_data_cfg(train, train))
    assert [p.kind for p in problems] == ["sandbox"] and problems[0].message.startswith("TypeError")


@pytest.mark.parametrize("prepare_only", [True, False])
def test_only_a_launch_that_trains_checks_the_sandbox(monkeypatch, prepare_only):
    """A preparation may run on another machine than the training, so it skips the sandbox check."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from orbit_verl.distill import preflight
    from orbit_verl.launch import cli, prepare

    for name in ("check_fits", "check_prepared", "check_settings", "check_resume"):
        monkeypatch.setattr(prepare, name, lambda cfg: [])
    monkeypatch.setattr(preflight, "check_think_budget", lambda cfg: [])
    monkeypatch.setattr(prepare, "check_code_sandbox", lambda cfg: [prepare.Problem("sandbox", "failed")])
    args = SimpleNamespace(prepare_only=prepare_only, force_prepare=False)
    plan = SimpleNamespace(gpus=0, gpu_source="test")
    problems = cli.run_checks(OmegaConf.create({"orbit": {"plugin": False}}), args=args, plan=plan, extra=[])
    assert [p.kind for p in problems] == ([] if prepare_only else ["sandbox", "gpus"])


# --------------------------------------------------------------------------- data and config


def test_code_rows_are_labelled_code(tmp_path):
    from orbit_verl.data import prepare_data as pdm

    mode = PROMPT_MODES["sft_think_code"]
    # Without an explicit data source, the prompt mode's default applies.
    assert pdm._resolve_data_source(None, mode) == "orbit/code"
    row = {"number": 0, "question": "q", "answer": STDIN_TRUTH, "messages": [], "n_tokens": 1}
    [record] = pdm._records([row], "train", mode, "orbit/code")
    assert record["ability"] == "code" and record["reward_model"]["ground_truth"] == STDIN_TRUTH
    assert record["agent_name"] == "orbit_sft_think_code"
    assert pdm._records([row], "train", PROMPT_MODES["sft_think"], "orbit/math")[0]["ability"] == "math"


def test_a_dataset_without_val_src_needs_held_out_rows(tmp_path):
    """val_size 0 without a val_src stops prepare_data instead of writing an empty val.parquet."""
    from orbit_verl.data import prepare_data as pdm

    with pytest.raises(ValueError, match="val_size must be at least 1"):
        pdm.convert(str(tmp_path / "train.jsonl"), str(tmp_path / "out"), prompt_mode="sft_think_code",
                    tokenizer_dir=str(tmp_path), max_prompt_length=2048, val_size=0, runs_root=str(tmp_path))


# --------------------------------------------------------------------------- the real sandbox


@needs_sandbox
def test_correct_program_scores_one_in_the_real_sandbox():
    result = code_score(ADD_PROGRAM)
    assert result["score"] == result["answer_correct"] == result["code_passed"] == 1.0
    assert result["code_wrong_answer"] == 0.0 and result["code_cpu_s"] > 0
    assert reward._code_pool is not None


@needs_sandbox
def test_wrong_program_scores_zero_in_the_real_sandbox():
    result = code_score(SUB_PROGRAM)
    assert result["score"] == result["code_passed"] == 0.0 and result["code_wrong_answer"] == 1.0


@needs_sandbox
def test_judge_defaults_in_the_real_sandbox():
    # The judge imports are on by default, so a program may use Counter without importing it.
    counter = "print(max(Counter(input().split()).values()))\n"
    truth = {"kind": "stdin", "tests": [{"input": "a b a\n", "output": "2\n"}]}
    assert code_score(counter, json.dumps(truth))["score"] == 1.0
    assert code_score(counter, json.dumps({**truth, "judge_imports": False}))["code_runtime_error"] == 1.0
    # The default tolerance accepts a fraction printed with fewer digits than the expected answer.
    third = "print(1 / 3)\n"
    truth = {"kind": "stdin", "tests": [{"input": "", "output": "0.333333333333333314829616256247\n"}]}
    assert code_score(third, json.dumps(truth))["score"] == 1.0
    assert code_score(third, json.dumps({**truth, "float_tolerance": None}))["code_wrong_answer"] == 1.0


@needs_sandbox
def test_assert_kind_in_the_real_sandbox():
    assert code_score(ADD_FUNCTION, ASSERT_TRUTH)["score"] == 1.0
    wrong = code_score("def add(a, b):\n    return a - b\n", ASSERT_TRUTH)
    assert wrong["score"] == 0.0 and wrong["code_test_failed"] == 1.0
    humaneval = {"kind": "assert", "tests": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
                 "entry_point": "add"}
    assert code_score(ADD_FUNCTION, json.dumps(humaneval))["score"] == 1.0


@needs_sandbox
def test_concurrent_code_rewards_in_the_real_sandbox():
    async def many():
        programs = [ADD_PROGRAM, SUB_PROGRAM] * 6
        return await asyncio.gather(*(
            reward.compute_score("orbit/code", f"```python\n{p}```", STDIN_TRUTH, code_threads=4) for p in programs
        ))

    results = asyncio.run(many())
    assert [r["score"] for r in results] == [1.0, 0.0] * 6
