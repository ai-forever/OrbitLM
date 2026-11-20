"""CPU tests for orbit_verl.data.reward: routing, the key set, MATH in the process pool, the
truncation and thinking rules, the rule of the thinking budget for forced answers, and the rows
without a gold answer. The code reward has its own tests in test_code_reward.py.

Run with the verl python and PYTHONPATH=<repo>/src. The test that loads verl's reward manager needs
the Qwen3 tokenizer of an SFT export (``ORBIT_TEST_SFT_EXPORT``, see conftest.py) and skips without it.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
import pytest

from orbit_verl import common
from orbit_verl.code_sandbox import REWARD_INFO_KEYS

# The caches go under the runs root of the person running the tests, or into a temporary dir.
common.apply_cache_env(os.environ.get("ORBIT_VERL_RUNS") or tempfile.mkdtemp(prefix="orbit_verl_tests_"))

from orbit_verl.data import reward  # noqa: E402
from orbit_verl.data.response_info import ANSWER_KEY, FINISH_KEY, FORCED_KEY, THINK_KEY  # noqa: E402
from orbit_verl.data.math_answer import math_reward  # noqa: E402

# Every response also carries the code metrics, as zeros when it ran no program, because a batch may
# mix math and code rows and verl expects one key set for the whole batch.
CODE_ZEROS = dict.fromkeys(REWARD_INFO_KEYS, 0.0)
KEYS = {"score", "acc", "answer_correct", "truncated", "think_unclosed", "looped", "think_forced", *REWARD_INFO_KEYS}

# Pool tasks are pickled by reference, so the spawned workers must be able to import the stand-in
# functions. They live in a small module written to a temporary directory on sys.path, because
# the workers do not import this test file.
HELPER_SOURCE = '''
import os, time

def slow(solution_str, ground_truth, pid_file):
    with open(pid_file, "w") as handle:
        handle.write(str(os.getpid()))
    time.sleep(120)
    return 1.0

def crash(solution_str, ground_truth):
    os._exit(3)
'''


def score(*args, **kwargs):
    return asyncio.run(reward.compute_score(*args, **kwargs))


def loop_info(finish="eos", think="none", answer=None, looped=False, forced=False):
    """The extra_info entries the agent loop adds to every response."""
    return {FINISH_KEY: finish, THINK_KEY: think, ANSWER_KEY: answer, reward.LOOP_KEY: looped, FORCED_KEY: forced}


def pool_info(domain="math", mode="think", finish="eos", think="closed", answer="The answer is 5.", forced=False):
    """The extra_info of a pool row (orbit/none) as compute_score receives it.

    verl merges the fields that the dataset builder wrote (orbit_verl.data.distill_data) with the ones
    that the agent loop adds to the response. The defaults describe a math row with thinking whose
    model closed its thinking block by itself and then answered.
    """
    return {"split": "train", "domain": domain, "mode": mode, **loop_info(finish, think, answer, forced=forced)}


def plain(value):
    """The full result of a response with no truncation, no thinking block, no repetition loop and no forced close."""
    return {"score": value, "acc": value, "answer_correct": value, "truncated": 0.0, "think_unclosed": 0.0,
            "looped": 0.0, "think_forced": 0.0, **CODE_ZEROS}


@pytest.fixture(autouse=True)
def fresh_pool():
    """Each test starts and ends without a pool, so one test's broken pool cannot leak into the next."""
    if reward._pool is not None:
        reward._discard_pool(reward._pool)
    yield
    if reward._pool is not None:
        reward._discard_pool(reward._pool)


@pytest.fixture
def helper(tmp_path, monkeypatch):
    (tmp_path / "orbit_reward_test_helper.py").write_text(HELPER_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import orbit_reward_test_helper

    return orbit_reward_test_helper


def test_unknown_data_source_raises():
    with pytest.raises(ValueError, match="Unknown data_source"):
        score("openai/gsm8k", "#### 72", "72")
    # GSM8K few-shot RL is not part of the release, so its data source is unknown as well.
    with pytest.raises(ValueError, match="Unknown data_source"):
        score("orbit/gsm8k", "#### 72", "72")
    # The tag does not make an unknown source known, and a known name after the tag does not count.
    with pytest.raises(ValueError, match="Unknown data_source"):
        score("orbit/maths#think", "#### 72", "72")
    with pytest.raises(ValueError, match="Unknown data_source"):
        score("eval#orbit/math", "#### 72", "72")


def test_tag_after_hash_keeps_the_reward_of_the_source():
    """Validation sets in two prompt modes differ only in the tag, and both must get the same check."""
    assert score("orbit/math#no_think", "\\boxed{72}", "72") == plain(1.0)
    assert score("orbit/math#think", "so \\boxed{\\frac12}", "\\dfrac{1}{2}") == plain(1.0)
    assert score("orbit/math#think", "so \\boxed{3}", "4")["score"] == 0.0
    # A code row with a tag still has its ground truth parsed as tests.
    with pytest.raises(ValueError, match="Code ground truth"):
        score("orbit/code#think", "```python\nprint(1)\n```", "not json")


def test_rows_without_gold_answer_score_zero_and_run_nothing(monkeypatch):
    async def no_code(*args, **kwargs):
        raise AssertionError("a orbit/none row must not run a program")

    monkeypatch.setattr(reward, "_code_score", no_code)
    for source in ("orbit/none", "orbit/none#chat"):
        # Even a text that would pass any check scores 0, and the ground truth is not read.
        assert score(source, "\\boxed{5} \\boxed{5}", "5") == plain(0.0)
        assert score(source, "anything", "", extra_info=loop_info()) == plain(0.0)
        # The diagnostics are still reported, and the truncation rule gives no reward either.
        result = score(source, "<think>and on", "", extra_info=loop_info("length", "unclosed", looped=True),
                       truncated_score=-1.0)
        assert result == {**plain(0.0), "truncated": 1.0, "think_unclosed": 1.0, "looped": 1.0}
    # No MATH check ran, so the pool was never started.
    assert reward._pool is None


def test_looped_is_reported_for_every_row():
    for case in (("orbit/math", "\\boxed{3}", "3"), ("orbit/none", "x", "")):
        assert score(*case, extra_info=loop_info(looped=True))["looped"] == 1.0
        assert score(*case, extra_info=loop_info(looped=False))["looped"] == 0.0
        # A response without the agent loop's fields, or from an agent loop without the detector.
        assert score(*case)["looped"] == 0.0
        assert score(*case, extra_info={FINISH_KEY: "eos", THINK_KEY: "none", ANSWER_KEY: None})["looped"] == 0.0
    # The repetition flag is a diagnostic: it does not change the reward.
    assert score("orbit/math", "\\boxed{3}", "3", extra_info=loop_info(looped=True))["score"] == 1.0
    assert reward.LOOP_KEY == "orbit_loop"


def test_math_through_the_pool():
    assert score("orbit/math", "so \\boxed{\\frac12}", "\\dfrac{1}{2}") == plain(1.0)
    assert score("orbit/math", "answer \\boxed{7}", "The sum is $\\boxed{7}$.")["score"] == 1.0
    assert score("orbit/math", "answer \\boxed{8}", "7")["score"] == 0.0
    assert score("orbit/math", "no box here", "7")["score"] == 0.0
    assert reward._pool is not None


def test_math_verify_is_active_in_the_pool():
    """0.5 equals 1/2 only through math_verify, which fails silently in a worker thread."""
    solution, gold = "so \\boxed{0.5}", "\\frac{1}{2}"
    in_thread = {}
    worker = threading.Thread(target=lambda: in_thread.setdefault("s", math_reward(solution, gold)))
    worker.start()
    worker.join()
    # This is the failure the pool exists for: verl calls sync rewards from threads.
    assert in_thread["s"] == 0.0
    assert score("orbit/math", solution, gold)["score"] == 1.0


def test_key_set_never_varies():
    cases = [
        ("orbit/math", "\\boxed{3}", "3"),
        ("orbit/math", "", "3"),
        ("orbit/math#no_think", "\\boxed{3}", "3"),
        ("orbit/none", "\\boxed{3}", ""),
        ("orbit/none#chat", "", ""),
    ]
    infos = [{"index": 1}, loop_info(), loop_info("length"), loop_info(think="unclosed"),
             loop_info(think="closed", answer="\\boxed{3}"), loop_info(looped=True),
             loop_info(think="closed", answer="\\boxed{3}", forced=True),
             loop_info("length", "closed", "\\boxed{3}", forced=True), pool_info(), pool_info("chat", forced=True)]
    # The rules of the thinking budget must not change the key set either, whether they are off or on.
    budget_rules = [{}, {"forced_score_scale": 0.5}]
    for case in cases:
        for info in infos:
            for rules in budget_rules:
                result = score(*case, extra_info=dict(info), reward_router_address=None, **rules)
                assert set(result) == KEYS
                assert result["score"] == result["acc"]
                assert all(isinstance(value, float) for value in result.values())


def test_concurrent_math_calls():
    async def many():
        golds = [str(i) for i in range(12)]
        return await asyncio.gather(
            *(reward.compute_score("orbit/math", f"\\boxed{{{g}}}", g) for g in golds),
            *(reward.compute_score("orbit/math", f"\\boxed{{{g}1}}", g) for g in golds),
        )

    results = asyncio.run(many())
    assert [r["score"] for r in results] == [1.0] * 12 + [0.0] * 12


def test_timeout_scores_zero_and_restarts_the_pool(helper, tmp_path, monkeypatch):
    import functools

    pid_file = tmp_path / "slow.pid"
    monkeypatch.setattr(reward, "MATH_TIMEOUT_S", 5.0)
    monkeypatch.setattr(reward, "_math_reward_in_subprocess", functools.partial(helper.slow, pid_file=str(pid_file)))
    started = time.monotonic()
    assert score("orbit/math", "\\boxed{1}", "1") == plain(0.0)
    assert time.monotonic() - started < 30
    assert reward._pool is None
    # The stuck worker must be gone, not left sleeping in the background.
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    monkeypatch.undo()
    assert score("orbit/math", "\\boxed{1}", "1")["score"] == 1.0
    assert reward._pool is not None


def test_broken_pool_scores_zero_and_recovers(helper, monkeypatch):
    monkeypatch.setattr(reward, "_math_reward_in_subprocess", helper.crash)
    assert score("orbit/math", "\\boxed{1}", "1") == plain(0.0)
    assert reward._pool is None
    monkeypatch.undo()
    assert score("orbit/math", "\\boxed{1}", "1")["score"] == 1.0


def test_loaded_the_way_verl_loads_it():
    import inspect

    from omegaconf import OmegaConf
    from verl.trainer.ppo.reward import get_custom_reward_fn

    config = OmegaConf.create(
        {"reward": {"custom_reward_function": {"path": "pkg://orbit_verl.data.reward", "name": "compute_score"}}}
    )
    fn = get_custom_reward_fn(config)
    # verl awaits a coroutine function directly and runs anything else in a thread.
    assert inspect.iscoroutinefunction(fn)
    result = asyncio.run(fn(data_source="orbit/math", solution_str="\\boxed{5}", ground_truth="5", extra_info={}))
    assert result == plain(1.0)


def test_reward_kwargs_reach_compute_score():
    """verl binds the config's reward_kwargs to compute_score, so a truncated_score set in the config must reach it."""
    from omegaconf import OmegaConf
    from verl.trainer.ppo.reward import get_custom_reward_fn

    config = OmegaConf.load(reward_config_path())
    config.reward.custom_reward_function.reward_kwargs.truncated_score = 0.25
    fn = get_custom_reward_fn(config)
    result = asyncio.run(
        fn(data_source="orbit/math", solution_str="\\boxed{5}", ground_truth="5", extra_info=loop_info("length"))
    )
    assert result["score"] == 0.25 and result["answer_correct"] == 1.0


def reward_config_path():
    from orbit_verl.data.prompts import repo_root

    return repo_root() / "configs" / "verl" / "orbit_base.yaml"


def test_base_config_zeroes_truncated_responses():
    from omegaconf import OmegaConf

    kwargs = OmegaConf.load(reward_config_path()).reward.custom_reward_function.reward_kwargs
    assert kwargs.truncated_score == 0.0
    # The repeated n-gram penalty is not part of the release, so the base config does not set it.
    assert "loop_penalty_coef" not in kwargs


def test_truncated_response_gets_truncated_score():
    # The answer is right, but the response ran into the length limit.
    for source, text, gold in (("orbit/math", "\\boxed{5} and more", "5"), ("orbit/math#think", "\\boxed{5} and on", "5")):
        result = score(source, text, gold, extra_info=loop_info("length"))
        assert result == {"score": 0.0, "acc": 0.0, "answer_correct": 1.0, "truncated": 1.0, "think_unclosed": 0.0,
                          "looped": 0.0, "think_forced": 0.0, **CODE_ZEROS}
        # None turns the rule off: the response is scored on its text.
        assert score(source, text, gold, extra_info=loop_info("length"), truncated_score=None)["score"] == 1.0
    # A response that ended on its own or at a stop string is scored on its text.
    for finish in ("eos", "stop"):
        assert score("orbit/math", "\\boxed{5}", "5", extra_info=loop_info(finish)) == plain(1.0)


def test_only_the_text_after_thinking_is_checked():
    # The thinking block holds a different number and box; only the answer after it counts.
    full = "<think>maybe #### 7 or \\boxed{7}</think>So \\boxed{5}, \\boxed{5}"
    after = "So \\boxed{5}, \\boxed{5}"
    for source in ("orbit/math", "orbit/math#think"):
        closed = loop_info(think="closed", answer=after)
        assert score(source, full, "5", extra_info=closed) == plain(1.0)
        assert score(source, full, "7", extra_info=closed)["score"] == 0.0
    # Without the agent loop's fields the whole text is checked, and its last box is the one after the block.
    assert score("orbit/math", full, "5")["score"] == 1.0
    # A thinking block with an empty answer after it scores 0.
    assert score("orbit/math", "<think>\\boxed{5}</think>", "5", extra_info=loop_info(think="closed", answer=""))["score"] == 0.0


def test_agent_loop_fields_reach_compute_score_through_verl(qwen3_tokenizer):
    """verl's naive reward manager must hand the agent loop's fields to compute_score.

    compute_score falls back to the whole text when the fields are missing. A verl change that
    stopped passing them would therefore turn both rules off without any error.
    """
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
    from verl.trainer.ppo.reward import get_custom_reward_fn

    tokenizer = qwen3_tokenizer
    config = OmegaConf.load(reward_config_path())
    manager = NaiveRewardManager(config, tokenizer, get_custom_reward_fn(config))

    def run(text, fields):
        # Built the way verl's agent loop builds the one-response batch it sends to a reward worker.
        prompt = tokenizer.encode("Q: x", add_special_tokens=False)
        response = tokenizer.encode(text, add_special_tokens=False)
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([prompt]),
                "responses": torch.tensor([response]),
                "attention_mask": torch.ones(1, len(prompt) + len(response), dtype=torch.long),
            },
            non_tensors={
                "data_source": np.array(["orbit/math"]),
                "reward_model": np.array([{"ground_truth": "5"}], dtype=object),
                "extra_info": np.array([{"index": 0}], dtype=object),
                "tool_extra_fields": np.array([fields], dtype=object),
                "__num_turns__": np.array([2]),
            },
        )
        return manager.loop.run_until_complete(manager.run_single(data))["reward_extra_info"]

    cut = run("\\boxed{5}", loop_info("length"))
    assert cut["score"] == 0.0 and cut["answer_correct"] == 1.0 and cut["truncated"] == 1.0
    assert run("\\boxed{5}", loop_info("eos"))["score"] == 1.0
    assert run("\\boxed{5}", loop_info(think="unclosed"))["think_unclosed"] == 1.0
    # The forced-close flag travels the same way. Without it, a forced answer would earn a full reward.
    forced = run("\\boxed{5}", loop_info(forced=True))
    assert forced["score"] == 0.0 and forced["answer_correct"] == 1.0 and forced["think_forced"] == 1.0


def test_unclosed_thinking_scores_zero():
    result = score("orbit/math", "<think>so \\boxed{5}", "5", extra_info=loop_info(think="unclosed"))
    assert result == {"score": 0.0, "acc": 0.0, "answer_correct": 0.0, "truncated": 0.0, "think_unclosed": 1.0,
                      "looped": 0.0, "think_forced": 0.0, **CODE_ZEROS}
    # Truncated and unclosed at once: both flags are set, and the reward is the truncation score.
    both = score("orbit/math", "<think>\\boxed{5}", "5", extra_info=loop_info("length", "unclosed"), truncated_score=-1.0)
    assert both["score"] == -1.0 and both["truncated"] == 1.0 and both["think_unclosed"] == 1.0


# --------------------------------------------------------------------------- the thinking budget

# A response whose thinking block the agent loop closed by force: the model's thinking, the inserted
# close, and a right answer after it.
FORCED_TEXT = "<think>long thinking</think>So \\boxed{5}, \\boxed{5}"
FORCED_ANSWER = "So \\boxed{5}, \\boxed{5}"


def test_rules_of_the_budget_are_off_by_default():
    # Passing the defaults explicitly changes nothing, for gold rows and pool rows alike.
    defaults = {"forced_score_scale": None}
    forced = loop_info(think="closed", answer=FORCED_ANSWER, forced=True)
    for case, info in ((("orbit/math", "\\boxed{5}", "5"), loop_info()),
                       (("orbit/math", FORCED_TEXT, "5"), loop_info(think="closed", answer=FORCED_ANSWER)),
                       (("orbit/math", FORCED_TEXT, "5"), forced),
                       (("orbit/none", "x", ""), pool_info())):
        assert score(*case, extra_info=info, **defaults) == score(*case, extra_info=info)
    # A pool row that closed its thinking by itself and answered still earns nothing.
    assert score("orbit/none", "<think>x</think>The answer is 5.", "", extra_info=pool_info()) == plain(0.0)


def test_forced_answer_scores_like_a_cut_one_by_default():
    # The answer after the forced close is right, but without the budget the model would have kept
    # thinking past it. A reward here would teach the model to think past the budget.
    for source in ("orbit/math", "orbit/math#think"):
        info = loop_info(think="closed", answer=FORCED_ANSWER, forced=True)
        result = score(source, FORCED_TEXT, "5", extra_info=info)
        assert result == {**plain(0.0), "answer_correct": 1.0, "think_forced": 1.0}
        # The forced answer gets whatever a cut answer gets.
        assert score(source, FORCED_TEXT, "5", extra_info=info, truncated_score=-1.0)["score"] == -1.0
        # With the truncation rule off, the forced answer is scored on its text like any other.
        assert score(source, FORCED_TEXT, "5", extra_info=info, truncated_score=None)["score"] == 1.0
    # The same answer without a forced close keeps its full reward.
    unforced = loop_info(think="closed", answer=FORCED_ANSWER)
    assert score("orbit/math", FORCED_TEXT, "5", extra_info=unforced) == plain(1.0)


def test_forced_score_scale_scales_the_correctness_of_forced_answers():
    forced = loop_info(think="closed", answer=FORCED_ANSWER, forced=True)
    result = score("orbit/math", FORCED_TEXT, "5", extra_info=forced, forced_score_scale=0.5)
    assert result == {**plain(0.5), "answer_correct": 1.0, "think_forced": 1.0}
    assert score("orbit/math", FORCED_TEXT, "5", extra_info=forced, forced_score_scale=0.5)["score"] == 0.5
    for scale in (0.0, 1.0):
        assert score("orbit/math", FORCED_TEXT, "5", extra_info=forced, forced_score_scale=scale)["score"] == scale
    # A wrong forced answer scores 0 whatever the scale.
    wrong = score("orbit/math", FORCED_TEXT, "7", extra_info=forced, forced_score_scale=0.5)
    assert wrong == {**plain(0.0), "think_forced": 1.0}
    # The scale replaces the cut-answer score for forced answers, so truncated_score no longer applies to them.
    assert score("orbit/math", FORCED_TEXT, "5", extra_info=forced, forced_score_scale=0.5,
                 truncated_score=-1.0)["score"] == 0.5
    # Answers that were not forced keep their plain reward under any scale.
    unforced = loop_info(think="closed", answer=FORCED_ANSWER)
    assert score("orbit/math", FORCED_TEXT, "5", extra_info=unforced, forced_score_scale=0.5) == plain(1.0)


def test_truncation_wins_over_a_forced_close():
    # The answer after a forced close ran into the length limit as well, so it is a cut answer.
    both = loop_info("length", "closed", FORCED_ANSWER, forced=True)
    result = score("orbit/math", FORCED_TEXT, "5", extra_info=both, forced_score_scale=1.0)
    assert result == {**plain(0.0), "answer_correct": 1.0, "truncated": 1.0, "think_forced": 1.0}
    assert score("orbit/math", FORCED_TEXT, "5", extra_info=both, forced_score_scale=1.0,
                 truncated_score=-1.0)["score"] == -1.0
    # With the truncation rule off, the forced rule decides.
    assert score("orbit/math", FORCED_TEXT, "5", extra_info=both, forced_score_scale=0.5,
                 truncated_score=None)["score"] == 0.5


def test_unknown_keyword_arguments_are_refused():
    # A misspelled reward kwarg would otherwise leave its rule off without a word.
    with pytest.raises(ValueError, match="forced_scale"):
        score("orbit/math", "\\boxed{5}", "5", forced_scale=0.5)
    with pytest.raises(ValueError, match="unknown keyword arguments"):
        score("orbit/none", "x", "", extra_info=pool_info(), forced_score=0.3)
    # verl's own two extras are accepted next to the config's keywords.
    accepted = score("orbit/math", "\\boxed{5}", "5", reward_router_address="127.0.0.1:8000", reward_model_tokenizer=None)
    assert accepted == plain(1.0)
    assert reward.VERL_REWARD_KWARGS == {"reward_router_address", "reward_model_tokenizer"}


def test_every_reward_kwarg_of_the_configs_is_accepted():
    """compute_score refuses keywords it does not know, so a config that sets one would fail at its first reward."""
    import yaml

    from orbit_verl.data.prompts import repo_root

    found = {}
    for path in sorted((repo_root() / "configs" / "verl").glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        function = ((doc.get("reward") or {}).get("custom_reward_function") or {}) if isinstance(doc, dict) else {}
        kwargs = function.get("reward_kwargs") or {}
        if kwargs:
            found[path.stem] = sorted(kwargs)
            assert score("orbit/math", "\\boxed{5}", "5", extra_info=loop_info(), **kwargs)["score"] == 1.0, path.name
    assert found["distill_rl_sft_16k"] == ["forced_score_scale"]
    assert {"orbit_base", "distill_rl_sft"} <= set(found)


def test_the_16k_reward_kwargs_reach_compute_score_through_verl():
    """verl binds the merged reward_kwargs of the 16K hybrid, and a forced answer must then score 0."""
    from omegaconf import OmegaConf
    from verl.trainer.ppo.reward import get_custom_reward_fn

    from orbit_verl.data.prompts import repo_root

    config_dir = repo_root() / "configs" / "verl"
    # The files in the order of their defaults lists, so later files override earlier ones as in hydra.
    names = ("orbit_base", "distill_sft", "distill_rl_sft", "distill_rl_sft_16k")
    config = OmegaConf.merge(*(OmegaConf.load(config_dir / f"{name}.yaml") for name in names))
    fn = get_custom_reward_fn(config)
    forced = loop_info(think="closed", answer="\\boxed{5}", forced=True)
    result = asyncio.run(fn(data_source="orbit/math", solution_str="<think>x</think>\\boxed{5}", ground_truth="5",
                            extra_info=forced, reward_router_address="127.0.0.1:8000", reward_model_tokenizer=None))
    assert result == {**plain(0.0), "answer_correct": 1.0, "think_forced": 1.0}
