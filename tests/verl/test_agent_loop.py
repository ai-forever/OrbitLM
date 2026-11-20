"""CPU tests for orbit_verl.data.agent_loop: the raw prompt ids, the response info for the reward
(how a response ended, where its thinking block ends, whether it loops), the loop-tail mask, the
no-think cap, the thinking budget (the forced close and the answer call after it), and the loop with
a fake server. The loop detector itself is tested in test_loops.py.

Run with the verl python and PYTHONPATH=<repo>/src. Most tests need the Qwen3 tokenizer of an SFT
export (ORBIT_TEST_SFT_EXPORT, see conftest.py) and skip without it.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from orbit_verl import common

common.apply_cache_env()

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap  # noqa: E402
from verl.utils.dataset.rl_dataset import RLHFDataset  # noqa: E402
from verl.workers.rollout.replica import TokenOutput  # noqa: E402

from orbit_verl.data import agent_loop as al  # noqa: E402
from orbit_verl.data import loops  # noqa: E402
from orbit_verl.data.prompts import (  # noqa: E402
    NO_THINK_MAX_TOKENS_ENV,
    PROMPT_MODES,
    THINK_BUDGET_ENV,
    prompt_token_ids,
    render_messages,
    repo_root,
)
from orbit_verl.data.response_info import (  # noqa: E402
    ANSWER_KEY,
    FINISH_KEY,
    FORCED_KEY,
    LOOP_KEY,
    LOOP_START_KEY,
    STOP_THINKING_TEXT,
    STOP_THINKING_TEXT_RU,
    STOP_THINKING_TEXTS,
    THINK_CLOSE,
    THINK_END_KEY,
    THINK_KEY,
    classify_finish,
    forced_close_ids,
    prompt_opens_think,
    split_think,
    think_token_ids,
)

RESPONSE_LENGTH = 64
# The ids of the Qwen3 tokenizer that the tests rely on.
END_OF_TEXT, IM_END, THINK_OPEN_ID, THINK_CLOSE_ID = 151643, 151645, 151667, 151668


@pytest.fixture(scope="module")
def tokenizer(qwen3_tokenizer):
    """The Qwen3 tokenizer of the SFT export; the tests that use it skip without one."""
    return qwen3_tokenizer


def test_the_tokenizer_is_the_chat_export_one(tokenizer):
    # A chat export ends a turn with <|im_end|>, and that is the eos the loop falls back to.
    assert tokenizer.eos_token_id == IM_END
    assert tokenizer.convert_tokens_to_ids(["<|endoftext|>", "<think>", "</think>"]) == [
        END_OF_TEXT, THINK_OPEN_ID, THINK_CLOSE_ID]


def test_registry_file_matches_prompt_modes():
    entries = OmegaConf.to_container(OmegaConf.load(repo_root() / "configs" / "verl" / "agent_loops.yaml"))
    assert len(entries) == len(PROMPT_MODES)
    for entry in entries:
        mode = PROMPT_MODES[entry["prompt_mode"]]
        assert entry["name"] == mode.agent_loop
        assert hydra.utils.get_class(entry["_target_"]) is al.OrbitLMAgentLoop


class FakeServerManager:
    """Stands in for verl's LLM server client: records each request and returns fixed tokens.

    ``replies`` holds one pair of token ids and log-probs per call, in the order of the calls. The
    last pair also answers every further call, so a test that gives one pair gets it from each call.
    """

    def __init__(self, replies):
        # A reply may carry a third item: more TokenOutput fields, such as num_preempted.
        self.replies = [(list(reply[0]), list(reply[1]), dict(reply[2]) if len(reply) > 2 else {}) for reply in replies]
        self.calls = []

    async def generate(self, *, request_id, prompt_ids, sampling_params, **kwargs):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": sampling_params,
                "priority": kwargs.get("priority"),
            }
        )
        token_ids, log_probs, more = self.replies[min(len(self.calls), len(self.replies)) - 1]
        return TokenOutput(token_ids=list(token_ids), log_probs=list(log_probs), **more)


# The log-prob that the fake server gives each id of a second call. It differs from the -0.5 of the
# first call, so a test can tell which call wrote each id of the merged response.
ANSWER_LOGPROB = -0.25
# Passed as ``extra_info`` to run a sample without that argument at all, as a row whose dataset wrote
# no extra_info would arrive.
NO_EXTRA_INFO = object()


def _loop_kwargs(tokenizer, server, response_length):
    """The constructor arguments that verl's AgentLoopWorker gives every agent loop.

    The registry entry adds ``prompt_mode`` and ``name`` to these.
    """
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"prompt_length": 1024, "response_length": response_length}},
            "data": {"apply_chat_template_kwargs": {}},
        }
    )
    return {
        "trainer_config": DictConfigWrap(config=config),
        "server_manager": server,
        "tokenizer": tokenizer,
        "processor": None,
        "hf_model_type": "ouro",
        "dataset_cls": RLHFDataset,
        "data_config": DictConfigWrap(config.data),
        "tools": ToolListWrap([]),
    }


def _run_loop(
    tokenizer,
    agent_name,
    completion,
    question="Tom buys 3 apples. How many?",
    token_ids=None,
    response_length=RESPONSE_LENGTH,
    answer_ids=None,
    extra_info=None,
    answer_extra=None,
):
    """Instantiate the loop the way verl's AgentLoopWorker does and run one sample.

    The fake server returns ``token_ids`` when they are given, and the encoded ``completion``
    otherwise. When ``answer_ids`` are given, the second call returns them instead, as the answer
    call after the thinking budget would. ``extra_info`` is an empty dict unless the test passes one,
    and NO_EXTRA_INFO leaves the argument out.
    """
    entries = OmegaConf.load(repo_root() / "configs" / "verl" / "agent_loops.yaml")
    entry = next(e for e in entries if e.name == agent_name)
    if token_ids is None:
        token_ids = tokenizer.encode(completion, add_special_tokens=False)
    token_ids = list(token_ids)
    replies = [(token_ids, [-0.5] * len(token_ids))]
    if answer_ids is not None:
        replies.append((list(answer_ids), [ANSWER_LOGPROB] * len(answer_ids), answer_extra or {}))
    server = FakeServerManager(replies)
    mode = PROMPT_MODES[entry.prompt_mode]
    messages = render_messages(mode, question)
    sampling_params = {"temperature": 0.9, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": True}
    run_kwargs = {"raw_prompt": messages, "index": 0}
    if extra_info is not NO_EXTRA_INFO:
        run_kwargs["extra_info"] = {} if extra_info is None else extra_info

    async def go():
        loop_obj = hydra.utils.instantiate(config=entry, **_loop_kwargs(tokenizer, server, response_length))
        assert loop_obj.prompt_mode is mode and loop_obj.agent_name == agent_name
        return await loop_obj.run(sampling_params, **run_kwargs)

    output = asyncio.run(go())
    return output, server, sampling_params, messages, token_ids


def _build_loop(tokenizer, agent_name, response_length):
    """Construct the loop of ``agent_name`` with the arguments of _run_loop, without running a sample.

    The loop is built directly rather than through hydra. Hydra wraps an error of the constructor in
    its own exception type, and the tests that use this helper check the constructor's ValueError.
    verl's base constructor looks up the running event loop, so the loop is built inside one.
    """
    mode = next(m for m in PROMPT_MODES.values() if m.agent_loop == agent_name)
    server = FakeServerManager([([], [])])

    async def go():
        return al.OrbitLMAgentLoop(
            prompt_mode=mode.name, name=agent_name, **_loop_kwargs(tokenizer, server, response_length)
        )

    return asyncio.run(go())


def _orbit_fields(output):
    """The response-info entries of ``extra_fields``, without the entries that verl adds itself."""
    return {key: value for key, value in output.extra_fields.items() if key.startswith("orbit_")}


def test_loop_run_sends_raw_prompt_ids_and_no_stop_strings(tokenizer, monkeypatch):
    monkeypatch.delenv(NO_THINK_MAX_TOKENS_ENV, raising=False)
    completion = "Tom has 3 apples, so \\boxed{3}.<|im_end|>"
    output, server, params, messages, token_ids = _run_loop(tokenizer, "orbit_sft_no_think", completion)
    sent = server.calls[0]["sampling_params"]
    # The SFT modes end at the end-of-turn token, a stop id of the model dir, so nothing else is sent.
    assert sent == params
    assert "stop" not in sent and "include_stop_str_in_output" not in sent and "stop_token_ids" not in sent
    expected_prompt = prompt_token_ids(tokenizer, messages)
    assert server.calls[0]["prompt_ids"] == expected_prompt == output.prompt_ids
    assert expected_prompt[0] == END_OF_TEXT
    assert output.response_ids == token_ids and token_ids[-1] == IM_END
    assert len(output.response_mask) == len(output.response_ids) == len(output.response_logprobs)
    assert output.response_mask == [1] * len(output.response_ids)
    assert output.extra_fields[FINISH_KEY] == "eos"
    # The prompt closed an empty thinking block, and the response has none of its own.
    assert output.extra_fields[THINK_KEY] == "none" and output.extra_fields[ANSWER_KEY] is None
    assert output.extra_fields[LOOP_KEY] is False and output.extra_fields[LOOP_START_KEY] == -1
    # The output must still convert the way verl converts it.
    converted = output.as_dict()
    assert converted["responses"].tolist() == output.response_ids


def test_loop_run_caps_at_the_response_length(tokenizer, monkeypatch):
    # A response longer than the response length is capped by verl, and it counts as cut.
    monkeypatch.delenv(NO_THINK_MAX_TOKENS_ENV, raising=False)
    completion = "x " * (RESPONSE_LENGTH * 2)
    output, _, _, _, token_ids = _run_loop(tokenizer, "orbit_sft_no_think", completion)
    assert output.response_ids == token_ids[:RESPONSE_LENGTH]
    assert output.extra_fields[FINISH_KEY] == "length"


def test_loop_run_reports_the_thinking_block(tokenizer, monkeypatch):
    monkeypatch.delenv(THINK_BUDGET_ENV, raising=False)
    # The sft_think prompt opens the block, so the response closes it with </think> before its answer.
    completion = "maybe \\boxed{1}</think>So the answer is \\boxed{2}.<|im_end|>"
    output, _, _, _, _ = _run_loop(tokenizer, "orbit_sft_think", completion)
    assert output.extra_fields[THINK_KEY] == "closed"
    assert output.extra_fields[ANSWER_KEY] == "So the answer is \\boxed{2}."
    assert output.extra_fields[FINISH_KEY] == "eos"

    # A thinking block that runs into the length limit is unclosed and truncated.
    output, _, _, _, _ = _run_loop(tokenizer, "orbit_sft_think", "step " * RESPONSE_LENGTH)
    assert output.extra_fields[THINK_KEY] == "unclosed" and output.extra_fields[ANSWER_KEY] is None
    assert output.extra_fields[FINISH_KEY] == "length"


# The response-info keys that every response carries, in every mode, whether or not its thinking
# block was closed by force.
RESPONSE_INFO_KEYS = frozenset(
    {FINISH_KEY, THINK_KEY, ANSWER_KEY, LOOP_KEY, LOOP_START_KEY, FORCED_KEY, THINK_END_KEY}
)


def test_loop_run_extra_fields_have_one_key_set(tokenizer, monkeypatch):
    """Every response must carry the same response-info keys, because verl concatenates the batches of
    all agent-loop workers and needs their keys to match. Every mode of the registry is run, so a new
    mode is checked too. A response that the thinking budget closed by force must carry the same keys
    as the others, because a batch can hold both kinds."""
    monkeypatch.delenv(THINK_BUDGET_ENV, raising=False)
    monkeypatch.delenv(NO_THINK_MAX_TOKENS_ENV, raising=False)
    keys = set()
    for mode in PROMPT_MODES.values():
        for completion in ("The answer is \\boxed{3}.<|im_end|>", "x " * (RESPONSE_LENGTH * 2)):
            output, _, _, _, _ = _run_loop(tokenizer, mode.agent_loop, completion)
            keys.add(frozenset(_orbit_fields(output)))
            assert output.extra_fields[FORCED_KEY] is False and output.extra_fields[THINK_END_KEY] == -1, mode.name

    monkeypatch.setenv(THINK_BUDGET_ENV, str(BUDGET))
    output, _, _, _, _ = _run_loop(
        tokenizer,
        "orbit_sft_think",
        "",
        token_ids=list(range(1000, 1000 + BUDGET)),
        answer_ids=[tokenizer.eos_token_id],
        response_length=BUDGET_RESPONSE_LENGTH,
        extra_info=TRAIN_ROW,
    )
    assert output.extra_fields[FORCED_KEY] is True
    keys.add(frozenset(_orbit_fields(output)))
    assert keys == {RESPONSE_INFO_KEYS}


def test_classify_finish():
    end = {0, 2}
    assert classify_finish([5, 6], stopped_at_string=True, max_new_tokens=2, end_token_ids=end) == "stop"
    assert classify_finish([5, 6], stopped_at_string=False, max_new_tokens=2, end_token_ids=end) == "length"
    # The model ended exactly at the limit with an end token: that is not a truncation.
    assert classify_finish([5, 2], stopped_at_string=False, max_new_tokens=2, end_token_ids=end) == "eos"
    # Shorter than the limit and not cut at a stop string: vLLM stopped on an end token.
    assert classify_finish([5], stopped_at_string=False, max_new_tokens=2, end_token_ids=end) == "eos"
    assert classify_finish([], stopped_at_string=False, max_new_tokens=1, end_token_ids=end) == "eos"


def test_split_think():
    o, c = 3, 4
    assert split_think([7, 8], o, c, starts_inside=False) == ("none", None)
    assert split_think([o, 7, c, 8, 9], o, c, starts_inside=False) == ("closed", 3)
    # The answer starts after the LAST close.
    assert split_think([o, 7, c, 8, o, 9, c, 5], o, c, starts_inside=False) == ("closed", 7)
    # A block opened again after the last close is unfinished.
    assert split_think([o, 7, c, 8, o, 9], o, c, starts_inside=False) == ("unclosed", None)
    assert split_think([o, 7, 8], o, c, starts_inside=False) == ("unclosed", None)
    # A prompt that opened the block: the response must close it.
    assert split_think([7, 8], o, c, starts_inside=True) == ("unclosed", None)
    assert split_think([7, c, 8], o, c, starts_inside=True) == ("closed", 2)


def test_prompt_opens_think(tokenizer):
    open_id, _ = think_token_ids(tokenizer)
    ids = tokenizer.encode("Solve it.<think>\n", add_special_tokens=False)
    assert prompt_opens_think(tokenizer, ids, open_id)
    # A <think> in the middle of the prompt does not count; only a block left open at its end does.
    middle = tokenizer.encode("A <think> tag in the question, then text.", add_special_tokens=False)
    assert open_id in middle and not prompt_opens_think(tokenizer, middle, open_id)
    for name, mode in PROMPT_MODES.items():
        prompt = prompt_token_ids(tokenizer, render_messages(mode, "What is 1+1?"))
        assert prompt_opens_think(tokenizer, prompt, open_id) == mode.opens_think, name


def test_end_token_ids_reads_generation_config(tokenizer, tmp_path):
    with_generation = tmp_path / "with_generation"
    with_generation.mkdir()
    # The qwen3 preset of prepare_model writes both end tokens into generation_config.json.
    (with_generation / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [IM_END, END_OF_TEXT], "pad_token_id": END_OF_TEXT}), encoding="utf-8")
    (with_generation / "config.json").write_text('{"eos_token_id": 7}', encoding="utf-8")
    assert al.end_token_ids(str(with_generation), tokenizer) == frozenset({IM_END, END_OF_TEXT})
    # Without generation_config.json vLLM reads config.json, and so does the agent loop.
    only_config = tmp_path / "only_config"
    only_config.mkdir()
    (only_config / "config.json").write_text(json.dumps({"eos_token_id": END_OF_TEXT}), encoding="utf-8")
    assert al.end_token_ids(str(only_config), tokenizer) == frozenset({tokenizer.eos_token_id, END_OF_TEXT})
    # Without a model dir the tokenizer's eos is used.
    assert al.end_token_ids(None, tokenizer) == frozenset({tokenizer.eos_token_id})


def test_max_new_tokens_follows_the_server(tmp_path):
    """The limit is the smallest of the response length, a max_tokens in the params, and the context left."""
    stub = SimpleNamespace(response_length=64, rollout_config=SimpleNamespace(max_model_len=None), config=SimpleNamespace())
    stub.model_path = lambda: al.OrbitLMAgentLoop.model_path(stub)
    limit = al.OrbitLMAgentLoop.max_new_tokens
    assert limit(stub, {}, 10) == 64
    assert limit(stub, {"max_tokens": 16}, 10) == 16
    # The single-turn loop cuts every response to response_length, so a larger max_tokens cannot raise the limit.
    assert limit(stub, {"max_tokens": 128}, 10) == 64
    stub.rollout_config.max_model_len = 50
    assert limit(stub, {}, 10) == 40
    # With max_model_len null the server uses the model's max_position_embeddings, and so does the loop.
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 60}), encoding="utf-8")
    stub.rollout_config.max_model_len = None
    stub.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(model=SimpleNamespace(path=str(tmp_path))))
    assert limit(stub, {}, 10) == 50


def test_response_info_when_the_prompt_opens_the_block(tokenizer):
    """A prompt that ends with <think> puts the response inside the block, so only a </think> closes it."""
    stub = SimpleNamespace(
        tokenizer=tokenizer,
        config=SimpleNamespace(),
        response_length=RESPONSE_LENGTH,
        rollout_config=SimpleNamespace(max_model_len=None),
        prompt_mode=PROMPT_MODES["sft_chat_think"],
    )
    stub.model_path = lambda: None
    stub.max_new_tokens = lambda params, prompt_len: al.OrbitLMAgentLoop.max_new_tokens(stub, params, prompt_len)
    prompt = tokenizer.encode("Solve it.<think>\n", add_special_tokens=False)

    def info(text):
        output = SimpleNamespace(prompt_ids=prompt, response_ids=tokenizer.encode(text, add_special_tokens=False))
        return al.OrbitLMAgentLoop.response_info(stub, output, {})

    closed = info("1+1 is 2.</think>So \\boxed{2}.")
    assert closed[THINK_KEY] == "closed" and closed[ANSWER_KEY] == "So \\boxed{2}."
    assert info("1+1 is 2, so \\boxed{2}.")[THINK_KEY] == "unclosed"


def test_the_release_tokenizer_writes_one_close_in_every_language(tokenizer):
    """The 16K run's student has the Qwen3 vocabulary, so both stop-thinking texts must pass the loop's
    checks on that tokenizer, stay within the pre-flight's bound, and survive the reward's decoding."""
    from orbit_verl.distill.preflight import STOP_THINKING_MAX_IDS

    tok = tokenizer
    for text in STOP_THINKING_TEXTS.values():
        ids = forced_close_ids(tok, text)  # raises if the round trip or the single </think> fails
        assert ids.count(THINK_CLOSE_ID) == 1
        assert len(ids) <= STOP_THINKING_MAX_IDS
        # This tokenizer keeps </think> in the text that the reward and the rollout dump decode.
        assert tok.decode(list(ids), skip_special_tokens=True) == text


def test_think_markers_of_the_release_tokenizer(tokenizer):
    """The Qwen3 tokenizer keeps <think> and </think> in decoded text, so the answer must leave them out."""
    tok = tokenizer
    open_id, close_id = think_token_ids(tok)
    assert (open_id, close_id) == (THINK_OPEN_ID, THINK_CLOSE_ID)
    ids = tok.encode("<think>a 7</think>So 5", add_special_tokens=False)
    # This tokenizer keeps the markers in the text the reward manager decodes.
    assert tok.decode(ids, skip_special_tokens=True) == "<think>a 7</think>So 5"
    status, start = split_think(ids, open_id, close_id, starts_inside=False)
    assert (status, start) == ("closed", ids.index(close_id) + 1)
    assert tok.decode(ids[start:], skip_special_tokens=True) == "So 5"
    assert prompt_opens_think(tok, tok.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False), open_id)
    # A template that writes an empty thinking block, as Qwen3 does with thinking turned off, leaves it closed.
    empty = tok.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
    assert not prompt_opens_think(tok, empty, open_id)


def test_unknown_prompt_mode_fails_at_construction(tokenizer):
    with pytest.raises(ValueError, match="Unknown prompt mode"):
        al.OrbitLMAgentLoop(prompt_mode="nope", name="x")


# The loop tests need responses longer than the detector's 512-id minimum, so they use a longer
# response length than the other tests.
LOOP_RESPONSE_LENGTH = 1536
# Ordinary text: 400 distinct ids, so none of its 16-grams repeats. Then one block of 13 other ids,
# repeated past the response length. The ids are given directly rather than encoded from text, so
# the loop starts exactly at index 400 whatever the tokenizer does with the text.
LOOP_PREFIX = list(range(1000, 1400))
LOOP_PERIOD = list(range(2000, 2013))
LOOPING_IDS = LOOP_PREFIX + LOOP_PERIOD * 120


def _run_looping(tokenizer, token_ids=LOOPING_IDS):
    return _run_loop(
        tokenizer, "orbit_sft_chat_no_think", "", token_ids=token_ids, response_length=LOOP_RESPONSE_LENGTH
    )[0]


@pytest.fixture
def no_caps(monkeypatch):
    """The loop tests run the sft_chat_no_think mode, so the no-think cap must be off for them."""
    monkeypatch.delenv(NO_THINK_MAX_TOKENS_ENV, raising=False)


def test_loop_is_reported_and_left_unmasked_by_default(tokenizer, monkeypatch, no_caps):
    """Without ORBIT_MASK_LOOP_TAIL the loop is only reported, and the output is what it was before
    the loop detector existed: the same ids, log-probs and an all-ones mask."""
    monkeypatch.delenv(loops.MASK_LOOP_TAIL_ENV, raising=False)
    output = _run_looping(tokenizer)
    assert output.extra_fields[FINISH_KEY] == "length"
    assert output.extra_fields[LOOP_KEY] is True and output.extra_fields[LOOP_START_KEY] == len(LOOP_PREFIX)
    assert output.response_ids == LOOPING_IDS[:LOOP_RESPONSE_LENGTH]
    assert output.response_logprobs == [-0.5] * LOOP_RESPONSE_LENGTH
    assert output.response_mask == [1] * LOOP_RESPONSE_LENGTH


def test_loop_tail_mask_when_enabled(tokenizer, monkeypatch, no_caps):
    """With the switch on, a looping response cut by the length limit keeps its ids and log-probs. Its
    mask is 1 up to the end of the first 16 ids of the first repeated copy, and 0 from there to the end."""
    monkeypatch.setenv(loops.MASK_LOOP_TAIL_ENV, "true")
    output = _run_looping(tokenizer)
    start = len(LOOP_PREFIX)
    # The first repeated copy of the 13-id block starts one period after the loop's start.
    cut = start + len(LOOP_PERIOD) + loops.LOOP_NGRAM
    assert output.extra_fields[FINISH_KEY] == "length" and output.extra_fields[LOOP_START_KEY] == start
    assert output.response_ids == LOOPING_IDS[:LOOP_RESPONSE_LENGTH]
    assert output.response_logprobs == [-0.5] * LOOP_RESPONSE_LENGTH
    assert len(output.response_mask) == len(output.response_ids)
    assert output.response_mask == [1] * cut + [0] * (LOOP_RESPONSE_LENGTH - cut)
    # verl turns the mask into the tensor that the losses use as their token mask.
    converted = output.as_dict()
    assert converted["response_mask"].sum().item() == cut
    assert converted["responses"].shape == converted["response_mask"].shape


def test_loop_tail_mask_needs_a_response_cut_by_the_length_limit(tokenizer, monkeypatch, no_caps):
    """A response that loops and then ends on an end token is reported as a loop but keeps its mask."""
    monkeypatch.setenv(loops.MASK_LOOP_TAIL_ENV, "true")
    ids = LOOPING_IDS[:1000] + [tokenizer.eos_token_id]
    output = _run_looping(tokenizer, ids)
    assert output.extra_fields[FINISH_KEY] == "eos" and output.extra_fields[LOOP_KEY] is True
    assert output.response_mask == [1] * len(ids)


def test_loop_tail_mask_leaves_a_clean_truncated_response_alone(tokenizer, monkeypatch, no_caps):
    monkeypatch.setenv(loops.MASK_LOOP_TAIL_ENV, "true")
    ids = list(range(1000, 1000 + LOOP_RESPONSE_LENGTH + 50))
    output = _run_looping(tokenizer, ids)
    assert output.extra_fields[FINISH_KEY] == "length"
    assert output.extra_fields[LOOP_KEY] is False and output.extra_fields[LOOP_START_KEY] == -1
    assert output.response_mask == [1] * LOOP_RESPONSE_LENGTH


def test_response_info_reports_the_loop_start(tokenizer):
    """response_info runs the detector on the response ids that verl trains on."""
    stub = SimpleNamespace(
        tokenizer=tokenizer,
        config=SimpleNamespace(),
        response_length=LOOP_RESPONSE_LENGTH,
        rollout_config=SimpleNamespace(max_model_len=None),
        prompt_mode=PROMPT_MODES["sft_chat_no_think"],
    )
    stub.model_path = lambda: None
    stub.max_new_tokens = lambda params, prompt_len: al.OrbitLMAgentLoop.max_new_tokens(stub, params, prompt_len)
    ids = LOOPING_IDS[:LOOP_RESPONSE_LENGTH]
    info = al.OrbitLMAgentLoop.response_info(stub, SimpleNamespace(prompt_ids=[], response_ids=ids), {})
    assert (info[LOOP_KEY], info[LOOP_START_KEY]) == loops.detect_loop(ids) == (True, len(LOOP_PREFIX))
    short = LOOPING_IDS[:300]
    info = al.OrbitLMAgentLoop.response_info(stub, SimpleNamespace(prompt_ids=[], response_ids=short), {})
    assert (info[LOOP_KEY], info[LOOP_START_KEY]) == (False, -1)


def test_misspelled_mask_switch_fails_at_construction(monkeypatch):
    monkeypatch.setenv(loops.MASK_LOOP_TAIL_ENV, "ture")
    with pytest.raises(ValueError, match=loops.MASK_LOOP_TAIL_ENV):
        al.OrbitLMAgentLoop(prompt_mode="sft_chat_think", name="orbit_sft_chat_think")


# A cap below RESPONSE_LENGTH, so that a capped answer is shorter than an uncapped one.
NO_THINK_CAP = 32


def test_no_think_answers_are_capped_and_count_as_cut(tokenizer, monkeypatch):
    """With the cap set, a no-think mode asks the server for at most that many tokens. An answer that
    fills the cap without an end token is cut by the length limit, as the reward must see it."""
    monkeypatch.setenv(NO_THINK_MAX_TOKENS_ENV, str(NO_THINK_CAP))
    capped = list(range(1000, 1000 + NO_THINK_CAP))
    for agent in ("orbit_sft_no_think", "orbit_sft_no_think_code", "orbit_sft_chat_no_think"):
        output, server, params, _, _ = _run_loop(tokenizer, agent, "", token_ids=capped)
        assert server.calls[0]["sampling_params"]["max_tokens"] == NO_THINK_CAP, agent
        assert "max_tokens" not in params  # the caller's dict is not modified
        assert output.response_ids == capped and output.extra_fields[FINISH_KEY] == "length", agent
        # The same answer one token shorter ended on its own.
        output, _, _, _, _ = _run_loop(tokenizer, agent, "", token_ids=capped[:-1])
        assert output.extra_fields[FINISH_KEY] == "eos", agent


def test_the_no_think_cap_leaves_think_modes_and_runs_without_it_alone(tokenizer, monkeypatch):
    monkeypatch.setenv(NO_THINK_MAX_TOKENS_ENV, str(NO_THINK_CAP))
    for agent in ("orbit_sft_think", "orbit_sft_think_code", "orbit_sft_chat_think", "orbit_sft_think_ru"):
        _, server, _, _, _ = _run_loop(tokenizer, agent, "The answer is 3.")
        assert "max_tokens" not in server.calls[0]["sampling_params"], agent
    monkeypatch.delenv(NO_THINK_MAX_TOKENS_ENV)
    _, server, _, _, _ = _run_loop(tokenizer, "orbit_sft_no_think", "The answer is 3.")
    assert "max_tokens" not in server.calls[0]["sampling_params"]


def test_a_bad_no_think_cap_fails_at_construction(monkeypatch):
    monkeypatch.setenv(NO_THINK_MAX_TOKENS_ENV, "4k")
    with pytest.raises(ValueError, match=NO_THINK_MAX_TOKENS_ENV):
        al.OrbitLMAgentLoop(prompt_mode="sft_think", name="orbit_sft_think")


# The thinking-budget tests use a budget well below their response length. The answer call after a
# forced close then has more room than MIN_ANSWER_TOKENS, so the constructor accepts the budget. The
# room also covers the Russian stop-thinking text, which takes more ids than the English one.
BUDGET = 128
BUDGET_RESPONSE_LENGTH = 1024
# The agents of the modes whose prompt opens a thinking block. Only these modes are limited.
THINK_AGENTS = ("orbit_sft_think", "orbit_sft_think_code", "orbit_sft_chat_think", "orbit_sft_think_ru")
# The extra_info of a training row and of a validation row, as both dataset builders write them.
TRAIN_ROW = {"index": 0, "split": "train"}
VAL_ROW = {"index": 0, "split": "val"}
# An answer that ends on its own, encoded with the end token in the tests below.
ANSWER_TEXT = "So the answer is \\boxed{2}."


def _stop_thinking_ids(tokenizer, text=STOP_THINKING_TEXT):
    """The ids that the loop inserts at the thinking budget, encoded here as forced_close_ids encodes them.

    The tests build their expected responses from this encoding rather than from the loop's own
    function, and test_the_tokenizer_writes_one_close checks that the two agree.
    """
    return tokenizer.encode(text, add_special_tokens=False)


def _ended_answer(tokenizer, text=ANSWER_TEXT):
    """The ids of an answer that the model ends with its end token."""
    return tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]


def test_the_think_agents_are_the_modes_that_open_a_block():
    assert {mode.agent_loop for mode in PROMPT_MODES.values() if mode.opens_think} == set(THINK_AGENTS)


def test_the_tokenizer_writes_one_close(tokenizer):
    """The stop-thinking text must encode to exactly one </think> and decode back to itself. Otherwise
    the loop refuses to start, and a forced response would not look closed to the reward."""
    _, close_id = think_token_ids(tokenizer)
    for text in STOP_THINKING_TEXTS.values():
        ids = _stop_thinking_ids(tokenizer, text)
        assert ids.count(close_id) == 1
        assert tokenizer.decode(ids, skip_special_tokens=False) == text
        assert forced_close_ids(tokenizer, text) == tuple(ids) == al.cached_forced_close_ids(tokenizer, text)
        # The ids after the close are the blank line that the answer follows.
        after_close = text.rpartition(THINK_CLOSE)[2]
        assert tokenizer.decode(ids[ids.index(close_id) + 1:], skip_special_tokens=True) == after_close


@pytest.mark.parametrize("answer_ends", [True, False], ids=["answer_ends", "answer_cut"])
@pytest.mark.parametrize("agent", THINK_AGENTS)
def test_a_block_still_open_at_the_budget_is_closed_by_force(tokenizer, monkeypatch, agent, answer_ends):
    """A training response that writes the whole budget inside its thinking block gets the stop-thinking
    ids and a second call for the answer. The inserted ids carry mask 0 and log-prob 0.0, and the
    response is judged on the answer call: it ends where that call ended."""
    monkeypatch.setenv(THINK_BUDGET_ENV, str(BUDGET))
    monkeypatch.delenv(loops.MASK_LOOP_TAIL_ENV, raising=False)
    _, close_id = think_token_ids(tokenizer)
    forced = _stop_thinking_ids(tokenizer)
    room = BUDGET_RESPONSE_LENGTH - BUDGET - len(forced)
    thinking = list(range(1000, 1000 + BUDGET))
    # An answer that fills all of its room without an end token was cut by the length limit.
    answer = _ended_answer(tokenizer) if answer_ends else list(range(3000, 3000 + room))
    output, server, params, _, _ = _run_loop(
        tokenizer,
        agent,
        "",
        token_ids=thinking,
        answer_ids=answer,
        response_length=BUDGET_RESPONSE_LENGTH,
        extra_info=TRAIN_ROW,
    )

    assert len(server.calls) == 2
    first, second = server.calls
    assert first["sampling_params"] == {**params, "max_tokens": BUDGET}
    assert "max_tokens" not in params  # the caller's dict is not modified
    assert first["prompt_ids"] == output.prompt_ids
    assert second["prompt_ids"] == output.prompt_ids + thinking + forced
    assert second["sampling_params"] == {**params, "max_tokens": room}
    # The answer call reuses the request id, so verl routes it to the replica that cached the thinking.
    assert first["request_id"] and second["request_id"] == first["request_id"]
    assert second["priority"] == first["priority"]

    assert output.response_ids == thinking + forced + answer
    assert output.response_mask == [1] * BUDGET + [0] * len(forced) + [1] * len(answer)
    assert output.response_logprobs == [-0.5] * BUDGET + [0.0] * len(forced) + [ANSWER_LOGPROB] * len(answer)
    info = output.extra_fields
    assert info[FORCED_KEY] is True and info[THINK_END_KEY] == BUDGET
    assert info[THINK_KEY] == "closed"
    assert info[ANSWER_KEY] == tokenizer.decode(forced[forced.index(close_id) + 1:] + answer, skip_special_tokens=True)
    if answer_ends:
        assert info[ANSWER_KEY] == STOP_THINKING_TEXT.rpartition(THINK_CLOSE)[2] + ANSWER_TEXT
    assert info[FINISH_KEY] == ("eos" if answer_ends else "length")
    assert info[LOOP_KEY] is False and info[LOOP_START_KEY] == -1
    # verl's metrics are a pydantic model; the answer call's time is added to the first call's.
    assert isinstance(output.metrics.generate_sequences, float) and output.metrics.generate_sequences >= 0
    # The output must still convert the way verl converts it, with the inserted ids out of the loss.
    converted = output.as_dict()
    assert converted["responses"].shape == converted["response_mask"].shape == converted["rollout_log_probs"].shape
    assert converted["response_mask"].sum().item() == BUDGET + len(answer)


@pytest.mark.parametrize("lang,text", [("ru", STOP_THINKING_TEXT_RU), ("en", STOP_THINKING_TEXT), (None, STOP_THINKING_TEXT)])
def test_the_stop_text_follows_the_row_language(tokenizer, monkeypatch, lang, text):
    """A Russian row gets the Russian stop-thinking text, so the answer after it stays in Russian; any
    other row, or a row without a language, gets the English one."""
    monkeypatch.setenv(THINK_BUDGET_ENV, str(BUDGET))
    monkeypatch.delenv(loops.MASK_LOOP_TAIL_ENV, raising=False)
    forced = _stop_thinking_ids(tokenizer, text)
    thinking = list(range(1000, 1000 + BUDGET))
    answer = _ended_answer(tokenizer)
    row = dict(TRAIN_ROW) if lang is None else {**TRAIN_ROW, "lang": lang}
    output, server, params, _, _ = _run_loop(
        tokenizer, "orbit_sft_think", "", token_ids=thinking, answer_ids=answer,
        response_length=BUDGET_RESPONSE_LENGTH, extra_info=row,
    )
    assert server.calls[1]["prompt_ids"] == output.prompt_ids + thinking + forced
    assert server.calls[1]["sampling_params"]["max_tokens"] == BUDGET_RESPONSE_LENGTH - BUDGET - len(forced)
    assert output.response_ids == thinking + forced + answer
    assert output.extra_fields[FORCED_KEY] is True and output.extra_fields[THINK_KEY] == "closed"


def test_the_answer_call_adds_its_preemptions_and_weight_version(tokenizer, monkeypatch):
    """The answer call is counted like a second turn: its preemptions add to the first call's, and the
    trajectory ends at the weight version of the answer call."""
    monkeypatch.setenv(THINK_BUDGET_ENV, str(BUDGET))
    output, _, _, _, _ = _run_loop(
        tokenizer, "orbit_sft_think", "", token_ids=list(range(1000, 1000 + BUDGET)),
        answer_ids=_ended_answer(tokenizer), response_length=BUDGET_RESPONSE_LENGTH, extra_info=TRAIN_ROW,
        answer_extra={"num_preempted": 2, "extra_fields": {"max_global_steps": 7}},
    )
    assert output.metrics.num_preempted == 2
    assert output.extra_fields["max_global_steps"] == 7


def test_a_block_closed_before_the_budget_continues_without_insertion(tokenizer, monkeypatch):
    """A response that closed its block and reached the budget while answering only goes on with its
    answer. Nothing is inserted, so the response is not forced and every id stays in the loss."""
    monkeypatch.setenv(THINK_BUDGET_ENV, str(BUDGET))
    monkeypatch.delenv(loops.MASK_LOOP_TAIL_ENV, raising=False)
    _, close_id = think_token_ids(tokenizer)
    thinking = list(range(1000, 1060))
    first_ids = thinking + [close_id] + list(range(1100, 1100 + BUDGET - len(thinking) - 1))
    answer = _ended_answer(tokenizer, " So \\boxed{2}.")
    output, server, params, _, _ = _run_loop(
        tokenizer,
        "orbit_sft_think",
        "",
        token_ids=first_ids,
        answer_ids=answer,
        response_length=BUDGET_RESPONSE_LENGTH,
        extra_info=TRAIN_ROW,
    )

    assert len(first_ids) == BUDGET and len(server.calls) == 2
    first, second = server.calls
    assert first["sampling_params"]["max_tokens"] == BUDGET
    assert second["prompt_ids"] == output.prompt_ids + first_ids
    assert second["sampling_params"] == {**params, "max_tokens": BUDGET_RESPONSE_LENGTH - BUDGET}
    assert second["request_id"] == first["request_id"]

    assert output.response_ids == first_ids + answer
    assert output.response_mask == [1] * (BUDGET + len(answer))
    assert output.response_logprobs == [-0.5] * BUDGET + [ANSWER_LOGPROB] * len(answer)
    info = output.extra_fields
    assert info[FORCED_KEY] is False and info[THINK_END_KEY] == -1
    assert info[THINK_KEY] == "closed" and info[FINISH_KEY] == "eos"
    assert info[ANSWER_KEY] == tokenizer.decode(first_ids[len(thinking) + 1:] + answer, skip_special_tokens=True)


def test_a_response_that_ends_before_the_budget_is_left_as_it_was(tokenizer, monkeypatch):
    """A response that ends on its end token before the budget needs no answer call. Its output is the
    output of the same run without a budget, apart from the max_tokens of the request."""
    ids = _ended_answer(tokenizer, "1+1 is 2.</think>So \\boxed{2}.")
    assert len(ids) < BUDGET
    run = dict(token_ids=ids, response_length=BUDGET_RESPONSE_LENGTH, extra_info=TRAIN_ROW)
    monkeypatch.delenv(THINK_BUDGET_ENV, raising=False)
    plain, plain_server, _, _, _ = _run_loop(tokenizer, "orbit_sft_think", "", **run)
    monkeypatch.setenv(THINK_BUDGET_ENV, str(BUDGET))
    output, server, _, _, _ = _run_loop(tokenizer, "orbit_sft_think", "", **run)

    assert "max_tokens" not in plain_server.calls[0]["sampling_params"]
    assert len(server.calls) == 1 and server.calls[0]["sampling_params"]["max_tokens"] == BUDGET
    assert output.response_ids == plain.response_ids == ids
    assert output.response_mask == plain.response_mask == [1] * len(ids)
    assert output.response_logprobs == plain.response_logprobs
    assert _orbit_fields(output) == _orbit_fields(plain)
    assert output.extra_fields[FORCED_KEY] is False and output.extra_fields[THINK_END_KEY] == -1
    assert output.extra_fields[THINK_KEY] == "closed" and output.extra_fields[FINISH_KEY] == "eos"


def test_a_response_that_ends_exactly_at_the_budget_is_not_continued(tokenizer, monkeypatch):
    """A response of exactly the budget whose last id is the end token ended on its own."""
    monkeypatch.setenv(THINK_BUDGET_ENV, str(BUDGET))
    ids = list(range(1000, 1000 + BUDGET - 1)) + [tokenizer.eos_token_id]
    output, server, _, _, _ = _run_loop(
        tokenizer,
        "orbit_sft_think",
        "",
        token_ids=ids,
        answer_ids=_ended_answer(tokenizer),
        response_length=BUDGET_RESPONSE_LENGTH,
        extra_info=TRAIN_ROW,
    )
    assert len(server.calls) == 1
    assert output.response_ids == ids and output.response_mask == [1] * BUDGET
    assert output.extra_fields[FINISH_KEY] == "eos"
    assert output.extra_fields[FORCED_KEY] is False and output.extra_fields[THINK_END_KEY] == -1


@pytest.mark.parametrize("agent", THINK_AGENTS)
def test_validation_rows_are_not_limited(tokenizer, monkeypatch, agent):
    """Validation measures how the model closes its block by itself, so its rows get no budget."""
    monkeypatch.setenv(THINK_BUDGET_ENV, str(BUDGET))
    ids = list(range(1000, 1000 + BUDGET + 10))
    output, server, _, _, _ = _run_loop(
        tokenizer,
        agent,
        "",
        token_ids=ids,
        answer_ids=_ended_answer(tokenizer),
        response_length=BUDGET_RESPONSE_LENGTH,
        extra_info=VAL_ROW,
    )
    assert len(server.calls) == 1
    assert "max_tokens" not in server.calls[0]["sampling_params"]
    assert output.response_ids == ids
    assert output.extra_fields[FORCED_KEY] is False and output.extra_fields[THINK_END_KEY] == -1


@pytest.mark.parametrize("extra_info", [NO_EXTRA_INFO, {}, {"index": 0}], ids=["missing", "empty", "no_split"])
@pytest.mark.parametrize("agent", THINK_AGENTS)
def test_a_row_without_a_split_is_refused_under_a_budget(tokenizer, monkeypatch, agent, extra_info):
    """A row without extra_info.split cannot be told to be a training or a validation row, so it is
    refused rather than limited or left alone by a guess."""
    monkeypatch.setenv(THINK_BUDGET_ENV, str(BUDGET))
    with pytest.raises(ValueError, match="extra_info.split"):
        _run_loop(tokenizer, agent, "1+1 is 2.", response_length=BUDGET_RESPONSE_LENGTH, extra_info=extra_info)


def test_modes_that_do_not_open_a_block_ignore_the_budget(tokenizer, monkeypatch):
    """The no-think modes run as without a budget, even on a row without a split. Their response length
    of 64 is below the budget, which the constructor accepts, because it checks the answer room only
    for the modes that open a block."""
    monkeypatch.setenv(THINK_BUDGET_ENV, str(BUDGET))
    monkeypatch.delenv(NO_THINK_MAX_TOKENS_ENV, raising=False)
    agents = [mode.agent_loop for mode in PROMPT_MODES.values() if not mode.opens_think]
    assert set(agents) == {"orbit_sft_no_think", "orbit_sft_no_think_code", "orbit_sft_chat_no_think",
                           "orbit_sft_no_think_ru"}
    for agent in agents:
        output, server, _, _, _ = _run_loop(tokenizer, agent, "The answer is 3.", extra_info={})
        assert len(server.calls) == 1, agent
        assert "max_tokens" not in server.calls[0]["sampling_params"], agent
        assert output.extra_fields[FORCED_KEY] is False and output.extra_fields[THINK_END_KEY] == -1, agent


@pytest.mark.parametrize("agent", THINK_AGENTS)
def test_a_budget_that_leaves_too_little_answer_room_fails_at_construction(tokenizer, monkeypatch, agent):
    """The answer after a forced close needs MIN_ANSWER_TOKENS of the response length. The largest budget
    that leaves exactly that much is accepted, and one token more is refused."""
    longest = max(len(_stop_thinking_ids(tokenizer, text)) for text in STOP_THINKING_TEXTS.values())
    largest = BUDGET_RESPONSE_LENGTH - longest - al.MIN_ANSWER_TOKENS
    monkeypatch.setenv(THINK_BUDGET_ENV, str(largest + 1))
    with pytest.raises(ValueError, match=THINK_BUDGET_ENV):
        _build_loop(tokenizer, agent, BUDGET_RESPONSE_LENGTH)
    monkeypatch.setenv(THINK_BUDGET_ENV, str(largest))
    loop_obj = _build_loop(tokenizer, agent, BUDGET_RESPONSE_LENGTH)
    assert loop_obj.think_budget == largest
    assert {text: list(ids) for text, ids in loop_obj.forced_ids_by_text.items()} == {
        text: _stop_thinking_ids(tokenizer, text) for text in STOP_THINKING_TEXTS.values()}


@pytest.mark.parametrize("value", ["6k", "0", "1.5", "-5"])
def test_a_bad_think_budget_fails_at_construction(monkeypatch, value):
    """The budget is read before the base constructor runs, so a bad value fails on the first sample
    of any mode, not only on the first sample of a mode that opens a block."""
    monkeypatch.setenv(THINK_BUDGET_ENV, value)
    for mode in ("sft_think", "sft_no_think"):
        with pytest.raises(ValueError, match=THINK_BUDGET_ENV):
            al.OrbitLMAgentLoop(prompt_mode=mode, name=PROMPT_MODES[mode].agent_loop)


# The forced-loop tests need thinking long enough to loop by the detector, which wants 512 ids or more,
# and room after the forced close for an answer longer than 512 ids.
LOOP_BUDGET = 1024
FORCED_LOOP_RESPONSE_LENGTH = 2048
# LOOPING_IDS cut at the budget: 400 ordinary ids, then the 13-id block repeated to the budget.
LOOPING_THINKING = LOOPING_IDS[:LOOP_BUDGET]
# Where the loop-tail mask of LOOPING_THINKING begins: one n-gram into the first repeated copy.
LOOPING_THINKING_CUT = len(LOOP_PREFIX) + len(LOOP_PERIOD) + loops.LOOP_NGRAM


def _run_forced_loop(tokenizer, answer):
    """Run an sft_think training row whose thinking loops until the budget, then ``answer``."""
    return _run_loop(
        tokenizer,
        "orbit_sft_think",
        "",
        token_ids=LOOPING_THINKING,
        answer_ids=answer,
        response_length=FORCED_LOOP_RESPONSE_LENGTH,
        extra_info=TRAIN_ROW,
    )[0]


def test_a_forced_response_masks_the_loop_in_its_thinking_only(tokenizer, monkeypatch):
    """With the switch on, the loop of a forced response is masked from its tail mask start up to the
    inserted ids. The inserted ids keep their mask 0, and the answer after them stays in the loss."""
    monkeypatch.setenv(THINK_BUDGET_ENV, str(LOOP_BUDGET))
    monkeypatch.setenv(loops.MASK_LOOP_TAIL_ENV, "true")
    forced = _stop_thinking_ids(tokenizer)
    answer = _ended_answer(tokenizer)
    output = _run_forced_loop(tokenizer, answer)
    cut = LOOPING_THINKING_CUT
    info = output.extra_fields
    assert info[FORCED_KEY] is True and info[THINK_END_KEY] == LOOP_BUDGET
    assert info[LOOP_KEY] is True and info[LOOP_START_KEY] == len(LOOP_PREFIX)
    assert info[FINISH_KEY] == "eos"
    # The mask changes, and the ids and log-probs do not.
    assert output.response_ids == LOOPING_THINKING + forced + answer
    assert output.response_logprobs == (
        [-0.5] * LOOP_BUDGET + [0.0] * len(forced) + [ANSWER_LOGPROB] * len(answer)
    )
    assert output.response_mask == [1] * cut + [0] * (LOOP_BUDGET - cut) + [0] * len(forced) + [1] * len(answer)


def test_a_forced_response_without_the_switch_masks_only_the_inserted_ids(tokenizer, monkeypatch):
    monkeypatch.setenv(THINK_BUDGET_ENV, str(LOOP_BUDGET))
    monkeypatch.delenv(loops.MASK_LOOP_TAIL_ENV, raising=False)
    forced = _stop_thinking_ids(tokenizer)
    answer = _ended_answer(tokenizer)
    output = _run_forced_loop(tokenizer, answer)
    assert output.extra_fields[LOOP_KEY] is True
    assert output.response_mask == [1] * LOOP_BUDGET + [0] * len(forced) + [1] * len(answer)


def test_a_loop_in_the_thinking_is_found_behind_a_long_answer(tokenizer, monkeypatch):
    """The detector reads only the last 1,024 ids. Behind an answer of 601 ids most of those are the
    answer, so a detector run on the whole response misses the loop. The loop runs it on the thinking
    and on the answer separately, and still finds the loop and masks it."""
    monkeypatch.setenv(THINK_BUDGET_ENV, str(LOOP_BUDGET))
    monkeypatch.setenv(loops.MASK_LOOP_TAIL_ENV, "true")
    forced = _stop_thinking_ids(tokenizer)
    answer = list(range(5000, 5600)) + [tokenizer.eos_token_id]
    output = _run_forced_loop(tokenizer, answer)
    cut = LOOPING_THINKING_CUT
    assert len(answer) > loops.LOOP_MIN_TOKENS
    assert loops.detect_loop(output.response_ids) == (False, -1)
    info = output.extra_fields
    assert info[LOOP_KEY] is True and info[LOOP_START_KEY] == len(LOOP_PREFIX)
    assert info[FORCED_KEY] is True and info[THINK_END_KEY] == LOOP_BUDGET and info[FINISH_KEY] == "eos"
    assert output.response_mask == [1] * cut + [0] * (LOOP_BUDGET - cut + len(forced)) + [1] * len(answer)
