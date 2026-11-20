"""The verl agent loop for OrbitLM: raw prompt tokens, response facts for the reward, and the length rules.

verl's stock ``SingleTurnAgentLoop`` tokenizes the prompt with the tokenizer's chat template. Our
prompt modes are the release chat format written out as text (``orbit_verl.data.prompts``), so
``OrbitLMAgentLoop`` builds the prompt ids with ``prompts.prompt_token_ids`` instead. The data tools
use the same function for their length filter, so the filter and the rollout never disagree.

It records in ``extra_fields``, for every response, how generation ended, where its thinking block
ends, and whether the response loops and from where (``orbit_verl.data.response_info`` explains
all three). The reward reads them: a response cut by the length limit gets ``truncated_score``, and
a response that ends inside a thinking block scores 0.

When ``ORBIT_MASK_LOOP_TAIL`` is on, a response that the length limit cut while it was looping
gets its response mask set to 0 from one n-gram into the first repeated copy to its end
(``orbit_verl.data.loops`` explains why). The ids, the log-probs and the reward are unchanged, so
the response is still scored as truncated, but no loss trains on the repeated tail. The variable
is off by default, so RL runs behave as before.

When ``ORBIT_NO_THINK_MAX_TOKENS`` is set, the modes whose prompt ends with a closed, empty
thinking block (``PromptMode.no_think``) generate at most that many tokens. Without the cap, a
looping no-think answer runs to the full response length and holds up the end of the rollout and
the teacher. The variable is unset by default, so RL runs behave as before.

When ``ORBIT_THINK_BUDGET`` is set, a training response of a mode that opens a thinking block
(``PromptMode.opens_think``) generates at most that many tokens first. If its block is still open
then, the loop appends Qwen3's stop-thinking sentence and "</think>" (``response_info.STOP_THINKING_TEXT``,
or its Russian version for a Russian row, so that the answer stays in Russian) with response mask 0, as
verl marks the tokens it inserts itself, and asks vLLM once more for the answer, in the rest of the
response length. A response that had closed its block before the budget
just continues its answer. Validation rows are never forced, so validation measures how the model
closes by itself. The variable is unset by default, so other runs behave as before.

Stop token ids are not set here. vLLM reads them from ``generation_config.json`` in the prepared
model dir, so the model preset is the one place that defines them.

``docs/verl/DESIGN.md`` summarises these rules. The registry entries that verl instantiates
are in ``configs/verl/agent_loops.yaml``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from orbit_verl.data.loops import detect_loop, mask_loop_tail, mask_loop_tail_enabled, tail_mask_start
from orbit_verl.data.prompts import get_mode, no_think_max_tokens, prompt_token_ids, think_budget
from orbit_verl.data.response_info import (
    ANSWER_KEY,
    FINISH_KEY,
    FORCED_KEY,
    LOOP_KEY,
    LOOP_START_KEY,
    STOP_THINKING_TEXTS,
    THINK_END_KEY,
    THINK_KEY,
    classify_finish,
    forced_close_ids,
    prompt_opens_think,
    split_think,
    stop_thinking_text,
    think_token_ids,
)

logger = logging.getLogger(__name__)

# verl looks up a token builder by the model's model_type. It has none for "ouro", so it warns and
# falls back to the default builder, once for every sample. The default builder is the right one
# for us, so the warning is only noise in every Ray log.
logging.getLogger("verl.utils.tokenizer.continuous_token_wiring").setLevel(logging.ERROR)

# End token ids per model dir. The agent loop needs them for every response, and reading the file
# once per process is enough, because a prepared model dir never changes during a run.
_END_TOKEN_IDS: dict[str, frozenset[int]] = {}


def end_token_ids(model_path: str | None, tokenizer: Any) -> frozenset[int]:
    """The ids that end generation for a model dir.

    vLLM stops at the tokenizer's eos and also at every eos id that ``generation_config.json`` in the
    prepared model dir lists, so this set holds both. When that file is missing, vLLM reads the eos
    ids from ``config.json`` instead, and so does this function. Without a model dir, as in the tests,
    only the tokenizer's eos is left.
    """
    key = str(model_path or "")
    cached = _END_TOKEN_IDS.get(key)
    if cached is not None:
        return cached
    ids: set[int] = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    path = os.path.join(key, "generation_config.json") if key else ""
    if path and not os.path.isfile(path):
        path = os.path.join(key, "config.json")
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle).get("eos_token_id")
        ids.update(int(i) for i in (value if isinstance(value, list) else [value]) if i is not None)
    _END_TOKEN_IDS[key] = frozenset(ids)
    return _END_TOKEN_IDS[key]


_CONTEXT_LENGTH: dict[str, int | None] = {}

# The least room the answer must have after the thinking budget and the inserted stop-thinking ids. A
# budget that leaves less is refused when the loop is built, because every forced answer would be cut.
MIN_ANSWER_TOKENS = 256

# The stop-thinking ids per tokenizer and text. verl builds one loop per sample, and encoding and
# checking each text once per process is enough.
_FORCED_IDS: dict[tuple[str, str], tuple[int, ...]] = {}


def cached_forced_close_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    """``response_info.forced_close_ids`` for this tokenizer and text, computed once per process."""
    key = (str(getattr(tokenizer, "name_or_path", "") or id(tokenizer)), text)
    if key not in _FORCED_IDS:
        _FORCED_IDS[key] = forced_close_ids(tokenizer, text)
    return _FORCED_IDS[key]


class RequestRecorder:
    """Stands in for the server manager during one rollout and remembers the request id it was given.

    verl makes the request id inside ``SingleTurnAgentLoop.run``, out of a subclass's reach. The answer
    call after a forced close reuses it, so the manager's sticky routing sends that call to the vLLM
    replica that holds the thinking in its prefix cache, and the long prefix is not computed again.
    """

    def __init__(self, inner: Any):
        self.inner = inner
        self.request_id: str | None = None

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        self.request_id = kwargs.get("request_id", args[0] if args else None)
        return await self.inner.generate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass(frozen=True)
class BudgetSplit:
    """Where a response continued after the thinking budget.

    ``think_end`` is the number of ids the first call wrote, which is where the inserted ids start.
    ``answer_start`` is where the second call's ids start, and ``answer_limit`` is how many it was
    allowed. ``forced`` is true when ids were inserted to close the thinking block. It is false when
    the block was already closed and only the answer went on, with nothing inserted.
    """

    think_end: int
    answer_start: int
    answer_limit: int
    forced: bool


def context_length(model_path: str | None) -> int | None:
    """``max_position_embeddings`` from the model dir's config.json, or None when it cannot be read.

    verl's vLLM server uses this value as ``max_model_len`` when ``rollout.max_model_len`` is null.
    The agent loop needs the same value to tell a response cut by the context from one that ended.
    """
    key = str(model_path or "")
    if key not in _CONTEXT_LENGTH:
        value = None
        path = os.path.join(key, "config.json") if key else ""
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                config = json.load(handle)
            value = config.get("max_position_embeddings") or (config.get("text_config") or {}).get(
                "max_position_embeddings"
            )
        _CONTEXT_LENGTH[key] = int(value) if value else None
    return _CONTEXT_LENGTH[key]


class OrbitLMAgentLoop(SingleTurnAgentLoop):
    """Single-turn agent loop for one OrbitLM prompt mode.

    verl builds one instance per sample, so the constructor only looks the mode up. ``name`` is the
    registry entry's name, which hydra passes as a constructor argument along with ``prompt_mode``.
    """

    def __init__(self, *args: Any, prompt_mode: str, name: str | None = None, **kwargs: Any):
        # Resolve the mode first, so a typo in agent_loops.yaml fails before any verl setup runs.
        self.prompt_mode = get_mode(prompt_mode)
        self.agent_name = name
        # The switch is read here rather than when a loop is found, so a misspelled value fails on
        # the first sample of the run instead of at the first loop, which may come much later.
        self.masks_loop_tail = mask_loop_tail_enabled()
        # The cap is read here for the same reason: a bad value fails on the first sample of any
        # mode, not only on the first no-think sample.
        self.no_think_max_tokens = no_think_max_tokens()
        # The thinking budget, read early for the same reason.
        self.think_budget = think_budget()
        super().__init__(*args, **kwargs)
        # The stop-thinking ids of every language, checked here so that a tokenizer that cannot write
        # one of them fails on the first sample, not on the first forced row of that language.
        self.forced_ids_by_text: dict[str, tuple[int, ...]] = {}
        if self.think_budget is not None and self.prompt_mode.opens_think:
            # The tokenizer exists only after the base constructor has run.
            self.forced_ids_by_text = {text: cached_forced_close_ids(self.tokenizer, text)
                                       for text in STOP_THINKING_TEXTS.values()}
            longest = max(len(ids) for ids in self.forced_ids_by_text.values())
            room = int(self.response_length) - self.think_budget - longest
            if room < MIN_ANSWER_TOKENS:
                raise ValueError(
                    f"ORBIT_THINK_BUDGET={self.think_budget} leaves {room} tokens of the response length "
                    f"{self.response_length} for the answer after up to {longest} stop-thinking ids; "
                    f"it needs at least {MIN_ANSWER_TOKENS}."
                )

    async def ct_build_initial_tokens(self, messages: list[dict], *args: Any, **kwargs: Any) -> list[int]:
        """Prompt token ids of the raw prompt text, without a chat template."""
        # Raw prompts are plain text. Tools and media need a chat template to be placed in the
        # prompt, so they cannot be honored here, and dropping them silently would hide a bug.
        extras = [key for key in ("tools", "images", "videos", "audios") if kwargs.get(key)]
        if args or extras:
            raise ValueError(
                f"Prompt mode {self.prompt_mode.name!r} renders raw text and supports no tools or "
                f"media, but got {extras or 'positional arguments'}."
            )
        loop = asyncio.get_running_loop()
        # Tokenizing a long prompt takes about a millisecond. That is long enough to stall the event
        # loop that drives every other sample, so it runs in the executor as verl's own tokenization does.
        prompt_ids = await loop.run_in_executor(None, lambda: prompt_token_ids(self.tokenizer, messages))
        return self._cap_text_prompt_length(prompt_ids)

    def budget_for(self, kwargs: Mapping[str, Any]) -> int | None:
        """The thinking budget of this sample, or None when its thinking is not limited.

        Only training rows of a mode that opens a thinking block are limited. Validation rows run to
        the full response length, so that validation measures how the model closes by itself. Both
        dataset builders write ``extra_info.split``. A row without it cannot be told apart, so it is
        refused rather than guessed.
        """
        if self.think_budget is None or not self.prompt_mode.opens_think:
            return None
        extra = kwargs.get("extra_info")
        split = extra.get("split") if isinstance(extra, Mapping) else None
        if split is None:
            raise ValueError(
                "ORBIT_THINK_BUDGET is set, but this row has no extra_info.split, so training and validation "
                "rows cannot be told apart."
            )
        return self.think_budget if split == "train" else None

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        """Generate one response, with the no-think length cap for a no-think mode.

        Under a thinking budget, generate up to the budget first and then the answer after a forced
        close (the module docstring). Then record in ``extra_fields`` how the response ended, where
        its thinking block ends and whether it loops, and mask the loop's tail when the switch is on.
        """
        # The caller's dict is shared by the n samples of a prompt, so it is copied, not changed.
        params = dict(sampling_params)
        if self.prompt_mode.no_think and self.no_think_max_tokens is not None:
            # verl's vLLM server generates at most max_tokens new tokens when the request names it.
            # max_new_tokens() reads the same key, so a no-think answer cut at the cap still counts
            # as cut by the length limit.
            cap = self.no_think_max_tokens
            params["max_tokens"] = cap if params.get("max_tokens") is None else min(int(params["max_tokens"]), cap)
        # The parameters of the full response, before the thinking budget caps the first call.
        full_params = dict(params)
        budget = self.budget_for(kwargs)
        recorder = None
        if budget is not None:
            params["max_tokens"] = budget if params.get("max_tokens") is None else min(int(params["max_tokens"]), budget)
            recorder = RequestRecorder(self.server_manager)
            self.server_manager = recorder
        try:
            output = await super().run(params, **kwargs)
        finally:
            if recorder is not None:
                self.server_manager = recorder.inner

        loop = asyncio.get_running_loop()
        split = None
        if budget is not None:
            extra = kwargs.get("extra_info")
            lang = extra.get("lang") if isinstance(extra, Mapping) else None
            split = await self.continue_after_budget(output, full_params, budget, recorder.request_id,
                                                     int(kwargs.get("priority", 0) or 0), lang)

        # Decoding the end of the prompt and the answer takes about as long as tokenizing a prompt.
        # That is long enough to stall the event loop that drives every other sample, so it also
        # runs in the executor.
        info = await loop.run_in_executor(None, lambda: self.response_info(output, params, split))
        output.extra_fields.update(info)
        if self.masks_loop_tail:
            if split is not None and split.forced:
                output.response_mask = self.mask_forced_loops(output, split, info[FINISH_KEY])
            elif info[FINISH_KEY] == "length" and info[LOOP_KEY]:
                # Only a response that the length limit cut is masked. A response that repeats itself for a while and
                # then ends on its own still reaches an answer, so all of its tokens stay in the loss.
                start = tail_mask_start(output.response_ids, info[LOOP_START_KEY])
                output.response_mask = mask_loop_tail(output.response_mask, start)
        return output

    async def continue_after_budget(self, output: AgentLoopOutput, full_params: dict[str, Any], budget: int,
                                    request_id: str | None, priority: int, lang: Any = None) -> BudgetSplit | None:
        """Write the answer of a response that the thinking budget stopped, or return None.

        Nothing happens when the first call ended before the budget. When it reached the budget with
        its thinking block still open, the stop-thinking ids are appended with response mask 0 and
        log-prob 0.0, the convention verl uses for the tokens it inserts itself. The text is the one of
        the row's language (``response_info.stop_thinking_text``). When the block was already closed,
        nothing is appended. Either way a second call writes on, with the room that the full response
        length leaves, and its ids, mask and log-probs are appended to the output.
        """
        ids = [int(token) for token in output.response_ids]
        end_ids = end_token_ids(self.model_path(), self.tokenizer)
        if classify_finish(ids, stopped_at_string=False, max_new_tokens=budget, end_token_ids=end_ids) != "length":
            return None
        markers = think_token_ids(self.tokenizer)
        if markers is None:
            return None
        status, _ = split_think(ids, markers[0], markers[1], starts_inside=True)
        insert = list(self.forced_ids_by_text[stop_thinking_text(lang)]) if status == "unclosed" else []
        room = self.max_new_tokens(full_params, len(output.prompt_ids)) - len(ids) - len(insert)
        if room <= 0:
            return None
        started = time.perf_counter()
        more = await self.server_manager.generate(
            request_id=request_id or uuid4().hex,
            prompt_ids=list(output.prompt_ids) + ids + insert,
            sampling_params={**full_params, "max_tokens": room},
            priority=priority,
        )
        new_ids = [int(token) for token in more.token_ids]
        length = int(self.response_length)
        output.response_ids = (ids + insert + new_ids)[:length]
        output.response_mask = ([int(m) for m in output.response_mask] + [0] * len(insert) + [1] * len(new_ids))[:length]
        if output.response_logprobs is not None:
            new_logprobs = list(more.log_probs) if more.log_probs else [0.0] * len(new_ids)
            output.response_logprobs = (list(output.response_logprobs) + [0.0] * len(insert) + new_logprobs)[:length]
        # verl's AgentLoopMetrics is a pydantic model, so its fields are attributes. The answer call's
        # time and preemptions add to the first call's, as verl's tool agent loop sums its turns. -1
        # means the first call reported no count.
        output.metrics.generate_sequences += time.perf_counter() - started
        if more.num_preempted is not None:
            output.metrics.num_preempted = max(output.metrics.num_preempted, 0) + more.num_preempted
        # The answer was written with the weights of the answer call, so the trajectory ends at that
        # call's version. min_global_steps stays the first call's, where the trajectory started.
        max_steps = (more.extra_fields or {}).get("max_global_steps")
        if max_steps is not None:
            output.extra_fields["max_global_steps"] = max_steps
        return BudgetSplit(think_end=len(ids), answer_start=len(ids) + len(insert), answer_limit=room,
                           forced=bool(insert))

    def mask_forced_loops(self, output: AgentLoopOutput, split: BudgetSplit, finish: str) -> list[int]:
        """The loop-tail mask of a response whose thinking block was closed by force.

        The thinking was cut by the budget, so a loop in it is masked from its tail mask start up to
        the inserted ids, and the answer after them stays in the loss. The answer is masked only when
        it loops itself and the length limit cut it, as for any other response.
        """
        mask = list(output.response_mask)
        thinking = output.response_ids[:split.think_end]
        looped, start = detect_loop(thinking)
        if looped:
            mask = mask_loop_tail(mask, tail_mask_start(thinking, start), end=split.think_end)
        if finish == "length":
            answer = output.response_ids[split.answer_start:]
            looped, start = detect_loop(answer)
            if looped:
                begin = tail_mask_start(answer, start)
                if begin >= 0:
                    mask = mask_loop_tail(mask, split.answer_start + begin)
        return mask

    def model_path(self) -> str | None:
        """The prepared model dir, or None when the config has none (as in the tests)."""
        model = getattr(getattr(self.config, "actor_rollout_ref", None), "model", None)
        return getattr(model, "path", None)

    def max_new_tokens(self, params: dict[str, Any], prompt_len: int) -> int:
        """The number of new tokens after which this response counts as cut by the length limit.

        verl's vLLM server lets a request generate ``max_tokens`` (or ``max_new_tokens``) tokens when
        the sampling params name one, and ``rollout.response_length`` tokens otherwise. It also keeps
        the prompt and the response within ``max_model_len``, which is the model's
        ``max_position_embeddings`` when ``rollout.max_model_len`` is null. The single-turn loop then
        cuts every response to ``response_length`` tokens, so that is a limit as well.
        """
        limit = int(self.response_length)
        for key in ("max_tokens", "max_new_tokens"):
            if params.get(key) is not None:
                limit = min(limit, int(params[key]))
        # getattr with a default works for both a DictConfig and verl's RolloutConfig dataclass.
        max_model_len = getattr(self.rollout_config, "max_model_len", None) or context_length(self.model_path())
        if max_model_len:
            limit = min(limit, int(max_model_len) - prompt_len)
        return max(limit, 1)

    def response_info(self, output: AgentLoopOutput, params: dict[str, Any],
                      split: BudgetSplit | None = None) -> dict[str, Any]:
        """How this response ended, where its thinking block ends, whether it loops and from where,
        and whether its thinking was closed by force, as the seven ``extra_fields`` entries that the
        reward and the loop-tail mask read.

        A response that went on after the thinking budget ends where its second call ended, so its
        finish is judged on the ids of that call against that call's limit. A forced response is
        checked for loops in its thinking and in its answer separately: the detector reads the last
        1,024 ids, and an answer of a few hundred ids would hide a loop in the thinking before it.
        """
        end_ids = end_token_ids(self.model_path(), self.tokenizer)
        if split is not None:
            finish = classify_finish(output.response_ids[split.answer_start:], stopped_at_string=False,
                                     max_new_tokens=split.answer_limit, end_token_ids=end_ids)
        else:
            finish = classify_finish(
                output.response_ids,
                stopped_at_string=False,
                max_new_tokens=self.max_new_tokens(params, len(output.prompt_ids)),
                end_token_ids=end_ids,
            )
        think, answer = "none", None
        markers = think_token_ids(self.tokenizer)
        if markers is not None:
            open_id, close_id = markers
            starts_inside = prompt_opens_think(self.tokenizer, output.prompt_ids, open_id)
            think, start = split_think(output.response_ids, open_id, close_id, starts_inside)
            if think == "closed":
                # The reward manager decodes the whole response with skip_special_tokens=True. The
                # answer is decoded the same way, so it holds the text that follows the thinking block.
                answer = self.tokenizer.decode(output.response_ids[start:], skip_special_tokens=True)
        # The detector runs on the ids that verl trains on. They include the final end token, as the
        # generated ids that the evaluation measures do.
        forced = split is not None and split.forced
        if forced:
            looped, loop_start = detect_loop(output.response_ids[:split.think_end])
            if not looped:
                answer_looped, answer_start = detect_loop(output.response_ids[split.answer_start:])
                looped, loop_start = answer_looped, (split.answer_start + answer_start if answer_looped else -1)
        else:
            looped, loop_start = detect_loop(output.response_ids)
        return {
            FINISH_KEY: finish,
            THINK_KEY: think,
            ANSWER_KEY: answer,
            LOOP_KEY: looped,
            LOOP_START_KEY: loop_start,
            FORCED_KEY: forced,
            THINK_END_KEY: split.think_end if forced else -1,
        }
