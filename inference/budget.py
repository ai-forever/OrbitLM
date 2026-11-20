"""Generation in one or two calls, with an optional thinking budget (budget forcing).

Both generate_vllm.py and generate_hf.py use this module, so the two engines follow the same rules.
An engine is a GenerateFn: it takes prompts as token ids with a token limit for each, and returns one
Completion per prompt.

Without a budget, every prompt gets one call of up to --max-new-tokens tokens.

With a budget (think mode only), the first call generates at most think_tokens tokens. Then:

- a response that ended on its own keeps its answer;
- a response that the budget cut while its thinking block was still open gets Qwen3's stop-thinking
  sentence and "</think>" appended, and a second call writes the answer in at most answer_tokens tokens;
- a response that had closed its block but was cut while answering gets a second call of at most
  answer_tokens tokens that continues the answer, with nothing inserted.

The stop-thinking sentences, English and Russian, are the ones the distillation rollouts insert at
their thinking budget (src/orbit_verl/data/response_info.py). A Russian prompt gets the Russian
sentence, so that the answer after it stays in Russian.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from orbit_verl.data.response_info import (
    STOP_THINKING_TEXTS,
    forced_close_ids,
    split_think,
    stop_thinking_text,
    think_token_ids,
)

# The default thinking budget and answer budget add up to 16K tokens. The distillation stage trains with
# the same 14,336-token thinking budget, and 2,048 tokens leave the answer enough room.
DEFAULT_THINK_BUDGET = 14336
DEFAULT_ANSWER_BUDGET = 2048


@dataclass(frozen=True)
class Completion:
    """What one engine call returned for one prompt.

    finish_reason is "stop" when the model ended its turn, and "length" when the token limit cut it.
    """

    token_ids: tuple[int, ...]
    finish_reason: str


GenerateFn = Callable[[list[list[int]], list[int]], list[Completion]]


@dataclass(frozen=True)
class Budget:
    """The thinking budget, the answer budget, and the token ids that close a thinking block by force."""

    think_tokens: int
    answer_tokens: int
    close_id: int
    stop_ids_by_text: dict[str, tuple[int, ...]]

    def stop_ids(self, lang: Any) -> tuple[int, ...]:
        """The ids of the stop-thinking sentence for a prompt of language lang (Russian for "ru")."""
        return self.stop_ids_by_text[stop_thinking_text(lang)]


def make_budget(tokenizer: Any, think_tokens: int, answer_tokens: int) -> Budget:
    """A Budget for this tokenizer. Every stop-thinking sentence is encoded and checked once, up front."""
    markers = think_token_ids(tokenizer)
    if markers is None:
        raise ValueError("this tokenizer has no single-token <think> and </think>, so it cannot use budget forcing")
    ids_by_text = {text: forced_close_ids(tokenizer, text) for text in STOP_THINKING_TEXTS.values()}
    return Budget(think_tokens=think_tokens, answer_tokens=answer_tokens, close_id=markers[1],
                  stop_ids_by_text=ids_by_text)


@dataclass
class Response:
    """The whole response to one prompt: the first call, the ids inserted at the budget, and the second call."""

    first: Completion
    inserted_ids: tuple[int, ...] = ()
    second: Optional[Completion] = None

    @property
    def forced_close(self) -> bool:
        """True when the thinking block was closed by force at the budget."""
        return bool(self.inserted_ids)

    @property
    def continued(self) -> bool:
        """True when a closed block's answer, cut at the budget, was continued by a second call."""
        return self.second is not None and not self.inserted_ids

    @property
    def token_ids(self) -> list[int]:
        """Every id after the prompt: the model's own ids and the inserted ones, in order."""
        second = self.second.token_ids if self.second is not None else ()
        return [*self.first.token_ids, *self.inserted_ids, *second]

    @property
    def generated_tokens(self) -> int:
        """How many tokens the model generated, without the inserted ones."""
        return len(self.first.token_ids) + (len(self.second.token_ids) if self.second is not None else 0)

    @property
    def finish_reason(self) -> str:
        """How the last call ended."""
        return (self.second or self.first).finish_reason


@dataclass(frozen=True)
class AnswerCall:
    """The second call of one response: its prompt ids, the ids inserted before it, and its token limit."""

    prompt_ids: list[int]
    inserted_ids: tuple[int, ...]
    max_tokens: int


def plan_answer_call(prompt_ids: Sequence[int], first: Completion, budget: Budget, lang: Any,
                     max_model_len: int) -> Optional[AnswerCall]:
    """The answer call after the thinking budget, or None when the response needs none.

    Only a response that the token limit cut needs one. When it is still inside its thinking block, the
    stop-thinking ids are inserted first. The answer call gets at most answer_tokens tokens and never
    runs past the model's context. When no room is left at all, there is no answer call.
    """
    if first.finish_reason != "length":
        return None
    inserted = () if budget.close_id in first.token_ids else budget.stop_ids(lang)
    prompt = [*prompt_ids, *first.token_ids, *inserted]
    room = min(budget.answer_tokens, max_model_len - len(prompt))
    if room <= 0:
        return None
    return AnswerCall(prompt_ids=prompt, inserted_ids=tuple(inserted), max_tokens=room)


def generate_responses(prompts: Sequence[Sequence[int]], langs: Sequence[Any], generate: GenerateFn, *,
                       max_new_tokens: Optional[int], max_model_len: int,
                       budget: Optional[Budget] = None) -> list[Response]:
    """Generate the responses to all prompts, in at most two engine calls.

    Every call goes to the engine as one batch, so vLLM can schedule all prompts together. The first
    call's limit is the thinking budget when there is one, else max_new_tokens, and in both cases at
    most what is left of the model's context.
    """
    first_limit = budget.think_tokens if budget is not None else max_new_tokens
    if first_limit is None:
        raise ValueError("max_new_tokens is needed when there is no budget")
    limits = []
    for index, prompt in enumerate(prompts):
        room = max_model_len - len(prompt)
        if room <= 0:
            raise ValueError(f"prompt {index} has {len(prompt)} tokens, which fills the context of {max_model_len}")
        limits.append(min(first_limit, room))
    responses = [Response(first=completion) for completion in generate([list(p) for p in prompts], limits)]
    if budget is None:
        return responses
    calls = {}
    for index, (prompt, response, lang) in enumerate(zip(prompts, responses, langs)):
        call = plan_answer_call(prompt, response.first, budget, lang, max_model_len)
        if call is not None:
            calls[index] = call
    if calls:
        answers = generate([call.prompt_ids for call in calls.values()], [call.max_tokens for call in calls.values()])
        for (index, call), answer in zip(calls.items(), answers):
            responses[index].inserted_ids = call.inserted_ids
            responses[index].second = answer
    return responses


@dataclass
class ResponseParts:
    """A response split for the output: thinking status, thinking ids and answer ids."""

    think: str
    thinking_ids: list[int] = field(default_factory=list)
    answer_ids: list[int] = field(default_factory=list)


def split_response(response: Response, mode: str, think_ids: tuple[int, int],
                   end_ids: Sequence[int]) -> ResponseParts:
    """Split a response into thinking and answer, without the end tokens and the inserted ids.

    think is "closed", "unclosed" or "none", as in the RL rollouts (response_info.split_think). In think
    mode the response starts inside the thinking block. In no_think mode the empty block is part of the
    prompt, and raw prompts have no block, so in both the whole response is the answer.
    """
    ends = set(end_ids)

    def strip_end(ids: Sequence[int]) -> list[int]:
        ids = list(ids)
        while ids and ids[-1] in ends:
            ids.pop()
        return ids

    if mode != "think":
        return ResponseParts(think="none", answer_ids=strip_end(response.token_ids))
    if response.forced_close:
        second = response.second.token_ids if response.second is not None else ()
        return ResponseParts(think="closed", thinking_ids=list(response.first.token_ids), answer_ids=strip_end(second))
    ids = response.token_ids
    status, answer_start = split_think(ids, think_ids[0], think_ids[1], starts_inside=True)
    if status == "closed":
        return ResponseParts(think="closed", thinking_ids=ids[:answer_start - 1], answer_ids=strip_end(ids[answer_start:]))
    return ResponseParts(think=status, thinking_ids=strip_end(ids))
