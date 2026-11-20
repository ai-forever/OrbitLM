"""What the agent loop learns about a finished response, handed to the reward through ``extra_fields``.

The reward function only receives the response as text, decoded with ``skip_special_tokens=True``.
Three facts it needs are lost in that text or are cheaper to find in the ids, so the agent loop
works them out from the token ids and attaches them to the response:

1. Whether generation ended on its own (an end token) or was cut by the length
   limit. verl's vLLM server reports both cases as "completed", so the loop decides from the
   response length and its last token.
2. Where a thinking block ends. Working from the ids does not depend on whether a tokenizer keeps
   ``<think>`` and ``</think>`` in the decoded text, which some tokenizers drop as special tokens.
3. Whether the response is caught in a repetition loop, and where the loop starts. The detector is
   the SFT evaluation's, ported to ``orbit_verl.data.loops``, so training and evaluation count the
   same responses as loops.

4. Whether the thinking block was closed by force at the thinking budget, and where the model's own
   ids end. ``orbit_verl.data.agent_loop`` does the forcing when ``ORBIT_THINK_BUDGET`` is set.

verl's reward manager merges each response's ``extra_fields`` into the ``extra_info`` that
``compute_score`` receives (the naive manager reads them as ``tool_extra_fields``). Every response
carries all seven keys. verl gathers the responses of each agent-loop worker into one batch, then
concatenates the batches of all workers, and that step expects every batch to have the same keys.

This module imports nothing from verl, so the reward workers and the tests can use it cheaply. The
inference package (``inference/``) imports the stop-thinking texts and the think helpers from here
in a python without verl, so their names and behaviour stay stable.
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

# How the response ended: "eos" (an end token, or any other early end) or "length" (cut at the
# maximum number of new tokens). "stop" (cut at a stop string) is defined for callers that use stop
# strings, but the release passes none, so its responses never get it.
FINISH_KEY = "orbit_finish"
# Thinking status: "none" (no thinking block), "closed" (the last block was closed),
# or "unclosed" (the response ends inside a thinking block).
THINK_KEY = "orbit_think"
# The decoded text after the last "</think>" when the status is "closed", else None.
ANSWER_KEY = "orbit_answer"
# True when the response loops by the evaluation's detector: at least 512 ids, and at least half of
# the 16-grams in its last 1,024 ids repeat an earlier one (``orbit_verl.data.loops.is_loop``).
LOOP_KEY = "orbit_loop"
# The index into the response ids where the repeating text starts (``loops.loop_start``), or -1
# when the response does not loop. The loop-tail mask turns the response mask off from here.
LOOP_START_KEY = "orbit_loop_start"
# True when the agent loop closed the thinking block by force at the thinking budget.
FORCED_KEY = "orbit_forced"
# The index into the response ids of the first id the agent loop inserted to close the block, or -1
# when the response was not forced. The ids before it are the model's thinking.
THINK_END_KEY = "orbit_think_end"

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# The ids inserted at the thinking budget: Qwen3's stop-thinking sentence (the thinking budget of the
# Qwen3 report, section 4.3), then the block's close in the release chat format. The sentence is
# Qwen3's own, so a Qwen3 teacher knows it and scores the answer after it as a normal continuation.
STOP_THINKING_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
    "directly now.\n</think>\n\n"
)
# The same sentence in Russian, for rows whose language is Russian. An English sentence inside Russian
# thinking could pull the answer that follows into English, and the release must keep answering
# Russian prompts in Russian.
STOP_THINKING_TEXT_RU = (
    "\n\nУчитывая, что время ограничено пользователем, я должен сразу дать решение на основе уже "
    "проведённых рассуждений.\n</think>\n\n"
)
# The stop-thinking text per row language (extra_info.lang). Any other language gets the English one.
STOP_THINKING_TEXTS = {"en": STOP_THINKING_TEXT, "ru": STOP_THINKING_TEXT_RU}


def stop_thinking_text(lang: Any) -> str:
    """The stop-thinking text for a row of language ``lang``: Russian for "ru", English otherwise."""
    return STOP_THINKING_TEXTS.get(str(lang or "").lower(), STOP_THINKING_TEXT)


def forced_close_ids(tokenizer: Any, text: str = STOP_THINKING_TEXT) -> tuple[int, ...]:
    """The ids of a stop-thinking text, checked to hold exactly one "</think>" and to decode back.

    A tokenizer that splits "</think>" into pieces would leave a forced response looking unclosed to
    ``split_think`` and the reward, so it is refused before any rollout.
    """
    markers = think_token_ids(tokenizer)
    if markers is None:
        raise ValueError("this tokenizer has no single-token <think> and </think>, so a thinking block cannot be closed by force")
    ids = tuple(int(i) for i in tokenizer.encode(text, add_special_tokens=False))
    if ids.count(markers[1]) != 1:
        raise ValueError(f"the stop-thinking text encodes to {ids.count(markers[1])} '</think>' ids instead of one")
    decoded = tokenizer.decode(list(ids), skip_special_tokens=False)
    if decoded != text:
        raise ValueError(f"the stop-thinking ids decode to {decoded!r} instead of {text!r}")
    return ids


def classify_finish(
    response_ids: Sequence[int],
    stopped_at_string: bool,
    max_new_tokens: int,
    end_token_ids: Collection[int],
) -> str:
    """Say how a response ended: "stop", "eos" or "length".

    vLLM ends a request early only for an end token, a stop token, a stop string, an abort or a
    detected repetition. So a response shorter than ``max_new_tokens`` that was not cut at a stop
    string ended on its own, and it counts as "eos". A response that reached ``max_new_tokens``
    counts as "length", unless its last token is an end token: then the model finished exactly at
    the limit.

    ``stopped_at_string`` gives "stop". Every caller in the release passes False, because no prompt
    mode has stop strings. The argument stays, because the inference package calls this function
    with that signature.
    """
    if stopped_at_string:
        return "stop"
    if len(response_ids) >= max_new_tokens and not (response_ids and response_ids[-1] in end_token_ids):
        return "length"
    return "eos"


def think_token_ids(tokenizer: Any) -> tuple[int, int] | None:
    """Ids of ``<think>`` and ``</think>`` when the tokenizer has each as a single token, else None.

    A tokenizer without these tokens maps them to its unknown token or splits them into pieces. In
    that case a response cannot contain a thinking block in the form this module looks for.
    """
    ids = []
    for marker in (THINK_OPEN, THINK_CLOSE):
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if not isinstance(token_id, int) or token_id == getattr(tokenizer, "unk_token_id", None):
            return None
        if tokenizer.convert_ids_to_tokens(token_id) != marker:
            return None
        ids.append(token_id)
    return ids[0], ids[1]


def prompt_opens_think(tokenizer: Any, prompt_ids: Sequence[int], open_id: int) -> bool:
    """True when the prompt ends by opening a thinking block, so the response starts inside it.

    Some chat templates end the generation prompt with "<think>\\n". Trailing tokens that decode to
    whitespace are skipped before the check. A ``<think>`` in the middle of the prompt does not
    count, because only a block that the prompt leaves open continues into the response.
    """
    for token_id in reversed(prompt_ids):
        if token_id == open_id:
            return True
        if tokenizer.decode([token_id], skip_special_tokens=False).strip():
            return False
    return False


def split_think(
    response_ids: Sequence[int], open_id: int, close_id: int, starts_inside: bool
) -> tuple[str, int | None]:
    """Return the thinking status of a response and the index where its answer starts.

    The status is "closed" when the response has a ``</think>`` and no ``<think>`` after the last
    one; the answer then starts right after that ``</think>``. It is "unclosed" when the response
    ends inside a thinking block: a ``<think>`` after the last ``</think>``, or no ``</think>`` at
    all in a response that opened one or started inside one. Otherwise it is "none", and the whole
    response is the answer.
    """
    ids = list(response_ids)
    last_close = max((i for i, t in enumerate(ids) if t == close_id), default=None)
    last_open = max((i for i, t in enumerate(ids) if t == open_id), default=None)
    if last_close is not None:
        if last_open is not None and last_open > last_close:
            return "unclosed", None
        return "closed", last_close + 1
    if last_open is not None or starts_inside:
        return "unclosed", None
    return "none", None
