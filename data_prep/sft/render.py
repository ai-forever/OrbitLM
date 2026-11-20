"""Render a chat and a response-only loss mask.

enable_thinking is passed to the tokenizer's chat template. The Qwen3.5-style
templates of the SFT tokenizers put the mode into the assistant prompt: the
prompt already contains <think>\\n, and in no_think it also contains the tail
\\n\\n</think>\\n\\n, which stays outside {% generation %} so the loss ignores it.

Without a rewrite table (a version without system_turns), a dataset system
that states the task is copied onto the first user turn, and a system that
only introduces the model (for example Llama) is dropped. With a rewrite
table (v5, where an empty table means no rewrites), a leading system stays
its own turn: the table replaces that exact string, and anything absent from
the table stays verbatim. The system turn is not a training target.

Loss is 1 on the assistant body, its closing <|im_end|> and the newline the
template writes after it. The role header and user turns stay 0. The dialog
then ends with <|endoftext|>, mask 0.
That token is the document-masking boundary. The supervised stop is <|im_end|>.
The loss mask comes from the template's {% generation %} spans, so a
tokenizer whose template has none is refused.
"""
import re

import numpy as np

END_OF_TEXT = "<|endoftext|>"
# The tokenizer's eos. Pad is the same id, so a mask built from token values
# would also drop a supervised <|im_end|>; the masks are built from positions.
EOS_ID = 151643
IM_END_ID = 151645

_THINK_SWITCH = re.compile(r"\s*/(?:no_)?think\s*$")
# Model self-introduction, not a task. The patterns are narrow on purpose: a
# persona prompt that also states a task stays, because the rest of its text
# is the task. This only matches "You are a helpful AI assistant named ..."
# and "I am Llama" in English or Russian.
_IDENTITY_SYSTEM = re.compile(
    r"you are a helpful ai assistant named\b"
    r"|\bi am llama\b"
    r"|я\s*[—–-]?\s*llama\b",
    re.IGNORECASE,
)
_ROLE_MAP = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "system": "system",
}


def strip_think_switch(text):
    """Remove a trailing /think or /no_think marker from a user turn.

    The mode is chosen by enable_thinking in the assistant prompt of the chat
    template, so a marker on the user text is removed.
    """
    return _THINK_SWITCH.sub("", text or "")


def unclosed_think(text):
    """True when an assistant turn opens <think> and never closes it.

    A closing tag that sits before the opening tag does not close it. That is
    the shape of the four OpenThoughts3 no_think rows whose second trace was
    left in the answer.
    """
    text = text or ""
    open_at = text.rfind("<think>")
    if open_at < 0:
        return False
    return "</think>" not in text[open_at:]


def _longest_true_run(eq):
    if eq.size == 0 or not bool(eq.any()):
        return []
    padded = np.empty(eq.size + 2, dtype=np.int8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = eq
    diffs = np.diff(padded)
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _phrase_repeat_limit(phrase, word_count):
    """How many back-to-back copies make this phrase a stuck loop."""
    letters = {char for char in phrase if char.isalpha()}
    if len(letters) < 2:
        return None
    if word_count == 1:
        if len(phrase) >= 3:
            return 10
        return None
    length = len(phrase)
    if length >= 80:
        return 4
    if length >= 40:
        return 5
    if length >= 20:
        return 6
    return None


def _word_loop(text):
    """True when the same words are repeated back to back."""
    words = text.split()
    count = len(words)
    if count < 4:
        return False
    hashes = np.fromiter((hash(word) for word in words), dtype=np.int64, count=count)
    max_phrase = min(64, count // 4)
    for phrase_len in range(1, max_phrase + 1):
        if count < phrase_len * 4:
            continue
        for start, end in _longest_true_run(hashes[:-phrase_len] == hashes[phrase_len:]):
            copies = (end - start) // phrase_len + 1
            if copies < 4:
                continue
            if words[start:end] != words[start + phrase_len:end + phrase_len]:
                continue
            phrase = " ".join(words[start:start + phrase_len])
            limit = _phrase_repeat_limit(phrase, phrase_len)
            if limit is not None and copies >= limit:
                return True
    return False


def _token_char_loop(token):
    """True when one long token is a short mixed string copied several times."""
    if len(token) < 48:
        return False
    chars = np.frombuffer(token.encode("utf-32-le"), dtype=np.uint32)
    size = int(chars.size)
    max_period = min(size // 6, 200)
    for period in range(8, max_period + 1):
        for start, end in _longest_true_run(chars[:-period] == chars[period:]):
            if end - start < period * 5:
                continue
            unit = token[start:start + period]
            letters = {ch for ch in unit if ch.isalnum()}
            if len(letters) >= 2:
                return True
    return False


def _char_flood(text):
    """True when one non-space character dominates a long turn."""
    if len(text) < 500:
        return False
    counts = {}
    for char in text:
        if char.isspace():
            continue
        counts[char] = counts.get(char, 0) + 1
    if not counts:
        return False
    return max(counts.values()) / len(text) >= 0.5


def is_degenerate_text(text):
    """True when a turn is a stuck loop or a single-character flood.

    A loop is the same phrase copied back to back. The phrase has to contain
    two different letters. One such word of 3 or more characters has to be
    copied 10 times; a phrase of 20 characters, 6 times; of 40, 5 times; of
    80, 4 times. A separator such as ----, a repeated ASCII-art row, a short
    code line shown a few times, and ordinary restatement are not loops. A
    turn of 500 or more characters that is at least half one non-space
    character is a flood.
    """
    text = text or ""
    if len(text) < 30:
        return False
    if _word_loop(text) or _char_flood(text):
        return True
    return any(_token_char_loop(token) for token in text.split() if len(token) >= 48)


def is_echo(user, assistant):
    """True when the assistant turn repeats the user turn verbatim."""
    user_text = " ".join((user or "").split())
    assistant_text = " ".join((assistant or "").split())
    if len(assistant_text) < 40:
        return False
    return user_text.casefold() == assistant_text.casefold()


def message_text(message):
    content = message.get("content")
    if content is None:
        content = message.get("value")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def is_identity_system(text):
    """True when a system turn introduces the model instead of stating a task."""
    return bool(_IDENTITY_SYSTEM.search(text or ""))


def _finish_dialog(turns, system_text, quality_filter=True):
    """Check the user/assistant body and optionally put one system turn in front."""
    if len(turns) < 2 or len(turns) % 2:
        return None
    for index, turn in enumerate(turns):
        expected = "user" if index % 2 == 0 else "assistant"
        if turn["role"] != expected:
            return None
        if quality_filter and is_degenerate_text(turn["content"]):
            return None
        if quality_filter and turn["role"] == "assistant" and is_echo(turns[index - 1]["content"], turn["content"]):
            return None
    if not system_text:
        return turns
    if quality_filter and is_degenerate_text(system_text):
        return None
    return [{"role": "system", "content": system_text}, *turns]


def prepare_messages(messages, rewrite=None, quality_filter=True):
    """Return a dialog, or None when it is unusable.

    rewrite is None for a version without system_turns. A system that states
    the task is then copied onto the first user turn, and a system that only
    introduces the model is dropped.

    rewrite is a dict for v5. Leading system turns stay a system turn. A
    string that is a key of the dict is replaced by its value; any other
    string stays verbatim. Several leading systems are joined. A replacement
    that is empty drops that system turn. A system turn after the dialog has
    started drops the whole dialog.

    A tool turn, an unknown role, an empty turn, an unclosed <think>, or a
    dialog that does not start with user and end with assistant drops the
    whole dialog. quality_filter also drops a degenerate turn (a stuck loop
    or a one-character flood) and an assistant turn that repeats the user.
    """
    if not messages:
        return None
    task_systems = []
    kept_systems = []
    started = False
    turns = []
    for message in messages:
        role = _ROLE_MAP.get((message.get("role") or message.get("from") or "").strip().lower())
        if role is None:
            return None
        if role == "system":
            if rewrite is not None and started:
                return None
            text = message_text(message).strip()
            if not text:
                continue
            if rewrite is None:
                if not is_identity_system(text):
                    task_systems.append(text)
                continue
            text = str(rewrite.get(text, text)).strip()
            if text:
                kept_systems.append(text)
            continue
        started = True
        text = message_text(message)
        if role == "user":
            text = strip_think_switch(text)
        elif unclosed_think(text):
            return None
        if not text.strip():
            return None
        turns.append({"role": role, "content": text})
    if rewrite is None and task_systems and turns:
        turns[0]["content"] = "\n\n".join(task_systems + [turns[0]["content"]])
    system_text = "\n\n".join(kept_systems) if rewrite is not None else ""
    return _finish_dialog(turns, system_text, quality_filter=quality_filter)


def prepare_tool_record(record, rewrite=None, quality_filter=True):
    """Return (turns, tools) for one stored tool dialog, or None.

    The stored shape is one tools list plus messages whose assistant calls
    live in tool_calls and whose chain lives in reasoning_content. An
    assistant turn may have empty content when it only calls a tool. A tool
    result is not checked for loops: those strings are masked out and are
    often JSON. Leading system turns are joined into one turn. rewrite
    replaces a listed system string the same way prepare_messages does.
    """
    if not isinstance(record, dict):
        return None
    messages = record.get("messages")
    tools = record.get("tools")
    if not isinstance(messages, list) or not messages:
        return None
    if not isinstance(tools, list) or not tools:
        return None
    leading = []
    body = []
    started = False
    last_user = ""
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = (message.get("role") or "").strip().lower()
        content = message_text(message)
        if role == "system":
            if started:
                return None
            text = content.strip()
            if not text:
                continue
            if rewrite is not None:
                text = str(rewrite.get(text, text)).strip()
            if not text:
                continue
            leading.append(text)
            continue
        started = True
        if role == "user":
            text = strip_think_switch(content)
            if not text.strip():
                return None
            if quality_filter and is_degenerate_text(text):
                return None
            last_user = text
            body.append({"role": "user", "content": text})
            continue
        if role == "assistant":
            reasoning = message.get("reasoning_content") or ""
            if not isinstance(reasoning, str):
                reasoning = ""
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                return None
            if unclosed_think(content) or unclosed_think(reasoning):
                return None
            if not content.strip() and not calls and not reasoning.strip():
                return None
            if quality_filter and is_degenerate_text(content):
                return None
            if quality_filter and reasoning and is_degenerate_text(reasoning):
                return None
            if quality_filter and content.strip() and is_echo(last_user, content):
                return None
            turn = {"role": "assistant", "content": content}
            if reasoning.strip():
                turn["reasoning_content"] = reasoning.strip()
            if calls:
                turn["tool_calls"] = calls
            body.append(turn)
            continue
        if role == "tool":
            body.append({"role": "tool", "content": content})
            continue
        return None
    if not any(turn["role"] == "user" for turn in body):
        return None
    if not any(turn["role"] == "assistant" for turn in body):
        return None
    turns = []
    if leading:
        system_text = "\n\n".join(leading)
        if quality_filter and is_degenerate_text(system_text):
            return None
        turns.append({"role": "system", "content": system_text})
    turns.extend(body)
    return turns, tools


def example_enable_thinking(mode, turns):
    """Resolve a stream's enable_thinking flag for one prepared dialog.

    True and false apply to the whole stream. "auto" is true when an assistant
    turn already contains a closed think span or a reasoning_content chain,
    which is how mixed sources such as T-Wix are split without wrapping a
    traced answer in a second empty think.
    """
    if mode == "auto":
        return any(
            turn["role"] == "assistant" and (
                ("<think>" in (turn.get("content") or "") and "</think>" in (turn.get("content") or ""))
                or bool((turn.get("reasoning_content") or "").strip())
            )
            for turn in turns
        )
    return bool(mode)


def _chat_template_supervises(tokenizer):
    return "{% generation %}" in (tokenizer.chat_template or "")


def _render_same_mode(batch, tokenizer, enable_thinking):
    encoded = tokenizer.apply_chat_template(
        batch,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=enable_thinking,
    )
    sequences = encoded["input_ids"]
    masks = encoded["assistant_masks"]
    eos_id = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    if eos_id != EOS_ID:
        raise RuntimeError(f"{END_OF_TEXT} id is {eos_id}, expected {EOS_ID}")
    rendered = []
    for ids, mask in zip(sequences, masks):
        ids = [int(token) for token in ids]
        mask = [int(value) for value in mask]
        ids.append(eos_id)
        mask.append(0)
        if len(ids) != len(mask):
            raise RuntimeError(f"mask length {len(mask)} != token length {len(ids)}")
        rendered.append((ids, mask))
    return rendered


def render_batch(batch, tokenizer, enable_thinking=False):
    """Tokenize prepared dialogs. enable_thinking is one flag or one per dialog."""
    if not batch:
        return []
    if not _chat_template_supervises(tokenizer):
        raise RuntimeError("the tokenizer's chat template has no {% generation %} span to build the loss mask from")
    if isinstance(enable_thinking, (list, tuple)):
        flags = list(enable_thinking)
        if len(flags) != len(batch):
            raise RuntimeError(f"{len(flags)} thinking flags for {len(batch)} dialogs")
    else:
        flags = [bool(enable_thinking)] * len(batch)
    order = []
    grouped = {True: [], False: []}
    for index, (turns, flag) in enumerate(zip(batch, flags)):
        flag = bool(flag)
        grouped[flag].append(turns)
        order.append((flag, len(grouped[flag]) - 1))
    rendered = {}
    for flag, turns in grouped.items():
        if turns:
            rendered[flag] = _render_same_mode(turns, tokenizer, flag)
    return [rendered[flag][index] for flag, index in order]


def render_turns(turns, tokenizer, enable_thinking=False):
    """Tokenize one prepared dialog. Returns (token ids, loss mask)."""
    return render_batch([turns], tokenizer, enable_thinking=enable_thinking)[0]


def mask_all_but_last_assistant(mask):
    """Loss on the last assistant span only. Earlier assistant spans become context.

    A span is a run of mask 1. The chat template opens one such run per
    assistant turn. ToolMind trains the last turn and leaves the shared
    prefix, including earlier assistant replies, at mask 0.
    """
    spans = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    if len(spans) <= 1:
        return list(mask)
    kept_start, kept_end = spans[-1]
    return [value if kept_start <= index < kept_end else 0 for index, value in enumerate(mask)]


def render_tool_dialog(turns, tools, tokenizer, enable_thinking=False):
    """Tokenize one tool dialog with its own tool list.

    Dialogs in one batch do not share a tool list, so this is one call of
    apply_chat_template. The end-of-text token is appended with mask 0, the
    same way render_turns closes a plain dialog.
    """
    encoded = tokenizer.apply_chat_template(
        turns,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=bool(enable_thinking),
    )
    ids = encoded["input_ids"]
    mask = encoded["assistant_masks"]
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids, mask = ids[0], mask[0]
    ids = [int(token) for token in ids]
    mask = [int(value) for value in mask]
    eos_id = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    if eos_id != EOS_ID:
        raise RuntimeError(f"{END_OF_TEXT} id is {eos_id}, expected {EOS_ID}")
    ids.append(eos_id)
    mask.append(0)
    if len(ids) != len(mask):
        raise RuntimeError(f"mask length {len(mask)} != token length {len(ids)}")
    return ids, mask
