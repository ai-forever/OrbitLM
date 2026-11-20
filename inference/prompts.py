"""Prompts for generation: reading them, finding their language, and rendering them like the SFT evaluation.

A prompt file is either .jsonl or .txt.

- .jsonl: one JSON object per line, with "prompt" (a string, sent as one user turn) or "messages"
  (a list of {"role", "content"} chat messages). Optional fields: "id" and "lang" ("en" or "ru").
- .txt: one prompt per non-empty line.

Chat prompts are rendered by eval/in_training/itr_render.py, the same code the SFT evaluation uses, and
checked against its token contract. Raw prompts go to the model as plain text, for pretraining and
context-extension exports that have no chat format.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

from inference import REPO_ROOT

import itr_render  # noqa: E402  (on sys.path through the inference package)

MODES = ("think", "no_think", "raw")
LANGUAGES = ("en", "ru")
# Generation stops at the end of the assistant turn and at the end of a document. The SFT evaluation and
# the RL rollouts stop at the same two tokens.
END_TOKENS = ("<|im_end|>", "<|endoftext|>")


class PromptError(ValueError):
    """A prompt file or a prompt cannot be used."""


@dataclass(frozen=True)
class PromptItem:
    """One prompt: its id, its chat messages, and the language given in the file (None when not given)."""

    id: str
    messages: tuple[dict[str, Any], ...]
    lang: Optional[str] = None

    @property
    def user_text(self) -> str:
        """The text of the last user message, which is what the language is detected from."""
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""


def load_prompts(path: Optional[Path] = None, texts: Optional[Sequence[str]] = None) -> list[PromptItem]:
    """The prompts of a .jsonl or .txt file, or of the --prompt strings."""
    if texts:
        items = [_item_from_text(str(index), text) for index, text in enumerate(texts)]
    elif path is None:
        raise PromptError("no prompts given")
    elif path.suffix == ".jsonl":
        items = _read_jsonl(path)
    elif path.suffix == ".txt":
        lines = [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines()]
        items = [_item_from_text(str(index), line) for index, line in enumerate(line for line in lines if line.strip())]
    else:
        raise PromptError(f"{path}: the prompts file must be .jsonl or .txt")
    if not items:
        raise PromptError(f"{path}: no prompts in the file")
    return items


def _item_from_text(item_id: str, text: str, lang: Optional[str] = None) -> PromptItem:
    return PromptItem(id=item_id, messages=({"role": "user", "content": text},), lang=lang)


def _read_jsonl(path: Path) -> list[PromptItem]:
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}, line {number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PromptError(f"{where}: not valid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise PromptError(f"{where}: every line must be a JSON object")
        item_id = str(row.get("id", len(items)))
        lang = row.get("lang")
        if lang is not None and str(lang).lower() not in LANGUAGES:
            raise PromptError(f"{where}: lang must be one of {list(LANGUAGES)}, got {lang!r}")
        lang = None if lang is None else str(lang).lower()
        if isinstance(row.get("prompt"), str):
            items.append(_item_from_text(item_id, row["prompt"], lang))
        elif isinstance(row.get("messages"), list) and row["messages"]:
            messages = tuple(dict(message) for message in row["messages"])
            if any("role" not in message or "content" not in message for message in messages):
                raise PromptError(f"{where}: every message needs a role and a content")
            items.append(PromptItem(id=item_id, messages=messages, lang=lang))
        else:
            raise PromptError(f"{where}: a line needs a string 'prompt' or a non-empty list 'messages'")
    return items


@lru_cache(maxsize=1)
def _chat_local() -> Any:
    """The SFT evaluation's helper module, loaded by its path, as the evaluation itself loads it."""
    path = REPO_ROOT / "eval" / "in_training" / "tasks" / "chat_local.py"
    spec = importlib.util.spec_from_file_location("itr_chat_local", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_language(text: str) -> str:
    """ "ru" when the prose of a text has at least as many Cyrillic letters as Latin ones, else "en".

    The letters are counted by the SFT evaluation's letter_counts, which leaves out formulas, code and
    links. A Russian math problem holds many Latin letters inside its LaTeX, and they must not make it
    count as English.
    """
    latin, cyrillic = _chat_local().letter_counts(text or "")
    return "ru" if cyrillic and cyrillic >= latin else "en"


def prompt_language(item: PromptItem, override: str = "auto") -> str:
    """The language of a prompt: the --lang value, else the file's "lang" field, else the detected one."""
    if override != "auto":
        return override
    return item.lang or detect_language(item.user_text)


def end_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """The ids of END_TOKENS in this tokenizer. A tokenizer without them cannot end a chat turn."""
    ids = []
    for token in END_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or token_id == getattr(tokenizer, "unk_token_id", None):
            raise PromptError(f"the tokenizer has no {token} token, so it is not a Qwen3 OrbitLM tokenizer")
        ids.append(int(token_id))
    return tuple(ids)


def render_ids(tokenizer: Any, item: PromptItem, mode: str, lead_eos: bool,
               contract: Optional[dict[str, Any]] = None) -> list[int]:
    """The prompt token ids of one item in one mode.

    think and no_think use itr_render.render and then check the ids against the SFT token contract, so
    a tokenizer whose chat template differs from the one the model was trained with is refused here
    instead of producing quietly worse answers. raw sends the text of a single user message as it is.
    """
    if mode == "raw":
        if len(item.messages) != 1:
            raise PromptError(f"prompt {item.id}: raw mode takes a plain prompt, not a conversation")
        return list(tokenizer.encode(item.user_text, add_special_tokens=False))
    text = itr_render.render(tokenizer, list(item.messages), mode, lead_eos)
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    problem = itr_render.contract_problem(ids, mode, lead_eos, contract or itr_render.expected_ids(tokenizer))
    if problem:
        raise PromptError(
            f"prompt {item.id} breaks the SFT chat format: {problem}. A pretraining export has no chat format; "
            "use --mode raw for it."
        )
    return ids
