"""Shared helpers of the inference tests: a stand-in for the Qwen3 tokenizer and a scripted engine.

The tests run on CPU without a GPU, vLLM or a tokenizer download. Run them from the repo root:

    python -m pytest -q tests/inference -p no:cacheprovider
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The Qwen3 ids of the special tokens that the chat format and budget forcing use.
SPECIAL = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645, "<think>": 151667, "</think>": 151668}
SPECIAL_BY_ID = {value: key for key, value in SPECIAL.items()}
SPECIAL_PATTERN = re.compile("|".join(re.escape(token) for token in SPECIAL))


class FakeTokenizer:
    """A character-level stand-in for the Qwen3 tokenizer.

    Special tokens keep their Qwen3 ids, and every other character is its own code point, so encoding
    and decoding are exact. The chat template writes the Qwen3 format, and think_prefill=False makes a
    template without the thinking prefill, which the SFT contract must refuse.
    """

    unk_token_id = None
    eos_token_id = SPECIAL["<|im_end|>"]
    pad_token_id = SPECIAL["<|endoftext|>"]
    name_or_path = "fake-qwen3"

    def __init__(self, think_prefill: bool = True):
        self.think_prefill = think_prefill

    def encode(self, text, add_special_tokens=False):
        ids, position = [], 0
        for match in SPECIAL_PATTERN.finditer(text):
            ids += [ord(char) for char in text[position:match.start()]]
            ids.append(SPECIAL[match.group()])
            position = match.end()
        return ids + [ord(char) for char in text[position:]]

    def decode(self, ids, skip_special_tokens=False):
        pieces = []
        for token_id in ids:
            if token_id in SPECIAL_BY_ID:
                pieces.append("" if skip_special_tokens else SPECIAL_BY_ID[token_id])
            else:
                pieces.append(chr(token_id))
        return "".join(pieces)

    def convert_tokens_to_ids(self, token):
        return SPECIAL.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return SPECIAL_BY_ID.get(token_id, chr(token_id))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                            chat_template=None, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if self.think_prefill:
                text += "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        return text


class ScriptedEngine:
    """A GenerateFn that answers every call from a script and records the calls it got.

    script(prompt_ids, limit, call_index) returns the Completion of one prompt.
    """

    def __init__(self, script):
        self.script = script
        self.calls = []

    def __call__(self, prompt_ids, limits):
        index = len(self.calls)
        self.calls.append(([list(p) for p in prompt_ids], list(limits)))
        return [self.script(list(p), limit, index) for p, limit in zip(prompt_ids, limits)]


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def make_tokenizer():
    return FakeTokenizer


@pytest.fixture
def scripted_engine():
    return ScriptedEngine


@pytest.fixture
def special():
    return dict(SPECIAL)
