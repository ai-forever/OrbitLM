"""Prompt files, prompt language, and rendering in the SFT evaluation's format."""
from __future__ import annotations

import json

import pytest

from inference import REPO_ROOT
from inference.prompts import (
    PromptError,
    PromptItem,
    detect_language,
    end_token_ids,
    load_prompts,
    prompt_language,
    render_ids,
)


@pytest.mark.parametrize("text, lang", [
    ("Найдите сумму всех действительных корней уравнения $x^2 - 7x + 10 = 0$.", "ru"),
    ("Сколько будет 2+2?", "ru"),
    ("Решите: \\(\\frac{a}{b} + \\sqrt{x}\\), ответ запишите в \\boxed{}.", "ru"),
    ("Find the sum of all real solutions of $x^2 - 7x + 10 = 0$.", "en"),
    ("What does the Russian word «мир» mean?", "en"),
    ("", "en"),
    ("$1+1$", "en"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_language_precedence():
    russian = PromptItem(id="0", messages=({"role": "user", "content": "Сколько будет 2+2?"},))
    assert prompt_language(russian) == "ru"
    assert prompt_language(russian, "en") == "en"
    tagged = PromptItem(id="1", messages=russian.messages, lang="en")
    assert prompt_language(tagged) == "en"
    assert prompt_language(tagged, "ru") == "ru"


def test_the_smoke_prompts_are_two_english_and_two_russian():
    items = load_prompts(REPO_ROOT / "inference" / "examples" / "math_smoke.jsonl")
    assert [prompt_language(item) for item in items] == ["en", "en", "ru", "ru"]


def test_load_jsonl(tmp_path):
    path = tmp_path / "p.jsonl"
    rows = [
        {"prompt": "hello"},
        {"id": "chat", "lang": "RU", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "привет"}]},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n\n", encoding="utf-8")
    first, second = load_prompts(path)
    assert (first.id, first.user_text, first.lang) == ("0", "hello", None)
    assert (second.id, second.user_text, second.lang, len(second.messages)) == ("chat", "привет", "ru", 2)


def test_load_txt_and_command_line(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("first\n\n  \nsecond\n", encoding="utf-8")
    assert [item.user_text for item in load_prompts(path)] == ["first", "second"]
    assert [(item.id, item.user_text) for item in load_prompts(texts=["a", "b"])] == [("0", "a"), ("1", "b")]


@pytest.mark.parametrize("name, content, message", [
    ("p.jsonl", "{not json\n", "not valid JSON"),
    ("p.jsonl", '["a list"]\n', "JSON object"),
    ("p.jsonl", '{"text": "no prompt field"}\n', "string 'prompt'"),
    ("p.jsonl", '{"prompt": "x", "lang": "de"}\n', "lang must be"),
    ("p.jsonl", '{"messages": [{"role": "user"}]}\n', "role and a content"),
    ("p.jsonl", "\n", "no prompts"),
    ("p.csv", "prompt\nx\n", ".jsonl or .txt"),
])
def test_bad_prompt_files(tmp_path, name, content, message):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    with pytest.raises(PromptError, match=message):
        load_prompts(path)


def test_render_think_and_no_think(tokenizer, special):
    item = PromptItem(id="0", messages=({"role": "user", "content": "2+2?"},))
    think = render_ids(tokenizer, item, "think", lead_eos=True)
    assert think[:2] == [special["<|endoftext|>"], special["<|im_start|>"]]
    assert tokenizer.decode(think).endswith("<|im_start|>assistant\n<think>\n")
    no_think = render_ids(tokenizer, item, "no_think", lead_eos=False)
    assert no_think[0] == special["<|im_start|>"]
    assert tokenizer.decode(no_think).endswith("<think>\n\n</think>\n\n")


def test_render_raw_sends_the_text_as_it_is(tokenizer):
    item = PromptItem(id="0", messages=({"role": "user", "content": "The capital of France is"},))
    assert tokenizer.decode(render_ids(tokenizer, item, "raw", lead_eos=True)) == "The capital of France is"
    chat = PromptItem(id="1", messages=({"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                        {"role": "user", "content": "c"}))
    with pytest.raises(PromptError, match="raw mode"):
        render_ids(tokenizer, chat, "raw", lead_eos=True)


def test_a_template_without_the_think_prefill_is_refused(make_tokenizer):
    item = PromptItem(id="7", messages=({"role": "user", "content": "2+2?"},))
    with pytest.raises(PromptError, match="prompt 7 breaks the SFT chat format"):
        render_ids(make_tokenizer(think_prefill=False), item, "think", lead_eos=True)


def test_end_token_ids(tokenizer, special):
    assert end_token_ids(tokenizer) == (special["<|im_end|>"], special["<|endoftext|>"])
