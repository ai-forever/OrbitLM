"""CPU tests for orbit_verl.data.prompts: the prompt-mode registry, rendering and token ids.

Run with the verl python and PYTHONPATH=<repo>/src. Tests that need the Qwen3 tokenizer read it from
the SFT export named by ORBIT_TEST_SFT_EXPORT (conftest.py) and skip without it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from orbit_verl import common

# The cache variables must be set before transformers is imported anywhere in this process.
common.apply_cache_env()

import yaml  # noqa: E402

from orbit_verl.data import prompts  # noqa: E402

# Every prompt mode of the release. Each one is the SFT chat format written out in a YAML template.
MODES = ["sft_think", "sft_no_think", "sft_think_code", "sft_no_think_code", "sft_chat_think", "sft_chat_no_think",
         "sft_think_ru", "sft_no_think_ru"]
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# Questions with the characters that break naive rendering: LaTeX braces, a literal "{question}",
# dollar signs, newlines and non-ASCII text.
QUESTIONS = [
    "Natalia sold clips to 48 of her friends in April. How many clips did she sell?",
    "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2\n\\end{array}\n\\right.\\]Find $a+b$.",
    "What is $\\frac{1}{2} + \\dfrac{3}{4}$? Write {question} literally {0} and {x}.",
    "Сколько будет 2+2? Ответ в градусах: 90°.",
]


def test_registry_is_consistent():
    assert sorted(prompts.PROMPT_MODES) == sorted(MODES)
    for name, mode in prompts.PROMPT_MODES.items():
        assert mode.name == name
        assert mode.agent_loop == f"orbit_{name}"
        assert mode.default_data_source in prompts.DATA_SOURCES
        assert mode.template_source == (f"configs/verl/prompts/{name}.yaml", "prompt_template")
    assert prompts.DATA_SOURCES == {"orbit/math": "math", "orbit/code": "code", "orbit/none": "none"}
    # The chat modes have no gold answers; the code modes are scored by the code reward.
    assert {name: prompts.PROMPT_MODES[name].default_data_source
            for name in ("sft_chat_think", "sft_chat_no_think", "sft_think_code", "sft_no_think_code")} == {
        "sft_chat_think": "orbit/none", "sft_chat_no_think": "orbit/none",
        "sft_think_code": "orbit/code", "sft_no_think_code": "orbit/code",
    }


def test_every_agent_loop_of_the_registry_is_declared_for_verl():
    entries = yaml.safe_load((prompts.repo_root() / "configs" / "verl" / "agent_loops.yaml").read_text(encoding="utf-8"))
    declared = {entry["name"]: entry["prompt_mode"] for entry in entries}
    assert declared == {mode.agent_loop: name for name, mode in prompts.PROMPT_MODES.items()}


def test_reward_of_reads_the_part_before_the_tag():
    assert prompts.reward_of("orbit/math") == "math"
    assert prompts.reward_of("orbit/math#no_think") == "math"
    assert prompts.reward_of("orbit/code#think#extra") == "code"
    assert prompts.reward_of("orbit/none#chat") == "none"
    assert prompts.reward_of("orbit/maths#think") is None
    assert prompts.reward_of("eval#orbit/math") is None
    assert prompts.reward_of("") is None


def test_repo_root_holds_the_package():
    root = prompts.repo_root()
    assert (root / "src" / "orbit_verl" / "data" / "prompts.py").is_file()


@pytest.mark.parametrize("name", MODES)
def test_templates_come_from_their_yaml(name):
    mode = prompts.PROMPT_MODES[name]
    rel_path, key = mode.template_source
    doc = yaml.safe_load((prompts.repo_root() / rel_path).read_text(encoding="utf-8"))
    # A completion ends at the end-of-turn token, so no template declares stop strings.
    assert set(doc) == {key}
    assert prompts.load_template(mode) == doc[key]


def test_no_think_marks_exactly_the_prompts_that_close_an_empty_thinking_block():
    # The flag must follow the template text, so a new no-think template cannot miss the length cap.
    flagged = {name for name, mode in prompts.PROMPT_MODES.items() if mode.no_think}
    assert flagged == {"sft_no_think", "sft_no_think_code", "sft_chat_no_think", "sft_no_think_ru"}
    for name, mode in prompts.PROMPT_MODES.items():
        template = prompts.load_template(mode)
        assert mode.no_think == template.endswith("<think>\n\n</think>\n\n"), name


@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("  ", None), ("4096", 4096), (" 512 ", 512)])
def test_the_no_think_cap_reads_a_positive_whole_number(value, expected):
    environ = {} if value is None else {prompts.NO_THINK_MAX_TOKENS_ENV: value}
    assert prompts.no_think_max_tokens(environ) == expected


@pytest.mark.parametrize("value", ["0", "-1", "4k", "1.5", "4_096", "+5", "²", "null"])
def test_the_no_think_cap_refuses_anything_else(value):
    with pytest.raises(ValueError, match=prompts.NO_THINK_MAX_TOKENS_ENV):
        prompts.no_think_max_tokens({prompts.NO_THINK_MAX_TOKENS_ENV: value})


def test_the_no_think_cap_is_off_without_the_variable(monkeypatch):
    monkeypatch.delenv(prompts.NO_THINK_MAX_TOKENS_ENV, raising=False)
    assert prompts.NO_THINK_MAX_TOKENS_ENV == "ORBIT_NO_THINK_MAX_TOKENS"
    assert prompts.no_think_max_tokens() is None
    monkeypatch.setenv(prompts.NO_THINK_MAX_TOKENS_ENV, "4096")
    assert prompts.no_think_max_tokens() == 4096


def test_opens_think_marks_exactly_the_prompts_that_open_a_thinking_block():
    # The agent loop caps the first call at the budget before the prompt reaches vLLM, so the static
    # flag must follow the template text.
    flagged = {name for name, mode in prompts.PROMPT_MODES.items() if mode.opens_think}
    assert flagged == {"sft_think", "sft_think_code", "sft_chat_think", "sft_think_ru"}
    for name, mode in prompts.PROMPT_MODES.items():
        template = prompts.load_template(mode)
        assert mode.opens_think == template.endswith("<think>\n"), name
        # A prompt either opens a thinking block or closes an empty one, never both.
        assert not (mode.opens_think and mode.no_think), name
    # The rendered prompts end inside the block too, whatever the question holds.
    for name in sorted(flagged):
        for question in QUESTIONS:
            assert prompts.render_messages(name, question)[0]["content"].endswith("<|im_start|>assistant\n<think>\n")


@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("  ", None), ("14336", 14336), (" 512 ", 512)])
def test_the_think_budget_reads_a_positive_whole_number(value, expected):
    environ = {} if value is None else {prompts.THINK_BUDGET_ENV: value}
    assert prompts.think_budget(environ) == expected


@pytest.mark.parametrize("value", ["0", "-1", "6k", "1.5", "14_336", "+5", "²", "null"])
def test_the_think_budget_refuses_anything_else(value):
    with pytest.raises(ValueError, match=prompts.THINK_BUDGET_ENV):
        prompts.think_budget({prompts.THINK_BUDGET_ENV: value})


def test_the_think_budget_is_off_without_the_variable(monkeypatch):
    monkeypatch.delenv(prompts.THINK_BUDGET_ENV, raising=False)
    assert prompts.THINK_BUDGET_ENV == "ORBIT_THINK_BUDGET"
    assert prompts.think_budget() is None
    monkeypatch.setenv(prompts.THINK_BUDGET_ENV, "14336")
    assert prompts.think_budget() == 14336


def test_the_budget_and_the_no_think_cap_read_their_own_variables():
    # The 16K hybrid sets both, so a mix-up would cap the thinking at the no-think cap or the reverse.
    both = {prompts.THINK_BUDGET_ENV: "14336", prompts.NO_THINK_MAX_TOKENS_ENV: "4096"}
    assert (prompts.think_budget(both), prompts.no_think_max_tokens(both)) == (14336, 4096)
    assert prompts.think_budget({prompts.NO_THINK_MAX_TOKENS_ENV: "4096"}) is None
    assert prompts.no_think_max_tokens({prompts.THINK_BUDGET_ENV: "14336"}) is None
    # An error names the variable that was read, so the shared parser cannot blame the other one.
    with pytest.raises(ValueError, match="^ORBIT_SOME_LIMIT='4k' "):
        prompts.token_count_env("ORBIT_SOME_LIMIT", {"ORBIT_SOME_LIMIT": "4k"})
    with pytest.raises(ValueError) as info:
        prompts.think_budget({prompts.THINK_BUDGET_ENV: "6k", prompts.NO_THINK_MAX_TOKENS_ENV: "4096"})
    assert prompts.NO_THINK_MAX_TOKENS_ENV not in str(info.value)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown prompt mode"):
        prompts.get_mode("nope")
    with pytest.raises(ValueError):
        prompts.render_messages("nope", "q")


def test_load_template_rejects_a_template_without_the_question_field(tmp_path, monkeypatch):
    (tmp_path / "t.yaml").write_text('prompt_template: "Q: {query}\\nA:"\n', encoding="utf-8")
    monkeypatch.setattr(prompts, "repo_root", lambda: tmp_path)
    mode = prompts.PromptMode("probe", "orbit_probe", "orbit/math", ("t.yaml", "prompt_template"))
    with pytest.raises(ValueError, match="question"):
        prompts.load_template(mode)


def test_rendered_prompts_look_right():
    think = prompts.render_messages("sft_think", "Q?")
    assert think == [{"role": "user", "content": (
        f"<|endoftext|><|im_start|>user\nQ?\n{MATH_INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n<think>\n")}]
    # The doubled braces of the YAML come out single.
    assert "\\boxed{}" in think[0]["content"] and "{{" not in think[0]["content"]
    chat = prompts.render_messages("sft_chat_no_think", "Hi!")[0]["content"]
    assert chat == "<|endoftext|><|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    for name in MODES:
        for question in QUESTIONS:
            [message] = prompts.render_messages(name, question)
            assert message["role"] == "user"
            assert message["content"] == prompts.load_template(name).format(question=question)
            assert question in message["content"]


@pytest.mark.parametrize("name", MODES)
def test_prompt_ids_are_the_raw_text_without_added_tokens(qwen3_tokenizer, name):
    """The ids are the template text encoded as it is: one leading <|endoftext|> and nothing appended."""
    eot = qwen3_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert eot == 151643
    assert prompts.missing_tokens(name, qwen3_tokenizer) == []
    for question in QUESTIONS:
        messages = prompts.render_messages(name, question)
        ids = prompts.prompt_token_ids(qwen3_tokenizer, messages)
        assert ids == qwen3_tokenizer.encode(messages[0]["content"], add_special_tokens=False)
        assert ids[0] == eot and eot not in ids[1:]
        assert qwen3_tokenizer.decode(ids) == messages[0]["content"]


def test_missing_tokens_finds_markup_the_tokenizer_splits():
    class Splitting:
        """A tokenizer that knows no special tokens, so it splits every markup item into characters."""

        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

    assert prompts.missing_tokens("sft_think", Splitting()) == list(prompts.PROMPT_MODES["sft_think"].required_tokens)


def test_import_reads_no_files_and_stays_light():
    """Ray workers import prompts per process; the import must not open templates or load heavy libs."""
    probe = r"""
import json, os, sys
opened = []
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if path.endswith((".yaml", ".yml", ".json", ".jinja")):
            opened.append(path)
sys.addaudithook(hook)
before = set(sys.modules)
import orbit_verl.data.prompts as p
new = set(sys.modules) - before
heavy = sorted(m for m in new if m.split(".")[0] in {"transformers", "yaml", "torch", "verl", "tokenizers"})
trainer = sorted(m for m in new if m == "orbit" or m.startswith("orbit."))
print(json.dumps({"opened": opened, "heavy": heavy, "trainer": trainer, "cache": p._TEMPLATE_CACHE}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(prompts.repo_root() / "src"), env.get("PYTHONPATH")]))
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True)
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"opened": [], "heavy": [], "trainer": [], "cache": {}}



def test_the_russian_math_modes_differ_from_the_english_ones_only_in_the_instruction():
    """The Russian instruction is the SFT eval's MGSM-ru one (chat_local.MATH_INSTRUCTION_RU), and the
    rest of the format is that of sft_think and sft_no_think."""
    english = "Please reason step by step, and put your final answer within \\boxed{}."
    russian = "Рассуждай пошагово и запиши окончательный ответ в \\boxed{}."
    for ru, en in (("sft_think_ru", "sft_think"), ("sft_no_think_ru", "sft_no_think")):
        ru_text = prompts.load_template(prompts.PROMPT_MODES[ru])
        en_text = prompts.load_template(prompts.PROMPT_MODES[en])
        assert russian in ru_text.format(question="Q") and english not in ru_text.format(question="Q")
        assert ru_text.replace(russian.replace("{}", "{{}}"), english.replace("{}", "{{}}")) == en_text
        mode = prompts.PROMPT_MODES[ru]
        assert (mode.opens_think, mode.no_think) == (prompts.PROMPT_MODES[en].opens_think, prompts.PROMPT_MODES[en].no_think)
