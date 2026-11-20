"""The prompt modes render exactly the chat format that the SFT model was trained and evaluated with.

The reference is the SFT eval's think prompt: <|endoftext|> followed by the export's chat template
applied to one user turn "<question>\\n<instruction>". The modes write that format out as text, so
these tests compare it with the template of a real SFT export. The SFT v5 export ships the tools
template, which renders a chat without a system turn exactly like the think template, so a v5 export
and an export with the think template both serve as the reference. The code modes sft_think_code and
sft_no_think_code use the user turn of the AceReason-1.1-SFT code rows instead:
"<question>\\n\\n<code instruction>".
The chat modes sft_chat_think and sft_chat_no_think, which distillation uses for prompts that are not
math problems, send the question as the whole user turn.

The tests that compare with the chat template need an SFT chat export (ORBIT_TEST_SFT_EXPORT,
see conftest.py) and skip without one.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orbit_verl.data.prompts import PROMPT_MODES, prompt_token_ids, render_messages, repo_root  # noqa: E402

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# The instruction that the code rows of nvidia AceReason-1.1-SFT put after every problem.
CODE_INSTRUCTION = (
    "Write Python code to solve the problem. Please place the solution code in the following format:\n"
    "```python\n# Your solution code here\n```"
)
QUESTIONS = [
    "What is $1+2$?",
    "Let \\[f(x) = \\left\\{\n\\begin{array}{cl} ax+3, &\\text{ if }x>2\n\\end{array}\n\\right.\\]Find $a+b$.",
    "What is $\\frac{1}{2}$? Write {question} literally {0} and {x}.",
    "Сколько будет 2+2?",
]


@pytest.fixture(scope="module")
def tokenizer(sft_export, qwen3_tokenizer):
    if not Path(sft_export, "chat_template.jinja").is_file():
        pytest.skip(f"{sft_export} has no chat_template.jinja, so it is not an SFT chat export")
    return qwen3_tokenizer


@pytest.mark.parametrize("think", [True, False])
@pytest.mark.parametrize("question", QUESTIONS)
def test_sft_modes_match_the_sft_chat_template(tokenizer, question, think):
    mode = PROMPT_MODES["sft_think" if think else "sft_no_think"]
    ids = prompt_token_ids(tokenizer, render_messages(mode, question))
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{question}\n{INSTRUCTION}"}],
        add_generation_prompt=True,
        enable_thinking=think,
        tokenize=False,
    )
    expected = [tokenizer.convert_tokens_to_ids("<|endoftext|>")] + tokenizer.encode(text, add_special_tokens=False)
    assert ids == expected


@pytest.mark.parametrize("question", QUESTIONS)
def test_sft_think_code_matches_the_sft_chat_template(tokenizer, question):
    ids = prompt_token_ids(tokenizer, render_messages(PROMPT_MODES["sft_think_code"], question))
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{question}\n\n{CODE_INSTRUCTION}"}],
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )
    expected = [tokenizer.convert_tokens_to_ids("<|endoftext|>")] + tokenizer.encode(text, add_special_tokens=False)
    assert ids == expected


@pytest.mark.parametrize("question", QUESTIONS)
def test_sft_think_code_renders_the_exact_prompt(question):
    # No tokenizer is needed here, so this test runs everywhere. Braces in the question stay as they are.
    expected = (
        f"<|endoftext|><|im_start|>user\n{question}\n\n{CODE_INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    assert render_messages(PROMPT_MODES["sft_think_code"], question) == [{"role": "user", "content": expected}]


def test_sft_think_code_is_registered():
    mode = PROMPT_MODES["sft_think_code"]
    assert (mode.agent_loop, mode.default_data_source) == ("orbit_sft_think_code", "orbit/code")
    # The response starts inside the thinking block, as in sft_think, so the reward reads the code after </think>.
    assert mode.required_tokens == PROMPT_MODES["sft_think"].required_tokens and mode.opens_think
    entries = yaml.safe_load((repo_root() / "configs" / "verl" / "agent_loops.yaml").read_text(encoding="utf-8"))
    assert [entry for entry in entries if entry["name"] == mode.agent_loop] == [{
        "name": "orbit_sft_think_code",
        "_target_": "orbit_verl.data.agent_loop.OrbitLMAgentLoop",
        "prompt_mode": "sft_think_code",
    }]


# The user turn of each mode that distillation added, and whether its assistant turn opens a thinking block.
DISTILL_MODES = {
    "sft_no_think_code": ("{question}\n\n" + CODE_INSTRUCTION, False),
    "sft_chat_think": ("{question}", True),
    "sft_chat_no_think": ("{question}", False),
}


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.parametrize("name", sorted(DISTILL_MODES))
def test_distill_modes_match_the_sft_chat_template(tokenizer, name, question):
    user, think = DISTILL_MODES[name]
    ids = prompt_token_ids(tokenizer, render_messages(PROMPT_MODES[name], question))
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user.replace("{question}", question)}],
        add_generation_prompt=True,
        enable_thinking=think,
        tokenize=False,
    )
    expected = [tokenizer.convert_tokens_to_ids("<|endoftext|>")] + tokenizer.encode(text, add_special_tokens=False)
    assert ids == expected


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.parametrize("name", sorted(DISTILL_MODES))
def test_distill_modes_render_the_exact_prompt(name, question):
    # No tokenizer is needed here, so this test runs everywhere. Braces in the question stay as they are.
    user, think = DISTILL_MODES[name]
    turn = user.replace("{question}", question)
    opening = "<think>\n" if think else "<think>\n\n</think>\n\n"
    expected = f"<|endoftext|><|im_start|>user\n{turn}<|im_end|>\n<|im_start|>assistant\n{opening}"
    assert render_messages(PROMPT_MODES[name], question) == [{"role": "user", "content": expected}]


def test_distill_modes_are_registered():
    entries = yaml.safe_load((repo_root() / "configs" / "verl" / "agent_loops.yaml").read_text(encoding="utf-8"))
    for name, (_, think) in DISTILL_MODES.items():
        mode = PROMPT_MODES[name]
        assert (mode.agent_loop, mode.opens_think, mode.no_think) == (f"orbit_{name}", think, not think)
        # A no-think prompt holds a closed thinking block, so its </think> must be one token as well.
        twin = PROMPT_MODES["sft_think" if think else "sft_no_think"]
        assert mode.required_tokens == twin.required_tokens
        assert [entry for entry in entries if entry["name"] == mode.agent_loop] == [{
            "name": mode.agent_loop,
            "_target_": "orbit_verl.data.agent_loop.OrbitLMAgentLoop",
            "prompt_mode": name,
        }]
    assert PROMPT_MODES["sft_no_think_code"].default_data_source == "orbit/code"
    assert PROMPT_MODES["sft_chat_think"].default_data_source == "orbit/none"
    assert PROMPT_MODES["sft_chat_no_think"].default_data_source == "orbit/none"


@pytest.mark.parametrize("name", ["sft_no_think", "sft_no_think_code", "sft_chat_no_think"])
def test_sft_no_think_prompt_closes_the_thinking_block(tokenizer, name):
    ids = prompt_token_ids(tokenizer, render_messages(PROMPT_MODES[name], QUESTIONS[0]))
    # The prompt ends with an empty, closed thinking block, so the response starts with the answer.
    assert tokenizer.decode(ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert ids[0] == tokenizer.convert_tokens_to_ids("<|endoftext|>")


@pytest.mark.parametrize("name", ["sft_think", "sft_think_code", "sft_chat_think"])
def test_sft_think_prompt_opens_the_thinking_block(tokenizer, name):
    ids = prompt_token_ids(tokenizer, render_messages(PROMPT_MODES[name], QUESTIONS[0]))
    # The prompt ends with "<|im_start|>assistant\n<think>\n", so the model's answer starts inside
    # the thinking block and the reward needs a </think> before it reads the answer.
    assert tokenizer.decode(ids[-5:]) == "<|im_start|>assistant\n<think>\n"
    assert ids[0] == tokenizer.convert_tokens_to_ids("<|endoftext|>")
