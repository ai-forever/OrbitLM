"""Prompt modes: what text the model sees, how it is tokenized, and where a completion stops.

``PROMPT_MODES`` is the single source of truth for the ways we prompt OrbitLM. The data converters,
the rollout agent loop and the parity tool all read it, so a mode means the same thing everywhere.
``docs/verl/DESIGN.md`` explains why the prompts are the release chat format written out as text.

Importing this module must stay cheap. Ray workers import it once per process to look up the
thinking flags, and they never need the template text. That is why templates are read
lazily, on the first ``load_template`` call, and why neither ``yaml`` nor ``transformers`` is
imported at module level.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any



@dataclass(frozen=True)
class PromptMode:
    """One way of turning a question into prompt tokens.

    Every mode is the release chat format written out as text, so it does not depend on the chat
    template that an export ships. ``template_source`` names the YAML file (relative to the repo root)
    and the key that hold the prompt template. A completion ends at the end-of-turn token
    ``<|im_end|>``, which is a stop id of the prepared model dir, so no mode needs stop strings.

    ``required_tokens`` lists markup that the prompt only works with when the tokenizer knows each
    item as one token. ``missing_tokens`` checks this, and the data tools refuse a tokenizer that
    would split such markup into ordinary pieces the model never saw in training.

    ``no_think`` marks the modes whose prompt already ends with an empty, closed thinking block, so
    the model answers without thinking. The agent loop gives these answers their own length cap
    when ``ORBIT_NO_THINK_MAX_TOKENS`` is set (``no_think_max_tokens``).

    ``opens_think`` marks the modes whose prompt ends with "<think>\n", so the response starts inside
    a thinking block. When ``ORBIT_THINK_BUDGET`` is set (``think_budget``), the agent loop closes
    such a block by force once the budget runs out, and the model answers in the rest of the response
    length. The flag is static because the cap has to be known before the prompt reaches vLLM.
    """

    name: str
    agent_loop: str
    default_data_source: str
    template_source: tuple[str, str]
    required_tokens: tuple[str, ...] = ()
    no_think: bool = False
    opens_think: bool = False


PROMPT_MODES: dict[str, PromptMode] = {
    mode.name: mode
    for mode in (
        # The release chat format in think mode, for the SFT chat export and the models trained
        # from it. The prompt is that format written out as text, so it does not depend on the chat
        # template that an export ships. No mode has a system turn, and for such a chat the SFT
        # export's template renders the same text that follows the leading <|endoftext|> here.
        # It is <|endoftext|> (packing puts it before most SFT dialogs, and the SFT eval adds it
        # too), the user turn "<question>\n<instruction>", and the assistant turn opened with
        # "<think>\n". The response therefore starts inside the thinking block, and only a
        # </think> closes it. The model ends its turn with <|im_end|>, a stop id of the qwen3 preset.
        PromptMode(
            name="sft_think",
            agent_loop="orbit_sft_think",
            default_data_source="orbit/math",
            template_source=("configs/verl/prompts/sft_think.yaml", "prompt_template"),
            required_tokens=("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>"),
            opens_think=True,
        ),
        # The same format without thinking: the assistant turn starts with an empty, closed
        # thinking block, as the SFT template renders it with enable_thinking=false. It is meant
        # for validation rows that measure the no-think answer next to the think answer.
        PromptMode(
            name="sft_no_think",
            agent_loop="orbit_sft_no_think",
            default_data_source="orbit/math",
            template_source=("configs/verl/prompts/sft_no_think.yaml", "prompt_template"),
            required_tokens=("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"),
            no_think=True,
        ),
        # The two math modes above for a Russian problem: the instruction after the question is the
        # Russian one of the SFT eval's MGSM-ru prompt, so a Russian problem reaches the model, and
        # the teacher, without English in its user turn. The distillation data gives these modes to
        # the math rows whose language is Russian (distill_data.LANG_FORMATS).
        PromptMode(
            name="sft_think_ru",
            agent_loop="orbit_sft_think_ru",
            default_data_source="orbit/math",
            template_source=("configs/verl/prompts/sft_think_ru.yaml", "prompt_template"),
            required_tokens=("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>"),
            opens_think=True,
        ),
        PromptMode(
            name="sft_no_think_ru",
            agent_loop="orbit_sft_no_think_ru",
            default_data_source="orbit/math",
            template_source=("configs/verl/prompts/sft_no_think_ru.yaml", "prompt_template"),
            required_tokens=("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"),
            no_think=True,
        ),
        # The think mode for code problems. The format is that of sft_think, but the user turn is
        # "<question>\n\n<instruction>" with the instruction that the code rows of nvidia
        # AceReason-1.1-SFT put after every problem, copied word for word. The SFT v5 mix that the
        # recipes start from does not hold those rows, but the instruction asks plainly for a
        # ```python block after </think>, and the code reward runs the last such block.
        PromptMode(
            name="sft_think_code",
            agent_loop="orbit_sft_think_code",
            default_data_source="orbit/code",
            template_source=("configs/verl/prompts/sft_think_code.yaml", "prompt_template"),
            required_tokens=("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>"),
            opens_think=True,
        ),
        # The code format of sft_think_code without thinking: the same user turn, and the assistant
        # turn opened with the empty, closed thinking block of sft_no_think. The model then writes
        # its ```python block right away. Distillation uses it for the no_think half of the code
        # prompts.
        PromptMode(
            name="sft_no_think_code",
            agent_loop="orbit_sft_no_think_code",
            default_data_source="orbit/code",
            template_source=("configs/verl/prompts/sft_no_think_code.yaml", "prompt_template"),
            required_tokens=("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"),
            no_think=True,
        ),
        # The release chat format with the user turn exactly as the user wrote it, for prompts that
        # are not math problems: instructions, chat and refusals. The math modes above append the
        # "\boxed{}" instruction to every question, which would be wrong for these prompts. The
        # rows have no gold answer, so their default data source is orbit/none, whose reward is 0.
        PromptMode(
            name="sft_chat_think",
            agent_loop="orbit_sft_chat_think",
            default_data_source="orbit/none",
            template_source=("configs/verl/prompts/sft_chat_think.yaml", "prompt_template"),
            required_tokens=("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>"),
            opens_think=True,
        ),
        # The same chat format without thinking, with the empty, closed thinking block of sft_no_think.
        PromptMode(
            name="sft_chat_no_think",
            agent_loop="orbit_sft_chat_no_think",
            default_data_source="orbit/none",
            template_source=("configs/verl/prompts/sft_chat_no_think.yaml", "prompt_template"),
            required_tokens=("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"),
            no_think=True,
        ),
    )
}

# Maps the parquet ``data_source`` value to the name of its reward. math checks the last boxed answer
# (``orbit_verl.data.math_answer``). code runs the response's program against the row's tests in the
# sandbox of ``orbit_verl.code_sandbox``. none is for prompts without a gold answer, such as most
# distillation prompts. Their reward is always 0, and only the diagnostics of the response are reported.
DATA_SOURCES: dict[str, str] = {
    "orbit/math": "math",
    "orbit/code": "code",
    "orbit/none": "none",
}

# A data_source may carry a tag after this character, as in "orbit/math#no_think". The reward is
# chosen by the part before it. verl groups the validation metrics by the whole data_source, so two
# validation sets with the same reward but different prompt modes are reported separately.
DATA_SOURCE_TAG = "#"


def reward_of(data_source: str) -> str | None:
    """The reward name that ``data_source`` selects, or None when its part before the tag is unknown."""
    return DATA_SOURCES.get(str(data_source).partition(DATA_SOURCE_TAG)[0])


# The run config caps the answers of the no_think modes through this variable. A config sets it
# under orbit.env, and the launcher copies that block into the env of every Ray process, the
# agent-loop workers included.
NO_THINK_MAX_TOKENS_ENV = "ORBIT_NO_THINK_MAX_TOKENS"


# The thinking budget of the modes that open a thinking block (``PromptMode.opens_think``). Like the
# no-think cap, a config sets it under orbit.env.
THINK_BUDGET_ENV = "ORBIT_THINK_BUDGET"


def token_count_env(name: str, environ: Mapping[str, str] | None = None) -> int | None:
    """A positive whole number of tokens from the variable ``name``, or None when it is unset or empty.

    A value that is not a positive whole number is refused. Reading "4k" or "1.5" some other way
    would silently train with a limit nobody asked for.
    """
    environ = os.environ if environ is None else environ
    raw = str(environ.get(name, "")).strip()
    if not raw:
        return None
    if not (raw.isascii() and raw.isdigit()) or int(raw) < 1:
        raise ValueError(f"{name}={raw!r} is not a positive whole number of tokens.")
    return int(raw)


def no_think_max_tokens(environ: Mapping[str, str] | None = None) -> int | None:
    """The cap on new tokens for the ``no_think`` modes, from ``ORBIT_NO_THINK_MAX_TOKENS``, or None.

    An unset or empty variable means no cap, so RL runs that never set it are unchanged.
    """
    return token_count_env(NO_THINK_MAX_TOKENS_ENV, environ)


def think_budget(environ: Mapping[str, str] | None = None) -> int | None:
    """The thinking budget of the ``opens_think`` modes, from ``ORBIT_THINK_BUDGET``, or None.

    An unset or empty variable means no budget, so runs that never set it are unchanged. With a
    budget, a training response whose thinking block is still open after that many tokens is closed
    by force (``orbit_verl.data.agent_loop``).
    """
    return token_count_env(THINK_BUDGET_ENV, environ)

# Template text per mode name, filled on first use.
_TEMPLATE_CACHE: dict[str, str] = {}


def repo_root() -> Path:
    """The repository that holds the running ``orbit_verl`` package (``<repo>/src/orbit_verl``)."""
    import orbit_verl

    return Path(orbit_verl.__file__).resolve().parents[2]


def get_mode(mode: PromptMode | str) -> PromptMode:
    """Accept a mode object or its name, and fail with the list of valid names otherwise."""
    if isinstance(mode, PromptMode):
        return mode
    try:
        return PROMPT_MODES[str(mode)]
    except KeyError:
        raise ValueError(f"Unknown prompt mode {mode!r}. Known modes: {sorted(PROMPT_MODES)}.") from None


def load_template(mode: PromptMode | str) -> str:
    """Return the mode's prompt template, a format string with one ``{question}`` field.

    The template is read from YAML on the first call and cached. Literal braces in the templates are
    doubled (``{{`` and ``}}``), because the template is rendered with ``str.format``.
    """
    mode = get_mode(mode)
    cached = _TEMPLATE_CACHE.get(mode.name)
    if cached is not None:
        return cached

    import yaml

    rel_path, key = mode.template_source
    path = repo_root() / rel_path
    with open(path, encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    template = doc.get(key) if isinstance(doc, dict) else None
    if not isinstance(template, str) or "{question}" not in template:
        raise ValueError(f"{path} has no string {key!r} with a {{question}} field.")
    _TEMPLATE_CACHE[mode.name] = template
    return template


def missing_tokens(mode: PromptMode | str, tokenizer: Any) -> list[str]:
    """The mode's ``required_tokens`` that ``tokenizer`` does not encode as exactly one token.

    The check encodes each item and counts the ids. Looking the item up with
    ``convert_tokens_to_ids`` is not enough, because a tokenizer may answer an unknown item with an
    unk id that is also the id of a real token.
    """
    mode = get_mode(mode)
    return [token for token in mode.required_tokens
            if len(tokenizer.encode(token, add_special_tokens=False)) != 1]


def render_messages(mode: PromptMode | str, question: str) -> list[dict[str, str]]:
    """Build the chat messages stored in the parquet ``prompt`` column for one question.

    The result is a single user message whose content is the complete prompt text. The agent loop
    later tokenizes that text as it is, without a chat template.
    """
    template = load_template(mode)
    return [{"role": "user", "content": template.format(question=question)}]


def prompt_token_ids(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> list[int]:
    """Token ids of a prompt, computed the same way by the data tools, the rollout and the parity tool.

    The concatenated message contents are encoded with ``add_special_tokens=False``, so no token is
    added that the model did not see in training. By default the Qwen3 tokenizer appends
    ``<|endoftext|>`` (id 151643), the document separator of pretraining, and the model would then
    start its answer right after an end-of-document marker. Any new code that tokenizes prompts must
    therefore pass ``add_special_tokens=False`` too.
    """
    text = "".join(str(message["content"]) for message in messages)
    return [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]
