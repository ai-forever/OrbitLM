"""Chat prompts of the SFT evaluation and the token contract they must meet.

The in-training evaluation of an SFT checkpoint renders every prompt with the chat template of the
exported tokenizer, which is the template the SFT run trains on. This module does that rendering and
checks the result against the token sequences of the think/no_think switch (README.md in this folder).
It needs only transformers, so the evaluation processes (itr_chat.py), the
trainer's startup check (the "+chat_render" preflight) and the inference scripts all use it, and a
prompt is rendered the same way everywhere.

A prompt is rendered in one of three modes:

* think: the template with enable_thinking=True, which must end with "<|im_start|>assistant\\n<think>\\n".
* no_think: the template with enable_thinking=False, which must end with the empty think block.
* no_prefill: the think render with its final "<think>\\n" removed. Gate G3 asks whether the model opens
  the think block by itself when a client sends no prefill.

The leading "<|endoftext|>" is added or removed here, so the result does not depend on whether a template
writes it. In SFT training that token is the end of the previous dialogue in the packed block, and
document masking turns it into the first token of the next dialogue's attention segment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

MODES = ("think", "no_think", "no_prefill")

LEAD_TEXT = "<|endoftext|>"
IM_START_TEXT = "<|im_start|>"
THINK_PREFILL = "<think>\n"
TAIL_TEXT = {
    "think": "<|im_start|>assistant\n<think>\n",
    "no_think": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "no_prefill": "<|im_start|>assistant\n",
}

# The ids of the prompt contract for the Qwen3 vocabulary that every OrbitLM SFT run uses. The contract
# itself is derived from the tokenizer in use (expected_ids); these literal ids are what the startup
# check compares that derivation with, so a different vocabulary stops the run instead of being
# checked against itself.
CONTRACT_TAIL_IDS = {
    "think": [151644, 77091, 198, 151667, 198],
    "no_think": [151644, 77091, 198, 151667, 271, 151668, 271],
    "no_prefill": [151644, 77091, 198],
}
CONTRACT_LEAD_ID = 151643
CONTRACT_IM_START_ID = 151644

# The probe conversations of the startup check: one English and one Russian single-turn prompt.
PROBE_MESSAGES = (
    [{"role": "user", "content": "What is 17 times 23? Answer briefly."}],
    [{"role": "user", "content": "Сколько будет 17 умножить на 23? Ответь коротко."}],
)
# A text whose ids must be the same in two tokenizer folders that claim to share a vocabulary.
VOCAB_PROBE_TEXT = (
    "<|endoftext|><|im_start|>user\nSolve $x^2=4$ и объясни.<|im_end|>\n<|im_start|>assistant\n"
    "<think>\nOkay, x = ±2.\n</think>\n\n\\boxed{2}<|im_end|>"
)


class ContractError(ValueError):
    """A rendered prompt does not have the token sequence that the prompt contract requires."""


def render(tokenizer: Any, messages: list[dict[str, Any]], mode: str, lead_eos: bool,
           chat_template: Optional[str] = None, template_args: Optional[dict[str, Any]] = None) -> str:
    """The prompt text of one conversation in one mode, with the leading-token policy applied.

    enable_thinking is always passed as an explicit boolean. Templates read an unset or None value
    differently: the stock Qwen3.5-4B template thinks, and the template of the SFT tokenizer folders
    that data_prep/tokenizer writes does not. The no-prefill prompt is the think prompt without its
    prefill, and the prefill must be there, because otherwise the template would not be a
    think-by-default template at all.
    """
    if mode not in MODES:
        raise ValueError(f"unknown prompt mode {mode!r}; known modes are {list(MODES)}")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=mode != "no_think",
        chat_template=chat_template,
        **(template_args or {}),
    )
    if mode == "no_prefill":
        if not text.endswith(THINK_PREFILL):
            raise ContractError(
                "the think render does not end with '<think>\\n', so there is no prefill to remove for the "
                f"no-prefill prompt; its end is {text[-60:]!r}"
            )
        text = text[: -len(THINK_PREFILL)]
    while text.startswith(LEAD_TEXT):
        text = text[len(LEAD_TEXT):]
    return LEAD_TEXT + text if lead_eos else text


def expected_ids(tokenizer: Any) -> dict[str, Any]:
    """The contract's ids, derived by encoding the literal strings with this tokenizer."""
    tails = {mode: list(tokenizer.encode(text, add_special_tokens=False)) for mode, text in TAIL_TEXT.items()}
    return {
        "tails": tails,
        "lead": int(tokenizer.convert_tokens_to_ids(LEAD_TEXT)),
        "im_start": int(tokenizer.convert_tokens_to_ids(IM_START_TEXT)),
    }


def contract_problem(ids: list[int], mode: str, lead_eos: bool, contract: dict[str, Any]) -> Optional[str]:
    """Why a prompt's ids break the contract, or None when they keep it.

    The head must be exactly one leading token followed by <|im_start|> (or <|im_start|> alone without
    the leading token), and the tail must be the mode's prefill. A doubled leading token would mean that
    something besides the template added one, which is the drift the check exists for.
    """
    tail = contract["tails"][mode]
    lead, im_start = contract["lead"], contract["im_start"]
    if lead_eos:
        if len(ids) < 2 or ids[0] != lead or ids[1] != im_start:
            return f"the prompt must start with [{lead}, {im_start}], but starts with {ids[:4]}"
    elif not ids or ids[0] != im_start:
        return f"the prompt must start with [{im_start}] (no leading token), but starts with {ids[:4]}"
    if list(ids[-len(tail):]) != tail:
        return f"the {mode} prompt must end with {tail}, but ends with {list(ids[-len(tail) - 2:])}"
    return None


def template_text(folder: Path) -> Optional[str]:
    """The chat template of a tokenizer folder: chat_template.jinja, else the tokenizer config's entry."""
    folder = Path(folder)
    jinja = folder / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    try:
        config = json.loads((folder / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    template = config.get("chat_template")
    if isinstance(template, list):
        # A list of named templates; the default one is what apply_chat_template uses.
        named = {entry.get("name"): entry.get("template") for entry in template if isinstance(entry, dict)}
        template = named.get("default")
    return template if isinstance(template, str) else None


def template_sha256(folder: Path) -> Optional[str]:
    text = template_text(folder)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None


def render_check(tokenizer_dir: str, train_tokenizer_dir: Optional[str] = None) -> dict[str, Any]:
    """The startup check of an SFT evaluation: render the probes and compare them with the contract.

    It renders the English and the Russian probe in every mode, with and without the leading token,
    tokenizes them as lm-eval's vLLM backend would (without special tokens, because the text already
    holds them), and checks every result against the contract. It also checks that the contract derived
    from this tokenizer has the literal ids above. When the training tokenizer folder is another folder,
    it checks that both folders give the same ids for a probe text. Any problem raises ContractError.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    contract = expected_ids(tokenizer)
    if (contract["tails"] != CONTRACT_TAIL_IDS or contract["lead"] != CONTRACT_LEAD_ID
            or contract["im_start"] != CONTRACT_IM_START_ID):
        raise ContractError(
            f"{tokenizer_dir}: the tokenizer encodes the prompt tails as {contract}, not as the contract's "
            f"{CONTRACT_TAIL_IDS} with leading token {CONTRACT_LEAD_ID}"
        )
    rendered = {}
    for index, messages in enumerate(PROBE_MESSAGES):
        for mode in MODES:
            for lead_eos in (True, False):
                text = render(tokenizer, messages, mode, lead_eos)
                ids = tokenizer.encode(text, add_special_tokens=False)
                problem = contract_problem(ids, mode, lead_eos, contract)
                if problem:
                    raise ContractError(f"{tokenizer_dir}: probe {index} in mode {mode} (lead_eos={lead_eos}): {problem}")
                rendered[f"{index}_{mode}_{int(lead_eos)}"] = len(ids)
    report: dict[str, Any] = {"probes": len(rendered), "template_sha256": template_sha256(Path(tokenizer_dir))}
    if train_tokenizer_dir and Path(train_tokenizer_dir).resolve() != Path(tokenizer_dir).resolve():
        other = AutoTokenizer.from_pretrained(train_tokenizer_dir)
        mine = tokenizer.encode(VOCAB_PROBE_TEXT, add_special_tokens=False)
        theirs = other.encode(VOCAB_PROBE_TEXT, add_special_tokens=False)
        if mine != theirs:
            raise ContractError(
                f"{tokenizer_dir} and the training tokenizer {train_tokenizer_dir} encode the same text "
                "differently, so the evaluated prompts would not use the vocabulary the model was trained on"
            )
        report["vocab_checked_against"] = str(train_tokenizer_dir)
    return report
