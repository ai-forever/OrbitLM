"""OrbitLM model code for verl and the tool that builds verl-ready model dirs.

- ``modelling_ouro.py`` and ``configuration_ouro.py``: the transformers-5 port of the trainer's model
  code. They are HF remote code: ``prepare_model`` copies them into each prepared model dir.
- ``prepare_model``: builds a model dir that verl and vLLM can load (``python -m
  orbit_verl.model.prepare_model --help``).
- ``parity_logprobs``: compares the ported HF model's log-probs with vLLM's on a GPU.

This file must stay cheap to import. The launcher reads ``PRESETS`` and ``MODEL_CODE_DIR`` from
processes that should not pay for importing torch or transformers, so nothing here imports them.
"""
from __future__ import annotations

from pathlib import Path

# The directory that holds the ported model code. prepare_model copies modelling_ouro.py and
# configuration_ouro.py from here into every prepared model dir.
MODEL_CODE_DIR = Path(__file__).resolve().parent

# The files of the ported model code that a prepared model dir needs, in the order they are copied.
MODEL_CODE_FILES = ("modelling_ouro.py", "configuration_ouro.py")

# Generation settings per tokenizer family. They go into the prepared dir's generation_config.json,
# which is where vLLM takes its stop tokens from. The release has one family, the Qwen3 tokenizer:
# <|im_end|> (151645) and <|endoftext|> (151643) both end a completion. Without <|im_end|> in this
# list, vLLM never stops a chat turn on its own.
PRESETS: dict[str, dict] = {
    "qwen3": {
        "vocab_size": 151936,
        "eos_token_ids": [151645, 151643],
        "pad_token_id": 151643,
        "bos_token_id": None,
    },
}


def preset_for_vocab(vocab_size: int) -> str:
    """Return the name of the preset whose vocabulary size equals ``vocab_size``.

    Raises ``ValueError`` when no preset matches. In that case the caller has to pick the stop
    tokens explicitly, because guessing them would let generation run past the end of an answer.
    """
    for name, preset in PRESETS.items():
        if preset["vocab_size"] == int(vocab_size):
            return name
    known = ", ".join(f"{name} ({preset['vocab_size']})" for name, preset in PRESETS.items())
    raise ValueError(
        f"No model preset has vocab_size {vocab_size}. Known presets: {known}. "
        "Pass --preset and the token ids (--eos-ids, --pad-id, --bos-id) explicitly."
    )
