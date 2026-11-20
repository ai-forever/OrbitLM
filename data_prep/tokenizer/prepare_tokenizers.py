"""Writes the three tokenizer directories that the release training configs read.

  tokenizer_qwen3_orbit           pretraining, 16K and 32K (no chat use)
  tokenizer_qwen3_4b_qwen35_think  the Qwen3.5-style think switch template, which the in-training
                                   evaluation of SFT renders its prompts with
  tokenizer_qwen3_4b_qwen35_tools  SFT v5 (stage 5): the same switch, a system turn and Qwen3
                                   tool calls

For chats without a system turn, the think and tools templates render the same text and token
ids; the tools template keeps a system turn, and the think template drops it.

All three share one vocabulary: the Qwen3 tokenizer (151,669 entries) plus twelve OrbitLM special
tokens at ids 151669-151680. The model declares vocab_size 151936, so the extra ids fit the
embedding table without resizing it. eos and pad are <|endoftext|> (151643) everywhere, because
that token is the document boundary of document masking, and the trainer refuses document
masking when eos is <|im_end|>.

The pretraining tokenizer is built from Qwen/Qwen3-1.7B-Base, the SFT ones from Qwen/Qwen3-4B.
Both bases have the same vocabulary; the base only decides which chat template the pretraining
directory ships and how its special tokens are written to disk. Each variant follows the recipe
that wrote the directory the training runs used, so the output matches those directories byte
for byte (see data_prep/README.md).

Usage:
  python data_prep/tokenizer/prepare_tokenizers.py --out-dir "$ORBIT_TOKENIZERS"
  python data_prep/tokenizer/prepare_tokenizers.py --out-dir DIR --only tokenizer_qwen3_orbit --offline
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
END_OF_TEXT = "<|endoftext|>"
END_OF_TEXT_ID = 151643
# The Qwen3 tokenizer holds this many entries; the OrbitLM specials start right after it.
QWEN3_TOKENIZER_LENGTH = 151669
DECLARED_VOCAB = 151936
# Code-related special tokens (in the StarCoder style) that Qwen3 lacks. They keep ids
# 151669-151680 in this order in every variant, so that the embedding rows of the trained
# checkpoints still match them.
ORBIT_SPECIALS = (
    "<file_sep>", "<filename>", "<gh_stars>", "<issue_start>", "<issue_comment>", "<issue_closed>",
    "<jupyter_start>", "<jupyter_text>", "<jupyter_code>", "<jupyter_output>", "<jupyter_script>",
    "<empty_output>",
)


@dataclass(frozen=True)
class Variant:
    base: str
    revision: str
    # None keeps the base model's own chat template; otherwise a file in templates/.
    template: str = None


VARIANTS = {
    "tokenizer_qwen3_orbit": Variant("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"),
    "tokenizer_qwen3_4b_qwen35_think": Variant(
        "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "qwen35_think.jinja"),
    "tokenizer_qwen3_4b_qwen35_tools": Variant(
        "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "qwen35_system_tools.jinja"),
}


def load_base(variant, offline):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(variant.base, revision=variant.revision, local_files_only=offline)
    if len(tokenizer) != QWEN3_TOKENIZER_LENGTH:
        raise SystemExit(f"{variant.base} has {len(tokenizer)} tokens, expected {QWEN3_TOKENIZER_LENGTH}")
    return tokenizer


def add_orbit_specials(tokenizer):
    """Appends the OrbitLM specials that the base lacks, as additional special tokens."""
    missing = [token for token in ORBIT_SPECIALS if token not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    if len(tokenizer) > DECLARED_VOCAB:
        raise SystemExit(f"tokenizer length {len(tokenizer)} exceeds the model vocab {DECLARED_VOCAB}")


def write_pretrain(variant, out, offline):
    """Writes the pretraining tokenizer.

    The base's own chat template stays. Only tokenizer_config.json is edited after saving, so
    special_tokens_map.json keeps the base's entries, which already name <|endoftext|> as eos
    and pad for a Base model. bos is set to <|endoftext|> in the config only. The config is
    rewritten without sorting and without a final newline, which is how this directory was
    first written.
    """
    tokenizer = load_base(variant, offline)
    add_orbit_specials(tokenizer)
    tokenizer.save_pretrained(str(out))
    config_path = out / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["vocab_size"] = DECLARED_VOCAB
    config["eos_token"], config["pad_token"], config["bos_token"] = END_OF_TEXT, END_OF_TEXT, END_OF_TEXT
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def write_sft(variant, out, offline):
    """Writes an SFT tokenizer: the base vocabulary, the OrbitLM specials and the variant's template.

    eos and pad are set before saving, so special_tokens_map.json names <|endoftext|> too. The
    template is also stored inside tokenizer_config.json, because some readers look only there.
    """
    tokenizer = load_base(variant, offline)
    add_orbit_specials(tokenizer)
    tokenizer.chat_template = (HERE / "templates" / variant.template).read_text(encoding="utf-8")
    tokenizer.eos_token = END_OF_TEXT
    tokenizer.pad_token = END_OF_TEXT
    tokenizer.save_pretrained(str(out))
    config_path = out / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["vocab_size"] = DECLARED_VOCAB
    config["eos_token"] = END_OF_TEXT
    config["pad_token"] = END_OF_TEXT
    config["chat_template"] = tokenizer.chat_template
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "chat_template.jinja").write_text(tokenizer.chat_template, encoding="utf-8")


def check_saved(out):
    """Reloads a written directory and checks the ids the trainer depends on. Returns its vocab."""
    from transformers import AutoTokenizer

    saved = AutoTokenizer.from_pretrained(str(out), local_files_only=True)
    if saved.eos_token != END_OF_TEXT or saved.eos_token_id != END_OF_TEXT_ID:
        raise SystemExit(f"{out}: eos must be {END_OF_TEXT} id {END_OF_TEXT_ID}, got {saved.eos_token} {saved.eos_token_id}")
    if saved.pad_token_id != END_OF_TEXT_ID:
        raise SystemExit(f"{out}: pad id {saved.pad_token_id}, expected {END_OF_TEXT_ID}")
    for offset, token in enumerate(ORBIT_SPECIALS):
        if saved.convert_tokens_to_ids(token) != QWEN3_TOKENIZER_LENGTH + offset:
            raise SystemExit(f"{out}: {token} is not at id {QWEN3_TOKENIZER_LENGTH + offset}")
    return saved.get_vocab()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True, help="the directories are written here, one per variant")
    parser.add_argument("--only", action="append", choices=sorted(VARIANTS), help="write only this variant (repeatable)")
    parser.add_argument("--offline", action="store_true", help="read the base tokenizers from the local Hugging Face cache only")
    args = parser.parse_args()
    vocabs = {}
    for name in args.only or list(VARIANTS):
        variant = VARIANTS[name]
        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        if variant.template is None:
            write_pretrain(variant, out, args.offline)
        else:
            write_sft(variant, out, args.offline)
        vocabs[name] = check_saved(out)
        print(f"wrote {out} tokens={len(vocabs[name])} eos={END_OF_TEXT_ID}")
    # The SFT run starts from the pretrained embedding table, so every variant must map the same
    # strings to the same ids.
    first = next(iter(vocabs.values()))
    for name, vocab in vocabs.items():
        if vocab != first:
            raise SystemExit(f"{name} has a different vocabulary from the other variants")


if __name__ == "__main__":
    main()
