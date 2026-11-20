"""Checks of the three release tokenizers that prepare_tokenizers.py writes (via the tokenizer_root fixture)."""
import json
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
NAMES = ("tokenizer_qwen3_orbit", "tokenizer_qwen3_4b_qwen35_think", "tokenizer_qwen3_4b_qwen35_tools")


def test_release_tokenizers_share_one_vocabulary_and_end_documents_with_endoftext(tokenizer_root):
    from transformers import AutoTokenizer

    vocab = None
    for name in NAMES:
        tok = AutoTokenizer.from_pretrained(str(tokenizer_root / name))
        assert len(tok) == 151681
        assert tok.eos_token_id == 151643 and tok.pad_token_id == 151643
        assert tok.convert_tokens_to_ids("<think>") == 151667
        assert tok.convert_tokens_to_ids("<file_sep>") == 151669
        assert tok.convert_tokens_to_ids("<empty_output>") == 151680
        assert json.loads((tokenizer_root / name / "tokenizer_config.json").read_text())["vocab_size"] == 151936
        if vocab is None:
            vocab = tok.get_vocab()
        assert tok.get_vocab() == vocab


def test_sft_tokenizers_ship_the_repository_templates(tokenizer_root):
    for name, template in (("tokenizer_qwen3_4b_qwen35_think", "qwen35_think.jinja"),
                           ("tokenizer_qwen3_4b_qwen35_tools", "qwen35_system_tools.jinja")):
        text = (TEMPLATES / template).read_text(encoding="utf-8")
        assert (tokenizer_root / name / "chat_template.jinja").read_text(encoding="utf-8") == text
        config = json.loads((tokenizer_root / name / "tokenizer_config.json").read_text(encoding="utf-8"))
        assert config["chat_template"] == text
        assert config["bos_token"] is None


def test_pretraining_tokenizer_keeps_the_base_template_and_sets_bos(tokenizer_root):
    config = json.loads((tokenizer_root / "tokenizer_qwen3_orbit" / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert config["bos_token"] == config["eos_token"] == config["pad_token"] == "<|endoftext|>"
    assert "chat_template" not in config
    assert (tokenizer_root / "tokenizer_qwen3_orbit" / "chat_template.jinja").is_file()
