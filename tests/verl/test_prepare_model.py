"""CPU tests of orbit_verl.model.prepare_model and the hf/compare phases of parity_logprobs.

Most tests build a fake export from a tiny ported model with the Qwen3 vocabulary, saved with
save_pretrained, plus the tokenizer files of a real SFT export, then prepare a model dir from it and
check it the way verl and vLLM will load it. They need ORBIT_TEST_SFT_EXPORT (conftest.py) and skip
without it.

The frozen-embedding-row detection is also tested on its own, on minimal exports that hold only
config.json, the model code and the embedding tensor. Those tests need no tokenizer.

Run with the verl python:
    PYTHONPATH=<repo>/src $PY -m pytest -q -p no:cacheprovider tests/verl/test_prepare_model.py
"""
from orbit_verl import common

# The cache variables must be set before transformers is imported, because transformers reads
# its module-cache location once, at import time.
common.apply_cache_env()

import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from orbit_verl.model import MODEL_CODE_DIR, MODEL_CODE_FILES, PRESETS, preset_for_vocab  # noqa: E402
from orbit_verl.model import configuration_ouro, modelling_ouro, parity_logprobs, prepare_model  # noqa: E402

QWEN3_VOCAB = 151936
# The vocabulary of an old model line that the release no longer prepares; it has no preset.
OLD_VOCAB = 49152
LOOPS = 4
# The Qwen3 ids that the qwen3 preset writes into generation_config.json.
IM_END, END_OF_TEXT = 151645, 151643

# Export code is parsed and must never run, so each fake export's modelling_ouro.py raises on import.
NEVER_IMPORT = "raise RuntimeError('the export code must not be loaded')\n"

# An embedding call that passes padding_idx from config.pad_token_id as the third positional argument.
# With pad_token_id 151643 and a trained row 151643, the code and the weights contradict each other.
PADDED_CODE = NEVER_IMPORT + '''
class OuroModel(OuroPreTrainedModel):
    def __init__(self, config: OuroConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
'''

# The release line still assigns self.padding_idx but builds the embedding without it.
RELEASE_CODE = NEVER_IMPORT + '''
class OuroModel(OuroPreTrainedModel):
    def __init__(self, config: OuroConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # NOT padding_idx=...: pad_token_id is a real token in the corpus.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
'''


def tiny_model(seed: int = 0, *, padded: bool = False) -> modelling_ouro.OuroForCausalLM:
    """A tiny Ouro model with the Qwen3 vocabulary and the special-token ids of the release exports.

    By default no row is frozen and no row is zero, as in the release line. ``padded`` gives it a
    frozen row 2 that is exactly zero, and config.pad_token_id 2 so that PADDED_CODE names that row;
    the port still supports such a model.
    """
    config = configuration_ouro.OuroConfig(
        vocab_size=QWEN3_VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=8, max_position_embeddings=4096,
        rope_theta=40000.0, total_ut_steps=LOOPS, early_exit_threshold=1.0, use_exit_gate=True,
        bos_token_id=END_OF_TEXT, eos_token_id=IM_END, pad_token_id=2 if padded else END_OF_TEXT,
        tie_word_embeddings=False, embedding_padding_idx=2 if padded else None,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    model = modelling_ouro.OuroForCausalLM(config).eval()
    with torch.no_grad():
        # Spread the logits so greedy argmax is stable against rounding noise.
        model.lm_head.weight.mul_(20.0)
        if padded:
            # Training keeps the frozen row at exactly zero; do not rely on the init doing it.
            model.model.embed_tokens.weight[2].zero_()
    return model


def write_export(export: Path, model: modelling_ouro.OuroForCausalLM, code: str, tokenizer_dir: Path) -> Path:
    """Lay out an export like the real ones: sharded safetensors, tokenizer links, foreign code."""
    export.mkdir(parents=True, exist_ok=True)
    # A small shard size forces a sharded checkpoint with model.safetensors.index.json.
    model.save_pretrained(export, max_shard_size="10MB")
    config = json.loads((export / "config.json").read_text())
    config["auto_map"] = dict(prepare_model.AUTO_MAP)
    # Real exports come from the trainer's code, whose config has no embedding_padding_idx.
    config.pop("embedding_padding_idx", None)
    (export / "config.json").write_text(json.dumps(config, indent=2))
    for name in prepare_model.TOKENIZER_FILES:
        if (tokenizer_dir / name).exists():
            os.symlink(tokenizer_dir / name, export / name)
    (export / "modelling_ouro.py").write_text(code)
    (export / "experiment_config.py").write_text(NEVER_IMPORT)
    return export


@pytest.fixture(scope="module")
def tokenizer_dir(sft_export) -> Path:
    """The SFT export whose Qwen3 tokenizer files the fake exports link."""
    return sft_export


@pytest.fixture(scope="module")
def fake_export(tmp_path_factory, tokenizer_dir):
    """An export of the release line: no padding_idx in the code and no zero row in the weights."""
    model = tiny_model()
    return write_export(tmp_path_factory.mktemp("export"), model, RELEASE_CODE, tokenizer_dir), model


@pytest.fixture(scope="module")
def padded_export(tmp_path_factory, tokenizer_dir):
    """An export whose code freezes row 2 and whose row 2 is exactly zero."""
    model = tiny_model(seed=2, padded=True)
    return write_export(tmp_path_factory.mktemp("padded_export"), model, PADDED_CODE, tokenizer_dir), model


@pytest.fixture(scope="module")
def unpadded_model():
    return tiny_model(seed=1)


@pytest.fixture()
def prepared(fake_export, tmp_path):
    export, model = fake_export
    out = tmp_path / "models" / "tiny"
    manifest = prepare_model.prepare_model_dir(str(export), str(out))
    return export, model, out, manifest


def mini_export(
    path: Path,
    code: str | None,
    *,
    vocab: int = QWEN3_VOCAB,
    zero_rows: tuple[int, ...] = (),
    config: dict | None = None,
    export_info: dict | None = None,
    sharded: bool = False,
    embed_key: str = prepare_model.EMBED_WEIGHT,
) -> Path:
    """An export with only what the padding detection reads: config, code, embedding, export_info."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "ouro", "vocab_size": vocab, **(config or {})}))
    if code is not None:
        (path / "modelling_ouro.py").write_text(code)
    if export_info is not None:
        (path / "export_info.json").write_text(json.dumps(
            {"checkpoint_metadata": {"model_config_overrides": export_info}}))
    embed = torch.randn(vocab, 4, generator=torch.Generator().manual_seed(0)) + 0.5
    for row in zero_rows:
        embed[row] = 0.0
    tensors = {embed_key: embed, "lm_head.weight": torch.randn(vocab, 4)}
    if sharded:
        save_file({"lm_head.weight": tensors.pop("lm_head.weight")}, str(path / "model-00001-of-00002.safetensors"))
        save_file(tensors, str(path / "model-00002-of-00002.safetensors"))
        weight_map = {"lm_head.weight": "model-00001-of-00002.safetensors",
                      embed_key: "model-00002-of-00002.safetensors"}
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file(tensors, str(path / "model.safetensors"))
    return path


def test_model_package_import_is_cheap_and_presets_resolve():
    # The launcher imports prepare_model for its pre-flight checks, so that must stay cheap too.
    code = ("import sys, orbit_verl.model, orbit_verl.model.prepare_model; "
            "print('torch' in sys.modules, 'transformers' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["False", "False"]
    assert preset_for_vocab(151936) == "qwen3"
    # The release has one tokenizer family; any other vocabulary has no preset.
    assert sorted(PRESETS) == ["qwen3"]
    for vocab in (OLD_VOCAB, 1000):
        with pytest.raises(ValueError, match="No model preset"):
            preset_for_vocab(vocab)
    assert PRESETS["qwen3"] == {"vocab_size": QWEN3_VOCAB, "eos_token_ids": [IM_END, END_OF_TEXT],
                                "pad_token_id": END_OF_TEXT, "bos_token_id": None}
    assert prepare_model.LINEAGES == ("release_qwen3", "unknown")


def test_an_export_without_a_preset_needs_explicit_token_ids():
    with pytest.raises(ValueError, match="No model preset"):
        prepare_model._resolve_token_ids("auto", OLD_VOCAB, None, None, None)
    with pytest.raises(ValueError, match="for vocab_size 151936"):
        prepare_model._resolve_token_ids("qwen3", OLD_VOCAB, None, None, None)
    name, ids = prepare_model._resolve_token_ids("qwen3", OLD_VOCAB, [0, 2], 0, "none")
    assert (name, ids) == ("qwen3", {"eos_token_ids": [0, 2], "pad_token_id": 0, "bos_token_id": None})
    with pytest.raises(ValueError, match="Unknown preset"):
        prepare_model._resolve_token_ids("fixed2", OLD_VOCAB, None, None, None)


def _init(body: str, embedding: str) -> str:
    return (
        "class OuroModel(OuroPreTrainedModel):\n"
        "    def __init__(self, config):\n"
        "        super().__init__(config)\n"
        + "".join(f"        {line}\n" for line in body.splitlines() if line)
        + f"        self.embed_tokens = {embedding}\n"
    )


@pytest.mark.parametrize(
    "source, expected",
    [
        (PADDED_CODE, ("config", "pad_token_id", "self.padding_idx")),
        (RELEASE_CODE, ("none", None, "")),
        (_init("", "nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)"),
         ("config", "pad_token_id", "config.pad_token_id")),
        (_init("", "torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.config.pad_token_id)"),
         ("config", "pad_token_id", "self.config.pad_token_id")),
        (_init("pad = config.pad_token_id", "nn.Embedding(config.vocab_size, config.hidden_size, pad)"),
         ("config", "pad_token_id", "pad")),
        (_init("", "nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=None)"), ("none", None, "None")),
        (_init("", "nn.Embedding(config.vocab_size, config.hidden_size, 7)"), ("literal", 7, "7")),
    ],
    ids=["padded", "release", "keyword", "self-config", "local-name", "literal-none", "literal-int"],
)
def test_code_padding_reads_the_embedding_call(tmp_path, source, expected):
    path = tmp_path / "modelling_ouro.py"
    path.write_text(source)
    assert prepare_model._code_padding(path) == expected


def test_code_padding_reads_our_own_port():
    """A verl checkpoint exported with the port's code must be detectable as well."""
    assert prepare_model._code_padding(MODEL_CODE_DIR / "modelling_ouro.py") == (
        "config", "embedding_padding_idx", "self.padding_idx")


@pytest.mark.parametrize(
    "source, reason",
    [
        (NEVER_IMPORT + "class Other:\n    pass\n", "OuroModel 0 times"),
        (_init("self.padding_idx = config.pad_token_id\nself.padding_idx = 0",
               "nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)"), "assigned 2 times"),
        (_init("", "nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=pick(config))"),
         "neither a literal"),
        (_init("", "MyEmbedding(config.vocab_size, config.hidden_size)"), "not by an Embedding call"),
        (_init("", "nn.Embedding(*sizes)"), "*args or **kwargs"),
        ("class OuroModel(:\n", "does not parse"),
        (None, "cannot read"),
    ],
    ids=["no-class", "two-assignments", "call", "custom-class", "star-args", "syntax", "missing-file"],
)
def test_code_padding_reports_unknown_patterns(tmp_path, source, reason):
    path = tmp_path / "modelling_ouro.py"
    if source is not None:
        path.write_text(source)
    kind, value, _ = prepare_model._code_padding(path)
    assert kind == "unknown" and reason in value


def test_detects_a_frozen_row(tmp_path, caplog):
    export = mini_export(tmp_path / "e", PADDED_CODE, zero_rows=(2,), config={"pad_token_id": 2},
                         export_info={"pad_token_id": "2"}, sharded=True)
    with caplog.at_level(logging.WARNING, logger=prepare_model.__name__):
        result = prepare_model.detect_embedding_padding(export)
    # A frozen row does not fit the release line, so the lineage is unknown.
    assert (result.embedding_padding_idx, result.lineage, result.source) == (2, "unknown", "detected")
    assert "lineage is recorded as unknown" in caplog.text
    evidence = result.evidence
    assert evidence["padding_from"] == "config.pad_token_id" and evidence["padding_expr"] == "self.padding_idx"
    assert (evidence["zero_rows"], evidence["zero_rows_count"], evidence["candidate_row"]) == ([2], 1, 2)
    assert evidence["export_info_value"] == 2 and evidence["candidate_row_norm"] == 0.0
    assert evidence["code_file_sha256"] == common.sha256_file(export / "modelling_ouro.py")
    json.dumps(evidence)  # it goes into the manifest


def test_detects_release_qwen3_line(tmp_path):
    export = mini_export(tmp_path / "e", RELEASE_CODE, vocab=QWEN3_VOCAB, config={"pad_token_id": 151643},
                         export_info={"pad_token_id": 151643})
    result = prepare_model.detect_embedding_padding(export)
    assert (result.embedding_padding_idx, result.lineage, result.source) == (None, "release_qwen3", "detected")
    assert result.evidence["zero_rows_count"] == 0 and result.evidence["padding_from"] == "none"


def test_unusual_vocab_or_extra_zero_rows_only_warn(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=prepare_model.__name__):
        old_vocab = prepare_model.detect_embedding_padding(mini_export(tmp_path / "a", RELEASE_CODE, vocab=OLD_VOCAB))
        extra = prepare_model.detect_embedding_padding(
            mini_export(tmp_path / "b", PADDED_CODE, zero_rows=(2, 5), config={"pad_token_id": 2}))
        # With a null pad, padding code passes padding_idx=None, which freezes nothing.
        null_pad = prepare_model.detect_embedding_padding(
            mini_export(tmp_path / "c", PADDED_CODE, config={"pad_token_id": None}))
    assert (old_vocab.embedding_padding_idx, old_vocab.lineage) == (None, "unknown")
    assert (extra.embedding_padding_idx, extra.lineage) == (2, "unknown")
    # No frozen row and the Qwen3 vocabulary: that is the release line, whatever the code looks like.
    assert (null_pad.embedding_padding_idx, null_pad.lineage) == (None, "release_qwen3")
    assert "lineage is recorded as unknown" in caplog.text
    assert "other rows are all zero" in caplog.text


@pytest.mark.parametrize(
    "kwargs, message",
    [
        # Padded code, but row 2 was trained: the code is not what trained this model.
        (dict(code=PADDED_CODE, config={"pad_token_id": 2}), "row is not zero"),
        # Unpadded code, but a row is exactly zero: only a frozen row stays zero.
        (dict(code=RELEASE_CODE, zero_rows=(7,)), "rows [7] are all zero"),
        (dict(code=PADDED_CODE, zero_rows=(2,), config={"pad_token_id": 2}, export_info={"pad_token_id": 0}),
         "export_info.json records 0"),
        (dict(code=PADDED_CODE, config={"pad_token_id": QWEN3_VOCAB + 5}), "outside vocab_size"),
        (dict(code=PADDED_CODE, zero_rows=(2,), config={"pad_token_id": "2"}), "config.json has pad_token_id='2'"),
        (dict(code=NEVER_IMPORT, zero_rows=(2,)), "All-zero embedding rows in the weights: [2]"),
        (dict(code=None), "cannot read modelling_ouro.py"),
        (dict(code=RELEASE_CODE, embed_key="model.embed.weight"), "Cannot read the token embedding"),
        (dict(code=RELEASE_CODE, vocab=1000, config={"vocab_size": 999}), "has 1000 rows"),
    ],
    ids=["padded-code-trained-row", "unpadded-code-zero-row", "export-info", "out-of-range", "string-pad",
         "unknown-code", "no-code", "no-embedding", "vocab-mismatch"],
)
def test_contradictions_are_refused(tmp_path, kwargs, message):
    code = kwargs.pop("code")
    with pytest.raises(prepare_model.PaddingDetectionError) as info:
        prepare_model.detect_embedding_padding(mini_export(tmp_path / "e", code, **kwargs))
    assert message in str(info.value)
    assert "--embedding-padding-idx" in str(info.value)


def test_override_decides_by_hand_and_keeps_the_evidence(tmp_path, caplog):
    contradicting = mini_export(tmp_path / "a", PADDED_CODE, config={"pad_token_id": 2})
    result = prepare_model.detect_embedding_padding(contradicting, override="none")
    assert (result.embedding_padding_idx, result.lineage, result.source) == (None, "unknown", "override")
    assert "row is not zero" in result.evidence["detection_error"]
    # A hand override may only name a row that training kept at zero. Row 5 holds trained values, so
    # freezing it is refused outright instead of being logged.
    with pytest.raises(prepare_model.PaddingDetectionError) as info:
        prepare_model.detect_embedding_padding(contradicting, override=5)
    assert "not zero in the weights" in str(info.value)

    padded = mini_export(tmp_path / "b", PADDED_CODE, zero_rows=(2,), config={"pad_token_id": 2})
    agreeing = prepare_model.detect_embedding_padding(padded, override="2")
    assert (agreeing.embedding_padding_idx, agreeing.lineage, agreeing.source) == (2, "unknown", "override")
    assert "detection_error" not in agreeing.evidence
    # None means "detect", like the other prepare_model arguments that are left unset.
    assert prepare_model.detect_embedding_padding(padded, override=None).source == "detected"
    with pytest.raises(ValueError, match="below vocab_size"):
        prepare_model.detect_embedding_padding(padded, override=QWEN3_VOCAB)
    for bad in ("-1", "two", True):
        with pytest.raises(ValueError, match="'auto', 'none' or a row index"):
            prepare_model.parse_padding_choice(bad)


def test_a_trained_padding_row_needs_the_override(tmp_path):
    """Code that passes padding_idx=config.pad_token_id (151643) while that row was trained is a
    contradiction. Detection refuses such an export, and --embedding-padding-idx none prepares it with
    the lineage unknown, which the release configs accept."""
    export = mini_export(tmp_path / "sft", PADDED_CODE, config={"pad_token_id": END_OF_TEXT},
                         export_info={"pad_token_id": END_OF_TEXT})
    with pytest.raises(prepare_model.PaddingDetectionError, match="row is not zero"):
        prepare_model.detect_embedding_padding(export)
    result = prepare_model.detect_embedding_padding(export, override="none")
    assert (result.embedding_padding_idx, result.lineage, result.source) == (None, "unknown", "override")


def test_export_without_rope_theta_is_refused(fake_export, tmp_path):
    """Without rope_theta the port would fall back to 10000, so prepare_model must refuse the export."""
    import shutil

    export, _ = fake_export
    broken = tmp_path / "export_no_rope"
    shutil.copytree(export, broken, symlinks=True)
    config = json.loads((broken / "config.json").read_text())
    config.pop("rope_theta", None)
    config.pop("rope_parameters", None)
    (broken / "config.json").write_text(json.dumps(config))
    out = tmp_path / "models" / "no_rope"
    with pytest.raises(ValueError, match="no rope_theta"):
        prepare_model.prepare_model_dir(str(broken), str(out))
    assert not out.exists()


def test_prepared_dir_layout_config_and_manifest(prepared, tokenizer_dir):
    export, _, out, manifest = prepared
    assert common.read_manifest(out / common.MODEL_MANIFEST, common.MODEL_MANIFEST_KIND) == manifest
    expected_keys = {
        "export_dir", "tokenizer_dir", "preset", "eos_token_ids", "pad_token_id", "bos_token_id",
        "vocab_size", "total_ut_steps", "max_position_embeddings", "config_overrides", "weights",
        "code", "code_sha256", "tokenizer_sha256", "lineage", "embedding_padding_idx",
        "embedding_padding_source", "embedding_padding_evidence", "rope_theta", "rope_scaling",
        "export_max_position_embeddings",
    }
    assert expected_keys <= manifest.keys()
    assert manifest["preset"] == "qwen3"
    assert (manifest["eos_token_ids"], manifest["pad_token_id"], manifest["bos_token_id"]) == (
        [IM_END, END_OF_TEXT], END_OF_TEXT, None)
    assert (manifest["vocab_size"], manifest["total_ut_steps"], manifest["max_position_embeddings"]) == (
        QWEN3_VOCAB, LOOPS, 4096)
    assert (manifest["weights"], manifest["code"]) == ("symlink", "copy")
    assert manifest["export_dir"] == str(export.resolve())
    assert manifest["tokenizer_sha256"] == common.tokenizer_sha256(tokenizer_dir)
    assert (manifest["lineage"], manifest["embedding_padding_idx"], manifest["embedding_padding_source"]) == (
        "release_qwen3", None, "detected")
    assert manifest["embedding_padding_evidence"]["zero_rows"] == []
    assert (manifest["rope_theta"], manifest["rope_scaling"], manifest["export_max_position_embeddings"]) == (
        40000.0, None, 4096)

    # Weights: absolute symlinks into the export, including every shard named by the index.
    index = json.loads((export / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) > 1
    for name in ["model.safetensors.index.json", *shards]:
        assert (out / name).is_symlink()
        target = os.readlink(out / name)
        assert os.path.isabs(target) and Path(target).resolve() == (export / name).resolve()
    # Tokenizer: symlinks for exactly the files the tokenizer dir has.
    linked = sorted(p.name for p in out.iterdir() if p.name in prepare_model.TOKENIZER_FILES)
    assert linked == sorted(n for n in prepare_model.TOKENIZER_FILES if (tokenizer_dir / n).exists())
    assert all((out / n).is_symlink() for n in linked)
    # Code: plain copies of the port, never the export's own code.
    for name in MODEL_CODE_FILES:
        assert not (out / name).is_symlink()
        assert (out / name).read_bytes() == (MODEL_CODE_DIR / name).read_bytes()
        assert manifest["code_sha256"][name] == common.sha256_file(MODEL_CODE_DIR / name)
    assert not (out / "experiment_config.py").exists()

    config = json.loads((out / "config.json").read_text())
    source = json.loads((export / "config.json").read_text())
    assert config["early_exit_threshold"] is None
    assert config["auto_map"] == prepare_model.AUTO_MAP
    assert config["embedding_padding_idx"] is None and "embedding_padding_idx" not in source
    for key in ("eos_token_id", "pad_token_id", "bos_token_id", "max_position_embeddings", "vocab_size", "rope_theta"):
        assert config.get(key) == source.get(key), key
    written = ("early_exit_threshold", "auto_map", "embedding_padding_idx")
    assert {k: v for k, v in config.items() if k not in written} == {k: v for k, v in source.items() if k not in written}
    generation = json.loads((out / "generation_config.json").read_text())
    # The qwen3 preset has no bos id, so none is written.
    assert generation == {"eos_token_id": [IM_END, END_OF_TEXT], "pad_token_id": END_OF_TEXT}
    # Nothing is left behind next to the output: no temporary or replaced dirs.
    assert sorted(p.name for p in out.parent.iterdir()) == ["tiny"]


def test_force_only_replaces_dirs_with_our_manifest(prepared, tmp_path, caplog):
    export, _, out, _ = prepared
    with pytest.raises(FileExistsError, match="--force"):
        prepare_model.prepare_model_dir(str(export), str(out))
    with caplog.at_level(logging.WARNING, logger=prepare_model.__name__):
        manifest = prepare_model.prepare_model_dir(str(export), str(out), force=True, max_position_embeddings=8192)
    assert "above the export's own 4096" in caplog.text
    assert manifest["max_position_embeddings"] == 8192
    assert manifest["export_max_position_embeddings"] == 4096
    assert manifest["config_overrides"]["max_position_embeddings"] == 8192
    assert json.loads((out / "config.json").read_text())["max_position_embeddings"] == 8192
    assert sorted(p.name for p in out.parent.iterdir()) == ["tiny"]
    # The export behind the replaced dir's symlinks is untouched.
    assert (export / "model.safetensors.index.json").exists()
    # The export's own length is not "above" anything.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=prepare_model.__name__):
        prepare_model.prepare_model_dir(str(export), str(out), force=True, max_position_embeddings=4096)
    assert "above" not in caplog.text

    foreign = tmp_path / "models" / "foreign"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("not ours")
    with pytest.raises(FileExistsError, match="no orbit_prepare.json"):
        prepare_model.prepare_model_dir(str(export), str(foreign), force=True)
    assert (foreign / "keep.txt").read_text() == "not ours"
    assert sorted(p.name for p in foreign.iterdir()) == ["keep.txt"]


def test_materialized_weights_and_linked_code(fake_export, tmp_path):
    export, _ = fake_export
    out = tmp_path / "copy"
    manifest = prepare_model.prepare_model_dir(str(export), str(out), materialize_weights=True, link_code=True)
    assert (manifest["weights"], manifest["code"]) == ("copy", "symlink")
    for name in prepare_model._weight_files(export):
        assert not (out / name).is_symlink()
        assert (out / name).read_bytes() == (export / name).read_bytes()
    for name in MODEL_CODE_FILES:
        assert (out / name).is_symlink()
        assert Path(os.readlink(out / name)) == MODEL_CODE_DIR / name


def test_presets_and_token_id_overrides(fake_export, tmp_path):
    export, _ = fake_export
    with pytest.raises(ValueError, match="Unknown preset"):
        prepare_model.prepare_model_dir(str(export), str(tmp_path / "a"), preset="fixed2")
    with pytest.raises(ValueError, match="below vocab_size"):
        prepare_model.prepare_model_dir(str(export), str(tmp_path / "b"), eos_token_ids=[IM_END, QWEN3_VOCAB])
    manifest = prepare_model.prepare_model_dir(
        str(export), str(tmp_path / "c"), eos_token_ids=[IM_END], pad_token_id=END_OF_TEXT, bos_token_id=END_OF_TEXT
    )
    assert (manifest["eos_token_ids"], manifest["pad_token_id"], manifest["bos_token_id"]) == (
        [IM_END], END_OF_TEXT, END_OF_TEXT)
    generation = json.loads((tmp_path / "c" / "generation_config.json").read_text())
    assert generation == {"eos_token_id": [IM_END], "pad_token_id": END_OF_TEXT, "bos_token_id": END_OF_TEXT}
    # Failed attempts leave nothing behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c"]


def test_refused_export_leaves_nothing_and_override_builds_it(unpadded_model, tokenizer_dir, tmp_path, capsys):
    """Padding code with a trained row 151643 is a contradiction; --embedding-padding-idx none decides it."""
    export = write_export(tmp_path / "export", unpadded_model, PADDED_CODE, tokenizer_dir)
    models = tmp_path / "models"
    with pytest.raises(prepare_model.PaddingDetectionError, match="row is not zero"):
        prepare_model.prepare_model_dir(str(export), str(models / "refused"))
    assert not models.exists() or not any(models.iterdir())

    assert prepare_model.main(["--export", str(export), "--out-dir", str(models / "cli")]) == 2
    assert "--embedding-padding-idx" in capsys.readouterr().err
    assert prepare_model.main(["--export", str(export), "--out-dir", str(models / "cli"),
                               "--embedding-padding-idx", "none", "--check", "--no-vllm-check"]) == 0
    assert "[check] OK" in capsys.readouterr().out
    manifest = common.read_manifest(models / "cli" / common.MODEL_MANIFEST, common.MODEL_MANIFEST_KIND)
    assert (manifest["embedding_padding_idx"], manifest["lineage"], manifest["embedding_padding_source"]) == (
        None, "unknown", "override")
    assert "row is not zero" in manifest["embedding_padding_evidence"]["detection_error"]
    assert json.loads((models / "cli" / "config.json").read_text())["embedding_padding_idx"] is None


def test_padded_export_loads_with_its_frozen_row(padded_export, tmp_path, caplog):
    """The port still freezes a row that an export's training froze, although no release model has one."""
    export, model = padded_export
    out = tmp_path / "padded"
    with caplog.at_level(logging.WARNING, logger=prepare_model.__name__):
        manifest = prepare_model.prepare_model_dir(str(export), str(out))
    assert (manifest["embedding_padding_idx"], manifest["lineage"], manifest["embedding_padding_source"]) == (
        2, "unknown", "detected")
    assert json.loads((out / "config.json").read_text())["embedding_padding_idx"] == 2
    summary = prepare_model.check_model_dir(str(out), load_weights=True, vllm_view=False)
    assert summary["loaded_padding_idx"] == 2 and summary["padding_row_zero"]
    # A config.json edited after preparing no longer matches the manifest's record.
    _rewrite_json(out / "config.json", embedding_padding_idx=5)
    with pytest.raises(RuntimeError, match="the manifest recorded 2"):
        prepare_model.check_model_dir(str(out), vllm_view=False)


def test_check_model_dir_loads_weights_like_verl(prepared):
    _, model, out, _ = prepared
    summary = prepare_model.check_model_dir(str(out), load_weights=True)
    assert summary["config_class"] == "OuroConfig"
    assert summary["early_exit_threshold"] is None
    assert (summary["missing_keys"], summary["unexpected_keys"], summary["mismatched_keys"]) == (0, 0, 0)
    assert summary["generation_eos"] == [IM_END, END_OF_TEXT]
    assert sorted(summary["vllm_eos"]) == [END_OF_TEXT, IM_END] or str(summary["vllm_eos"]).startswith("skipped")
    assert summary["forward_finite"] and summary["forward_shape"] == [1, 8, QWEN3_VOCAB]
    assert (summary["lineage"], summary["embedding_padding_idx"]) == ("release_qwen3", None)
    assert summary["loaded_padding_idx"] is None and "padding_row_zero" not in summary
    assert summary["rope_theta"] == 40000.0
    if "vllm_rope_theta" in summary:
        assert (summary["vllm_rope_theta"], summary["vllm_max_position_embeddings"]) == (40000.0, 4096)

    # The remote code copied into the dir reproduces the in-memory model on the final loop.
    from transformers import AutoModelForCausalLM

    loaded = AutoModelForCausalLM.from_pretrained(str(out), trust_remote_code=True, attn_implementation="sdpa").eval()
    assert type(loaded).__module__ != modelling_ouro.__name__
    assert loaded.model.embed_tokens.padding_idx is None
    ids = torch.randint(3, QWEN3_VOCAB, (1, 12), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False, exit_at_step=LOOPS - 1).logits
        actual = loaded(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_check_model_dir_reports_problems(prepared):
    _, _, out, _ = prepared
    (out / "generation_config.json").write_text(json.dumps({"eos_token_id": [IM_END], "pad_token_id": END_OF_TEXT}))
    with pytest.raises(RuntimeError, match="generation eos"):
        prepare_model.check_model_dir(str(out), vllm_view=False)


def _rewrite_json(path: Path, **changes) -> None:
    """Apply changes to a JSON file; a value of ... deletes the key."""
    data = json.loads(path.read_text())
    for key, value in changes.items():
        if value is ...:
            data.pop(key, None)
        else:
            data[key] = value
    path.write_text(json.dumps(data))


def test_dir_prepared_before_padding_detection_is_refused(prepared):
    """A dir without the frozen-row record: no embedding_padding_idx in config.json, no lineage in the manifest."""
    _, _, out, _ = prepared
    _rewrite_json(out / "config.json", embedding_padding_idx=...)
    _rewrite_json(out / common.MODEL_MANIFEST, embedding_padding_idx=..., lineage=...)
    manifest = common.read_manifest(out / common.MODEL_MANIFEST, common.MODEL_MANIFEST_KIND)
    problems = prepare_model.recorded_padding_problems(out, manifest)
    assert len(problems) == 1 and "without a record of the frozen embedding row" in problems[0]
    assert "--force" in problems[0]
    with pytest.raises(RuntimeError, match="without a record of the frozen embedding row"):
        prepare_model.check_model_dir(str(out), vllm_view=False)


def test_edited_padding_row_is_refused(prepared):
    _, _, out, _ = prepared
    _rewrite_json(out / "config.json", embedding_padding_idx=5)
    with pytest.raises(RuntimeError, match="the manifest recorded None"):
        prepare_model.check_model_dir(str(out), vllm_view=False)


def test_cli_prepare_and_check(fake_export, tmp_path, capsys):
    export, _ = fake_export
    out = tmp_path / "cli"
    assert prepare_model.main(["--export", str(export), "--out-dir", str(out), "--check", "--no-vllm-check"]) == 0
    printed = capsys.readouterr().out
    assert "[check] OK" in printed
    assert "[prepare]   lineage: release_qwen3" in printed and "[prepare]   embedding_padding_idx: None" in printed
    # Without --force an existing dir is an error; checking alone needs no --export.
    assert prepare_model.main(["--export", str(export), "--out-dir", str(out)]) == 2
    assert prepare_model.main(["--out-dir", str(out), "--check", "--no-vllm-check"]) == 0
    assert prepare_model.main(["--out-dir", str(out)]) == 2


def _greedy_with_logprobs(model, prompt_ids: list[int], steps: int) -> tuple[list[int], list[float]]:
    ids, logprobs = list(prompt_ids), []
    with torch.no_grad():
        for _ in range(steps):
            logits = model(input_ids=torch.tensor([ids]), use_cache=False, exit_at_step=LOOPS - 1).logits[0, -1]
            log_softmax = logits.float().log_softmax(-1)
            token = int(log_softmax.argmax())
            ids.append(token)
            logprobs.append(float(log_softmax[token]))
    return ids[len(prompt_ids):], logprobs


def test_parity_hf_and_compare_phases_on_cpu(prepared, tmp_path):
    """A synthetic vllm.json made from the model itself must pass; shifted log-probs must fail."""
    _, model, out, _ = prepared
    from transformers import AutoTokenizer

    # On transformers 5 the tokenizer load also reads the config's auto_map, which needs trust_remote_code.
    tokenizer = AutoTokenizer.from_pretrained(str(out), trust_remote_code=True)
    records = []
    for index, text in enumerate(["Question: What is 2 + 3?\nAnswer:", "Problem: Compute 7 * 6.\nSolution:"]):
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        completion, logprobs = _greedy_with_logprobs(model, prompt_ids, 5)
        records.append({"index": index, "source": "test", "prompt_ids": prompt_ids, "completion_ids": completion,
                        "completion_logprobs": logprobs, "finish_reason": "length", "text": tokenizer.decode(completion)})
    parity_dir = tmp_path / "parity"
    parity_dir.mkdir()
    (parity_dir / "vllm.json").write_text(json.dumps({"model_dir": str(out), "prompts": records}))

    hf_args = ["--model-dir", str(out), "--out-dir", str(parity_dir), "--device", "cpu", "--dtype", "float32", "--attn", "sdpa"]
    assert parity_logprobs.main(["--phase", "hf", *hf_args]) == 0
    assert parity_logprobs.main(["--phase", "compare", "--out-dir", str(parity_dir)]) == 0
    report = json.loads((parity_dir / "report.json").read_text())
    assert report["passed"] and report["tokens"] == 10
    assert report["mean_abs_dlogp"] < 1e-4 and report["greedy_agreement"] == 1.0
    assert report["max_abs_packed_vs_padded"] < 1e-4

    for record in records:
        record["completion_logprobs"] = [lp - 1.0 for lp in record["completion_logprobs"]]
    (parity_dir / "vllm.json").write_text(json.dumps({"model_dir": str(out), "prompts": records}))
    assert parity_logprobs.main(["--phase", "compare", "--out-dir", str(parity_dir)]) == 1
    assert not json.loads((parity_dir / "report.json").read_text())["passed"]


def test_real_sft_export_detects_the_release_line(sft_export):
    """Detection only, on the real SFT export of the tests, without loading the model. The scan reads
    its 151,936 embedding rows in about 2 s. The export's code builds the embedding without
    padding_idx and no row is zero, so it is the release line, and --embedding-padding-idx none agrees
    with the detection."""
    result = prepare_model.detect_embedding_padding(sft_export)
    assert (result.embedding_padding_idx, result.lineage, result.source) == (None, "release_qwen3", "detected")
    assert result.evidence["embedding_shape"][0] == QWEN3_VOCAB and result.evidence["zero_rows_count"] == 0
    override = prepare_model.detect_embedding_padding(sft_export, override="none")
    assert (override.embedding_padding_idx, override.lineage, override.source) == (None, "release_qwen3", "override")
    assert "detection_error" not in override.evidence
