"""The one-loop draft dir: config with total_ut_steps=1, every other file linked, never copied."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from inference.make_draft import DraftDirError, default_draft_dir, main, make_draft

CONFIG = {"model_type": "ouro", "total_ut_steps": 4, "hidden_size": 8, "max_position_embeddings": 32768}


@pytest.fixture
def model_dir(tmp_path):
    """A fake HF export: a config, two weight shards (one of them a symlink, as in a prepared model dir) and a tokenizer."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    (model / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 64)
    elsewhere = tmp_path / "store" / "shard2.safetensors"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"\1" * 64)
    (model / "model-00002-of-00002.safetensors").symlink_to(elsewhere)
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model


def test_the_draft_links_every_file_and_changes_only_the_loop_count(model_dir, tmp_path):
    draft = make_draft(model_dir, tmp_path / "draft")
    config = json.loads((draft / "config.json").read_text(encoding="utf-8"))
    assert config == {**CONFIG, "total_ut_steps": 1}
    assert not (draft / "config.json").is_symlink()
    names = sorted(entry.name for entry in draft.iterdir())
    assert names == sorted(entry.name for entry in model_dir.iterdir())
    for name in names:
        if name == "config.json":
            continue
        link = draft / name
        assert link.is_symlink(), f"{name} was copied"
        # Every link ends at the real file, also for a shard that is itself a link in the model dir.
        assert Path(os.readlink(link)) == (model_dir / name).resolve()
    assert (draft / "model-00002-of-00002.safetensors").read_bytes() == b"\1" * 64


def test_a_second_call_reuses_the_draft(model_dir, tmp_path):
    first = make_draft(model_dir, tmp_path / "draft")
    stamp = (first / "config.json").stat().st_mtime_ns
    assert make_draft(model_dir, tmp_path / "draft") == first
    assert (first / "config.json").stat().st_mtime_ns == stamp


def test_an_unfinished_draft_is_completed(model_dir, tmp_path):
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "tokenizer.json").symlink_to((model_dir / "tokenizer.json").resolve())
    make_draft(model_dir, draft)
    assert json.loads((draft / "config.json").read_text())["total_ut_steps"] == 1


@pytest.mark.parametrize("damage, message", [
    (lambda d, m: (d / "tokenizer.json").write_text("{}"), "is a copy"),
    (lambda d, m: (d / "tokenizer.json").symlink_to(m / "config.json"), "points to"),
    (lambda d, m: (d / "notes.txt").write_text("x"), "does not have"),
    (lambda d, m: (d / "config.json").write_text(json.dumps({**CONFIG, "total_ut_steps": 2})), "is not the model's config"),
])
def test_a_dir_that_is_not_a_draft_of_this_model_is_refused(model_dir, tmp_path, damage, message):
    draft = tmp_path / "draft"
    draft.mkdir()
    damage(draft, model_dir)
    with pytest.raises(DraftDirError, match=message):
        make_draft(model_dir, draft)


def test_bad_models_and_places_are_refused(model_dir, tmp_path):
    with pytest.raises(DraftDirError, match="inside it"):
        make_draft(model_dir, model_dir / "draft")
    with pytest.raises(DraftDirError, match="no config.json"):
        make_draft(tmp_path / "store", tmp_path / "d")
    dense = tmp_path / "dense"
    dense.mkdir()
    (dense / "config.json").write_text(json.dumps({**CONFIG, "total_ut_steps": 1}))
    with pytest.raises(DraftDirError, match="fewer loops than the model, but .* has total_ut_steps=1"):
        make_draft(dense, tmp_path / "d1")


def test_default_draft_dir_is_per_model_and_under_the_temp_dir(tmp_path):
    first, second = default_draft_dir(tmp_path / "a"), default_draft_dir(tmp_path / "b")
    assert first == default_draft_dir(tmp_path / "a") and first != second
    assert Path(tempfile.gettempdir()) in first.parents
    assert first.name.endswith("_loops1")


def test_command_line(model_dir, tmp_path, capsys):
    assert main([str(model_dir), str(tmp_path / "draft")]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "draft")
    with pytest.raises(SystemExit) as raised:
        main([str(model_dir), str(model_dir / "draft")])
    assert raised.value.code == 2
    assert "inside it" in capsys.readouterr().err
