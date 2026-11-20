"""CPU tests of scripts/verl/export_hf_fp32.py, which turns a verl actor checkpoint into an fp32 HF export.

The checkpoints are tiny and written the way verl's FSDP engine writes them: one
``model_world_size_<W>_rank_<R>.pt`` file per rank. The sharded case runs two gloo processes on the
CPU, so the shards are real DTensors split along their first dimension. The base export holds a
config, a tokenizer file, a chat template and weights of the same names and shapes. The script runs
as a subprocess, as a user runs it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from test_launch import REPO  # noqa: E402

SCRIPT = REPO / "scripts" / "verl" / "export_hf_fp32.py"
# Two tensors: one that FSDP shards along dim 0 and one small enough to compare by eye.
SHAPES = {"model.embed_tokens.weight": (8, 4), "model.norm.weight": (4,)}


def _full_tensors() -> dict:
    generator = torch.Generator().manual_seed(0)
    return {name: torch.randn(shape, generator=generator, dtype=torch.float32) for name, shape in SHAPES.items()}


def _base_export(root: Path) -> Path:
    """A base export whose weights have the checkpoint's names and shapes, in bf16 as a trainer export may be."""
    base = root / "base_export"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "ouro", "vocab_size": 8}))
    (base / "tokenizer.json").write_text("{}")
    (base / "chat_template.jinja").write_text("{{ messages }}")
    (base / "export_info.json").write_text(json.dumps({"from": "the trainer"}))
    zeros = {name: torch.zeros(shape, dtype=torch.bfloat16) for name, shape in SHAPES.items()}
    safetensors_torch.save_file(zeros, str(base / "model.safetensors"))
    return base


def _plain_checkpoint(root: Path, tensors: dict, dtype=torch.float32) -> Path:
    """A one-rank checkpoint of plain tensors."""
    ckpt = root / "ckpt" / "global_step_2" / "actor"
    ckpt.mkdir(parents=True)
    torch.save({name: value.to(dtype) for name, value in tensors.items()}, ckpt / "model_world_size_1_rank_0.pt")
    return ckpt


def _save_dtensor_shard(rank: int, world: int, init_file: str, ckpt: str) -> None:
    """Run in each of ``world`` processes: shard the full tensors as FSDP does and save this rank's part."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        state = {}
        for name, value in _full_tensors().items():
            placement = Shard(0) if value.dim() == 2 else Replicate()
            state[name] = distribute_tensor(value, mesh, [placement])
        torch.save(state, Path(ckpt) / f"model_world_size_{world}_rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


def _sharded_checkpoint(root: Path, world: int = 2) -> Path:
    ckpt = root / "ckpt" / "global_step_2" / "actor"
    ckpt.mkdir(parents=True)
    torch.multiprocessing.spawn(_save_dtensor_shard, args=(world, str(root / "gloo_init"), str(ckpt)), nprocs=world)
    return ckpt


def _export(*argv: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, env=env)


def _read(path: Path) -> dict:
    return safetensors_torch.load_file(str(path))


def _check_export(out: Path, base: Path, ckpt: Path) -> None:
    merged = _read(out / "model.safetensors")
    full = _full_tensors()
    assert set(merged) == set(full)
    for name, value in full.items():
        assert merged[name].dtype == torch.float32 and torch.equal(merged[name], value), name
    # Everything but the weights comes from the base export unchanged.
    for name in ("config.json", "tokenizer.json", "chat_template.jinja"):
        assert (out / name).read_bytes() == (base / name).read_bytes()
    assert sorted(path.name for path in out.iterdir()) == [
        "chat_template.jinja", "config.json", "export_info.json", "model.safetensors", "tokenizer.json"]
    info_text = (out / "export_info.json").read_text()
    info = json.loads(info_text)
    # The export names its sources by run and folder name, never by this machine's paths.
    assert info["source_checkpoint"] == f"{ckpt.parents[2].name}/ckpt/global_step_2/actor"
    assert info["non_weight_files_from"] == base.name
    assert str(base.parent) not in info_text


def test_a_one_rank_checkpoint_of_plain_tensors(tmp_path):
    base, ckpt = _base_export(tmp_path), _plain_checkpoint(tmp_path, _full_tensors())
    out = tmp_path / "out"
    proc = _export("--ckpt", str(ckpt), "--base-export", str(base), "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    _check_export(out, base, ckpt)
    assert json.loads((out / "export_info.json").read_text())["checked_against_verl_bf16_merge"] is False


def test_machine_paths_stay_out_and_symlinks_keep_their_names(tmp_path):
    """A base export from an older exporter may record the local FA3 build in config.json."""
    base = _base_export(tmp_path)
    ckpt = _plain_checkpoint(tmp_path / "first_run_name", _full_tensors())
    kernel = str(tmp_path / "local_fa3_build")
    (base / "config.json").write_text(json.dumps({"model_type": "ouro", "vocab_size": 8,
                                                  "_attn_kernel_repo_path": kernel}))
    # The run and the base export are reached through symlinks with other names. The export must
    # record those names, not the names of the folders that the symlinks point to.
    links = tmp_path / "published"
    (links / "runs").mkdir(parents=True)
    (links / "runs" / "rl_run").symlink_to(ckpt.parents[2], target_is_directory=True)
    (links / "start_export").symlink_to(base, target_is_directory=True)
    out = tmp_path / "out"
    proc = _export("--ckpt", str(links / "runs" / "rl_run" / "ckpt" / "global_step_2" / "actor"),
                   "--base-export", str(links / "start_export"), "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    assert json.loads((out / "config.json").read_text()) == {"model_type": "ouro", "vocab_size": 8}
    info_text = (out / "export_info.json").read_text()
    info = json.loads(info_text)
    assert info["source_checkpoint"] == "rl_run/ckpt/global_step_2/actor"
    assert info["non_weight_files_from"] == "start_export"
    assert "first_run_name" not in info_text and base.name not in info_text and str(tmp_path) not in info_text


def test_dtensor_shards_are_concatenated_in_rank_order(tmp_path):
    base, ckpt = _base_export(tmp_path), _sharded_checkpoint(tmp_path)
    # The shards really are halves, so a merge in the wrong order would not pass the check below.
    local = torch.load(ckpt / "model_world_size_2_rank_1.pt", weights_only=False)["model.embed_tokens.weight"]
    assert tuple(local._local_tensor.shape) == (4, 4)
    out = tmp_path / "out"
    proc = _export("--ckpt", str(ckpt), "--base-export", str(base), "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    _check_export(out, base, ckpt)


def test_the_check_against_verl_s_bf16_merge(tmp_path):
    base, ckpt = _base_export(tmp_path), _plain_checkpoint(tmp_path, _full_tensors())
    verl_merge = tmp_path / "verl_bf16.safetensors"
    safetensors_torch.save_file({name: value.to(torch.bfloat16) for name, value in _full_tensors().items()},
                                str(verl_merge))
    out = tmp_path / "out"
    proc = _export("--ckpt", str(ckpt), "--base-export", str(base), "--out", str(out), "--check-against",
                   str(verl_merge))
    assert proc.returncode == 0, proc.stderr
    assert "match verl's merge after bf16 rounding" in proc.stdout
    assert json.loads((out / "export_info.json").read_text())["checked_against_verl_bf16_merge"] is True

    # A tensor that differs after bf16 rounding stops the export before anything is written.
    perturbed = {name: value.to(torch.bfloat16) for name, value in _full_tensors().items()}
    perturbed["model.norm.weight"] = perturbed["model.norm.weight"] + 1
    safetensors_torch.save_file(perturbed, str(verl_merge))
    proc = _export("--ckpt", str(ckpt), "--base-export", str(base), "--out", str(tmp_path / "out2"),
                   "--check-against", str(verl_merge))
    assert proc.returncode != 0 and "model.norm.weight differs from verl's bf16 merge" in proc.stderr
    assert not (tmp_path / "out2").exists()


def test_refusals(tmp_path):
    base = _base_export(tmp_path)
    ckpt = _plain_checkpoint(tmp_path, _full_tensors())
    out = tmp_path / "out"
    out.mkdir()
    proc = _export("--ckpt", str(ckpt), "--base-export", str(base), "--out", str(out))
    assert proc.returncode != 0 and "exists" in proc.stderr

    # The checkpoint must hold the fp32 master weights.
    bf16 = _plain_checkpoint(tmp_path / "bf16", _full_tensors(), dtype=torch.bfloat16)
    proc = _export("--ckpt", str(bf16), "--base-export", str(base), "--out", str(tmp_path / "out_bf16"))
    assert proc.returncode != 0 and "expected the fp32 master weights" in proc.stderr

    # A rank file that is missing would silently drop half of every sharded tensor.
    (ckpt / "model_world_size_1_rank_0.pt").rename(ckpt / "model_world_size_2_rank_1.pt")
    proc = _export("--ckpt", str(ckpt), "--base-export", str(base), "--out", str(tmp_path / "out_missing"))
    assert proc.returncode != 0 and "some shards are missing" in proc.stderr

    # Weights whose names differ from the base export's are refused.
    other = _plain_checkpoint(tmp_path / "other", {"model.other.weight": torch.zeros(2)})
    proc = _export("--ckpt", str(other), "--base-export", str(base), "--out", str(tmp_path / "out_other"))
    assert proc.returncode != 0 and "tensor names differ from the base export" in proc.stderr
