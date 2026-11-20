#!/usr/bin/env python3
"""Export a verl actor checkpoint as an HF model dir with fp32 weights.

    $PY scripts/verl/export_hf_fp32.py --ckpt <run dir>/ckpt/global_step_N/actor \\
        --base-export <the HF export the run started from> --out <new export dir> \\
        [--check-against <verl bf16 merge>/model.safetensors]

verl's own ``python -m verl.model_merger merge --backend fsdp`` casts every shard to bf16. The
checkpoint holds the fp32 master weights, and our SFT exports and evaluations use fp32, so this tool
concatenates the FSDP shards itself and keeps them in fp32.

The output dir gets one ``model.safetensors`` with the merged weights. Every other file (config.json,
the model code, the tokenizer, the chat template) is copied from ``--base-export``, so the new export
loads exactly like the start export and differs from it only in its weights. The one change is that
config.json loses the keys that hold a path on the training machine (``MACHINE_LOCAL_CONFIG_KEYS``).
``export_info.json`` records where the weights came from: the run name and step of the checkpoint and
the name of the base export, without the folders above them. Both names are taken from the paths as
given, without following symlinks, so a folder reached through a symlink is recorded under the
symlink's name.

``--check-against`` takes the safetensors file of verl's bf16 merge of the same checkpoint. Every
merged tensor, rounded to bf16, must then be bit-identical to verl's, which proves that the shards
were put together in the right order. Without it the tool only checks names, dtypes and shapes
against the base export.

The checkpoint must come from verl's FSDP engine with a one-dimensional device mesh, which is what
every run of this repo uses: one ``model_world_size_<W>_rank_<R>.pt`` file per rank. Run it on a CPU
node; it needs about twice the fp32 model size in memory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

SHARD_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")
# The weight files of an export. They are replaced by the merged weights, while every other file of
# the base export is copied unchanged.
WEIGHT_FILES = ("model.safetensors.index.json",)
# Model config keys that hold a path on the training machine, the same keys as
# orbit.export.MACHINE_LOCAL_CONFIG_KEYS. The trainer's exporter leaves them out, but a base export
# written by an older exporter may still carry them. This script runs in the verl environment, which
# does not import the trainer package, so the list is repeated here.
MACHINE_LOCAL_CONFIG_KEYS = ("_attn_kernel_repo_path",)


def shard_files(ckpt: Path) -> list[Path]:
    """The rank files of the checkpoint, in rank order. Refuses a checkpoint with missing ranks."""
    found: dict[int, dict[int, Path]] = {}
    for path in ckpt.iterdir():
        match = SHARD_RE.match(path.name)
        if match:
            found.setdefault(int(match.group(1)), {})[int(match.group(2))] = path
    if len(found) != 1:
        raise SystemExit(f"{ckpt} must hold the shards of exactly one world size, found {sorted(found) or 'none'}.")
    ((world, ranks),) = found.items()
    if sorted(ranks) != list(range(world)):
        raise SystemExit(f"{ckpt} has ranks {sorted(ranks)} of world size {world}; some shards are missing.")
    return [ranks[rank] for rank in range(world)]


def merge_shards(paths: list[Path]) -> dict:
    """Concatenate the per-rank tensors into full fp32 tensors."""
    import torch
    from torch.distributed.tensor import DTensor, Replicate, Shard

    shards = [torch.load(str(path), map_location="cpu", weights_only=False) for path in paths]
    merged = {}
    for key in sorted(shards[0]):
        parts = [shard[key] for shard in shards]
        if isinstance(parts[0], DTensor):
            (placement,) = parts[0].placements
            local = [part._local_tensor for part in parts]
            if isinstance(placement, Shard):
                full = torch.cat(local, dim=placement.dim)
            elif isinstance(placement, Replicate):
                full = local[0]
            else:
                raise SystemExit(f"{key}: unexpected placement {placement}")
        else:
            full = parts[0]
        if full.dtype != torch.float32:
            raise SystemExit(f"{key} is {full.dtype} in the checkpoint, expected the fp32 master weights.")
        merged[key] = full.contiguous()
    return merged


def base_tensors(base: Path) -> dict[str, list[int]]:
    """The tensor names and shapes of the base export's safetensors files."""
    from safetensors import safe_open

    shapes = {}
    for path in sorted(base.glob("*.safetensors")):
        with safe_open(str(path), "pt") as handle:
            for key in handle.keys():
                shapes[key] = list(handle.get_slice(key).get_shape())
    return shapes


def check_against_verl(merged: dict, verl_bf16: Path) -> None:
    """Every merged tensor, rounded to bf16, must equal verl's bf16 merge bit for bit."""
    import torch
    from safetensors import safe_open

    with safe_open(str(verl_bf16), "pt") as handle:
        names = set(handle.keys())
        if names != set(merged):
            raise SystemExit(f"tensor names differ from verl's merge: {sorted(names ^ set(merged))[:5]}")
        for key in sorted(merged):
            ref = handle.get_tensor(key)
            if ref.shape != merged[key].shape or not torch.equal(merged[key].to(torch.bfloat16), ref):
                raise SystemExit(f"{key} differs from verl's bf16 merge")


def copy_config(src: Path, dst: Path) -> None:
    """Copy the base export's config.json without the keys that hold a path on the training machine.

    A config.json without such keys is copied byte for byte. Otherwise it is written again the way
    transformers writes it, with sorted keys and an indent of 2.
    """
    config = json.loads(src.read_text(encoding="utf-8"))
    if not any(key in config for key in MACHINE_LOCAL_CONFIG_KEYS):
        shutil.copy2(src, dst)
        return
    for key in MACHINE_LOCAL_CONFIG_KEYS:
        config.pop(key, None)
    dst.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def recorded_name(path: Path, parts: int) -> str:
    """The last ``parts`` components of ``path``, made absolute without following symlinks."""
    return "/".join(Path(os.path.abspath(path)).parts[-parts:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, type=Path, help="the actor dir of a verl checkpoint")
    parser.add_argument("--base-export", required=True, type=Path, help="the HF export the run started from")
    parser.add_argument("--out", required=True, type=Path, help="the new export dir; it must not exist")
    parser.add_argument("--check-against", type=Path, help="model.safetensors of verl's bf16 merge of --ckpt")
    args = parser.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new dir.")
    merged = merge_shards(shard_files(args.ckpt))
    expected = base_tensors(args.base_export)
    if set(expected) != set(merged):
        raise SystemExit(f"tensor names differ from the base export: {sorted(set(expected) ^ set(merged))[:5]}")
    wrong = [key for key in merged if list(merged[key].shape) != expected[key]]
    if wrong:
        raise SystemExit(f"tensor shapes differ from the base export: {wrong[:5]}")
    if args.check_against is not None:
        check_against_verl(merged, args.check_against)
        print(f"{len(merged)} tensors match verl's merge after bf16 rounding")

    from safetensors.torch import save_file

    # The new dir is filled under a temporary name and renamed at the end, so an interrupted export
    # never leaves a dir that looks complete.
    tmp = args.out.with_name(f".{args.out.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for path in sorted(args.base_export.iterdir()):
        if path.suffix == ".safetensors" or path.name in WEIGHT_FILES or path.name == "export_info.json":
            continue
        if path.name == "config.json":
            copy_config(path, tmp / path.name)
            continue
        (shutil.copytree if path.is_dir() else shutil.copy2)(path, tmp / path.name)
    save_file(merged, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    # An export is meant to be copied to other machines and published. Its sources are therefore
    # named as <run name>/ckpt/global_step_N/actor and by the base export's folder name, because
    # the full paths would only describe the machine that made the export. The names are the ones
    # on the command line. Symlinks are not followed, so a folder reached through a symlink is
    # recorded under the symlink's name.
    info = {
        "source_checkpoint": recorded_name(args.ckpt, 4),
        "weights": "fp32 FSDP shards concatenated by scripts/verl/export_hf_fp32.py",
        "non_weight_files_from": recorded_name(args.base_export, 1),
        "checked_against_verl_bf16_merge": args.check_against is not None,
    }
    (tmp / "export_info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    tmp.rename(args.out)
    print(f"wrote {args.out} ({len(merged)} fp32 tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
