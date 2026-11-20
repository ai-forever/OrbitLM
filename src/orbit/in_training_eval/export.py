"""The HF export of the live model that the evaluation processes load, and the chat export of the export scripts."""
from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

from .files import write_json_atomic


def tokenizer_stop_ids(tokenizer_config_path: Path) -> tuple[list[int], Optional[int], Optional[int]]:
    """(eos ids, pad id, bos id) for generation_config.json, from the run's tokenizer config.

    The eos list is the configured eos plus the ids of <|endoftext|> and <|im_end|> when the
    tokenizer has them, so a completion stops at the end of a document and at the end of a turn,
    as in the paper's evaluations. For the run's Qwen3 tokenizer, whose eos is <|endoftext|>, that is
    [151643, 151645].
    """
    payload = json.loads(Path(tokenizer_config_path).read_text(encoding="utf-8"))
    token_to_id = {
        value.get("content"): int(key)
        for key, value in (payload.get("added_tokens_decoder") or {}).items()
        if isinstance(value, dict)
    }
    eos_ids: list[int] = []
    for token in (payload.get("eos_token"), "<|endoftext|>", "<|im_end|>"):
        if isinstance(token, dict):
            token = token.get("content")
        token_id = token_to_id.get(token)
        if token_id is not None and token_id not in eos_ids:
            eos_ids.append(token_id)
    if not eos_ids:
        raise ValueError(f"{tokenizer_config_path}: no eos token id found in added_tokens_decoder.")

    def lookup(name: str) -> Optional[int]:
        token = payload.get(name)
        if isinstance(token, dict):
            token = token.get("content")
        return token_to_id.get(token)

    return eos_ids, lookup("pad_token"), lookup("bos_token")


# The special tokens of a released chat model. The training tokenizer keeps <|endoftext|> as eos,
# because the trainer refuses document masking when the eos is <|im_end|>.
SHIPPING_SPECIALS = {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": None, "add_bos_token": False}


def set_shipping_specials(tokenizer_config_path: Path) -> None:
    """Rewrites an exported tokenizer config to the special tokens that a released chat model ships with.

    ship_chat_export uses it for the --chat-export option of orbit.export.
    """
    path = Path(tokenizer_config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(SHIPPING_SPECIALS)
    write_json_atomic(path, payload)


def ship_chat_export(output_dir: Path) -> None:
    """Gives an exported chat model the special tokens that a released Qwen3 chat model ships with.

    The chat model ends every turn with <|im_end|>, but the training tokenizer names <|endoftext|>
    as eos. A raw export therefore makes every tool that stops on the tokenizer's eos, on config.json
    or on generation_config.json run past the end of the answer. This sets <|im_end|> as eos in the
    tokenizer config, the special tokens map and config.json, and writes generation_config.json with
    both stop ids and no sampling defaults. tokenizer.json and the chat template are left untouched,
    so token ids do not change. Use it for chat exports only: a base model ends documents with
    <|endoftext|> and never writes <|im_end|>.
    """
    output_dir = Path(output_dir)
    if not (output_dir / "chat_template.jinja").is_file():
        raise ValueError(f"{output_dir}: no chat_template.jinja, so this is not a chat export.")
    tokenizer_config = output_dir / "tokenizer_config.json"
    set_shipping_specials(tokenizer_config)
    eos_ids, pad_id, _ = tokenizer_stop_ids(tokenizer_config)
    im_end_id = eos_ids[0]

    special_tokens_map = output_dir / "special_tokens_map.json"
    if special_tokens_map.exists():
        payload = json.loads(special_tokens_map.read_text(encoding="utf-8"))
        eos = payload.get("eos_token")
        payload["eos_token"] = dict(eos, content=SHIPPING_SPECIALS["eos_token"]) if isinstance(eos, dict) else SHIPPING_SPECIALS["eos_token"]
        write_json_atomic(special_tokens_map, payload)

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["eos_token_id"] = im_end_id
    config["pad_token_id"] = pad_id
    write_json_atomic(config_path, config)
    # Only token ids go here. vLLM would also take sampling fields as default sampling parameters.
    write_json_atomic(output_dir / "generation_config.json", {"eos_token_id": eos_ids, "pad_token_id": pad_id})


def write_eval_export(
    state: dict[str, Any],
    *,
    model_config: Any,
    out_dir: Path,
    tokenizer_source: Path,
    tokenizer_config: Path,
    max_position_embeddings: int,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Writes the gathered weights as an HF folder that vLLM's in-tree Ouro can load.

    The weights are bf16 safetensors under their HF names. config.json is the live model config
    with a window of at least ``max_position_embeddings`` (vLLM sizes its RoPE table from it),
    the run's rope_theta, and early_exit_threshold null, so the HF forward is the fixed-depth path
    that vLLM also runs. The tokenizer files come from the run's own tokenizer folder, never from
    the package default. The folder is built under <out_dir>.partial and renamed when complete;
    when anything fails, neither folder is left behind. ``metadata`` is added to
    eval_export.json, the last file written.
    """
    out_dir = Path(out_dir)
    staging = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    try:
        return _write_eval_export(
            state,
            model_config=model_config,
            staging=staging,
            out_dir=out_dir,
            tokenizer_source=tokenizer_source,
            tokenizer_config=tokenizer_config,
            max_position_embeddings=max_position_embeddings,
            metadata=metadata or {},
        )
    except BaseException:
        # A failed export is most often a full disk, and 3.7 GB of half-written files would keep
        # it full. The step is not retried, so nothing would ever clean them up.
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        raise


def _write_eval_export(
    state: dict[str, Any],
    *,
    model_config: Any,
    staging: Path,
    out_dir: Path,
    tokenizer_source: Path,
    tokenizer_config: Path,
    max_position_embeddings: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    from ..export import (
        copy_remote_code_files,
        maybe_export_tokenizer,
        normalize_key,
        save_sharded_state_dict,
        set_hf_config_metadata,
    )

    staging.mkdir(parents=True)
    tensors: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        tensor = value.detach()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.bfloat16)
        tensors[normalize_key(str(key))] = tensor.to("cpu").contiguous()

    config = copy.deepcopy(model_config)
    config.max_position_embeddings = max(int(config.max_position_embeddings), int(max_position_embeddings))
    config.early_exit_threshold = None
    # The trainer records its own FA3 build here. That build loads only under the torch of the
    # training environment, and the HF modelling code prefers this path over ORBIT_ATTN_KERNEL_PATH.
    config._attn_kernel_repo_path = None
    for name in ("dtype", "torch_dtype"):
        try:
            setattr(config, name, torch.bfloat16)
        except Exception:
            pass  # transformers versions differ in which of the two names is settable
    set_hf_config_metadata(config, attn_impl="sdpa")
    config.save_pretrained(staging)

    eos_ids, pad_id, bos_id = tokenizer_stop_ids(tokenizer_config)
    generation = {"eos_token_id": eos_ids, "pad_token_id": pad_id, "bos_token_id": bos_id}
    write_json_atomic(staging / "generation_config.json", generation)
    copy_remote_code_files(staging)
    maybe_export_tokenizer(
        staging,
        tokenizer_source=str(tokenizer_source),
        tokenizer_config_path=Path(tokenizer_config),
    )
    save_sharded_state_dict(tensors, staging, safe_serialization=True, max_shard_size="5GB")
    info = {
        "tensors": len(tensors),
        "bytes": int(sum(t.numel() * t.element_size() for t in tensors.values())),
        "vocab_size": int(config.vocab_size),
        "max_position_embeddings": int(config.max_position_embeddings),
        "rope_theta": float(getattr(config, "rope_theta", 0.0) or 0.0),
        "eos_token_ids": eos_ids,
        "tokenizer_source": str(tokenizer_source),
        **metadata,
    }
    write_json_atomic(staging / "eval_export.json", info)
    os.replace(staging, out_dir)
    return info
