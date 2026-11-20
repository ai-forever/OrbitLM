"""Training data: MDS token shards mixed by one Mosaic StreamingDataset, cut into (x, y) pairs.

Every stream of a config is a folder of MDS shards (or a folder of such folders). The streams are
mixed by sampling weight inside a single StreamingDataset (build_stage1_mixture_dataset and
build_streaming_mixture_dataset), and MdsTokenIterableDataset turns each sample into the input
and label tensors of one row. A sample holds seq_len + 1 tokens, because the labels are the
inputs shifted by one.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .optional import STREAMING_IMPORT_ERROR, Stream, StreamingDataset
from .runtime import format_count


def resolve_dataset_roots(dataset_root: Path) -> list[Path]:
    if (dataset_root / "index.json").exists():
        return [dataset_root]
    roots = discover_mds_dirs(dataset_root)
    if roots:
        return roots
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root does not exist: {dataset_root!s}")
    try:
        preview = sorted(
            f"{path.name}/" if path.is_dir() else path.name
            for path in dataset_root.iterdir()
        )[:32]
    except Exception as exc:
        preview = [f"<failed to list directory: {exc!r}>"]
    raise RuntimeError(
        f"No MDS index.json found under {dataset_root!s}. "
        f"First entries: {preview}"
    )


def filter_streaming_kwargs(kwargs: dict) -> dict:
    if StreamingDataset is None:
        return kwargs
    try:
        sig = inspect.signature(StreamingDataset)
    except (TypeError, ValueError):
        return kwargs
    allowed = set(sig.parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


def filter_stream_kwargs(kwargs: dict) -> dict:
    if Stream is None:
        return kwargs
    try:
        sig = inspect.signature(Stream)
    except (TypeError, ValueError):
        return kwargs
    allowed = set(sig.parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}

def discover_mds_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    dirs: set[Path] = set()
    seen_dirs: set[str] = set()
    for current, dirnames, filenames in os.walk(root, followlinks=True):
        current_path = Path(current)
        try:
            current_real = str(current_path.resolve())
        except OSError:
            dirnames[:] = []
            continue
        if current_real in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(current_real)

        pruned_dirnames = []
        for dirname in dirnames:
            child = current_path / dirname
            try:
                child_real = str(child.resolve())
            except OSError:
                continue
            if child_real not in seen_dirs:
                pruned_dirnames.append(dirname)
        dirnames[:] = pruned_dirnames

        if "index.json" in filenames:
            dirs.add(current_path)
    return sorted(dirs)


def decode_sample_dtype_code(raw_dtype) -> Optional[int]:
    if raw_dtype is None:
        return None
    if isinstance(raw_dtype, (int, np.integer)):
        return int(raw_dtype)
    if isinstance(raw_dtype, (bytes, bytearray)):
        blob = bytes(raw_dtype)
        if len(blob) == 1:
            if 48 <= blob[0] <= 57:  # ASCII digits
                return int(blob.decode("utf-8"))
            return int(blob[0])
        try:
            return int(blob.decode("utf-8").strip())
        except Exception:
            return int.from_bytes(blob, byteorder="little", signed=False)
    if isinstance(raw_dtype, str):
        text = raw_dtype.strip().lower()
        if text in {"int32", "uint32"}:
            return 1
        if text in {"int64", "uint64"}:
            return 2
        if text in {"int16", "uint16"}:
            return 4
        if text in {"int8", "uint8"}:
            return 0
        if text.isdigit():
            return int(text)
    try:
        return int(raw_dtype)
    except Exception:
        return None


def tokens_bytes_to_ids(sample, override_dtype_code: Optional[int] = None) -> np.ndarray:
    data = sample["tokens"]
    # The config's stage1_token_dtype arrives here as override_dtype_code and replaces the dtype
    # label stored in the sample. A wrong label would make the token bytes be read at the wrong
    # width, which gives ids outside the vocabulary.
    if override_dtype_code is not None:
        dtype_code = override_dtype_code
    else:
        dtype_code = decode_sample_dtype_code(sample.get("dtype", 1))
    dtype_map = {
        0: np.uint8,
        1: np.int32,
        2: np.int64,
        4: np.int16,
    }

    if isinstance(data, np.ndarray):
        ids = data
    else:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            data = bytes(data)
        dt = dtype_map.get(dtype_code)
        if dt is None:
            # Conservative fallback when dtype metadata is absent/corrupt.
            if len(data) % 8 == 0:
                dt = np.int64
            elif len(data) % 4 == 0:
                dt = np.int32
            elif len(data) % 2 == 0:
                dt = np.int16
            else:
                dt = np.uint8
        ids = np.frombuffer(data, dtype=dt)

    if np.issubdtype(ids.dtype, np.floating):
        raise RuntimeError(f"Unsupported floating token dtype in sample: {ids.dtype}")
    if ids.dtype == np.int8:
        ids = ids.view(np.uint8)
    elif ids.dtype == np.int16:
        ids = ids.view(np.uint16)
    elif ids.dtype == np.int32 and np.any(ids < 0):
        ids = ids.view(np.uint32)
    return ids.astype(np.int64, copy=False)


def trim_padding(ids: np.ndarray, pad_id: Optional[int]) -> np.ndarray:
    if pad_id is None:
        return ids
    end = len(ids)
    while end > 0 and int(ids[end - 1]) == pad_id:
        end -= 1
    return ids[:end]


def build_xy_from_sample(
    sample,
    ids: np.ndarray,
    *,
    block_size: int,
    pad_id: Optional[int],
    eos_id: Optional[int],
    require_loss_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the (x, y) training pair for one MDS sample.

    When the sample carries a per-token "loss_mask" (the packed SFT shards that
    data_prep/sft writes), labels are masked by POSITION: y positions whose mask
    is 0 (prompt tokens, padding) become -100, and supervised tokens keep their
    values even when they equal the pad/eos id. Such streams are read with
    labels_premasked, which requires the column on every sample.

    Without a mask (pretraining shards), trailing padding is trimmed by token
    value and only the padded tail of the labels is masked, again by position.
    """
    raw_mask = sample.get("loss_mask") if hasattr(sample, "get") else None
    needed = block_size + 1

    if raw_mask is None:
        if require_loss_mask:
            raise RuntimeError(
                "labels_premasked is enabled but the stream sample has no "
                "'loss_mask' column; this stream would train on unmasked labels."
            )
        ids = trim_padding(ids, pad_id)
        real_len = len(ids)                       # genuine (non-padding) tokens
        if real_len < needed:
            fill_id = pad_id if pad_id is not None else (eos_id or 0)
            ids = np.pad(ids, (0, needed - real_len), constant_values=fill_id)
        else:
            ids = ids[:needed]
            real_len = needed
        x = torch.tensor(ids[:block_size], dtype=torch.long)
        y = ids[1:needed].astype(np.int64, copy=True)
        # Mask the padded tail by POSITION, not by pad VALUE. y[k] predicts ids[k+1];
        # ids[real_len] is the first pad token (also the eos target, since pad_id ==
        # eos_id for our tokenizer), so we KEEP y[real_len-1] supervised (predict eos
        # after the last real token) and mask only y[real_len:] (predict pad after pad).
        # Value masking (labels == pad_id -> -100) would instead drop EVERY eos target,
        # and the model would never learn to end a document. Interior eos/document
        # separators are never trailing, so they stay supervised.
        if real_len < needed:
            y[real_len:] = -100
        return x, torch.from_numpy(y)

    mask = np.frombuffer(raw_mask, dtype=np.uint8)
    if len(mask) != len(ids):
        raise RuntimeError(
            f"loss_mask length {len(mask)} does not match tokens length {len(ids)}"
        )
    # Packed blocks are written at exactly block_size + 1; pad defensively.
    if len(ids) < needed:
        fill_id = pad_id if pad_id is not None else (eos_id or 0)
        ids = np.pad(ids, (0, needed - len(ids)), constant_values=fill_id)
        mask = np.pad(mask, (0, needed - len(mask)), constant_values=0)
    else:
        ids = ids[:needed]
        mask = mask[:needed]
    x = torch.tensor(ids[:block_size], dtype=torch.long)
    y = ids[1:needed].astype(np.int64, copy=True)
    y[mask[1:needed] == 0] = -100
    return x, torch.from_numpy(y)


class MdsTokenIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, block_size: int, pad_id: Optional[int], eos_id: Optional[int],
                 token_dtype_code: Optional[int] = None, require_loss_mask: bool = False):
        super().__init__()
        self.dataset = dataset
        self.block_size = block_size
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.token_dtype_code = token_dtype_code
        self.require_loss_mask = require_loss_mask

    def __iter__(self):
        for sample in self.dataset:
            ids = tokens_bytes_to_ids(sample, self.token_dtype_code)
            yield build_xy_from_sample(
                sample,
                ids,
                block_size=self.block_size,
                pad_id=self.pad_id,
                eos_id=self.eos_id,
                require_loss_mask=self.require_loss_mask,
            )


def build_streaming_mixture_dataset(
    streams: list[Stream],
    *,
    seed: int,
    shuffle: bool,
    shuffle_block_size: Optional[int],
    batch_size: int,
    predownload: Optional[int],
    cache_limit: Optional[str],
    epoch_size: Optional[int],
    partition_algo: Optional[str],
    num_canonical_nodes: Optional[int],
) -> "StreamingDataset":
    """Build one Mosaic StreamingDataset that mixes all streams internally."""
    if StreamingDataset is None:
        raise RuntimeError(
            "streaming is required to read .mds shards. Import failed with: "
            f"{STREAMING_IMPORT_ERROR}"
        )
    if not streams:
        raise RuntimeError("Mosaic random mixing requires at least one stream.")

    kwargs = {
        "streams": streams,
        "shuffle": bool(shuffle),
        "shuffle_seed": int(seed),
        "batch_size": int(batch_size),
        "download_timeout": 60,
        "sampling_method": "balanced",
        "batching_method": "random",
    }
    if shuffle_block_size is not None:
        kwargs["shuffle_block_size"] = int(shuffle_block_size)
    if predownload is not None and predownload > 0:
        kwargs["predownload"] = int(predownload)
    if cache_limit is not None:
        kwargs["cache_limit"] = cache_limit
    if epoch_size is not None:
        kwargs["epoch_size"] = int(epoch_size)
    if partition_algo is not None:
        kwargs["partition_algo"] = partition_algo
    if num_canonical_nodes is not None:
        kwargs["num_canonical_nodes"] = int(num_canonical_nodes)

    filtered_kwargs = filter_streaming_kwargs(kwargs)
    if "streams" not in filtered_kwargs:
        raise RuntimeError(
            "Installed streaming.StreamingDataset does not expose multi-stream support."
        )
    return StreamingDataset(**filtered_kwargs)


def build_stage1_mixture_dataset(
    root: Path,
    entries: list[tuple[str, str]],
    token_budget: int,
    total_samples: int,
    dataset_specs: dict[str, dict[str, float]],
    log_fn=None,
) -> tuple[list[Stream], int, list[str]]:
    """Builds one Mosaic Stream per MDS folder of every config stream.

    A stream path that is not absolute is a folder under root. A stream folder that holds several
    MDS folders gets one Stream per folder, and its proportion is split evenly between them.
    token_budget and total_samples only feed the log lines.
    """
    if StreamingDataset is None or Stream is None:
        raise RuntimeError(
            "streaming is required to read .mds shards. Import failed with: "
            f"{STREAMING_IMPORT_ERROR}"
        )

    streams = []
    labels = []
    # The proportions of a config are relative weights; they are divided by their sum here.
    total_prop = sum(float(dataset_specs[key]["proportion"]) for key, _ in entries)
    if total_prop <= 0:
        raise RuntimeError("Stage 1 dataset proportions sum to zero.")

    for key, folder in entries:
        prop = float(dataset_specs[key]["proportion"]) / total_prop
        target_tokens = int(token_budget * prop)
        if log_fn is not None:
            log_fn(
                f"stage1_dataset={key} folder={folder} target_tokens={format_count(target_tokens)} "
                f"epoch_samples={format_count(int(total_samples * prop))}"
            )
        dataset_root = Path(folder)
        if not dataset_root.is_absolute():
            dataset_root = root / folder
        roots = resolve_dataset_roots(dataset_root)
        per_root_prop = prop / float(max(1, len(roots)))
        for sub_root in roots:
            stream_kwargs = dict(
                local=str(sub_root),
                remote=None,
                split=None,
                proportion=float(per_root_prop),
            )
            streams.append(Stream(**filter_stream_kwargs(stream_kwargs)))
            labels.append(key)

    if not streams:
        raise RuntimeError("No Stage 1 datasets could be constructed.")
    return streams, total_samples, labels
