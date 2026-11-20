"""Pack rendered dialogs into fixed blocks and write OrbitLM SFT MDS shards.

A block is exactly context_size tokens (training seq_len + 1). Dialogs longer
than that are dropped, never truncated. Short dialogs are concatenated until
the next one would not fit, then the block is padded. Padding uses the
tokenizer pad id, which for Qwen3 equals <|endoftext|>, with loss_mask 0.

The columns are the ones the trainer reads for premasked labels: tokens
(uint32 bytes), loss_mask (uint8 bytes, one per token) and dtype (the label
"uint32").
"""
import os

import numpy as np

from render import EOS_ID

MDS_COLUMNS = {"tokens": "bytes", "loss_mask": "bytes", "dtype": "str"}
DTYPE_LABEL = "uint32"
SHARD_SIZE_LIMIT = "128mb"


def pack_dialogs(dialogs, context_size, pad_id=EOS_ID, min_resp_tokens=2, stats=None):
    """Yield (tokens, loss_mask) blocks of length context_size.

    `dialogs` yields (token ids, loss mask) pairs. A dialog whose mask sums to
    less than min_resp_tokens, or whose length exceeds context_size, is
    skipped. The remainder of the last block is padded with pad_id / mask 0.
    Counts are added to `stats` when that dict is passed.
    """
    if stats is None:
        stats = {}
    current_tokens = []
    current_mask = []
    dropped_long = 0
    dropped_short = 0
    kept = 0
    assistant_tokens = 0
    dialog_tokens = 0

    def flush():
        nonlocal current_tokens, current_mask
        if not current_tokens:
            return None
        pad = context_size - len(current_tokens)
        block = (current_tokens + [pad_id] * pad, current_mask + [0] * pad)
        current_tokens, current_mask = [], []
        return block

    for ids, mask in dialogs:
        if len(ids) != len(mask):
            raise RuntimeError(f"dialog mask length {len(mask)} != tokens {len(ids)}")
        supervised = sum(mask)
        if supervised < min_resp_tokens:
            dropped_short += 1
            continue
        if len(ids) > context_size:
            dropped_long += 1
            continue
        if current_tokens and len(current_tokens) + len(ids) > context_size:
            block = flush()
            if block is not None:
                yield block
        current_tokens.extend(ids)
        current_mask.extend(mask)
        kept += 1
        assistant_tokens += supervised
        dialog_tokens += len(ids)

    block = flush()
    if block is not None:
        yield block

    stats["kept"] = stats.get("kept", 0) + kept
    stats["dropped_long"] = stats.get("dropped_long", 0) + dropped_long
    stats["dropped_short"] = stats.get("dropped_short", 0) + dropped_short
    stats["assistant_tokens"] = stats.get("assistant_tokens", 0) + assistant_tokens
    stats["dialog_tokens"] = stats.get("dialog_tokens", 0) + dialog_tokens


def write_blocks(blocks, out_dir, compression=None):
    """Write packed blocks to one MDS directory. Returns the number of blocks."""
    from streaming import MDSWriter

    os.makedirs(out_dir, exist_ok=True)
    written = 0
    with MDSWriter(
        columns=MDS_COLUMNS,
        out=out_dir,
        compression=compression,
        size_limit=SHARD_SIZE_LIMIT,
    ) as writer:
        for tokens, mask in blocks:
            if len(tokens) != len(mask):
                raise RuntimeError("block mask/token length mismatch")
            writer.write({
                "tokens": np.asarray(tokens, dtype=np.uint32).tobytes(),
                "loss_mask": np.asarray(mask, dtype=np.uint8).tobytes(),
                "dtype": DTYPE_LABEL,
            })
            written += 1
    return written
