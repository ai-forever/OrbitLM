"""Tokenizes raw text files into one 4K pretraining stream: <out-root>/<name>/merged_shards/.

The 4K configs read the English streams from $ORBIT_DATA/pretrain_4k/en and the Russian ones
from $ORBIT_DATA/pretrain_4k/ru, so --out-root is one of these two.

The 4K pretraining runs read streams that were written by MosaicML llm-foundry's ConcatTokens
conversion. This script writes the same format, with the stream writer and the shard merge of
the long-context package (data_prep/long_context/mds.py):

  - every raw file is converted on its own, in sorted path order;
  - the "text" field of every record is tokenized as it is, without any cleaning and without
    special tokens, and one <|endoftext|> (151643) is appended to it;
  - the token stream of a file is cut into blocks of 4,096 tokens, and the tokens after the
    last full block of the file are dropped;
  - a block is one MDS sample with the columns dtype (uint32, value 1) and tokens (the uint32
    token bytes), in uncompressed shards of at most 128 MiB (8,185 samples);
  - the shards of all files are then moved into merged_shards/ under one index.json and
    renamed shard.00000000.mds, shard.00000001.mds, and so on.

Each file therefore ends with one partial shard, and the 16K and 32K stages rely on that to find
the file boundaries again (long_context/mds.py, split_runs).

The blocks are 4,096 tokens and not seq_len + 1 = 4,097, because that is what the 4K runs
trained on. The trainer pads a 4,096-token sample by one token, so the last position of every
sample has an <|endoftext|> target that is not in the text. The 16K and 32K streams do not have
this, because their builders write seq_len + 1 tokens per sample.

Records are read from .jsonl, .jsonl.gz, .jsonl.zst, .jsonl.zstd and .parquet files. A
file that is converted completely leaves a finished part directory with an index.json, so a
stopped run can be restarted and skips those files. The merge runs once every file is done.

Usage:
  tokenize_4k.py --out-root DIR --name NAME --src DIR [--src DIR ...] [--include REGEX] [--workers N]
  tokenize_4k.py --out-root "$ORBIT_DATA/pretrain_4k/en" \
      --name nemotron-cc --src "$ORBIT_RAW/pretrain4k/nemotron-cc" --include '^quality=high'
data_prep/README.md lists the source directories of every stream of the two 4K configs.
"""
import argparse
import gzip
import io
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "long_context"))
import documents  # noqa: E402
import mds  # noqa: E402

BLOCK_TOKENS = 4096
SUFFIXES = (".jsonl", ".jsonl.gz", ".jsonl.zst", ".jsonl.zstd", ".parquet")
# Texts are tokenized in batches of this many records, which is much faster than one at a time
# and gives the same token ids.
ENCODE_BATCH = 1024


def list_files(srcs, include):
    """Returns (source directory, path relative to it) of every raw file, in the order they are converted."""
    pattern = re.compile(include) if include else None
    found = []
    for src in srcs:
        for current, dirs, files in os.walk(src):
            # Hidden directories hold download caches and lock files, not data.
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for name in files:
                if not name.endswith(SUFFIXES) or name.startswith("."):
                    continue
                if pattern and not pattern.search(name):
                    continue
                found.append((src, os.path.relpath(os.path.join(current, name), src)))
    return sorted(found, key=lambda item: (item[0], item[1]))


def iter_texts(path, text_field):
    """Yields the text of every record of one raw file, in file order."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=ENCODE_BATCH, columns=[text_field]):
            yield from batch.column(0).to_pylist()
        return
    if path.endswith((".zst", ".zstd")):
        import zstandard

        handle = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8")
    elif path.endswith(".gz"):
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = open(path, encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)[text_field]


def iter_documents(path, text_field):
    """Yields every record of a file as token ids followed by one <|endoftext|>."""
    tokenizer = documents.tokenizer()
    batch = []
    for text in iter_texts(path, text_field):
        if not isinstance(text, str):
            raise ValueError(f"{path}: a record has no text in the field {text_field!r}")
        batch.append(text)
        if len(batch) == ENCODE_BATCH:
            yield from _encode(tokenizer, batch)
            batch = []
    if batch:
        yield from _encode(tokenizer, batch)


def _encode(tokenizer, texts):
    for encoding in tokenizer.encode_batch(texts, add_special_tokens=False):
        yield np.array(encoding.ids + [documents.EOS], dtype=np.uint32)


def concat_blocks(docs, block_tokens):
    """Joins documents into one token stream and yields it in full blocks, dropping the remainder."""
    pending, pending_tokens = [], 0
    for doc in docs:
        pending.append(doc)
        pending_tokens += len(doc)
        if pending_tokens < block_tokens:
            continue
        stream = np.concatenate(pending)
        full = len(stream) // block_tokens * block_tokens
        yield from stream[:full].reshape(-1, block_tokens)
        rest = stream[full:]
        pending, pending_tokens = [rest], len(rest)


def convert_file(job):
    """Converts one raw file into a part directory. Returns (part directory, blocks written)."""
    path, part_dir, text_field, block_tokens = job
    if os.path.exists(os.path.join(part_dir, "index.json")):
        with open(os.path.join(part_dir, "index.json")) as f:
            return part_dir, sum(s["samples"] for s in json.load(f)["shards"])
    # A part without an index.json was stopped halfway, so it starts again from nothing.
    shutil.rmtree(part_dir, ignore_errors=True)
    writer = mds.StreamWriter(part_dir, block_tokens)
    for block in concat_blocks(iter_documents(path, text_field), block_tokens):
        writer(block)
    writer.close()
    return part_dir, writer.blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="the stream to write: <out-root>/<name>/merged_shards")
    ap.add_argument("--src", action="append", required=True, help="a directory of raw files, searched recursively; repeatable")
    ap.add_argument("--include", default="", help="keep only files whose name matches this regular expression")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--out-root", required=True, help="the directory of the stream directories")
    ap.add_argument("--block-tokens", type=int, default=BLOCK_TOKENS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    stream_dir = os.path.join(a.out_root, a.name)
    merged = os.path.join(stream_dir, "merged_shards")
    if os.path.exists(os.path.join(merged, "index.json")):
        raise SystemExit(f"{merged} is already merged; remove it to rebuild")
    files = list_files(a.src, a.include)
    if not files:
        raise SystemExit(f"no raw files under {a.src}")
    jobs = [(os.path.join(src, rel), os.path.join(stream_dir, "parts", f"{i:06d}"), a.text_field, a.block_tokens)
            for i, (src, rel) in enumerate(files)]
    print(f"{a.name}: {len(jobs)} raw files", flush=True)
    total = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for done, (part_dir, blocks) in enumerate(pool.map(convert_file, jobs), start=1):
            total += blocks
            print(f"  {done}/{len(jobs)} {os.path.basename(part_dir)}: {blocks} blocks", flush=True)
    os.makedirs(merged, exist_ok=True)
    samples = mds.merge_parts(merged, [job[1] for job in jobs])
    os.rmdir(os.path.join(stream_dir, "parts"))
    print(f"merged {a.name}: {samples} samples, {samples * a.block_tokens / 1e9:.3f}B tokens", flush=True)


if __name__ == "__main__":
    main()
