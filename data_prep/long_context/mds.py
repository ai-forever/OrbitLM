"""MDS shards: reading the existing 4K pretraining shards, and writing the packed streams.

A stream is a directory of MDS shards under one index.json, in the format the OrbitLM trainer
reads: a uint32 "tokens" column and a "dtype" column. Every sample of a stream is one block of
exactly seq_len + 1 tokens, because the trainer takes seq_len + 1 tokens per sample (the inputs
plus the shifted labels). A block of only seq_len tokens would be padded by one token, and
every sequence would end with a fake EOS target.

All blocks of a stage have the same length. The trainer mixes streams by sample counts, so
equal sample lengths are what make the yaml proportions equal token proportions.
"""
import bisect
import json
import os
import shutil
from collections import defaultdict

import numpy as np

from documents import EOS

DTYPE_CODE_UINT32 = 1
MDS_COLUMNS = {"dtype": "uint32", "tokens": "bytes"}
SHARD_SIZE_LIMIT = 1 << 27  # 128 MiB, the same as the existing 4K shards


# ---------------------------------------------------------------- reading the 4K MDS shards

def find_index(stream_dir):
    """Returns the directory that holds the stream's index.json (the root or merged_shards/)."""
    for candidate in (stream_dir, os.path.join(stream_dir, "merged_shards")):
        if os.path.exists(os.path.join(candidate, "index.json")):
            return candidate
    raise FileNotFoundError(f"no index.json under {stream_dir}")


def read_shards(stream_dir):
    """Returns (directory of the shards, list of shard entries of its index.json)."""
    root = find_index(stream_dir)
    with open(os.path.join(root, "index.json")) as f:
        shards = json.load(f)["shards"]
    return root, shards


def split_runs(shards):
    """Groups shard indices into runs, one run per raw input file of the original tokenization.

    Each raw file had its own writer, so its shards are a contiguous token stream that ends with
    one partial shard (fewer samples than a full shard). Documents continue across shards only
    inside a run, so documents are rebuilt run by run.
    """
    full = max(s["samples"] for s in shards)
    runs, current = [], []
    for i, s in enumerate(shards):
        current.append(i)
        if s["samples"] < full:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def read_shard_tokens(path):
    """Reads all samples of one MDS shard as one flat uint32 token array.

    The shard layout is: the sample count, the sample offsets, then the samples. Each sample
    here is the size of the tokens column (uint32), the dtype column (uint32), then the token
    bytes, so the first two uint32 words of every sample are skipped.
    """
    with open(path, "rb") as f:
        buf = f.read()
    n = int(np.frombuffer(buf, np.uint32, 1, 0)[0])
    offsets = np.frombuffer(buf, np.uint32, n + 1, 4).astype(np.int64)
    sizes = np.diff(offsets)
    if n and np.all(sizes == sizes[0]):
        # All samples have the same size, so the whole shard can be viewed as one 2-D array.
        arr = np.frombuffer(buf, np.uint32, count=int(offsets[-1] - offsets[0]) // 4, offset=int(offsets[0]))
        return arr.reshape(n, -1)[:, 2:].reshape(-1)
    parts = []
    for i in range(n):
        sample = np.frombuffer(buf, np.uint32, count=int(sizes[i]) // 4, offset=int(offsets[i]))
        parts.append(sample[2:])
    return np.concatenate(parts) if parts else np.zeros(0, np.uint32)


def iter_run_docs(root, shards, run, stats=None):
    """Yields the documents of one run, each ending with its EOS.

    The piece after the last EOS of a run is the document that the original tokenizer cut off
    when it dropped its unfinished last block, so it is not yielded. When the run is read to its
    end, the size of that piece is added to stats["cut_run_tail_tokens"].
    """
    carry = np.zeros(0, np.uint32)
    for shard_index in run:
        tokens = read_shard_tokens(os.path.join(root, shards[shard_index]["raw_data"]["basename"]))
        if carry.size:
            tokens = np.concatenate([carry, tokens])
        start = 0
        for end in np.flatnonzero(tokens == EOS):
            if end + 1 - start > 1:  # two EOS in a row would be an empty document
                yield tokens[start:end + 1]
            start = end + 1
        carry = tokens[start:]
    if stats is not None:
        stats["cut_run_tail_tokens"] += int(carry.size)


# ---------------------------------------------------------------- packing

class Packer:
    """Packs documents into blocks of exactly block_tokens tokens, cutting as few documents as possible.

    A document longer than a block is first cut into full-block windows of itself, so a long
    document fills whole blocks on its own. Its remainder, and every shorter document, goes into
    a buffer. A block is built from the buffer by taking the longest document first and then,
    repeatedly, the longest document that still fits the free space (best fit).

    When nothing in the buffer fits the last gap, the gap is filled with the start of the
    shortest document that is longer than the gap, and the rest of that document goes back into
    the buffer. This keeps every block exactly full, which the trainer's mixing needs, and cuts
    at most one document per block.
    """

    def __init__(self, emit, block_tokens, buffer_docs=4096):
        self.emit = emit
        self.block_tokens = block_tokens
        self.buffer_docs = buffer_docs
        self.lengths = []                       # sorted lengths of the buffered documents
        self.docs_by_length = defaultdict(list)
        self.buffered = 0
        self.stats = defaultdict(int)

    def _push(self, doc):
        # Documents often arrive as slices of a whole 128 MB shard read into memory. Keeping the
        # slice would keep the entire shard alive while the document waits in the buffer, and
        # thousands of waiting documents would pin hundreds of shards and run the machine out of
        # memory. Copying keeps only the document itself.
        doc = np.array(doc, dtype=np.uint32, copy=True)
        bisect.insort(self.lengths, len(doc))
        self.docs_by_length[len(doc)].append(doc)
        self.buffered += 1

    def _pop(self, length):
        del self.lengths[bisect.bisect_left(self.lengths, length)]
        doc = self.docs_by_length[length].pop()
        if not self.docs_by_length[length]:
            del self.docs_by_length[length]
        self.buffered -= 1
        return doc

    def add(self, doc):
        self.stats["docs_in"] += 1
        self.stats["tokens_in"] += len(doc)
        while len(doc) >= self.block_tokens:
            self.emit(doc[:self.block_tokens])
            self.stats["blocks"] += 1
            self.stats["window_blocks"] += 1
            doc = doc[self.block_tokens:]
        if len(doc):
            self._push(doc)
        while self.buffered > self.buffer_docs:
            self._build_block()

    def _build_block(self):
        """Emits one block built from the buffer; returns False when the buffer ran dry first."""
        parts = [self._pop(self.lengths[-1])]
        gap = self.block_tokens - len(parts[0])
        while gap > 0 and self.buffered:
            i = bisect.bisect_right(self.lengths, gap)
            if i > 0:
                doc = self._pop(self.lengths[i - 1])
                parts.append(doc)
                gap -= len(doc)
            else:
                doc = self._pop(self.lengths[0])  # the shortest document, which is longer than the gap
                parts.append(doc[:gap])
                self._push(doc[gap:])
                self.stats["cut_docs"] += 1
                gap = 0
        if gap > 0:
            # This only happens at the very end, when the last documents cannot fill a block.
            # The partial block is dropped.
            self.stats["dropped_tail_tokens"] += self.block_tokens - gap
            return False
        self.emit(np.concatenate(parts))
        self.stats["blocks"] += 1
        return True

    def flush(self):
        """Packs everything that is still buffered; the last partial block is dropped."""
        while self.buffered:
            if not self._build_block():
                break
        self.lengths.clear()
        self.docs_by_length.clear()
        self.buffered = 0


# ---------------------------------------------------------------- writing streams

class StreamWriter:
    """Writes blocks of exactly block_tokens tokens to an MDS directory, in the trainer's columns."""

    def __init__(self, out_dir, block_tokens):
        from streaming import MDSWriter
        os.makedirs(out_dir, exist_ok=True)
        self.writer = MDSWriter(out=out_dir, columns=MDS_COLUMNS, compression=None, size_limit=SHARD_SIZE_LIMIT)
        self.block_tokens = block_tokens
        self.blocks = 0

    def __call__(self, block):
        assert len(block) == self.block_tokens, len(block)
        self.writer.write({"dtype": DTYPE_CODE_UINT32,
                           "tokens": np.ascontiguousarray(block, dtype=np.uint32).tobytes()})
        self.blocks += 1

    def close(self):
        self.writer.finish()


def merge_parts(stream_dir, part_dirs):
    """Moves the shards of several part directories into stream_dir under one index.json.

    The trainer treats every directory that holds an index.json as a separate source and splits
    the stream's proportion between them equally, so a finished stream must have exactly one
    index.json, and the part directories are removed. Returns the number of samples.
    """
    shards = []
    for part_dir in part_dirs:
        with open(os.path.join(part_dir, "index.json")) as f:
            index = json.load(f)
        for s in index["shards"]:
            new_name = f"shard.{len(shards):08d}.mds"
            os.replace(os.path.join(part_dir, s["raw_data"]["basename"]), os.path.join(stream_dir, new_name))
            shards.append(dict(s, raw_data=dict(s["raw_data"], basename=new_name)))
    tmp = os.path.join(stream_dir, "index.json.tmp")
    with open(tmp, "w") as f:
        json.dump({"version": 2, "shards": shards}, f)
    os.replace(tmp, os.path.join(stream_dir, "index.json"))
    for part_dir in part_dirs:
        shutil.rmtree(part_dir)
    return sum(s["samples"] for s in shards)
