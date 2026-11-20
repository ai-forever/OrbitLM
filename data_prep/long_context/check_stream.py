"""Checks a finished stream before it goes into a training yaml.

It checks that the stream has exactly one index.json, that every sample is exactly the stage's
block length in uint32 tokens, that every token id is inside the vocabulary, and that mosaicml
streaming reads the stream the way the trainer does. With --decode it prints the text around a
document boundary inside one sample, so a human can see how documents are joined.

Usage: check_stream.py --stage 32k STREAM_DIR [--decode]
"""
import argparse
import json
import os

import numpy as np

import documents
import mds
import stages

VOCAB_SIZE = 151936


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages.add_stage_argument(ap)
    ap.add_argument("stream_dir")
    ap.add_argument("--decode", action="store_true", help="print decoded text around a document boundary")
    a = ap.parse_args()
    block_tokens = stages.get(a.stage).BLOCK_TOKENS
    stream_dir = a.stream_dir

    roots = [d for d, _, files in os.walk(stream_dir) if "index.json" in files]
    assert roots == [stream_dir], f"expected exactly one index.json at the stream root, found {roots}"
    with open(os.path.join(stream_dir, "index.json")) as f:
        index = json.load(f)
    samples = sum(s["samples"] for s in index["shards"])
    bad_shards, max_id, eos, total = 0, 0, 0, 0
    for s in index["shards"]:
        tokens = mds.read_shard_tokens(os.path.join(stream_dir, s["raw_data"]["basename"]))
        if tokens.size != s["samples"] * block_tokens:
            bad_shards += 1
        max_id = max(max_id, int(tokens.max()))
        eos += int(np.count_nonzero(tokens == documents.EOS))
        total += tokens.size
    print(f"shards {len(index['shards'])}, samples {samples}, tokens {total / 1e9:.3f}B, bad shards {bad_shards}, "
          f"max id {max_id}, EOS per sample {eos / max(samples, 1):.2f}")
    assert bad_shards == 0 and max_id < VOCAB_SIZE

    from streaming import StreamingDataset
    dataset = StreamingDataset(local=stream_dir, remote=None, shuffle=False, batch_size=1)
    for i in (0, len(dataset) // 2, len(dataset) - 1):
        ids = np.frombuffer(dataset[i]["tokens"], np.uint32)
        assert ids.size == block_tokens and int(dataset[i]["dtype"]) == mds.DTYPE_CODE_UINT32
    print("streaming reader ok:", len(dataset), "samples")

    if a.decode:
        tok = documents.tokenizer()
        ids = np.frombuffer(dataset[len(dataset) // 3]["tokens"], np.uint32)
        ends = np.flatnonzero(ids == documents.EOS)
        print("EOS positions in one sample:", ends[:20].tolist(), "...")
        if ends.size:
            k = int(ends[0])
            print("around the first EOS:", repr(tok.decode(ids[max(0, k - 40):k + 40].tolist(), skip_special_tokens=False)))
        print("sample end:", repr(tok.decode(ids[-60:].tolist(), skip_special_tokens=False)))


if __name__ == "__main__":
    main()
