"""Round trip of the 4K pretraining writer: raw files in, MDS stream out, read back as the trainer reads it."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PRETRAIN = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PRETRAIN))
sys.path.insert(0, PRETRAIN)
sys.path.insert(0, os.path.join(REPO, "src"))

import tokenize_4k  # noqa: E402
from tokenize_4k import documents, mds  # noqa: E402

EOS = documents.EOS
BLOCK = 16


def test_concat_blocks_cuts_full_blocks_and_drops_the_rest():
    rng = np.random.default_rng(0)
    docs = [np.append(rng.integers(0, 1000, int(n), dtype=np.uint32), np.uint32(EOS)) for n in rng.integers(1, 40, 30)]
    stream = np.concatenate(docs)
    blocks = list(tokenize_4k.concat_blocks(iter(docs), BLOCK))
    assert all(len(block) == BLOCK for block in blocks)
    assert len(blocks) == len(stream) // BLOCK
    assert np.array_equal(np.concatenate(blocks), stream[:len(blocks) * BLOCK])


@pytest.fixture
def orbit_tokenizer(tokenizer_root, monkeypatch):
    """Makes the writer use the release pretraining tokenizer, also in its worker processes."""
    from tokenizers import Tokenizer

    path = str(tokenizer_root / "tokenizer_qwen3_orbit" / "tokenizer.json")
    tokenizer = Tokenizer.from_file(path)
    monkeypatch.setattr(documents, "_tokenizer", tokenizer)
    # On macOS the worker processes start fresh instead of forking, so they do not see the
    # attribute set above. They load the tokenizer from ORBIT_TOKENIZERS, which therefore has to
    # name the same directory.
    monkeypatch.setenv("ORBIT_TOKENIZERS", str(tokenizer_root))
    return tokenizer


def write_raw(raw):
    """Writes a jsonl file, a zstd jsonl file that the include filter skips, and a parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zstandard

    (raw / "a").mkdir(parents=True)
    small = ["The first document.", "A second one, with <think> written literally.", ""]
    with open(raw / "a" / "keep_0.jsonl", "w", encoding="utf-8") as f:
        for text in small:
            f.write(json.dumps({"text": text, "id": 1}) + "\n")
    with open(raw / "a" / "skip_1.jsonl.zstd", "wb") as f:
        f.write(zstandard.ZstdCompressor().compress(json.dumps({"text": "never read"}).encode() + b"\n"))
    large = [f"Paragraph {i}: " + "numbers and words " * (i % 7 + 1) for i in range(60)]
    pq.write_table(pa.table({"text": large}), raw / "a" / "keep_2.parquet")
    (raw / ".cache").mkdir()
    (raw / ".cache" / "keep_3.jsonl").write_text(json.dumps({"text": "a download cache file"}) + "\n")
    return {"keep_0.jsonl": small, "keep_2.parquet": large}


def expected_blocks(tokenizer, texts):
    ids = []
    for encoding in tokenizer.encode_batch(texts, add_special_tokens=False):
        ids += encoding.ids + [EOS]
    full = len(ids) // BLOCK * BLOCK
    return np.array(ids[:full], dtype=np.uint32).reshape(-1, BLOCK)


def test_stream_round_trip_through_the_trainer_reader(tmp_path, monkeypatch, orbit_tokenizer):
    raw = tmp_path / "raw"
    texts = write_raw(raw)
    out = tmp_path / "pretrain4k"
    monkeypatch.setattr(sys, "argv", ["tokenize_4k.py", "--name", "toy", "--src", str(raw), "--include", "^keep_",
                                      "--out-root", str(out), "--block-tokens", str(BLOCK), "--workers", "2"])
    tokenize_4k.main()

    stream = out / "toy"
    assert sorted(os.listdir(stream)) == ["merged_shards"]
    merged = stream / "merged_shards"
    index = json.loads((merged / "index.json").read_text())
    assert index["version"] == 2
    names = [shard["raw_data"]["basename"] for shard in index["shards"]]
    assert names == [f"shard.{i:08d}.mds" for i in range(len(names))]
    assert sorted(os.listdir(merged)) == sorted(names + ["index.json"])
    for shard in index["shards"]:
        assert shard["column_names"] == ["dtype", "tokens"]
        assert shard["column_encodings"] == ["uint32", "bytes"]
        assert shard["compression"] is None and shard["size_limit"] == 1 << 27

    # One shard per raw file here, in sorted file order; the literal <think> stays a control token.
    want = [expected_blocks(orbit_tokenizer, texts[name]) for name in ("keep_0.jsonl", "keep_2.parquet")]
    assert [shard["samples"] for shard in index["shards"]] == [len(w) for w in want]
    assert 151667 in want[0]
    want = np.concatenate(want)

    from streaming import StreamingDataset

    from orbit.data import build_xy_from_sample, tokens_bytes_to_ids

    dataset = StreamingDataset(local=str(merged), remote=None, shuffle=False, batch_size=1)
    assert len(dataset) == len(want)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert int(sample["dtype"]) == 1
        ids = tokens_bytes_to_ids(sample)
        assert ids.tolist() == want[i].tolist()
        x, y = build_xy_from_sample(sample, ids, block_size=BLOCK, pad_id=EOS, eos_id=EOS)
        assert len(x) == len(y) == BLOCK
        # The trainer reads BLOCK + 1 tokens, so the last label of a full block is a padded
        # <|endoftext|> that the text does not have.
        if ids[-1] != EOS:
            assert x.tolist() == ids.tolist() and int(y[-1]) == EOS

    # The 16K and 32K stages find the files again as runs and rebuild their documents.
    root, shards = mds.read_shards(str(stream))
    runs = mds.split_runs(shards)
    assert runs == [[0], [1]]
    docs = [doc.tolist() for run in runs for doc in mds.iter_run_docs(root, shards, run)]
    originals = [encoding.ids + [EOS] for encoding in orbit_tokenizer.encode_batch(texts["keep_2.parquet"], add_special_tokens=False)]
    assert docs[-1] in originals
