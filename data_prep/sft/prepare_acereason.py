"""Keep a fixed code sample from a local nvidia/AceReason-1.1-SFT download.

The download is read from $ORBIT_RAW/hf/nvidia__AceReason-1.1-SFT, where
data_prep/long_context/download_hf.py puts Hugging Face sources, unless
--root names another folder.

The full dump has about 1.3M code samples and 2.7M math samples. v4 uses
100k code rows. v8 uses a fresh 200k reservoir with the same seed. The 200k
draw is not a superset of the 100k draw. Neither sample is read back in as
source data.

  python data_prep/sft/prepare_acereason.py
  python data_prep/sft/prepare_acereason.py --target 200000 --out-dir /path/code_200k
"""
import argparse
import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 100_000
SEED = 46
# Already-written samples. A later draw must not treat them as source rows.
SAMPLE_DIRS = ("code_100k", "code_200k")
SAMPLE_FILES = ("code_100k.parquet", "code_200k.parquet")
ROWS_PER_PART = 10_000


def arrow_files(root):
    patterns = [
        os.path.join(root, "sft_data.parquet", "*.arrow"),
        os.path.join(root, "data", "*.arrow"),
        os.path.join(root, "*.arrow"),
        os.path.join(root, "**", "*.parquet"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern, recursive=True))
    kept = []
    for path in set(found):
        if not os.path.isfile(path):
            continue
        parts = path.split(os.sep)
        if ".cache" in parts or any(name in SAMPLE_DIRS for name in parts):
            continue
        if os.path.basename(path) in SAMPLE_FILES:
            continue
        kept.append(path)
    return sorted(kept)


def iter_rows(path):
    """Yield code rows from an Arrow IPC file or a parquet file."""
    if path.endswith(".parquet"):
        table = pq.read_table(path, columns=["category", "source", "input", "output"])
    else:
        with pa.memory_map(path, "r") as source:
            try:
                reader = pa.ipc.open_file(source)
                table = reader.read_all()
            except pa.ArrowInvalid:
                source.seek(0)
                reader = pa.ipc.open_stream(source)
                table = reader.read_all()
        table = table.select(["category", "source", "input", "output"])
    mask = pc.equal(table.column("category"), "code")
    table = table.filter(mask)
    if table.num_rows == 0:
        return
    columns = [table.column(name).to_pylist() for name in ("category", "source", "input", "output")]
    for category, source, question, answer in zip(*columns):
        yield {
            "category": category,
            "source": source,
            "input": question,
            "output": answer,
        }


def sample_code(root, target=TARGET, seed=SEED):
    files = arrow_files(root)
    if not files:
        raise FileNotFoundError(f"no AceReason files under {root}")
    rng = np.random.default_rng(seed)
    kept = []
    seen = 0
    for index, path in enumerate(files, start=1):
        for row in iter_rows(path):
            seen += 1
            if len(kept) < target:
                kept.append(row)
                continue
            slot = int(rng.integers(0, seen))
            if slot < target:
                kept[slot] = row
        if index % 20 == 0 or index == len(files):
            print(f"  scanned {index}/{len(files)} files, code rows {seen}", flush=True)
    if len(kept) < target:
        raise RuntimeError(f"found {len(kept)} code rows, wanted {target}")
    return kept, seen


def default_root():
    """The AceReason download under $ORBIT_RAW."""
    raw = os.environ.get("ORBIT_RAW")
    if not raw:
        raise SystemExit("set ORBIT_RAW first; data_prep/README.md describes the ORBIT_* roots")
    return os.path.join(raw, "hf", "nvidia__AceReason-1.1-SFT")


def _table(rows):
    return pa.table({
        "category": [row["category"] for row in rows],
        "source": [row["source"] for row in rows],
        "input": [row["input"] for row in rows],
        "output": [row["output"] for row in rows],
    })


def write_parts(rows, out_dir, rows_per_part=ROWS_PER_PART):
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)
    written = 0
    part = 0
    for start in range(0, len(rows), rows_per_part):
        chunk = rows[start:start + rows_per_part]
        path = os.path.join(out_dir, f"part-{part:02d}.parquet")
        pq.write_table(_table(chunk), path)
        written += len(chunk)
        part += 1
    return written, part


def write_sample(root=None, target=TARGET, seed=SEED, out_dir=None):
    root = root or default_root()
    kept, seen = sample_code(root, target, seed)
    if out_dir:
        written, parts = write_parts(kept, out_dir)
        print(
            f"wrote {out_dir}: {written} code rows in {parts} parts, sampled from {seen}",
            flush=True,
        )
        return out_dir
    out = os.path.join(root, "code_100k.parquet")
    pq.write_table(_table(kept), out)
    print(f"wrote {out}: {len(kept)} code rows sampled from {seen}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Reservoir-sample AceReason code rows.")
    parser.add_argument("--root", default=None, help="default: $ORBIT_RAW/hf/nvidia__AceReason-1.1-SFT")
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default="", help="directory of part-*.parquet; default is one code_100k.parquet")
    args = parser.parse_args()
    write_sample(args.root, args.target, args.seed, args.out_dir or None)


if __name__ == "__main__":
    main()
