"""Tokenizes the long code subset of ProLong into the prolong_code document store of the 32K stage.

The source is prolong-text, a text copy of the princeton-nlp ProLong training data (241 parquet
files with the columns text and source; see paths.PROLONG_TEXT). Its rows with the source
'thestackv1_concat_by_repo-long/65536-65536' are 64K-token windows (Llama-3 tokens) of The Stack
v1 files concatenated by repository. Each such row becomes one document here, re-tokenized with
our Qwen3 tokenizer.

Only this subset is used. The long books subset of the same data comes from SlimPajama's Books
split, which includes Books3, and it is left out because the model will be released publicly.

The row groups that hold code rows are listed in a fixed order and shuffled with --seed. Part P
takes every Nth of them and stops once it has written --target-tokens / N tokens, so the store
is a reproducible random sample of the subset.

Output: <32K root>/work/docs/prolong_code (part P writes store part P) and
<32K root>/manifests/build/prolong_code.part_PP.json.

Usage: build_prolong_code.py --part P --nparts N --target-tokens 4.5e9
"""
import argparse
import glob
import json
import os
import random
import time

import pyarrow.compute as pc
import pyarrow.parquet as pq

import documents
import paths

SUBSET = "thestackv1_concat_by_repo-long/65536-65536"
# A census of the source column found the code rows only in the first two files (77,452 rows in
# file 0 and 20,851 in file 1, 98,303 in total). The rest of the data is books and short text.
CODE_FILES = ("train-00000-of-00241.parquet", "train-00001-of-00241.parquet")
BATCH = 16


def code_row_groups():
    """Lists (file name, row group) for every row group of the files that hold code rows."""
    groups = []
    for name in CODE_FILES:
        parquet = pq.ParquetFile(os.path.join(paths.PROLONG_TEXT, name))
        for row_group in range(parquet.num_row_groups):
            groups.append((name, row_group))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--target-tokens", type=float, required=True, help="tokens of the whole store")
    ap.add_argument("--seed", type=int, default=32768)
    a = ap.parse_args()

    found = sorted(os.path.basename(f) for f in glob.glob(os.path.join(paths.PROLONG_TEXT, "*.parquet")))
    assert found[:2] == list(CODE_FILES), f"unexpected source files in {paths.PROLONG_TEXT}: {found[:2]}"
    groups = code_row_groups()
    random.Random(a.seed).shuffle(groups)
    mine = groups[a.part::a.nparts]
    target = a.target_tokens / a.nparts
    writer = documents.DocStoreWriter(os.path.join(paths.CTX32K_ROOT, "work", "docs", "prolong_code"), a.part)
    stats = {"row_groups": 0, "rows_code": 0, "rows_other": 0, "tokens": 0}
    start = time.time()
    for name, row_group in mine:
        table = pq.ParquetFile(os.path.join(paths.PROLONG_TEXT, name)).read_row_group(row_group, columns=["text", "source"])
        texts = table.filter(pc.equal(table["source"], SUBSET))["text"].to_pylist()
        stats["rows_other"] += table.num_rows - len(texts)
        for i in range(0, len(texts), BATCH):
            for doc in documents.encode_docs(texts[i:i + BATCH]):
                writer.add(doc)
                stats["rows_code"] += 1
                stats["tokens"] += len(doc)
        stats["row_groups"] += 1
        print(f"{time.strftime('%H:%M:%S')} part {a.part}: {name} row group {row_group}, {json.dumps(stats)}, "
              f"{stats['tokens'] / max(time.time() - start, 1) / 1e6:.2f}M tok/s", flush=True)
        if stats["tokens"] >= target:
            break
    stats.update(writer.close())
    stats["minutes"] = round((time.time() - start) / 60, 1)
    manifest_dir = os.path.join(paths.CTX32K_ROOT, "manifests", "build")
    os.makedirs(manifest_dir, exist_ok=True)
    with open(os.path.join(manifest_dir, f"prolong_code.part_{a.part:02d}.json"), "w") as f:
        json.dump({"args": vars(a), "stats": stats}, f, indent=1)
    print("DONE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
