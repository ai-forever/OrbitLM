"""Checks a document store and prints a few decoded documents for a human to read.

For every store file it checks that the offsets match the token file, that every document ends
with EOS and has no other EOS, that no document is empty, and that every token id is inside the
vocabulary. --min-doc-tokens and --max-doc-tokens also check the length range, counted without
the final EOS. Then it prints --show documents spread over the first store file, or, with
--grep, the first documents whose decoded text contains the given string. --files K checks only
the first K store files, which keeps a look at a big store quick.

Usage: check_store.py STORE_DIR [--min-doc-tokens N] [--max-doc-tokens N] [--files K] [--show N] [--grep TEXT]
"""
import argparse
import os

import numpy as np

import documents

VOCAB_SIZE = 151936
HEAD_CHARS, TAIL_CHARS = 1500, 200


def check_file(store, name):
    """Checks one store file and returns the lengths of its documents (with their EOS)."""
    tokens = np.fromfile(os.path.join(store, name + ".u32"), dtype=np.uint32)
    offsets = np.load(os.path.join(store, name + ".idx.npy"))
    assert offsets[0] == 0 and offsets[-1] == tokens.size, (name, offsets[-1], tokens.size)
    lengths = np.diff(offsets)
    assert lengths.min() > 1, f"{name}: an empty document"
    assert np.array_equal(np.flatnonzero(tokens == documents.EOS), offsets[1:] - 1), \
        f"{name}: EOS is missing at a document end or appears inside a document"
    assert int(tokens.max()) < VOCAB_SIZE, f"{name}: token id {int(tokens.max())} is outside the vocabulary"
    return lengths


def show(text, label):
    tail = f"\n[...]\n{text[-TAIL_CHARS:]}" if len(text) > HEAD_CHARS + TAIL_CHARS else ""
    print(f"===== {label}\n{text[:HEAD_CHARS]}{tail}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("--min-doc-tokens", type=int, default=0)
    ap.add_argument("--max-doc-tokens", type=int, default=0)
    ap.add_argument("--files", type=int, default=0, help="check only the first K store files")
    ap.add_argument("--show", type=int, default=3, help="decoded documents to print")
    ap.add_argument("--grep", default=None, help="print only documents whose text contains this string")
    a = ap.parse_args()

    files = documents.docstore_files(a.store)
    assert files, f"no store files in {a.store}"
    if a.files:
        files = files[:a.files]
    lengths = np.concatenate([check_file(a.store, name) for name in files])
    body = lengths - 1
    print(f"{a.store}: files {len(files)}, docs {lengths.size}, tokens {lengths.sum()}, "
          f"tokens per document without EOS: min {body.min()}, "
          f"p10/p50/p90/p99 {np.percentile(body, [10, 50, 90, 99]).astype(int).tolist()}, max {body.max()}")
    assert body.min() >= a.min_doc_tokens, f"a document is shorter than {a.min_doc_tokens} tokens"
    assert not a.max_doc_tokens or body.max() <= a.max_doc_tokens, f"a document is longer than {a.max_doc_tokens} tokens"

    tok = documents.tokenizer()
    shown = 0
    if a.grep is not None:
        for name in files:
            for i, doc in enumerate(documents.iter_docstore_file(a.store, name)):
                if shown == a.show:
                    return
                text = tok.decode(doc.tolist(), skip_special_tokens=False)
                if a.grep in text:
                    show(text, f"{name} document {i}, {len(doc)} tokens")
                    shown += 1
        print(f"documents shown: {shown}")
        return
    docs = list(documents.iter_docstore_file(a.store, files[0]))
    step = max(1, len(docs) // max(a.show, 1))
    for i in range(0, len(docs), step)[:a.show]:
        show(tok.decode(docs[i].tolist(), skip_special_tokens=False), f"{files[0]} document {i}, {len(docs[i])} tokens")


if __name__ == "__main__":
    main()
