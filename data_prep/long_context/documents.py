"""Turning text into training documents, and the document store that holds them.

A document is a uint32 array of token ids that ends with exactly one EOS. Builders render their
source as plain text, tokenize it with encode_docs and write the documents to a document store
with DocStoreWriter. pack.py later reads the stores back and packs the documents into the
fixed-length blocks of a stream, so packing can be re-run without rendering again.
"""
import os

import numpy as np

import paths

EOS = 151643  # <|endoftext|>; it ends every document
# Every id from EOS upward is an added (control) token in this tokenizer.
FIRST_ADDED_TOKEN_ID = EOS
THINK_TOKEN_IDS = (151667, 151668)  # <think>, </think>
THINK_TAGS = ("<think>", "</think>")

_tokenizer = None
_added_token_strings = None


def tokenizer():
    """Returns the OrbitLM Qwen3 tokenizer (tokenizers.Tokenizer), loaded once per process."""
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer
        _tokenizer = Tokenizer.from_file(paths.TOKENIZER_JSON)
    return _tokenizer


def added_token_strings():
    """Returns every added-token string of the tokenizer (<|endoftext|>, <think>, <filename>, ...).

    Any of them written literally in a document would be encoded as a control token, so
    clean_text removes all of them from document text. The longest strings come first, so that a
    string which contains a shorter one is removed whole.
    """
    global _added_token_strings
    if _added_token_strings is None:
        contents = {t.content for t in tokenizer().get_added_tokens_decoder().values()}
        _added_token_strings = sorted(contents, key=lambda s: (-len(s), s))
    return _added_token_strings


def clean_text(text, keep=()):
    """Removes every added-token string from text, except the strings listed in keep.

    None is treated as an empty text.
    """
    text = text or ""
    for s in added_token_strings():
        if s in text and s not in keep:
            text = text.replace(s, "")
    return text


def encode_docs(texts):
    """Tokenizes a batch of document texts and appends EOS to each; returns uint32 arrays.

    Every added-token string is removed first, except <think> and </think>. render_chat puts
    those tags into the text on purpose, and removing them here would erase the reasoning markup.
    A literal <think> in a plain document therefore survives as a control token, and pack.py
    drops such documents from streams that are packed without --allow-think.
    """
    encodings = tokenizer().encode_batch([clean_text(t, keep=THINK_TAGS) for t in texts], add_special_tokens=False)
    return [np.array(e.ids + [EOS], dtype=np.uint32) for e in encodings]


def render_chat(turns, system=None):
    """Renders one conversation as plain text, without ChatML tokens.

    The format is "System: ..." (only when there is a system prompt), then alternating
    "User: ..." and "Assistant: ..." blocks separated by blank lines. turns is a list of dicts
    with the role "user" or "assistant", the content, and optionally the reasoning. Reasoning
    is written as <think>...</think> only for the last assistant turn, the way Qwen3 drops the
    reasoning of earlier turns.
    """
    parts = []
    if system and system.strip():
        parts.append("System: " + clean_text(system).strip())
    last_assistant = max((i for i, t in enumerate(turns) if t["role"] == "assistant"), default=-1)
    for i, t in enumerate(turns):
        content = clean_text(t.get("content")).strip()
        if t["role"] == "user":
            parts.append("User: " + content)
        elif t["role"] == "assistant":
            reasoning = clean_text(t.get("reasoning")).strip() if i == last_assistant else ""
            parts.append("Assistant: " + (f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content))
        else:
            raise ValueError(f"unexpected role {t['role']!r}")
    return "\n\n".join(parts)


class DocStoreWriter:
    """Writes tokenized documents (each ending with EOS) to one part of a document store.

    A store is a directory of numbered parts. Each part is split into files of about
    flush_tokens tokens, and each file is a flat uint32 token file (.u32) plus an int64 offsets
    file (.idx.npy) with one entry more than the file has documents. Both are written under a
    temporary name and renamed when complete, so a crashed builder leaves no truncated token
    file behind.
    """

    def __init__(self, store_dir, part, flush_tokens=200_000_000):
        os.makedirs(store_dir, exist_ok=True)
        self.base = os.path.join(store_dir, f"part_{part:04d}")
        self.flush_tokens = flush_tokens
        self.buffer, self.buffered_tokens, self.next_file = [], 0, 0
        self.docs = self.tokens = 0

    def add(self, doc):
        self.buffer.append(np.asarray(doc, dtype=np.uint32))
        self.buffered_tokens += len(doc)
        self.docs += 1
        self.tokens += len(doc)
        if self.buffered_tokens >= self.flush_tokens:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return
        name = f"{self.base}_{self.next_file:03d}"
        offsets = np.zeros(len(self.buffer) + 1, np.int64)
        offsets[1:] = np.cumsum([len(d) for d in self.buffer])
        np.concatenate(self.buffer).tofile(name + ".u32.tmp")
        np.save(name + ".idx.tmp.npy", offsets)
        os.replace(name + ".u32.tmp", name + ".u32")
        os.replace(name + ".idx.tmp.npy", name + ".idx.npy")
        self.buffer, self.buffered_tokens, self.next_file = [], 0, self.next_file + 1

    def close(self):
        """Writes the last file and returns the part's totals as {"docs": ..., "tokens": ...}."""
        self._flush()
        return {"docs": self.docs, "tokens": self.tokens}


def docstore_files(store_dir):
    """Returns the names of the finished files of a store (without extension), in sorted order."""
    return sorted(f[:-len(".u32")] for f in os.listdir(store_dir) if f.endswith(".u32"))


def iter_docstore_file(store_dir, name):
    """Yields the documents of one store file, in the order they were written."""
    tokens = np.fromfile(os.path.join(store_dir, name + ".u32"), dtype=np.uint32)
    offsets = np.load(os.path.join(store_dir, name + ".idx.npy"))
    for i in range(len(offsets) - 1):
        yield tokens[offsets[i]:offsets[i + 1]]
