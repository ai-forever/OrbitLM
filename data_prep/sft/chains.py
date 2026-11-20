"""Longest reasoning chain of one dialog, in tokens of the chain text alone.

The text is the same span the length report measures: everything between
<think> and the first </think>, trimmed like the Qwen3.5 chat template.
Markers are not counted. A tool turn stores that span in reasoning_content
instead, and that text is a chain too. The length is the tokenizer's count
of the text by itself, not its length inside the packed dialog. An example's
length is the longest such chain. A dialog with no chain is kept.
"""
import numpy as np


def reasoning_text(content):
    """Reasoning span of one assistant turn, or None when it has no closed think.

    Matches chat_template_qwen35.jinja: the text before the first </think>,
    after the last <think> in that head, trimmed. An empty span is "" so a
    closed think with no words still counts as a chain of length 0.
    """
    content = content or ""
    if "</think>" not in content:
        return None
    head = content.split("</think>", 1)[0].rstrip("\n")
    if "<think>" in head:
        head = head.split("<think>")[-1].lstrip("\n")
    return head.strip()


def assistant_chain_texts(turns):
    """Chain text of every assistant turn. Empty when the dialog has no chain."""
    texts = []
    for turn in turns:
        if turn.get("role") != "assistant":
            continue
        reasoning = turn.get("reasoning_content") or ""
        if "</think>" in reasoning:
            extracted = reasoning_text(reasoning)
            if extracted is not None:
                texts.append(extracted)
        elif reasoning.strip():
            texts.append(reasoning.strip())
        extracted = reasoning_text(turn.get("content") or "")
        if extracted is not None:
            texts.append(extracted)
    return texts


def _token_counts(tokenizer, texts):
    counts = np.zeros(len(texts), dtype=np.int32)
    pending = []
    pending_at = []

    def flush():
        if not pending:
            return
        encoded = tokenizer(pending, add_special_tokens=False, padding=False)
        for index, ids in zip(pending_at, encoded["input_ids"]):
            counts[index] = len(ids)
        pending.clear()
        pending_at.clear()

    for index, text in enumerate(texts):
        if not text:
            continue
        pending.append(text)
        pending_at.append(index)
        if len(pending) >= 32:
            flush()
    flush()
    return counts


def longest_lengths(tokenizer, turns_list):
    """Longest chain of each dialog, or None when that dialog has no chain."""
    flat = []
    spans = []
    for turns in turns_list:
        texts = assistant_chain_texts(turns)
        spans.append((len(flat), len(texts)))
        flat.extend(texts)
    counts = _token_counts(tokenizer, flat)
    lengths = []
    for start, size in spans:
        if size == 0:
            lengths.append(None)
            continue
        lengths.append(int(counts[start:start + size].max()))
    return lengths


def keep_mask(tokenizer, turns_list, limit):
    """True when the dialog stays. limit None keeps everything.

    A dialog is dropped only when its longest chain is strictly longer than
    the limit. No chain, or a chain of exactly the limit, stays.
    """
    if limit is None:
        return [True] * len(turns_list)
    return [length is None or length <= limit for length in longest_lengths(tokenizer, turns_list)]


def percentile(values, q):
    """Smallest value at which the cumulative weight reaches q percent.

    Same rank as the length report: uniform weight per example that has a
    chain. Empty input returns None.
    """
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    cumulative = np.cumsum(np.ones(values.size, dtype=np.float64))
    rank = float(cumulative[-1]) * (float(q) / 100.0)
    index = int(np.searchsorted(cumulative, rank, side="left"))
    index = min(index, int(ordered.size) - 1)
    return int(ordered[index])


def chain_threshold(values, floor, q):
    """max(floor, percentile). No chains in the stream leaves the floor."""
    rank = percentile(values, q)
    if rank is None:
        return int(floor), None
    return max(int(floor), rank), rank
