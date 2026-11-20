"""Tokenize one input file into one MDS part. Used by build.py's process pool.

The tokenizer is loaded once per worker. Dialogs are rendered and packed in
batches so a large parquet file is not held in memory as token lists.
"""
import json
import os

import numpy as np

import benches
import chains
import dialogs
from blocks import pack_dialogs, write_blocks
from render import (
    EOS_ID,
    example_enable_thinking,
    mask_all_but_last_assistant,
    render_batch,
    render_tool_dialog,
)
from system_prompts import load_rewrite_table

TOKENIZER = None
REWRITE = None
BATCH = 32
# How often a long stream rewrites progress_current.json while it tokenizes.
PROGRESS_EVERY = 2000


def init_worker(tokenizer_path, rewrite_path=None, system_turns=False):
    global TOKENIZER, REWRITE
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
    REWRITE = load_rewrite_table(rewrite_path, system_turns)


def _dialogs(task, seen):
    kind = task["kind"]
    limit = task["limit"]
    if kind == "bench":
        yield from benches.iter_bench_dialogs(task["raw_root"], task.get("sources"), limit=limit)
        return
    spec = {
        "kind": kind,
        "path": task["path"],
        "dedup_prompts": task.get("dedup_prompts", False),
        "record_prompts": False,
        "byte_start": task.get("byte_start"),
        "byte_end": task.get("byte_end"),
    }
    yield from dialogs.iter_stream_dialogs(spec, limit=limit, seen_prompts=seen, rewrite=REWRITE)


def note_progress(task, rendered):
    """Rewrite the current-stream heartbeat. Safe to call from a worker."""
    path = task.get("progress_path")
    if not path:
        return
    payload = {
        "stream": task.get("stream_id"),
        "file": task.get("part_label") or os.path.basename(task.get("path") or "") or task.get("stream_id"),
        "rendered": int(rendered),
    }
    if task.get("progress_phase"):
        payload["phase"] = task["progress_phase"]
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")
    os.replace(tmp, path)


def _apply_loss_span(task, item):
    if not task.get("loss_last_assistant"):
        return item
    ids, mask = item
    return ids, mask_all_but_last_assistant(mask)


def _prepared(task, seen):
    """Dialogs the builder would pack, as (turns, tools). tools is None otherwise."""
    if task["kind"] == "tool_jsonl":
        spec = {
            "path": task["path"],
            "byte_start": task.get("byte_start"),
            "byte_end": task.get("byte_end"),
        }
        for turns, tools in dialogs.iter_tool_dialogs(spec, limit=task["limit"], rewrite=REWRITE):
            yield turns, tools
        return
    for turns in _dialogs(task, seen):
        yield turns, None


def _seen_prompts(task):
    keys_path = task.get("prompt_keys_path")
    if not keys_path:
        return None
    return set(int(value) for value in np.load(keys_path).tolist())


def measure_chains(task):
    """Longest chain of every dialog in one file that has a chain.

    Uses the same preparation as tokenization, including the quality filter
    and T-Wix prompt dedup. The returned array is the population of the
    percentile: dialogs with no chain are counted in n_dialogs and omitted
    here.
    """
    task = dict(task)
    task["progress_phase"] = "chain"
    seen = _seen_prompts(task)
    lengths = []
    pending = []
    scanned = 0
    noted = 0

    def flush():
        if not pending:
            return
        for value in chains.longest_lengths(TOKENIZER, pending):
            if value is not None:
                lengths.append(value)
        pending.clear()

    for turns, _tools in _prepared(task, seen):
        pending.append(turns)
        scanned += 1
        if len(pending) >= BATCH:
            flush()
        if scanned - noted >= PROGRESS_EVERY:
            note_progress(task, scanned)
            noted = scanned
    flush()
    note_progress(task, scanned)
    return {
        "n_dialogs": scanned,
        "lengths": np.asarray(lengths, dtype=np.int32),
    }


def _rendered(task, seen, stats):
    mode = task.get("enable_thinking", False)
    rendered = 0
    noted = 0
    limit = task.get("chain_max")

    def count(n):
        nonlocal rendered, noted
        rendered += n
        if rendered - noted >= PROGRESS_EVERY:
            note_progress(task, rendered)
            noted = rendered

    pending_turns = []
    pending_tools = []

    def flush():
        if not pending_turns:
            return
        keep = chains.keep_mask(TOKENIZER, pending_turns, limit)
        plain = []
        flags = []
        tool_rows = []
        for turns, tools, ok in zip(pending_turns, pending_tools, keep):
            if not ok:
                stats["dropped_chain"] = stats.get("dropped_chain", 0) + 1
                continue
            if tools is None:
                plain.append(turns)
                flags.append(example_enable_thinking(mode, turns))
            else:
                tool_rows.append((turns, tools))
        pending_turns.clear()
        pending_tools.clear()
        if plain:
            for item in render_batch(plain, TOKENIZER, enable_thinking=flags):
                yield _apply_loss_span(task, item)
            count(len(plain))
        for turns, tools in tool_rows:
            flag = example_enable_thinking(mode, turns)
            try:
                item = _apply_loss_span(
                    task, render_tool_dialog(turns, tools, TOKENIZER, enable_thinking=flag),
                )
            except Exception as exc:
                stats["dropped_render"] = stats.get("dropped_render", 0) + 1
                if stats["dropped_render"] <= 3:
                    print(
                        f"drop render {task.get('stream_id')}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                continue
            count(1)
            yield item

    for turns, tools in _prepared(task, seen):
        pending_turns.append(turns)
        pending_tools.append(tools)
        if len(pending_turns) >= BATCH:
            yield from flush()
    yield from flush()
    note_progress(task, rendered)


def encode_file(task):
    """Render one file (or the bench list) and optionally write an MDS part.

    Returns a stats dict. `blocks` is the number of packed blocks written or,
    in stats-only mode, the number that would be written.
    """
    seen = _seen_prompts(task)
    stats = {"read_files": 1}
    if task.get("chain_max") is not None:
        stats["dropped_chain"] = 0
    context = task["context_size"]
    blocks = pack_dialogs(
        _rendered(task, seen, stats),
        context,
        pad_id=TOKENIZER.pad_token_id if TOKENIZER.pad_token_id is not None else EOS_ID,
        min_resp_tokens=task["min_resp_tokens"],
        stats=stats,
    )
    out_dir = task.get("out_dir")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # Write only when the first block exists, so an empty file leaves no shard.
        first = next(blocks, None)
        if first is None:
            stats["blocks"] = 0
            return stats
        def rest():
            yield first
            yield from blocks
        stats["blocks"] = write_blocks(rest(), out_dir)
    else:
        stats["blocks"] = sum(1 for _ in blocks)
    stats["packed_tokens"] = stats["blocks"] * context
    return stats
