"""Token counts for the OpenThoughts3 topic split.

Uses the same prepare_messages, rewrite table, chain threshold, and chat
template as the v6 pack. Dialogs longer than seq_len + 1 are not counted.
"""
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq

import chains
from label_ot3 import (
    OUT_PATH, PARTIAL_ROOT, SMOLTALK, STATS_PATH, TOKENIZER,
    first_user, load_partial, question_key, save_partial,
)
from render import prepare_messages, render_batch

CONTEXT = 32768 + 1
TOPICS = ("code", "math", "science", "other")
_COUNT = {}


def _empty():
    return {topic: Counter() for topic in TOPICS}


def _init_count(domains, rewrite, tokenizer_path):
    from transformers import AutoTokenizer

    _COUNT["domains"] = domains
    _COUNT["rewrite"] = rewrite
    _COUNT["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_path)


def _count_file(payload):
    path, thinking, chain_max = payload
    domains = _COUNT["domains"]
    rewrite = _COUNT["rewrite"]
    tokenizer = _COUNT["tokenizer"]
    counts = _empty()
    kept = 0
    table = pq.ParquetFile(path)
    for batch in table.iter_batches(batch_size=64, columns=["messages"]):
        pending = []
        topics = []
        for messages in batch.column("messages").to_pylist():
            prepared = prepare_messages(messages, rewrite=rewrite, quality_filter=True)
            if prepared is None:
                continue
            domain = domains.get(question_key(first_user(messages)))
            topic = domain if domain in {"math", "code", "science"} else "other"
            pending.append(prepared)
            topics.append(topic)
        if not pending:
            continue
        keep = chains.keep_mask(tokenizer, pending, chain_max)
        kept_turns = [turns for turns, ok in zip(pending, keep) if ok]
        kept_topics = [topic for topic, ok in zip(topics, keep) if ok]
        if not kept_turns:
            continue
        for (ids, mask), topic in zip(
            render_batch(kept_turns, tokenizer, enable_thinking=thinking),
            kept_topics,
        ):
            if len(ids) > CONTEXT:
                continue
            counts[topic]["examples"] += 1
            counts[topic]["tokens"] += len(ids)
            counts[topic]["assistant"] += int(sum(mask))
            kept += 1
    return counts, kept


def _count_record(part, part_kept):
    return {
        "kept": part_kept,
        "topics": {
            topic: {field: int(part[topic][field]) for field in ("examples", "tokens", "assistant")}
            for topic in TOPICS
        },
    }


def _apply_count(counts, kept, record):
    for topic in TOPICS:
        for field, value in record["topics"][topic].items():
            counts[topic][field] += value
    return kept + int(record["kept"])


def write_counts(domains, rewrite, limits, out_path=OUT_PATH, jobs=24):
    import glob
    from concurrent.futures import as_completed

    result = {}
    with open(STATS_PATH, encoding="utf-8") as handle:
        packed = json.load(handle)
    for name, (pattern, thinking) in SMOLTALK.items():
        files = sorted(glob.glob(pattern))
        counts = _empty()
        kept = 0
        pending = []
        resumed = 0
        for path in files:
            saved = load_partial(os.path.join(PARTIAL_ROOT, "count", name, os.path.basename(path) + ".json"))
            if saved is None:
                pending.append(path)
                continue
            resumed += 1
            kept = _apply_count(counts, kept, saved)
        done = resumed
        if resumed:
            print(f"  {name} resumed {resumed}/{len(files)} count files, kept {kept}", flush=True)
        if pending:
            with ProcessPoolExecutor(
                max_workers=min(jobs, len(pending)),
                mp_context=multiprocessing.get_context("fork"),
                initializer=_init_count,
                initargs=(domains, rewrite, TOKENIZER),
            ) as pool:
                payloads = [(path, thinking, limits[name]) for path in pending]
                futures = {pool.submit(_count_file, payload): payload[0] for payload in payloads}
                for future in as_completed(futures):
                    path = futures[future]
                    part, part_kept = future.result()
                    record = _count_record(part, part_kept)
                    save_partial(
                        os.path.join(PARTIAL_ROOT, "count", name, os.path.basename(path) + ".json"),
                        record,
                    )
                    kept = _apply_count(counts, kept, record)
                    done += 1
                    if done % 10 == 0 or done == len(files):
                        print(f"  {name} counted {done}/{len(files)} files, kept {kept}", flush=True)
        expected = int(packed[name]["kept"])
        print(f"{name}: counted {kept}, v6 kept {expected}", flush=True)
        if kept == 0 or abs(kept - expected) / expected > 0.01:
            raise SystemExit(f"{name} kept {kept} disagrees with v6 kept {expected}")
        result[name] = {
            "kept": kept,
            "parts": {
                topic: {
                    "examples": int(counts[topic]["examples"]),
                    "tokens": int(counts[topic]["tokens"]),
                    "assistant": int(counts[topic]["assistant"]),
                }
                for topic in TOPICS
                if counts[topic]["examples"]
            },
        }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"wrote {out_path}", flush=True)
    return result
