"""Join SmolTalk OpenThoughts3 rows to the original domain labels.

The SmolTalk export has no domain. The raw parquet does: math, code, science.
The join key is the user question after lowercasing and collapsing whitespace.
Rows that pass prepare_messages and the v6 chain threshold are counted.
Unmatched rows are topic other. A large unmatched share stops the count.

The inputs are the OpenThoughts3-1.2M and SmolTalk2 downloads in
$ORBIT_RAW/hf and the v6 build in $ORBIT_DATA/sft/v6. The counts go to
$ORBIT_DATA/sft/code. --rewrite-table names the table of the v6 build, if it
used one.

  python data_prep/sft/label_ot3.py --match-only
  python data_prep/sft/label_ot3.py
"""
import argparse
import glob
import hashlib
import json
import multiprocessing
import os
import pickle
import sys

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from render import prepare_messages  # noqa: E402
from system_prompts import load_rewrite_table  # noqa: E402

ROOTS = ("ORBIT_RAW", "ORBIT_DATA", "ORBIT_TOKENIZERS")


def _under(root, *parts):
    """A path under one of the ORBIT_* roots. main() stops when a root is not set."""
    return os.path.join(os.environ.get(root, ""), *parts)


RAW_GLOB = _under("ORBIT_RAW", "hf", "open-thoughts__OpenThoughts3-1.2M", "data", "train-*.parquet")
SMOLTALK = {
    "OpenThoughts3_1.2M_think": (
        _under("ORBIT_RAW", "hf", "HuggingFaceTB__smoltalk2", "SFT", "OpenThoughts3_1.2M_think-*.parquet"),
        True,
    ),
    "OpenThoughts3_1.2M_no_think_no_think": (
        _under("ORBIT_RAW", "hf", "HuggingFaceTB__smoltalk2", "SFT", "OpenThoughts3_1.2M_no_think_no_think-*.parquet"),
        False,
    ),
}
STATS_PATH = _under("ORBIT_DATA", "sft", "v6", "stats.json")
TOKENIZER = _under("ORBIT_TOKENIZERS", "tokenizer_qwen3_4b_qwen35_tools")
OUT_PATH = _under("ORBIT_DATA", "sft", "code", "ot3_topics.json")
INDEX_PATH = _under("ORBIT_DATA", "sft", "code", "ot3_domain_index.pkl")
# One JSON per finished input file. A restarted run skips those files.
PARTIAL_ROOT = _under("ORBIT_DATA", "sft", "code", "ot3_partial")
_MATCH = {"domains": None, "rewrite": None}
TOPICS = ("code", "math", "science", "other")
# Above this share of filtered rows, the join is not trusted.
UNMATCHED_STOP = 0.02


def question_key(text):
    normalized = " ".join((text or "").lower().split())
    digest = hashlib.blake2b(normalized.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def first_user(messages):
    for message in messages or []:
        role = (message.get("role") or message.get("from") or "").strip().lower()
        if role in {"user", "human"}:
            return message.get("content") or message.get("value") or ""
    return ""


def load_domains():
    """Map a question key to math, code, or science. Conflicts become None."""
    found = {}
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise FileNotFoundError(RAW_GLOB)
    for index, path in enumerate(files, start=1):
        table = pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=2048, columns=["conversations", "domain"]):
            for conv, domain in zip(
                batch.column("conversations").to_pylist(),
                batch.column("domain").to_pylist(),
            ):
                if domain not in {"math", "code", "science"}:
                    continue
                key = question_key(first_user(conv))
                previous = found.get(key, domain)
                found[key] = domain if previous == domain else None
        if index % 20 == 0 or index == len(files):
            print(f"  indexed {index}/{len(files)} raw files, {len(found)} questions", flush=True)
    return found


def chain_limits():
    with open(STATS_PATH, encoding="utf-8") as handle:
        stats = json.load(handle)
    return {name: int(stats[name]["chain_max"]) for name in SMOLTALK}


def load_rewrite(path=None):
    """The v6 build keeps system turns, so no table means an empty one."""
    return load_rewrite_table(path, system_turns=True)


def iter_smoltalk(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    for path in files:
        table = pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=256, columns=["messages"]):
            for row in batch.column("messages").to_pylist():
                yield row


def topic_of(messages, domains):
    domain = domains.get(question_key(first_user(messages)))
    if domain in {"math", "code", "science"}:
        return domain, "matched"
    if domain is None and question_key(first_user(messages)) in domains:
        return "other", "conflict"
    return "other", "unmatched"


def _match_file(path):
    domains = _MATCH["domains"]
    rewrite = _MATCH["rewrite"]
    counts = {"rows": 0, "prepared": 0, "matched": 0, "conflict": 0, "unmatched": 0}
    misses = []
    table = pq.ParquetFile(path)
    for batch in table.iter_batches(batch_size=256, columns=["messages"]):
        for messages in batch.column("messages").to_pylist():
            counts["rows"] += 1
            prepared = prepare_messages(messages, rewrite=rewrite, quality_filter=True)
            if prepared is None:
                continue
            counts["prepared"] += 1
            _topic, status = topic_of(messages, domains)
            counts[status] += 1
            if status == "unmatched" and len(misses) < 2:
                misses.append(first_user(messages)[:240])
    return counts, misses


def partial_path(kind, stream, source):
    return os.path.join(PARTIAL_ROOT, kind, stream, os.path.basename(source) + ".json")


def load_partial(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def save_partial(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")
    os.replace(temporary, path)


def match_report(domains, rewrite, jobs=32):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    _MATCH["domains"] = domains
    _MATCH["rewrite"] = rewrite
    report = {}
    misses = []
    for name, (pattern, _thinking) in SMOLTALK.items():
        files = sorted(glob.glob(pattern))
        counts = {"rows": 0, "prepared": 0, "matched": 0, "conflict": 0, "unmatched": 0}
        pending = []
        resumed = 0
        for path in files:
            saved = load_partial(partial_path("match", name, path))
            if saved is None:
                pending.append(path)
                continue
            resumed += 1
            for key, value in saved["counts"].items():
                counts[key] += value
            for text in saved.get("misses") or []:
                if len(misses) < 8:
                    misses.append((name, text))
        done = resumed
        if resumed:
            print(f"  {name} resumed {resumed}/{len(files)} match files", flush=True)
        # Fork, so workers see _MATCH. Spawn would reload this file as __main__.
        if pending:
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=min(jobs, len(pending)), mp_context=context) as pool:
                futures = {pool.submit(_match_file, path): path for path in pending}
                for future in as_completed(futures):
                    path = futures[future]
                    part_counts, part_misses = future.result()
                    save_partial(partial_path("match", name, path), {
                        "counts": part_counts,
                        "misses": part_misses,
                    })
                    for key, value in part_counts.items():
                        counts[key] += value
                    for text in part_misses:
                        if len(misses) < 8:
                            misses.append((name, text))
                    done += 1
                    if done % 20 == 0 or done == len(files):
                        print(f"  {name} matched {done}/{len(files)} files", flush=True)
        prepared = counts["prepared"] or 1
        counts["unmatched_share"] = counts["unmatched"] / prepared
        report[name] = counts
        print(
            f"{name}: rows={counts['rows']} prepared={counts['prepared']} "
            f"matched={counts['matched']} conflict={counts['conflict']} "
            f"unmatched={counts['unmatched']} ({counts['unmatched_share']:.4%})",
            flush=True,
        )
    if misses:
        print("unmatched examples:", flush=True)
        for name, text in misses:
            print(f"--- {name} ---", flush=True)
            print(text.replace("\n", " | "), flush=True)
    return report


def load_or_build_domains():
    if os.path.isfile(INDEX_PATH):
        with open(INDEX_PATH, "rb") as handle:
            domains = pickle.load(handle)
        print(f"loaded domain index {len(domains)} questions from {INDEX_PATH}", flush=True)
        return domains
    domains = load_domains()
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "wb") as handle:
        pickle.dump(domains, handle, protocol=pickle.HIGHEST_PROTOCOL)
    conflicts = sum(1 for value in domains.values() if value is None)
    print(f"domain index {len(domains)} questions, {conflicts} conflicts", flush=True)
    return domains


def main():
    parser = argparse.ArgumentParser(description="Label SmolTalk OT3 rows by the original domain.")
    parser.add_argument("--match-only", action="store_true")
    parser.add_argument("--rewrite-table", default=None, help="the rewrite table of the v6 build, if any")
    args = parser.parse_args()
    for root in ROOTS:
        if not os.environ.get(root):
            raise SystemExit(f"set {root} first; data_prep/README.md describes the ORBIT_* roots")
    domains = load_or_build_domains()
    rewrite = load_rewrite(args.rewrite_table)
    report = match_report(domains, rewrite)
    worst = max(item["unmatched_share"] for item in report.values())
    if worst > UNMATCHED_STOP:
        raise SystemExit(
            f"unmatched share {worst:.4%} is above {UNMATCHED_STOP:.0%}; "
            "not writing topic counts"
        )
    if args.match_only:
        return
    # Imported here so --match-only stays free of the tokenizer.
    from label_ot3_count import write_counts  # noqa: E402

    write_counts(domains, rewrite, chain_limits(), OUT_PATH)


if __name__ == "__main__":
    main()
