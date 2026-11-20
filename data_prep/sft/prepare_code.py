"""Build the jsonl streams of the v8 code part.

Sources are Hugging Face downloads in $ORBIT_RAW/hf, in the <org>__<name>
folders that data_prep/long_context/download_hf.py writes. Outputs land in
$ORBIT_DATA/sft/code/, which versions/v8_smoltalk2.yaml reads. Each line is
{"messages": [{"role", "content"}, ...]}. AceReason is sampled separately
by prepare_acereason.py.

  python data_prep/sft/prepare_code.py
"""
import glob
import json
import math
import os
import re

import numpy as np
import pyarrow.parquet as pq

# The roots are read when the module is imported. main() stops when one is not set.
DATA = os.path.join(os.environ.get("ORBIT_RAW", ""), "hf")
OUT_DIR = os.path.join(os.environ.get("ORBIT_DATA", ""), "sft", "code")
SEED = 46
KODCODE_CAP = 80_000

# Field values use underscores. The paper writes the same subsets as
# "Data Structure" and "Code Contests".
KODCODE_DROP = {"Leetcode", "Codeforces", "Apps", "Taco", "Code_Contests"}
KODCODE_KEEP = {"Prefill", "Algorithm", "Data_Structure", "Docs", "Package", "Evol", "Filter"}
KODCODE_ROOT = os.path.join(DATA, "KodCode__KodCode-V1-SFT-R1", "data")
OSS_PATH = os.path.join(
    DATA, "bigcode__self-oss-instruct-sc2-exec-filter-50k", "data", "train-00000-of-00001.parquet",
)
FEEDBACK_PATH = os.path.join(DATA, "m-a-p__Code-Feedback", "Code-Feedback.jsonl")
BIRD_PATH = os.path.join(DATA, "birdsql__bird23-train-filtered", "data", "train-00000-of-00001.jsonl")
BIRD_MEANINGS = os.path.join(DATA, "birdsql__bird23-train-filtered", "train_column_meaning.json")

_FROM = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}
_EXEC_START = re.compile(r"(?i)^execution result\s*:")
_DIAGNOSTIC = re.compile(
    r"(?im)^(?:"
    r"traceback \(most recent call last\)"
    r"|syntaxerror:"
    r"|indentationerror:"
    r"|nameerror:"
    r"|typeerror:"
    r"|valueerror:"
    r"|compilation (?:failed|error)"
    r"|compiler error"
    r"|error:"
    r")"
)
_NEW_TASK = re.compile(
    r"(?i)^(?:write |implement |create |develop |design |you are tasked|please |extend |provide |given )"
)


def user_turn_is_execution(text):
    """True when a user turn is an interpreter dump or a compiler error.

    A later turn that opens like a new assignment is not feedback, even if
    it mentions the word error. Code-Feedback marks real runs with
    'Execution result:'.
    """
    body = (text or "").strip()
    if not body:
        return False
    if _EXEC_START.match(body):
        return True
    first = body.split("\n", 1)[0].strip()
    if _NEW_TASK.match(first):
        return False
    return _DIAGNOSTIC.search(body) is not None


def dialog_has_execution(messages):
    return any(
        turn.get("role") == "user" and user_turn_is_execution(turn.get("content") or "")
        for turn in messages
    )


def subset_quotas(counts, cap):
    """How many rows to keep from each subset. Shares stay when the cap binds."""
    total = sum(counts.values())
    if total <= cap:
        return dict(counts)
    names = sorted(counts)
    raw = {name: cap * counts[name] / total for name in names}
    quotas = {name: int(math.floor(raw[name])) for name in names}
    left = cap - sum(quotas.values())
    order = sorted(names, key=lambda name: (raw[name] - quotas[name], name), reverse=True)
    for name in order:
        if left <= 0:
            break
        quotas[name] += 1
        left -= 1
    return quotas


def sample_positions(n_rows, quota, rng):
    if quota >= n_rows:
        return None
    picked = rng.choice(n_rows, size=quota, replace=False)
    return set(int(index) for index in picked)


def kodcode_messages(conversations):
    messages = []
    for turn in conversations:
        role = _FROM.get(turn.get("from"))
        text = turn.get("value") or ""
        if role is None or not text.strip():
            return None
        if role == "assistant" and "<think>" in text and "</think>" not in text:
            return None
        messages.append({"role": role, "content": text})
    if not messages or messages[-1]["role"] != "assistant":
        return None
    return messages


def bird_schema(db_id, meanings):
    tables = {}
    for key, description in meanings.items():
        database, table, column = key.split("|", 2)
        if database != db_id:
            continue
        tables.setdefault(table, []).append((column, " ".join(str(description).split())))
    if not tables:
        raise KeyError(db_id)
    lines = [f"Database `{db_id}` schema:", ""]
    for table, columns in tables.items():
        lines.append(f"CREATE TABLE {table} (")
        for index, (column, description) in enumerate(columns):
            comma = "," if index + 1 < len(columns) else ""
            lines.append(f"  {column}{comma}  -- {description}")
        lines.append(")")
        lines.append("")
    return "\n".join(lines).strip()


def bird_user(schema, question, evidence):
    lines = [schema, "", f"Question: {question.strip()}"]
    if evidence and str(evidence).strip():
        lines.append(f"Evidence: {str(evidence).strip()}")
    lines.append("Write a SQL query that answers the question.")
    return "\n".join(lines)


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(rows)}", flush=True)


def _kodcode_files():
    files = sorted(glob.glob(os.path.join(KODCODE_ROOT, "train-*.parquet")))
    if not files:
        raise FileNotFoundError(KODCODE_ROOT)
    return files


def kodcode_counts(files=None):
    counts = {}
    unexpected = {}
    files = files or _kodcode_files()
    for path in files:
        table = pq.read_table(path, columns=["subset"])
        for name in table.column("subset").to_pylist():
            if name in KODCODE_DROP:
                continue
            if name not in KODCODE_KEEP:
                unexpected[name] = unexpected.get(name, 0) + 1
                continue
            counts[name] = counts.get(name, 0) + 1
    return counts, unexpected


def write_kodcode(out_path, seed=SEED, cap=KODCODE_CAP):
    files = _kodcode_files()
    counts, unexpected = kodcode_counts(files)
    print("kodcode kept subsets:", flush=True)
    for name in sorted(counts):
        print(f"  {name}: {counts[name]}", flush=True)
    if unexpected:
        shown = ", ".join(f"{name}={count}" for name, count in sorted(unexpected.items()))
        raise RuntimeError(f"unexpected KodCode subsets: {shown}")
    quotas = subset_quotas(counts, cap)
    print(f"kodcode quotas (cap {cap}): {quotas}", flush=True)
    rng = np.random.default_rng(seed)
    want = {
        name: sample_positions(counts[name], quotas[name], rng)
        for name in counts
    }
    seen = {name: 0 for name in counts}
    rows = []
    for path in files:
        table = pq.read_table(path, columns=["subset", "conversations"])
        for subset, conversations in zip(
            table.column("subset").to_pylist(),
            table.column("conversations").to_pylist(),
        ):
            if subset not in counts:
                continue
            index = seen[subset]
            seen[subset] += 1
            chosen = want[subset]
            if chosen is not None and index not in chosen:
                continue
            messages = kodcode_messages(conversations)
            if messages is None:
                continue
            rows.append({"messages": messages})
    _write_jsonl(out_path, rows)
    return len(rows)


def write_self_oss(out_path):
    table = pq.read_table(OSS_PATH, columns=["instruction", "response"])
    rows = []
    for instruction, response in zip(
        table.column("instruction").to_pylist(),
        table.column("response").to_pylist(),
    ):
        if not (instruction or "").strip() or not (response or "").strip():
            continue
        rows.append({"messages": [
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": response.strip()},
        ]})
    _write_jsonl(out_path, rows)
    return len(rows)


def _feedback_preview(path, limit=30):
    hits = []
    misses = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            messages = json.loads(line)["messages"]
            users = [turn for turn in messages if turn.get("role") == "user"]
            later = users[1:]
            matched = next(
                (turn.get("content") or "" for turn in users if user_turn_is_execution(turn.get("content") or "")),
                "",
            )
            if matched:
                if len(hits) < limit:
                    hits.append(matched)
            elif later and len(misses) < limit:
                misses.append(later[0].get("content") or "")
            if len(hits) >= limit and len(misses) >= limit:
                break
    return hits, misses


def write_code_feedback(out_path, preview=True):
    if preview:
        hits, misses = _feedback_preview(FEEDBACK_PATH)
        print(f"code-feedback filter, {len(hits)} hits and {len(misses)} misses:", flush=True)
        for index, text in enumerate(hits, start=1):
            print(f"--- hit {index} ---", flush=True)
            print(text[:500].replace("\n", " | "), flush=True)
        for index, text in enumerate(misses, start=1):
            print(f"--- miss {index} ---", flush=True)
            print(text[:500].replace("\n", " | "), flush=True)
    kept = 0
    seen = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(FEEDBACK_PATH, encoding="utf-8") as source, open(out_path, "w", encoding="utf-8") as dest:
        for line in source:
            seen += 1
            messages = json.loads(line)["messages"]
            if not dialog_has_execution(messages):
                continue
            dest.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {out_path}: {kept} of {seen}", flush=True)
    return kept


def write_bird(out_path):
    with open(BIRD_MEANINGS, encoding="utf-8") as handle:
        meanings = json.load(handle)
    schemas = {}
    rows = []
    missing = 0
    with open(BIRD_PATH, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            db_id = row["db_id"]
            if db_id not in schemas:
                try:
                    schemas[db_id] = bird_schema(db_id, meanings)
                except KeyError:
                    missing += 1
                    continue
            sql = (row.get("SQL") or "").strip()
            question = row.get("question") or ""
            if not sql or not question.strip():
                continue
            rows.append({"messages": [
                {"role": "user", "content": bird_user(schemas[db_id], question, row.get("evidence"))},
                {"role": "assistant", "content": sql},
            ]})
    if missing:
        raise RuntimeError(f"bird databases with no schema: {missing}")
    _write_jsonl(out_path, rows)
    return len(rows)


def main():
    for name in ("ORBIT_RAW", "ORBIT_DATA"):
        if not os.environ.get(name):
            raise SystemExit(f"set {name} first; data_prep/README.md describes the ORBIT_* roots")
    os.makedirs(OUT_DIR, exist_ok=True)
    write_kodcode(os.path.join(OUT_DIR, "kodcode_r1.jsonl"))
    write_self_oss(os.path.join(OUT_DIR, "self_oss_instruct.jsonl"))
    write_code_feedback(os.path.join(OUT_DIR, "code_feedback.jsonl"))
    write_bird(os.path.join(OUT_DIR, "bird_sql.jsonl"))


if __name__ == "__main__":
    main()
