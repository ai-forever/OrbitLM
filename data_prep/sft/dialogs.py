"""Load chat sources and turn them into prepared user/assistant dialogs.

T-Wix keeps subsets general, reasoning and long_context, and drops
english_corpus. GrandMaster dialogs whose user prompts already appear in
T-Wix are dropped, so the same question is not learned once with a trace and
once without. Aya is the original Russian question/answer subset, not the
SmolTalk Qwen rewrite.
"""
import glob
import hashlib
import json
import os
import unicodedata

import pyarrow.parquet as pq

from render import prepare_messages, prepare_tool_record, strip_think_switch

TWIX_KEEP = ("general", "reasoning", "long_context")
AYA_RUSSIAN = {"russian", "rus", "ru"}


def keep_twix_subset(subset):
    return subset in TWIX_KEEP


def norm_prompt(text):
    text = unicodedata.normalize("NFKC", strip_think_switch(text)).lower()
    return " ".join(text.split())


def user_prompt_key(turns):
    """Stable integer key of the user turns. GrandMaster drops keys already seen in T-Wix."""
    users = [norm_prompt(turn["content"]) for turn in turns if turn["role"] == "user"]
    blob = "\x1e".join(users).encode("utf-8")
    digest = hashlib.blake2b(blob, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def expand_paths(pattern):
    if isinstance(pattern, (list, tuple)):
        found = []
        for item in pattern:
            found.extend(expand_paths(item))
        return found
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches
    if os.path.isfile(pattern):
        return [pattern]
    raise FileNotFoundError(f"no files match {pattern}")


def iter_parquet(paths, columns, limit=0):
    seen = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048, columns=columns):
            for row in batch.to_pylist():
                yield row
                seen += 1
                if limit and seen >= limit:
                    return


def _jsonl_row(raw):
    line = raw.decode("utf-8").strip()
    if not line:
        return None
    return json.loads(line)


def iter_jsonl_range(path, start, end, limit=0):
    """Yield JSON objects from path over the byte half-open range [start, end).

    end None means the end of the file. The range must start and end on a
    line boundary, except that the last line of a file with no trailing
    newline is included when end is the file size. A torn line is not a
    valid record.
    """
    start = 0 if start is None else int(start)
    seen = 0
    with open(path, "rb") as handle:
        handle.seek(start)
        if end is None:
            end = os.path.getsize(path)
        else:
            end = int(end)
        pending = b""
        pos = start
        while pos < end:
            chunk = handle.read(min(1 << 20, end - pos))
            if not chunk:
                break
            pos += len(chunk)
            pending += chunk
            while True:
                newline = pending.find(b"\n")
                if newline < 0:
                    break
                raw = pending[:newline]
                pending = pending[newline + 1:]
                row = _jsonl_row(raw)
                if row is None:
                    continue
                yield row
                seen += 1
                if limit and seen >= limit:
                    return
        if pending:
            row = _jsonl_row(pending)
            if row is not None:
                yield row


def _line_boundary(handle, offset, size):
    """File offset of the first line that begins at or after offset."""
    if offset <= 0:
        return 0
    if offset >= size:
        return size
    handle.seek(offset - 1)
    if handle.read(1) == b"\n":
        return offset
    handle.seek(offset)
    while True:
        pos = handle.tell()
        chunk = handle.read(1 << 20)
        if not chunk:
            return size
        newline = chunk.find(b"\n")
        if newline >= 0:
            return pos + newline + 1


def jsonl_slices(path, pieces, min_bytes=32 * 1024 * 1024):
    """Cover path with at most `pieces` byte ranges, cut on line boundaries.

    A line is never split. A line longer than the nominal width collapses
    neighboring ranges, so the result can be shorter than `pieces`. A file
    too small for two ranges of min_bytes is returned as one range.
    """
    size = os.path.getsize(path)
    pieces = int(pieces)
    if pieces <= 1 or size <= 0:
        return [(0, size)]
    count = min(pieces, max(1, size // int(min_bytes)))
    if count <= 1:
        return [(0, size)]
    bounds = [0]
    with open(path, "rb") as handle:
        for index in range(1, count):
            bounds.append(_line_boundary(handle, size * index // count, size))
    bounds.append(size)
    slices = []
    start = 0
    for end in bounds[1:]:
        if end > start:
            slices.append((start, end))
            start = end
    return slices or [(0, size)]


def iter_jsonl(paths, limit=0, byte_start=None, byte_end=None):
    if byte_start is not None or byte_end is not None:
        if len(paths) != 1:
            raise ValueError("a jsonl byte range applies to one file")
        yield from iter_jsonl_range(paths[0], byte_start, byte_end, limit)
        return
    seen = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                seen += 1
                if limit and seen >= limit:
                    return


def _as_messages(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return None
    return raw


def custom_instructions(row):
    """Persona text SmolTalk stores beside the message list."""
    kwargs = row.get("chat_template_kwargs") if isinstance(row, dict) else None
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs)
    if not isinstance(kwargs, dict):
        return ""
    extra = kwargs.get("custom_instructions")
    return "" if extra is None else str(extra).strip()


def smoltalk_messages(row):
    """Message list, with custom_instructions as a leading system turn.

    The column is read only for a mix that keeps system turns (v5, with or
    without a rewrite table), so the rows of a mix without system turns stay
    unchanged. A message list that already has a system turn keeps that text.
    """
    messages = _as_messages(row.get("messages"))
    if messages is None:
        return None
    extra = custom_instructions(row)
    if not extra:
        return messages
    for message in messages:
        if isinstance(message, dict) and (message.get("role") or "").strip().lower() == "system":
            return messages
    return [{"role": "system", "content": extra}, *messages]


def messages_jsonl(row):
    """The message list of a messages-format jsonl row (stream kind jsonl).

    A row is {"messages": [{"role": ..., "content": ...}, ...]}. A "conversation"
    column holding the same list is accepted too, so that any chat export in
    this shape can be added as a stream.
    """
    return _as_messages(row.get("messages") or row.get("conversation"))


def twix_messages(row):
    if not keep_twix_subset(row.get("subset")):
        return None
    return _as_messages(row.get("messages"))


def aya_messages(row):
    language = str(row.get("language") or row.get("language_code") or "").strip().lower()
    if language not in AYA_RUSSIAN:
        return None
    question = row.get("inputs") or row.get("input") or ""
    answer = row.get("targets") or row.get("target") or row.get("output") or ""
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def acereason_messages(row):
    """AceReason row: input is the question, output is a DeepSeek-R1 trace plus the answer.

    Only code rows are kept. A file that is already the code subset still has
    category set, so a math row cannot slip in.
    """
    category = str(row.get("category") or "").strip().lower()
    if category != "code":
        return None
    question = row.get("input") or ""
    answer = row.get("output") or ""
    return [
        {"role": "user", "content": str(question)},
        {"role": "assistant", "content": str(answer)},
    ]


def grandmaster_messages(row):
    conversation = row.get("conversation") or row.get("messages")
    if conversation is not None:
        return _as_messages(conversation)
    # Some exports store the dialog as a list of contents plus a separate answer.
    inputs = row.get("inputs")
    generated = row.get("generated")
    if isinstance(inputs, list) and generated:
        messages = []
        for item in inputs:
            if isinstance(item, dict):
                messages.append(item)
            else:
                messages.append({"role": "user", "content": str(item)})
        answer = generated[0] if isinstance(generated, list) else generated
        messages.append({"role": "assistant", "content": str(answer)})
        return messages
    return None


def iter_stream_rows(spec, limit=0, with_custom_instructions=False):
    """Yield raw rows for a version-yaml stream."""
    kind = spec["kind"]
    paths = expand_paths(spec["path"])
    if kind in ("smoltalk", "twix", "aya", "grandmaster", "acereason"):
        if kind == "twix":
            columns = ["messages", "subset"]
        elif kind == "smoltalk":
            columns = ["messages", "chat_template_kwargs"] if with_custom_instructions else ["messages"]
        elif kind == "acereason":
            columns = ["input", "output", "category"]
        else:
            columns = None
        yield from iter_parquet(paths, columns, limit=limit)
        return
    if kind == "jsonl":
        yield from iter_jsonl(
            paths, limit=limit,
            byte_start=spec.get("byte_start"), byte_end=spec.get("byte_end"),
        )
        return
    raise ValueError(f"unknown stream kind {kind!r}")


MESSAGE_FNS = {
    "smoltalk": smoltalk_messages,
    "jsonl": messages_jsonl,
    "twix": twix_messages,
    "aya": aya_messages,
    "grandmaster": grandmaster_messages,
    "acereason": acereason_messages,
}


def iter_stream_dialogs(spec, limit=0, seen_prompts=None, rewrite=None, quality_filter=True):
    """Yield prepared dialogs for one stream.

    When seen_prompts is a set, dialogs whose user-prompt key is already in it
    are skipped (GrandMaster vs T-Wix). The key of every yielded dialog is
    added to the set when the stream asks for that via spec['record_prompts'].
    rewrite is the v5 system-prompt table. None folds a task system into the
    first user turn, for a version without system_turns.
    quality_filter drops loops, floods and echoes. Length scans turn it off
    so a long trace is still counted.
    """
    kind = spec["kind"]
    message_fn = MESSAGE_FNS[kind]
    record = bool(spec.get("record_prompts"))
    dedup = bool(spec.get("dedup_prompts"))
    for row in iter_stream_rows(spec, limit=limit, with_custom_instructions=rewrite is not None):
        messages = message_fn(row)
        if messages is None:
            continue
        turns = prepare_messages(messages, rewrite=rewrite, quality_filter=quality_filter)
        if turns is None:
            continue
        if dedup or record:
            key = user_prompt_key(turns)
            if dedup and seen_prompts is not None and key in seen_prompts:
                continue
            if seen_prompts is not None and (record or dedup):
                seen_prompts.add(key)
        yield turns


def iter_tool_dialogs(spec, limit=0, rewrite=None, quality_filter=True):
    """Yield (turns, tools) for a tool_jsonl stream.

    limit counts raw rows, the same way the other stream iterators do.
    A row the tool preparation drops is not yielded.
    """
    for row in iter_jsonl(
        expand_paths(spec["path"]), limit=limit,
        byte_start=spec.get("byte_start"), byte_end=spec.get("byte_end"),
    ):
        prepared = prepare_tool_record(row, rewrite=rewrite, quality_filter=quality_filter)
        if prepared is None:
            continue
        yield prepared
