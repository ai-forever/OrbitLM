"""Builds the Russian chat document stores of the 16K stage: twix, grandmaster and oasst2_ru
from Hugging Face sources, and ru_sft from the Russian SFT chats that the user supplies in
paths.RUSSIAN_SFT. Each subcommand reads one source, renders every kept conversation with
documents.render_chat, tokenizes it and writes it to work/docs/<stream>.

Usage:
  build_ru_open.py twix        [--part P --nparts N] [--limit K] [--out DIR] [--overwrite]
  build_ru_open.py grandmaster [--part P --nparts N] [--limit K] [--out DIR] [--overwrite]
  build_ru_open.py oasst2_ru   [--part P --nparts N] [--limit K] [--out DIR] [--overwrite]
  build_ru_open.py ru_sft      [--part P --nparts N] [--limit K] [--out DIR] [--overwrite]

--limit and --out are for tests: they stop after K examples and write the store and the
manifest into DIR instead of the real store.

Order matters: grandmaster drops prompts that are also T-Wix user prompts, so it reads the
prompt hashes that the twix run saves, and twix must finish first.

Splitting: twix gives part P every Nth parquet file, grandmaster every Nth line of its single
jsonl file, oasst2_ru every Nth conversation tree, and ru_sft every Nth role file. Exact
deduplication is done inside a part, so a source that is split into several parts is only
deduplicated within each part. A single part keeps deduplication exact, and the three Hugging
Face sources are small enough to run that way.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np

import documents
import paths

ROOT = paths.CTX16K_ROOT
RAW = paths.RAW_HF
WORK = os.path.join(ROOT, "work", "ru_open")
MAX_TOKENS = 16384  # rendered tokens without the EOS, the cap the 16K stage packs these streams with
# A text this long cannot fit into 16,384 tokens: it would need more than 12 characters per
# token, while Russian and English prose tokenize at 3 to 5. Such texts are dropped as too long
# without spending time on tokenizing them.
MAX_CHARS = 200_000
BATCH = 1000


# ---------------------------------------------------------------- shared helpers

# Qwen3 data marks thinking prompts with a trailing "/think" (or "/no_think") switch. Our plain
# text format has no such switch, since the reasoning is always shown in <think> tags, so the
# marker is removed from user turns.
_THINK_SWITCH = re.compile(r"\s*/(?:no_)?think\s*$")


def strip_think_switch(text):
    return _THINK_SWITCH.sub("", text or "")


def norm(text):
    """Normalizes a prompt for exact deduplication: Unicode NFKC, lower case, one space for
    every run of whitespace, and no Qwen3 think switch."""
    text = unicodedata.normalize("NFKC", strip_think_switch(text)).lower()
    return " ".join(text.split())


def hash64(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def split_think(text):
    """Splits an assistant message into (reasoning, answer).

    Returns (None, text) when the message has no think tags, and None when a <think> is never
    closed, because such an answer was cut off in the middle of its reasoning.
    """
    text = text or ""
    close = text.find("</think>")
    if close < 0:
        return None if "<think>" in text else (None, text)
    open_ = text.find("<think>")
    start = open_ + len("<think>") if 0 <= open_ < close else 0
    before = text[:open_] if 0 <= open_ < close else ""
    reasoning = text[start:close].strip()
    answer = (before + text[close + len("</think>"):]).strip()
    return reasoning, answer


# The language filter counts letters outside code and formulas. Code identifiers and LaTeX
# commands are Latin letters, so a Russian question about code or math would otherwise look
# English. These patterns are Rust regular expressions for polars.
_CODE_AND_MATH = [
    r"(?s)```.*?(?:```|$)",                          # fenced code blocks, also an unclosed last one
    r"`[^`\n]*`",                                    # inline code
    r"(?s)\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)",       # display math and \( \) math
    r"\$[^$\n]+\$",                                  # inline $ math
    r"\\[A-Za-z]+",                                  # any LaTeX command left over
]


def cyrillic_shares(texts):
    """Returns two lists: the share of Cyrillic letters among all letters of each text, counted
    outside code and formulas (used by the filter), and the same share over the raw text
    (reported only, to show how many examples the plain rule would drop)."""
    import polars as pl
    s = pl.Series(texts, dtype=pl.Utf8)
    stripped = s
    for pat in _CODE_AND_MATH:
        stripped = stripped.str.replace_all(pat, " ")
    raw_c = s.str.count_matches(r"\p{Cyrillic}").to_list()
    raw_l = s.str.count_matches(r"\p{L}").to_list()
    st_c = stripped.str.count_matches(r"\p{Cyrillic}").to_list()
    st_l = stripped.str.count_matches(r"\p{L}").to_list()
    filt, raw = [], []
    for rc, rl, sc, sl in zip(raw_c, raw_l, st_c, st_l):
        raw.append(rc / rl if rl else 0.0)
        # A text made only of code or formulas has no letters left after stripping, so it is
        # judged by its raw letters instead.
        filt.append(sc / sl if sl else (rc / rl if rl else 0.0))
    return filt, raw


class Builder:
    """Collects rendered candidates, tokenizes them in batches, applies the length cap and the
    exact deduplication, and writes the survivors to the document store.

    Deduplication keys are checked twice: before tokenizing, against keys already written (to
    skip work on obvious duplicates), and again at write time, because two copies can sit in
    the same batch. A key is remembered only when its document is written, so a duplicate of an
    example dropped for length can still be kept.
    """

    def __init__(self, a, stream):
        self.a, self.stream = a, stream
        self.store = os.path.join(a.out, stream) if a.out else os.path.join(ROOT, "work", "docs", stream)
        prefix = os.path.join(self.store, f"part_{a.part:04d}_")
        old = sorted(glob.glob(prefix + "*"))
        if old and not a.overwrite:
            raise SystemExit(f"{self.store} already has files of part {a.part}; pass --overwrite to rebuild")
        for f in old:
            os.remove(f)
        self.writer = documents.DocStoreWriter(self.store, a.part)
        self.pending = []
        self.seen = set()
        self.st = Counter()
        self.by_group = defaultdict(Counter)
        self.examples = defaultdict(list)
        self.t0 = time.time()

    def drop(self, reason, group=None):
        self.st["drop_" + reason] += 1
        if group is not None:
            self.by_group[group]["drop_" + reason] += 1

    def is_dup(self, key):
        return key in self.seen

    def add(self, text, key, group, on_write=None):
        """Queues one rendered example; on_write is called with it once it is written."""
        if len(text) > MAX_CHARS:
            self.drop("too_long", group)
            return
        self.pending.append((text, key, group, on_write))
        if len(self.pending) >= BATCH:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        docs = documents.encode_docs([p[0] for p in self.pending])
        for (text, key, group, on_write), doc in zip(self.pending, docs):
            if len(doc) - 1 > MAX_TOKENS:
                self.drop("too_long", group)
                continue
            if key in self.seen:
                self.drop("duplicate", group)
                continue
            self.seen.add(key)
            self.writer.add(doc)
            self.st["docs_written"] += 1
            self.st["tokens_written"] += len(doc)
            self.by_group[group]["docs_written"] += 1
            self.by_group[group]["tokens_written"] += len(doc)
            if len(self.examples[group]) < 3:
                self.examples[group].append(text)
            if on_write:
                on_write()
        self.pending = []

    def progress(self, what):
        print(f"{time.strftime('%H:%M:%S')} {self.stream} part {self.a.part}: {what}, "
              f"{json.dumps(dict(self.st), ensure_ascii=False)}", flush=True)

    def finish(self, extra=None):
        self.flush()
        closed = self.writer.close()
        assert closed["docs"] == self.st["docs_written"]
        # Three rendered examples, taken from different groups first so a reader sees each kind.
        picks = []
        for k in range(3):
            picks += [ex[k] for ex in self.examples.values() if len(ex) > k]
        picks = picks[:3]
        man = {
            "stream": self.stream,
            "script": os.path.abspath(__file__),
            "args": vars(self.a),
            "part": self.a.part,
            "nparts": self.a.nparts,
            "store": self.store,
            "examples_read": self.st["examples_read"],
            "drops": {k[5:]: v for k, v in sorted(self.st.items()) if k.startswith("drop_")},
            "docs_written": self.st["docs_written"],
            "tokens_written": self.st["tokens_written"],
            "tokens_note": "tokens include one EOS per document",
            "counters": dict(sorted(self.st.items())),
            "by_group": {g: dict(sorted(c.items())) for g, c in sorted(self.by_group.items())},
            "rendered_examples_first_300_chars": [p[:300] for p in picks],
            "elapsed_s": round(time.time() - self.t0, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            man.update(extra)
        if self.a.out:
            path = os.path.join(self.a.out, f"{self.stream}.manifest.json")
        elif self.a.nparts == 1:
            path = os.path.join(ROOT, "manifests", "build", f"{self.stream}.json")
        else:
            path = os.path.join(ROOT, "manifests", "build", f"{self.stream}.part_{self.a.part:02d}_of_{self.a.nparts:02d}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        self.progress(f"finished, manifest {path}")
        return man


def parse_turns(msgs, stats_group, b):
    """Turns a list of {role, content} messages into (system, turns) for render_chat.

    Returns None, after counting the reason, for anything that is not an optional leading
    system message followed by strictly alternating user and assistant turns that start with a
    user turn and end with an assistant turn.
    """
    system = None
    if msgs and msgs[0]["role"] == "system":
        system = msgs[0]["content"] or ""
        msgs = msgs[1:]
    if not msgs or len(msgs) % 2:
        b.drop("bad_roles", stats_group)
        return None
    turns = []
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] != want:
            b.drop("bad_roles", stats_group)
            return None
        content = m["content"] or ""
        if want == "user":
            stripped = strip_think_switch(content)
            if stripped != content:
                b.st["think_switch_removed"] += 1
            if not stripped.strip():
                b.drop("empty_turn", stats_group)
                return None
            turns.append({"role": "user", "content": stripped})
        else:
            sp = split_think(content)
            if sp is None:
                b.drop("unclosed_think", stats_group)
                return None
            reasoning, answer = sp
            if not answer.strip():
                b.drop("empty_turn", stats_group)
                return None
            t = {"role": "assistant", "content": answer}
            if reasoning:
                t["reasoning"] = reasoning
            turns.append(t)
    return system, turns


def prompt_key(system, turns):
    """The deduplication key of a conversation: its system prompt and all of its user turns."""
    users = [norm(t["content"]) for t in turns if t["role"] == "user"]
    return hash64(norm(system or "") + "\x1f" + "\x1e".join(users))


# ---------------------------------------------------------------- T-Wix

TWIX_KEEP = ("general", "reasoning", "long_context")


def build_twix(a):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(RAW, "t-tech__T-Wix", "data", "train-*.parquet")))[a.part::a.nparts]
    b = Builder(a, "twix")
    user_hashes = []
    shown_lang = 0
    done = False
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for batch in pf.iter_batches(batch_size=2000, columns=["id", "messages", "subset"]):
            cands = []
            for r in batch.to_pylist():
                if a.limit and b.st["examples_read"] >= a.limit:
                    done = True
                    break
                b.st["examples_read"] += 1
                sub = r["subset"]
                b.by_group[sub]["examples_read"] += 1
                if sub == "english_corpus":
                    b.drop("english_corpus", sub)
                    continue
                if sub not in TWIX_KEEP:
                    b.drop("unknown_subset", sub)
                    continue
                parsed = parse_turns(r["messages"], sub, b)
                if parsed is None:
                    continue
                cands.append((sub, r["id"], parsed))
            if cands:
                texts = ["\n".join((t.get("reasoning") or "") + "\n" + t["content"] for t in p[1])
                         for _, _, p in cands]
                shares, raw_shares = cyrillic_shares(texts)
                for (sub, rid, (system, turns)), share, raw, lt in zip(cands, shares, raw_shares, texts):
                    if raw < 0.5:
                        b.st["plain_rule_would_drop"] += 1
                        if share >= 0.5:
                            b.st["kept_by_code_math_rule"] += 1
                    if share < 0.5:
                        b.drop("not_cyrillic", sub)
                        if shown_lang < 5:
                            shown_lang += 1
                            print(f"DROPPED not_cyrillic share={share:.2f} raw={raw:.2f} {sub} {rid}: "
                                  f"{lt[:300]!r}", flush=True)
                        continue
                    key = prompt_key(system, turns)
                    if b.is_dup(key):
                        b.drop("duplicate", sub)
                        continue
                    users = [hash64(norm(t["content"])) for t in turns if t["role"] == "user"]
                    b.add(documents.render_chat(turns, system=system), key, sub,
                          on_write=lambda users=users: user_hashes.extend(users))
            if done:
                break
        b.progress(f"{os.path.basename(fpath)} done")
        if done:
            break
    b.flush()
    # The user prompts of every written T-Wix conversation, for the grandmaster run.
    hdir = a.out or WORK
    os.makedirs(hdir, exist_ok=True)
    hpath = os.path.join(hdir, f"twix_user_prompts.part_{a.part:04d}.npy")
    np.save(hpath + ".tmp.npy", np.unique(np.array(user_hashes, dtype=np.uint64)))
    os.replace(hpath + ".tmp.npy", hpath)
    b.finish({"source": os.path.join(RAW, "t-tech__T-Wix"), "files": [os.path.basename(f) for f in files],
              "language_rule": "drop when Cyrillic letters are under 50% of all letters of the user and "
                               "assistant text (reasoning included), counted outside code blocks, inline "
                               "code and LaTeX math; plain_rule_would_drop counts the same rule on raw text",
              "twix_user_prompt_hashes": hpath})


# ---------------------------------------------------------------- GrandMaster-PRO-MAX-Qwen3-synthetic

def load_twix_prompt_hashes(a):
    hdir = a.twix_hashes or WORK
    files = sorted(glob.glob(os.path.join(hdir, "twix_user_prompts.part_*.npy")))
    if not files:
        raise SystemExit(f"no twix_user_prompts.part_*.npy in {hdir}: run the twix build first")
    return set(np.concatenate([np.load(f) for f in files]).tolist()), files


def build_grandmaster(a):
    src = os.path.join(RAW, "RefalMachine__GrandMaster-PRO-MAX-Qwen3-synthetic", "train.jsonl")
    twix, twix_files = load_twix_prompt_hashes(a)
    b = Builder(a, "grandmaster")
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % a.nparts != a.part:
                continue
            if a.limit and b.st["examples_read"] >= a.limit:
                break
            b.st["examples_read"] += 1
            r = json.loads(line)
            inputs = r.get("inputs") or []
            if len(inputs) != 1 or inputs[0].get("role") != "user" or not (inputs[0].get("content") or "").strip():
                b.drop("bad_input", "all")
                continue
            prompt = strip_think_switch(inputs[0]["content"])
            key = hash64(norm(prompt))
            if key in twix:
                b.drop("in_twix", "all")
                continue
            if b.is_dup(key):
                b.drop("duplicate", "all")
                continue
            # The first answer whose reasoning is closed and whose final answer is not empty.
            chosen = None
            for k, g in enumerate(r.get("generated") or []):
                sp = split_think(g) if isinstance(g, str) else None
                if sp and sp[0] is not None and sp[0].strip() and sp[1].strip():
                    chosen = (k, sp)
                    break
            if chosen is None:
                b.drop("no_closed_think", "all")
                continue
            k, (reasoning, answer) = chosen
            b.st[f"answer_index_{k}"] += 1
            turns = [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": answer, "reasoning": reasoning}]
            b.add(documents.render_chat(turns), key, "all")
            if b.st["examples_read"] % 10000 == 0:
                b.progress(f"{b.st['examples_read']} read")
    b.finish({"source": src, "twix_prompt_hash_files": twix_files, "twix_prompt_hashes": len(twix),
              "answer_rule": "first of the 3 generated answers with a closed </think>, non-empty "
                             "reasoning and a non-empty final answer"})


# ---------------------------------------------------------------- OASST2, Russian

# Replies that speak about themselves as a language model or an AI. The patterns need "я" (or a
# verb like "являюсь" or "будучи") next to the AI words, because a reply that merely talks about
# AI as a topic, as in "такие как искусственный интеллект", must stay. "Как ИИ" therefore only
# counts at the start of a sentence or right after "я".
_SELF_AI = re.compile("|".join([
    r"как\s+(?:большая\s+)?языков(?:ая|ой)\s+модел",
    r"\bя\s*[—–-]?\s*(?:[\w-]+\s+){0,3}?"
    r"(?:языков(?:ая|ой|ую)\s+модел|нейросет|нейронн(?:ая|ой)\s+сет|искусственн(?:ый|ым)\s+интеллект|ии\b|чат-?бот)",
    r"(?:являюсь|будучи|я\s+всего\s+лишь)\s+(?:[\w-]+\s+){0,2}?"
    r"(?:языков|нейросет|нейронн|искусствен|ии\b|ai\b|программой\b|(?:чат-?)?бот(?:ом)?\b)",
    r"(?:^|[.!?…]\s+|\bя,?\s+)как\s+(?:ии|искусственный\s+интеллект|модель\s+ии)\b",
    # "Я всего лишь лингвистическая модель", including misspelled adjectives. The adjective
    # must end like one, so "я построил модель" does not match.
    r"\bя\s*[—–-]?\s*(?:всего\s+лишь\s+|лишь\s+|просто\s+|только\s+|же\s+|ведь\s+)?(?:[\w-]+(?:ая|ой)\s+)+модел(?:ь|ью)\b",
    r"as an ai language model|as a language model|as an ai\b|\bi am an ai\b|\bi'm an ai\b",
]), re.I | re.M)


def build_oasst2_ru(a):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(RAW, "OpenAssistant__oasst2", "data", "*.parquet")))
    cols = ["message_id", "parent_id", "text", "role", "lang", "rank", "deleted", "review_result",
            "created_date", "message_tree_id"]
    b = Builder(a, "oasst2_ru")
    msgs, children, roots = {}, defaultdict(list), []
    for fpath in files:
        for batch in pq.ParquetFile(fpath).iter_batches(batch_size=5000, columns=cols):
            for m in batch.to_pylist():
                b.st["messages_read"] += 1
                if m["lang"] != "ru":
                    continue
                b.st["ru_messages"] += 1
                # Deleted messages and messages that failed review are skipped with their subtree.
                if m["deleted"] or m["review_result"] is False:
                    b.st["ru_messages_deleted_or_failed_review"] += 1
                    continue
                if hash64(m["message_tree_id"]) % a.nparts != a.part:
                    continue
                msgs[m["message_id"]] = m
    for mid, m in msgs.items():
        if m["parent_id"] is None:
            if m["role"] == "prompter":
                roots.append(mid)
        elif m["parent_id"] in msgs:
            children[m["parent_id"]].append(mid)
    roots.sort(key=lambda x: msgs[x]["created_date"])

    def self_ai(text):
        return bool(_SELF_AI.search(text or ""))

    def best_assistant(pid, count=False):
        """The best-ranked reply that does not describe itself as an AI. Rank 0 is best, and an
        unranked reply (usually the only reply) comes after every ranked one. The statistics are
        counted only for the path that is actually taken, not for look-ahead calls."""
        cands = [c for c in children[pid] if msgs[c]["role"] == "assistant" and (msgs[c]["text"] or "").strip()]
        good = [c for c in cands if not self_ai(msgs[c]["text"])]
        good.sort(key=lambda c: (msgs[c]["rank"] is None, msgs[c]["rank"] or 0, msgs[c]["created_date"]))
        if count:
            b.st["replies_self_ai_skipped"] += len(cands) - len(good)
            if good and msgs[good[0]]["rank"] is None:
                b.st["steps_unranked_reply"] += 1
            elif good and msgs[good[0]]["rank"] != 0:
                b.st["steps_best_rank_not_0"] += 1
        return good[0] if good else None

    depth_cache = {}

    def depth(uid):
        """How many turns the best-reply path starting at this user message would contain."""
        if uid not in depth_cache:
            aid = best_assistant(uid)
            d = 0
            if aid is not None:
                nxt = [depth(c) for c in children[aid] if msgs[c]["role"] == "prompter"]
                d = 2 + max(nxt, default=0)
            depth_cache[uid] = d
        return depth_cache[uid]

    def next_user(aid):
        """The follow-up question that continues the conversation furthest, the earliest on ties.
        User messages carry no ranking, so the length of the thread is the signal used."""
        cands = [c for c in children[aid] if msgs[c]["role"] == "prompter"
                 and (msgs[c]["text"] or "").strip() and depth(c) > 0]
        return min(cands, key=lambda c: (-depth(c), msgs[c]["created_date"])) if cands else None

    for rid in roots:
        if a.limit and b.st["examples_read"] >= a.limit:
            break
        b.st["examples_read"] += 1
        turns, uid = [], rid
        while uid is not None:
            aid = best_assistant(uid, count=True)
            if aid is None:
                break
            turns.append({"role": "user", "content": msgs[uid]["text"]})
            turns.append({"role": "assistant", "content": msgs[aid]["text"]})
            uid = next_user(aid)
        if not turns:
            b.drop("no_usable_reply", "all")
            continue
        b.st[f"turn_pairs_{min(len(turns) // 2, 5)}{'+' if len(turns) // 2 >= 5 else ''}"] += 1
        key = prompt_key(None, turns)
        if b.is_dup(key):
            b.drop("duplicate", "all")
            continue
        b.add(documents.render_chat(turns), key, "all")
    b.finish({"source": files, "trees_with_ru_root": len(roots),
              "path_rule": "from the root prompt, the best-ranked assistant reply at every step "
                           "(rank 0, unranked after ranked), skipping replies that call themselves "
                           "a language model or an AI; the follow-up user turn with the longest "
                           "continuation; the path ends on an assistant turn"})


# ---------------------------------------------------------------- Russian SFT chats

# The role files of paths.RUSSIAN_SFT that the ru_sft store reads. The tasks and education roles
# hold short task items rather than conversations, so the context-extension recipe keeps them out of
# this chat stream. The SFT version yaml reads every role.
RU_SFT_ROLES = ("science", "math", "code", "dialog", "creative", "qa_reasoning", "summarization")


def build_ru_sft(a):
    """Renders the user's Russian SFT chats, one messages-format jsonl file per role.

    Every line is {"messages": [{"role": ..., "content": ...}, ...]}. A chat goes through the
    same role check, think split and prompt deduplication as T-Wix. The chats are expected to be
    clean already, so no further content filter runs here. A role file that is missing is
    skipped and listed in the manifest, so a user who has only some of the roles can still build
    the store.
    """
    present, missing = [], []
    for role in RU_SFT_ROLES:
        path = os.path.join(paths.RUSSIAN_SFT, f"{role}.jsonl")
        (present if os.path.isfile(path) else missing).append(path)
    if not present:
        raise SystemExit(f"no role file of {', '.join(RU_SFT_ROLES)} in {paths.RUSSIAN_SFT}; "
                         "data_prep/README.md describes the folder")
    files = present[a.part::a.nparts]
    b = Builder(a, "ru_sft")
    done = False
    for fpath in files:
        role = os.path.basename(fpath)[:-len(".jsonl")]
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if a.limit and b.st["examples_read"] >= a.limit:
                    done = True
                    break
                b.st["examples_read"] += 1
                b.by_group[role]["examples_read"] += 1
                parsed = parse_turns(json.loads(line).get("messages") or [], role, b)
                if parsed is None:
                    continue
                system, turns = parsed
                key = prompt_key(system, turns)
                if b.is_dup(key):
                    b.drop("duplicate", role)
                    continue
                b.add(documents.render_chat(turns, system=system), key, role)
        b.progress(f"{os.path.basename(fpath)} done")
        if done:
            break
    b.finish({"source": paths.RUSSIAN_SFT, "files": [os.path.basename(f) for f in files],
              "missing_role_files": [os.path.basename(f) for f in missing]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream", choices=["twix", "grandmaster", "oasst2_ru", "ru_sft"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--twix-hashes", default=None, help="directory with twix_user_prompts.part_*.npy")
    a = ap.parse_args()
    assert 0 <= a.part < a.nparts
    {"twix": build_twix, "grandmaster": build_grandmaster, "oasst2_ru": build_oasst2_ru,
     "ru_sft": build_ru_sft}[a.stream](a)


if __name__ == "__main__":
    main()
