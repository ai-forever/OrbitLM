"""Renders two English chat sources into document stores for the 16K stage.

Streams:
  nemotron_if   nvidia/Nemotron-Instruction-Following-Chat-v1 (chat_if.jsonl, structured_outputs.jsonl)
  smoltalk2_en  HuggingFaceTB/smoltalk2, config SFT, three *_think subsets

Every conversation is rendered with documents.render_chat, tokenized with EOS appended, and
written to work/docs/<stream>/ (pack.py packs it later).

Rows are dropped when a turn claims a model identity (Nemotron/NVIDIA or any other named model
or AI lab), when the rendered conversation is longer than 16,384 tokens, when it is an exact
duplicate of an earlier row, or when it is malformed. Every drop is counted by reason.

Parts: a row goes to part (hash of its dedup key) mod nparts. Every part therefore reads all
input rows, but duplicates always land in the same part, so deduplicating inside each part is
exact for the whole source. The sources are small enough that reading them nparts times is cheap
next to tokenizing.

Usage:
  build_chat_en.py build --stream nemotron_if --part P --nparts N [--limit ROWS] [--out DIR]
  build_chat_en.py manifest --stream nemotron_if --nparts N
"""
import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter

import documents
import paths

ROOT = paths.CTX16K_ROOT
WORK = os.path.join(ROOT, "work", "chat_en")
MAX_TOKENS = 16384  # rendered tokens, without the EOS that encode_docs appends

SOURCES = {
    "nemotron_if": [
        ("chat_if", "jsonl", os.path.join(paths.RAW_HF, "nvidia__Nemotron-Instruction-Following-Chat-v1/data/chat_if.jsonl")),
        ("structured_outputs", "jsonl", os.path.join(paths.RAW_HF, "nvidia__Nemotron-Instruction-Following-Chat-v1/data/structured_outputs.jsonl")),
    ],
    "smoltalk2_en": [
        (n, "parquet", os.path.join(paths.RAW_HF, f"HuggingFaceTB__smoltalk2/SFT/{n}-00000-of-00001.parquet"))
        for n in ("smoltalk_systemchats_Qwen3_32B_think", "multi_turn_reasoning_if_think",
                  "smoltalk_everyday_convs_reasoning_Qwen3_32B_think")
    ],
}

# ---------------------------------------------------------------- identity filter
#
# The 16K stage trains on every token, user turns included, so a conversation in which the
# assistant introduces itself as another model teaches our model that identity. Nemotron data
# signs about 19% of its answers with "Nemotron, created by NVIDIA", and SmolTalk2 has system
# prompts such as "You are the AI model named GPT-5".
#
# Plain mentions of the companies stay: answers about NVIDIA graphics cards or OpenAI's API are
# normal content. Only first-person claims are identity claims, so every pattern below except
# the Nemotron one needs "I am", "my name is", "you are" or a similar self-reference.

ORG = (r"(?:nvidia|open\s?ai|anthropic|google(?:\s+deepmind)?|deepmind|alibaba(?:\s+cloud)?|meta(?:\s+ai)?"
       r"|mistral(?:\s+ai)?|deepseek|hugging\s?face|microsoft|xai|x\.ai|moonshot(?:\s+ai)?|zhipu(?:\s+ai)?"
       r"|baidu|tencent|cohere)\b")
# Model names. Bard, Kimi and Ernie are also ordinary words or first names, so only their
# unambiguous forms are listed.
NAME = (r"(?:chat\s?gpt|gpt-?\s?\d(?:\.\d)?(?:o|-?turbo)?|gpt|open\s?ai|claude|gemini|google\s+bard|llama(?:\s?\d(?:\.\d)?)?"
        r"|qwen\d*(?:\.\d)?|tongyi(?:\s+qianwen)?|mistral|mixtral|deepseek|smollm\d*|copilot|grok|nemotron"
        r"|ernie\s?bot|phi-\d)\b")
VERB = (r"(?:created|developed|trained|built|made|designed|engineered|crafted|forged|programmed|fine-?tuned"
        r"|brought\s+to\s+life|powered)")
# A dash may sit between the verb and "by": "I was built — by NVIDIA, yes".
BY_ORG = (VERB + r"\s+(?:[—–-]\s*)?(?:\w+\s+){0,2}by\s+(?:the\s+(?:team|folks|people|engineers|researchers)\s+(?:at|of)\s+)?"
          + ORG)
# Running on someone's servers is also a statement of where the model comes from.
RUN_BY_ORG = r"(?:managed|run|operated|hosted|owned)\s+by\s+" + ORG
AI_NOUN = r"(?:ai|assistant|model|llm|chatbot|language\s+model)"

# First-person forms for answers, second-person forms for system prompts ("You are ...").
SELF_BE = {"assistant": r"\b(?:I\s+was|I['’]m|I\s+am|I['’]ve\s+been|I\s+have\s+been)\b",
           "system": r"\b(?:you\s+were|you['’]re|you\s+are|you['’]ve\s+been|you\s+have\s+been)\b"}
SELF_NAME = {"assistant": r"\b(?:I['’]m|I\s+am|my\s+name\s+is|call\s+me|I\s+go\s+by)\s+(?:called\s+|named\s+|actually\s+|just\s+)?",
             "system": r"\b(?:you['’]re|you\s+are|your\s+name\s+is)\s+(?:called\s+|named\s+|actually\s+)?"}
# "My creators" always means the model's makers, so the organisation may come a little later
# ("my creator's workshop at 3 a.m. NVIDIA."), punctuation included. "My team" and "my
# developers" are left out on purpose: answers often write letters for the user, and "my team
# at Google" in a cover letter is the user's team, not the model's.
CREATORS = r"(?:creators?|makers?|designers?)(?:['’]s?)?"
SELF_CREATOR = {"assistant": r"\bmy\s+[*_]*" + CREATORS + r"\b[\s\S]{0,120}?\b",
                "system": r"\byour\s+[*_]*" + CREATORS + r"\b[\s\S]{0,120}?\b"}
SELF_OBJ = {"assistant": r"me", "system": r"you"}
SELF_POSS = {"assistant": r"my", "system": r"your"}


def _identity_patterns(side):
    pats = {
        # "I was built by NVIDIA", "I'm a large language model developed by NVIDIA".
        "self_made_by_org": SELF_BE[side] + r"[^.!?\n]{0,90}?\b(?:" + BY_ORG + "|" + RUN_BY_ORG + ")",
        # "I'm an AI assistant from NVIDIA".
        "self_ai_from_org": SELF_BE[side] + r"\s+(?:\w+\s+){0,4}" + AI_NOUN + r"\s+(?:from|by|at|of)\s+" + ORG,
        # "I'm NVIDIA's AI", "I am OpenAI's model".
        "self_orgs_ai": SELF_BE[side] + r"\s+(?:\w+\s+){0,2}?" + ORG + r"['’]s\s+(?:own\s+)?(?:\w+\s+)?" + AI_NOUN,
        # "I'm ChatGPT", "my name is OpenAI", "You are GPT-5". No free word is allowed between
        # the verb and the name, because "I'm using GPT-4" is not a claim.
        "self_named_model": SELF_NAME[side] + r"(?:the\s+)?" + NAME,
        # "I am an AI model named GPT-5".
        "self_model_named": SELF_BE[side] + r"\s+(?:\w+\s+){0,3}" + AI_NOUN + r"\s+(?:named|called)\s+" + NAME,
        # "my creators at OpenAI".
        "self_creator_org": SELF_CREATOR[side] + ORG,
        # "Companies like NVIDIA (yeah, my creators)".
        "org_my_creator": ORG + r"[\s\S]{0,40}?\b" + SELF_POSS[side] + r"\s+(?:creators?|makers?)\b",
        # "NVIDIA made me", "NVIDIA didn't build me just to fix Bluetooth".
        "org_made_self": (ORG + r"\s+(?:[\w’']+\s+){0,2}(?:made|created|built|build|trained|designed|developed|make|create|train|design)\s+"
                          + SELF_OBJ[side] + r"\b"),
        # "As an AI assistant created by NVIDIA", "someone like me, an AI built by NVIDIA",
        # "I may be an AI built by NVIDIA", "being an AI developed by NVIDIA", "not even one created by NVIDIA".
        "as_ai_made_by_org": (r"\b(?:as|being|be|even|am|I['’]m|" + SELF_OBJ[side] + r",?)\s+(?:an?\s+(?:\w+\s+){0,3}" + AI_NOUN
                              + r"|one|someone|something)[,\s]+(?:\w+\s+){0,2}?" + BY_ORG),
        # "NVIDIA's AI models (like me)", "developers like me at NVIDIA".
        "org_ai_like_self": (ORG + r"['’]s\s+(?:\w+\s+){0,3}(?:ai|models?|assistants?|systems?)\s*[,(]?\s*(?:like|such\s+as|including)\s+"
                             + SELF_OBJ[side] + r"\b"),
        "like_self_at_org": r"\b(?:like|including)\s+" + SELF_OBJ[side] + r"\s+at\s+" + ORG,
        # "my NVIDIA-built neurons".
        "self_org_built": r"\b" + SELF_POSS[side] + r"\s+" + ORG + r"[-\s](?:built|made|designed|trained|powered)\b",
        # "forged in the quiet labs of NVIDIA", "I was still a gleam in NVIDIA's R&D lab".
        "self_in_org_lab": (SELF_BE[side] + r"[^.!?\n]{0,90}?\b(?:(?:labs?|workshops?|servers?|data\s+cent(?:er|re)s?)\s+(?:of|at)\s+"
                            + ORG + "|" + ORG + r"['’]s\s+(?:[\w&]+\s+){0,2}(?:labs?|workshops?)\b)"),
        # "As someone who's been created by NVIDIA".
        "as_someone_made_by_org": r"\bas\s+(?:someone|something|one)\s+(?:who|that)(?:['’]s|\s+was|\s+is|\s+has)?\s+(?:been\s+)?" + BY_ORG,
        # "I'm proud to be part of NVIDIA's broader mission".
        "self_part_of_org": SELF_BE[side] + r"\s+(?:\w+\s+){0,3}part\s+of\s+" + ORG,
    }
    if side == "assistant":
        # Nemotron answers often end with a signature that has no "I" in it, such as
        # "— Created by NVIDIA, but endlessly curious", "(created by NVIDIA — but today, just a
        # witness)" or "— N, created by NVIDIA". The verb then starts the line or follows a dash,
        # which a sentence about a product almost never does. After a bracket or a full stop only
        # "created" counts, because "(made by NVIDIA)" and "Developed by NVIDIA, CUDA ..." are
        # ordinary product descriptions.
        pats["signoff_made_by_org"] = (r"(?:(?:^|\n|[—–>_]|\s-\s)\s*" + BY_ORG + r"|(?:\(|[.!?]\s)\s*created\s+by\s+" + ORG
                                       + r"|[—–]\s*[A-Z]\w{0,12},\s+" + BY_ORG + ")")
    if side == "system":
        # A system prompt can name the model without "you": "The AI model named GPT-5 ...". In
        # answers the same words usually describe a real model ("a language model called GPT-3"),
        # so this pattern is used for system prompts only.
        pats["model_named"] = r"\b" + AI_NOUN + r"\s+(?:named|called)\s+" + NAME
    return {k: re.compile(v, re.I) for k, v in pats.items()}


# System prompts are checked with both sets: "You are GPT-5" is a claim, and so is an instruction
# that quotes the answer to give, such as "this is correct: 'my name is OpenAI'".
ID_PATTERNS = {"assistant": _identity_patterns("assistant"),
               "system": {**_identity_patterns("system"),
                          **{"sys_first_person_" + k: v for k, v in _identity_patterns("assistant").items()}}}
# Misspellings such as "Nemtron" occur in the signatures too.
NEMOTRON = re.compile(r"nem[a-z]{0,3}tron", re.I)
# A cheap pre-check: the detailed patterns run only on texts that mention an organisation or a
# model name at all, which is a small fraction of all texts.
TRIGGER = re.compile(ORG + "|" + NAME, re.I)
NVIDIA_WORD = re.compile(r"nvidia", re.I)
# A deliberately loose pattern for rows that were kept but still mention NVIDIA near a
# self-reference or a "made by" phrase. It drops nothing; a sample of its hits is saved in the
# statistics so a human can see what the identity filter lets through.
SUSPECT = re.compile(r"(?:created|built|made|crafted|designed|developed|trained|engineered|powered|born)\s+(?:\w+\s+){0,2}"
                     r"(?:by|at|in)\s+(?:the\s+\w+\s+(?:at|of)\s+)?nvidia|nvidia[^.\n]{0,40}\b(?:me|my|I)\b"
                     r"|\b(?:me|my|I)\b[^.\n]{0,40}nvidia", re.I)


def identity_hit(texts_by_side):
    """Returns (pattern name, matched text, context) for the first identity claim, or None.

    texts_by_side maps "assistant" and "system" to lists of texts. User turns are checked only
    for the word Nemotron, because a user asking "are you ChatGPT?" is not a claim.
    """
    for side, texts in texts_by_side.items():
        for t in texts:
            # Markdown emphasis splits the phrases the patterns look for ("I *was* created by
            # **NVIDIA**"), so asterisks are removed before matching.
            t = t.replace("*", "")
            m = NEMOTRON.search(t)
            if m:
                return "nemotron", m.group(0), t[max(0, m.start() - 150):m.end() + 100]
            if side == "user" or not TRIGGER.search(t):
                continue
            for name, p in ID_PATTERNS[side].items():
                m = p.search(t)
                if m:
                    return name, m.group(0), t[max(0, m.start() - 150):m.end() + 100]
    return None


# ---------------------------------------------------------------- reading rows


def iter_rows(kind, path):
    if kind == "jsonl":
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=2000):
            yield from b.to_pylist()


def split_think(text):
    """Splits an assistant text that carries its reasoning inline as <think>...</think>.

    Returns (reasoning, answer), or None when a <think> is opened and never closed, because
    such a text is a generation cut off in the middle of its reasoning.
    """
    if "</think>" not in text:
        if "<think>" in text:
            return None
        return "", text
    head, _, tail = text.partition("</think>")
    # Some generators drop the opening tag, so everything before </think> is the reasoning.
    reasoning = head.split("<think>", 1)[1] if "<think>" in head else head
    if "<think>" in tail or "</think>" in tail:
        return None
    return reasoning.strip(), tail.strip()


def to_conversation(stream, row):
    """Turns one source row into (system, turns), or raises Drop with the reason."""
    if stream == "nemotron_if":
        msgs = row["messages"]
        system = None
        if msgs and msgs[0]["role"] == "system":
            system = (msgs[0].get("content") or "").strip() or None
            msgs = msgs[1:]
        turns = []
        for m in msgs:
            role = m["role"]
            if role not in ("user", "assistant"):
                raise Drop(f"role_{role}")
            content = m.get("content") or ""
            reasoning = (m.get("reasoning_content") or "").strip()
            if role == "assistant" and ("<think>" in content or "</think>" in content):
                sp = split_think(content)
                if sp is None:
                    raise Drop("unclosed_think")
                inline_r, content = sp
                reasoning = reasoning or inline_r
            turns.append({"role": role, "content": content, "reasoning": reasoning})
    else:
        kw = row.get("chat_template_kwargs") or {}
        if kw.get("python_tools") or kw.get("xml_tools"):
            raise Drop("has_tools")
        system = (kw.get("custom_instructions") or "").strip() or None
        turns = []
        for m in row["messages"]:
            role = m["role"]
            if role == "system":
                if turns or system:
                    raise Drop("extra_system")
                system = (m.get("content") or "").strip() or None
                continue
            if role not in ("user", "assistant"):
                raise Drop(f"role_{role}")
            content, reasoning = m.get("content") or "", ""
            if role == "assistant":
                sp = split_think(content)
                if sp is None:
                    raise Drop("unclosed_think")
                reasoning, content = sp
            turns.append({"role": role, "content": content, "reasoning": reasoning})
    # A conversation that ends with an unanswered user turn is trimmed to its last answer.
    while turns and turns[-1]["role"] == "user":
        turns.pop()
    if not any(t["role"] == "assistant" for t in turns):
        raise Drop("no_assistant")
    if turns[0]["role"] != "user" or any(a["role"] == b["role"] for a, b in zip(turns, turns[1:])):
        raise Drop("bad_alternation")
    if any(not t["content"].strip() for t in turns):
        raise Drop("empty_turn")
    return system, turns


class Drop(Exception):
    pass


def norm(s):
    return " ".join((s or "").lower().split())


def dedup_key(system, turns):
    # The prompt is the system prompt plus every user turn: two rows that ask the same things
    # in the same order are the same example even if the answers were sampled differently.
    s = norm(system) + "\x00" + "\x00".join(norm(t["content"]) for t in turns if t["role"] == "user")
    return hashlib.md5(s.encode("utf-8")).digest()


# ---------------------------------------------------------------- build


def build(a):
    out = a.out or os.path.join(ROOT, "work", "docs", a.stream)
    stats_path = os.path.join(a.stats_dir or WORK, f"{a.stream}.part_{a.part:02d}.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    # A rebuild must start from an empty part: new files would otherwise sit next to the old
    # run's files and pack.py would pack both.
    if os.path.isdir(out) and any(f.startswith(f"part_{a.part:04d}_") for f in os.listdir(out)):
        raise SystemExit(f"{out} already has files of part {a.part}; remove them to rebuild")
    writer = documents.DocStoreWriter(out, a.part)
    st = Counter()
    per_file = {}
    lens = Counter()
    examples, id_examples, kept_nvidia = [], [], []
    seen = set()
    pending = []  # (rendered text, source file name) waiting to be tokenized as one batch
    t0 = time.time()

    def flush():
        docs = documents.encode_docs([p[0] for p in pending])
        for (text, fname), doc in zip(pending, docs):
            n = len(doc) - 1
            if n > MAX_TOKENS:
                st["drop_too_long"] += 1
                per_file[fname]["drop_too_long"] += 1
                continue
            writer.add(doc)
            st["docs"] += 1
            st["tokens"] += len(doc)
            per_file[fname]["docs"] += 1
            per_file[fname]["tokens"] += len(doc)
            lens[next(b for b in (1024, 2048, 4096, 8192, 16384) if n <= b)] += 1
            if sum(e["file"] == fname for e in examples) < 2:
                examples.append({"file": fname, "text": text[:300]})
        pending.clear()

    for fname, kind, path in SOURCES[a.stream]:
        pf = per_file.setdefault(fname, Counter())
        for i, row in enumerate(iter_rows(kind, path)):
            if a.limit and i >= a.limit:
                break
            try:
                system, turns = to_conversation(a.stream, row)
            except Drop as e:
                # Malformed rows cannot be hashed by prompt, so they are split by row number;
                # this only affects which part counts the drop.
                if i % a.nparts == a.part:
                    st["rows"] += 1
                    pf["rows"] += 1
                    st[f"drop_{e}"] += 1
                    pf[f"drop_{e}"] += 1
                continue
            key = dedup_key(system, turns)
            if int.from_bytes(key[:8], "little") % a.nparts != a.part:
                continue
            st["rows"] += 1
            pf["rows"] += 1
            if key in seen:
                st["drop_duplicate"] += 1
                pf["drop_duplicate"] += 1
                continue
            seen.add(key)
            last = max(j for j, t in enumerate(turns) if t["role"] == "assistant")
            answers = [t["content"] for t in turns if t["role"] == "assistant"] + [turns[last]["reasoning"]]
            hit = identity_hit({"system": [system] if system else [],
                                "user": [t["content"] for t in turns if t["role"] == "user"],
                                "assistant": answers})
            if hit:
                st["drop_identity"] += 1
                st[f"identity_{hit[0]}"] += 1
                pf["drop_identity"] += 1
                if len(id_examples) < 8:
                    id_examples.append({"pattern": hit[0], "match": hit[1], "context": hit[2].replace("\n", " ")})
                continue
            for t in answers:
                if NVIDIA_WORD.search(t):
                    m = SUSPECT.search(t)
                    if m:
                        st["kept_suspect_nvidia"] += 1
                        if len(kept_nvidia) < 30:
                            kept_nvidia.append(t[max(0, m.start() - 150):m.end() + 100].replace("\n", " "))
                        break
            if turns[last]["reasoning"]:
                st["with_reasoning"] += 1
            if system:
                st["with_system"] += 1
            pending.append((documents.render_chat(turns, system=system), fname))
            if len(pending) >= 512:
                flush()
            if st["rows"] % 20000 == 0:
                el = time.time() - t0
                print(f"[{el:7.0f}s] {fname} row {i} part rows {st['rows']} docs {st['docs']} tokens {st['tokens'] / 1e6:.1f}M", flush=True)
        if pending:
            flush()
    res = writer.close()
    assert res["docs"] == st["docs"] and res["tokens"] == st["tokens"]
    result = {"stream": a.stream, "part": a.part, "nparts": a.nparts, "limit": a.limit, "out": out,
              "stats": dict(st), "per_file": {k: dict(v) for k, v in per_file.items()},
              "length_buckets": {f"<={k}": v for k, v in sorted(lens.items())},
              "examples": examples, "identity_drop_examples": id_examples,
              "kept_suspect_nvidia_mentions": kept_nvidia, "seconds": round(time.time() - t0)}
    tmp = stats_path + ".tmp"
    json.dump(result, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, stats_path)
    print(json.dumps(result, indent=1, ensure_ascii=False))


def _spread(examples, k):
    """Picks k examples, taking them from as many different source files as possible."""
    firsts = [e for i, e in enumerate(examples) if all(x["file"] != e["file"] for x in examples[:i])]
    rest = [e for e in examples if e not in firsts]
    return [e["text"] for e in (firsts + rest)[:k]]


def manifest(a):
    parts = [json.load(open(os.path.join(WORK, f"{a.stream}.part_{p:02d}.json"))) for p in range(a.nparts)]
    assert all(p["nparts"] == a.nparts and not p["limit"] for p in parts), "stats are from a different run"
    tot, per_file, lens = Counter(), {}, Counter()
    for p in parts:
        tot.update(p["stats"])
        for f, s in p["per_file"].items():
            per_file.setdefault(f, Counter()).update(s)
        lens.update(p["length_buckets"])
    drops = {k[5:]: v for k, v in sorted(tot.items()) if k.startswith("drop_")}
    ident = {k[9:]: v for k, v in sorted(tot.items()) if k.startswith("identity_")}
    m = {"stream": a.stream, "docstore": os.path.join(ROOT, "work", "docs", a.stream),
         "sources": [p for _, _, p in SOURCES[a.stream]],
         "examples_read": tot["rows"], "drops": drops, "identity_drops_by_pattern": ident,
         "docs_written": tot["docs"], "tokens_written": tot["tokens"],
         "with_reasoning": tot["with_reasoning"], "with_system": tot["with_system"],
         "per_file": {k: dict(v) for k, v in per_file.items()},
         "length_buckets": dict(lens), "max_doc_tokens": MAX_TOKENS,
         "rendered_examples_first_300_chars": _spread(parts[0]["examples"], 3),
         "identity_drop_examples": [e for p in parts for e in p["identity_drop_examples"]][:5]}
    path = os.path.join(ROOT, "manifests", "build", f"{a.stream}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(m, open(path + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    print(json.dumps(m, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "manifest"])
    ap.add_argument("--stream", required=True, choices=sorted(SOURCES))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="read only the first ROWS rows of every file (tests)")
    ap.add_argument("--out", help="document store directory (default work/docs/<stream>)")
    ap.add_argument("--stats-dir", help="where the part statistics go (default work/chat_en)")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else manifest(a)
