"""Classify dataset system prompts and scrub model-identity cards.

A clean prompt names no known model or lab and is kept verbatim. An identity
card is a short prompt whose only use of those names is to introduce the
model; the names are removed and the task, if any, stays. A disputed prompt
mentions a name inside a quotation, a long instruction, or a phrase the
scrubber cannot clear. It is kept verbatim until a person edits it.

Phrases such as "developed by" do not by themselves mark a prompt. The name
that follows has to be on the list.
"""
import json
import re

# Longer aliases come first so "smollm3" is not recorded as "smollm".
_MODELS = (
    ("smollm", r"smollm\s*\d+(?:\.\d+)?"),
    ("smollm", r"smollm"),
    ("qwen", r"qwen\s*\d+(?:\.\d+)?"),
    ("qwen", r"qwen"),
    ("llama", r"llama\s*-?\s*\d+(?:\.\d+)?"),
    ("llama", r"llama"),
    ("chatgpt", r"chatgpt"),
    ("gpt", r"gpt\s*-?\s*\d+(?:\.\d+)?"),
    ("gpt", r"gpt"),
    ("claude", r"claude(?:\s*-?\s*\d+(?:\.\d+)?)?"),
    ("deepseek", r"deepseek"),
    ("mistral", r"mistral"),
    ("olmo", r"olmo"),
    ("gemini", r"gemini"),
    ("gemma", r"gemma"),
)
_ORGS = (
    ("huggingface", r"hugging\s*face"),
    ("alibaba", r"alibaba"),
    ("openai", r"openai"),
    ("anthropic", r"anthropic"),
    ("allenai", r"allen\s*ai"),
    ("allenai", r"allen\s+institute"),
    ("allenai", r"ai2"),
    ("deepmind", r"deepmind"),
)
# "Meta" and "Google" alone occur in ordinary instructions. Only the
# attribution forms count.
_ATTRIBUTION = (
    ("meta", r"(?:by|from)\s+meta\b|meta\s+ai"),
    ("google", r"(?:by|from)\s+google\b|google\s+deepmind"),
)

_NAME_RES = tuple(
    (label, re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE))
    for label, pattern in (_MODELS + _ORGS + _ATTRIBUTION)
)
_MODEL_ALT = "|".join(pattern for _label, pattern in _MODELS)
_ORG_ALT = "|".join(pattern for _label, pattern in _ORGS)
_MODEL_RE = re.compile(rf"\b(?:{_MODEL_ALT})\b", re.IGNORECASE)
_ORG_RE = re.compile(rf"\b(?:{_ORG_ALT})\b", re.IGNORECASE)

_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'|«[^»]*»")
_BY_ORG = re.compile(
    rf"(?:,\s*|\s+)(?:trained|developed|created|built|made|produced)\s+by\s+"
    rf"(?:the\s+)?(?:{_ORG_ALT}|meta(?:\s+ai)?|google(?:\s+deepmind)?)\b",
    re.IGNORECASE,
)
_NAMED_MODEL = re.compile(
    rf"\b(?:named|called)\s+(?:the\s+)?(?:{_MODEL_ALT})\b",
    re.IGNORECASE,
)
_MODEL_IS = re.compile(
    rf"\b(?:you are|you're|i am|i'm)\s+(?:the\s+)?(?:ai\s+)?(?:model|assistant)\s+"
    rf"(?:{_MODEL_ALT})\b",
    re.IGNORECASE,
)
_ASK_MODEL = re.compile(
    r"[^.!?\n]*\bif the user asks what model you are\b[^.!?\n]*[.!?]?",
    re.IGNORECASE,
)
_RU_COMPANY = re.compile(
    rf"(?:компания|компанией)\s+(?:{_ORG_ALT})\b[^.!?\n]*",
    re.IGNORECASE,
)
_RU_NAMED = re.compile(
    rf"\bпод названием\s+(?:{_MODEL_ALT})\b",
    re.IGNORECASE,
)
_RU_ASK_MODEL = re.compile(
    r"[^.!?\n]*если пользователь спросит, какая модель вы[^.!?\n]*[.!?]?",
    re.IGNORECASE,
)
_RU_BOILER = re.compile(
    r"^(?:вы являетесь(?:\s+ai-)?\s*моделью(?:\s+ии)?|вы\s*[—–-]\s*(?:ai-)?модель)(?:\s+ии)?\.?$",
    re.IGNORECASE,
)
_SPACE = re.compile(r"[ \t]{2,}")
_EMPTY_PARENS = re.compile(r"\(\s*\)")
_DANGLING = re.compile(r"\s+([,.;:)])")
_LEAD_PUNCT = re.compile(r"^[,;:\s]+")
_BOILER = re.compile(
    r"^(?:you are(?: the)?(?: an?)?(?: ai)?(?: model| assistant)?|ты)\.?$",
    re.IGNORECASE,
)
_FALLBACK_EN = "You are a helpful AI assistant."
_FALLBACK_RU = "Ты — полезный ассистент."
# A short prompt can still be an identity card. A longer one that merely
# mentions a name is a task, and a person decides what to do with it.
_IDENTITY_MAX_CHARS = 500


def _spans(pattern, text):
    return [(match.start(), match.end()) for match in pattern.finditer(text or "")]


def _quoted_spans(text):
    return _spans(_QUOTED, text)


def _inside(span, ranges):
    start, end = span
    return any(start >= left and end <= right for left, right in ranges)


def _hits(text):
    quotes = _quoted_spans(text)
    hits = []
    for label, pattern in _NAME_RES:
        for match in pattern.finditer(text or ""):
            hits.append({
                "name": label,
                "start": match.start(),
                "end": match.end(),
                "quoted": _inside((match.start(), match.end()), quotes),
            })
    return hits


def _tidy(text):
    text = _EMPTY_PARENS.sub("", text)
    text = _DANGLING.sub(r"\1", text)
    text = _SPACE.sub(" ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [_LEAD_PUNCT.sub("", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    text = text.strip(" ,;.")
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _fallback(original):
    if re.search(r"[А-Яа-яЁё]", original or ""):
        return _FALLBACK_RU
    return _FALLBACK_EN


def _drop_identity_sentences(text):
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    kept = []
    for piece in pieces:
        if not piece.strip():
            continue
        has_name = _MODEL_RE.search(piece) or _ORG_RE.search(piece)
        asks = _ASK_MODEL.search(piece)
        bio = re.search(r"нейросетев", piece, re.IGNORECASE)
        if asks or (has_name and bio):
            continue
        if has_name and _RU_COMPANY.search(piece):
            continue
        kept.append(piece.strip())
    return " ".join(kept)


def scrub_identity(text):
    """Remove model and lab names from a short self-introduction.

    The result can still contain a name. Callers treat that as unresolved
    and do not apply the rewrite.
    """
    original = text or ""
    scrubbed = _BY_ORG.sub("", original)
    scrubbed = _NAMED_MODEL.sub("", scrubbed)
    scrubbed = _RU_NAMED.sub("", scrubbed)
    scrubbed = _MODEL_IS.sub("You are a helpful AI assistant", scrubbed)
    scrubbed = _ASK_MODEL.sub("", scrubbed)
    scrubbed = _RU_ASK_MODEL.sub("", scrubbed)
    scrubbed = _RU_COMPANY.sub("", scrubbed)
    scrubbed = _drop_identity_sentences(scrubbed)
    scrubbed = _tidy(scrubbed)
    if not scrubbed or _BOILER.match(scrubbed) or _RU_BOILER.match(scrubbed):
        return _fallback(original)
    return scrubbed


def classify_system(text):
    """Return bucket, matched names, and the text to train on.

    bucket is clean, identity, or disputed. The training text differs from
    the original only for an identity card.
    """
    original = text or ""
    hits = _hits(original)
    names = []
    for hit in hits:
        if hit["name"] not in names:
            names.append(hit["name"])
    if not hits:
        return {"bucket": "clean", "names": [], "rewritten": original, "reason": ""}
    if all(hit["quoted"] for hit in hits):
        return {"bucket": "disputed", "names": names, "rewritten": original, "reason": "quoted"}
    if len(original) > _IDENTITY_MAX_CHARS:
        return {"bucket": "disputed", "names": names, "rewritten": original, "reason": "long"}
    rewritten = scrub_identity(original)
    if rewritten != original and not _hits(rewritten):
        return {"bucket": "identity", "names": names, "rewritten": rewritten, "reason": "scrubbed"}
    return {"bucket": "disputed", "names": names, "rewritten": original, "reason": "unresolved"}


def load_rewrite_table(path, system_turns=False):
    """Exact original system text to the text training should use.

    Without system_turns, no path returns None, and the caller then folds a
    task system into the user turn. With system_turns, no path returns an
    empty table instead. An empty table keeps system turns and changes
    nothing, which is how v5 builds without a table.

    data_prep/README.md describes the table and how
    collect_system_prompts.py writes one.
    """
    if not path:
        return {} if system_turns else None
    with open(path, "r", encoding="utf-8") as handle:
        table = json.load(handle)
    if not isinstance(table, dict):
        raise ValueError(f"rewrite table must be an object: {path}")
    return {str(key): "" if value is None else str(value) for key, value in table.items()}


def system_texts_from_row(row):
    """System texts on one raw row.

    SmolTalk stores persona text in chat_template_kwargs.custom_instructions
    when the message list itself has no system turn. Tool schemas are not
    system prompts and are not returned.
    """
    if not isinstance(row, dict):
        return []
    messages = row.get("messages")
    if messages is None:
        messages = row.get("conversation")
    texts = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = (message.get("role") or message.get("from") or "").strip().lower()
            if role != "system":
                continue
            content = message.get("content")
            if content is None:
                content = message.get("value")
            content = "" if content is None else str(content).strip()
            if content:
                texts.append(content)
    if texts:
        return texts
    kwargs = row.get("chat_template_kwargs") or {}
    if isinstance(kwargs, dict):
        extra = kwargs.get("custom_instructions")
        extra = "" if extra is None else str(extra).strip()
        if extra:
            return [extra]
    return []


def row_has_tools(row):
    kwargs = row.get("chat_template_kwargs") if isinstance(row, dict) else None
    if not isinstance(kwargs, dict):
        return False
    for key in ("python_tools", "xml_tools"):
        value = kwargs.get(key)
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
    return False
