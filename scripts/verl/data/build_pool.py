"""Builds the prompt pool for OrbitLM on-policy distillation (scripts/verl/data/README.md has the result).

The output is about 32,000 unique single-turn user prompts without responses. The student samples an
answer to each prompt and the teacher scores it, so only the prompt text, a mode and a few labels are
stored. The script only reads the source datasets and writes nothing outside --out.

Steps, in the order the script runs them:

1. Eval index. Every eval set we report on is read from the parquet/jsonl snapshots of an offline
   Hugging Face hub cache (--hf-hub): MATH-500 and the full MATH test, GSM8K test, IFEval, MGSM-ru,
   HumanEval, MBPP test, MMLU-redux, MMLU test, MMLU-Pro test, AIME 2024/2025, OASST2 validation,
   CRUXEval, HellaSwag and WinoGrande validation, plus chat_probe_trivial.jsonl. For each item the index stores its 13-word
   spans, its alphanumeric-only form for exact matches, and for math items a signature of its LaTeX
   formulas and of its numbers. The signatures catch Russian translations of English eval problems,
   which share no words with the original.
   An IFEval 13-word span that occurs in three or more IFEval prompts is a constraint sentence
   ("first repeat the request word for word without change ..."). Such spans are kept out of the match
   index, so that IF training prompts which only reuse a constraint phrasing are not counted as
   contaminated; a copy of an IFEval prompt still matches through its task text. After the sources are
   loaded, an IFEval span that 20 or more IF training prompts contain is treated as template text too,
   because the same constraint has many phrasings in IFEval itself. stats.json reports
   how many prompts would have been dropped by those template spans alone. All other eval sets use
   every span.
2. RL-pool index. When --rl-math is given, its problems are indexed the same way, so that math
   prompts can be kept disjoint from them. Pass the math set that the RL gold rows are drawn from.
3. Source loading, in parallel over files. Each loader keeps the first user turn of dialogs that have
   exactly one user turn and no non-empty system message. The one exception is smol_magpie_ultra,
   where every dialog has three user turns by construction; its first turn is a standalone generated
   query, so that first turn is used. Source-specific filters run here (Big-Math solve rate,
   solutions inside the Russian math problems, LMSYS moderation and redaction, pictures and
   interactive problems in code sets, the IF target of Nemotron-IF). Duplicates inside a source are
   removed by the normalised-text hash (lowercase, whitespace collapsed).
   Contest statements (TACO, APPS, CodeContests and the same kind of rows in AceReason) get a short
   English instruction to solve the problem in Python 3, plus the starter code when the row has one. Dedup and
   decontamination use the statement without that instruction.
4. Quota filling. The plan below gives a quota per domain, language and source. Each source's
   candidates are shuffled with a fixed seed and taken in that order. A candidate is dropped by the
   first check it fails: too short, chat special tokens in the text, wrong or unsupported language
   (for English chat, IF and refusal prompts also a Latin-script prompt that is not English), identity
   question, a model name, PII or profanity (chat sources), duplicate of an already accepted
   prompt, eval overlap, RL-pool overlap (math), more than 1,024 tokens with the Qwen3 tokenizer, or near
   duplicate of an accepted prompt (80% of its 13-word spans already seen). If a source runs out, the
   other sources of the same domain and language fill the gap. A cell whose sources cannot fill it
   stays short, and stats.json reports the shortfall.
5. Modes. Within each domain and language, the accepted prompts are split exactly half think and half
   no_think in a seeded random order.
6. Output: prompts.jsonl in a seeded random order, so any prefix keeps roughly the full mix, and
   stats.json with counts and drop reasons.

Run with the verl env's python, CPU only. It reads only the sources and writes only --out:
    $PY scripts/verl/data/build_pool.py --out <dir> --raw-hf <dir> --hf-cache <dir> --hf-hub <dir> \
        --tokenizer <dir> [--ru-math <jsonl>] [--ru-sft <dir>] [--rl-math <jsonl>] [--trivial <jsonl>] \
        [--workers 64]
--raw-hf is the folder that data_prep/long_context/download_hf.py fills, $ORBIT_RAW/hf, after both of
its plans have run. --hf-cache is a Hugging Face home whose datasets/ folder holds the `datasets` caches
of the code and LMSYS sources. ``load_pools`` lists the sub-paths read below each of them. With 64
workers the build takes about five minutes.

The Russian math and code cells, and a part of the Russian chat cell, come from two inputs that the
user supplies, and both are optional:

- ``--ru-math``: a JSONL file of Russian math problems with reference answers, one gold math row per
  line (``instruction`` is the problem, ``response`` the answer). Its problems fill the ru_math pool.
  The same file can feed ``passrate.sh --kind math_ru``, which makes the Russian keep of the RL recipe.
- ``--ru-sft``: a folder of Russian SFT chats, one ``<role>.jsonl`` file per role in messages format
  (``{"messages": [{"role": "user", "content": ...}, ...]}``). ``math.jsonl`` fills ru_sft_math,
  ``code.jsonl`` fills ru_sft_code, and the chat-like roles (``CHAT_ROLES``) fill ru_sft_chat. A
  missing role file is skipped.

Without them, the Russian math and code cells stay empty, the public Russian chat sources fill the
rest of the Russian chat cell, and the Russian share of the pool drops.
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import statistics
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import pyarrow as pa
import pyarrow.parquet as pq

# The trivial prompts of the SFT evaluation's chat probe, which the pool must not contain either.
# scripts/verl/data -> the repo root is three levels up.
TRIVIAL_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                               "eval", "in_training", "tasks", "chat_probe_trivial.jsonl")

# The seed of every shuffle and draw. The pool of the release recipes was built with it.
SEED = 20261023
NGRAM = 13
TEMPLATE_GRAM_MIN_ITEMS = 3
# Only IFEval has template spans that training data shares by design: its constraint sentences.
TEMPLATE_SETS = {"ifeval"}
IF_POOLS = ("dolci_precise_if", "dolci_precise_if_ru", "nemotron_if")
IF_TEMPLATE_MIN_PROMPTS = 20
RL_POOL_OVERLAP = 0.5
NEAR_DUP_OVERLAP = 0.8
MAX_TOKENS = 1024
MIN_CHARS = 8

# Quotas: (domain, lang, cell total, [(pool, quota), ...]). The totals give math 35%, code 15%,
# instruction following 20%, chat 25% and refusals 5% of 32,000, with Russian at 10,980 (34.3%).
# The ru_math and ru_sft_* pools come from the optional inputs --ru-math and --ru-sft.
PLAN = [
    ("math", "en", 6720, [("omr", 3700), ("bigmath", 3020)]),
    ("math", "ru", 4480, [("ru_math", 3000), ("ru_sft_math", 1480)]),
    ("code", "en", 3200, [("taco", 700), ("apps", 300), ("code_contests", 600), ("acereason_code", 1600)]),
    ("code", "ru", 1600, [("ru_sft_code", 1600)]),
    ("if", "en", 5900, [("dolci_precise_if", 3400), ("nemotron_if", 2500)]),
    ("if", "ru", 500, [("dolci_precise_if_ru", 500)]),
    ("chat", "en", 3600, [("wildchat", 2000), ("magpie_ultra", 1600)]),
    ("chat", "ru", 4400, [("wildchat_ru", 600), ("twix", 1600), ("grandmaster_ru", 900),
                          ("ru_sft_chat", 700), ("lmsys_ru", 600)]),
    ("refusal", "en", 1600, [("wildjailbreak", 650), ("wildguardmix", 600), ("coconot", 350)]),
]
# The role files of --ru-sft that fill the Russian chat pool.
CHAT_ROLES = ("creative", "dialog", "science", "summarization")

# ----------------------------------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------------------------------
WS = re.compile(r"\s+")
WORD = re.compile(r"[^\W_]+")


def norm_text(s):
    """The dedup key: lowercase with whitespace collapsed."""
    return WS.sub(" ", s.lower()).strip()


def norm_alnum(s):
    """The exact-match key of the RL math decontamination, extended to Cyrillic: letters and digits only."""
    return "".join(WORD.findall(unicodedata.normalize("NFKC", s).lower()))


def grams(s, n=NGRAM):
    w = WORD.findall(unicodedata.normalize("NFKC", s).lower())
    return {" ".join(w[k:k + n]) for k in range(len(w) - n + 1)}


def gram_hashes(s):
    return {hash(g) for g in grams(s)}


def blake(s, size=8):
    return hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=size).hexdigest()


CODE_FENCE = re.compile(r"```.*?```", re.S)
LATEX_CMD = re.compile(r"\\[A-Za-z]+")
# Letters that Russian does not use but Ukrainian, Belarusian, Kazakh, Kyrgyz and similar languages do.
NON_RU_CYR = set("іїєґўәғқңөұүһӣӯҳҷ")


def lang_of(s):
    """Script-based tag: ru if at least 30% of the letters are Cyrillic, en if at least 70% are ASCII Latin.

    Code blocks and LaTeX commands are removed first, because their Latin letters say nothing about the
    language of the request. Cyrillic text with letters that Russian lacks (Ukrainian, Kazakh, Kyrgyz)
    is tagged other.
    """
    t = LATEX_CMD.sub(" ", CODE_FENCE.sub(" ", s))
    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return "none"
    cyr = [ch for ch in letters if "Ѐ" <= ch <= "ӿ"]
    if len(cyr) >= 0.3 * len(letters):
        lower = [ch.lower() for ch in cyr]
        if sum(ch in NON_RU_CYR for ch in lower) >= 2:
            return "other"
        return "ru"
    latin = sum(("a" <= ch.lower() <= "z") for ch in letters)
    return "en" if latin >= 0.7 * len(letters) else "other"


LATEX_SPAN = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


LATEX_NOISE = re.compile(r"\\(displaystyle|left|right|bigl|bigr|Bigl|Bigr|big|Big|text|mathrm|operatorname|"
                         r"qquad|quad|,|;|!|:)")


def math_signatures(s):
    """Language-independent fingerprints of a math problem: its formulas and the sequence of its numbers.

    A translation keeps both, so they link a Russian problem to the English original. The formula
    signature is the LaTeX content reduced to letters, digits and operators (\\dfrac and \\frac are
    the same, spacing and sizing commands are dropped), because a translator often re-spaces or
    re-delimits formulas. It counts only when at least 15 characters remain. The number signature is
    the ordered list of numbers and counts only with at least five numbers and eight digits, so that
    short or generic problems do not collide by chance.
    """
    spans = ["".join(m.groups(default="")) for m in LATEX_SPAN.finditer(s)]
    latex = re.sub(r"\\[dt]frac", r"\\frac", "".join(spans))
    latex = re.sub(r"[^A-Za-z0-9+\-*/=^<>]", "", LATEX_NOISE.sub("", latex))
    nums = [x.replace(",", ".") for x in NUMBER.findall(s)]
    digits = sum(ch.isdigit() for x in nums for ch in x)
    return (latex if len(latex) >= 15 else None), (tuple(nums) if len(nums) >= 5 and digits >= 8 else None)


SPECIAL = re.compile(r"<\|(im_start|im_end|endoftext)\|>|</?think>")

# Function words for telling English from other languages in Latin script. The script tag alone calls
# a prompt English when most of its letters are ASCII Latin, which also holds for Spanish or Malagasy,
# and for an Arabic question followed by a long English IF constraint.
EN_FUNCTION_WORDS = set("""the and of to is are was were for with that this what how why which who be can from an or
not have has does will would should could please your you it its they their there these those about into if than then
when where my we our write give make any all some each more most only also just like been being did at by on as his
her him them""".split())
FOREIGN_FUNCTION_WORDS = set("""de la en un el los las del que por para una cómo qué está es pero muy también puedes sobre entre
como le les des du est et une pour dans qui avec sur pas vous sont cette ce au aux der das und ist nicht mit ein eine
für auf sich auch wie oder wird sind zu dem os não uma com você isso são dos das mais ao il gli che della sono questo
anche essere più nel alla het een van niet zijn ook wat voor maar deze yang dan dari ini itu untuk dengan adalah tidak
akan saya bisa ny ary amin ireto izy ve raha ho tsy dia izany waa iyo oo waxaa ee kwa katika hii ang ng mga ay""".split())


def latin_prompt_not_english(s):
    """True for a prompt in Latin script that is not English, or that mixes in another script.

    It fires when at least 5% of the letters (and at least four) are non-ASCII and not Greek
    (diacritics, Arabic, CJK, Cyrillic and so on; Greek is left out because English math uses it), so
    that an English prompt with "café" or "naïve" stays English. It also fires when the first 80 words
    hold at least three common function words of another language and 1.5 times more of them than of
    English function words.
    """
    t = LATEX_CMD.sub(" ", CODE_FENCE.sub(" ", s))
    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return False
    foreign_letters = sum(1 for ch in letters if ord(ch) > 127 and not ("\u0370" <= ch <= "\u03ff"))
    if foreign_letters >= max(4, 0.05 * len(letters)):
        return True
    words = re.findall(r"[^\W\d_]+", t.lower())[:80]
    if len(words) < 6:
        return False
    foreign = sum(w in FOREIGN_FUNCTION_WORDS for w in words)
    english = sum(w in EN_FUNCTION_WORDS for w in words)
    return foreign >= 3 and foreign > 1.5 * english

IDENTITY = re.compile("|".join([
    r"\bwho\s+(are|r)\s+(you|u)\b", r"\bwhat'?s\s+your\s+name\b", r"\bwhat\s+is\s+your\s+name\b",
    r"\b(what|which)\s+(model|llm|ai|language\s+model|version|company|organization)\s+(are|r)\s+(you|u)\b",
    r"\b(who|which\s+company|what\s+company|which\s+organi[sz]ation)\s+(made|created|built|trained|developed|designed|owns)\s+you\b",
    r"\bare\s+you\s+(a\s+|an\s+)?(chatgpt|gpt|human|bot|robot|ai|real\s+person|sentient|conscious|language\s+model)\b",
    r"\b(tell\s+me\s+about|introduce)\s+yourself\b",
    r"\byour\s+(creator|developer|maker|training\s+data|knowledge\s+cut-?off|underlying\s+model|architecture)\b",
    r"\bwhat\s+can\s+you\s+do\b",
    r"\bкто\s+(ты|вы)\b", r"\bкак\s+(тебя|вас)\s+зовут\b", r"\b(тво[её]|ваше)\s+имя\b",
    r"\b(какая|что\s+за|какой)\s+(ты|вы)\s+(модель|нейросеть|программа|ии|бот)\b",
    r"\b(кто|какая\s+компания)\s+(тебя|вас)\s+(создал|сделал|разработал|обучил|написал)",
    r"\b(ты|вы)\s+(человек|робот|бот|нейросеть|chatgpt)\s*\?", r"\bрасскажи\s+о\s+себе\b", r"\bпредставься\b",
    r"\bна\s+какой\s+модели\b", r"\bкакая\s+у\s+(тебя|вас)\s+версия\b", r"\bчто\s+(ты|вы)\s+умеете?\b",
    r"\bчто\s+ты\s+умеешь\b",
]), re.I)
# Model and developer names. Answers to such prompts would carry the teacher's own identity into the student.
MODEL_NAMES = re.compile(r"qwen|通义|tongyi|nemotron|\bolmo\b|llama", re.I)
# Other assistant and developer names. A prompt that addresses another assistant invites the teacher
# to say who it really is.
MODEL_NAMES_EXTRA = re.compile(r"chat\s?gpt|openai|\bgpt-?[345]|claude|anthropic|deepseek|gemini|mistral|"
                               r"alibaba|nvidia|allen\s?ai|\bai2\b|meta\s+ai",
                               re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)|"
                   r"(?<!\d)\+\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3}[\s-]?\d{2,4}(?!\d)")
PROFANITY = re.compile(r"\b(ху[йеёия]\w*|пизд\w*|[её]б[аaл]\w*|бля[дт]\w*|сук[аи]\b|мудак\w*|пид[оа]р\w*|"
                       r"залуп\w*|шлюх\w*|fuck\w*|cunt\w*|nigg\w*)", re.I)

# ----------------------------------------------------------------------------------------------
# Programming statements: a short instruction for statements that do not say what to produce
# ----------------------------------------------------------------------------------------------
CODE_WRAP = [
    "Solve the following programming problem in Python 3. Put the complete solution in one ```python code block.\n\n{s}",
    "Write a Python 3 program that solves the problem below. Give the full code in a single ```python block.\n\n{s}",
    "{s}\n\nSolve this problem in Python 3 and put the complete solution in one ```python code block.",
    "{s}\n\nWrite a Python 3 solution for this problem. Present the whole program in a single ```python code block.",
]
def wrap_statement(statement, starter=""):
    """Returns the prompt for a bare contest statement: one of four instruction phrasings plus the statement.

    Every statement from these sources gets the instruction, because many of them say "write a function"
    without naming a language, and the evals we care about (HumanEval, MBPP) are Python. The phrasing is
    chosen by a hash of the statement, so a rebuild gives the same prompt.
    """
    s = statement.strip()
    if starter and starter.strip():
        s += "\n\nUse this starter code:\n```python\n" + starter.strip("\n") + "\n```"
    variant = int(blake(statement, 2), 16) % len(CODE_WRAP)
    return CODE_WRAP[variant].format(s=s)


# ----------------------------------------------------------------------------------------------
# Loaders. Each returns (records, counter) for one file. A record is a dict with the final prompt
# "text", the "core" text used for dedup and decontamination, the output "source" label, a
# "ref" back to the source row and the script "lang" tag.
# ----------------------------------------------------------------------------------------------
def first_user_turn(messages, allow_multiturn=False, role_key="role", content_key="content"):
    """Returns (prompt, None) for an allowed dialog or (None, reason) for a dropped one."""
    if not messages:
        return None, "empty_dialog"
    roles = [m.get(role_key) for m in messages]
    for m in messages:
        if m.get(role_key) == "system" and (m.get(content_key) or "").strip():
            return None, "system_prompt"
    rest = [m for m in messages if m.get(role_key) != "system"]
    if not rest or rest[0].get(role_key) not in ("user", "human"):
        return None, "not_user_first"
    n_user = sum(r in ("user", "human") for r in roles)
    if n_user > 1 and not allow_multiturn:
        return None, "multi_turn"
    text = rest[0].get(content_key)
    if not isinstance(text, str):
        return None, "non_text_content"
    return text, None


def rec(text, source, ref, core=None):
    core = text if core is None else core
    return {"text": text, "core": core, "source": source, "ref": str(ref), "lang": lang_of(core if core else text)}


def dedup_local(records, counter):
    seen, out = set(), []
    for r in records:
        k = blake(norm_text(r["core"]))
        if k in seen:
            counter["dup_in_source"] += 1
            continue
        seen.add(k)
        out.append(r)
    return out


def ld_omr(f):
    c = collections.Counter()
    t = pq.read_table(f, columns=["problem", "problem_source", "problem_type"])
    out = []
    for p, s, ty in zip(t["problem"].to_pylist(), t["problem_source"].to_pylist(), t["problem_type"].to_pylist()):
        c["rows"] += 1
        out.append(rec(p or "", "OpenMathReasoning/cot", f"{s}|{ty}"))
    return dedup_local(out, c), c


def ld_bigmath(f):
    c = collections.Counter()
    t = pq.read_table(f, columns=["problem", "source", "llama8b_solve_rate"])
    out = []
    for i, (p, s, r) in enumerate(zip(t["problem"].to_pylist(), t["source"].to_pylist(),
                                      t["llama8b_solve_rate"].to_pylist())):
        c["rows"] += 1
        if r is None:
            c["solve_rate_missing"] += 1
            continue
        if not (0.1 <= r <= 0.9):
            c["solve_rate_outside_0.1_0.9"] += 1
            continue
        out.append(rec(p or "", "Big-Math-RL-Verified", f"{s}|row{i}"))
    return dedup_local(out, c), c


# A problem text that already holds an answer or a worked solution teaches nothing, because the
# student only has to copy it. Such problems are dropped.
RU_SOLUTION_IN_PROBLEM = re.compile(r"\*\*Ответ|Ответ\s*:|##\s*Answer|Answer\s*:|\\boxed|Шаг\s*1|###|\n---\n|"
                                    r"Решение\s*:|Таким образом|Итак,", re.I)


def ld_ru_math(f):
    """The problems of --ru-math, a gold math jsonl: ``instruction`` is the problem."""
    c = collections.Counter()
    out = []
    name = os.path.basename(f)
    with open(f, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            c["rows"] += 1
            p = json.loads(line).get("instruction") or ""
            if RU_SOLUTION_IN_PROBLEM.search(p):
                c["solution_inside_problem"] += 1
                continue
            out.append(rec(p, "ru_math", f"{name}:{i}"))
    return dedup_local(out, c), c


def ld_ru_sft(f):
    """The first user turns of one role file of --ru-sft, a messages-format jsonl."""
    c = collections.Counter()
    out = []
    role = os.path.basename(f)[:-len(".jsonl")]
    with open(f, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            c["rows"] += 1
            d = json.loads(line)
            text, why = first_user_turn(d.get("messages") or [])
            if why:
                c[why] += 1
                continue
            out.append(rec(text, f"russian_sft/{role}", f"{role}.jsonl:{i}"))
    return dedup_local(out, c), c


def open_arrow(f):
    try:
        return list(pa.ipc.open_stream(pa.memory_map(f)))
    except Exception:
        r = pa.ipc.open_file(pa.memory_map(f))
        return [r.get_batch(i) for i in range(r.num_record_batches)]


ACE_BARE = {"OpenCodeReasoning", "TACO", "apps", "leetcode"}


def ld_acereason(f):
    c = collections.Counter()
    out = []
    base = os.path.basename(f)
    for b in open_arrow(f):
        cats = b.column(b.schema.get_field_index("category")).to_pylist()
        srcs = b.column(b.schema.get_field_index("source")).to_pylist()
        inps = b.column(b.schema.get_field_index("input"))
        for i, (cat, s) in enumerate(zip(cats, srcs)):
            if cat != "code":
                continue
            c["rows"] += 1
            x = inps[i].as_py() or ""
            text = wrap_statement(x) if s in ACE_BARE else x
            out.append(rec(text, f"AceReason-1.1-SFT/code/{s}", f"{base}", core=x))
    return dedup_local(out, c), c


INTERACTIVE = re.compile(r"this is an interactive problem|interactive problem", re.I)


def ld_taco(f):
    c = collections.Counter()
    out = []
    t = pa.Table.from_batches(open_arrow(f)).select(["question", "starter_code", "picture_num", "url", "source"])
    for q, st, pic, url, s in zip(*[t[k].to_pylist() for k in t.column_names]):
        c["rows"] += 1
        q = q or ""
        try:
            pic_n = int(float(pic)) if pic not in (None, "") else 0
        except ValueError:
            pic_n = 0
        if pic_n > 0 or "<image>" in q:
            c["references_picture"] += 1
            continue
        if INTERACTIVE.search(q):
            c["interactive"] += 1
            continue
        out.append(rec(wrap_statement(q, st or ""), "TACO/train", url or s, core=q))
    return dedup_local(out, c), c


def ld_apps(f):
    c = collections.Counter()
    out = []
    t = pa.Table.from_batches(open_arrow(f)).select(["problem_id", "question", "starter_code"])
    for pid, q, st in zip(*[t[k].to_pylist() for k in t.column_names]):
        c["rows"] += 1
        q = q or ""
        if "<image>" in q:
            c["references_picture"] += 1
            continue
        if INTERACTIVE.search(q):
            c["interactive"] += 1
            continue
        out.append(rec(wrap_statement(q, st or ""), "APPS/train", f"problem_id={pid}", core=q))
    return dedup_local(out, c), c


def ld_code_contests(f):
    c = collections.Counter()
    out = []
    t = pa.Table.from_batches(open_arrow(f)).select(["name", "description"])
    for n, q in zip(t["name"].to_pylist(), t["description"].to_pylist()):
        c["rows"] += 1
        q = q or ""
        if "<image>" in q:
            c["references_picture"] += 1
            continue
        if INTERACTIVE.search(q):
            c["interactive"] += 1
            continue
        out.append(rec(wrap_statement(q), "CodeContests/train", n, core=q))
    return dedup_local(out, c), c


def ld_jsonl_range(args):
    """Reads Nemotron-IF chat_if.jsonl between two byte offsets, so that the 7 GB file parses in parallel."""
    f, start, end = args
    c = collections.Counter()
    out = []
    with open(f, "rb") as fh:
        if start:
            fh.seek(start - 1)
            if fh.read(1) != b"\n":
                fh.readline()
        while fh.tell() < end:
            line = fh.readline()
            if not line:
                break
            c["rows"] += 1
            d = json.loads(line)
            if d.get("capability_target") != "instruction_following":
                c["capability_target_not_if"] += 1
                continue
            text, why = first_user_turn(d.get("messages") or [])
            if why:
                c[why] += 1
                continue
            out.append(rec(text, "Nemotron-Instruction-Following-Chat-v1/chat_if", d.get("uuid")))
    return out, c


DOLCI_KEEP = {"Wildchat": "Dolci-Instruct/Wildchat", "Dolci Instruct Precise IF": "Dolci-Instruct/Precise IF",
              "WildJailbreak": "Dolci-Instruct/WildJailbreak", "WildGuardMix": "Dolci-Instruct/WildGuardMix",
              "CoCoNot": "Dolci-Instruct/CoCoNot"}


def ld_dolci(f):
    c = collections.Counter()
    out = []
    t = pq.read_table(f, columns=["id", "messages", "source_dataset"])
    for i, m, s in zip(t["id"].to_pylist(), t["messages"].to_pylist(), t["source_dataset"].to_pylist()):
        if s not in DOLCI_KEEP:
            continue
        c[f"rows:{s}"] += 1
        text, why = first_user_turn(m)
        if why:
            c[f"{why}:{s}"] += 1
            continue
        out.append(rec(text, DOLCI_KEEP[s], i))
    return out, c


def ld_magpie(f):
    c = collections.Counter()
    out = []
    t = pq.read_table(f, columns=["messages"])
    base = os.path.basename(f)
    for i, m in enumerate(t["messages"].to_pylist()):
        c["rows"] += 1
        text, why = first_user_turn(m, allow_multiturn=True)
        if why:
            c[why] += 1
            continue
        if sum(x.get("role") == "user" for x in m) > 1:
            c["first_turn_of_multi_turn_used"] += 1
        out.append(rec(text, "smoltalk2/smol_magpie_ultra", f"{base}:{i}"))
    return dedup_local(out, c), c


def ld_twix(f):
    c = collections.Counter()
    out = []
    t = pq.read_table(f, columns=["id", "messages", "subset"])
    for i, m, s in zip(t["id"].to_pylist(), t["messages"].to_pylist(), t["subset"].to_pylist()):
        c["rows"] += 1
        if s != "general":
            c["subset_not_general"] += 1
            continue
        text, why = first_user_turn(m)
        if why:
            c[why] += 1
            continue
        out.append(rec(text, "T-Wix/general", i))
    return dedup_local(out, c), c


def ld_grandmaster(f):
    c = collections.Counter()
    out = []
    t = pq.read_table(f, columns=["source", "conversation", "prompt_lang"])
    base = os.path.basename(f)
    for i, (s, m, pl) in enumerate(zip(t["source"].to_pylist(), t["conversation"].to_pylist(),
                                       t["prompt_lang"].to_pylist())):
        c["rows"] += 1
        if pl != "ru":
            c["prompt_lang_not_ru"] += 1
            continue
        text, why = first_user_turn(m)
        if why:
            c[why] += 1
            continue
        out.append(rec(text, "GrandMaster-PRO-MAX/ru", f"{base}:{i}|{s}"))
    return dedup_local(out, c), c


def ld_lmsys(f):
    c = collections.Counter()
    out = []
    for b in open_arrow(f):
        langs = b.column(b.schema.get_field_index("language")).to_pylist()
        idx = [i for i, l in enumerate(langs) if l == "Russian"]
        if not idx:
            continue
        sub = b.take(pa.array(idx))
        for cid, conv, turn, mod, red in zip(*[sub.column(sub.schema.get_field_index(k)).to_pylist() for k in
                                               ("conversation_id", "conversation", "turn", "openai_moderation",
                                                "redacted")]):
            c["rows_russian"] += 1
            if red:
                c["pii_redacted_by_lmsys"] += 1
                continue
            if mod and any(x.get("flagged") or any((x.get("categories") or {}).values()) for x in mod):
                c["openai_moderation_flagged"] += 1
                continue
            text, why = first_user_turn(conv)
            if why:
                c[why] += 1
                continue
            out.append(rec(text, "LMSYS-Chat-1M/ru", cid))
    return out, c


# ----------------------------------------------------------------------------------------------
# Eval and RL-pool indexes
# ----------------------------------------------------------------------------------------------
def snap(hub, repo):
    s = sorted(glob.glob(f"{hub}/datasets--{repo.replace('/', '--')}/snapshots/*"))
    assert s, repo
    return s[0]


def read_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


# Multiple-choice questions are indexed with their options; the question alone is also an exact key,
# because a training prompt may quote the question without the options.
QUESTION_ONLY = []


def eval_sets(hub, trivial):
    """Returns {eval name: [(text, is_math)]} read from the offline snapshots of ``hub``."""
    E = {}
    E["math500"] = [(r["problem"], True) for r in read_jsonl(snap(hub, "HuggingFaceH4/MATH-500") + "/test.jsonl")]
    E["math_test_full"] = [(p, True) for f in glob.glob(snap(hub, "EleutherAI/hendrycks_math") + "/*/test-*.parquet")
                           for p in pq.read_table(f, columns=["problem"])["problem"].to_pylist()]
    E["gsm8k_test"] = [(q, True) for q in pq.read_table(glob.glob(snap(hub, "openai/gsm8k") + "/main/test-*.parquet")[0],
                                                        columns=["question"])["question"].to_pylist()]
    E["ifeval"] = [(r["prompt"], False) for r in read_jsonl(snap(hub, "google/IFEval") + "/ifeval_input_data.jsonl")]
    t = pq.read_table(glob.glob(snap(hub, "juletxara/mgsm") + "/ru/test-*.parquet")[0])
    E["mgsm_ru"] = [(q, True) for q in t["question"].to_pylist()]
    t = pq.read_table(glob.glob(snap(hub, "openai/openai_humaneval") + "/openai_humaneval/test-*.parquet")[0])
    E["humaneval"] = [(p, False) for p in t["prompt"].to_pylist()]
    t = pq.read_table(glob.glob(snap(hub, "google-research-datasets/mbpp") + "/full/test-*.parquet")[0])
    E["mbpp_test"] = [(x + "\n" + "\n".join(tl), False) for x, tl in zip(t["text"].to_pylist(), t["test_list"].to_pylist())]
    items = []
    for f in glob.glob(snap(hub, "fxmarty/mmlu-redux-2.0-ok") + "/*/data-*.arrow"):
        tb = pa.Table.from_batches(open_arrow(f))
        for q, ch in zip(tb["question"].to_pylist(), tb["choices"].to_pylist()):
            items.append((q + "\n" + "\n".join(map(str, ch or [])), False))
            QUESTION_ONLY.append((q, "mmlu_redux"))
    E["mmlu_redux"] = items
    items = []
    for f in glob.glob(snap(hub, "cais/mmlu") + "/*/test-*.parquet"):
        tb = pq.read_table(f, columns=["question", "choices"])
        for q, ch in zip(tb["question"].to_pylist(), tb["choices"].to_pylist()):
            items.append((q + "\n" + "\n".join(map(str, ch or [])), False))
            QUESTION_ONLY.append((q, "mmlu_test"))
    E["mmlu_test"] = items
    tb = pq.read_table(glob.glob(snap(hub, "TIGER-Lab/MMLU-Pro") + "/data/test-*.parquet")[0], columns=["question", "options"])
    E["mmlu_pro_test"] = [(q + "\n" + "\n".join(map(str, o or [])), False)
                          for q, o in zip(tb["question"].to_pylist(), tb["options"].to_pylist())]
    QUESTION_ONLY.extend((q, "mmlu_pro_test") for q in tb["question"].to_pylist())
    tb = pq.read_table(glob.glob(snap(hub, "Maxwell-Jia/AIME_2024") + "/*.parquet")[0])
    E["aime2024"] = [(p, True) for p in tb["Problem"].to_pylist()]
    E["aime2025"] = [(r["problem"], True) for r in read_jsonl(snap(hub, "math-ai/aime25") + "/test.jsonl")]
    tb = pq.read_table(glob.glob(snap(hub, "OpenAssistant/oasst2") + "/data/validation-*.parquet")[0], columns=["text", "role"])
    E["oasst2_validation"] = [(x, False) for x, r in zip(tb["text"].to_pylist(), tb["role"].to_pylist()) if r == "prompter"]
    E["chat_probe_trivial"] = [(r["prompt"], False) for r in read_jsonl(trivial)]
    E["cruxeval"] = [(r["code"], False) for r in read_jsonl(snap(hub, "cruxeval-org/cruxeval") + "/test.jsonl")]
    tb = pq.read_table(glob.glob(snap(hub, "Rowan/hellaswag") + "/data/validation-*.parquet")[0], columns=["ctx"])
    E["hellaswag_val"] = [(x, False) for x in tb["ctx"].to_pylist()]
    tb = pq.read_table(glob.glob(snap(hub, "allenai/winogrande") + "/winogrande_xl/validation-*.parquet")[0], columns=["sentence"])
    E["winogrande_val"] = [(x, False) for x in tb["sentence"].to_pylist()]
    return E


class EvalIndex:
    """13-word spans, exact keys and math signatures of all eval sets, each mapped to its eval name."""

    def __init__(self, hub, trivial):
        sets = eval_sets(hub, trivial)
        self.sizes = {k: len(v) for k, v in sets.items()}
        self.gram = {}
        self.template_gram = {}
        self.exact = {}
        self.latex = {}
        self.nums = {}
        for name, items in sets.items():
            per_item = collections.Counter()
            for text, is_math in items:
                per_item.update(gram_hashes(text))
                k = norm_alnum(text)
                if k:
                    self.exact.setdefault(k, name)
                if is_math:
                    lx, nm = math_signatures(text)
                    if lx:
                        self.latex.setdefault(lx, name)
                    if nm:
                        self.nums.setdefault(nm, name)
            for g, n in per_item.items():
                if name in TEMPLATE_SETS and n >= TEMPLATE_GRAM_MIN_ITEMS:
                    self.template_gram.setdefault(g, name)
                else:
                    self.gram.setdefault(g, name)
        for q, name in QUESTION_ONLY:
            k = norm_alnum(q)
            if len(k) >= 20:
                self.exact.setdefault(k, name)
        self.template_counts = collections.Counter(self.template_gram.values())

    def promote_templates(self, hashes):
        """Moves spans that many training prompts share out of the match index into the template set."""
        for g in hashes:
            name = self.gram.pop(g, None)
            if name:
                self.template_gram[g] = name
        self.template_counts = collections.Counter(self.template_gram.values())

    def match(self, core, gh, is_math, xling):
        """Returns (eval name or None, how, template-only eval name or None).

        The number signature is used only for Russian candidates (xling), because English problems are
        already covered by the word spans and short number lists collide by chance.
        """
        k = norm_alnum(core)
        if k in self.exact:
            return self.exact[k], "exact", None
        for g in gh:
            if g in self.gram:
                return self.gram[g], "13gram", None
        if is_math:
            lx, nm = math_signatures(core)
            if lx and lx in self.latex:
                return self.latex[lx], "latex_signature", None
            if xling and nm and nm in self.nums:
                return self.nums[nm], "number_signature", None
        for g in gh:
            if g in self.template_gram:
                return None, None, self.template_gram[g]
        return None, None, None


class RlMathIndex:
    """The math problems of --rl-math: exact keys, 13-word spans and math signatures.

    Without a path the index is empty and matches nothing.
    """

    def __init__(self, path=None):
        self.exact, self.gram, self.latex, self.nums = set(), set(), set(), set()
        self.n = 0
        if path is None:
            return
        with open(path) as fh:
            for line in fh:
                q = json.loads(line)["instruction"]
                self.n += 1
                self.exact.add(norm_alnum(q))
                self.gram |= gram_hashes(q)
                lx, nm = math_signatures(q)
                if lx:
                    self.latex.add(lx)
                if nm:
                    self.nums.add(nm)

    def match(self, core, gh, xling):
        if norm_alnum(core) in self.exact:
            return "exact"
        if gh and sum(g in self.gram for g in gh) >= RL_POOL_OVERLAP * len(gh):
            return "13gram"
        lx, nm = math_signatures(core)
        if lx and lx in self.latex:
            return "latex_signature"
        if xling and nm and nm in self.nums:
            return "number_signature"
        return None


# ----------------------------------------------------------------------------------------------
# Loading all pools
# ----------------------------------------------------------------------------------------------
def byte_ranges(f, parts):
    size = os.path.getsize(f)
    step = size // parts + 1
    return [(f, i * step, min(size, (i + 1) * step)) for i in range(parts)]


def load_pools(workers, paths, missing):
    """Reads every source in parallel. ``paths`` holds the source dirs of the command line.

    Below --raw-hf, plain downloads of the Hugging Face repos in the <org>__<name> folders that
    download_hf.py writes. Its default plan gives nvidia__OpenMathReasoning,
    SynthLabsAI__Big-Math-RL-Verified, nvidia__AceReason-1.1-SFT, t-tech__T-Wix and
    nvidia__Nemotron-Instruction-Following-Chat-v1. data_prep/sft/download_plan.json gives
    allenai__Dolci-Instruct-SFT, HuggingFaceTB__smoltalk2 and Vikhrmodels__GrandMaster-PRO-MAX.
    Below the datasets/ folder of --hf-cache, the `datasets` caches of BAAI/taco, codeparrot/apps,
    deepmind/code_contests and lmsys/lmsys-chat-1m, as ``datasets.load_dataset`` writes them. The
    optional --ru-math file and --ru-sft folder are described in the module docstring. ``missing``
    collects the optional inputs that were not given or not found, so that stats.json can name them.
    """
    R = paths["raw_hf"]
    D = paths["hf_cache"] + "/datasets"
    jobs = []  # (pool name, function, argument)
    for f in sorted(glob.glob(R + "/nvidia__OpenMathReasoning/data/cot-*.parquet")):
        jobs.append(("omr", ld_omr, f))
    jobs.append(("bigmath", ld_bigmath, R + "/SynthLabsAI__Big-Math-RL-Verified/data/train-00000-of-00001.parquet"))
    if paths["ru_math"]:
        jobs.append(("ru_math", ld_ru_math, paths["ru_math"]))
    else:
        missing.append("--ru-math")
    if paths["ru_sft_dir"]:
        roles = [("ru_sft_math", "math"), ("ru_sft_code", "code"), *(("ru_sft_chat", role) for role in CHAT_ROLES)]
        for pool, role in roles:
            f = os.path.join(paths["ru_sft_dir"], f"{role}.jsonl")
            if os.path.isfile(f):
                jobs.append((pool, ld_ru_sft, f))
            else:
                missing.append(f)
    else:
        missing.append("--ru-sft")
    for f in sorted(glob.glob(R + "/nvidia__AceReason-1.1-SFT/sft_data.parquet/*.arrow")):
        jobs.append(("acereason_code", ld_acereason, f))
    for f in sorted(glob.glob(D + "/BAAI___taco/ALL/*/*/taco-train-*.arrow")):
        jobs.append(("taco", ld_taco, f))
    for f in sorted(glob.glob(D + "/codeparrot___apps/all/*/*/apps-train*.arrow")):
        jobs.append(("apps", ld_apps, f))
    for f in sorted(glob.glob(D + "/deepmind___code_contests/default/*/*/code_contests-train-*.arrow")):
        jobs.append(("code_contests", ld_code_contests, f))
    for a in byte_ranges(R + "/nvidia__Nemotron-Instruction-Following-Chat-v1/data/chat_if.jsonl", 48):
        jobs.append(("nemotron_if", ld_jsonl_range, a))
    for f in sorted(glob.glob(R + "/allenai__Dolci-Instruct-SFT/data/*.parquet")):
        jobs.append(("dolci", ld_dolci, f))
    for f in sorted(glob.glob(R + "/HuggingFaceTB__smoltalk2/SFT/smoltalk_smollm3_smol_magpie_ultra_no_think-*.parquet")):
        jobs.append(("magpie_ultra", ld_magpie, f))
    for f in sorted(glob.glob(R + "/t-tech__T-Wix/data/*.parquet")):
        jobs.append(("twix", ld_twix, f))
    for f in sorted(glob.glob(R + "/Vikhrmodels__GrandMaster-PRO-MAX/data/*.parquet")):
        jobs.append(("grandmaster_ru", ld_grandmaster, f))
    for f in sorted(glob.glob(D + "/lmsys___lmsys-chat-1m/default/*/*/*.arrow")):
        jobs.append(("lmsys_ru", ld_lmsys, f))

    records = collections.defaultdict(list)
    struct = collections.defaultdict(collections.Counter)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        futs = [(name, ex.submit(fn, arg)) for name, fn, arg in jobs]
        for name, fu in futs:
            recs, c = fu.result()
            records[name].extend(recs)
            struct[name].update(c)

    # Dolci holds five pools; WildChat and Precise IF are split by script into English and Russian.
    dolci = records.pop("dolci")
    dc = struct.pop("dolci")
    split = {"Dolci-Instruct/Wildchat": ("wildchat", "wildchat_ru"),
             "Dolci-Instruct/Precise IF": ("dolci_precise_if", "dolci_precise_if_ru"),
             "Dolci-Instruct/WildJailbreak": ("wildjailbreak", None),
             "Dolci-Instruct/WildGuardMix": ("wildguardmix", None), "Dolci-Instruct/CoCoNot": ("coconot", None)}
    for r in dolci:
        en, ru = split[r["source"]]
        records[ru if (ru and r["lang"] == "ru") else en].append(r)
    # The load-stage counters of the five Dolci pools come from the same files, so they are kept together.
    struct["dolci_instruct (wildchat, precise_if, wildjailbreak, wildguardmix, coconot; en and ru)"] = dc

    # Pools assembled from several files get one more dedup pass across their files.
    for name in list(records):
        records[name] = dedup_local(records[name], struct[name])
        struct[name]["candidates_after_load"] = len(records[name])
    return records, struct


# ----------------------------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------------------------
CHAT_LIKE = {"wildchat", "wildchat_ru", "magpie_ultra", "twix", "grandmaster_ru", "ru_sft_chat", "lmsys_ru"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--raw-hf", required=True,
                    help="dir of the plain Hugging Face downloads that download_hf.py writes, $ORBIT_RAW/hf "
                         "(load_pools lists them)")
    ap.add_argument("--hf-cache", required=True,
                    help="a Hugging Face home whose datasets/ dir holds the datasets caches of BAAI/taco, "
                         "codeparrot/apps, deepmind/code_contests and lmsys/lmsys-chat-1m")
    ap.add_argument("--ru-math", default=None,
                    help="optional: a jsonl of Russian math problems with reference answers (instruction, response)")
    ap.add_argument("--ru-sft", dest="ru_sft_dir", default=None,
                    help="optional: a dir of Russian SFT chats, one <role>.jsonl per role in messages format")
    ap.add_argument("--rl-math", default=None,
                    help="optional: a math jsonl (instruction column) whose problems the pool must not contain, "
                         "for example the source of the RL math keep")
    ap.add_argument("--hf-hub", required=True, help="offline Hugging Face hub cache with the eval datasets")
    ap.add_argument("--trivial", default=TRIVIAL_DEFAULT,
                    help="chat_probe_trivial.jsonl of the SFT evaluation (default: the repo's copy)")
    ap.add_argument("--tokenizer", required=True, help="the Qwen3 tokenizer dir that counts prompt tokens")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ev = EvalIndex(args.hf_hub, args.trivial)
    log("eval index:", ev.sizes, "grams", len(ev.gram), "template grams", len(ev.template_gram))
    rl_math = RlMathIndex(args.rl_math)
    log("RL pool index:", rl_math.n, "problems", "" if args.rl_math else "(no --rl-math)")
    missing = []
    pools, struct = load_pools(args.workers, {"raw_hf": args.raw_hf, "hf_cache": args.hf_cache,
                                              "ru_math": args.ru_math, "ru_sft_dir": args.ru_sft_dir}, missing)
    if missing:
        log("optional inputs not given or not found, so their pools stay empty:", missing)
    log("pools loaded:", {k: len(v) for k, v in pools.items()})

    # IFEval constraint sentences come in many phrasings, so one phrasing may occur in only one or two
    # IFEval prompts and still be a template that thousands of IF training prompts reuse. A span that
    # at least IF_TEMPLATE_MIN_PROMPTS different IF training prompts contain is treated as template text.
    freq = collections.Counter()
    for pool in IF_POOLS:
        for r in pools.get(pool, []):
            freq.update(g for g in gram_hashes(r["core"]) if ev.gram.get(g) == "ifeval")
    shared = {g for g, n in freq.items() if n >= IF_TEMPLATE_MIN_PROMPTS}
    ev.promote_templates(shared)
    log("IFEval spans shared by >=", IF_TEMPLATE_MIN_PROMPTS, "IF training prompts:", len(shared))

    rng = random.Random(SEED)
    for name in sorted(pools):
        random.Random(f"{SEED}:{name}").shuffle(pools[name])
    cursor = collections.Counter()
    scan = collections.defaultdict(collections.Counter)
    eval_hits = collections.defaultdict(collections.Counter)
    template_only = collections.defaultdict(collections.Counter)
    seen_norm = set()
    accepted_grams = set()
    selected = []

    def take(pool, domain, lang, want):
        """Scans the pool from its cursor and accepts up to `want` prompts; returns how many it took."""
        got = 0
        cand = pools.get(pool, [])
        sc = scan[pool]
        while got < want and cursor[pool] < len(cand):
            r = cand[cursor[pool]]
            cursor[pool] += 1
            sc["scanned"] += 1
            text, core = r["text"].strip(), r["core"].strip()
            if len(text) < MIN_CHARS or len(core) < MIN_CHARS:
                sc["shorter_than_8_chars"] += 1
                continue
            if SPECIAL.search(text):
                sc["chat_special_tokens"] += 1
                continue
            if r["lang"] != lang:
                sc[f"lang_{r['lang']}_not_{lang}"] += 1
                continue
            if lang == "en" and domain in ("chat", "if", "refusal") and latin_prompt_not_english(core):
                sc["latin_script_prompt_not_english"] += 1
                continue
            if IDENTITY.search(text):
                sc["identity_question"] += 1
                continue
            if MODEL_NAMES.search(text):
                sc["model_name_qwen_nemotron_olmo_llama"] += 1
                continue
            if MODEL_NAMES_EXTRA.search(text):
                sc["model_name_other_assistant_or_developer"] += 1
                continue
            if pool in CHAT_LIKE and (EMAIL.search(text) or PHONE.search(text)):
                sc["pii_email_or_phone"] += 1
                continue
            if pool in CHAT_LIKE and PROFANITY.search(text):
                sc["profanity"] += 1
                continue
            key = blake(norm_text(core))
            if key in seen_norm:
                sc["duplicate_across_sources"] += 1
                continue
            gh = gram_hashes(core)
            name, how, tmpl = ev.match(core, gh, domain == "math", r["lang"] == "ru")
            if name:
                sc["eval_overlap"] += 1
                eval_hits[name][how] += 1
                continue
            if tmpl:
                template_only[tmpl][pool] += 1
            if domain == "math":
                how = rl_math.match(core, gh, r["lang"] == "ru")
                if how:
                    sc["rl_pool_overlap_" + how] += 1
                    continue
            n_tok = len(tok.encode(text, add_special_tokens=False))
            if n_tok > MAX_TOKENS:
                sc["longer_than_1024_tokens"] += 1
                continue
            if gh and sum(g in accepted_grams for g in gh) >= NEAR_DUP_OVERLAP * len(gh):
                sc["near_duplicate_80pct_13grams"] += 1
                continue
            seen_norm.add(key)
            accepted_grams.update(gh)
            selected.append({"prompt": text, "domain": domain, "lang": lang, "source": r["source"],
                             "source_ref": r["ref"], "pool": pool, "n_tokens": n_tok})
            sc["accepted"] += 1
            got += 1
        return got

    shortfall = {}
    for domain, lang, total, sources in PLAN:
        got = 0
        for pool, quota in sources:
            got += take(pool, domain, lang, quota)
        # A pool that ran dry leaves a gap; the other pools of the cell fill it in plan order.
        for pool, _ in sources:
            if got >= total:
                break
            got += take(pool, domain, lang, total - got)
        if got < total:
            shortfall[f"{domain}/{lang}"] = total - got
        log(f"{domain}/{lang}: {got}/{total}")

    # Modes: exactly half think inside each domain and language, in a seeded random order.
    cells = collections.defaultdict(list)
    for s in selected:
        cells[(s["domain"], s["lang"])].append(s)
    for key in sorted(cells):
        items = cells[key]
        order = list(range(len(items)))
        random.Random(f"{SEED}:mode:{key}").shuffle(order)
        for rank, i in enumerate(order):
            items[i]["mode"] = "think" if rank % 2 == 0 else "no_think"

    rng.shuffle(selected)
    ids = set()
    for s in selected:
        s["id"] = "opd-" + blake(norm_text(s["prompt"]), 6)
        assert s["id"] not in ids
        ids.add(s["id"])

    fields = ["id", "prompt", "mode", "domain", "lang", "source", "source_ref"]
    with open(os.path.join(args.out, "prompts.jsonl"), "w") as fh:
        for s in selected:
            fh.write(json.dumps({k: s[k] for k in fields}, ensure_ascii=False) + "\n")

    n = len(selected)
    cnt = lambda key: dict(sorted(collections.Counter(key(s) for s in selected).items()))
    tok_by_domain = {}
    for d in sorted({s["domain"] for s in selected}):
        L = sorted(s["n_tokens"] for s in selected if s["domain"] == d)
        tok_by_domain[d] = {"p50": L[len(L) // 2], "p90": L[int(len(L) * 0.9)], "max": L[-1],
                            "mean": round(statistics.mean(L), 1)}
    drop_totals = collections.Counter()
    for p, c in scan.items():
        for k, v in c.items():
            if k not in ("scanned", "accepted"):
                drop_totals[k] += v
    stats = {
        "total": n,
        "seed": SEED,
        "config": {"ngram": NGRAM, "template_gram_min_items": TEMPLATE_GRAM_MIN_ITEMS,
                   "if_template_min_training_prompts": IF_TEMPLATE_MIN_PROMPTS,
                   "rl_pool_overlap": RL_POOL_OVERLAP, "near_dup_overlap": NEAR_DUP_OVERLAP,
                   "max_tokens": MAX_TOKENS, "min_chars": MIN_CHARS, "tokenizer": args.tokenizer,
                   "plan": [{"domain": d, "lang": l, "target": t, "sources": dict(s)} for d, l, t, s in PLAN]},
        "shares": {"domain": {k: round(v / n, 4) for k, v in cnt(lambda s: s["domain"]).items()},
                   "lang": {k: round(v / n, 4) for k, v in cnt(lambda s: s["lang"]).items()},
                   "mode": {k: round(v / n, 4) for k, v in cnt(lambda s: s["mode"]).items()}},
        "counts": {
            "domain": cnt(lambda s: s["domain"]),
            "lang": cnt(lambda s: s["lang"]),
            "mode": cnt(lambda s: s["mode"]),
            "domain_lang_mode": cnt(lambda s: f"{s['domain']}|{s['lang']}|{s['mode']}"),
            "source": cnt(lambda s: s["source"]),
            "pool": cnt(lambda s: s["pool"]),
        },
        "shortfall_vs_plan": shortfall,
        "optional_inputs_missing": missing,
        "prompt_tokens_by_domain": tok_by_domain,
        "dropped": {
            "totals_over_scanned_candidates": dict(drop_totals.most_common()),
            "load_stage_per_pool": {k: dict(v) for k, v in sorted(struct.items())},
            "scan_stage_per_pool": {k: dict(v) for k, v in sorted(scan.items())},
            "eval_overlap_by_eval_set": {k: dict(v) for k, v in sorted(eval_hits.items())},
            "kept_although_matching_eval_template_spans": {k: dict(v) for k, v in sorted(template_only.items())},
        },
        "eval_index": {"items": ev.sizes, "template_grams_by_set": dict(ev.template_counts)},
        "rl_pool_problems_indexed": rl_math.n,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "stats.json"), "w") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=1)
    log("done:", n, "prompts", stats["shares"], "shortfall", shortfall)


if __name__ == "__main__":
    main()
