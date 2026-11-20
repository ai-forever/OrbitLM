"""Every filesystem location that the data build reads or writes.

All locations hang off three environment variables, and the sub-paths below them are fixed, so
a tree built on one machine has the same layout on another (data_prep/README.md lists it):

  ORBIT_RAW         raw sources: Hugging Face downloads and the other raw inputs. Only
                     download_hf.py writes here; every build step only reads it.
  ORBIT_DATA        everything the build writes: the 4K pretraining streams and the 16K and
                     32K stage roots
  ORBIT_TOKENIZERS  the directories that data_prep/tokenizer/prepare_tokenizers.py writes

A script stops with a message naming the variable when one of them is not set.
"""
import os


def _root(variable):
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(f"set {variable} first; data_prep/README.md describes the three ORBIT_* roots")
    return value


RAW = _root("ORBIT_RAW")
DATA = _root("ORBIT_DATA")
TOKENIZERS = _root("ORBIT_TOKENIZERS")

# The stage roots: <root>/work/ (document stores and builder intermediates), streams/,
# manifests/ and logs/.
CTX16K_ROOT = os.path.join(DATA, "ctx16k")
CTX32K_ROOT = os.path.join(DATA, "ctx32k")
TOKENIZER_JSON = os.path.join(TOKENIZERS, "tokenizer_qwen3_orbit", "tokenizer.json")

# The 4K pretraining streams that data_prep/pretrain/tokenize_4k.py writes, one directory per
# stream: the English ones under en/ and the Russian ones under ru/. The 16K and 32K stages
# repack several of them instead of tokenizing their text again.
TOKENIZED_4K_EN = os.path.join(DATA, "pretrain_4k", "en")
TOKENIZED_4K_RU = os.path.join(DATA, "pretrain_4k", "ru")

# Hugging Face downloads, one directory per repository (<org>__<name>/). download_hf.py writes
# them, and most builders read them.
RAW_HF = os.path.join(RAW, "hf")
# A text copy of ProLong's 64K training data: 241 parquet files with the columns text and source.
# data_prep/README.md says what build_prolong_code.py reads from it.
PROLONG_TEXT = os.path.join(RAW, "prolong-text", "data")
# Russian SFT chats that the user supplies, one messages-format jsonl file per role
# (<role>.jsonl, for example math.jsonl). build_ru_open.py ru_sft renders them into the ru_sft
# store, and the SFT version yaml reads the same files.
RUSSIAN_SFT = os.path.join(RAW, "russian_sft")
# The Dolma 3 mix: olmOCR science PDFs and arXiv in dolma_3_mix_6t, Stack-Edu in
# dolma_3_mix_excluded. data_prep/README.md says what is known about where both come from.
DOLMA3_MIX = os.path.join(RAW, "dolma3", "dolma_3_mix_6t", "data")
DOLMA3_EXCLUDED = os.path.join(RAW, "dolma3", "dolma_3_mix_excluded")


def config_path(path):
    """Returns a path in the form that the training configs use.

    A path under ORBIT_DATA is written as ${ORBIT_DATA}/..., so that a config block written by
    a build script can be copied into a config without carrying this machine's paths. Any other
    path is returned unchanged.
    """
    data = os.path.abspath(DATA)
    full = os.path.abspath(path)
    if full == data:
        return "${ORBIT_DATA}"
    if os.path.commonpath([full, data]) != data:
        return path
    return "${ORBIT_DATA}/" + os.path.relpath(full, data).replace(os.sep, "/")
