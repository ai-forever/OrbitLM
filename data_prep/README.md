# Data preparation

Русская версия ниже.

This folder builds the tokenized data of training stages 1 to 5: the tokenizers, the 4K
pretraining streams, the 16K and 32K context-extension streams and the SFT v5 mix.

## Setup

Run the commands from the repository root in the data environment of the root `README.md`. The SFT
`--decode` check and the tests also import `orbit`, so they need `requirements/train.txt` in the
same environment. The configs read the outputs here:

| Stage | Written to | Config entry |
|---|---|---|
| Tokenizers | `$ORBIT_TOKENIZERS/<name>/` | `tokenizer_config: ${ORBIT_TOKENIZERS}/<name>/tokenizer_config.json` |
| 4K, stages 1 and 2 | `$ORBIT_DATA/pretrain_4k/en/<stream>/merged_shards/`, `$ORBIT_DATA/pretrain_4k/ru/<stream>/merged_shards/` | `stage1_root:` is the `en` root; Russian streams are named by full path |
| 16K, stage 3 | `$ORBIT_DATA/ctx16k/streams/<stream>/` | `stage1_root: ${ORBIT_DATA}/ctx16k/streams` |
| 32K, stage 4 | `$ORBIT_DATA/ctx32k/streams/<stream>/` | `stage1_root: ${ORBIT_DATA}/ctx32k/streams` |
| SFT v5, stage 5 | `$ORBIT_DATA/sft/v5/<stream>/` | `stage1_root: ${ORBIT_DATA}/sft/v5` |

`python data_prep/long_context/download_hf.py [--plan PLAN.json]` downloads the Hugging Face
sources at pinned revisions into `$ORBIT_RAW/hf/<org>__<name>/`: the 16K sources by default, and
the rest of SFT with `--plan data_prep/sft/download_plan.json`. `SynthLabsAI/Big-Math-RL-Verified`
is gated, so accept its terms on the Hub and log in with `huggingface-cli login` or `HF_TOKEN`
first.

## Russian data that you supply

The repository ships no Russian data, and `docs/DATA.md` describes the formats. Pretraining text
goes into the folders `web`, `books`, `science`, `wikipedia` and `wikiquote` as `.jsonl`,
`.jsonl.gz`, `.jsonl.zst`, `.jsonl.zstd` or `.parquet` files with a `text` field, and each folder
becomes the 4K stream `ru_<folder>`. SFT chats go into `$ORBIT_RAW/russian_sft/<role>.jsonl`, one
messages-format chat per line, for the roles `science`, `math`, `code`, `dialog`, `tasks`,
`creative`, `education`, `qa_reasoning` and `summarization`. An assistant turn may open with
`<think>…</think>`. The builders only check roles and
drop exact duplicates and overlong chats, so clean the chats first. SFT stops when a role file is
missing. The ru_sft store of 16K and 32K reads every role except `tasks` and `education` and stops
only when no file exists.

To build without one of these streams, delete it from the config's `stage1_streams` and
`stage1_dataset_proportions`, and for SFT from the version yaml. For 16K and 32K, also delete it
under the same name from `STREAMS` and from the mix (`MIX` in `long_context/stage_16k.py`,
`LONG_SHARES` or `GENERAL_SHARES` in `long_context/stage_32k.py`). The Russian streams there are
ru_web_le8k, ru_wiki, ru_books or ru_books_long, and ru_sft. Without the pretraining text, also
delete `RU` from the stage's `PREFLIGHT_PATHS`, or the preflight stops the build. Then run
`mix.py` again.

## Tokenizers

```bash
python data_prep/tokenizer/prepare_tokenizers.py --out-dir "$ORBIT_TOKENIZERS"
```

| Directory | Used by | Built from |
|---|---|---|
| `tokenizer_qwen3_orbit` | 4K, 16K, 32K | `Qwen/Qwen3-1.7B-Base` and its own chat template |
| `tokenizer_qwen3_4b_qwen35_think` | the in-training evaluation of SFT (`eval_tokenizer_source`) | `Qwen/Qwen3-4B` and `templates/qwen35_think.jinja` |
| `tokenizer_qwen3_4b_qwen35_tools` | SFT v5 and its chat export | `Qwen/Qwen3-4B` and `templates/qwen35_system_tools.jinja` |

With `--offline` the script reads the bases from the local Hugging Face cache only. eos and pad are
`<|endoftext|>` (id 151643), the document boundary. The bases are pinned, and the output matches the
release byte for byte only with the transformers version of `requirements/data_prep.txt`.

## Pretraining 4K (stages 1 and 2)

`pretrain/tokenize_4k.py` writes one stream per call. `--name` is the directory name that the
configs use, `--out-root` is the `en` or `ru` root, and the raw layout is up to you.

```bash
EN=$ORBIT_DATA/pretrain_4k/en
RU=$ORBIT_DATA/pretrain_4k/ru
python data_prep/pretrain/tokenize_4k.py --out-root $EN --name nemotron-cc --src "$ORBIT_RAW/pretrain4k/nemotron-cc" --include '^quality=high'
python data_prep/pretrain/tokenize_4k.py --out-root $EN --name Nemotron-CC-Math-v1 --src "$ORBIT_RAW/pretrain4k/Nemotron-CC-Math-v1"
python data_prep/pretrain/tokenize_4k.py --out-root $RU --name web --src "$ORBIT_RAW/pretrain4k_ru/web"
# ... one call per English stream below and per Russian folder.
```

| Config stream | `--name` | Raw source |
|---|---|---|
| nemotron_cc_high | `nemotron-cc` | NVIDIA Nemotron-CC, quality=high: kind actual and the synthetic kinds distill, diverse_qa_pairs, extract_knowledge, knowledge_list, wrap_medium; 11,108 `.jsonl.zstd` files named `quality=high_kind=…_kind2=…_CC-MAIN-…-part-NNNNN` |
| nemotron_cc_math | `Nemotron-CC-Math-v1` | `nvidia/Nemotron-CC-Math-v1`, subset `3`, parquet `part_000000` to `000056` |
| megamath_high_quality | `MegaMath-high-quality` | `LLM360/MegaMath`, subset `megamath-web-pro`, parquet `000_00000` to `000_00099` |
| opc_annealing_corpus | `opc-annealing-corpus` | `OpenCoder-LLM/opc-annealing-corpus`: algorithmic_corpus, synthetic_code_snippet, synthetic_qa |
| nemotron_synthetic_code | `Nemotron-Post-Training_merged` | jsonl text renderings of `nvidia/Nemotron-Instruction-Following-Chat-v1` (chat_if, structured_outputs), `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2`, `nvidia/Nemotron-Science-v1`, `nvidia/Nemotron-SpecializedDomains-Finance-v1` |
| nemotron_sft_merged | `Nemotron-Cascade-SFT-Stage-1_merged` | jsonl text renderings of `nvidia/Nemotron-Cascade-SFT-Stage-1`: code, general_science, math |
| nemotron_pretraining_specialized (stage 2 only) | `Nemotron-Pretraining-Specialized-v1` | `nvidia/Nemotron-Pretraining-Specialized-v1` |

Any plain-text rendering of the two chat sets can stand in. The 16K and 32K stages find the
raw-file boundaries again from the partial last shard of each file, so the 16K nemotron_cc stream
matches the release only when `nemotron-cc` is rebuilt from the same 11,108 files.

## Context extension 16K (stage 3)

This stage writes 25 streams of 16,385-token samples for a 200B-token run, as
`long_context/stage_16k.py` defines them, with the shares in `MIX`. The `build_*.py` builders write
document stores to `$ORBIT_DATA/ctx16k/work/docs/<store>/`. `run_tasks.py` then packs, merges and
checks the streams and can be restarted at any time, and `mix.py` writes the config's data block.

Besides the downloads, the 4K streams and the Russian SFT chats, the build reads Dolma 3 from
`allenai/dolma3_mix-6T`, whose revision is not pinned. `$ORBIT_RAW/dolma3/dolma_3_mix_6t/data/`
mirrors its `data/` folder, from which the builders read
`olmocr_science_pdfs-science_math_and_technology-part1` and `-part2`,
`olmocr_science_pdfs-health`, `olmocr_science_pdfs-literature` and `rpj-proofpile-arxiv`, with the
arXiv file names of the dataset. `$ORBIT_RAW/dolma3/dolma_3_mix_excluded/` holds its
`stack_edu-Python` and `stack_edu-Java`.

```bash
LC=data_prep/long_context
python $LC/download_hf.py                         # Big-Math is gated, see Setup
# Question-answering and math train splits.
python $LC/build_bench_train.py
python $LC/build_bench_math.py omr-scan --part P --nparts 2           # for P in 0..1
python $LC/build_bench_math.py omr-select
python $LC/build_bench_math.py omr --part P --nparts 2 --overwrite    # for P in 0..1
python $LC/build_bench_math.py bigmath --overwrite                    # needs omr-select
python $LC/build_bench_math.py deepmind --part P --nparts 2 --per-module 1000000 --overwrite   # for P in 0..1
python $LC/build_bench_math.py manifest
python $LC/build_bench_mcq.py qasc --overwrite
python $LC/build_bench_mcq.py manifest
# Instructions.
python $LC/build_ru_open.py twix
python $LC/build_ru_open.py grandmaster           # needs the T-Wix prompt hashes that twix saves
python $LC/build_ru_open.py oasst2_ru
python $LC/build_ru_open.py ru_sft                # the Russian SFT chats in $ORBIT_RAW/russian_sft
python $LC/build_chat_en.py build --stream nemotron_if --part P --nparts 2   # for P in 0..1
python $LC/build_chat_en.py manifest --stream nemotron_if --nparts 2
python $LC/build_chat_en.py build --stream smoltalk2_en --part 0 --nparts 1
python $LC/build_chat_en.py manifest --stream smoltalk2_en --nparts 1
# Reasoning: the step order is in the docstring of build_reasoning.py.
python $LC/build_reasoning.py scan --stream acereason
python $LC/build_reasoning.py scan --stream openthoughts3   # needs the acereason scan
python $LC/build_reasoning.py select --stream acereason     # and the same for openthoughts3
python $LC/build_reasoning.py build --stream acereason      # tier r0-2; the same for openthoughts3
#   ... then anskey, build --ranks 2:4 (acereason) / 2:8 (openthoughts3), count --label r0-2,
#   extend, the count stages that extend asks for, build --tier backfill, verify, manifest.
# Long documents.
python $LC/build_pdf_arxiv.py build --stream olmocr_science --part P --nparts 160 --overwrite   # for P in 0..159
python $LC/build_pdf_arxiv.py build --stream olmocr_health --part P --nparts 80 --overwrite     # for P in 0..79
python $LC/build_pdf_arxiv.py build --stream olmocr_literature --part P --nparts 2 --overwrite  # for P in 0..1
python $LC/build_pdf_arxiv.py build --stream arxiv --part P --nparts 10 --overwrite             # for P in 0..9
#   after each stream: build_pdf_arxiv.py manifest --stream S --nparts N; build_pdf_arxiv.py crossdup --stream S
python $LC/build_finepdfs_ru.py --part P --nparts 36        # for P in 0..35
python $LC/build_finepdfs_ru.py --manifest --nparts 36
python $LC/build_code_repos.py route --lang Python --part 0 --nparts 1
python $LC/build_code_repos.py route --lang Java --part 0 --nparts 1
python $LC/build_code_repos.py build --part P --nparts 48   # for P in 0..47, after both routes
python $LC/build_code_repos.py manifest
# Packing, checks and the mix.
python $LC/run_tasks.py --stage 16k --dry-run
python $LC/run_tasks.py --stage 16k
python $LC/mix.py --stage 16k                     # writes $ORBIT_DATA/ctx16k/manifests/stage16k_data.yaml
```

A command marked "for P in …" runs once per part, and the parts can run in parallel.
`run_tasks.py --stage 16k --tasks FILE.json` spreads such a list over the CPUs of one machine.
`check_stream.py` checks a stream the way the trainer reads it, and `check_store.py` prints
documents of a store. At the release shares `mix.py` warns that bench_train and bench_mcq exceed
`MAX_PASSES`; lower their shares in `MIX` to repeat them less.

To match the release, delete the `part_0029_*` files of olmocr_health before packing, because the
release left out build part 29 of 80. The release stores acereason and openthoughts3 were built in
several passes, the first acereason tier with an older `documents.py`. `build_reasoning.py` models
that difference with `old_library_bad` and `recheck`, but a rebuild from scratch can still differ
slightly.

## Context extension 32K (stage 4)

This stage writes 29 streams of 32,769-token samples, 8 long and 21 short, for a 20B-token run,
all defined in `long_context/stage_32k.py`. It reads the 16K document stores, the 4K streams and a
text copy of ProLong's training data in `$ORBIT_RAW/prolong-text/data/`.

```bash
LC=data_prep/long_context
python $LC/run_tasks.py --stage 32k --dry-run     # creates $ORBIT_DATA/ctx32k if it is missing
python $LC/run_tasks.py --stage 32k               # build_prolong_code (8 parts), then pack, merge and check of every stream
python $LC/mix.py --stage 32k                     # writes $ORBIT_DATA/ctx32k/manifests/stage32k_data.yaml
```

The long streams, 40% of the tokens, hold documents of at least 16,384 tokens, cut at 131,072.
The other 60% repack the other 16K streams and the 4K STEM streams as 32K blocks. The caps
(`--target-tokens`) are about 1.5 times what the run needs, so a change of `LONG_SHARES`,
`STEM_SHARES` or `GENERAL_SHARES` needs only a `mix.py` rerun while no stream exceeds
`MAX_PASSES`. bench_mcq is small and gets many passes, which `mix.py` reports.

The ProLong copy holds the 64K-token windows of `princeton-nlp/prolong-data-64K`, decoded from
Llama-3 tokens back to text: 241 parquet files, `train-00000-of-00241.parquet` to
`train-00240-of-00241.parquet`, with the columns `text` and `source`. `build_prolong_code.py`
reads only the first two files, which hold all code rows, and stops when the sorted file list does
not start with them. It keeps the rows whose `source` is
`thestackv1_concat_by_repo-long/65536-65536`.

## SFT v5 (stage 5)

This stage packs whole chats into 41 streams of 32,769-token blocks, about 11.7B tokens, in
`$ORBIT_DATA/sft/v5/<stream>/` and drops longer chats. The targets are the assistant answers with
their closing `<|im_end|>` and newline. Next to the streams it writes `<name>.resolved.yaml` and
the config's data block `<name>_data.yaml`. The mix is `data_prep/sft/versions/v5_smoltalk2.yaml`,
the mix of `configs/train/sft/sft_v5.yaml`, rendered with the tools tokenizer.

```bash
python data_prep/long_context/download_hf.py      # the whole 16K plan (Big-Math is gated, see Setup); SFT reads only T-Wix and the bench_train sources of it
python data_prep/long_context/download_hf.py --plan data_prep/sft/download_plan.json
python data_prep/sft/prepare_tools.py             # writes $ORBIT_DATA/sft/tools/*.jsonl
V=data_prep/sft/versions/v5_smoltalk2.yaml
python data_prep/sft/build.py --version $V --build --jobs 16
python data_prep/sft/build.py --version data_prep/sft/versions/v5_smoltalk2.yaml --progress   # from another shell, while it runs
python data_prep/sft/build.py --version $V --resolve
python data_prep/sft/build.py --version $V --decode OpenThoughts3_1.2M_think   # checks one block against the trainer
```

The build reads the SmolTalk2 SFT splits, T-Wix, Aya (Russian rows), GrandMaster-PRO-MAX, Dolci
and the bench_train sources under `$ORBIT_RAW/hf/`. The SFT plan pins only SmolTalk2, so the other
sources can come at newer revisions than in the release. `prepare_tools.py` also needs two
100k-row tool-calling samples,
`$ORBIT_RAW/nemotron_tool_samples/Nemotron-SFT-Agentic-v2/tool_calling_100k.jsonl` and
`$ORBIT_RAW/nemotron_tool_samples/Nemotron-Post-Training-Dataset-v1/tool_calling_100k.parquet`.
Any 100k-row sample of the tool-calling rows of those two NVIDIA datasets can stand in. A stream
of your own messages-format chats needs only a version yaml entry of the `jsonl` kind. The epoch
counts that `--resolve` prints are for the version's 10B budget, while the v5 run trains on 14.7B
tokens.

The v5 yaml keeps every system prompt verbatim, while the release run rewrote some SmolTalk ones.
`collect_system_prompts.py --out DIR` writes a rewrite table for them to `DIR/rewrite_table.json`.
It maps an exact prompt to its replacement, and a `null` or empty value drops the turn. Prompts
that the script cannot settle stay verbatim and are listed in `DIR/disputed.jsonl` and
`DIR/REVIEW.md` for a person to add, and `--saved FILE` keeps the manual edits of an earlier table.
The table matches the release one except for two persona prompts in which "OpenAI" was replaced by
"John" by hand. Pass it with `build.py --rewrite-table FILE`, or as `rewrite_table: FILE` in the
version yaml, relative to the yaml. A named table that is missing stops the build.

## SFT v6 to v9 (later mixes)

These four versions are later SFT mixes built with the same code. They are not the release recipe,
and the repository has no training config for them. To train one, copy
`configs/train/sft/sft_v5.yaml` and replace its data block with the `<name>_data.yaml` that
`--resolve` writes. The header of each version yaml explains its numbers.

| Version | What changes |
|---|---|
| v6 | the v5 mix with a chain filter: a chat is dropped when its longest reasoning chain is longer than the larger of 8,192 tokens and the 95th percentile of its stream |
| v7 | v6 plus ToolMind (open_datasets) in `addition_tools`, with loss on the last assistant turn only |
| v8 | v6 plus ToolMind and a `code` part: 200k AceReason code rows, KodCode, self-oss-instruct, Code-Feedback and BIRD |
| v9 | the v8 streams with a larger tool part |

```bash
python data_prep/long_context/download_hf.py --plan data_prep/sft/download_plan_v7_v9.json   # about 98 GB, 93 GB of it AceReason
python data_prep/sft/prepare_tools.py --only toolmind_open    # v7 to v9: $ORBIT_DATA/sft/tools/toolmind_open.jsonl
python data_prep/sft/prepare_acereason.py --target 200000 --out-dir $ORBIT_RAW/hf/nvidia__AceReason-1.1-SFT/code_200k   # v8, v9
python data_prep/sft/prepare_code.py                          # v8, v9: $ORBIT_DATA/sft/code/*.jsonl
V=data_prep/sft/versions/v9_smoltalk2.yaml
python data_prep/sft/build.py --version $V --build --jobs 16
python data_prep/sft/build.py --version $V --resolve
```

These steps come after the v5 steps above, which download the shared sources and write the other
tool files. Set `V` to the version you build. v6 needs none of the downloads above.

The yamls of v7 to v9 name an earlier build in `stats_root`. It lets a build reuse the streams of
that earlier build instead of packing them again. To reuse them, link the stream directories of
the earlier build into the new `out_root` and pack only the new streams with `--streams`.
`--resolve` then reads the earlier counts from `stats_root/stats.json` and puts the new ones on
top. Without an earlier build, build every stream as above. `stats_root` may then be missing, and
`--resolve` takes every count from the new `out_root`. In both cases `--resolve` stops when a
stream has no count.

`label_ot3.py`, with `label_ot3_count.py`, measures how much of the OpenThoughts3 streams is code,
math and science after the v6 filter. Those shares set the code fraction of v8 and v9, and a build
does not need them. It reads OpenThoughts3-1.2M from the 16K plan and the v6 build.

## Tests

```bash
python -m pytest -q data_prep -p no:cacheprovider
```

The tests run on CPU with synthetic data. They take the tokenizers from `$ORBIT_TOKENIZERS` or
write them from the local Hugging Face cache, and the tests that need them are skipped when
neither is available.

---

# Подготовка данных

Эта папка собирает токенизированные данные стадий обучения с 1 по 5: токенизаторы, потоки
претрейна 4K, потоки расширения контекста до 16K и 32K и смесь SFT v5.

## Установка

Команды запускаются из корня репозитория в окружении для данных, которое описано в корневом
`README.md`. Проверка SFT `--decode` и тесты импортируют ещё и `orbit`, поэтому им нужен
`requirements/train.txt` в том же окружении. Конфиги читают результаты по этим путям:

| Стадия | Куда пишется | Запись в конфиге |
|---|---|---|
| Токенизаторы | `$ORBIT_TOKENIZERS/<name>/` | `tokenizer_config: ${ORBIT_TOKENIZERS}/<name>/tokenizer_config.json` |
| 4K, стадии 1 и 2 | `$ORBIT_DATA/pretrain_4k/en/<stream>/merged_shards/`, `$ORBIT_DATA/pretrain_4k/ru/<stream>/merged_shards/` | `stage1_root:` указывает на корень `en`; русские потоки задаются полным путём |
| 16K, стадия 3 | `$ORBIT_DATA/ctx16k/streams/<stream>/` | `stage1_root: ${ORBIT_DATA}/ctx16k/streams` |
| 32K, стадия 4 | `$ORBIT_DATA/ctx32k/streams/<stream>/` | `stage1_root: ${ORBIT_DATA}/ctx32k/streams` |
| SFT v5, стадия 5 | `$ORBIT_DATA/sft/v5/<stream>/` | `stage1_root: ${ORBIT_DATA}/sft/v5` |

`python data_prep/long_context/download_hf.py [--plan PLAN.json]` скачивает источники с Hugging Face
на закреплённых ревизиях в `$ORBIT_RAW/hf/<org>__<name>/`: по умолчанию источники 16K, а с
`--plan data_prep/sft/download_plan.json` остальные источники SFT.
`SynthLabsAI/Big-Math-RL-Verified` выдаётся только по запросу, поэтому сначала примите его условия
на Hub и войдите через `huggingface-cli login` или `HF_TOKEN`.

## Русские данные, которые вы готовите

Русских данных в репозитории нет, их форматы описаны в `docs/DATA.md`. Текст для претрейна
кладётся в папки `web`, `books`, `science`, `wikipedia` и `wikiquote` файлами `.jsonl`,
`.jsonl.gz`, `.jsonl.zst`, `.jsonl.zstd` или `.parquet` с полем `text`, и каждая папка становится
потоком 4K `ru_<folder>`. Чаты для SFT кладутся в `$ORBIT_RAW/russian_sft/<role>.jsonl`, по одному
чату в формате messages на строку, для ролей `science`, `math`, `code`, `dialog`, `tasks`,
`creative`, `education`, `qa_reasoning` и `summarization`. Ответ ассистента может начинаться с
`<think>…</think>`. Сборщики только проверяют роли и
убирают точные дубликаты и слишком длинные чаты, так что чистить чаты нужно заранее. SFT
останавливается, если нет файла какой-то роли. Хранилище ru_sft для 16K и 32K читает все роли,
кроме `tasks` и `education`, и останавливается, только если нет ни одного файла.

Чтобы собрать данные без одного из этих потоков, удалите его из `stage1_streams` и
`stage1_dataset_proportions` в конфиге, а для SFT из yaml версии. Для 16K и 32K удалите его под тем
же именем ещё из `STREAMS` и из смеси (`MIX` в `long_context/stage_16k.py`, `LONG_SHARES` или
`GENERAL_SHARES` в `long_context/stage_32k.py`). Русские потоки там называются ru_web_le8k,
ru_wiki, ru_books или ru_books_long и ru_sft. Без текста для претрейна удалите также `RU` из
`PREFLIGHT_PATHS` стадии, иначе сборку остановит предварительная проверка. Затем снова запустите
`mix.py`.

## Токенизаторы

```bash
python data_prep/tokenizer/prepare_tokenizers.py --out-dir "$ORBIT_TOKENIZERS"
```

| Папка | Где используется | Из чего собрана |
|---|---|---|
| `tokenizer_qwen3_orbit` | 4K, 16K, 32K | `Qwen/Qwen3-1.7B-Base` и его собственный чат-шаблон |
| `tokenizer_qwen3_4b_qwen35_think` | оценка SFT во время обучения (`eval_tokenizer_source`) | `Qwen/Qwen3-4B` и `templates/qwen35_think.jinja` |
| `tokenizer_qwen3_4b_qwen35_tools` | SFT v5 и его чат-экспорт | `Qwen/Qwen3-4B` и `templates/qwen35_system_tools.jinja` |

С `--offline` скрипт берёт базовые токенизаторы только из локального кэша Hugging Face. eos и pad
равны `<|endoftext|>` (id 151643), это граница документов. Базовые токенизаторы закреплены на
ревизиях, и результат совпадает с релизным байт в байт только с версией transformers из
`requirements/data_prep.txt`.

## Претрейн 4K (стадии 1 и 2)

`pretrain/tokenize_4k.py` пишет один поток за вызов. `--name` задаёт имя папки, которое используют
конфиги, `--out-root` указывает на корень `en` или `ru`, а раскладку исходных файлов вы выбираете
сами.

```bash
EN=$ORBIT_DATA/pretrain_4k/en
RU=$ORBIT_DATA/pretrain_4k/ru
python data_prep/pretrain/tokenize_4k.py --out-root $EN --name nemotron-cc --src "$ORBIT_RAW/pretrain4k/nemotron-cc" --include '^quality=high'
python data_prep/pretrain/tokenize_4k.py --out-root $EN --name Nemotron-CC-Math-v1 --src "$ORBIT_RAW/pretrain4k/Nemotron-CC-Math-v1"
python data_prep/pretrain/tokenize_4k.py --out-root $RU --name web --src "$ORBIT_RAW/pretrain4k_ru/web"
# ... по одному вызову на каждый английский поток из таблицы ниже и на каждую русскую папку.
```

| Поток в конфиге | `--name` | Источник |
|---|---|---|
| nemotron_cc_high | `nemotron-cc` | NVIDIA Nemotron-CC, quality=high: kind actual и синтетические kind distill, diverse_qa_pairs, extract_knowledge, knowledge_list, wrap_medium; 11 108 файлов `.jsonl.zstd` с именами `quality=high_kind=…_kind2=…_CC-MAIN-…-part-NNNNN` |
| nemotron_cc_math | `Nemotron-CC-Math-v1` | `nvidia/Nemotron-CC-Math-v1`, подмножество `3`, parquet с `part_000000` по `000056` |
| megamath_high_quality | `MegaMath-high-quality` | `LLM360/MegaMath`, подмножество `megamath-web-pro`, parquet с `000_00000` по `000_00099` |
| opc_annealing_corpus | `opc-annealing-corpus` | `OpenCoder-LLM/opc-annealing-corpus`: algorithmic_corpus, synthetic_code_snippet, synthetic_qa |
| nemotron_synthetic_code | `Nemotron-Post-Training_merged` | текстовые версии в jsonl для `nvidia/Nemotron-Instruction-Following-Chat-v1` (chat_if, structured_outputs), `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2`, `nvidia/Nemotron-Science-v1`, `nvidia/Nemotron-SpecializedDomains-Finance-v1` |
| nemotron_sft_merged | `Nemotron-Cascade-SFT-Stage-1_merged` | текстовые версии в jsonl для `nvidia/Nemotron-Cascade-SFT-Stage-1`: code, general_science, math |
| nemotron_pretraining_specialized (только стадия 2) | `Nemotron-Pretraining-Specialized-v1` | `nvidia/Nemotron-Pretraining-Specialized-v1` |

Для двух наборов чатов подойдёт любая текстовая версия. Стадии 16K и 32K восстанавливают границы
исходных файлов по неполному последнему шарду каждого файла, поэтому поток nemotron_cc стадии 16K
совпадёт с релизным, только если `nemotron-cc` пересобран из тех же 11 108 файлов.

## Расширение контекста до 16K (стадия 3)

Стадия пишет 25 потоков с примерами по 16 385 токенов для прогона на 200 млрд токенов. Потоки
задаёт `long_context/stage_16k.py`, их доли лежат в `MIX`. Сборщики `build_*.py` пишут хранилища
документов в `$ORBIT_DATA/ctx16k/work/docs/<store>/`. Затем `run_tasks.py` упаковывает, сливает и
проверяет потоки, его можно перезапустить в любой момент, а `mix.py` пишет блок данных конфига.

Кроме скачанного, потоков 4K и русских чатов для SFT сборка читает Dolma 3 из
`allenai/dolma3_mix-6T`, ревизия которого не закреплена. `$ORBIT_RAW/dolma3/dolma_3_mix_6t/data/`
повторяет его папку `data/`, и сборщики читают из неё
`olmocr_science_pdfs-science_math_and_technology-part1` и `-part2`,
`olmocr_science_pdfs-health`, `olmocr_science_pdfs-literature` и `rpj-proofpile-arxiv`, а файлы
arXiv ожидают с именами из датасета. В `$ORBIT_RAW/dolma3/dolma_3_mix_excluded/` лежат его
`stack_edu-Python` и `stack_edu-Java`.

```bash
LC=data_prep/long_context
python $LC/download_hf.py                         # Big-Math выдаётся по запросу, см. раздел «Установка»
# Обучающие сплиты вопросно-ответных и математических датасетов.
python $LC/build_bench_train.py
python $LC/build_bench_math.py omr-scan --part P --nparts 2           # для P от 0 до 1
python $LC/build_bench_math.py omr-select
python $LC/build_bench_math.py omr --part P --nparts 2 --overwrite    # для P от 0 до 1
python $LC/build_bench_math.py bigmath --overwrite                    # после omr-select
python $LC/build_bench_math.py deepmind --part P --nparts 2 --per-module 1000000 --overwrite   # для P от 0 до 1
python $LC/build_bench_math.py manifest
python $LC/build_bench_mcq.py qasc --overwrite
python $LC/build_bench_mcq.py manifest
# Инструкции.
python $LC/build_ru_open.py twix
python $LC/build_ru_open.py grandmaster           # нужны хэши промптов T-Wix, которые сохраняет twix
python $LC/build_ru_open.py oasst2_ru
python $LC/build_ru_open.py ru_sft                # русские чаты для SFT в $ORBIT_RAW/russian_sft
python $LC/build_chat_en.py build --stream nemotron_if --part P --nparts 2   # для P от 0 до 1
python $LC/build_chat_en.py manifest --stream nemotron_if --nparts 2
python $LC/build_chat_en.py build --stream smoltalk2_en --part 0 --nparts 1
python $LC/build_chat_en.py manifest --stream smoltalk2_en --nparts 1
# Рассуждения: порядок шагов описан в docstring build_reasoning.py.
python $LC/build_reasoning.py scan --stream acereason
python $LC/build_reasoning.py scan --stream openthoughts3   # после scan для acereason
python $LC/build_reasoning.py select --stream acereason     # и то же для openthoughts3
python $LC/build_reasoning.py build --stream acereason      # уровень r0-2; то же для openthoughts3
#   ... затем anskey, build --ranks 2:4 (acereason) / 2:8 (openthoughts3), count --label r0-2,
#   extend, шаги count, которые запросит extend, build --tier backfill, verify, manifest.
# Длинные документы.
python $LC/build_pdf_arxiv.py build --stream olmocr_science --part P --nparts 160 --overwrite   # для P от 0 до 159
python $LC/build_pdf_arxiv.py build --stream olmocr_health --part P --nparts 80 --overwrite     # для P от 0 до 79
python $LC/build_pdf_arxiv.py build --stream olmocr_literature --part P --nparts 2 --overwrite  # для P от 0 до 1
python $LC/build_pdf_arxiv.py build --stream arxiv --part P --nparts 10 --overwrite             # для P от 0 до 9
#   после каждого потока: build_pdf_arxiv.py manifest --stream S --nparts N; build_pdf_arxiv.py crossdup --stream S
python $LC/build_finepdfs_ru.py --part P --nparts 36        # для P от 0 до 35
python $LC/build_finepdfs_ru.py --manifest --nparts 36
python $LC/build_code_repos.py route --lang Python --part 0 --nparts 1
python $LC/build_code_repos.py route --lang Java --part 0 --nparts 1
python $LC/build_code_repos.py build --part P --nparts 48   # для P от 0 до 47, после обоих route
python $LC/build_code_repos.py manifest
# Упаковка, проверки и смесь.
python $LC/run_tasks.py --stage 16k --dry-run
python $LC/run_tasks.py --stage 16k
python $LC/mix.py --stage 16k                     # пишет $ORBIT_DATA/ctx16k/manifests/stage16k_data.yaml
```

Команда с пометкой «для P от … до …» запускается один раз на каждую часть, и части можно
запускать параллельно. `run_tasks.py --stage 16k --tasks FILE.json` раскладывает такой список по
всем CPU одной машины. `check_stream.py` проверяет поток так, как его читает трейнер, а
`check_store.py` печатает документы из хранилища. При релизных долях `mix.py` предупреждает, что
bench_train и bench_mcq превышают `MAX_PASSES`. Чтобы они повторялись реже, уменьшите их доли в
`MIX`.

Чтобы совпасть с релизом, удалите перед упаковкой файлы `part_0029_*` из olmocr_health: в релизе
нет части сборки 29 из 80. Релизные хранилища acereason и openthoughts3 собирались в несколько
заходов, причём первый уровень acereason собран более ранней версией `documents.py`.
`build_reasoning.py` воспроизводит это различие через `old_library_bad` и `recheck`, но сборка с
нуля всё равно может немного отличаться.

## Расширение контекста до 32K (стадия 4)

Стадия пишет 29 потоков с примерами по 32 769 токенов, 8 длинных и 21 короткий, для прогона на
20 млрд токенов. Всё о ней задано в `long_context/stage_32k.py`. Она читает хранилища документов
16K, потоки 4K и текстовую копию обучающих данных ProLong в `$ORBIT_RAW/prolong-text/data/`.

```bash
LC=data_prep/long_context
python $LC/run_tasks.py --stage 32k --dry-run     # создаёт $ORBIT_DATA/ctx32k, если её нет
python $LC/run_tasks.py --stage 32k               # build_prolong_code (8 частей), затем упаковка, слияние и проверка каждого потока
python $LC/mix.py --stage 32k                     # пишет $ORBIT_DATA/ctx32k/manifests/stage32k_data.yaml
```

На длинные потоки приходится 40% токенов, в них документы от 16 384 токенов, обрезанные на 131 072.
Остальные 60% составляют прочие потоки 16K и STEM-потоки 4K, заново упакованные в блоки 32K.
Ограничения (`--target-tokens`) примерно в 1,5 раза больше, чем нужно прогону, поэтому после
правки `LONG_SHARES`, `STEM_SHARES` или `GENERAL_SHARES` достаточно снова запустить `mix.py`, пока
ни один поток не превышает `MAX_PASSES`. bench_mcq маленький и получает много проходов, о чём
сообщает `mix.py`.

Копия ProLong содержит окна по 64K токенов из `princeton-nlp/prolong-data-64K`, декодированные из
токенов Llama-3 обратно в текст: 241 файл parquet, от `train-00000-of-00241.parquet` до
`train-00240-of-00241.parquet`, со столбцами `text` и `source`. `build_prolong_code.py` читает
только первые два файла, в которых лежат все строки с кодом, и останавливается, если
отсортированный список файлов начинается не с них. Он оставляет строки, у которых `source` равен
`thestackv1_concat_by_repo-long/65536-65536`.

## SFT v5 (стадия 5)

Стадия упаковывает чаты целиком в 41 поток блоков по 32 769 токенов, всего около 11,7 млрд токенов,
в `$ORBIT_DATA/sft/v5/<stream>/`, а более длинные чаты отбрасывает. Цели обучения: ответы
ассистента вместе с закрывающим `<|im_end|>` и переводом строки. Рядом с потоками стадия пишет
`<name>.resolved.yaml` и блок данных конфига `<name>_data.yaml`. Смесь задана в
`data_prep/sft/versions/v5_smoltalk2.yaml`: это смесь `configs/train/sft/sft_v5.yaml`, которая
рендерится токенизатором tools.

```bash
python data_prep/long_context/download_hf.py      # весь план 16K (Big-Math выдаётся по запросу, см. раздел «Установка»); SFT читает из него только T-Wix и источники bench_train
python data_prep/long_context/download_hf.py --plan data_prep/sft/download_plan.json
python data_prep/sft/prepare_tools.py             # пишет $ORBIT_DATA/sft/tools/*.jsonl
V=data_prep/sft/versions/v5_smoltalk2.yaml
python data_prep/sft/build.py --version $V --build --jobs 16
python data_prep/sft/build.py --version data_prep/sft/versions/v5_smoltalk2.yaml --progress   # из другой оболочки, пока идёт сборка
python data_prep/sft/build.py --version $V --resolve
python data_prep/sft/build.py --version $V --decode OpenThoughts3_1.2M_think   # сверяет один блок с трейнером
```

Сборка читает сплиты SFT из SmolTalk2, T-Wix, Aya (русские строки), GrandMaster-PRO-MAX, Dolci и
источники bench_train из `$ORBIT_RAW/hf/`. План SFT закрепляет только SmolTalk2, так что остальные
источники могут прийти в более новых ревизиях, чем в релизе. `prepare_tools.py` нужны ещё две
выборки tool calling по 100 тысяч строк:
`$ORBIT_RAW/nemotron_tool_samples/Nemotron-SFT-Agentic-v2/tool_calling_100k.jsonl` и
`$ORBIT_RAW/nemotron_tool_samples/Nemotron-Post-Training-Dataset-v1/tool_calling_100k.parquet`.
Подойдёт любая выборка в 100 тысяч строк tool calling из этих двух датасетов NVIDIA. Для потока из
собственных чатов в формате messages достаточно записи в yaml версии с типом `jsonl`. Число эпох,
которое печатает `--resolve`, посчитано по бюджету версии в 10 млрд токенов, а прогон v5 обучается
на 14,7 млрд.

yaml версии v5 оставляет все системные промпты как есть, а в релизном прогоне часть системных
промптов SmolTalk заменили. `collect_system_prompts.py --out DIR` пишет для них таблицу замен в
`DIR/rewrite_table.json`. Она сопоставляет точному тексту промпта его замену, а значение `null` или
пустая строка убирает реплику. Промпты, по которым скрипт не может решить, остаются как есть и
перечислены в `DIR/disputed.jsonl` и `DIR/REVIEW.md`, чтобы их внёс в таблицу человек, а
`--saved FILE` сохраняет ручные правки прежней таблицы. Таблица совпадает с релизной, кроме двух
промптов с персонажем, в которых «OpenAI» вручную заменили на «John». Передайте её через
`build.py --rewrite-table FILE` или укажите в yaml версии строкой `rewrite_table: FILE`, путь
считается от yaml. Если указанной таблицы нет, сборка останавливается.

## SFT с v6 по v9 (поздние смеси)

Эти четыре версии появились позже и собираются тем же кодом. Это не релизный рецепт, и конфигов
трейнера для них в репозитории нет. Чтобы обучить одну из них, скопируйте
`configs/train/sft/sft_v5.yaml` и замените в нём блок данных на `<name>_data.yaml`, который пишет
`--resolve`. Числа каждой версии объяснены в шапке её yaml.

| Версия | Что меняется |
|---|---|
| v6 | смесь v5 с фильтром по цепочкам: чат отбрасывается, если его самая длинная цепочка рассуждения длиннее большего из двух чисел, 8192 токенов и 95-го перцентиля его потока |
| v7 | v6 и ToolMind (open_datasets) в `addition_tools`, лосс только на последней реплике ассистента |
| v8 | v6, ToolMind и часть `code`: 200 тысяч строк кода из AceReason, KodCode, self-oss-instruct, Code-Feedback и BIRD |
| v9 | потоки v8 с большей долей tool calling |

```bash
python data_prep/long_context/download_hf.py --plan data_prep/sft/download_plan_v7_v9.json   # около 98 ГБ, из них 93 ГБ AceReason
python data_prep/sft/prepare_tools.py --only toolmind_open    # v7, v8, v9: $ORBIT_DATA/sft/tools/toolmind_open.jsonl
python data_prep/sft/prepare_acereason.py --target 200000 --out-dir $ORBIT_RAW/hf/nvidia__AceReason-1.1-SFT/code_200k   # v8, v9
python data_prep/sft/prepare_code.py                          # v8, v9: $ORBIT_DATA/sft/code/*.jsonl
V=data_prep/sft/versions/v9_smoltalk2.yaml
python data_prep/sft/build.py --version $V --build --jobs 16
python data_prep/sft/build.py --version $V --resolve
```

Эти шаги идут после шагов v5 выше: те скачивают общие источники и пишут остальные файлы tool
calling. В `V` подставьте версию, которую собираете. Для v6 загрузки выше не нужны.

В yaml версий с v7 по v9 в `stats_root` указана более ранняя сборка. Так сборка может взять потоки
ранней сборки, а не упаковывать их заново. Для этого сделайте ссылки на каталоги потоков ранней
сборки в новом `out_root` и упакуйте только новые потоки через `--streams`. Тогда `--resolve`
возьмёт прежние числа из `stats_root/stats.json` и положит новые поверх. Без ранней сборки
собирайте все потоки, как выше. Тогда `stats_root` может не существовать, и `--resolve` берёт все
числа из нового `out_root`. В обоих случаях `--resolve` останавливается, если для какого-то потока
чисел нет.

`label_ot3.py` вместе с `label_ot3_count.py` измеряет, какая доля потоков OpenThoughts3 после
фильтра v6 приходится на код, математику и естественные науки. По этим долям выбрана доля кода в
v8 и v9, для сборки они не нужны. Скрипт читает OpenThoughts3-1.2M из плана 16K и сборку v6.

## Тесты

```bash
python -m pytest -q data_prep -p no:cacheprovider
```

Тесты идут на CPU на синтетических данных. Токенизаторы они берут из `$ORBIT_TOKENIZERS` или
собирают из локального кэша Hugging Face, а тесты, которым токенизаторы нужны, пропускаются, если
нет ни того, ни другого.
