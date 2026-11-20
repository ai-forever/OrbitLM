# Data tools of distillation and RL

Русская версия ниже.

These tools build the inputs of the verl recipes: the prompt pool and gold rows of distillation
(`configs/verl/distill_rl_sft_16k.yaml`) and the gold rows and validation sets of RL
(`configs/verl/rl_sft_mix.yaml`). Run them from the repo root with `$VERL_PYTHON`, the verl env's python;
`-m orbit_verl.data.code_data` imports the repository installed in that env. Steps 1 to 4 run on the CPU.
Step 5 runs the model and needs a runs root (`--runs-root` or `$ORBIT_VERL_RUNS`). You supply three inputs
of your own: math problems for `math_keep.jsonl`, Russian math problems and Russian SFT chats.

## Steps

```
# 1. Pool v1: prompts.jsonl (32,000 rows when every cell fills) and stats.json
$VERL_PYTHON scripts/verl/data/build_pool.py --out <v1 dir> --raw-hf $ORBIT_RAW/hf --hf-cache <hf_cache> \
    --hf-hub <hub cache> --tokenizer <Qwen3 tokenizer> [--rl-math <math set>] [--ru-math <ru math set>] \
    [--ru-sft $ORBIT_RAW/russian_sft]
# 2. Pool v2, the one the runs use: prompts.jsonl and fix_log.json
$VERL_PYTHON scripts/verl/data/fix_pool.py --pool <v1 dir>/prompts.jsonl --hf-hub <hub cache> --out <v2 dir> \
    [--flagged <ids.json>]
# 3. Code problems with tests: train.jsonl and val.jsonl (the release command)
$VERL_PYTHON -m orbit_verl.data.code_data build --out-dir <out> \
    --codeforces <hf_cache>/datasets/open-r1___codeforces \
    --code-contests <hf_cache>/datasets/deepmind___code_contests \
    --taco <hf_cache>/datasets/BAAI___taco \
    --primeintellect <DeepCoder-Preview-Dataset>/primeintellect \
    --lcb <LiveCodeBench code_generation_lite dir> --validate
# 4. Validation sets; the docstring has each command, and every input path must name the pinned revision
$VERL_PYTHON scripts/verl/data/build_val.py chat|lcb|math500|gsm8k|mgsm-ru ...
# 5. Pass rates of the start export; <work dir>/<kind>/results/<tag>/keep.jsonl is a gold input
bash scripts/verl/data/passrate/passrate.sh --kind math|code|math_ru --export <HF export> --model-name <name> \
    --tag <tag> --src <jsonl> --python <verl python> --runs-root <dir> [--work-dir <dir>] [--count N] \
    [--shard-from I --shard-to J | --merge-only | --prepare-only | --dry-run] [-- <hydra overrides>]
```

| file in `--data` of the run scripts | built by |
|---|---|
| `prompts.jsonl` | step 2 |
| `code_train.jsonl` | step 3, its `train.jsonl` renamed |
| `code_val.jsonl`, `lcb_val78.jsonl` | step 3 (`val.jsonl`) or step 4 `lcb`; the two files are identical |
| `math500.jsonl`, `gsm8k_en_paired250.jsonl`, `mgsm_ru.jsonl`, `chat_probe_ru.jsonl`, `chat_probe_en.jsonl` | step 4 |
| `math_keep.jsonl` | step 5, `--kind math --src <your math set>` |
| `code_keep.jsonl` | step 5, `--kind code --src code_train.jsonl` |
| `ru_math_keep.jsonl` | step 5, `--kind math_ru --src <your Russian math set>`; only the `mix_ru` RL arm reads it |

## Row formats

All files are jsonl, one object per line.

- Pool row: `id`, `prompt`, `mode` (`think`, `no_think`), `domain` (`math`, `code`, `if`, `chat`, `refusal`),
  `lang` (`en`, `ru`), `source`, `source_ref`.
- Gold math row (family `math` or `math_ru`): `instruction`, `response` (the gold answer or a solution
  ending in `\boxed{}`, compared with the last `\boxed{}` of the model's answer), `source`, `unique_id`.
- Gold code row (family `code`): `instruction`, `response` (the tests as a JSON string, see
  `orbit_verl.data.reward.parse_code_ground_truth`), `source`, `unique_id` and the fields of `code_data`.
- Keep row: the input row plus `n`, `n_correct`, `pass_rate`, `mean_response_len`, `truncated_frac`.
- Validation row: `instruction`, `response`. The chat probes have an empty `response` and add `lang`,
  `domain` and `source`.
- Russian SFT chat: `{"messages": [{"role": ..., "content": ...}, ...]}`, one `<role>.jsonl` per role (see
  `data_prep/README.md`).

## Inputs and pitfalls

`--raw-hf` is `$ORBIT_RAW/hf` after `data_prep/long_context/download_hf.py` has run its default plan and
`data_prep/sft/download_plan.json`. `--hf-cache` is a Hugging Face home with the `datasets` caches that
`load_pools` in `build_pool.py` lists. `--hf-hub` holds the eval sets that the pool is decontaminated
against. `--rl-math` takes the math set behind `math_keep.jsonl` and keeps the pool's math disjoint from it.
Without `--ru-math` (gold math rows) and `--ru-sft`, the Russian math and code cells and part of the
Russian chat cell stay short, and `stats.json` reports the shortfall. `--flagged` is
`{"ids": {reason: [id, ...]}}`, for example from reading v1 by hand.

The math set needs short gold answers as `\boxed{...}`, since the reward can check nothing else. Leave out
proofs, critiques, problems that need a figure, multiple-choice rows and duplicates whose answers
disagree, and keep it disjoint from MATH-500 and the GSM8K test split. `math_keep.jsonl` needs at least
6,454 rows, because distillation draws that many. The Russian math set has the same format and can also go to
`--ru-math`.

Measure every keep on the start export, SFT v5 at step 7000 (`$ORBIT_EXPORTS/sft_v5_step7000`), since a
keep measured on another checkpoint holds other problems. Each prompt gets n = 8 answers scored by the RL
reward, and the keep holds prompts with 1 to 7 correct. Answers cut by the length limit or with an
unclosed thinking block count as wrong. Prompts over 1,024 tokens (`math`, `math_ru`) or 2,048 tokens
(`code`) are dropped, and `data.max_prompt_length=N` after `--` changes the limit. To split a tag over
machines that share the runs root, run once with `--prepare-only`, then on each machine with its own
`--shard-from`/`--shard-to`, then once with `--merge-only`. The header of `passrate.sh` lists every
option. Its OpenMathReasoning filters (`--drop-omr-*`) only see rows labelled `"source": "OMR"`.

The release code set read the `datasets` caches of open-r1/codeforces
(`default/0.0.0/fbe3f6e903ee854eec2e69e9d96d0306cde59baf`), deepmind/code_contests
(`default/0.0.0/802411c3010cb00d1b05bad57ca77365a3c699d6`) and BAAI/TACO (config `ALL`). Only 78 LCB
problems from 2025-02-01 on pass the test caps, so `val.jsonl` has 78 rows. `--validate` runs the
reference solutions under time limits, so on much slower CPUs a few more problems can fail it.

---

# Инструменты данных для дистилляции и RL

Эти инструменты готовят входы рецептов verl: пул промптов и эталонные строки для дистилляции
(`configs/verl/distill_rl_sft_16k.yaml`), эталонные строки и валидационные наборы для RL
(`configs/verl/rl_sft_mix.yaml`). Запускайте их из корня репозитория через `$VERL_PYTHON`, Python
окружения verl; `-m orbit_verl.data.code_data` импортирует репозиторий, установленный в этом окружении.
Шаги с 1 по 4 идут на CPU. Шаг 5 запускает модель, и ему нужен корень прогонов (`--runs-root` или
`$ORBIT_VERL_RUNS`). Три входа вы готовите сами: задачи по математике для `math_keep.jsonl`, задачи по
математике на русском и русские SFT-диалоги.

## Шаги

```
# 1. Пул v1: prompts.jsonl (32 000 строк, если заполнены все ячейки) и stats.json
$VERL_PYTHON scripts/verl/data/build_pool.py --out <v1 dir> --raw-hf $ORBIT_RAW/hf --hf-cache <hf_cache> \
    --hf-hub <hub cache> --tokenizer <Qwen3 tokenizer> [--rl-math <math set>] [--ru-math <ru math set>] \
    [--ru-sft $ORBIT_RAW/russian_sft]
# 2. Пул v2, на котором идут прогоны: prompts.jsonl и fix_log.json
$VERL_PYTHON scripts/verl/data/fix_pool.py --pool <v1 dir>/prompts.jsonl --hf-hub <hub cache> --out <v2 dir> \
    [--flagged <ids.json>]
# 3. Задачи по программированию с тестами: train.jsonl и val.jsonl (команда релиза)
$VERL_PYTHON -m orbit_verl.data.code_data build --out-dir <out> \
    --codeforces <hf_cache>/datasets/open-r1___codeforces \
    --code-contests <hf_cache>/datasets/deepmind___code_contests \
    --taco <hf_cache>/datasets/BAAI___taco \
    --primeintellect <DeepCoder-Preview-Dataset>/primeintellect \
    --lcb <LiveCodeBench code_generation_lite dir> --validate
# 4. Валидационные наборы; все команды есть в docstring, и каждый путь входа должен содержать закреплённую ревизию
$VERL_PYTHON scripts/verl/data/build_val.py chat|lcb|math500|gsm8k|mgsm-ru ...
# 5. Pass rate стартового экспорта; <work dir>/<kind>/results/<tag>/keep.jsonl идёт в эталонные входы
bash scripts/verl/data/passrate/passrate.sh --kind math|code|math_ru --export <HF export> --model-name <name> \
    --tag <tag> --src <jsonl> --python <verl python> --runs-root <dir> [--work-dir <dir>] [--count N] \
    [--shard-from I --shard-to J | --merge-only | --prepare-only | --dry-run] [-- <hydra overrides>]
```

| файл в `--data` скриптов запуска | откуда |
|---|---|
| `prompts.jsonl` | шаг 2 |
| `code_train.jsonl` | шаг 3, его `train.jsonl` под новым именем |
| `code_val.jsonl`, `lcb_val78.jsonl` | шаг 3 (`val.jsonl`) или шаг 4 `lcb`; файлы совпадают |
| `math500.jsonl`, `gsm8k_en_paired250.jsonl`, `mgsm_ru.jsonl`, `chat_probe_ru.jsonl`, `chat_probe_en.jsonl` | шаг 4 |
| `math_keep.jsonl` | шаг 5, `--kind math --src <your math set>` |
| `code_keep.jsonl` | шаг 5, `--kind code --src code_train.jsonl` |
| `ru_math_keep.jsonl` | шаг 5, `--kind math_ru --src <your Russian math set>`; его читает только RL-ветка `mix_ru` |

## Форматы строк

Все файлы в формате jsonl, по одному объекту на строку.

- Строка пула: `id`, `prompt`, `mode` (`think`, `no_think`), `domain` (`math`, `code`, `if`, `chat`,
  `refusal`), `lang` (`en`, `ru`), `source`, `source_ref`.
- Эталонная строка по математике (семейство `math` или `math_ru`): `instruction`, `response` (эталонный
  ответ или решение, которое кончается на `\boxed{}`; с ним сравнивается последний `\boxed{}` ответа
  модели), `source`, `unique_id`.
- Эталонная строка по коду (семейство `code`): `instruction`, `response` (тесты строкой JSON, см.
  `orbit_verl.data.reward.parse_code_ground_truth`), `source`, `unique_id` и поля `code_data`.
- Строка keep: входная строка и поля `n`, `n_correct`, `pass_rate`, `mean_response_len`, `truncated_frac`.
- Валидационная строка: `instruction`, `response`. У чат-проб `response` пустой, и к нему добавлены `lang`,
  `domain` и `source`.
- Русский SFT-диалог: `{"messages": [{"role": ..., "content": ...}, ...]}`, по файлу `<role>.jsonl` на роль
  (см. `data_prep/README.md`).

## Входы и подводные камни

В `--raw-hf` передайте `$ORBIT_RAW/hf` после того, как `data_prep/long_context/download_hf.py` выполнил
свой план по умолчанию и `data_prep/sft/download_plan.json`. В `--hf-cache` нужна папка Hugging Face с
кешами `datasets`, которые перечислены в `load_pools` скрипта `build_pool.py`. В `--hf-hub` лежат
оценочные наборы, пересечения с которыми удаляются из пула. `--rl-math` принимает математический набор для
`math_keep.jsonl`, и математика пула с ним не пересекается. Без `--ru-math` (эталонные строки по
математике) и `--ru-sft` русские ячейки математики и кода и часть русской ячейки чата остаются
недобранными, и `stats.json` показывает нехватку. `--flagged` имеет вид `{"ids": {reason: [id, ...]}}`;
такой список получается, например, при ручном просмотре v1.

В математическом наборе нужны короткие эталонные ответы в виде `\boxed{...}`: другие награда проверить не
может. Уберите доказательства, разборы решений, задачи с рисунком, задачи с вариантами ответа и дубли с
расходящимися ответами. Набор не должен пересекаться с MATH-500 и тестовой частью GSM8K. В
`math_keep.jsonl` нужно не меньше 6454 строк, столько берёт дистилляция. Русский математический набор
имеет тот же формат, и его же можно подать в `--ru-math`.

Каждый keep измеряйте на стартовом экспорте, SFT v5 на шаге 7000 (`$ORBIT_EXPORTS/sft_v5_step7000`): keep с
другого чекпоинта содержит другие задачи. На каждый промпт модель даёт n = 8 ответов, их оценивает награда
RL, и в keep попадают промпты, у которых от 1 до 7 верных ответов. Ответ, обрезанный по длине или с
незакрытым блоком рассуждений, считается неверным. Промпты длиннее 1024 токенов (`math`, `math_ru`) или
2048 токенов (`code`) выбрасываются, а `data.max_prompt_length=N` после `--` меняет лимит. Чтобы разделить
тег между машинами с общим корнем прогонов, один раз запустите команду с `--prepare-only`, затем на каждой
машине со своим `--shard-from`/`--shard-to`, а в конце один раз с `--merge-only`. Все опции перечислены в
шапке `passrate.sh`. Его фильтры OpenMathReasoning (`--drop-omr-*`) видят только строки с
`"source": "OMR"`.

Релизный набор задач по программированию собран из кешей `datasets` для open-r1/codeforces
(`default/0.0.0/fbe3f6e903ee854eec2e69e9d96d0306cde59baf`), deepmind/code_contests
(`default/0.0.0/802411c3010cb00d1b05bad57ca77365a3c699d6`) и BAAI/TACO (конфиг `ALL`). Ограничения на
тесты проходят только 78 задач LCB с 2025-02-01, поэтому в `val.jsonl` 78 строк. `--validate` запускает
эталонные решения с лимитами времени, так что на заметно более медленных CPU его может не пройти ещё
несколько задач.
