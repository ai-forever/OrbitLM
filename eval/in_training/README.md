# In-training evaluation

Русская версия ниже.

Every `eval_every` optimizer steps the trainer pauses, evaluates an export of the model with lm-eval
0.4.13 on vLLM on all GPUs, logs the scores to TensorBoard (and ClearML when it is enabled) and
resumes. Stages 4 and 5 use it to pick checkpoints. Commands run from the repository root in the
training environment, with the repository installed (`python -m pip install -e . --no-deps`).

| Suite | Used by |
|---|---|
| `suites/base.yaml` | stage 4, and the default: the paper's few-shot base protocol with RULER 4K-32K |
| `suites/sft.yaml` | stage 5: 0-shot chat benchmarks in think and no_think mode, the gates and four base benchmarks |
| `suites/smoke.yaml`, `suites/sft_smoke.yaml` | quick GPU checks of the two suites above |

Each suite file describes its protocol.

## Setup

Evaluation processes run with `eval_python`, the interpreter of a separate vLLM environment pinned by
`requirements/eval.txt`. The configs of stages 4 and 5 read `ORBIT_EVAL_PYTHON`, `ORBIT_LM_EVAL` and
`ORBIT_EVAL_CACHE` and refuse to load without them. `ORBIT_TOKENIZERS` is described in
`src/orbit/README.md`. Once:

```bash
export ORBIT_EVAL_PYTHON=/path/to/eval/env/bin/python  # the interpreter of the environment with vLLM
export ORBIT_LM_EVAL=/path/to/lm_eval_0.4.13           # a new, empty folder
export ORBIT_EVAL_CACHE=/path/to/eval_cache            # the offline cache, filled once with network access

# 1. The evaluation environment, IFEval's langdetect and immutabledict included (skip in an
#    environment that already has these packages).
$ORBIT_EVAL_PYTHON -m pip install -r requirements/eval.txt

# 2. lm-eval 0.4.13 and the RULER generator's wonderwords, into a folder of their own.
$ORBIT_EVAL_PYTHON -m pip install --no-deps --no-compile --target $ORBIT_LM_EVAL \
    lm_eval==0.4.13 wonderwords==3.0.1

# 3. The offline cache: every task runs once with lm-eval's dummy model, online.
for suite in base sft; do
  python -m orbit.train_eval prefetch --suite eval/in_training/suites/$suite.yaml \
      --eval-python $ORBIT_EVAL_PYTHON --harness $ORBIT_LM_EVAL --cache-dir $ORBIT_EVAL_CACHE \
      --tokenizer $ORBIT_TOKENIZERS/tokenizer_qwen3_orbit
done
```

Install lm-eval only into a new, empty folder. Over an older install, `pip install --target` keeps the
old files, and the startup check accepts only an exact pip install of 0.4.13. `prefetch --only mbpp`
adds one benchmark to an existing cache.

If the evaluation environment cannot be changed and lacks IFEval's two packages, which only `sft.yaml`
needs, install them into a folder of their own. Pass it as `eval_extra_pythonpath` to the trainer and
as `--extra-pythonpath` to `prefetch`:

```bash
$ORBIT_EVAL_PYTHON -m pip install --no-deps --require-hashes --no-build-isolation \
    --target /path/to/eval_extras -r eval/in_training/requirements-sft-eval.txt
```

`python -m orbit.train --help` lists the `eval_*` keys. A relative `eval_suite` resolves against the
config file's folder. Every node checks the evaluation environment at startup, so a broken setup stops
the run before its first step.

## Known traps

- RULER fails when the system's libstdc++ is older than the evaluation environment's and lacks
  `CXXABI_1.3.15`, because nltk then cannot import `sqlite3`. The callback preloads
  `<eval env>/lib/libstdc++.so.6` unless a newer copy is on `LD_LIBRARY_PATH` or in the system
  folders. The `libstdc++:` line of the startup log and the `libstdc++=` field of `preflight ok` show
  the choice. The same setup can pass on one machine and fail on another.
- `TMPDIR` must be shorter than about 70 characters, because CRUXEval, HumanEval and MBPP keep a Unix
  socket there.
- CUDA libraries on the trainer's `LD_LIBRARY_PATH` or `LD_PRELOAD` that do not match the evaluation
  environment's torch fail the startup matmul.

## Chat suites

A suite with a `chat:` section renders its prompts with the chat template of `eval_tokenizer_source`.
The startup check refuses a template whose prompts end in other ids:

| Mode | The prompt ends with | Ids |
|---|---|---|
| think | `<\|im_start\|>assistant\n<think>\n` | `151644 77091 198 151667 198` |
| no_think | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | `151644 77091 198 151667 271 151668 271` |
| no_prefill | `<\|im_start\|>assistant\n` | `151644 77091 198` |

## Scores, files and failures

Scores go to `eval_<benchmark>` with series `raw` (`eval_ruler` has one series per length), chat
accuracy to `eval_chat_<bench>` and gates to `eval_gate_<gate>`. Each step's work is kept in
`run_dir/eval/step_XXXXXXXX/`, with `summary.json` at the end. The export `run_dir/eval/model` is
deleted after the pause unless `eval_keep_exports` is set.

A failed evaluation does not stop training: a benchmark with a failed unit is not reported, and
`eval_failures` counts it. A step interrupted by SIGTERM is evaluated again after the resume, and one
cut off by two crashes is given up. The exception is a GPU whose memory the evaluation processes did
not free: all ranks then stop after the pause, with the step's checkpoint on disk.

## Tests

CPU tests, with stand-ins for lm-eval and vLLM, run in the trainer environment with
`requirements/dev.txt`. After a deliberate change to `base.yaml` or `smoke.yaml`, rewrite the golden
snapshot with `python tests/train/base_golden.py --write`.

```bash
python -m pytest -q tests/train/test_train_eval.py tests/train/test_train_eval_chat.py
```

The one-GPU end-to-end check trains two steps and evaluates after each with `smoke.yaml`. It needs a
base HF export of the 1.4B model and a streams root with the MDS streams `nemotron_cc` and `ru_wiki` of
16,385-token samples:

```bash
OUT=/path/to/empty/folder                      # the run's outputs
HF_DIR=$ORBIT_EXPORTS/stage3_16k_step47684    # any base export of the 1.4B model
STREAMS_ROOT=/path/to/streams                  # holds nemotron_cc/ and ru_wiki/
torchrun --standalone --nproc_per_node 1 -m orbit.train \
    --config tests/train/configs/e2e_in_training_eval_1gpu.yaml \
    --save-dir $OUT --tensorboard-root $OUT/tb --init-from-hf $HF_DIR --stage1-root $STREAMS_ROOT \
    --tokenizer-config $ORBIT_TOKENIZERS/tokenizer_qwen3_orbit/tokenizer_config.json \
    --eval-python $ORBIT_EVAL_PYTHON --eval-harness-path $ORBIT_LM_EVAL --eval-cache-dir $ORBIT_EVAL_CACHE
```

It passes when the log shows two `in_training_eval rank=0 step=N scores ...` lines with
`failed_units=0` and both `summary.json` files say `"status": "ok"`.

---

# Оценка во время обучения

Каждые `eval_every` шагов оптимизатора трейнер встаёт на паузу, оценивает экспорт модели через
lm-eval 0.4.13 на vLLM на всех GPU, пишет результаты в TensorBoard (и в ClearML, если он включён) и
продолжает обучение. По этим оценкам выбирают чекпоинты стадий 4 и 5. Команды выполняются из корня
репозитория в окружении обучения, где установлен сам репозиторий (`python -m pip install -e . --no-deps`).

| Набор | Где используется |
|---|---|
| `suites/base.yaml` | стадия 4 (это набор по умолчанию): базовый few-shot протокол статьи с RULER от 4K до 32K |
| `suites/sft.yaml` | стадия 5: 0-shot бенчмарки в формате чата в режимах think и no_think, гейты и четыре базовых бенчмарка |
| `suites/smoke.yaml`, `suites/sft_smoke.yaml` | быстрые проверки двух наборов выше на GPU |

Протокол каждого набора описан в его файле.

## Подготовка

Процессы оценки запускаются интерпретатором `eval_python` из отдельного окружения с vLLM, состав
которого закреплён в `requirements/eval.txt`. Конфиги стадий 4 и 5 читают переменные
`ORBIT_EVAL_PYTHON`, `ORBIT_LM_EVAL` и `ORBIT_EVAL_CACHE` и без них не загружаются. Про
`ORBIT_TOKENIZERS` см. `src/orbit/README.md`. Один раз выполните:

```bash
export ORBIT_EVAL_PYTHON=/path/to/eval/env/bin/python  # интерпретатор окружения с vLLM
export ORBIT_LM_EVAL=/path/to/lm_eval_0.4.13           # новая пустая папка
export ORBIT_EVAL_CACHE=/path/to/eval_cache            # офлайн-кэш, заполняется один раз при доступе к сети

# 1. The evaluation environment, IFEval's langdetect and immutabledict included (skip in an
#    environment that already has these packages).
$ORBIT_EVAL_PYTHON -m pip install -r requirements/eval.txt

# 2. lm-eval 0.4.13 and the RULER generator's wonderwords, into a folder of their own.
$ORBIT_EVAL_PYTHON -m pip install --no-deps --no-compile --target $ORBIT_LM_EVAL \
    lm_eval==0.4.13 wonderwords==3.0.1

# 3. The offline cache: every task runs once with lm-eval's dummy model, online.
for suite in base sft; do
  python -m orbit.train_eval prefetch --suite eval/in_training/suites/$suite.yaml \
      --eval-python $ORBIT_EVAL_PYTHON --harness $ORBIT_LM_EVAL --cache-dir $ORBIT_EVAL_CACHE \
      --tokenizer $ORBIT_TOKENIZERS/tokenizer_qwen3_orbit
done
```

Ставьте lm-eval только в новую пустую папку. Поверх старой установки `pip install --target` оставляет
старые файлы, а стартовая проверка принимает только lm-eval 0.4.13 ровно в том виде, в каком его
установил pip. `prefetch --only mbpp` добавляет один бенчмарк в готовый кэш.

Если окружение оценки менять нельзя и в нём нет двух пакетов IFEval (они нужны только `sft.yaml`),
установите их в отдельную папку. Трейнеру её передают ключом `eval_extra_pythonpath`, а `prefetch`
флагом `--extra-pythonpath`:

```bash
$ORBIT_EVAL_PYTHON -m pip install --no-deps --require-hashes --no-build-isolation \
    --target /path/to/eval_extras -r eval/in_training/requirements-sft-eval.txt
```

Ключи `eval_*` перечисляет `python -m orbit.train --help`. Относительный путь в `eval_suite`
отсчитывается от папки конфига. При старте каждый узел проверяет окружение оценки, так что
неисправная настройка останавливает прогон ещё до первого шага.

## Известные ловушки

- RULER падает, если системная копия libstdc++ более старая, чем в окружении оценки, и в ней нет
  `CXXABI_1.3.15`: тогда nltk не может импортировать `sqlite3`. Колбэк заранее подгружает
  `<eval env>/lib/libstdc++.so.6`, если в `LD_LIBRARY_PATH` или в системных папках нет более новой
  копии. Что выбрано, показывают строка `libstdc++:` стартового лога и поле `libstdc++=` строки
  `preflight ok`. Одна и та же настройка может работать на одной машине и падать на другой.
- `TMPDIR` должен быть короче примерно 70 символов, потому что CRUXEval, HumanEval и MBPP держат там
  Unix-сокет.
- Библиотеки CUDA из `LD_LIBRARY_PATH` или `LD_PRELOAD` трейнера, которые не подходят к torch
  окружения оценки, проваливают проверочное умножение матриц при старте.

## Наборы в формате чата

Набор с секцией `chat:` рендерит промпты шаблоном чата из `eval_tokenizer_source`. Стартовая проверка
отвергает шаблон, у которого промпты заканчиваются другими id:

| Режим | Конец промпта | Id |
|---|---|---|
| think | `<\|im_start\|>assistant\n<think>\n` | `151644 77091 198 151667 198` |
| no_think | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | `151644 77091 198 151667 271 151668 271` |
| no_prefill | `<\|im_start\|>assistant\n` | `151644 77091 198` |

## Результаты, файлы и сбои

Оценки пишутся в `eval_<benchmark>` с серией `raw` (у `eval_ruler` по серии на каждую длину), точность
в формате чата в `eval_chat_<bench>`, гейты в `eval_gate_<gate>`. Файлы оценки каждого шага лежат в
`run_dir/eval/step_XXXXXXXX/`, в конце там появляется `summary.json`. Экспорт `run_dir/eval/model`
удаляется после паузы, если не задан `eval_keep_exports`.

Сбой оценки не останавливает обучение: бенчмарк, у которого упала хотя бы одна единица работы, не
выводится, а `eval_failures` учитывает сбой. Шаг, прерванный SIGTERM, после возобновления оценивается
заново; если оценку шага дважды прервало падение, от неё отказываются. Исключение одно: если процессы
оценки не освободили память GPU, все ранги останавливаются после паузы, а чекпоинт шага остаётся на
диске.

## Тесты

CPU-тесты с заглушками вместо lm-eval и vLLM запускаются в окружении трейнера с
`requirements/dev.txt`. После намеренного изменения `base.yaml` или `smoke.yaml` перезапишите
эталонный снимок командой `python tests/train/base_golden.py --write`.

```bash
python -m pytest -q tests/train/test_train_eval.py tests/train/test_train_eval_chat.py
```

Сквозная проверка на одном GPU обучает два шага и после каждого оценивает модель набором
`smoke.yaml`. Ей нужны базовый HF-экспорт модели 1.4B и корень потоков с MDS-потоками `nemotron_cc` и
`ru_wiki` из сэмплов по 16 385 токенов:

```bash
OUT=/path/to/empty/folder                      # выходные файлы прогона
HF_DIR=$ORBIT_EXPORTS/stage3_16k_step47684    # любой базовый экспорт модели 1.4B
STREAMS_ROOT=/path/to/streams                  # содержит nemotron_cc/ и ru_wiki/
torchrun --standalone --nproc_per_node 1 -m orbit.train \
    --config tests/train/configs/e2e_in_training_eval_1gpu.yaml \
    --save-dir $OUT --tensorboard-root $OUT/tb --init-from-hf $HF_DIR --stage1-root $STREAMS_ROOT \
    --tokenizer-config $ORBIT_TOKENIZERS/tokenizer_qwen3_orbit/tokenizer_config.json \
    --eval-python $ORBIT_EVAL_PYTHON --eval-harness-path $ORBIT_LM_EVAL --eval-cache-dir $ORBIT_EVAL_CACHE
```

Проверка пройдена, если в логе есть две строки `in_training_eval rank=0 step=N scores ...` с
`failed_units=0`, а в обоих файлах `summary.json` стоит `"status": "ok"`.
