# Inference

Русская версия ниже.

These scripts generate with OrbitLM HF exports and render prompts like the SFT evaluation.
`generate_vllm.py` is the main path. It runs in the verl environment (vLLM 0.26) or with a newer vLLM
in any environment where the repository is installed (`pip install -e . --no-deps`): vLLM removed its
Ouro model after 0.26, and the `orbit_vllm` plugin brings it back (checked with vLLM 0.34).
`generate_hf.py`, for debugging and parity checks, runs one prompt at a time with transformers 4.56 to
5.23, for example in the training environment. Exports made by older versions of this repository carry
model code that needs the `kernels` package and transformers below 5.2; copying
`src/orbit/modelling_ouro.py` into such an export lifts both limits. Run the commands from the repo root.

## Commands

```bash
python inference/generate_vllm.py $MODEL --prompts prompts.jsonl --output answers.jsonl \
    --speculative --budget-forcing --think-budget 14336 --answer-budget 2048
```

A greedy parity check, with a bf16 KV cache and no speculation:

```bash
python inference/generate_vllm.py $MODEL --prompts inference/examples/math_smoke.jsonl \
    --kv-cache-dtype auto --temperature 0 --max-new-tokens 1024 --output vllm_greedy.jsonl
python inference/generate_hf.py $MODEL --prompts inference/examples/math_smoke.jsonl \
    --temperature 0 --max-new-tokens 1024 --output hf_greedy.jsonl
```

The outputs match up to the first near-tie of the two best bf16 logits. A gap of more than a few
rounding steps there means one engine computes the model differently.

`$MODEL` is an HF export. The think and no_think modes need an SFT chat export, which the exporter
writes with `--chat-export` (see `src/orbit/README.md`). A pretraining or context-extension export runs
with `--mode raw`. Sampling defaults to temperature 0.6, top_p 0.95 and top_k 20, and `--temperature 0`
is greedy. `--loops N` runs N of the model's loops (default: all, and two is the useful minimum) and
answers from the last one. `python inference/make_draft.py $MODEL DRAFT_DIR` prepares the one-loop
draft of `--speculative` for reuse with `--draft-dir DRAFT_DIR`.

## Prompts and output

`--prompts` takes a `.jsonl` file with `prompt` or `messages` (a chat, in the export's chat template)
and optional `id` and `lang` (`en` or `ru`), or a `.txt` file with one prompt per line.
`--prompt TEXT` can be repeated instead. `--mode think` (default) ends the prompt with an open thinking
block and `no_think` with an empty one. Prompts in both modes start with `<|endoftext|>` unless
`--no-lead-eos` is set. A bad prompt fails before the engine starts, with exit code 2.

Each output record holds `id`, `lang`, `mode`, `loops`, `prompt`, `prompt_tokens`, `thinking`, `answer`,
`think` (`closed`, `unclosed` or `none`), `forced_close`, `continued_after_budget`, `finish_reason`,
`completion_tokens` and `text`, the whole response with special tokens. Without `--output` the records
go to standard output and all logs go to standard error, so `> answers.jsonl` gives clean JSONL.

## Budget forcing

With `--budget-forcing` (think mode only), a response still thinking after `--think-budget` tokens
(default 14,336) gets Qwen3's stop-thinking sentence and `</think>`, and a second call writes the answer
in at most `--answer-budget` tokens (default 2,048). A response cut off while answering gets up to
`--answer-budget` more tokens. A Russian prompt gets the Russian sentence from `STOP_THINKING_TEXTS` in
`src/orbit_verl/data/response_info.py`. The language is `--lang`, else the prompt's `lang` field, else
detected from the letters of the last user message, so write the instruction of a Russian prompt in
Russian too.

## Speed and memory

Against a bf16 KV cache without speculation (vLLM 0.26, one H200), the default FP8 KV cache holds 2.1
times as many tokens. `--speculative` (a one-loop greedy draft, 4 tokens per step) decodes 1.7 to 2.2
times faster when greedy and 1.9 times faster at temperature 0.9, with 1.5 times the KV capacity. It
keeps the full model's output distribution.

## Environment fixes

The scripts print each fix they apply, and a variable you set yourself wins. They remove the variables
of a distributed launcher (`RANK`, `WORLD_SIZE`, `MASTER_*`, `TORCHELASTIC_*` and the like), with which
vLLM hangs waiting for peers. A `flash_attn` that fails to import is hidden from vLLM, and the engine
then starts with `VLLM_WORKER_MULTIPROC_METHOD=spawn`. `ORBIT_HIDE_FLASH_ATTN=1` forces the hide, `0`
turns the check off. vLLM also gets `VLLM_USE_DEEP_GEMM=0`, `VLLM_USE_AOT_COMPILE=0` (with torch 2.10
and later the AOT cache mis-measures free memory from the second run on) and, without `nvcc`,
`VLLM_USE_FLASHINFER_SAMPLER=0`. `generate_hf.py` turns off the cuDNN attention kernel, which fails on
long contexts in some environments.

## Tests

```bash
python -m pytest -q tests/inference -p no:cacheprovider
```

The tests run on CPU. Run them in both environments, because `tests/inference/test_hf_generate.py`
skips itself in the verl one.

---

# Инференс

Скрипты генерируют на HF-экспортах OrbitLM и рендерят промпты так же, как SFT-оценка. Основной путь:
`generate_vllm.py`. Он работает в окружении verl (vLLM 0.26) или с более новым vLLM в любом окружении,
куда установлен репозиторий (`pip install -e . --no-deps`). Модель Ouro убрали из vLLM после 0.26, и её
возвращает плагин `orbit_vllm` (проверено на vLLM 0.34). `generate_hf.py` нужен для отладки и сверки и
прогоняет промпты по одному с transformers от 4.56 до 5.23, например в окружении обучения. В экспортах
из старых версий репозитория лежит код модели, которому нужны пакет `kernels` и transformers ниже 5.2.
Если скопировать в такой экспорт `src/orbit/modelling_ouro.py`, оба ограничения снимаются. Команды
запускаются из корня репозитория.

## Команды

```bash
python inference/generate_vllm.py $MODEL --prompts prompts.jsonl --output answers.jsonl \
    --speculative --budget-forcing --think-budget 14336 --answer-budget 2048
```

Жадная сверка, с KV-кэшем в bf16 и без спекуляции:

```bash
python inference/generate_vllm.py $MODEL --prompts inference/examples/math_smoke.jsonl \
    --kv-cache-dtype auto --temperature 0 --max-new-tokens 1024 --output vllm_greedy.jsonl
python inference/generate_hf.py $MODEL --prompts inference/examples/math_smoke.jsonl \
    --temperature 0 --max-new-tokens 1024 --output hf_greedy.jsonl
```

Выходы совпадают до первой почти ничьей двух лучших логитов в bf16. Если разрыв там больше
нескольких шагов округления, один из движков считает модель иначе.

`$MODEL` указывает на HF-экспорт. Режимам think и no_think нужен чат-экспорт SFT, его пишет скрипт
экспорта с флагом `--chat-export` (см. `src/orbit/README.md`). Экспорт претрейна или расширения
контекста запускается с `--mode raw`. По умолчанию сэмплирование идёт с температурой 0,6, top_p 0,95
и top_k 20, а `--temperature 0` даёт жадную генерацию. `--loops N` запускает N проходов модели (по
умолчанию все, разумный минимум два) и берёт ответ с последнего.
`python inference/make_draft.py $MODEL DRAFT_DIR` заранее собирает однопроходный черновик для
`--speculative`, его передают через `--draft-dir DRAFT_DIR`.

## Промпты и выход

`--prompts` принимает `.jsonl` с полем `prompt` или `messages` (чат, его рендерит чат-шаблон
экспорта) и необязательными `id` и `lang` (`en` или `ru`), либо `.txt` с одним промптом на строку.
Вместо файла можно повторять `--prompt TEXT`. `--mode think` (по умолчанию) заканчивает промпт
открытым блоком рассуждения, `no_think` пустым. В обоих режимах промпт начинается с `<|endoftext|>`,
если не задан `--no-lead-eos`. Плохой промпт даёт ошибку ещё до старта движка, с кодом выхода 2.

Каждая запись выхода содержит `id`, `lang`, `mode`, `loops`, `prompt`, `prompt_tokens`, `thinking`,
`answer`, `think` (`closed`, `unclosed` или `none`), `forced_close`, `continued_after_budget`,
`finish_reason`, `completion_tokens` и `text`, то есть всю генерацию со спецтокенами. Без `--output`
записи идут в стандартный вывод, а все логи в stderr, поэтому `> answers.jsonl` даёт чистый JSONL.

## Budget forcing

С `--budget-forcing` (только в режиме think) генерация, которая после `--think-budget` токенов (по
умолчанию 14 336) всё ещё рассуждает, получает фразу Qwen3 о завершении рассуждения и `</think>`, а
второй вызов пишет ответ не длиннее `--answer-budget` токенов (по умолчанию 2048). Генерация,
оборванная посреди ответа, получает ещё до `--answer-budget` токенов. Русский промпт получает русскую
фразу из `STOP_THINKING_TEXTS` в `src/orbit_verl/data/response_info.py`. Язык берётся из `--lang`,
иначе из поля `lang` промпта, иначе определяется по буквам последнего сообщения пользователя, поэтому
инструкцию в русском промпте тоже пишите по-русски.

## Скорость и память

По сравнению с KV-кэшем в bf16 без спекуляции (vLLM 0.26, одна H200) FP8 KV-кэш, который включён по
умолчанию, вмещает в 2,1 раза больше токенов. `--speculative` (жадный однопроходный черновик, 4 токена
за шаг) декодирует от 1,7 до 2,2 раза быстрее при жадной генерации и в 1,9 раза быстрее при
температуре 0,9, а KV-кэш с ним вмещает в 1,5 раза больше. Распределение выходов полной модели при
этом не меняется.

## Исправления окружения

Скрипты печатают каждое исправление, а переменная, заданная вами, важнее значения по умолчанию.
Переменные распределённого лаунчера (`RANK`, `WORLD_SIZE`, `MASTER_*`, `TORCHELASTIC_*` и подобные)
удаляются, потому что с ними vLLM зависает в ожидании пиров. `flash_attn`, который не импортируется,
скрывается от vLLM, и движок тогда стартует с `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
`ORBIT_HIDE_FLASH_ATTN=1` скрывает его принудительно, `0` отключает проверку. Для vLLM ставятся также
`VLLM_USE_DEEP_GEMM=0`, `VLLM_USE_AOT_COMPILE=0` (с torch 2.10 и новее AOT-кэш со второго запуска
неверно измеряет свободную память), а без `nvcc` ещё и `VLLM_USE_FLASHINFER_SAMPLER=0`.
`generate_hf.py` отключает ядро внимания cuDNN, которое в некоторых окружениях падает на длинных
контекстах.

## Тесты

```bash
python -m pytest -q tests/inference -p no:cacheprovider
```

Тесты идут на CPU. Запускайте их в обоих окружениях, потому что в окружении verl
`tests/inference/test_hf_generate.py` пропускает себя.
