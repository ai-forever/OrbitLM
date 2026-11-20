# orbit: the OrbitLM trainer

Русская версия ниже.

`train.py` runs stages 1 to 5, and the config picks the stage. `export.py` turns a checkpoint into
the HF folder that the next stage, `src/orbit_verl` and inference load.

## Stages

Context extension sets `seq_len` 16384 or 32768, `rope_theta` (without it the model keeps the value
of its `init_from_hf` export) and `document_masking: true`, which stops attention at every
`<|endoftext|>`. SFT adds `labels_premasked: true`, so only the `loss_mask` tokens of the SFT shards
are trained. Document masking needs `attn_impl: kernels-community/vllm-flash-attn3`, and the trainer
stops with an error when the tokenizer config's `eos_token` is `<|im_end|>`.

## Running

`scripts/train/run_local.sh <config.yaml> [GPUs] [options]` runs on one machine, by default on every
GPU. The options override the config (`python -m orbit.train --help`):

```bash
scripts/train/run_local.sh configs/train/smoke/sft.yaml 1 --max-steps 2    # one GPU, two steps
```

A scheduler other than torchrun must start every process with `RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
`LOCAL_WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT`. Every node needs the data, exports, tokenizers
and evaluation environment at the same paths. With `$ORBIT_RUNS` on local disks, the in-training
evaluation waits until it times out and the export cannot merge a step. Exact resume and the same
data order need the same GPU count.

## Paths

Configs name locations as `${NAME}`. An unset variable stops the run with its name, and repository
paths are relative to the config file. `data_prep/README.md` gives the layout of `$ORBIT_DATA`. A run
folder is `$ORBIT_RUNS/<run_name>` (`orbit_stage1_4k`, `orbit_stage2_4k`, `orbit_stage3_16k`,
`orbit_stage4_32k`, `orbit_sft_v5`) with `checkpoints/step_<N>/`, `latest.txt` and
`step_trace.jsonl`, next to `tensorboard/` and `nan_dumps/`. Without `ORBIT_ATTN_KERNEL_PATH`, every
rank downloads the FA3 kernel from the HF Hub at start.

## Config keys that matter

| Keys | Meaning |
|---|---|
| `total_ut_steps`, `beta` | the number of loops (4) and the weight of the exit entropy in the loss (`losses.py`) |
| `init_from_hf` | the architecture comes from the export's `config.json` |
| `tokens_per_step`, `batch_size` | `grad_accum` = tokens_per_step / (batch_size x seq_len x GPUs), rounded up |
| `token_budget` | gives `max_steps` = `token_budget / tokens_per_step` unless `max_steps` is set |
| `lr_scheduler`, `lr_cosine_steps` | `warmup_cosine` or `constant`; `auto` ends the cosine at `max_steps` |
| `stage1_streams`, `stage1_dataset_proportions` | name: folder (relative to `stage1_root` or absolute), mixed by relative weight |
| `clearml_enabled`, `clearml_project`, `clearml_env` | ClearML is off by default; `clearml_env` is a dotenv file with the credentials |
| `detect_nan`, `dump_nonfinite_batch`, `max_nonfinite_grad_skips` | name the first parameter with a non-finite gradient and save that batch; such steps are skipped, and more than the limit stop the run |

A key of an earlier trainer (`train.REMOVED_CONFIG_KEYS`) at any value but the one this trainer
implements, or a value outside an option's choices, stops the run with an error.

## Export

`python -m orbit.export` takes a run folder (its latest step), a step folder such as
`$ORBIT_RUNS/orbit_sft_v5/checkpoints/step_00007000`, or a `latest.txt`. The architecture,
`rope_theta` and `max_position_embeddings` come from the checkpoint. `--attn-impl` (default `eager`)
sets the attention implementation written to `config.json`. `export_info.json` records the run name
as typed and does not follow symlinks, so export through a symlink to publish a run under another
name.

## Evaluation and tests

Stages 4 and 5 pause every `eval_every` steps for the evaluation in `eval/in_training/README.md`.
`tests/train/README.md` lists the CPU tests (with `requirements/train.txt` and
`requirements/dev.txt`) and the GPU checks.

---

# orbit: трейнер OrbitLM

`train.py` проводит стадии с 1 по 5, а стадию выбирает конфиг. `export.py` превращает чекпоинт в
папку HF, которую загружают следующая стадия, `src/orbit_verl` и инференс.

## Стадии

Расширение контекста задаёт `seq_len` 16 384 или 32 768, `rope_theta` (без этого ключа модель
сохраняет значение из своего экспорта `init_from_hf`) и `document_masking: true`, при котором
внимание обрывается на каждом `<|endoftext|>`. SFT добавляет `labels_premasked: true`, и обучаются
только токены из `loss_mask` шардов SFT. Маскированию документов нужен
`attn_impl: kernels-community/vllm-flash-attn3`, а если в конфиге токенизатора `eos_token` равен
`<|im_end|>`, трейнер останавливается с ошибкой.

## Запуск

`scripts/train/run_local.sh <config.yaml> [GPUs] [options]` запускает обучение на одной машине, по
умолчанию на всех GPU. Опции переопределяют конфиг (`python -m orbit.train --help`):

```bash
scripts/train/run_local.sh configs/train/smoke/sft.yaml 1 --max-steps 2    # одна GPU, два шага
```

Любой другой планировщик должен задавать каждому процессу те же переменные, что и torchrun: `RANK`,
`LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR` и `MASTER_PORT`. Данные, экспорты,
токенизаторы и окружение оценки должны лежать на всех узлах по одним и тем же путям. Если
`$ORBIT_RUNS` находится на локальных дисках, оценка во время обучения ждёт до тайм-аута, а экспорт не
может собрать шаг. Точное возобновление и тот же порядок данных требуют того же числа GPU.

## Пути

Конфиги задают пути через `${NAME}`. Незаданная переменная останавливает прогон с её именем в
сообщении, а пути внутри репозитория отсчитываются от папки конфига. Структура `$ORBIT_DATA` описана
в `data_prep/README.md`. Папка прогона лежит в `$ORBIT_RUNS/<run_name>` (`orbit_stage1_4k`,
`orbit_stage2_4k`, `orbit_stage3_16k`, `orbit_stage4_32k`, `orbit_sft_v5`) и содержит
`checkpoints/step_<N>/`, `latest.txt` и `step_trace.jsonl`, а рядом лежат `tensorboard/` и
`nan_dumps/`. Без `ORBIT_ATTN_KERNEL_PATH` каждый ранг при старте скачивает ядро FA3 с HF Hub.

## Важные ключи конфига

| Ключи | Смысл |
|---|---|
| `total_ut_steps`, `beta` | число циклов (4) и вес энтропии распределения выхода в функции потерь (`losses.py`) |
| `init_from_hf` | архитектура берётся из `config.json` экспорта |
| `tokens_per_step`, `batch_size` | `grad_accum` = tokens_per_step / (batch_size x seq_len x GPUs) с округлением вверх |
| `token_budget` | даёт `max_steps` = `token_budget / tokens_per_step`, если `max_steps` не задан |
| `lr_scheduler`, `lr_cosine_steps` | `warmup_cosine` или `constant`; при `auto` косинус заканчивается на `max_steps` |
| `stage1_streams`, `stage1_dataset_proportions` | имя потока: папка (относительно `stage1_root` или абсолютный путь), потоки смешиваются по относительным весам |
| `clearml_enabled`, `clearml_project`, `clearml_env` | ClearML по умолчанию выключен; `clearml_env` указывает на dotenv-файл с учётными данными |
| `detect_nan`, `dump_nonfinite_batch`, `max_nonfinite_grad_skips` | называют первый параметр с неконечным градиентом и сохраняют этот батч; такие шаги пропускаются, а когда их становится больше лимита, прогон останавливается |

Прогон останавливается с ошибкой, если ключ прежних версий трейнера (`train.REMOVED_CONFIG_KEYS`)
имеет значение, которого этот трейнер не реализует, или значение опции не входит в её допустимый
набор.

## Экспорт

`python -m orbit.export` принимает папку прогона (берёт её последний шаг), папку шага, например
`$ORBIT_RUNS/orbit_sft_v5/checkpoints/step_00007000`, или `latest.txt`. Архитектура, `rope_theta` и
`max_position_embeddings` берутся из чекпоинта. `--attn-impl` (по умолчанию `eager`) задаёт
реализацию внимания, которую экспорт записывает в `config.json`. `export_info.json` хранит имя
прогона в том виде, в каком его указали в команде, и не раскрывает симлинки. Чтобы опубликовать
прогон под другим именем, экспортируйте его через симлинк с этим именем.

## Оценка и тесты

Стадии 4 и 5 каждые `eval_every` шагов прерываются на оценку, описанную в
`eval/in_training/README.md`. `tests/train/README.md` перечисляет тесты на CPU (с пакетами
`requirements/train.txt` и `requirements/dev.txt`) и проверки на GPU.
