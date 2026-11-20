# Distillation and RL on verl

Русская версия ниже.

The release model is RL on verl 0.9.1 with responses of up to 16K tokens, started from the SFT v5
export at step 7000. The code also has on-policy distillation from Qwen3-8B, and it works well too:
RL can start from its export instead of the SFT export. Commands run from the repo root in the verl
env. The code design is in `docs/verl/DESIGN.md`, the data tools in `scripts/verl/data/README.md`.

| recipe | config | training |
|---|---|---|
| distillation | `configs/verl/distill_rl_sft_16k.yaml` | on-policy distillation from Qwen3-8B plus GRPO on gold rows, thinking budget 14,336 |
| RL | `configs/verl/rl_sft_mix.yaml` | GRPO on gold math and code; the release model uses the config's default loss `rltt_exit`, which weights every loop by the exit gate |

## The release model

```bash
RECIPE_TAG=lr1e6_fp8tis_rlttexit bash scripts/verl/run_rl.sh mix \
    --export $ORBIT_EXPORTS/sft_v5_step7000 --model-name sft_v5_step7000 --data <data dir> -- \
    actor_rollout_ref.actor.policy_loss.loss_mode=rltt_exit trainer.total_training_steps=150
$VERL_PYTHON scripts/verl/export_hf_fp32.py --ckpt $ORBIT_VERL_RUNS/runs/rl_mix__lr1e6_fp8tis_rlttexit/ckpt/global_step_90/actor \
    --base-export $ORBIT_EXPORTS/sft_v5_step7000 --out $ORBIT_EXPORTS/orbitlm_1.8b
```

The RL run uses the `mix` arm with the default loss `rltt_exit` and a cap of 150 steps; the release
model is its step 90, chosen by the validation scores. On 80 GB GPUs add `--gpu-mem 0.7` before `--`.

## Setup

```bash
bash requirements/verl/build_env.sh /path/to/envs/orbit-verl [flash_attn-2.8.3-...whl]
```

Keep vLLM at 0.26, the last release with the Ouro model. flash-attn has no official wheel for torch 2.11
with CUDA 13; `build_env.sh` says how to build one. Set `ORBIT_VERL_RUNS` to the runs root (no default,
`--runs-root` replaces it), which holds `models/`, `data/`, `runs/<run name>/` and `cache/<host>/`. Set
`VERL_PYTHON` to the env's python. Without internet access, export `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`.

## Inputs

The launcher prepares the model dir from the export itself. It refuses an export whose code passes
`padding_idx=151643` to the embedding while that row is not zero in the weights. Prepare such an export
by hand once:

```bash
$VERL_PYTHON -m orbit_verl.model.prepare_model --export <export> \
    --out-dir $ORBIT_VERL_RUNS/models/<name> --preset qwen3 --embedding-padding-idx none --check
```

The data are jsonl files in one dir, passed as `--data`:

| file | role | rows used |
|---|---|---|
| `prompts.jsonl` | distillation prompt pool | all |
| `math_keep.jsonl` | gold math | RL: all in `math`, 3,800 in the mixed arms; distillation: 6,454 |
| `code_train.jsonl`, `code_val.jsonl` | distillation: gold code, validation | 3,300 / all |
| `code_keep.jsonl` | RL: gold code | all |
| `ru_math_keep.jsonl` | RL `mix_ru`: Russian gold math | all |
| `math500.jsonl` | validation, think and no-think | 500 (no-think in RL: 200) |
| `mgsm_ru.jsonl` | validation | 250 |
| `lcb_val78.jsonl`, `gsm8k_en_paired250.jsonl` | RL validation | 78 / 250 |
| `chat_probe_ru.jsonl`, `chat_probe_en.jsonl` | RL validation, no gold answer | 109 / 100 |

## Distillation

```bash
bash scripts/verl/run_distill.sh --export <export> --teacher <Qwen3-8B dir> --data <data dir> \
    [--arm opd] [--print | --dry-run | --prepare-only]
```

One 8-GPU node (6 GPUs for the student, 2 for the teacher) runs 200 steps of 132 prompts x 4 answers,
with validation and a checkpoint every 10 steps. `--arm opd` gives the gold rows only the teacher's signal.
In both run scripts `--print` shows the launcher arguments and `--dry-run` runs every check without
starting. Choose the checkpoint by the validation numbers.

To start RL from a distillation step, export it to fp32 first (here step 200 of a run started from
the SFT v5 export):

```bash
$VERL_PYTHON scripts/verl/export_hf_fp32.py --ckpt $ORBIT_VERL_RUNS/runs/distill_rl_16k/ckpt/global_step_200/actor \
    --base-export $ORBIT_EXPORTS/sft_v5_step7000 --out $ORBIT_EXPORTS/distill_v5_step200
```

Then pass `--export $ORBIT_EXPORTS/distill_v5_step200 --model-name distill_v5_step200` to
`run_rl.sh`, and the same export as `--base-export` when exporting the RL step.

## RL

```bash
bash scripts/verl/run_rl.sh ARM --export <export> --data <data dir> [--gpu-mem 0.7] [--gate] \
    [--print | --dry-run | --prepare-only]
```

`ARM` is `math`, `code`, `mix` (3,800 math rows and all code rows) or `mix_ru` (`mix` plus `RU_COUNT` rows
of `RU_KEEP`, by default all of `ru_math_keep.jsonl`). Each arm runs on one 8-GPU node with 64 prompts x 8
answers per step and lr 1e-6 after 10 warmup steps, and saves every 5 steps and validates every 10. Use
`--gpu-mem 0.7` on 80 GB GPUs. The run name is `rl_<ARM>__lr1e6_fp8tis`. Overrides after `--` need
`RECIPE_TAG=<tag>`, which replaces `lr1e6_fp8tis`.

verl silently drops code prompts when the sandbox fails, so check a new kind of machine first:

1. `bash scripts/verl/run_rl.sh smoke <same options>` runs 2 steps of 16 prompts.
2. `SMOKE_STEPS=3 bash scripts/verl/run_rl.sh smoke <same options>` resumes it for step 3.
3. `$VERL_PYTHON scripts/verl/sandbox_gate.py $ORBIT_VERL_RUNS/runs/rl_smoke__lr1e6_fp8tis` checks both
   launch logs for scored code rows.
4. `--gate` on a full arm refuses to start it until the gate passes.

Run the watcher next to every arm and take the checkpoint before its first alarm.
`total_training_steps: 200` is only a cap.

## Several nodes

Run the command once with `--prepare-only` on a machine that sees the runs root (no GPU needed). Then run
it with `--no-prepare` once per node, never once per GPU, with `RANK`, `WORLD_SIZE` and `MASTER_ADDR`
(node 0) set:

```bash
torchrun --nnodes N --node_rank R --nproc_per_node 1 --master_addr HOST --master_port 29500 \
    --no-python bash scripts/verl/run_rl.sh mix --no-prepare <same options>
```

Every node needs the runs root on a shared filesystem and the export (for distillation also the teacher)
at the same path. Put `--no-prepare` before any `--`. Rank 0 logs to `runs/<run name>/launch.log`, the
other ranks to `launch.node<rank>.log` next to it.

## Monitoring

```bash
$VERL_PYTHON scripts/verl/watch.py $ORBIT_VERL_RUNS/runs/<run> --follow 600               # distillation
$VERL_PYTHON scripts/verl/watch.py $ORBIT_VERL_RUNS/runs/<run> --rl --val-dir <data dir> --follow 600
```

The watcher needs no GPU, and its docstring defines the rules. Exit code 0 means nothing fired, 3 a
notify rule, 4 a hard stop. It never stops a run itself; with `--follow --rl` it exits at the first hard
stop, so that a wrapper can.

## Export

```bash
$VERL_PYTHON scripts/verl/export_hf_fp32.py --ckpt $ORBIT_VERL_RUNS/runs/<run>/ckpt/global_step_N/actor \
    --base-export <start export> --out <new export dir> [--check-against <verl bf16 merge>/model.safetensors]
```

verl's own merger casts to bf16. This tool keeps fp32 and copies every other file from the start export.
`--check-against` compares the result, rounded to bf16, with verl's merge bit for bit.

## Smoke test

```bash
bash scripts/verl/smoke.sh rl      --export <export> --data <dir> [-- OVERRIDE ...]
bash scripts/verl/smoke.sh distill --export <export> --data <dir> --teacher <Qwen3-8B dir> [-- OVERRIDE ...]
```

A smoke runs 2 steps of 4 prompts with responses of at most 1,024 tokens, `rl` on 1 GPU and `distill`
on 2. `--data` needs `math_keep.jsonl`, `code_keep.jsonl`, `math500.jsonl`, `lcb_val78.jsonl`, and for
`distill` also `prompts.jsonl` and `mgsm_ru.jsonl`. A healthy smoke exits with 0 and shows `kl_loss` 0 at
step 1 (RL), a teacher loss in `actor/distillation/loss` (distillation), non-zero `code_passed` in some code
rollouts and `global_step_2` under `runs/smoke_<kind>/ckpt`. Delete `runs/smoke_<kind>` before a rerun. At
1,024 tokens most RL answers are cut and score 0. For non-zero advantages, raise `data.max_response_length`
together with the three token budgets named in `scripts/verl/smoke.sh`.

## Launcher and tests

The run scripts start `scripts/verl/launch.py`. A relaunch under the same run name resumes from the last
checkpoint and refuses changed recipe keys. Launcher exit codes: 0 success, 1 training error,
2 configuration or pre-flight error, 3 no Ray cluster.

```bash
ORBIT_TEST_SFT_EXPORT=$ORBIT_EXPORTS/sft_v5_step7000 CUDA_VISIBLE_DEVICES= \
    $VERL_PYTHON -m pytest -q tests/verl
```

The tests run on CPU. Without `ORBIT_TEST_SFT_EXPORT` the tokenizer tests skip. The sandbox tests need
Linux with user namespaces, Landlock and seccomp.

---

# Дистилляция и RL на verl

Релизная модель обучена RL на verl 0.9.1 с ответами длиной до 16K токенов, старт с экспорта SFT v5
на шаге 7000. В коде есть и on-policy дистилляция из Qwen3-8B, и она тоже неплохо работает: RL может
стартовать с её экспорта вместо экспорта SFT. Команды запускаются из корня репозитория в окружении
verl. Устройство кода описано в `docs/verl/DESIGN.md`, инструменты данных в
`scripts/verl/data/README.md`.

| рецепт | конфиг | что обучает |
|---|---|---|
| дистилляция | `configs/verl/distill_rl_sft_16k.yaml` | on-policy дистилляция из Qwen3-8B плюс GRPO на gold-строках, бюджет рассуждения 14 336 |
| RL | `configs/verl/rl_sft_mix.yaml` | GRPO на gold-строках математики и кода; релизная модель обучена с лоссом конфига по умолчанию `rltt_exit`, он взвешивает все проходы гейтом выхода |

## Релизная модель

```bash
RECIPE_TAG=lr1e6_fp8tis_rlttexit bash scripts/verl/run_rl.sh mix \
    --export $ORBIT_EXPORTS/sft_v5_step7000 --model-name sft_v5_step7000 --data <data dir> -- \
    actor_rollout_ref.actor.policy_loss.loss_mode=rltt_exit trainer.total_training_steps=150
$VERL_PYTHON scripts/verl/export_hf_fp32.py --ckpt $ORBIT_VERL_RUNS/runs/rl_mix__lr1e6_fp8tis_rlttexit/ckpt/global_step_90/actor \
    --base-export $ORBIT_EXPORTS/sft_v5_step7000 --out $ORBIT_EXPORTS/orbitlm_1.8b
```

RL идёт на плече `mix` с лоссом по умолчанию `rltt_exit` и не больше 150 шагов; релизная модель
взята с его шага 90 по валидационным метрикам. На GPU с 80 ГБ добавьте `--gpu-mem 0.7` перед `--`.

## Установка

```bash
bash requirements/verl/build_env.sh /path/to/envs/orbit-verl [flash_attn-2.8.3-...whl]
```

vLLM держите на версии 0.26: это последний релиз с моделью Ouro. Официальной сборки flash-attn для
torch 2.11 с CUDA 13 нет; как её собрать, написано в `build_env.sh`. В `ORBIT_VERL_RUNS` укажите корень
прогонов (значения по умолчанию нет, его заменяет `--runs-root`), в нём лежат `models/`, `data/`,
`runs/<run name>/` и `cache/<host>/`. В `VERL_PYTHON` укажите python окружения. Без доступа в интернет
задайте `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`.

## Входы

Папку модели из экспорта лаунчер готовит сам. Он не принимает экспорт, чей код передаёт эмбеддингу
`padding_idx=151643`, если эта строка в весах ненулевая. Такой экспорт один раз подготовьте вручную:

```bash
$VERL_PYTHON -m orbit_verl.model.prepare_model --export <export> \
    --out-dir $ORBIT_VERL_RUNS/models/<name> --preset qwen3 --embedding-padding-idx none --check
```

Данные лежат файлами jsonl в одной папке, её передают в `--data`:

| файл | роль | сколько строк |
|---|---|---|
| `prompts.jsonl` | пул промптов дистилляции | все |
| `math_keep.jsonl` | gold-математика | RL: все в `math`, 3800 в смешанных плечах; дистилляция: 6454 |
| `code_train.jsonl`, `code_val.jsonl` | дистилляция: gold-код, валидация | 3300 / все |
| `code_keep.jsonl` | RL: gold-код | все |
| `ru_math_keep.jsonl` | RL `mix_ru`: русская gold-математика | все |
| `math500.jsonl` | валидация, think и no-think | 500 (no-think в RL: 200) |
| `mgsm_ru.jsonl` | валидация | 250 |
| `lcb_val78.jsonl`, `gsm8k_en_paired250.jsonl` | валидация RL | 78 / 250 |
| `chat_probe_ru.jsonl`, `chat_probe_en.jsonl` | валидация RL, без gold-ответа | 109 / 100 |

## Дистилляция

```bash
bash scripts/verl/run_distill.sh --export <export> --teacher <Qwen3-8B dir> --data <data dir> \
    [--arm opd] [--print | --dry-run | --prepare-only]
```

Один узел с 8 GPU (6 под студента, 2 под учителя) делает 200 шагов по 132 промпта x 4 ответа и каждые
10 шагов проводит валидацию и сохраняет чекпоинт. С `--arm opd` gold-строки получают только сигнал
учителя. В обоих скриптах запуска `--print` показывает аргументы лаунчера, а `--dry-run` выполняет все
проверки и ничего не запускает. Чекпоинт выбирайте по валидационным метрикам.

Чтобы стартовать RL с шага дистилляции, сначала экспортируйте его в fp32 (здесь шаг 200 прогона,
стартовавшего с экспорта SFT v5):

```bash
$VERL_PYTHON scripts/verl/export_hf_fp32.py --ckpt $ORBIT_VERL_RUNS/runs/distill_rl_16k/ckpt/global_step_200/actor \
    --base-export $ORBIT_EXPORTS/sft_v5_step7000 --out $ORBIT_EXPORTS/distill_v5_step200
```

Затем передайте в `run_rl.sh` `--export $ORBIT_EXPORTS/distill_v5_step200 --model-name
distill_v5_step200`, а при экспорте шага RL укажите тот же экспорт в `--base-export`.

## RL

```bash
bash scripts/verl/run_rl.sh ARM --export <export> --data <data dir> [--gpu-mem 0.7] [--gate] \
    [--print | --dry-run | --prepare-only]
```

`ARM` принимает значения `math`, `code`, `mix` (3800 строк математики и все строки кода) или `mix_ru`
(`mix` плюс `RU_COUNT` строк из `RU_KEEP`, по умолчанию весь `ru_math_keep.jsonl`). Каждое плечо идёт
на одном узле с 8 GPU: 64 промпта x 8 ответов на шаг, lr 1e-6 после 10 шагов разогрева, чекпоинт каждые
5 шагов, валидация каждые 10. На GPU с 80 ГБ задайте `--gpu-mem 0.7`. Прогон называется
`rl_<ARM>__lr1e6_fp8tis`. Переопределениям после `--` нужен `RECIPE_TAG=<tag>`, он заменяет
`lr1e6_fp8tis`.

Если песочница не работает, verl молча выбрасывает промпты с кодом, поэтому машину нового типа сначала
проверьте:

1. `bash scripts/verl/run_rl.sh smoke <same options>` делает 2 шага по 16 промптов.
2. `SMOKE_STEPS=3 bash scripts/verl/run_rl.sh smoke <same options>` продолжает его до шага 3.
3. `$VERL_PYTHON scripts/verl/sandbox_gate.py $ORBIT_VERL_RUNS/runs/rl_smoke__lr1e6_fp8tis` проверяет по
   логам обоих запусков, что строки кода получили оценку.
4. С `--gate` полное плечо не стартует, пока проверка не пройдена.

Рядом с каждым плечом держите вотчер и берите чекпоинт до его первой тревоги.
`total_training_steps: 200` задаёт только верхнюю границу.

## Несколько узлов

Один раз запустите команду с `--prepare-only` на машине, которая видит корень прогонов (GPU не нужна).
Потом запустите её с `--no-prepare` по одному разу на каждом узле, а не на каждой GPU, с заданными
`RANK`, `WORLD_SIZE` и `MASTER_ADDR` (адрес узла 0):

```bash
torchrun --nnodes N --node_rank R --nproc_per_node 1 --master_addr HOST --master_port 29500 \
    --no-python bash scripts/verl/run_rl.sh mix --no-prepare <same options>
```

Корень прогонов должен лежать на общей для всех узлов файловой системе, а экспорт (для дистилляции ещё и
учитель) на каждом узле по одному и тому же пути. `--no-prepare` ставьте до `--`. Ранг 0 пишет лог в
`runs/<run name>/launch.log`, остальные ранги в `launch.node<rank>.log` рядом с ним.

## Мониторинг

```bash
$VERL_PYTHON scripts/verl/watch.py $ORBIT_VERL_RUNS/runs/<run> --follow 600               # дистилляция
$VERL_PYTHON scripts/verl/watch.py $ORBIT_VERL_RUNS/runs/<run> --rl --val-dir <data dir> --follow 600
```

Вотчеру не нужна GPU, а правила описаны в его docstring. Код выхода 0 значит, что ничего не сработало,
3 значит, что сработало правило уровня notify, 4 означает жёсткую остановку. Сам вотчер прогон не
останавливает, но с `--follow --rl` завершается на первой жёсткой остановке, чтобы это сделала обёртка.

## Экспорт

```bash
$VERL_PYTHON scripts/verl/export_hf_fp32.py --ckpt $ORBIT_VERL_RUNS/runs/<run>/ckpt/global_step_N/actor \
    --base-export <start export> --out <new export dir> [--check-against <verl bf16 merge>/model.safetensors]
```

Штатное слияние verl приводит веса к bf16. Этот инструмент сохраняет fp32 и копирует все остальные файлы
из стартового экспорта. `--check-against` побитово сравнивает результат, округлённый до bf16, со
слиянием verl.

## Смоук-тест

```bash
bash scripts/verl/smoke.sh rl      --export <export> --data <dir> [-- OVERRIDE ...]
bash scripts/verl/smoke.sh distill --export <export> --data <dir> --teacher <Qwen3-8B dir> [-- OVERRIDE ...]
```

Смоук делает 2 шага по 4 промпта с ответами не длиннее 1024 токенов, `rl` на 1 GPU, `distill` на 2. В
`--data` нужны `math_keep.jsonl`, `code_keep.jsonl`, `math500.jsonl`, `lcb_val78.jsonl`, а для `distill`
ещё `prompts.jsonl` и `mgsm_ru.jsonl`. Успешный смоук завершается с кодом 0: `kl_loss` на шаге 1 равен 0
(RL), лосс учителя виден в `actor/distillation/loss` (дистилляция), в части роллаутов кода `code_passed`
ненулевой, а в `runs/smoke_<kind>/ckpt` лежит `global_step_2`. Перед повторным запуском удалите
`runs/smoke_<kind>`. При 1024 токенах большинство ответов RL обрезаются и получают 0. Чтобы получить
ненулевые преимущества, увеличьте `data.max_response_length` вместе с тремя бюджетами токенов, которые
перечислены в `scripts/verl/smoke.sh`.

## Лаунчер и тесты

Скрипты запуска вызывают `scripts/verl/launch.py`. Повторный запуск под тем же именем прогона продолжает
с последнего чекпоинта и отказывается стартовать, если изменились ключи рецепта. Коды выхода лаунчера:
0 успех, 1 ошибка обучения, 2 ошибка конфига или предварительных проверок, 3 кластер Ray не собрался.

```bash
ORBIT_TEST_SFT_EXPORT=$ORBIT_EXPORTS/sft_v5_step7000 CUDA_VISIBLE_DEVICES= \
    $VERL_PYTHON -m pytest -q tests/verl
```

Тесты идут на CPU. Без `ORBIT_TEST_SFT_EXPORT` тесты токенизатора пропускаются. Тестам песочницы нужен
Linux с user namespaces, Landlock и seccomp.
