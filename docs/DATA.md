# Данные

Все обучающие данные собирают скрипты, вручную готовятся только несколько входов. Сборка запускается
из корня репозитория в окружении данных (`requirements/data_prep.txt` и установленный репозиторий).
Исходные данные лежат в `$ORBIT_RAW`, токенизированные в `$ORBIT_DATA`.

## Порядок сборки

1. Токенизаторы: `data_prep/tokenizer/prepare_tokenizers.py`.
2. Стадии 1 и 2 (4K): `data_prep/pretrain/tokenize_4k.py`, по одному вызову на поток.
3. Стадия 3 (16K): сборщики `data_prep/long_context/build_*.py`, затем `run_tasks.py --stage 16k`
   и `mix.py --stage 16k`.
4. Стадия 4 (32K): `run_tasks.py --stage 32k` и `mix.py --stage 32k`.
5. Стадия 5 (SFT): `data_prep/sft/prepare_tools.py`, затем
   `data_prep/sft/build.py --version data_prep/sft/versions/v5_smoltalk2.yaml --build` и та же
   команда с `--resolve`.
6. Стадия 6 (RL) и дистилляция: инструменты из `scripts/verl/data/` собирают для них папку `--data`.

Полные команды есть в `data_prep/README.md` и `scripts/verl/data/README.md`.

## Входы, которые готовятся вручную

Подойдут любые данные описанного ниже вида. Конфиги сохраняют доли, которые эти потоки имели в
релизе. Чтобы обучаться без какого-то из них, его можно удалить из конфига (для 16K и 32K ещё и из файла
стадии); трейнер распределит его долю между остальными потоками.

| Вход | Формат | Где используется |
|---|---|---|
| Русский текст в пяти папках: `web`, `books`, `science`, `wikipedia`, `wikiquote` | `.jsonl`, `.jsonl.gz`, `.jsonl.zst`, `.jsonl.zstd` или `.parquet` с полем `text`; `tokenize_4k.py` пишет каждую папку в `$ORBIT_DATA/pretrain_4k/ru/` | стадии с 1 по 4 |
| Русские чаты: `$ORBIT_RAW/russian_sft/<kind>.jsonl` для `science`, `math`, `code`, `dialog`, `tasks`, `creative`, `education`, `qa_reasoning`, `summarization` | один чат на строку, `{"messages": [...]}`: необязательная системная реплика, затем реплики пользователя и ассистента по очереди, первая от пользователя, последняя от ассистента | стадии с 3 по 5 и дистилляция |
| Математические задачи с короткими ответами | jsonl-строки с полями `instruction`, `response` в виде `\boxed{...}` и `source`; `scripts/verl/data/passrate/passrate.sh --kind math` оставляет задачи, которые модель решает лишь иногда, в `math_keep.jsonl` (не меньше 6454 строк) | стадия 6 и дистилляция |
| Русские математические задачи | тот же формат; `passrate.sh --kind math_ru` пишет `ru_math_keep.jsonl` | плечо RL `mix_ru` |

Отбор по pass rate лучше всего делать на экспорте после SFT.