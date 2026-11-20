# Tests of the OrbitLM trainer

Русская версия ниже.

Run everything from the repository root, in the trainer's environment with the test packages and the
repository installed:

```bash
python -m pip install -r requirements/train.txt -r requirements/dev.txt
python -m pip install -e . --no-deps
```

## CPU tests

```bash
python -m pytest -q tests/train
```

Besides `test_trainer_cpu.py` and `test_release_configs.py`, this collects the `unit` group of
`test_document_masking.py` and the tests of the in-training evaluation. Those tests and the
evaluation's one-GPU run are described in `eval/in_training/README.md`.

## GPU checks of document masking

Document masking stops attention at every EOS (`<|endoftext|>`, id 151643). In
`test_document_masking.py`, groups `a` and `b` compare masked attention with dense attention and with
each document run alone, group `c` checks the backward pass and that no gradient crosses a document,
and group `d` only measures the step time (peak memory 36 GB at batch 4). `check_dense_unchanged.py` checks
that with masking off the outputs equal those of a reference package, for example `src/orbit` of an
earlier commit. The two-GPU run passes when both FSDP steps finish and the log shows
`document_masking=1 boundary_token_id=151643`.

Groups a-d and `check_dense_unchanged.py` need a Hopper GPU (H100 or H200) for the FA3 kernel and are
skipped with a message without one. On a Hopper GPU, a kernel that does not load (a wrong
`ORBIT_ATTN_KERNEL_PATH`, a build for another torch) is a failure. They also need an HF export of the
1.4B model and a streams root with MDS streams of 16,385-token samples: `nemotron_cc`, plus `ru_wiki`
and `opc` for group d. Without a local build, the kernel is downloaded from the HF Hub.

```bash
export ORBIT_TEST_HF_DIR=/path/to/hf_export             # --hf-dir
export ORBIT_TEST_STREAMS_ROOT=/path/to/streams         # --streams-root
export ORBIT_ATTN_KERNEL_PATH=/path/to/vllm-flash-attn3 # --kernel-path; the trainer reads it too

python tests/train/test_document_masking.py --out /tmp/document_masking_report.json
python tests/train/test_document_masking.py --groups unit,a,b
python tests/train/check_dense_unchanged.py --reference-package <package folder>

# End-to-end on two GPUs. OUT is any empty folder for the run's outputs.
torchrun --standalone --nproc_per_node 2 -m orbit.train \
    --config tests/train/configs/e2e_document_masking_2gpu.yaml \
    --save-dir $OUT --tensorboard-root $OUT/tb --dump-nonfinite-batch $OUT/nan_dumps \
    --init-from-hf $ORBIT_TEST_HF_DIR --stage1-root $ORBIT_TEST_STREAMS_ROOT \
    --tokenizer-config $ORBIT_TOKENIZERS/tokenizer_qwen3_orbit/tokenizer_config.json
```

`test_document_masking.py` prints a JSON report with a `pass` entry per group. Both scripts exit with
0 when everything that ran passed or was skipped for lack of a Hopper GPU, and with 1 when a check
failed or the kernel did not load.

---

# Тесты трейнера OrbitLM

Все команды выполняются из корня репозитория, в окружении трейнера с тестовыми пакетами и
установленным репозиторием:

```bash
python -m pip install -r requirements/train.txt -r requirements/dev.txt
python -m pip install -e . --no-deps
```

## CPU-тесты

```bash
python -m pytest -q tests/train
```

Кроме `test_trainer_cpu.py` и `test_release_configs.py`, сюда входят группа `unit` из
`test_document_masking.py` и тесты оценки во время обучения. Эти тесты и прогон оценки на одном GPU
описаны в `eval/in_training/README.md`.

## GPU-проверки маскирования документов

Маскирование документов обрывает внимание на каждом EOS (`<|endoftext|>`, id 151643). В
`test_document_masking.py` группы `a` и `b` сравнивают маскированное внимание с плотным и с прогоном
каждого документа по отдельности, группа `c` проверяет обратный проход и то, что градиент не переходит
из документа в документ, а группа `d` только замеряет время шага (пиковая память 36 GB при batch 4).
`check_dense_unchanged.py` проверяет, что при выключенном маскировании выходы совпадают с эталонным
пакетом, например с `src/orbit` из более раннего коммита. Прогон на двух GPU считается успешным, если
оба шага FSDP завершились, а в логе есть `document_masking=1 boundary_token_id=151643`.

Группам a-d и `check_dense_unchanged.py` нужен GPU Hopper (H100 или H200) для ядра FA3, без него они
пропускаются с сообщением. На Hopper ядро, которое не загрузилось (неверный `ORBIT_ATTN_KERNEL_PATH`,
сборка под другую версию torch), считается провалом. Ещё им нужны HF-экспорт модели 1.4B и корень
потоков с MDS-потоками из сэмплов по 16 385 токенов: `nemotron_cc`, а для группы d также `ru_wiki` и
`opc`. Без локальной сборки ядро скачивается с HF Hub.

```bash
export ORBIT_TEST_HF_DIR=/path/to/hf_export             # --hf-dir
export ORBIT_TEST_STREAMS_ROOT=/path/to/streams         # --streams-root
export ORBIT_ATTN_KERNEL_PATH=/path/to/vllm-flash-attn3 # --kernel-path; его читает и трейнер

python tests/train/test_document_masking.py --out /tmp/document_masking_report.json
python tests/train/test_document_masking.py --groups unit,a,b
python tests/train/check_dense_unchanged.py --reference-package <package folder>

# End-to-end on two GPUs. OUT is any empty folder for the run's outputs.
torchrun --standalone --nproc_per_node 2 -m orbit.train \
    --config tests/train/configs/e2e_document_masking_2gpu.yaml \
    --save-dir $OUT --tensorboard-root $OUT/tb --dump-nonfinite-batch $OUT/nan_dumps \
    --init-from-hf $ORBIT_TEST_HF_DIR --stage1-root $ORBIT_TEST_STREAMS_ROOT \
    --tokenizer-config $ORBIT_TOKENIZERS/tokenizer_qwen3_orbit/tokenizer_config.json
```

`test_document_masking.py` печатает JSON-отчёт с полем `pass` для каждой группы. Оба скрипта
завершаются с кодом 0, если всё запущенное прошло или было пропущено из-за отсутствия Hopper, и с
кодом 1, если проверка провалилась или ядро не загрузилось.
