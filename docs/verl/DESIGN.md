# Distillation and RL on verl: design

Русская версия ниже.

The contracts and invariants to keep when you change `src/orbit_verl/`. Paths below are relative to it,
and `docs/verl/README.md` says how to run the code.

## Model port

`model/` ports the trainer's `modelling_ouro.py` and `configuration_ouro.py` to transformers 5, and the
file headers list every change. The port imports nothing from `verl` or `orbit_verl` and must keep working
on transformers 4.57. The RLTT engine relies on this forward contract:

- `return_orbit_aux=True, compute_logits=False` runs all `total_ut_steps` loops and returns
  `loop_hidden_states` (one normed `[B, L, H]` per loop) and `loop_gate_logits` (one `[B, L, 1]` per loop).
- With `early_exit_threshold=None` the plain forward returns the final loop's logits, the `fixed_last`
  mode that vLLM implements.
- `flash_attention_2` handles verl's packed batches (`attention_mask=None`, positions restarting per
  sequence). The hub attention kernels drop those inputs and must not run under verl.

`python -m orbit_verl.model.parity_logprobs --phase vllm|hf|compare` fails when the mean |Δ log p| exceeds
0.05 or the greedy agreement is below 0.95. verl casts the temperature to the logits dtype (0.9 becomes
0.8984375 in bf16), and the RLTT forward must divide by the same value (`stock_temperature`), or the KL
does not start at 0.

## RLTT loss

`rltt/` holds the forward, the loss, the FSDP engine and the exit drift alarm, and `plugin.py` registers
them in verl. A forward with `rltt_labels` returns per-loop log-probs `[B, L, T]`, the final-loop log-probs
and the exit pdf. It must all run inside the forward, because FSDP gathers the weights only while the
owning module runs. The per-loop log-probs go through verl's chunked `FusedLinearForPPO`, so no
`[tokens, vocab]` logits are kept for backward.

- The loss is ratio-free REINFORCE, which is correct only with one optimizer step per batch. verl's
  `old_log_probs` are final-loop values, so PPO clipping against the loop mixture would cut the gradient.
- The exit pdf is detached (`orbit.rltt.train_exit_gate: false`). A loss that trains the gate pushes it
  toward loop 0 on every token with a negative advantage, and length and reward follow.
- The rollout KV cache is fp8, so the loss multiplies each token's loss by the capped importance weight
  of `algorithm.rollout_correction`.
- `rltt_last` must give the gradients of `vanilla`, stock GRPO on the final loop. The CPU tests compare
  the RLTT weightings with `tests/verl/rltt_reference.py`, a frozen copy of the reference code.

## Frozen embedding row

`prepare_model` reads the export's `modelling_ouro.py` as text, without importing it, to find the
embedding's `padding_idx`. It checks that row against the weights and records `embedding_padding_idx` in
`config.json` and the lineage in the manifest. The release line freezes no row (`null`, lineage
`release_qwen3`), because its pad 151643 is also the eos and must stay trainable. The launcher refuses
`pad_token_id` and `embedding_padding_idx` in `override_config`: verl overwrites `pad_token_id`, and a
frozen 151643 would leave a model that cannot learn to stop.

## Prompts and agent loop

`data/prompts.py` (templates in `configs/verl/prompts/`) writes the release chat format as text:
`<|endoftext|>`, the user turn with the instruction of the row's kind, and the assistant turn opened with
`<think>\n` or with an empty closed think block. For one user turn this must match both
`data_prep/tokenizer/templates/qwen35_think.jinja` and the SFT v5 tools template. The two differ on system
and tool turns, so no mode may add a system turn; a test checks the modes against a real SFT export.
Prompts are tokenized with `add_special_tokens=False` everywhere. Russian math rows get the Russian
instruction, or the teacher thinks in English. Each dataset row names its agent loop, so one batch mixes
modes.

`data/agent_loop.py` reads its switches from `orbit.env`. With `ORBIT_THINK_BUDGET`, a training think
response still open at the budget gets Qwen3's stop-thinking sentence and `</think>` (Russian for Russian
rows) with response mask 0, and vLLM writes the answer on the same request id. Validation is never
forced. `ORBIT_MASK_LOOP_TAIL` masks the tail of a cut looping response and stays off in the hybrid
recipes, so the GRPO penalty reaches those tokens.

## Reward

`data/reward.py` routes by the `data_source` before `#`: `orbit/math` checks the last `\boxed{}`,
`orbit/code` runs the last ```python block in the sandbox, and `orbit/none` gives 0. Only the text after
the last `</think>` is scored. A cut response gets `truncated_score` (0), and so does a forced think close
unless `forced_score_scale` is set, so RL never rewards an unfinished answer. math_verify needs a main
thread for `signal.alarm`, so MATH checks run in a spawn process pool. Every result has the same keys,
because verl fails a batch whose rows differ in keys. An unknown data source raises.

## Distillation

The per-token advantage is `log p_teacher - log p_student` (k1, clamped to ±10 nats). The student samples
at temperature 1.0, because verl divides its logits by the rollout temperature and the teacher scores at
1.0. The teacher sees the sequence without the leading `<|endoftext|>`, and `distill/teacher.py` shifts
its rows back. Pre-flight checks that the teacher's tokenizer matches ours on every Qwen3 id. Only the
final loop trains. The answers of a mini-batch must divide evenly among the student's GPUs.

## Sandbox and launcher

Every reward worker runs `code_sandbox.self_check()` before its first code reward and refuses code
rewards when a protection is missing; the launcher runs the same check before training on code rows. A
prepared dir (model dir or dataset) writes its manifest last, and the launcher refuses one that does not
fit the config. Multi-node and `--no-prepare` launches never prepare. Before Ray starts, the launcher
removes `RANK`, `TORCHELASTIC_*` and the other launcher variables. The inference package imports
`orbit_verl.launch.env.remove_launcher_env` and the response-info helpers, so their names and behaviour
must stay stable.

---

# Дистилляция и RL на verl: устройство

Контракты и инварианты, которые нужно сохранить при изменении `src/orbit_verl/`. Пути ниже указаны
относительно этой папки, а как запускать код, описано в `docs/verl/README.md`.

## Порт модели

`model/` переносит `modelling_ouro.py` и `configuration_ouro.py` трейнера на transformers 5, все изменения
перечислены в заголовках файлов. Порт ничего не импортирует из `verl` и `orbit_verl` и должен работать и
на transformers 4.57. Движок RLTT опирается на такой контракт forward:

- `return_orbit_aux=True, compute_logits=False` прогоняет все `total_ut_steps` проходов и возвращает
  `loop_hidden_states` (по одному нормированному `[B, L, H]` на проход) и `loop_gate_logits` (по одному
  `[B, L, 1]` на проход).
- При `early_exit_threshold=None` обычный forward возвращает логиты последнего прохода. Это режим
  `fixed_last`, который реализует vLLM.
- `flash_attention_2` обрабатывает упакованные батчи verl (`attention_mask=None`, позиции начинаются
  заново в каждой последовательности). Ядра внимания из хаба отбрасывают эти входы, и под verl их
  запускать нельзя.

`python -m orbit_verl.model.parity_logprobs --phase vllm|hf|compare` падает, если среднее |Δ log p| больше
0,05 или доля совпадений при жадной генерации ниже 0,95. verl приводит температуру к типу логитов (0,9
превращается в 0,8984375 в bf16), и forward RLTT должен делить на то же значение (`stock_temperature`),
иначе KL на старте не равен 0.

## Лосс RLTT

В `rltt/` лежат forward, лосс, FSDP-движок и сигнал дрейфа выхода, а `plugin.py` регистрирует их в verl.
Forward с `rltt_labels` возвращает лог-вероятности по проходам `[B, L, T]`, лог-вероятности последнего
прохода и распределение выхода. Всё это должно считаться внутри forward, потому что FSDP собирает веса
только на время работы модуля, которому они принадлежат. Лог-вероятности по проходам идут через блочный
`FusedLinearForPPO` из verl, поэтому логиты `[tokens, vocab]` для backward не хранятся.

- Лосс устроен как REINFORCE без отношения вероятностей и корректен только при одном шаге оптимизатора
  на батч. `old_log_probs` в verl посчитаны по последнему проходу, поэтому клиппинг PPO относительно
  смеси проходов срезал бы градиент.
- Распределение выхода отсоединено от графа (`orbit.rltt.train_exit_gate: false`). Лосс, который обучает
  гейт, толкает его к проходу 0 на каждом токене с отрицательным преимуществом, а следом за ним идут
  длина ответов и награда.
- KV-кэш роллаутов хранится в fp8, поэтому лосс умножает вклад каждого токена на ограниченный
  сверху вес важности из `algorithm.rollout_correction`.
- `rltt_last` должен давать те же градиенты, что и `vanilla`, штатный GRPO на последнем проходе.
  CPU-тесты сравнивают взвешивания RLTT с `tests/verl/rltt_reference.py`, замороженной копией эталонного
  кода.

## Замороженная строка эмбеддинга

`prepare_model` читает `modelling_ouro.py` экспорта как текст, не импортируя его, и находит `padding_idx`
эмбеддинга. Он сверяет эту строку с весами и записывает `embedding_padding_idx` в `config.json`, а lineage
в манифест. В релизной линии замороженной строки нет (`null`, lineage `release_qwen3`), потому что её pad
151643 служит ещё и eos и должен обучаться. Лаунчер не принимает `pad_token_id` и `embedding_padding_idx`
в `override_config`: verl перезаписывает `pad_token_id`, а модель с замороженным 151643 не научилась бы
останавливаться.

## Промпты и агентный цикл

`data/prompts.py` (шаблоны в `configs/verl/prompts/`) записывает релизный чат-формат текстом:
`<|endoftext|>`, реплика пользователя с инструкцией для вида строки и реплика ассистента, которая
открывается `<think>\n` или пустым закрытым блоком рассуждения. Для одной реплики пользователя текст
должен совпадать и с `data_prep/tokenizer/templates/qwen35_think.jinja`, и с шаблоном с инструментами из
SFT v5. Они расходятся на системных репликах и репликах инструментов, поэтому ни один режим не должен
добавлять системную реплику; тест сверяет режимы с настоящим экспортом SFT. Промпты везде
токенизируются с `add_special_tokens=False`. Строки русской математики получают русскую инструкцию,
иначе учитель рассуждает по-английски. Каждая строка датасета сама задаёт свой agent loop, поэтому в
одном батче смешиваются режимы.

`data/agent_loop.py` читает свои переключатели из `orbit.env`. С `ORBIT_THINK_BUDGET` обучающий
think-ответ, у которого к концу бюджета рассуждение ещё не закрыто, получает стоп-фразу Qwen3 и
`</think>` (для русских строк русскую фразу) с маской ответа 0, и vLLM дописывает ответ с тем же request
id. На валидации рассуждение не закрывается принудительно. `ORBIT_MASK_LOOP_TAIL` маскирует хвост
обрезанного зациклившегося ответа и выключен в гибридных рецептах, чтобы штраф GRPO доходил до этих
токенов.

## Награда

`data/reward.py` выбирает проверку по части `data_source` до `#`: `orbit/math` сверяет последний
`\boxed{}`, `orbit/code` запускает последний блок ```python в песочнице, а `orbit/none` даёт 0.
Оценивается только текст после последнего `</think>`. Обрезанный ответ получает `truncated_score` (0), и
ответ с принудительно закрытым рассуждением тоже, если не задан `forced_score_scale`, поэтому RL никогда
не награждает незаконченный ответ. math_verify нужен главный поток для `signal.alarm`, поэтому проверки
MATH идут в пуле процессов spawn. У каждого результата один и тот же набор ключей, потому что verl роняет
батч, строки которого различаются ключами. Неизвестный источник данных вызывает исключение.

## Дистилляция

Преимущество на токен равно `log p_teacher - log p_student` (k1, обрезанный до ±10 нат). Студент
сэмплирует при температуре 1,0, потому что verl делит его логиты на температуру роллаута, а учитель
оценивает при 1,0. Учитель видит последовательность без начального `<|endoftext|>`, и
`distill/teacher.py` сдвигает его строки обратно. Предварительная проверка сверяет токенизатор учителя с
нашим на каждом id Qwen3. Обучается только последний проход. Ответы мини-батча должны делиться поровну
между GPU студента.

## Песочница и лаунчер

Каждый воркер награды вызывает `code_sandbox.self_check()` перед первой наградой за код и отказывается
считать награды за код, если какой-то защиты нет; лаунчер выполняет ту же проверку перед обучением на
строках кода. Подготовленная папка (папка модели или датасет) пишет манифест последним, и лаунчер
отказывается от папки, которая не подходит конфигу. Многоузловые запуски и запуски с `--no-prepare`
ничего не готовят. Перед стартом Ray лаунчер удаляет `RANK`, `TORCHELASTIC_*` и другие переменные
лаунчеров. Пакет инференса импортирует `orbit_verl.launch.env.remove_launcher_env` и хелперы response
info, поэтому их имена и поведение должны оставаться прежними.
