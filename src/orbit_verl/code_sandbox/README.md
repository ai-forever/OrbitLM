# orbit_verl.code_sandbox

Русская версия ниже.

The executor of the RL code reward. It runs model-written programs and their tests on the training
machine without a container, using only the Python standard library. It needs the repository installed
(`python -m pip install -e . --no-deps`).

```python
from orbit_verl.code_sandbox import execute, ExecutorPool, Limits, self_check, extract_code, reward_info

assert all(self_check().values())        # once per worker: do all protections hold on this machine?
program = extract_code(response_text)    # the last ```python block of the answer
r = execute(program, [{"input": "1 2\n", "output": "3\n"}], kind="stdin")
r = execute(program, cases, kind="stdin", judge_imports=True, float_tolerance=1e-6)        # LCB style
r = execute(program, "assert add(1, 2) == 3\n", kind="assert")                            # MBPP style
r = execute(program, check_src, kind="assert", entry_point="find_zero", test_context=prompt)  # HumanEval
r = execute(program, tests, kind="assert", judge_imports=True, limits=Limits().for_cases(n))  # LeetCode
reward = 1.0 if r["passed"] else 0.0
metrics = reward_info(r)                 # a fixed set of float metrics for the logs
with ExecutorPool() as pool:             # the default size is the machine's CPU quota
    results = pool.map([{"program": p, "tests": t, "kind": "stdin"} for p, t in batch])
```

`r["passed"]` is True only when every test passed with a valid verdict. `r["reason"]` is `passed`,
`policy_violation` (the program never ran), `syntax_error`, `wrong_answer`, `test_failed`,
`runtime_error`, `timeout`, `memory_limit`, `output_limit`, `early_exit` (`sys.exit` before the tests
ended), `non_plain_result` (a value that could fake a comparison), `no_verdict`, `verdict_tampered`,
`bad_tests` (no cases, or no assert ran) or `sandbox_error`, which never counts as a pass. `detail` and
`stdout_tail` show what went wrong. `reward_info(None)` gives the same metric keys as zeros, because
verl needs them for every response of a batch.

## What it protects against

The model under training learns anything that raises its reward, so a program must not read the
expected answers, fake a verdict, reach the network, touch other processes or files, or see the
trainer's secrets. A protection that fails to set up makes the run a `sandbox_error`.

1. Each run starts under `unshare --user --map-current-user --pid` with an empty environment, in a
   directory on local disk that is deleted afterwards. Landlock allows reading only the system and
   Python library directories, which shuts out home and data directories, `/proc`, `/tmp`, `/dev/shm`,
   GPU devices, most of `/etc`, and TCP from Landlock ABI 4 on. Seccomp denies sockets, ptrace, mounts,
   namespaces, bpf, io_uring and similar calls. Rlimits cap CPU time, memory, output, processes and
   open files.
2. The static policy (`policy.py`, `rules.py`) rejects denied imports, attributes and builtins (`eval`,
   `exec`, ...) and introspection before the run, and wrapped builtins repeat the checks inside it.
3. In assert mode the tests see the real builtins and modules even when the program redefines them.
   Every value the program returns is checked, so an object that always compares equal fails.

The verdict is one line with a per-run nonce on a separate pipe, and printed output or exit codes never
count as one. Pass the HumanEval prompt as `test_context`, or a program can redefine a helper that the
tests call (`poly` in HumanEval/32).

## Judging and limits

In stdin mode each case runs in a fresh sandbox, and an output matches when its whitespace-separated
tokens equal the expected ones. Pass `float_tolerance=1e-6` for AtCoder and LCB data, because many
tasks accept that error and LCB's reference outputs print more digits than a correct program does. For
LCB-style data pass `judge_imports=True` in both modes: it adds the imports that LeetCode and
LiveCodeBench give every program (`List`, `Counter`, `inf`, ...) and `sys.setrecursionlimit(50000)`.

`Limits()` gives a run 6 CPU seconds (`time_s`, LiveCodeBench's budget per case), `2 * time_s + 2`
seconds of wall clock, 1024 MB of address space, 16 MB of output, 16 processes and threads, 64 open
files and a read-only work dir (`writable_work_dir=True` opens it). An assert run checks all its cases
in one sandbox, so give it `Limits().for_cases(n, cap_s=30)`: `time_s` per case, 30 seconds at most. On a
machine slower than the one LCB graded on, under 1% of correct LCB programs exceed the CPU limit and a
few need more than 1 GB. They lose their reward.

## Kernel requirements and the self-check

The protections need Linux on x86_64 with user namespaces, Landlock and seccomp. Check every machine that computes
rewards:

```bash
python -c "from orbit_verl.code_sandbox import self_check; print(self_check())"
```

Every value must be True: `ran` (the sandbox started and returned a verdict), `outside_blocked` (a
canary directory and file outside the run directory cannot be listed or read), `home_blocked`,
`proc_blocked`, `tmp_blocked`, `socket_blocked` and `env_empty`. Each reward worker runs the check
before its first code reward and refuses code rewards when it fails. The launcher runs it before
training whenever the dataset has code rows (`orbit_verl.data.reward.check_sandbox`).

## Tests

```
python -m pytest -q tests/verl/test_code_sandbox_policy.py tests/verl/test_code_sandbox_executor.py
```

The policy tests check that typical correct solutions pass and that every rule fires. The executor
tests start real sandboxes, so they need Linux with the kernel features above. They check that the
sandbox holds with the policy off, that every limit stops a program, and that early exit, fake verdicts
and comparison hacks never pass.

---

# orbit_verl.code_sandbox

Исполнитель для награды за код в RL. Он запускает программы модели и их тесты прямо на машине
обучения, без контейнера, и обходится стандартной библиотекой Python. Репозиторий должен быть
установлен (`python -m pip install -e . --no-deps`).

```python
from orbit_verl.code_sandbox import execute, ExecutorPool, Limits, self_check, extract_code, reward_info

assert all(self_check().values())        # один раз на воркер: держатся ли все защиты на этой машине?
program = extract_code(response_text)    # последний блок ```python в ответе
r = execute(program, [{"input": "1 2\n", "output": "3\n"}], kind="stdin")
r = execute(program, cases, kind="stdin", judge_imports=True, float_tolerance=1e-6)        # в стиле LCB
r = execute(program, "assert add(1, 2) == 3\n", kind="assert")                            # в стиле MBPP
r = execute(program, check_src, kind="assert", entry_point="find_zero", test_context=prompt)  # HumanEval
r = execute(program, tests, kind="assert", judge_imports=True, limits=Limits().for_cases(n))  # LeetCode
reward = 1.0 if r["passed"] else 0.0
metrics = reward_info(r)                 # фиксированный набор float-метрик для логов
with ExecutorPool() as pool:             # по умолчанию размер равен квоте CPU машины
    results = pool.map([{"program": p, "tests": t, "kind": "stdin"} for p, t in batch])
```

`r["passed"]` равно True, только если все тесты прошли с корректным вердиктом. `r["reason"]` принимает
значения `passed`, `policy_violation` (программа не запускалась), `syntax_error`, `wrong_answer`,
`test_failed`, `runtime_error`, `timeout`, `memory_limit`, `output_limit`, `early_exit` (`sys.exit` до
конца тестов), `non_plain_result` (значение, которым можно подделать сравнение), `no_verdict`,
`verdict_tampered`, `bad_tests` (нет кейсов или не выполнился ни один assert) и `sandbox_error`,
который никогда не засчитывается как успех. Что пошло не так, видно в `detail` и `stdout_tail`.
`reward_info(None)` даёт те же ключи метрик с нулями, потому что verl нужны одинаковые ключи для всех
ответов батча.

## От чего защищает

Обучаемая модель усвоит всё, что поднимает награду, поэтому программа не должна читать ожидаемые
ответы, подделывать вердикт, выходить в сеть, трогать чужие процессы и файлы и видеть секреты
трейнера. Если какую-то из защит не удалось настроить, запуск получает `sandbox_error`.

1. Каждый запуск стартует под `unshare --user --map-current-user --pid` с пустым окружением, в
   каталоге на локальном диске, который потом удаляется. Landlock разрешает только чтение системных
   каталогов и библиотек Python, так что домашние каталоги и каталоги данных, `/proc`, `/tmp`,
   `/dev/shm`, устройства GPU и большая часть `/etc` недоступны, а начиная с ABI 4 и TCP. Seccomp
   запрещает сокеты, ptrace, монтирование, создание namespace, bpf, io_uring и похожие вызовы. Rlimits
   ограничивают время CPU, память, вывод, число процессов и открытых файлов.
2. Статическая политика (`policy.py`, `rules.py`) ещё до запуска отклоняет запрещённые импорты,
   атрибуты и встроенные функции (`eval`, `exec`, ...) и интроспекцию, а внутри песочницы те же
   проверки повторяют обёрнутые встроенные функции.
3. В режиме assert тесты видят настоящие встроенные функции и модули, даже если программа их
   переопределила. Каждое значение, которое возвращает программа, проверяется, поэтому объект, равный
   чему угодно, не проходит.

Вердикт приходит одной строкой с одноразовым nonce по отдельному pipe, а напечатанный вывод и коды
выхода вердиктом не считаются. Промпт HumanEval передавайте в `test_context`, иначе программа может
переопределить хелпер, который вызывают тесты (`poly` в HumanEval/32).

## Проверка решений и лимиты

В режиме stdin каждый кейс идёт в свежей песочнице, и вывод совпадает, если его токены, разделённые
пробелами, равны ожидаемым. Для данных AtCoder и LCB передавайте `float_tolerance=1e-6`, потому что
многие задачи допускают такую погрешность, а эталонные выводы LCB печатают больше знаков, чем
правильная программа. Для данных в стиле LCB передавайте `judge_imports=True` в обоих режимах: он
добавляет импорты, которые LeetCode и LiveCodeBench дают каждой программе (`List`, `Counter`, `inf`,
...), и `sys.setrecursionlimit(50000)`.

`Limits()` даёт запуску 6 секунд CPU (`time_s`, бюджет LiveCodeBench на кейс), `2 * time_s + 2`
секунд по часам, 1024 МБ адресного пространства, 16 МБ вывода, 16 процессов и потоков выполнения, 64
открытых файла и рабочий каталог только для чтения (`writable_work_dir=True` открывает запись). Запуск
в режиме assert проверяет все кейсы в одной песочнице, поэтому ему нужен
`Limits().for_cases(n, cap_s=30)`: `time_s` на кейс, но не больше 30 секунд всего. На машине
медленнее той, где проверял LCB, меньше 1% правильных программ LCB не укладываются в лимит CPU, а
нескольким нужно больше 1 ГБ. Такие программы теряют награду.

## Требования к ядру и самопроверка

Защитам нужен Linux на x86_64 с user namespace, Landlock и seccomp. Проверьте каждую машину, на которой
считаются награды:

```bash
python -c "from orbit_verl.code_sandbox import self_check; print(self_check())"
```

Все значения должны быть True: `ran` (песочница запустилась и вернула вердикт), `outside_blocked`
(каталог и файл-канарейку вне каталога запуска нельзя ни перечислить, ни прочитать), `home_blocked`,
`proc_blocked`, `tmp_blocked`, `socket_blocked` и `env_empty`. Каждый воркер награды запускает
проверку перед первой наградой за код и при неудаче отказывается считать награды за код. Лаунчер
запускает её перед обучением, если в датасете есть строки с кодом
(`orbit_verl.data.reward.check_sandbox`).

## Тесты

```
python -m pytest -q tests/verl/test_code_sandbox_policy.py tests/verl/test_code_sandbox_executor.py
```

Тесты политики проверяют, что типичные правильные решения проходят, а каждое правило срабатывает.
Тесты исполнителя запускают настоящие песочницы, поэтому им нужен Linux с перечисленными выше
возможностями ядра. Они проверяют, что с выключенной политикой песочница держится сама, что каждый
лимит останавливает программу и что ранний выход, поддельные вердикты и трюки со сравнением никогда
не проходят.
