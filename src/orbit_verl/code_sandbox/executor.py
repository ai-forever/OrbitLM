"""Parent-side API of the executor: execute(), Limits and ExecutorPool.

execute(program, tests, kind, limits) checks the program against the static policy, runs it in the sandbox
(runner.py) and returns a plain dict:

    passed            True only if every test passed and every run produced a valid verdict record
    reason            "passed" or why not (see REASONS)
    policy_violations the policy reasons when reason == "policy_violation", else []
    runtime           wall-clock seconds spent in sandbox runs
    stdout_tail       the end of the program's stdout from the last run, for debugging
    shadowed_names    assert mode: names such as abs or math that the program defined and the tests read. The
                      tests used the real builtin or module; a non-empty list is a reward-hacking signal.
    plus tests_passed, tests_total, cpu_time and detail.

Two test styles are supported:
  kind="stdin":  tests is a list of {"input": str, "output": str}. Each case runs in a fresh sandbox with the
                 input as a regular file on stdin. Stdout matches when its whitespace-separated tokens equal the
                 expected ones, where two numbers also match if they are equal as decimals ("1.0" and "1.000"),
                 or within float_tolerance if one is given.
  kind="assert": tests is Python source with assert statements (MBPP style) or a check(candidate) function
                 (HumanEval style, pass entry_point to call it). Program and tests run in one sandbox.

Everything here is standard library. Each run gets its own directory under tmp_root (default /tmp, which should
be a local disk) and the directory is deleted when the run ends.
"""

import builtins
import dataclasses
import decimal
import importlib
import json
import marshal
import math
import multiprocessing
import os
import re
import secrets
import select
import shutil
import signal
import stat
import subprocess
import symtable
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from . import policy, rules

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")
# The runner and the rules it imports are copied into every run directory, which is on local disk. When the
# package lives on a shared filesystem, starting the sandbox from the package directory would make every run
# read them over the network, and those reads can cap throughput before the CPU does. The sources are read once
# per process.
_RUNNER_FILES = ("runner.py", "rules.py")
_runner_sources = None


def _runner_files():
    global _runner_sources
    if _runner_sources is None:
        package_dir = os.path.dirname(RUNNER)
        sources = {}
        for name in _RUNNER_FILES:
            with open(os.path.join(package_dir, name), "rb") as f:
                sources[name] = f.read()
        _runner_sources = sources
    return _runner_sources
UNSHARE = shutil.which("unshare") or "/usr/bin/unshare"
SETUP_FAILED_EXIT = 70  # Must match runner.SETUP_FAILED_EXIT.

REASONS = (
    "passed", "policy_violation", "syntax_error", "wrong_answer", "runtime_error", "timeout", "memory_limit",
    "output_limit", "test_failed", "early_exit", "non_plain_result", "no_verdict", "verdict_tampered",
    "bad_tests", "sandbox_error",
)

# Runner status -> reason. "ok" is not listed: it means the run itself succeeded.
_STATUS_REASON = {
    "exit": "runtime_error",
    "exception": "runtime_error",
    "memory_error": "memory_limit",
    "syntax_error": "syntax_error",
    "early_exit": "early_exit",
    "assert_failed": "test_failed",
    "not_plain": "non_plain_result",
    "tests_error": "bad_tests",
    "no_asserts": "bad_tests",
    "setup_error": "sandbox_error",
}


@dataclasses.dataclass(frozen=True)
class Limits:
    """Per-run limits. In stdin mode a run is one test case; in assert mode it is the whole test suite."""

    # CPU seconds (RLIMIT_CPU, rounded up to whole seconds by the kernel API). Six seconds per case is
    # LiveCodeBench's budget. On a slower machine some correct LCB programs need more.
    time_s: float = 6.0
    wall_s: float = 0.0         # wall-clock seconds; 0 means 2 * time_s + 2
    memory_mb: int = 1024       # address space (RLIMIT_AS)
    output_mb: int = 16         # largest file the program may write, which includes stdout (RLIMIT_FSIZE)
    nproc: int = 16             # processes and threads (RLIMIT_NPROC)
    nofile: int = 64            # open file descriptors (RLIMIT_NOFILE)
    # Read-only by default. AppArmor can forbid mounting a private tmpfs, so writes would hit disk.
    writable_work_dir: bool = False
    # With a writable work dir, RLIMIT_FSIZE caps each file but not how many there are. The parent therefore
    # measures the work dir while the program runs and kills the run above either of these.
    work_dir_mb: int = 64
    work_dir_files: int = 1000
    # A network namespace is redundant here: seccomp already denies socket() and socketpair(), and Landlock
    # denies TCP. The kernel serialises network namespace creation, which caps a whole machine at roughly 190
    # sandbox starts per second instead of about 2000, so it is off unless asked for.
    network_namespace: bool = False
    stdout_tail_chars: int = 2000

    def wall_seconds(self):
        return self.wall_s if self.wall_s > 0 else 2 * self.time_s + 2

    def for_cases(self, cases, cap_s=30.0):
        """Returns limits for one assert-mode run that holds `cases` test cases.

        In assert mode all cases share one CPU budget, so a correct program with many heavy cases (LiveCodeBench
        function-call tasks) can run out of time. This gives time_s per case, capped at cap_s, because a program
        that loops forever burns the whole budget before it fails.
        """
        seconds = min(float(cap_s), self.time_s * max(1, int(cases)))
        wall = 0.0 if self.wall_s <= 0 else min(self.wall_s * max(1, int(cases)), 2 * float(cap_s) + 2)
        return dataclasses.replace(self, time_s=seconds, wall_s=wall)


def execute(program, tests, kind="stdin", limits=None, *, entry_point=None, test_context=None,
            judge_imports=False, float_tolerance=None, check_policy=True, python=None, tmp_root=None,
            stop_on_first_failure=True):
    """Checks and runs one program against its tests. See the module docstring for the result format.

    entry_point     assert mode: the function the tests check. The harness appends check(entry_point), and the
                    tests get the program's object under this name even if it is a builtin's name.
    test_context    assert mode: trusted source, such as a HumanEval prompt, whose definitions the tests should
                    use instead of the program's. HumanEval/32 checks find_zero with the prompt's poly(); without
                    the context a program could redefine poly and pass with any answer.
    judge_imports   prepend judge_prelude(program): the imports LeetCode-style judges provide (List, Counter, inf,
                    ...) for the names the program uses without importing them.
    float_tolerance stdin mode: numbers within this absolute or relative difference also match.
    check_policy    False skips the static policy and the runtime guards. It exists so the tests can show that
                    the sandbox holds on its own; never use it for rewards.
    """
    limits = limits or Limits()
    python = python or sys.executable
    tmp_root = tmp_root or "/tmp"
    result = {"passed": False, "reason": "", "policy_violations": [], "runtime": 0.0, "stdout_tail": "",
              "tests_passed": 0, "tests_total": 0, "cpu_time": 0.0, "detail": "", "shadowed_names": []}

    if judge_imports and isinstance(program, str):
        program = judge_prelude(program) + program
    if check_policy:
        ok, reasons = policy.check_program(program)
        if not ok:
            is_syntax = len(reasons) == 1 and reasons[0].startswith("syntax_error")
            result["reason"] = "syntax_error" if is_syntax else "policy_violation"
            result["policy_violations"] = [] if is_syntax else reasons
            result["detail"] = reasons[0]
            return result

    if kind == "stdin":
        cases = [_normalise_case(c) for c in tests]
        result["tests_total"] = len(cases)
        if not cases:
            # With no cases there is no evidence the program works, so it must not earn a reward.
            result.update(reason="bad_tests", detail="no test cases")
            return result
        for index, (stdin_text, expected) in enumerate(cases):
            run = _run_once(program, "stdin", None, stdin_text.encode("utf-8"), limits, check_policy, python,
                            tmp_root)
            _accumulate(result, run, limits)
            reason = _classify(run, limits)
            if reason is None:
                got = run["stdout"].decode("utf-8", "replace")
                if outputs_match(got, expected, float_tolerance):
                    result["tests_passed"] += 1
                    continue
                reason = "wrong_answer"
                result["detail"] = f"case {index}: expected {_shorten(expected)!r}, got {_shorten(got)!r}"
            else:
                result["detail"] = f"case {index}: {_run_detail(run, limits)}"
            result["reason"] = result["reason"] or reason
            if stop_on_first_failure:
                break
        if result["tests_passed"] == result["tests_total"] and not result["reason"]:
            result["passed"] = True
            result["reason"] = "passed"
        return result

    if kind == "assert":
        tests_src = "\n".join(tests) if isinstance(tests, (list, tuple)) else str(tests)
        if entry_point:
            if not entry_point.isidentifier():
                raise ValueError(f"bad entry_point {entry_point!r}")
            tests_src += f"\n\ncheck({entry_point})\n"
        result["tests_total"] = 1
        run = _run_once(program, "assert", tests_src, b"", limits, check_policy, python, tmp_root,
                        entry_point=entry_point, context=test_context)
        _accumulate(result, run, limits)
        if run["record"]:
            result["shadowed_names"] = run["record"][3]
        reason = _classify(run, limits)
        if reason is None:
            result.update(passed=True, reason="passed", tests_passed=1)
        else:
            result.update(reason=reason, detail=_run_detail(run, limits))
        return result

    raise ValueError(f"unknown kind {kind!r}; expected 'stdin' or 'assert'")


def outputs_match(got, expected, float_tolerance=None):
    """Compares a program's stdout with the expected output, token by token after splitting on whitespace.

    Two differing tokens still match when both are plain decimal numbers with the same value ("1.0" and "1.000",
    "-0" and "0"), as in LiveCodeBench's checker. With float_tolerance they also match when they differ by at
    most that much, absolutely or relative to the expected value. The tolerance only applies when the expected
    token is written with a decimal point or an exponent. An integer answer therefore still has to be exact,
    even when the task also prints fractional numbers. Text tokens must be equal.
    """
    got_tokens, expected_tokens = got.split(), expected.split()
    if got_tokens == expected_tokens:
        return True
    if len(got_tokens) != len(expected_tokens):
        return False
    for a, b in zip(got_tokens, expected_tokens):
        if a != b and not _same_number(a, b, float_tolerance):
            return False
    return True


_NUMBER_CHARS = frozenset("0123456789+-.eE")


def _same_number(a, b, float_tolerance):
    # Only plain decimal notation is compared as numbers. Underscores, "nan", "inf" and hex never match a
    # different spelling, because a judge would not accept them either.
    if not (_NUMBER_CHARS.issuperset(a) and _NUMBER_CHARS.issuperset(b)) or len(a) > 4000 or len(b) > 4000:
        return False
    try:
        x, y = decimal.Decimal(a), decimal.Decimal(b)
    except (decimal.InvalidOperation, ValueError):
        return False
    if not (x.is_finite() and y.is_finite()):
        return False
    if x == y:
        return True
    if float_tolerance is None or not any(ch in b for ch in ".eE"):
        return False
    try:
        fx, fy = float(x), float(y)
    except (OverflowError, ValueError):
        return False
    return abs(fx - fy) <= float_tolerance * max(1.0, abs(fy))


# LeetCode and LiveCodeBench run every program after star imports from these modules, in this order, so a later
# module wins when two export the same name. typing comes last there too; here a name typing shares with
# collections (Counter, OrderedDict, ...) is taken from collections, which builds the same objects.
_PRELUDE_FROM = ("string", "re", "datetime", "collections", "heapq", "bisect", "copy", "math", "random",
                 "statistics", "itertools", "functools", "operator", "json", "typing")
# The judges then import these as modules, so 'datetime' or 'copy' finally names the module.
_PRELUDE_MODULES = ("string", "re", "datetime", "collections", "heapq", "bisect", "copy", "math", "random",
                    "statistics", "itertools", "functools", "operator", "io", "sys", "json")
_BUILTIN_NAMES = frozenset(dir(builtins))
_module_exports = {}


def _exports(module_name):
    names = _module_exports.get(module_name)
    if names is None:
        module = importlib.import_module(module_name)
        names = getattr(module, "__all__", None) or [n for n in dir(module) if not n.startswith("_")]
        # A name the policy would reject must never be put into a program by the harness itself.
        names = [n for n in names if rules.module_attr_problem(module_name, n) is None]
        _module_exports[module_name] = names
    return names


def _module_scope_reads(program):
    """Returns the names the program looks up in module scope and does not bind there.

    It uses symtable, the compiler's own scope analysis, so it follows Python's rules exactly. A method that uses
    a name the class body imported still reads it from module scope, because class scope is not visible to
    methods. An annotation runs in the scope that holds the def, so an import inside the method does not cover
    it. A local variable that happens to be called 'mode' or 'e' is not a module-scope read.
    """
    top = symtable.symtable(program, "<solution>", "exec")
    bound = {sym.get_name() for sym in top.get_symbols()
             if sym.is_assigned() or sym.is_imported() or sym.is_namespace()}
    reads = {sym.get_name() for sym in top.get_symbols() if sym.is_referenced()}
    stack = list(top.get_children())
    while stack:
        table = stack.pop()
        reads.update(sym.get_name() for sym in table.get_symbols() if sym.is_referenced() and sym.is_global())
        stack.extend(table.get_children())
    return reads - bound


def judge_prelude(program):
    """Returns import lines that give a program the names LeetCode-style judges import for it.

    LeetCode and LiveCodeBench run a program after lines like 'from collections import *' and
    'from math import *', plus sys.setrecursionlimit(50000). Correct submissions therefore use Counter, List or
    inf without importing them. Copying those lines would not work here: the policy rejects star imports of sys
    and io, and 'from math import *' would replace the builtin pow with math.pow. So the prelude imports, by
    name, exactly the names the program reads from module scope without binding there, and never shadows a
    builtin.
    """
    try:
        free = _module_scope_reads(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return ""
    free -= _BUILTIN_NAMES
    modules = [m for m in _PRELUDE_MODULES if m in free]
    owner = {}
    for module_name in _PRELUDE_FROM:
        for name in _exports(module_name):
            if name in free and name not in modules and not (module_name == "typing" and name in owner):
                owner[name] = module_name
    lines = ["import sys", "sys.setrecursionlimit(50000)"]
    if modules:
        lines.append("import " + ", ".join(modules))
    by_module = {}
    for name, module_name in owner.items():
        by_module.setdefault(module_name, []).append(name)
    for module_name in _PRELUDE_FROM:
        if module_name in by_module:
            lines.append(f"from {module_name} import {', '.join(sorted(by_module[module_name]))}")
    return "\n".join(lines) + "\n"


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Returns the program in a model response: the last fenced python block, else the last fenced block.

    A response without any fence is returned as it is, so a bare program still runs. The last block is taken
    because reasoning models often sketch code before they write the final answer.
    """
    blocks = _FENCE.findall(text)
    if not blocks:
        return text
    python_blocks = [body for lang, body in blocks if lang.lower() in ("python", "python3", "py")]
    return (python_blocks or [body for _, body in blocks])[-1]


# The metrics reward_info returns, always all of them, because verl expects every response of a batch to have
# the same keys. A response that is not code gets reward_info(None): every value is 0.
REWARD_INFO_KEYS = ("code_passed", "code_shadowed", "code_cpu_s") + tuple(f"code_{r}" for r in REASONS[1:])


def reward_info(result):
    """Turns an execute() result into the fixed set of float metrics in REWARD_INFO_KEYS.

    code_<reason> is 1 for the reason the program failed, so the mean over a batch is the share of each reason.
    code_shadowed is 1 when the program defined a builtin or module name that the tests read, which is the
    signature of the shadowing reward hack even when the run failed anyway.
    """
    info = dict.fromkeys(REWARD_INFO_KEYS, 0.0)
    if result:
        info["code_passed"] = float(bool(result["passed"]))
        info["code_shadowed"] = float(bool(result.get("shadowed_names")))
        info["code_cpu_s"] = float(result.get("cpu_time", 0.0))
        key = "code_" + str(result["reason"])
        if key in info:
            info[key] = 1.0
    return info


def _normalise_case(case):
    if isinstance(case, dict):
        return str(case.get("input", "")), str(case.get("output", ""))
    stdin_text, expected = case
    return str(stdin_text), str(expected)


def _shorten(text, limit=200):
    return text if len(text) <= limit else text[:limit] + "..."


def _accumulate(result, run, limits):
    result["runtime"] += run["wall"]
    result["cpu_time"] += run["cpu"]
    tail = run["stdout"][-4 * limits.stdout_tail_chars:].decode("utf-8", "replace")
    result["stdout_tail"] = tail[-limits.stdout_tail_chars:]


def _run_detail(run, limits):
    parts = []
    if run["record"]:
        parts.append(run["record"][0])
        if run["record"][2]:
            parts.append(run["record"][2])
    elif run["cpu"] >= math.ceil(limits.time_s) - 0.05:
        parts.append(f"CPU limit of {math.ceil(limits.time_s)}s reached")
    if run["verdict_problem"]:
        parts.append(run["verdict_problem"])
    if run["timed_out"]:
        parts.append(f"wall clock exceeded after {run['wall']:.2f}s")
    if run["disk_exceeded"]:
        parts.append("the writable work dir grew past its size or file-count limit")
    if run["returncode"]:
        parts.append(f"exit status {run['returncode']}")
    # When the kernel kills the program, the unshare wrapper prints its own complaint about restoring its signal
    # mask. That line says nothing about the program, so it is left out.
    stderr = "\n".join(line for line in run["stderr_tail"].decode("utf-8", "replace").strip().splitlines()
                       if not line.startswith("unshare: ")).strip()
    if stderr:
        parts.append("stderr: " + stderr[-600:])
    return "; ".join(parts)


def _classify(run, limits):
    """Returns None if the run succeeded, otherwise the reason it did not."""
    cpu_limit = math.ceil(limits.time_s)
    if run["disk_exceeded"]:
        return "output_limit"
    if run["timed_out"]:
        return "timeout"
    if run["verdict_problem"] == "tampered":
        return "verdict_tampered"
    if run["out_size"] >= limits.output_mb * 2 ** 20:
        return "output_limit"
    record = run["record"]
    if record is None:
        # No record means the harness never reached its end: the process was killed (CPU limit, memory,
        # signal) or left through a path that skips the harness, such as os._exit.
        if run["cpu"] >= cpu_limit - 0.05:
            return "timeout"
        if run["returncode"] == SETUP_FAILED_EXIT:
            return "sandbox_error"
        if run["returncode"] < 0:
            return "runtime_error"
        return "no_verdict"
    if run["returncode"] != 0:
        return "verdict_tampered"
    status = record[0]
    if status == "ok":
        return None
    return _STATUS_REASON.get(status, "verdict_tampered")


def _parse_verdict(data, nonce):
    """Accepts exactly one well-formed record that starts with this run's nonce, and nothing else."""
    if not data:
        return None, None
    if data.count(b"\n") != 1 or not data.endswith(b"\n"):
        return None, "tampered"
    parts = data[:-1].split(b" ", 4)
    if len(parts) != 5 or not secrets.compare_digest(parts[0], nonce):
        return None, "tampered"
    try:
        status = parts[1].decode("ascii")
        code = int(parts[2])
    except (UnicodeDecodeError, ValueError):
        return None, "tampered"
    shadowed = [] if parts[3] == b"-" else parts[3].decode("utf-8", "replace").split(",")
    return (status, code, parts[4].decode("utf-8", "replace"), shadowed), None


def _run_once(program, mode, tests_src, stdin_bytes, limits, guards, python, tmp_root, entry_point=None,
              context=None):
    """Runs the sandbox once and returns raw facts about the run; _classify interprets them."""
    run_dir = tempfile.mkdtemp(prefix="sbx_", dir=tmp_root)
    fds = []
    proc = None
    try:
        os.mkdir(os.path.join(run_dir, "work"), 0o700)
        for name, source in _runner_files().items():
            with open(os.path.join(run_dir, name), "wb") as f:
                f.write(source)
        spec = {
            "mode": mode, "program": program, "tests": tests_src, "guards": bool(guards),
            "entry_point": entry_point, "context": context,
            "writable_work_dir": bool(limits.writable_work_dir),
            "limits": {
                "cpu_seconds": math.ceil(limits.time_s),
                "memory_bytes": int(limits.memory_mb) * 2 ** 20,
                "fsize_bytes": int(limits.output_mb) * 2 ** 20,
                "nproc": int(limits.nproc),
                "nofile": int(limits.nofile),
            },
        }
        # marshal is built into the interpreter, so the runner can read the spec without importing json (and
        # the re and enum modules json pulls in) on every sandbox start.
        with open(os.path.join(run_dir, "spec.marshal"), "wb") as f:
            f.write(marshal.dumps(spec))

        # Stdin is a regular file, not a pipe, so os.fstat(0).st_size works as it does on a judge.
        stdin_path = os.path.join(run_dir, "stdin")
        with open(stdin_path, "wb") as f:
            f.write(stdin_bytes)
        stdin_fd = os.open(stdin_path, os.O_RDONLY)
        fds.append(stdin_fd)
        # Stdout and stderr are files whose descriptors the parent keeps. The parent reads them through these
        # descriptors, never by path, so swapping the paths for symlinks would not make it read other files.
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        stdout_fd = os.open(os.path.join(run_dir, "stdout"), flags, 0o600)
        fds.append(stdout_fd)
        stderr_fd = os.open(os.path.join(run_dir, "stderr"), flags, 0o600)
        fds.append(stderr_fd)

        nonce = secrets.token_hex(16).encode("ascii")
        nonce_r, nonce_w = os.pipe()
        fds += [nonce_r, nonce_w]
        os.write(nonce_w, nonce)
        verdict_r, verdict_w = os.pipe()
        fds += [verdict_r, verdict_w]

        namespaces = ["--user", "--map-current-user", "--pid", "--fork", "--kill-child"]
        if limits.network_namespace:
            namespaces.append("--net")
        argv = [UNSHARE, *namespaces, "--", python, "-I", "-S", "-B", "-X", "utf8",
                os.path.join(run_dir, "runner.py"), run_dir, str(nonce_r), str(verdict_w)]
        start = time.monotonic()
        # env={} keeps every secret in the parent's environment (tokens, keys) out of the sandbox entirely,
        # including the copy the kernel would otherwise place on the new process's stack.
        proc = subprocess.Popen(argv, stdin=stdin_fd, stdout=stdout_fd, stderr=stderr_fd,
                                pass_fds=(nonce_r, verdict_w), env={}, start_new_session=True, close_fds=True)
        for fd in (nonce_r, nonce_w, verdict_w):
            os.close(fd)
            fds.remove(fd)
        watch_dir = os.path.join(run_dir, "work") if limits.writable_work_dir else None
        killed = _wait_or_kill(proc, limits.wall_seconds(), watch_dir, int(limits.work_dir_mb) * 2 ** 20,
                               int(limits.work_dir_files))
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        wall = time.monotonic() - start

        verdict = _read_capped(verdict_r, 1 << 16, deadline=time.monotonic() + 2)
        record, problem = _parse_verdict(verdict, nonce)
        out_size = os.fstat(stdout_fd).st_size
        stdout = os.pread(stdout_fd, limits.output_mb * 2 ** 20, 0)
        err_size = os.fstat(stderr_fd).st_size
        stderr_tail = os.pread(stderr_fd, 2000, max(0, err_size - 2000))
        return {"record": record, "verdict_problem": problem, "timed_out": killed == "timeout",
                "disk_exceeded": killed == "disk",
                "returncode": proc.returncode, "wall": wall, "cpu": usage.ru_utime + usage.ru_stime,
                "stdout": stdout, "out_size": out_size, "stderr_tail": stderr_tail}
    finally:
        if proc is not None and proc.returncode is None:
            _kill_group(proc.pid)
            try:
                os.waitpid(proc.pid, 0)
            except ChildProcessError:
                pass
            proc.returncode = -signal.SIGKILL
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
        _remove_tree(run_dir)


def _wait_or_kill(proc, timeout, watch_dir=None, max_bytes=0, max_files=0):
    """Waits for the process to exit, without reaping it. Returns None, "timeout" or "disk".

    It kills the process group after timeout seconds. With watch_dir it also measures that directory every 50 ms
    and kills the run once it holds more than max_bytes or max_files. A pidfd wakes up exactly when the process
    exits; Popen.wait(timeout) polls with sleeps of up to 50 ms, which would add more overhead than a whole
    short run takes.
    """
    deadline = time.monotonic() + timeout
    try:
        pidfd = os.pidfd_open(proc.pid)
    except (AttributeError, OSError):
        pidfd = None
    poller = None
    if pidfd is not None:
        poller = select.poll()
        poller.register(pidfd, select.POLLIN)
    killed = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                killed = "timeout"
                break
            step = remaining if watch_dir is None else min(remaining, 0.05)
            if _exited_within(proc, poller, step):
                break
            if watch_dir is not None and _dir_exceeds(watch_dir, max_bytes, max_files):
                killed = "disk"
                break
    finally:
        if pidfd is not None:
            os.close(pidfd)
    if killed:
        # unshare runs in its own process group. Killing the group kills unshare, and --kill-child makes the
        # kernel kill the sandbox's pid 1, which tears down the whole pid namespace with it.
        _kill_group(proc.pid)
    return killed


def _exited_within(proc, poller, seconds):
    if poller is not None:
        return bool(poller.poll(max(1, int(seconds * 1000))))
    deadline = time.monotonic() + seconds
    while True:
        if os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.005)


def _dir_exceeds(path, max_bytes, max_files):
    """Says whether a directory tree holds more than max_bytes on disk or more than max_files entries.

    Symlinks are not followed, so the program cannot point the count at other directories. Counting stops at
    the first limit, so a directory with millions of entries costs at most max_files stat calls.
    """
    total, count = 0, 0
    stack = [path]
    while stack:
        try:
            entries = os.scandir(stack.pop())
        except OSError:
            continue
        with entries:
            for entry in entries:
                count += 1
                if count > max_files:
                    return True
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                total += info.st_blocks * 512
                if total > max_bytes:
                    return True
                if stat.S_ISDIR(info.st_mode):
                    stack.append(entry.path)
    return False


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _read_capped(fd, cap, deadline):
    """Reads the verdict pipe until EOF, at most cap bytes, and gives up at the deadline.

    The sandbox is dead when this runs, so EOF normally comes at once. The deadline only matters if something
    in the sandbox still held the pipe open; the run then counts as failed instead of hanging the worker.
    """
    chunks, total = [], 0
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    while total <= cap:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not poller.poll(int(remaining * 1000) + 1):
            chunks.append(b"\n<verdict pipe still open>")
            break
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _remove_tree(path):
    """Deletes a run directory even if the program changed permissions inside a writable work dir."""
    def on_error(func, failed_path, _exc):
        try:
            os.chmod(os.path.dirname(failed_path), stat.S_IRWXU)
            os.chmod(failed_path, stat.S_IRWXU)
        except OSError:
            pass
        try:
            if os.path.isdir(failed_path) and not os.path.islink(failed_path):
                shutil.rmtree(failed_path, ignore_errors=True)
            else:
                os.unlink(failed_path)
        except OSError:
            pass

    shutil.rmtree(path, onerror=on_error)


# The probe program of self_check. The two paths are filled in by self_check, as Python string literals.
_SELF_CHECK_PROBE = r'''
import json, os, socket
def blocked(fn):
    try:
        fn()
        return False
    except OSError:
        return True
canary_dir, home_dir = CANARY_DIR, HOME_DIR
print(json.dumps({
    "outside_blocked": blocked(lambda: os.listdir(canary_dir))
                       and blocked(lambda: open(os.path.join(canary_dir, "canary")).read()),
    "home_blocked": blocked(lambda: os.listdir(home_dir)),
    "proc_blocked": blocked(lambda: open("/proc/self/status").read()),
    "tmp_blocked": blocked(lambda: os.listdir("/tmp")),
    "socket_blocked": blocked(lambda: socket.socket(socket.AF_UNIX)),
    "env_empty": not os.environ,
}))
'''


def self_check(python=None, tmp_root=None):
    """Runs one probe program in the sandbox and reports which protections are active on this machine.

    Call it once when a reward worker starts. Every value in the returned dict should be True.

    The probe only tries paths that exist and that this process can read, so a missing path can never make a
    check pass. ``outside_blocked`` lists a canary directory and reads a canary file that this function writes
    under ``tmp_root``, outside the run directory. ``home_blocked`` lists the home directory of this process.
    When this process cannot list its own home either, the probe lists the canary directory in its place.
    """
    tmp_root = tmp_root or "/tmp"
    canary_dir = tempfile.mkdtemp(prefix="sbx_canary_", dir=tmp_root)
    try:
        with open(os.path.join(canary_dir, "canary"), "w") as f:
            f.write("parent secret")
        home_dir = os.path.expanduser("~")
        try:
            os.listdir(home_dir)
        except OSError:
            home_dir = canary_dir
        probe = _SELF_CHECK_PROBE.replace("CANARY_DIR", repr(canary_dir)).replace("HOME_DIR", repr(home_dir))
        run = _run_once(probe, "stdin", None, b"", Limits(), False, python or sys.executable, tmp_root)
    finally:
        shutil.rmtree(canary_dir, ignore_errors=True)
    report = {"ran": _classify(run, Limits()) is None}
    try:
        report.update(json.loads(run["stdout"].decode()))
    except ValueError:
        report["detail"] = _run_detail(run, Limits())
    return report


def available_cpus():
    """CPUs this process may actually use: the affinity mask, capped by the cgroup CPU quota.

    A container often sees every CPU of the node but may only use a quota of them, for example 12 of 192.
    A pool sized by the visible count oversubscribes the quota, and the kernel then throttles every sandbox.
    """
    cpus = len(os.sched_getaffinity(0))
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            with open(path) as f:
                fields = f.read().split()
            if path.endswith("cpu.max"):
                quota, period = fields[0], fields[1]
            else:
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                    quota, period = fields[0], f.read().strip()
            if quota not in ("max", "-1"):
                cpus = min(cpus, max(1, int(quota) // int(period)))
            break
        except (OSError, ValueError, IndexError):
            continue
    return cpus


def _execute_job(job):
    return execute(**job)


class ExecutorPool:
    """Runs many execute() calls in parallel, one sandbox per worker at a time.

    The workers are processes by default (forkserver start method). A process pool keeps the policy check and
    the result handling out of the caller's GIL, and forkserver starts workers from a small clean process
    instead of forking a large trainer process that may hold CUDA or Ray state. Pass processes=False for a
    thread pool, which starts instantly and is enough for small batches.

        with ExecutorPool(workers=64) as pool:
            results = pool.map([{"program": p, "tests": t, "kind": "stdin"} for p, t in batch])
    """

    def __init__(self, workers=None, processes=True, **defaults):
        self.workers = workers or available_cpus()
        self.defaults = defaults
        if processes:
            context = multiprocessing.get_context("forkserver")
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=context)
        else:
            self._pool = ThreadPoolExecutor(max_workers=self.workers)

    def submit(self, program, tests, kind="stdin", limits=None, **kwargs):
        job = dict(self.defaults, program=program, tests=tests, kind=kind, **kwargs)
        if limits is not None:
            job["limits"] = limits
        return self._pool.submit(_execute_job, job)

    def map(self, jobs):
        futures = [self._pool.submit(_execute_job, dict(self.defaults, **job)) for job in jobs]
        return [f.result() for f in futures]

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


__all__ = ["execute", "Limits", "ExecutorPool", "self_check", "available_cpus", "judge_prelude", "outputs_match",
           "extract_code", "reward_info", "REWARD_INFO_KEYS", "REASONS"]
