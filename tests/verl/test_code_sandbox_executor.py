"""End-to-end tests of orbit_verl.code_sandbox. They start real sandboxes, so run them on the machine that will run
rewards, with PYTHONPATH=<repo>/src (see src/orbit_verl/code_sandbox/README.md). Elsewhere than Linux the module
is skipped.
"""

import glob
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest

if not sys.platform.startswith("linux"):
    raise unittest.SkipTest("the sandbox needs Linux (Landlock, seccomp, user namespaces)")

import orbit_verl.code_sandbox  # noqa: E402
from orbit_verl.code_sandbox import ExecutorPool, Limits, execute, outputs_match, self_check  # noqa: E402
from orbit_verl.code_sandbox.executor import _run_once  # noqa: E402
from test_code_sandbox_policy import PYRIVAL_FAST_IO, THREAD_RECURSION  # noqa: E402

SHORT = Limits(time_s=1, wall_s=4, memory_mb=512, output_mb=1)
PACKAGE_DIR = os.path.dirname(os.path.abspath(orbit_verl.code_sandbox.__file__))


def runner_processes():
    """Pids of sandbox processes that are still alive. Used to prove nothing outlives a run.

    Every sandbox runs a copy of runner.py from its own run directory under /tmp, so any live process whose
    command line names such a copy is a leftover.
    """
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                args = f.read().split(b"\0")
        except OSError:
            continue
        if any(a.startswith(b"/tmp/sbx_") and a.endswith(b"/runner.py") for a in args):
            found.append(int(pid))
    return found


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def probe(body, **kwargs):
    """Runs a program with the policy off and returns the JSON object it printed.

    The policy is off on purpose: these tests show that the sandbox alone stops what the policy would also
    have rejected.
    """
    program = "import json, os, sys\nresults = {}\n" + body + "\nprint(json.dumps(results))\n"
    result = execute(program, [{"input": "", "output": ""}], "stdin", kwargs.pop("limits", SHORT),
                     check_policy=False, **kwargs)
    try:
        return json.loads(result["stdout_tail"]), result
    except ValueError:
        raise AssertionError(f"probe produced no JSON: {result}")


ATTEMPT = '''
def attempt(name, fn):
    try:
        value = fn()
        results[name] = "ALLOWED " + repr(value)[:80]
    except Exception as exc:
        results[name] = "blocked " + type(exc).__name__
'''


class CorrectSolutionsPass(unittest.TestCase):
    def test_stdin_a_plus_b(self):
        program = "a, b = map(int, input().split())\nprint(a + b)\n"
        cases = [{"input": "1 2\n", "output": "3\n"}, {"input": "-5 7", "output": "2"},
                 {"input": "10 20\n", "output": "30   \n\n"}]
        result = execute(program, cases, "stdin")
        self.assertTrue(result["passed"], result)
        self.assertEqual((result["tests_passed"], result["tests_total"]), (3, 3))

    def test_stdin_idioms(self):
        cases = [{"input": "3\n1 2 3\n", "output": "6\n"}, {"input": "2\n10 -4\n", "output": "6"}]
        programs = {
            "pyrival_fast_io": PYRIVAL_FAST_IO,
            "open_zero": "data = open(0).read().split()\nprint(sum(map(int, data[1:])))\n",
            "os_read_fstat": "import io, os\ninput = io.BytesIO(os.read(0, os.fstat(0).st_size)).readline\n"
                             "input()\nprint(sum(map(int, input().split())))\n",
            "stdin_buffer": "import sys\ndata = sys.stdin.buffer.read().split()\n"
                            "sys.stdout.write(str(sum(map(int, data[1:]))) + '\\n')\n",
            "exit_after_print": "import sys\nn = int(input())\nprint(sum(map(int, input().split())))\nsys.exit(0)\n"
                                "print('never')\n",
            "exit_builtin_midway": "n = int(input())\ns = sum(map(int, input().split()))\nif s:\n"
                                   "    print(s)\n    exit()\nprint(0)\n",
        }
        for name, program in programs.items():
            with self.subTest(name=name):
                result = execute(program, cases, "stdin")
                self.assertTrue(result["passed"], f"{name}: {result}")

    def test_thread_recursion_idiom(self):
        result = execute(THREAD_RECURSION, [{"input": "100000\n", "output": "100000\n"}], "stdin")
        self.assertTrue(result["passed"], result)

    def test_assert_style_mbpp(self):
        program = "def remove_occ(s, ch):\n    return s.replace(ch, '')\n"
        tests = ['assert remove_occ("hello", "l") == "heo"', 'assert remove_occ("abcda", "a") == "bcd"']
        result = execute(program, tests, "assert")
        self.assertTrue(result["passed"], result)

    def test_assert_style_humaneval(self):
        program = ("from typing import List\n\n"
                   "def has_close(numbers: List[float], t: float) -> bool:\n"
                   "    s = sorted(numbers)\n"
                   "    return any(b - a < t for a, b in zip(s, s[1:]))\n")
        tests = ("def check(candidate):\n"
                 "    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n"
                 "    assert candidate([1.0, 2.8, 3.0], 0.3) == True\n"
                 "    assert abs(len([candidate([1.0], 1.0)]) - 1) < 1e-9\n")
        result = execute(program, tests, "assert", entry_point="has_close")
        self.assertTrue(result["passed"], result)

    def test_assert_accepts_ordinary_return_types(self):
        program = '''
from collections import Counter, namedtuple, OrderedDict
P = namedtuple("P", "x y")

class Pair:
    def __init__(self, a, b):
        self.a, self.b = a, b
    def __eq__(self, other):
        return (self.a, self.b) == (other.a, other.b)

class Node:
    def __init__(self, val, nxt=None):
        self.val, self.nxt = val, nxt

def counts(s): return Counter(s)
def point(): return P(1, 2)
def squares(n): return (i * i for i in range(n))
def chain(n): return Node(1, Node(2)) if n else None
def longest(pairs): return max(p.b - p.a for p in pairs)
def nested(): return {"a": [1, (2, 3.5)], "b": {4, 5}, "c": OrderedDict(x=None)}
'''
        tests = '''
assert counts("aab") == {"a": 2, "b": 1}
assert point() == (1, 2)
assert list(squares(4)) == [0, 1, 4, 9]
def values(node):
    out = []
    while node:
        out.append(node.val)
        node = node.nxt
    return out
assert values(chain(1)) == [1, 2]
assert longest([Pair(1, 5), Pair(2, 3)]) == 4
assert nested() == {"a": [1, (2, 3.5)], "b": {4, 5}, "c": {"x": None}}
'''
        result = execute(program, tests, "assert")
        self.assertTrue(result["passed"], result)


class FailuresAreReported(unittest.TestCase):
    def test_wrong_answer(self):
        result = execute("print(int(input()) + 1)\n", [{"input": "1", "output": "3"}], "stdin")
        self.assertEqual(result["reason"], "wrong_answer", result)
        self.assertIn("expected", result["detail"])

    def test_runtime_error(self):
        result = execute("print(1 // 0)\n", [{"input": "", "output": "1"}], "stdin")
        self.assertEqual(result["reason"], "runtime_error", result)
        self.assertIn("ZeroDivisionError", result["detail"])

    def test_nonzero_exit(self):
        result = execute("import sys\nprint(3)\nsys.exit(1)\n", [{"input": "", "output": "3"}], "stdin")
        self.assertEqual(result["reason"], "runtime_error", result)

    def test_syntax_error(self):
        result = execute("def f(:\n", [{"input": "", "output": ""}], "stdin")
        self.assertEqual(result["reason"], "syntax_error", result)

    def test_failed_assert_reports_line(self):
        result = execute("def f(x):\n    return x\n", "assert f(1) == 1\nassert f(2) == 3\n", "assert")
        self.assertEqual(result["reason"], "test_failed", result)
        self.assertIn("line 2: assert f(2) == 3", result["detail"])

    def test_policy_violation_skips_execution(self):
        before = runner_processes()
        result = execute("import subprocess\nprint(1)\n", [{"input": "", "output": "1"}], "stdin")
        self.assertEqual(result["reason"], "policy_violation")
        self.assertEqual(result["runtime"], 0.0)
        self.assertTrue(result["policy_violations"][0].startswith("denied_import"), result)
        self.assertEqual(before, runner_processes())

    def test_no_cases_is_not_a_pass(self):
        result = execute("print(1)\n", [], "stdin")
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "bad_tests")


class SandboxBoundary(unittest.TestCase):
    """Policy off: each attempt must be stopped by the sandbox itself."""

    def test_self_check(self):
        report = self_check()
        self.assertTrue(all(report.values()), report)
        expected = {"ran", "outside_blocked", "home_blocked", "proc_blocked", "tmp_blocked", "socket_blocked",
                    "env_empty"}
        self.assertEqual(set(report), expected)

    def test_package_home_and_tmp(self):
        marker = os.path.join(PACKAGE_DIR, "policy.py")
        evil = os.path.join(PACKAGE_DIR, "evil_from_sandbox.txt")
        # Only directories that exist and that this process can list are probed, because listing a missing
        # directory fails inside the sandbox too and would count as blocked without proving anything.
        candidates = (os.path.expanduser("~"), os.path.dirname(PACKAGE_DIR))
        outside_dirs = [path for path in candidates if os.path.isdir(path)]
        for path in outside_dirs:
            os.listdir(path)
        with tempfile.NamedTemporaryFile("w", dir="/tmp", prefix="sbx_marker_", delete=False) as f:
            f.write("parent secret")
            tmp_marker = f.name
        try:
            body = ATTEMPT + f'''
attempt("read package file", lambda: open({marker!r}).read(10))
for path in {outside_dirs!r}:
    attempt("list " + path, lambda path=path: os.listdir(path))
attempt("write file into the package", lambda: open({evil!r}, "w").write("x"))
attempt("write into python prefix", lambda: open(sys.prefix + "/lib/evil.txt", "w").write("x"))
attempt("read parent file in /tmp", lambda: open({tmp_marker!r}).read())
attempt("list /tmp", lambda: os.listdir("/tmp"))
attempt("read /etc/passwd", lambda: open("/etc/passwd").read(10))
attempt("write in work dir", lambda: open("scratch.txt", "w").write("x"))
attempt("read stdlib (allowed)", lambda: len(open(os.__file__).read()) > 0)
'''
            results, _ = probe(body)
        finally:
            os.unlink(tmp_marker)
        for name, outcome in results.items():
            with self.subTest(name=name):
                if name == "read stdlib (allowed)":
                    self.assertTrue(outcome.startswith("ALLOWED"), outcome)
                else:
                    self.assertTrue(outcome.startswith("blocked"), f"{name}: {outcome}")
        self.assertFalse(os.path.exists(evil))

    def test_writable_work_dir_is_opt_in(self):
        body = ATTEMPT + 'attempt("write in work dir", lambda: open("scratch.txt", "w").write("x"))\n'
        limits = Limits(time_s=1, wall_s=4, memory_mb=512, output_mb=1, writable_work_dir=True)
        results, _ = probe(body, limits=limits)
        self.assertEqual(results["write in work dir"], "ALLOWED 1")

    def test_proc_and_devices(self):
        devices = sorted(glob.glob("/dev/nvidia*"))
        body = ATTEMPT + f'''
attempt("read /proc/self/environ", lambda: open("/proc/self/environ", "rb").read())
attempt("read /proc/self/mem", lambda: open("/proc/self/mem", "rb"))
attempt("read /proc/1/cmdline", lambda: open("/proc/1/cmdline", "rb").read())
attempt("list /proc", lambda: os.listdir("/proc"))
attempt("list /dev/shm", lambda: os.listdir("/dev/shm"))
attempt("open /dev/zero", lambda: open("/dev/zero", "rb").read(1))
for dev in {devices!r}:
    attempt("open " + dev, lambda dev=dev: open(dev, "rb"))
attempt("write /dev/null (allowed)", lambda: open("/dev/null", "w").write("x"))
'''
        results, _ = probe(body)
        if not devices:
            print("\n  note: no /dev/nvidia* nodes on this host; GPU check covered by /dev/zero and /dev/shm")
        for name, outcome in results.items():
            with self.subTest(name=name):
                expected = "ALLOWED" if "(allowed)" in name else "blocked"
                self.assertTrue(outcome.startswith(expected), f"{name}: {outcome}")
        self.assertTrue(any(name.startswith("open /dev/nvidia") for name in results) or not devices)

    def test_network(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        unix_path = tempfile.mktemp(prefix="sbx_sock_", dir="/tmp")
        unix_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        unix_server.bind(unix_path)
        unix_server.listen(1)
        try:
            body = ATTEMPT + f'''
import socket
attempt("tcp to parent on localhost", lambda: socket.create_connection(("127.0.0.1", {port}), timeout=2))
attempt("tcp to the internet", lambda: socket.create_connection(("1.1.1.1", 443), timeout=2))
attempt("udp socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
attempt("unix socket to parent", lambda: socket.socket(socket.AF_UNIX).connect({unix_path!r}))
attempt("socketpair", lambda: socket.socketpair())
'''
            results, _ = probe(body)
        finally:
            server.close()
            unix_server.close()
            os.unlink(unix_path)
        for name, outcome in results.items():
            with self.subTest(name=name):
                self.assertTrue(outcome.startswith("blocked"), f"{name}: {outcome}")

    def test_optional_network_namespace(self):
        limits = Limits(time_s=1, wall_s=4, memory_mb=512, output_mb=1, network_namespace=True)
        body = ATTEMPT + 'import socket\nattempt("udp socket", lambda: socket.socket(socket.AF_INET, ' \
                         'socket.SOCK_DGRAM))\n'
        results, _ = probe(body, limits=limits)
        self.assertTrue(results["udp socket"].startswith("blocked"), results)

    def test_signals_cannot_reach_outside(self):
        helper = subprocess.Popen(["sleep", "60"])
        try:
            body = ATTEMPT + f'''
import signal
attempt("kill helper process", lambda: os.kill({helper.pid}, signal.SIGKILL))
attempt("kill test process", lambda: os.kill({os.getpid()}, signal.SIGTERM))
attempt("kill pid 1 of the machine", lambda: os.kill(1, 0) or "own namespace init")
attempt("kill -1 (everything visible)", lambda: os.kill(-1, signal.SIGKILL) or "nothing else visible")
results["own pid"] = os.getpid()
'''
            results, _ = probe(body)
            time.sleep(0.2)
            self.assertIsNone(helper.poll(), "the helper process was killed from the sandbox")
        finally:
            helper.kill()
            helper.wait()
        self.assertEqual(results["own pid"], 1, "the program should be pid 1 of a fresh pid namespace")
        self.assertTrue(results["kill helper process"].startswith("blocked ProcessLookupError"), results)
        self.assertTrue(results["kill test process"].startswith("blocked ProcessLookupError"), results)

    def test_environment_is_empty(self):
        os.environ["SBX_TEST_SECRET"] = "s3cr3t-value"
        try:
            body = ATTEMPT + 'results["environ"] = dict(os.environ)\n'
            results, _ = probe(body)
        finally:
            del os.environ["SBX_TEST_SECRET"]
        self.assertEqual(results["environ"], {})


class LimitsStopRunaways(unittest.TestCase):
    def setUp(self):
        self.rss_before = rss_mb()

    def tearDown(self):
        # A killed sandbox needs a moment to disappear from /proc. On a busy machine that takes longer than on an idle
        # one, so the check waits up to three seconds before it calls a process a survivor.
        deadline = time.monotonic() + 3.0
        while runner_processes() and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertEqual(runner_processes(), [], "sandbox processes outlived the run")
        self.assertLess(rss_mb() - self.rss_before, 50, "the parent's memory grew")
        ok = execute("print(2)\n", [{"input": "", "output": "2"}], "stdin")
        self.assertTrue(ok["passed"], "the parent cannot run a normal program after a runaway one")

    def test_cpu_time(self):
        start = time.monotonic()
        result = execute("while True:\n    pass\n", [{"input": "", "output": ""}], "stdin", SHORT)
        self.assertEqual(result["reason"], "timeout", result)
        self.assertLess(time.monotonic() - start, 4.5)

    def test_wall_time_without_cpu(self):
        result = execute("import time\ntime.sleep(100)\n", [{"input": "", "output": ""}], "stdin", SHORT)
        self.assertEqual(result["reason"], "timeout", result)
        self.assertLess(result["runtime"], 5)

    def test_memory(self):
        for name, program in {"one block": "x = bytearray(2 * 1024 ** 3)\n",
                              "gradual": "x = []\nwhile True:\n    x.append([0] * 10 ** 6)\n"}.items():
            with self.subTest(name=name):
                result = execute(program, [{"input": "", "output": ""}], "stdin", SHORT)
                self.assertEqual(result["reason"], "memory_limit", result)

    def test_output_size(self):
        program = "import sys\nwhile True:\n    sys.stdout.write('x' * 65536)\n"
        result = execute(program, [{"input": "", "output": ""}], "stdin", SHORT)
        self.assertEqual(result["reason"], "output_limit", result)

    def test_writable_work_dir_is_capped(self):
        # RLIMIT_FSIZE caps each file but not how many there are. The parent measures the work dir instead.
        limits = Limits(time_s=5, wall_s=10, memory_mb=512, output_mb=1, writable_work_dir=True,
                        work_dir_mb=8, work_dir_files=200)
        many_files = "i = 0\nwhile True:\n    open('f%d' % i, 'wb').write(b'0' * 4096)\n    i += 1\n"
        many_bytes = "i = 0\nwhile True:\n    open('f%d' % i, 'wb').write(b'0' * 1000000)\n    i += 1\n"
        for name, program in {"many small files": many_files, "many large files": many_bytes}.items():
            with self.subTest(name=name):
                start = time.monotonic()
                result = execute(program, [{"input": "", "output": ""}], "stdin", limits, check_policy=False)
                self.assertEqual(result["reason"], "output_limit", result)
                self.assertLess(time.monotonic() - start, 3)

    def test_file_size_in_writable_work_dir(self):
        limits = Limits(time_s=1, wall_s=4, memory_mb=512, output_mb=1, writable_work_dir=True)
        body = ATTEMPT + '''
attempt("write 2 MB file", lambda: open("big.bin", "wb").write(b"0" * (2 << 20)))
results["size on disk"] = os.stat("big.bin").st_size
'''
        results, _ = probe(body, limits=limits)
        self.assertTrue(results["write 2 MB file"].startswith("blocked"), results)
        self.assertLessEqual(results["size on disk"], 1 << 20)

    def test_process_count(self):
        body = '''
import time
children = 0
for _ in range(100):
    try:
        pid = os.fork()
    except OSError:
        break
    if pid == 0:
        time.sleep(60)
        os._exit(0)
    children += 1
results["children"] = children
'''
        limits = Limits(time_s=1, wall_s=4, memory_mb=512, output_mb=1, nproc=8)
        start = time.monotonic()
        results, _ = probe(body, limits=limits)
        self.assertLess(results["children"], 8)
        self.assertLess(time.monotonic() - start, 3, "sleeping children should die with the sandbox's pid 1")

    def test_fork_bomb(self):
        program = "import os\nwhile True:\n    try:\n        os.fork()\n    except OSError:\n        pass\n"
        limits = Limits(time_s=1, wall_s=3, memory_mb=512, output_mb=1, nproc=16)
        result = execute(program, [{"input": "", "output": ""}], "stdin", limits, check_policy=False)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "timeout", result)


class SentinelCannotBeFaked(unittest.TestCase):
    """Assert-style tests: only the harness's nonce-tagged record can produce a pass."""

    tests = "assert add(1, 2) == 3\nassert add(2, 2) == 4\n"

    def run_assert(self, program, check_policy=True):
        return execute(program, self.tests, "assert", SHORT, check_policy=check_policy)

    def test_early_exits(self):
        programs = {
            "sys.exit(0) at import": ("import sys\ndef add(a, b):\n    return 0\nsys.exit(0)\n", "early_exit"),
            "raise SystemExit": ("def add(a, b):\n    return 0\nraise SystemExit(0)\n", "early_exit"),
            "exit() inside the tested function": ("def add(a, b):\n    exit(0)\n", "early_exit"),
        }
        for name, (program, reason) in programs.items():
            with self.subTest(name=name):
                result = self.run_assert(program)
                self.assertFalse(result["passed"], result)
                self.assertEqual(result["reason"], reason, result)

    def test_hard_exits_with_policy_off(self):
        programs = {
            "os._exit(0)": "import os\ndef add(a, b):\n    return 0\nos._exit(0)\n",
            "SIGKILL to itself": "import os, signal\ndef add(a, b):\n    return 0\n"
                                 "os.kill(os.getpid(), signal.SIGKILL)\n",
            "abort": "import os\ndef add(a, b):\n    return 0\nos.abort()\n",
            "kill own process group": "import os, signal\ndef add(a, b):\n    return 0\n"
                                      "os.killpg(0, signal.SIGKILL)\n",
        }
        for name, program in programs.items():
            with self.subTest(name=name):
                result = self.run_assert(program, check_policy=False)
                self.assertFalse(result["passed"], result)

    def test_printed_sentinels_are_ignored(self):
        fake = "0" * 32 + " ok 0 \\n"
        program = ("import sys\ndef add(a, b):\n    return 0\n"
                   "for line in ['ALL TESTS PASSED', 'OK', 'SENTINEL', '" + fake + "']:\n"
                   "    print(line)\n    print(line, file=sys.stderr)\n")
        result = self.run_assert(program)
        self.assertEqual(result["reason"], "test_failed", result)

    def test_forged_records_on_every_fd(self):
        forge = ("import os\nfor fd in range(3, 64):\n    try:\n        os.write(fd, b'" + "0" * 32
                 + " ok 0 forged\\n')\n    except OSError:\n        pass\n")
        wrong_then_exit = "def add(a, b):\n    return 0\n" + forge + "os._exit(0)\n"
        result = self.run_assert(wrong_then_exit, check_policy=False)
        self.assertFalse(result["passed"], result)
        self.assertIn(result["reason"], ("verdict_tampered", "no_verdict"), result)
        # Even a correct program loses its pass if it writes anything on the verdict channel.
        correct_but_forging = "def add(a, b):\n    return a + b\n" + forge
        result = self.run_assert(correct_but_forging, check_policy=False)
        self.assertEqual(result["reason"], "verdict_tampered", result)

    def test_nonce_is_not_readable_from_any_fd(self):
        program = '''
import os
found = []
for fd in range(3, 64):
    try:
        data = os.read(fd, 4096)
    except OSError:
        continue
    found.append((fd, data))
def add(a, b):
    return 0 if found else a + b
'''
        result = self.run_assert(program, check_policy=False)
        self.assertTrue(result["passed"], f"some fd was readable: {result}")

    def test_always_equal_objects(self):
        programs = {
            "__eq__ always true": "class A:\n    def __eq__(self, o):\n        return True\n"
                                  "def add(a, b):\n    return A()\n",
            "int subclass with __eq__": "class I(int):\n    def __eq__(self, o):\n        return True\n"
                                        "    __hash__ = int.__hash__\ndef add(a, b):\n    return I(0)\n",
            "__eq__ inside a list": "class A:\n    def __eq__(self, o):\n        return True\n"
                                    "def add(a, b):\n    return [A()][0]\n",
        }
        for name, program in programs.items():
            with self.subTest(name=name):
                result = self.run_assert(program)
                self.assertEqual(result["reason"], "non_plain_result", result)

    def test_arithmetic_and_truthiness_hacks(self):
        sub_hack = ("class Z:\n    def __sub__(self, o):\n        return 0\n    def __rsub__(self, o):\n"
                    "        return 0\ndef area(r):\n    return Z()\n")
        result = execute(sub_hack, "assert abs(area(1) - 3.14159) < 1e-3\n", "assert", SHORT)
        self.assertEqual(result["reason"], "non_plain_result", result)
        bool_hack = "class T:\n    def __bool__(self):\n        return True\ndef is_prime(n):\n    return T()\n"
        result = execute(bool_hack, "assert is_prime(7)\nassert is_prime(13)\n", "assert", SHORT)
        self.assertEqual(result["reason"], "non_plain_result", result)
        list_hack = ("class A:\n    def __eq__(self, o):\n        return True\ndef f():\n"
                     "    return [A(), A()]\n")
        result = execute(list_hack, "assert f() == [1, 2]\n", "assert", SHORT)
        self.assertEqual(result["reason"], "non_plain_result", result)

    def test_vacuous_tests_are_not_a_pass(self):
        # A HumanEval-style check() that is never called, or a test file without asserts, would pass any program.
        program = "def add(a, b):\n    return 0\n"
        never_called = execute(program, "def check(candidate):\n    assert candidate(1, 2) == 3\n", "assert", SHORT)
        self.assertEqual(never_called["reason"], "bad_tests", never_called)
        no_asserts = execute(program, "x = add(1, 2)\n", "assert", SHORT)
        self.assertEqual(no_asserts["reason"], "bad_tests", no_asserts)
        # Tests guarded by a main check now run: the test namespace is called __main__.
        guarded = execute(program, "if __name__ == '__main__':\n    assert add(1, 2) == 3\n", "assert", SHORT)
        self.assertEqual(guarded["reason"], "test_failed", guarded)

    def test_stdin_exit_after_correct_output_is_a_pass_but_hard_exit_is_not(self):
        cases = [{"input": "", "output": "42"}]
        self.assertTrue(execute("print(42)\nexit()\n", cases, "stdin")["passed"])
        hard = execute("import os, sys\nprint(42)\nsys.stdout.flush()\nos._exit(0)\n", cases, "stdin",
                       check_policy=False)
        self.assertEqual(hard["reason"], "no_verdict", hard)


BOOLY = "class T:\n    def __bool__(self):\n        return True\n"


class TestsSeeOnlyTheirOwnNames(unittest.TestCase):
    """The program cannot change what a name in the tests means, except for the function under test."""

    def run_hack(self, program, tests, **kwargs):
        result = execute(program, tests, "assert", SHORT, **kwargs)
        self.assertFalse(result["passed"], result)
        return result

    def test_shadowed_builtins_are_ignored(self):
        cases = {
            "abs": ("def area(r):\n    return 0.0\ndef abs(y):\n    return 0.0\n",
                    "assert abs(area(5) - 78.5398) < 1e-3\n"),
            "len": ("def f(x):\n    return []\ndef len(o):\n    return 3\n", "assert len(f(1)) == 3\n"),
            "sorted": ("def f(x):\n    return 'junk'\ndef sorted(o):\n    return [1, 2, 3]\n",
                       "assert sorted(f([3, 1, 2])) == [1, 2, 3]\n"),
            "max": ("def f(x):\n    return [1]\ndef max(o):\n    return 9\n", "assert max(f(1)) == 9\n"),
            "set": ("def similar(a, b):\n    return [9]\ndef set(x):\n    return 1\n",
                    "assert set(similar((3, 4), (4, 5))) == set((4,))\n"),
            "isinstance": ("def f():\n    return 'x'\ndef isinstance(a, b):\n    return True\n",
                           "assert isinstance(f(), int)\n"),
        }
        for name, (program, tests) in cases.items():
            with self.subTest(name=name):
                result = self.run_hack(program, tests)
                self.assertIn(result["reason"], ("test_failed", "runtime_error"), result)
                self.assertEqual(result["shadowed_names"], [name], result)
        humaneval = self.run_hack("def candidate(x):\n    return 0.0\ndef abs(y):\n    return 0.0\n",
                                  "def check(candidate):\n    assert abs(candidate(5) - 25.0) < 1e-6\n",
                                  entry_point="candidate")
        self.assertEqual(humaneval["reason"], "test_failed", humaneval)

    def test_shadowed_modules_are_ignored(self):
        program = ("class math:\n    @staticmethod\n    def isclose(a, b, **kwargs):\n        return True\n"
                   "def area(r):\n    return 0.0\n")
        result = self.run_hack(program, "assert math.isclose(area(1), 3.14159)\n")
        self.assertEqual(result["reason"], "test_failed", result)
        self.assertEqual(result["shadowed_names"], ["math"])

    def test_imports_of_the_solution_get_the_same_view(self):
        program = "def f(x):\n    return 0.0\ndef abs(y):\n    return 0.0\n"
        result = self.run_hack(program, "from solution import *\nassert abs(f(5) - 25.0) < 1e-6\n")
        self.assertEqual(result["reason"], "test_failed", result)
        method_hack = BOOLY + "class S:\n    def ok(self):\n        return T()\n"
        result = self.run_hack(method_hack, "import solution\nassert all([solution.S().ok()])\n")
        self.assertEqual(result["reason"], "non_plain_result", result)
        ok = execute("def f(x):\n    return x * x\n", "from solution import f\nassert f(5) == 25\n", "assert", SHORT)
        self.assertTrue(ok["passed"], ok)

    def test_trusted_context_overrides_oracle_helpers(self):
        # HumanEval/32 checks find_zero with poly() from the prompt. Redefining poly must not help.
        context = ("import math\ndef poly(xs, x):\n    return sum(c * math.pow(x, i) for i, c in enumerate(xs))\n"
                   "def find_zero(xs):\n    \"\"\"Docstring only, as in the prompt.\"\"\"\n")
        tests = ("def check(candidate):\n    for coeffs in ([1, 2], [-6, 11, -6, 1]):\n"
                 "        assert abs(poly(coeffs, candidate(list(coeffs)))) < 1e-4\n")
        hack = "def poly(xs, x):\n    return 0.0\ndef find_zero(xs):\n    return 123.0\n"
        result = self.run_hack(hack, tests, entry_point="find_zero", test_context=context)
        self.assertEqual(result["reason"], "test_failed", result)
        correct = context.replace("    \"\"\"Docstring only, as in the prompt.\"\"\"\n", (
            "    lo, hi = -1.0, 1.0\n    while poly(xs, lo) * poly(xs, hi) > 0:\n        lo, hi = 2 * lo, 2 * hi\n"
            "    while hi - lo > 1e-10:\n        mid = (lo + hi) / 2\n"
            "        if poly(xs, mid) * poly(xs, lo) > 0:\n            lo = mid\n        else:\n            hi = mid\n"
            "    return lo\n"))
        ok = execute(correct, tests, "assert", SHORT, entry_point="find_zero", test_context=context)
        self.assertTrue(ok["passed"], ok)

    def test_legitimate_names_still_work(self):
        programs = {
            "entry point named like a builtin": (
                "def sum(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n",
                "def check(candidate):\n    assert candidate([1, 2]) == 3\n", {"entry_point": "sum"}),
            "own abs helper": ("def abs(x):\n    return x if x >= 0 else -x\ndef dist(a, b):\n    return abs(a - b)\n",
                               "def check(candidate):\n    assert abs(candidate(1, 4) - 3) < 1e-9\n",
                               {"entry_point": "dist"}),
            "tests use the program's math import": (
                "import math\ndef area(r):\n    return math.pi * r * r\n",
                "assert math.isclose(area(1), 3.141592653589793)\n", {}),
            "helper the tests call": ("def helper(x):\n    return x + 1\ndef f(x):\n    return helper(x) * 2\n",
                                      "assert f(1) == 4\nassert helper(1) == 2\n", {}),
        }
        for name, (program, tests, kwargs) in programs.items():
            with self.subTest(name=name):
                result = execute(program, tests, "assert", SHORT, **kwargs)
                self.assertTrue(result["passed"], result)


class ValueGuardsCoverEveryPath(unittest.TestCase):
    """Values from the program are checked whichever way they reach the tests."""

    def test_hacks_through_iterators_callables_and_attributes(self):
        hacks = {
            "all() over a returned generator": (BOOLY + "def f(n):\n    return (T() for _ in range(n))\n",
                                                "assert all(f(3))\n"),
            "sum() over a returned map": ("class R:\n    def __radd__(self, o):\n        return 10\n"
                                          "def f(n):\n    return map(lambda i: R(), range(n))\n",
                                          "assert sum(f(3)) == 10\n"),
            "map() over the function under test": (BOOLY + "def is_prime(n):\n    return T()\n",
                                                   "assert all(map(is_prime, [2, 3, 5]))\n"),
            "a returned closure": (BOOLY + "def make():\n    return lambda i: T()\n",
                                   "f = make()\nassert all(f(i) for i in range(3))\n"),
            "a method of a returned object": (BOOLY + "class S:\n    def ok(self, i):\n        return T()\n"
                                              "def make():\n    return S()\n",
                                              "s = make()\nassert all(s.ok(i) for i in range(3))\n"),
            "an attribute of a returned object": ("class R:\n    def __radd__(self, o):\n        return 6\n"
                                                  "class N:\n    def __init__(self):\n        self.val = R()\n"
                                                  "def f():\n    return [N()]\n",
                                                  "assert sum(n.val for n in f()) == 6\n"),
            "a slot of a returned object": (BOOLY + "class N:\n    __slots__ = ('v',)\n    def __init__(self):\n"
                                            "        self.v = T()\ndef f():\n    return N()\n",
                                            "assert all([f().v])\n"),
            "an iterator inside a list": (BOOLY + "def f():\n    return [iter([T()])]\n", "assert all(f()[0])\n"),
            "a registered virtual number": (
                "import numbers\nclass E:\n    def __eq__(s, o):\n        return True\nclass V:\n"
                "    @property\n    def numerator(self):\n        return E()\n    @property\n"
                "    def denominator(self):\n        return E()\nnumbers.Rational.register(V)\n"
                "def f():\n    return V()\n", "from fractions import Fraction\nassert f() == Fraction(1, 3)\n"),
            "a thread still running": ("import threading, time\nthreading.Thread(target=time.sleep, args=(2,), "
                                       "daemon=True).start()\ndef f():\n    return [1]\n", "assert f() == [1]\n"),
        }
        for name, (program, tests) in hacks.items():
            with self.subTest(name=name):
                result = execute(program, tests, "assert", SHORT)
                self.assertEqual(result["reason"], "non_plain_result", result)

    def test_ordinary_values_still_pass(self):
        program = (
            "import datetime\nfrom functools import lru_cache\nfrom typing import List\n\n"
            "@lru_cache(maxsize=None)\ndef fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n\n"
            "def squares(n):\n    for i in range(n):\n        yield i * i\n\n"
            "def adder(k):\n    return lambda x: x + k\n\n"
            "def next_day(d):\n    return d + datetime.timedelta(days=1)\n\n"
            "class ListNode:\n    def __init__(self, val, nxt=None):\n        self.val, self.next = val, nxt\n\n"
            "def reverse(head):\n    prev = None\n    while head:\n"
            "        head.next, prev, head = prev, head, head.next\n    return prev\n\n"
            "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
            "        seen = {}\n        for i, x in enumerate(nums):\n            if target - x in seen:\n"
            "                return [seen[target - x], i]\n            seen[x] = i\n")
        tests = (
            "import datetime\n"
            "class Wrap:\n    def __init__(self, xs):\n        self.xs = xs\n"
            "    def __eq__(self, other):\n        return self.xs == other.xs\n"
            "def to_list(node):\n    out = []\n    while node:\n        out.append(node.val)\n"
            "        node = node.next\n    return out\n"
            "assert fib(30) == 832040 and all(fib(i) >= 0 for i in range(10))\n"
            "assert list(squares(4)) == [0, 1, 4, 9] and sum(squares(4)) == 14\n"
            "assert list(map(adder(1), [1, 2])) == [2, 3]\n"
            "assert next_day(datetime.date(2024, 2, 28)) == datetime.date(2024, 2, 29)\n"
            "assert Wrap(to_list(reverse(ListNode(1, ListNode(2))))) == Wrap([2, 1])\n"
            "assert Solution().twoSum([2, 7, 11, 15], 9) == [0, 1]\n")
        result = execute(program, tests, "assert", SHORT)
        self.assertTrue(result["passed"], result)


class OutputComparison(unittest.TestCase):
    def test_tokens_and_numbers(self):
        self.assertTrue(outputs_match("1 2\n", "1  2"))
        self.assertTrue(outputs_match("1.0\n", "1.000"))
        self.assertTrue(outputs_match("-0", "0"))
        self.assertFalse(outputs_match("1_0", "10"))
        self.assertFalse(outputs_match("Infinity", "inf"))
        self.assertFalse(outputs_match("0.3333333", "0.33333333"))
        self.assertTrue(outputs_match("0.3333333", "0.33333333", float_tolerance=1e-6))
        self.assertTrue(outputs_match("215.913355350494", "215.913355350494384765625", float_tolerance=1e-6))
        # An integer answer stays exact under a tolerance, because the expected token has no decimal point.
        self.assertFalse(outputs_match("1000000.5", "1000000", float_tolerance=1e-6))
        self.assertFalse(outputs_match("3.0000001", "3", float_tolerance=1e-6))
        self.assertFalse(outputs_match("YES", "yes"))
        self.assertFalse(outputs_match("1 2", "1 2 3"))

    def test_numeric_match_in_execute(self):
        result = execute("print(f'{1/4:.4f}')\n", [{"input": "", "output": "0.25"}], "stdin")
        self.assertTrue(result["passed"], result)


class JudgeConventions(unittest.TestCase):
    def test_judge_imports(self):
        program = ("class Solution:\n    def count(self, a: List[int]) -> int:\n"
                   "        return max(Counter(a).values(), default=-inf)\n")
        tests = "assert Solution().count([1, 1, 2]) == 2\n"
        self.assertEqual(execute(program, tests, "assert", SHORT)["reason"], "runtime_error")
        result = execute(program, tests, "assert", SHORT, judge_imports=True)
        self.assertTrue(result["passed"], result)

    def test_for_cases_scales_the_budget(self):
        limits = Limits(time_s=2).for_cases(10, cap_s=12)
        self.assertEqual((limits.time_s, limits.wall_seconds()), (12, 26))


class RuntimeGuards(unittest.TestCase):
    """The guards inside the sandbox enforce the policy even for code the static check never saw."""

    def run_guarded(self, program):
        run = _run_once(program, "stdin", None, b"", SHORT, True, sys.executable, "/tmp")
        return run["record"], run["stdout"].decode()

    def test_denied_paths_raise(self):
        programs = {
            "import subprocess": ("import subprocess\n", "ImportError"),
            "from os import system": ("from os import system\n", "ImportError"),
            "eval": ("eval('1')\n", "NameError"),
            "exec": ("exec('x = 1')\n", "NameError"),
            "open for write": ("open('x.txt', 'w')\n", "PermissionError"),
            "getattr sys.modules": ("import sys\ngetattr(sys, 'modules')\n", "PermissionError"),
            "getattr __class__": ("getattr(1, '__class__')\n", "PermissionError"),
            "setattr module": ("import math\nsetattr(math, 'pi', 3)\n", "PermissionError"),
            "computed getattr": ("import os\ngetattr(os, '_ex' + 'it')(0)\n", "PermissionError"),
        }
        for name, (program, error) in programs.items():
            with self.subTest(name=name):
                record, _ = self.run_guarded(program)
                self.assertIsNotNone(record)
                self.assertEqual(record[0], "exception", record)
                self.assertTrue(record[2].startswith(error), record)

    def test_allowed_paths_work(self):
        program = ("import collections.abc, os.path, sys\nfrom collections import abc\nfrom os import read\n"
                   "print(os.path.exists('nope'), isinstance([], abc.Sequence), getattr(sys, 'maxsize') > 0)\n")
        record, stdout = self.run_guarded(program)
        self.assertEqual(record[0], "ok", record)
        self.assertEqual(stdout.split(), ["False", "True", "True"])


class Pool(unittest.TestCase):
    def jobs(self, n):
        good = "a, b = map(int, input().split())\nprint(a + b)\n"
        bad = "a, b = map(int, input().split())\nprint(a - b)\n"
        cases = [{"input": "2 3", "output": "5"}, {"input": "4 4", "output": "8"}]
        return [{"program": good if i % 2 == 0 else bad, "tests": cases, "kind": "stdin"} for i in range(n)]

    def check_pool(self, processes):
        jobs = self.jobs(64)
        workers = min(32, len(os.sched_getaffinity(0)))
        start = time.monotonic()
        with ExecutorPool(workers=workers, processes=processes) as pool:
            results = pool.map(jobs)
        elapsed = time.monotonic() - start
        self.assertEqual([r["passed"] for r in results], [i % 2 == 0 for i in range(64)])
        self.assertTrue(all(r["reason"] == "wrong_answer" for r in results[1::2]))
        runs = sum(r["tests_passed"] + (0 if r["passed"] else 1) for r in results)
        print(f"\n  pool(processes={processes}, workers={workers}): 64 programs, {runs} sandbox runs in "
              f"{elapsed:.2f}s -> {runs / elapsed:.0f} runs/s")
        self.assertEqual(runner_processes(), [])

    def test_process_pool(self):
        self.check_pool(processes=True)

    def test_thread_pool(self):
        self.check_pool(processes=False)

    def test_submit(self):
        with ExecutorPool(workers=2, processes=False) as pool:
            future = pool.submit("print(7)\n", [{"input": "", "output": "7"}], "stdin")
            self.assertTrue(future.result()["passed"])


if __name__ == "__main__":
    unittest.main()
