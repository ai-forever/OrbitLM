"""Static policy of orbit_verl.code_sandbox: correct competitive-programming idioms pass, and every rule fires
on its target. Pure Python, so these tests run anywhere with PYTHONPATH=<repo>/src.
"""

import os
import subprocess
import sys
import unittest

import orbit_verl
from orbit_verl.code_sandbox import judge_prelude
from orbit_verl.code_sandbox.policy import check_program

# The src directory that holds orbit_verl, for the test that needs a fresh interpreter.
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(orbit_verl.__file__)))

PYRIVAL_FAST_IO = '''
import os
import sys
from io import BytesIO, IOBase

BUFSIZE = 8192


class FastIO(IOBase):
    newlines = 0

    def __init__(self, file):
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None

    def read(self):
        while True:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()

    def readline(self):
        while self.newlines == 0:
            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()

    def flush(self):
        if self.writable:
            os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\\r\\n")

n = int(input())
print(sum(map(int, input().split())))
'''

THREAD_RECURSION = '''
import sys
import threading

sys.setrecursionlimit(1 << 25)
threading.stack_size(1 << 27)


def main():
    n = int(sys.stdin.readline())

    def depth(k):
        return 0 if k == 0 else depth(k - 1) + 1

    print(depth(n))


threading.Thread(target=main).start()
'''

ACCEPTED = {
    "pyrival_fast_io": PYRIVAL_FAST_IO,
    "thread_recursion": THREAD_RECURSION,
    "open_zero": "data = open(0).read().split()\nprint(sum(map(int, data)))\n",
    "open_zero_binary": "data = open(0, 'rb').read()\nprint(len(data))\n",
    "os_read_fstat": "import io, os\ninput = io.BytesIO(os.read(0, os.fstat(0).st_size)).readline\n"
                     "print(int(input()))\n",
    "stdin_buffer": "import sys\ninput = sys.stdin.buffer.readline\nsys.stdout.write(str(int(input())) + '\\n')\n",
    "special_methods": '''
class Node:
    __slots__ = ("key", "value")
    __hash__ = None

    def __init__(self, key, value):
        self.key, self.value = key, value

    def __lt__(self, other):
        return self.key < other.key

    def __eq__(self, other):
        return self.key == other.key

    def __repr__(self):
        return f"{type(self).__name__}({self.key!r})"


class Child(Node):
    def __init__(self, key):
        super().__init__(key, None)
        self.__private = 1
''',
    "main_guard": "def main():\n    print(1)\n\nif __name__ == '__main__':\n    main()\n",
    "algorithms": '''
import heapq, bisect, math, itertools, functools, operator
from collections import deque, Counter, defaultdict, OrderedDict, namedtuple
from fractions import Fraction
from decimal import Decimal, getcontext
getcontext().prec = 50

@functools.lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

P = namedtuple("P", "x y")
pts = sorted([P(1, 2), P(0, 1)], key=operator.attrgetter("x"))
h = [3, 1, 2]
heapq.heapify(h)
print(bisect.bisect_left([1, 2, 3], 2), math.gcd(4, 6), list(itertools.accumulate([1, 2])),
      Counter("aab").most_common(1), deque([1]).popleft(), Fraction(1, 3), Decimal(1) / 7, fib(30), pts)
print(functools.reduce(operator.add, [1, 2, 3]), operator.itemgetter(1)([1, 2]))
''',
    "typing_dataclass": '''
from __future__ import annotations
from typing import List, Optional, Dict
from dataclasses import dataclass, field

@dataclass(order=True)
class Item:
    weight: int
    tags: List[str] = field(default_factory=list)

def solve(items: List[Item]) -> Optional[int]:
    return max((i.weight for i in items), default=None)
''',
    "star_imports": "from math import *\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\n"
                    "from bisect import *\nfrom typing import *\nprint(gcd(2, 4), deque(), list(permutations([1])))\n",
    "formatting": "x = 1.5\nprint('{:.6f}'.format(x), f'{x:.3f}', '%d' % 3, '{0} {1}'.format(1, 2), "
                  "'{p.real}'.format(p=x))\n",
    "sys_knobs": "import sys\nsys.setrecursionlimit(10**6)\nsys.set_int_max_str_digits(0)\n"
                 "print(1, file=sys.stderr)\nsys.stdout.flush()\nsys.exit(0)\n",
    "exit_builtin": "print(1)\nexit()\n",
    "literal_getattr": "class A:\n    pass\na = A()\nsetattr(a, 'x', 1)\nprint(getattr(a, 'x', 0), "
                       "hasattr(a, '__iter__'))\n",
    "os_path_template": "import os, sys\nif os.path.exists('input.txt'):\n    sys.stdin = open('input.txt')\n"
                        "print(input())\n",
    "match_statement": "class Pt:\n    __match_args__ = ('x',)\n    def __init__(self, x):\n        self.x = x\n"
                       "match Pt(1):\n    case Pt(x=1):\n        print('one')\n    case Pt(y):\n        pass\n",
    "string_io_output": "import io, sys\nbuf = io.StringIO()\nbuf.write('a\\n')\nsys.stdout.write(buf.getvalue())\n",
    "wraps_decorator": "import functools\ndef deco(f):\n    @functools.wraps(f)\n    def g(*a):\n"
                       "        return f(*a)\n    return g\n",
    "text_and_time": "import re, json, time, random, string, datetime\nrandom.seed(1)\n"
                     "print(re.findall(r'\\d+', 'a1b22'), json.loads('[1]'), time.time() > 0, "
                     "string.ascii_lowercase[:3], datetime.date(2024, 1, 1).isoformat())\n",
    "collections_abc": "import collections.abc\nfrom collections import abc\n"
                       "print(isinstance([], collections.abc.Sequence), isinstance({}, abc.Mapping))\n",
    "own_name_shadows_builtin": "def compile(s):\n    return s.upper()\nvars = [1, 2]\nprint(compile('x'), vars)\n",
    "threading_lock": "import threading\nlock = threading.Lock()\nwith lock:\n    pass\n",
    "hashlib_md5": "import hashlib\ndef string_to_md5(text):\n    return hashlib.md5(text.encode('ascii')).hexdigest()\n",
    "regex_with_backslashes": "import re\nprint(re.findall('\\d+', 'a1b22'))\n",
}

# (program, rule id that must be reported)
REJECTED = [
    ("import subprocess", "denied_import"),
    ("import socket", "denied_import"),
    ("import ctypes", "denied_import"),
    ("import shutil", "denied_import"),
    ("from importlib import import_module", "denied_import"),
    ("import multiprocessing", "denied_import"),
    ("import pickle", "denied_import"),
    ("import marshal", "denied_import"),
    ("import gc", "denied_import"),
    ("import inspect", "denied_import"),
    ("import builtins", "denied_import"),
    ("import resource", "denied_import"),
    ("import signal", "denied_import"),
    ("import pathlib", "denied_import"),
    ("import numpy", "denied_import"),
    ("import json.decoder", "denied_import"),
    ("from . import helper", "denied_import"),
    ("from os import system", "denied_module_attr"),
    ("import os\nos.system('ls')", "denied_module_attr"),
    ("import os\nos.remove('x')", "denied_module_attr"),
    ("import os\nos.kill(1, 9)", "denied_module_attr"),
    ("import os\nos._exit(0)", "denied_module_attr"),
    ("import os\nos.fork()", "denied_module_attr"),
    ("import os\nprint(os.environ)", "denied_module_attr"),
    ("import os.path\nos.path.realpath('.')", "denied_module_attr"),
    ("import sys\nsys.modules['os']", "denied_module_attr"),
    ("import sys\nsys._getframe()", "denied_module_attr"),
    ("import sys\nsys.settrace(None)", "denied_module_attr"),
    ("import sys\nsys.exc_info()", "denied_module_attr"),
    ("import sys\nprint(sys.argv)", "denied_module_attr"),
    ("import sys as s\ns.path.append('/tmp')", "denied_module_attr"),
    ("import io\nio.open('x')", "denied_module_attr"),
    ("from io import FileIO", "denied_module_attr"),
    ("import threading\nthreading.settrace(None)", "denied_module_attr"),
    ("import threading\nthreading._shutdown()", "denied_module_attr"),
    ("import typing\ntyping.sys.modules", "denied_module_attr"),
    ("import dataclasses\ndataclasses.inspect", "denied_module_attr"),
    ("import random\nrandom._os", "denied_module_attr"),
    ("import typing\ntyping.get_type_hints(int)", "denied_module_attr"),
    ("from functools import singledispatch", "denied_module_attr"),
    ("from typing import *\nget_type_hints(int)", "denied_module_attr"),
    ("import string\nstring.Formatter()", "denied_module_attr"),
    ("import dataclasses\ndataclasses.make_dataclass('X', [])", "denied_module_attr"),
    ("import hashlib\nhashlib.file_digest(open(0, 'rb'), 'md5')", "denied_module_attr"),
    ("from hashlib import _hashlib", "denied_module_attr"),
    ("from os import *", "star_import"),
    ("import sys\nsys.excepthook = print", "module_attr_write"),
    ("import math\nmath.pi = 3", "module_attr_write"),
    ("import math\ndel math.pi", "module_attr_write"),
    ("import math\nsetattr(math, 'pi', 3)", "module_attr_write"),
    ("from fractions import Fraction\nFraction.limit_denominator = None", "imported_object_write"),
    ("import sys\nx = sys", "module_as_value"),
    ("import sys\nprint(sys)", "module_as_value"),
    ("import os\nf = [os][0]", "module_as_value"),
    ("eval('1')", "denied_builtin"),
    ("exec('x = 1')", "denied_builtin"),
    ("compile('1', 'f', 'eval')", "denied_builtin"),
    ("__import__('os')", "denied_builtin"),
    ("breakpoint()", "denied_builtin"),
    ("globals()['x'] = 1", "denied_builtin"),
    ("vars()", "denied_builtin"),
    ("locals()", "denied_builtin"),
    ("help(print)", "denied_builtin"),
    ("f = open", "builtin_as_value"),
    ("g = getattr", "builtin_as_value"),
    ("list(map(getattr, [1], ['real']))", "builtin_as_value"),
    ("from operator import attrgetter\nf = attrgetter", "builtin_as_value"),
    ("open('out.txt', 'w')", "open_mode"),
    ("open('out.txt', 'a')", "open_mode"),
    ("open('out.txt', mode='r+')", "open_mode"),
    ("open('out.txt', 'xb')", "open_mode"),
    ("m = 'w'\nopen('out.txt', m)", "open_mode"),
    ("open('x', opener=None)", "open_mode"),
    ("open('x', **{'mode': 'w'})", "open_mode"),
    ("name = 'real'\ngetattr(1, name)", "dynamic_attr"),
    ("getattr(1, '__class__')", "dynamic_attr"),
    ("class A:\n    pass\nsetattr(A(), 'x' + 'y', 1)", "dynamic_attr"),
    ("class A:\n    pass\ndelattr(A, '__doc__')", "dynamic_attr"),
    ("import operator\noperator.attrgetter('__class__')(1)", "dynamic_attr"),
    ("import operator\nn = 'x'\noperator.attrgetter(n)", "dynamic_attr"),
    ("import operator\noperator.methodcaller('__reduce__')(1)", "dynamic_attr"),
    ("import functools\nfunctools.update_wrapper(1, 2, ('x',))", "dynamic_attr"),
    ("import functools\nfunctools.wraps(len, assigned=('x',))", "dynamic_attr"),
    ("x = 1\nx.__class__", "dunder_attr"),
    ("int.__bases__", "dunder_attr"),
    ("object.__subclasses__()", "dunder_attr"),
    ("def f(): pass\nf.__globals__", "dunder_attr"),
    ("class A: pass\nA().__dict__", "dunder_attr"),
    ("def f(): pass\nf.__code__", "dunder_attr"),
    ("def f(): pass\nf.__closure__", "dunder_attr"),
    ("int.__mro__", "dunder_attr"),
    ("x = 1\nx.__getattribute__('real')", "dunder_attr"),
    ("[].__reduce_ex__(2)", "dunder_attr"),
    ("print.__self__", "dunder_attr"),
    ("class A: pass\nA.__lt__ = lambda a, b: True", "dunder_attr"),
    ("try:\n    1 / 0\nexcept Exception as e:\n    e.__traceback__", "dunder_attr"),
    ("print(__builtins__)", "dunder_name"),
    ("print(__loader__)", "dunder_name"),
    ("print(__spec__)", "dunder_name"),
    ("print(__file__)", "dunder_name"),
    ("__builtins__ = {}", "dunder_name"),
    ("g = (i for i in [1])\ng.gi_frame", "introspection_attr"),
    ("g = (i for i in [1])\ng.gi_code.co_consts", "introspection_attr"),
    ("def f(t):\n    return t.tb_frame.f_back.f_locals", "introspection_attr"),
    ("def f(c):\n    return c.cell_contents", "introspection_attr"),
    ("class C: pass\nmatch 1:\n    case C(gi_frame=f):\n        pass", "introspection_attr"),
    ("'{0.__class__}'.format(1)", "format_traversal"),
    ("'{0.gi_frame}'.format(1)", "format_traversal"),
    ("'{x:{0._y}}'.format(1, x=2)", "format_traversal"),
    ("fmt = '{}'\nfmt.format(1)", "format_traversal"),
    ("str.format('{}', 1)", "format_traversal"),
    ("('{}' * 2).format(1, 2)", "format_traversal"),
    ("_sbx_ret_ = 1", "reserved_name"),
    ("def _sbx_val_(x):\n    return x", "reserved_name"),
    ("import math as _sbx_m", "reserved_name"),
    ("x = 'f_locals'", "protected_name"),
    ("x = '__globals__'", "protected_name"),
    ("dict(gi_frame=1)", "protected_name"),
    ("def f(:\n    pass", "syntax_error"),
    ("x = '\0'", "syntax_error"),
    ("x = 1\n" * 40000, "too_large"),
]


class PolicyAcceptsCorrectSolutions(unittest.TestCase):
    def test_accepted_programs(self):
        for name, program in ACCEPTED.items():
            with self.subTest(name=name):
                ok, reasons = check_program(program)
                self.assertTrue(ok, f"{name} was rejected: {reasons}")


class PolicyRejectsEachRule(unittest.TestCase):
    def test_each_rule_fires(self):
        for program, rule in REJECTED:
            with self.subTest(program=program[:60]):
                ok, reasons = check_program(program)
                self.assertFalse(ok, f"accepted: {program!r}")
                rules = {r.split(":", 1)[0] for r in reasons}
                self.assertIn(rule, rules, f"{program!r} gave {reasons}")

    def test_every_rule_id_is_covered(self):
        expected = {rule for _, rule in REJECTED}
        documented = {
            "denied_import", "denied_module_attr", "star_import", "module_attr_write", "imported_object_write",
            "module_as_value", "denied_builtin", "builtin_as_value", "open_mode", "dynamic_attr", "dunder_attr",
            "dunder_name", "introspection_attr", "format_traversal", "reserved_name", "protected_name",
            "syntax_error", "too_large",
        }
        self.assertEqual(expected, documented)

    def test_reasons_carry_rule_and_line(self):
        ok, reasons = check_program("x = 1\nimport socket\n")
        self.assertFalse(ok)
        self.assertEqual(reasons, ["denied_import: line 2: import of 'socket' is not allowed"])

    def test_parse_warnings_do_not_reach_the_trainer(self):
        # Regexes such as "\d" in plain strings make Python 3.12 warn at parse time. Thousands of rollouts per
        # step would flood the trainer's stderr, so check_program must stay silent. A fresh interpreter is used,
        # because the warnings registry of this one may already have seen the warning.
        code = ("from orbit_verl.code_sandbox.policy import check_program\n"
                "ok, _ = check_program('import re\\nx = re.findall(\"\\\\d+\", \"a1\")\\n')\n"
                "assert ok\n")
        env = dict(os.environ, PYTHONPATH=SRC_DIR)
        done = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stderr, "")

    def test_judge_prelude_passes_the_policy(self):
        program = ("class Solution:\n    def f(self, a: List[int]) -> int:\n        c = Counter(a)\n"
                   "        return max(c.values(), default=-inf) + pow(2, 3) + reduce(add, a, 0)\n")
        prelude = judge_prelude(program)
        self.assertIn("from typing import List", prelude)
        self.assertIn("from collections import Counter", prelude)
        self.assertNotIn("pow", prelude)  # A builtin is never replaced by math.pow.
        ok, reasons = check_program(prelude + program)
        self.assertTrue(ok, reasons)

    def test_judge_prelude_covers_signatures_of_late_imports(self):
        # LCB 3411 (GPT-4O): the annotation needs List when the def runs, before the import inside the method.
        program = ("class Solution:\n    def f(self, q: List[List[int]]) -> List[int]:\n"
                   "        from typing import List\n        return [len(x) for x in q]\n")
        self.assertIn("from typing import List", judge_prelude(program))
        # A name the program defines at module level is still left to the program.
        self.assertNotIn("Counter", judge_prelude("from collections import Counter\ndef f(c: Counter): pass\n"))

    def test_judge_prelude_follows_python_scopes(self):
        # LCB 3384 (MetaStone-L1-7B): a class-body import is invisible to the methods, which read module scope.
        program = ("class Solution:\n    from collections import defaultdict\n"
                   "    def f(self, a):\n        d = defaultdict(int)\n        return len(d)\n")
        self.assertIn("from collections import defaultdict", judge_prelude(program))
        # A local variable named like a math or statistics export is not a module-scope read.
        program = "def g(xs):\n    mode = max(xs)\n    e = 1\n    return mode + e\n"
        self.assertEqual(judge_prelude(program).count("\n"), 2)

    def test_deep_nesting_does_not_crash_the_checker(self):
        program = "x = " + "[" * 150 + "]" * 150 + "\n"
        ok, reasons = check_program(program)
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(reasons, list)


if __name__ == "__main__":
    unittest.main()
