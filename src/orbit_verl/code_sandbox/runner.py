"""In-sandbox half of the executor: confines this process, runs one program, and reports a verdict.

The parent (executor.py) starts this script as

    unshare --user --map-current-user --pid --fork --kill-child -- \\
        python -I -S -B -X utf8 runner.py RUN_DIR NONCE_FD VERDICT_FD

so it already runs in fresh user and pid namespaces: it is pid 1 of its own namespace, sees no other
processes and cannot signal them. (A network namespace is optional, see Limits.network_namespace; the
seccomp filter below already makes sockets impossible.) Before any untrusted code runs, this script

  1. reads a per-run nonce from NONCE_FD straight into malloc'd memory, so the nonce never exists as a Python
     object while the program runs;
  2. reads RUN_DIR/spec.marshal (program, tests, limits);
  3. applies Landlock (read-only system and Python library paths, nothing else; no TCP), a seccomp filter
     (no sockets, ptrace, mounts, namespaces, bpf, ...), and rlimits (CPU, address space, file size,
     processes, open files, no core dumps), and clears the environment.

It then runs the program (and, in assert mode, the tests) and writes exactly one record to VERDICT_FD:

    <nonce> <status> <exit code> <shadowed names or -> <detail>\\n

"Shadowed names" are names such as abs or math that the program defined and the tests read. The tests got the
real builtin or module instead (see build_test_namespace).

The parent trusts only a record that carries its nonce. Stdout, the exit code and anything else the program can
produce are never treated as the verdict, so exiting early or printing a fake success line cannot pass. The
program could write to VERDICT_FD, but without the nonce its bytes only make the parent reject the run.

Keep this file standard-library only and self-contained apart from rules.py, which is imported from this
directory before the sandbox closes. Imports are kept to the minimum on the common path and the rest are
imported only when needed: every sandbox start pays for each module, about a millisecond apiece when the
interpreter lives on a shared filesystem.
"""

import _signal
import builtins
import ctypes
import marshal
import os
import resource
import struct
import sys

_ModuleType = type(sys)

_HERE = os.path.dirname(os.path.abspath(__file__))

# Exit code for "the sandbox could not be set up". The parent reports it as sandbox_error, never as a pass.
SETUP_FAILED_EXIT = 70

NONCE_LEN = 32

# Landlock filesystem rights (linux/landlock.h). The ABI version decides which ones the kernel knows.
LL_EXECUTE, LL_WRITE_FILE, LL_READ_FILE, LL_READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
LL_REFER, LL_TRUNCATE, LL_IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
LL_FILE_ONLY = LL_EXECUTE | LL_WRITE_FILE | LL_READ_FILE | LL_TRUNCATE | LL_IOCTL_DEV
LL_READ_TREE = LL_EXECUTE | LL_READ_FILE | LL_READ_DIR

# Syscalls the program never needs. Denying socket() and socketpair() is what isolates the network: no
# socket of any family can exist in the sandbox. It also closes the gap Landlock ABI 4 leaves open, which is
# connecting to pathname UNIX sockets such as Ray's under /tmp/ray. x86_64 numbers.
_SECCOMP_DENY_X86_64 = (
    41,   # socket
    53,   # socketpair
    101,  # ptrace
    103,  # syslog
    155,  # pivot_root
    161,  # chroot
    163,  # acct
    165, 166,  # mount, umount2
    167, 168, 169,  # swapon, swapoff, reboot
    175, 176, 313,  # init_module, delete_module, finit_module
    179,  # quotactl
    246, 320,  # kexec_load, kexec_file_load
    248, 249, 250,  # add_key, request_key, keyctl
    272, 308,  # unshare, setns
    298,  # perf_event_open
    300,  # fanotify_init
    303, 304,  # name_to_handle_at, open_by_handle_at
    310, 311,  # process_vm_readv, process_vm_writev
    321,  # bpf
    323,  # userfaultfd
    425, 426, 427,  # io_uring_setup, io_uring_enter, io_uring_register
    428, 429, 430, 431, 432, 433, 442,  # open_tree, move_mount, fsopen, fsconfig, fsmount, fspick, mount_setattr
    438, 440,  # pidfd_getfd, process_madvise
)


# ----------------------------------------------------------------------------------------------------------
# Confinement
# ----------------------------------------------------------------------------------------------------------

def _syscall(libc):
    fn = libc.syscall
    fn.restype = ctypes.c_long
    return fn


def read_nonce(libc, fd):
    """Reads the nonce from fd into malloc'd memory and returns only its address.

    The nonce stays outside the Python heap, so reprs, frames or format strings cannot reveal it. Reading it
    back needs ctypes or /proc/self/mem, and the program has neither.
    """
    libc.malloc.restype = ctypes.c_void_p
    libc.read.restype = ctypes.c_ssize_t
    addr = libc.malloc(ctypes.c_size_t(NONCE_LEN))
    if not addr:
        raise MemoryError("malloc failed")
    got = 0
    while got < NONCE_LEN:
        n = libc.read(ctypes.c_int(fd), ctypes.c_void_p(addr + got), ctypes.c_size_t(NONCE_LEN - got))
        if n <= 0:
            raise OSError("short read on the nonce pipe")
        got += n
    os.close(fd)
    return addr


def apply_landlock(libc, ro_dirs, ro_files, rw_files, work_dir, writable_work_dir):
    """Restricts this process (and every child) to the listed paths. Returns the Landlock ABI version.

    Everything not listed is denied: home and data directories, /proc, /dev (except /dev/null), /tmp. On ABI 4+
    TCP bind and connect are denied as well. Fails if Landlock is unavailable, because the parent must not run
    untrusted code without it.
    """
    create, add_rule, restrict = 444, 445, 446  # Same numbers on x86_64 and aarch64.
    syscall = _syscall(libc)
    abi = syscall(ctypes.c_long(create), None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    if abi < 1:
        raise OSError(ctypes.get_errno(), "Landlock is not available")
    fs_all = (1 << 13) - 1
    if abi >= 2:
        fs_all |= LL_REFER
    if abi >= 3:
        fs_all |= LL_TRUNCATE
    if abi >= 5:
        fs_all |= LL_IOCTL_DEV
    if abi >= 6:
        attr = struct.pack("QQQ", fs_all, 3, 3)  # Also scope abstract UNIX sockets and signals.
    elif abi >= 4:
        attr = struct.pack("QQ", fs_all, 3)  # Deny TCP bind and connect: no rules grant them.
    else:
        attr = struct.pack("Q", fs_all)
    attr_buf = ctypes.create_string_buffer(attr, len(attr))
    ruleset = syscall(ctypes.c_long(create), attr_buf, ctypes.c_size_t(len(attr)), ctypes.c_uint32(0))
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

    def allow(path, access):
        try:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except FileNotFoundError:
            return
        try:
            if not os.path.isdir(path):
                access &= LL_FILE_ONLY
            access &= fs_all
            rule = ctypes.create_string_buffer(struct.pack("=Qi", access, fd), 12)
            if syscall(ctypes.c_long(add_rule), ctypes.c_long(ruleset), ctypes.c_long(1), rule,
                       ctypes.c_uint32(0)) != 0:
                raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
        finally:
            os.close(fd)

    try:
        for path in ro_dirs:
            allow(path, LL_READ_TREE)
        for path in ro_files:
            allow(path, LL_READ_FILE)
        for path in rw_files:
            allow(path, LL_READ_FILE | LL_WRITE_FILE | LL_TRUNCATE)
        allow(work_dir, fs_all if writable_work_dir else LL_READ_TREE)
        if libc.prctl(ctypes.c_int(38), ctypes.c_ulong(1), ctypes.c_ulong(0), ctypes.c_ulong(0),
                      ctypes.c_ulong(0)) != 0:  # PR_SET_NO_NEW_PRIVS, required by restrict_self.
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
        if syscall(ctypes.c_long(restrict), ctypes.c_long(ruleset), ctypes.c_uint32(0)) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset)
    return abi


def apply_seccomp(libc):
    """Installs a deny-list seccomp filter: listed syscalls fail with EPERM, other architectures are killed."""
    if os.uname().machine != "x86_64":
        raise OSError("the seccomp deny list is only defined for x86_64")

    class SockFilter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte),
                    ("k", ctypes.c_uint32)]

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]

    ld_abs, jeq, jge, ret = 0x20, 0x15, 0x35, 0x06
    allow, deny_eperm, kill = 0x7FFF0000, 0x00050000 | 1, 0x80000000
    deny = _SECCOMP_DENY_X86_64
    deny_index = 6 + len(deny)
    prog = [
        (ld_abs, 0, 0, 4),                 # Load seccomp_data.arch.
        (jeq, 1, 0, 0xC000003E),           # x86_64 continues; i386 (int 0x80) and others are killed.
        (ret, 0, 0, kill),
        (ld_abs, 0, 0, 0),                 # Load seccomp_data.nr.
        (jge, deny_index - 5, 0, 0x40000000),  # x32 syscall numbers are denied.
    ]
    for i, nr in enumerate(deny):
        prog.append((jeq, deny_index - (6 + i), 0, nr))
    prog += [(ret, 0, 0, allow), (ret, 0, 0, deny_eperm)]
    filters = (SockFilter * len(prog))(*[SockFilter(*ins) for ins in prog])
    fprog = SockFprog(len(prog), filters)
    if libc.prctl(ctypes.c_int(22), ctypes.c_ulong(2), ctypes.byref(fprog), ctypes.c_ulong(0),
                  ctypes.c_ulong(0)) != 0:  # PR_SET_SECCOMP, SECCOMP_MODE_FILTER
        raise OSError(ctypes.get_errno(), "installing the seccomp filter failed")


def apply_rlimits(limits):
    def cap(which, value):
        resource.setrlimit(which, (value, value))

    cap(resource.RLIMIT_CORE, 0)
    # Soft and hard CPU limits are equal on purpose. The kernel then sends SIGKILL at the limit; a SIGXCPU
    # would be ignored, because this process is pid 1 of its namespace.
    cap(resource.RLIMIT_CPU, max(1, int(limits["cpu_seconds"])))
    cap(resource.RLIMIT_AS, int(limits["memory_bytes"]))
    cap(resource.RLIMIT_FSIZE, int(limits["fsize_bytes"]))
    cap(resource.RLIMIT_NPROC, int(limits["nproc"]))
    cap(resource.RLIMIT_NOFILE, int(limits["nofile"]))
    # With SIGXFSZ ignored, writing past the file-size limit fails with EFBIG instead of killing the process,
    # so the run ends with a readable error and the parent can report output_limit.
    _signal.signal(_signal.SIGXFSZ, _signal.SIG_IGN)


def python_lib_dirs():
    """The directories this interpreter loads its standard library and shared libraries from."""
    dirs = []
    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix}:
        dirs.append(os.path.join(prefix, "lib"))
    return dirs


# ----------------------------------------------------------------------------------------------------------
# Runtime guards: the tables in rules.py, enforced on dynamic paths the static check cannot see
# ----------------------------------------------------------------------------------------------------------

def make_program_builtins(rules):
    """Builds the builtins namespace for the program: denied builtins removed, risky ones wrapped."""
    real_import = builtins.__import__
    real_getattr, real_setattr, real_delattr, real_hasattr = getattr, setattr, delattr, hasattr
    real_open = open
    module_type = _ModuleType
    allowed_modules = rules.ALLOWED_MODULES
    module_attr_problem = rules.module_attr_problem
    attribute_name_problem = rules.attribute_name_problem
    canonical = rules.canonical_module_name

    def check(obj, name, store):
        if type(name) is not str:
            raise TypeError("attribute name must be a string")
        if isinstance(obj, module_type):
            problem = module_attr_problem(canonical(obj), name, store)
        else:
            problem = attribute_name_problem(name, store)
        if problem:
            raise PermissionError(problem)

    def guarded_getattr(obj, name, *default):
        check(obj, name, False)
        return real_getattr(obj, name, *default)

    def guarded_hasattr(obj, name):
        if type(name) is not str:
            raise TypeError("attribute name must be a string")
        return real_hasattr(obj, name)

    def guarded_setattr(obj, name, value):
        check(obj, name, True)
        return real_setattr(obj, name, value)

    def guarded_delattr(obj, name):
        check(obj, name, True)
        return real_delattr(obj, name)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or type(name) is not str or name not in allowed_modules:
            raise ImportError(f"import of {name!r} is not allowed")
        module = real_import(name, globals, locals, fromlist, level)
        if fromlist:
            for item in fromlist:
                if item == "*":
                    if isinstance(allowed_modules[name], frozenset):
                        raise ImportError(f"'from {name} import *' is not allowed")
                    continue
                problem = module_attr_problem(name, item) if type(item) is str else "bad import name"
                if problem:
                    raise ImportError(problem)
        return module

    def guarded_open(file, mode="r", *args, **kwargs):
        opener = args[5] if len(args) > 5 else kwargs.get("opener")
        if type(mode) is not str or any(ch in mode for ch in "wax+") or opener is not None:
            raise PermissionError("open() is read-only in the sandbox")
        return real_open(file, mode, *args, **kwargs)

    def exit_(code=None):
        raise SystemExit(code)

    namespace = dict(builtins.__dict__)
    for name in rules.DENIED_BUILTINS | {"__loader__", "__spec__", "copyright", "credits", "license"}:
        namespace.pop(name, None)
    namespace.update({
        "__import__": guarded_import, "open": guarded_open, "getattr": guarded_getattr,
        "setattr": guarded_setattr, "delattr": guarded_delattr, "hasattr": guarded_hasattr,
        "exit": exit_, "quit": exit_,
    })
    return namespace


class NotPlainValue(BaseException):
    """A value reached the tests in a form that could change how a check in the tests behaves.

    It derives from BaseException so that "except Exception" in test code does not swallow it.
    """


def make_value_guards():
    """Returns the helpers the instrumented tests use on every value that comes from the program.

    The classic reward hack for assert-style tests is to return an object whose __eq__ always answers True, or
    whose __sub__ returns 0 so that abs(result - expected) < eps holds. The guards accept only values whose
    behaviour in a test comes from trusted code: builtin scalars and containers, a few standard-library types,
    classes the tests define themselves, and program classes that override none of the methods a comparison, a
    truth test or arithmetic would call. Containers, and the attributes of program objects, are checked item by
    item. Types are read through type.__dict__ and C-level descriptors, so no method of a program object runs
    inside a check.

    The result is a dict of the names the instrumented tests call:
      _sbx_ret_(callee, value)  every value a call hands to the tests. A returned function is wrapped so that its
                                own results are checked too, and a returned iterator is wrapped so that each item
                                is checked when the tests consume it. Anything else is checked at once.
      _sbx_val_(value)          each operand of a comparison in an assert, and each bare asserted expression.
      _sbx_hit_()               counts the assert statements that ran. A test file that asserts nothing passes
                                any program, so the runner reports it instead of passing it.
      _sbx_trust_(cls)          marks a class the tests (or the trusted context) define. Its instances keep
                                their own comparison methods, because those come from the test author.
    plus "hits", a function that returns the count.
    """
    import collections
    import decimal
    import fractions
    import functools
    import gc
    import itertools
    import types

    modules = sys.modules
    # Builtin functions and method descriptors compare by identity, and calling one runs no program code.
    atoms = frozenset({int, float, complex, bool, str, bytes, bytearray, type(None), type(Ellipsis), range,
                       slice, fractions.Fraction, decimal.Decimal, types.BuiltinFunctionType,
                       types.MethodWrapperType, types.WrapperDescriptorType, types.MethodDescriptorType,
                       types.ClassMethodDescriptorType})
    sequences = frozenset({list, tuple, collections.deque})
    sets = frozenset({set, frozenset})
    mappings = frozenset({dict, collections.Counter, collections.OrderedDict, collections.defaultdict})
    views = frozenset({type({}.keys()), type({}.values()), type({}.items())})
    iterators = frozenset({
        types.GeneratorType, map, filter, zip, enumerate, reversed, type(iter([])), type(iter(())),
        type(iter(range(1))), type(iter("")), type(iter(b"")), type(iter({})), type(iter(set())),
        type(iter({}.values())), type(iter({}.items())), type(reversed([])),
        itertools.chain, itertools.count, itertools.cycle, itertools.repeat, itertools.islice,
        itertools.accumulate, itertools.product, itertools.permutations, itertools.combinations,
        itertools.combinations_with_replacement, itertools.groupby, itertools.starmap,
        itertools.zip_longest, itertools.takewhile, itertools.dropwhile, itertools.filterfalse,
        itertools.compress, itertools.pairwise,
    })
    # Callables whose results the guards can intercept by wrapping them. Instances of program classes that define
    # __call__ are not wrapped, because a wrapper would hide their other methods; they are checked as values.
    function_like = frozenset({types.FunctionType, types.MethodType, functools.partial,
                               functools._lru_cache_wrapper})
    trusted = atoms | sequences | sets | mappings | views | iterators | {object}
    # The checks look types up by id. "t in atoms" would hash t, and a program metaclass with its own __hash__
    # would then run program code inside the check. These standard types live as long as the interpreter, so
    # their ids stay valid.
    atom_ids = frozenset(map(id, atoms))
    container_ids = frozenset(map(id, sequences | sets | views))
    mapping_ids = frozenset(map(id, mappings))
    iterator_ids = frozenset(map(id, iterators))
    function_ids = frozenset(map(id, function_like))
    trusted_ids = frozenset(map(id, trusted))
    # Methods that a comparison, a truth test, a conversion or arithmetic in the tests would call on a program
    # object. numerator, denominator, real and imag are here because Fraction and Decimal read them from the other
    # operand when they compare themselves with a registered "virtual" number.
    unsafe_names = frozenset(
        {f"__{name}__" for name in (
            "eq ne lt le gt ge bool len contains iter next getitem float int index complex abs round trunc "
            "floor ceil neg pos invert getattr getattribute class instancecheck subclasscheck"
        ).split()}
        | {f"__{prefix}{op}__" for op in
           "add sub mul matmul truediv floordiv mod divmod pow lshift rshift and xor or".split()
           for prefix in ("", "r", "i")}
        | {"numerator", "denominator", "real", "imag"}
    )
    get_mro = type.__dict__["__mro__"].__get__
    get_dict = type.__dict__["__dict__"].__get__
    get_name = type.__dict__["__name__"].__get__
    get_flags = type.__dict__["__flags__"].__get__
    heap_type_flag = 1 << 9
    getset_type = type(type.__dict__["__dict__"])
    member_type = type(type.__dict__["__dictoffset__"])
    generic_getattr = object.__getattribute__
    # Classes are looked up by id, so that no __hash__ or __eq__ of a program metaclass runs. The list keeps the
    # classes alive, so an id cannot be reused by another object while the tests run.
    test_class_ids = set()
    test_classes = []
    class_cache = {}
    assert_count = [0]

    def stdlib_atoms():
        """Immutable standard-library types whose comparisons are native, if the program or tests loaded them."""
        found = set()
        dt = modules.get("datetime")
        if dt is not None:
            found.update(map(id, (dt.date, dt.datetime, dt.time, dt.timedelta, dt.timezone)))
        arr = modules.get("array")
        if arr is not None:
            found.add(id(arr.array))
        return found

    def class_ok(cls):
        if id(cls) in test_class_ids:
            return True
        if type(cls) is not type:
            return False  # A custom metaclass could change isinstance or comparisons of the class itself.
        verdict = class_cache.get(cls)
        if verdict is None:
            verdict = True
            extra = None
            for klass in get_mro(cls):
                if id(klass) in trusted_ids or id(klass) in test_class_ids:
                    continue
                if extra is None:
                    extra = stdlib_atoms()
                if id(klass) in extra:
                    continue
                namespace = get_dict(klass)
                if any(type(key) is str and key in unsafe_names for key in namespace):
                    verdict = False
                    break
                # A class that replaces its __dict__ descriptor could run code when the guard reads attributes.
                descriptor = namespace.get("__dict__")
                if descriptor is not None and type(descriptor) is not getset_type:
                    verdict = False
                    break
            class_cache[cls] = verdict
        return verdict

    def attribute_values(obj, cls):
        """The attribute values of a Python-level object: its instance dict and its __slots__ members."""
        if not get_flags(cls) & heap_type_flag:
            return ()  # Builtin C types expose only their own fields, which the tests cannot be fooled by.
        values = []
        try:
            instance_dict = generic_getattr(obj, "__dict__")
        except (AttributeError, TypeError):
            instance_dict = None
        if type(instance_dict) is dict:
            values.extend(dict.values(instance_dict))
        for klass in get_mro(cls):
            if id(klass) in trusted_ids or not get_flags(klass) & heap_type_flag:
                continue
            for item in get_dict(klass).values():
                if type(item) is member_type:
                    try:
                        values.append(item.__get__(obj, cls))
                    except AttributeError:
                        pass  # A slot that was never assigned.
        return values

    def walk(value):
        # A running program thread could swap an item after it was checked, so values are only accepted while
        # no program thread runs. Only threads made with threading can exist: _thread is not importable.
        threading = modules.get("threading")
        if threading is not None and threading.active_count() > 1:
            raise NotPlainValue("a program thread was still running while the tests checked a value")
        stack = [value]
        seen = set()
        budget = 10_000_000
        while stack:
            v = stack.pop()
            budget -= 1
            if budget < 0:
                raise NotPlainValue("the value is too large to check")
            t = type(v)
            if id(t) in atom_ids or t is type:
                continue
            if id(t) in iterator_ids:
                if v is value:
                    continue  # ret() wraps a top-level iterator, so each item is checked when it is consumed.
                raise NotPlainValue("an iterator nested inside a value cannot be checked")
            if id(v) in seen:
                continue
            if id(t) in container_ids:
                seen.add(id(v))
                stack.extend(v)
                continue
            if id(t) in mapping_ids:
                seen.add(id(v))
                stack.extend(dict.keys(v))
                stack.extend(dict.values(v))
                continue
            if not class_ok(t):
                raise NotPlainValue(
                    f"a value of type '{get_name(t)}' overrides comparison, arithmetic or container methods")
            seen.add(id(v))
            # class_ok has ruled out overridden __iter__ and __class__, so these use the builtin implementations.
            if issubclass(t, dict):
                stack.extend(dict.keys(v))
                stack.extend(dict.values(v))
            elif issubclass(t, (list, tuple, set, frozenset, collections.deque)):
                stack.extend(v)
            if id(t) not in test_class_ids:
                stack.extend(attribute_values(v, t))

    def check(value):
        # The cyclic garbage collector can run __del__ methods of program objects at any allocation. Pausing it
        # keeps program code from running, and changing the value, in the middle of a check.
        enabled = gc.isenabled()
        gc.disable()
        try:
            walk(value)
        finally:
            if enabled:
                gc.enable()

    def checked_items(iterator):
        for item in iterator:
            yield ret(None, item)

    def checked_calls(function):
        def call(*args, **kwargs):
            return ret(None, function(*args, **kwargs))
        return call

    def ret(callee, value):
        t = type(value)
        if callee is not None and t is callee:
            return value  # The tests constructed one of the program's own classes (Pair(1, 2), Solution()).
        if id(t) in iterator_ids:
            return checked_items(value)
        if id(t) in function_ids:
            return checked_calls(value)
        check(value)
        return value

    def val(value):
        check(value)
        return value

    def hit():
        assert_count[0] += 1

    def trust(cls):
        if issubclass(type(cls), type):
            test_class_ids.add(id(cls))
            test_classes.append(cls)
        return cls

    return {"_sbx_ret_": ret, "_sbx_val_": val, "_sbx_hit_": hit, "_sbx_trust_": trust,
            "hits": lambda: assert_count[0]}


def analyse_tests(tree, program_module):
    """Returns (loaded, defined): every name the tests read, and the names they define with def, class or import.

    Names the tests import from the program's own module ("from solution import f") are program code, so they
    are not counted as defined by the tests.
    """
    import ast

    loaded, defined = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != program_module:
                    defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module != program_module:
                defined.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
    return loaded, defined


def make_program_view(program_globals, entry_point, ret, rules):
    """A module object that hands the tests the program's names the same way the test namespace does.

    Some test files import the program ("from solution import f" or "from solution import *"). They get this
    view instead of the program's module. It resolves names lazily, on first import, with the rules of
    build_test_namespace: builtin and module names are never taken from the program (except the entry point),
    functions come wrapped and other values are checked. A star import gets only the public names that pass
    the checks, so an unused odd global cannot fail the run.
    """
    view = _ModuleType("solution")
    builtin_names = set(builtins.__dict__)
    module_names = rules.MODULE_NAMES

    def resolve(name):
        if type(name) is not str or name.startswith("__") or name not in program_globals:
            raise AttributeError(f"module 'solution' has no attribute {name!r}")
        if name != entry_point:
            if name in module_names:
                return sys.modules.get(name) or builtins.__import__(name)
            if name in builtin_names:
                raise AttributeError(f"the tests use the real builtin {name!r}, not the program's")
        value = program_globals[name]
        return value if issubclass(type(value), type) else ret(None, value)

    def module_getattr(name):
        if name == "__all__":
            public = []
            for key in list(program_globals):
                if type(key) is str and not key.startswith("_") and key not in module_names and (
                        key not in builtin_names or key == entry_point):
                    try:
                        resolve(key)
                    except NotPlainValue:
                        continue
                    public.append(key)
            return public
        return resolve(name)

    view.__dict__["__getattr__"] = module_getattr
    return view


def build_test_namespace(program_globals, loaded, entry_point, context_code, guards, rules):
    """Builds the globals the tests run in. Returns (namespace, shadowed, trusted_roots).

    The tests see only what they are entitled to from the program: the entry point and the other names they
    read that the program defines. Everything else resolves the way the test author meant it:
      * builtin names (abs, len, sorted, ...) always mean the real builtins, even if the program defines a
        function of the same name. Otherwise "def abs(x): return 0" would make abs(f(x) - y) < eps pass;
      * module names (math, re, ...) always mean the real modules;
      * names the trusted context defines (the helpers in a HumanEval prompt, such as poly) come from the context.
    Only the entry point may take a builtin's or a module's name. "shadowed" lists the names the program tried to
    give the tests and did not, for the metrics. trusted_roots are the names whose calls need no result check.
    """
    namespace = {"__builtins__": builtins, "__name__": "__main__"}
    namespace.update((name, helper) for name, helper in guards.items() if name.startswith("_sbx_"))
    if context_code is not None:
        exec(context_code, namespace)
        for value in list(namespace.values()):
            if issubclass(type(value), type):
                guards["_sbx_trust_"](value)
    context_names = set(namespace)
    builtin_names = set(builtins.__dict__)
    module_names = rules.MODULE_NAMES
    ret = guards["_sbx_ret_"]
    shadowed = []
    from_program = set()
    wanted = set(loaded)
    if entry_point:
        wanted.add(entry_point)
    for name in sorted(wanted):
        if name not in program_globals or name.startswith("__"):
            continue
        value = program_globals[name]
        if name != entry_point:
            if name in context_names:
                continue
            if name in module_names:
                real = sys.modules.get(name) or builtins.__import__(name)
                if value is not real:
                    shadowed.append(name)
                namespace[name] = real
                continue
            if name in builtin_names:
                shadowed.append(name)
                continue
        # Classes are handed over as they are, so the tests can construct them and call isinstance with them.
        namespace[name] = value if issubclass(type(value), type) else ret(None, value)
        from_program.add(name)
    trusted_roots = (context_names | builtin_names | (module_names & set(namespace))) - from_program
    # The harness's own helpers are never trusted roots: _sbx_ret_(None, f(x)).g() must still check g's result.
    trusted_roots = {name for name in trusted_roots if not name.startswith(rules.RESERVED_PREFIX)}
    return namespace, shadowed, trusted_roots


def instrument_tests(tree, trusted_roots):
    """Rewrites the parsed tests so that every value coming from the program passes the value guards.

      * Every call whose callee is not trusted becomes _sbx_ret_(f, f(...)), so its result is checked as soon as
        it is returned. Trusted callees are the builtins, the test's own functions, classes and imports, and the
        trusted context. Everything else may be program code: program functions, methods of program objects, and
        functions the program returned.
      * Inside every assert, each operand of a comparison and any bare tested expression becomes _sbx_val_(...),
        and a call to _sbx_hit_() runs before the assert, to count it.
      * Every class the tests define gets the _sbx_trust_ decorator, so the test author's own comparison methods
        are accepted.
    Line numbers are kept, so a failing assert is reported with its original line.
    """
    import ast

    def call(name, node, *args):
        return ast.copy_location(
            ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[]), node)

    def root_name(expr):
        while isinstance(expr, (ast.Attribute, ast.Call, ast.Subscript)):
            expr = expr.func if isinstance(expr, ast.Call) else expr.value
        return expr.id if isinstance(expr, ast.Name) else None

    class Instrumenter(ast.NodeTransformer):
        def visit_ClassDef(self, node):
            self.generic_visit(node)
            node.decorator_list.insert(0, ast.copy_location(ast.Name(id="_sbx_trust_", ctx=ast.Load()), node))
            return node

        def visit_Call(self, node):
            # The root is read before the children are rewritten. Afterwards the root of f(x).g() would be the
            # harness's own _sbx_ret_, and the outer call would wrongly count as trusted.
            root = root_name(node.func)
            self.generic_visit(node)
            if root in trusted_roots:
                return node
            if isinstance(node.func, ast.Name):
                callee = ast.copy_location(ast.Name(id=node.func.id, ctx=ast.Load()), node)
            else:
                callee = ast.copy_location(ast.Constant(value=None), node)
            return call("_sbx_ret_", node, callee, node)

        def visit_Assert(self, node):
            self.generic_visit(node)
            node.test = self.guard_test(node.test)
            counter = ast.copy_location(ast.Expr(value=call("_sbx_hit_", node)), node)
            return [counter, node]

        def guard_test(self, expr):
            if isinstance(expr, ast.Compare):
                expr.left = call("_sbx_val_", expr.left, expr.left)
                expr.comparators = [call("_sbx_val_", c, c) for c in expr.comparators]
                return expr
            if isinstance(expr, ast.BoolOp):
                expr.values = [self.guard_test(v) for v in expr.values]
                return expr
            if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
                expr.operand = self.guard_test(expr.operand)
                return expr
            return call("_sbx_val_", expr, expr)

    tree = Instrumenter().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


# ----------------------------------------------------------------------------------------------------------
# Running the program
# ----------------------------------------------------------------------------------------------------------

def _text(obj, limit=1000):
    """Converts an untrusted object to a plain str without trusting its str subclass methods."""
    try:
        s = str(obj)
        s = str.encode(s, "utf-8", "backslashreplace").decode("utf-8")
    except BaseException:
        s = "<unprintable>"
    return s[:limit]


def _describe(exc):
    return f"{type.__dict__['__name__'].__get__(type(exc))}: {_text(exc)}"


def _exit_status(exc):
    """Turns SystemExit into an exit code the way the interpreter does: None is 0, a non-int prints and is 1."""
    code = exc.code
    if code is None:
        return 0, ""
    if isinstance(code, int):
        return int(code), ""
    return 1, _text(code)


def main():
    run_dir, nonce_fd, verdict_fd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    libc = ctypes.CDLL(None, use_errno=True)
    real_stderr_fd = 2
    try:
        nonce_addr = read_nonce(libc, nonce_fd)
    except BaseException as exc:
        os.write(real_stderr_fd, f"runner: cannot read the nonce: {exc}\n".encode())
        os._exit(SETUP_FAILED_EXIT)

    # Everything the verdict path needs is bound now, before the program runs, so later changes to modules
    # cannot alter how the record is written.
    string_at, write, encode = ctypes.string_at, os.write, str.encode

    def emit(status, exit_code=0, detail="", shadowed=()):
        names = ",".join(shadowed)
        record = b"".join([
            string_at(nonce_addr, NONCE_LEN), b" ", encode(status, "ascii"), b" ", b"%d" % exit_code, b" ",
            encode(names, "utf-8", "backslashreplace") if names else b"-", b" ",
            encode(detail, "utf-8", "backslashreplace").replace(b"\n", b"\\n")[:4000], b"\n",
        ])
        view = memoryview(record)
        while view:
            view = view[write(verdict_fd, view):]

    try:
        with open(os.path.join(run_dir, "spec.marshal"), "rb") as f:
            spec = marshal.loads(f.read())
        sys.path.insert(0, _HERE)
        import rules
        sys.path.remove(_HERE)

        work_dir = os.path.join(run_dir, "work")
        apply_landlock(libc, ["/usr", "/lib", "/lib64", "/bin"] + python_lib_dirs(),
                       ["/etc/ld.so.cache", "/etc/localtime"], ["/dev/null"], work_dir,
                       bool(spec.get("writable_work_dir")))
        apply_seccomp(libc)
        apply_rlimits(spec["limits"])
        os.chdir(work_dir)
        os.environ.clear()
        sys.argv = [""]
        # Correct solutions sometimes print integers longer than CPython's default 4300-digit guard. The CPU
        # limit already bounds the cost, so the guard is lifted to match how the reference outputs were made.
        sys.set_int_max_str_digits(0)
    except BaseException as exc:
        emit("setup_error", 0, _describe(exc))
        os._exit(0)

    mode = spec["mode"]
    program = spec["program"]
    guards_on = bool(spec.get("guards", True))
    entry_point = spec.get("entry_point")
    stderr = sys.stderr

    def print_traceback(exc):
        # Imported only on failure: traceback and linecache pull in a dozen modules.
        try:
            import linecache
            import traceback
            linecache.cache["<solution>"] = (len(program), None, program.splitlines(True), "<solution>")
            traceback.print_exception(exc, file=stderr)
        except BaseException:
            pass

    module_name = "__main__" if mode == "stdin" else "solution"
    module = _ModuleType(module_name)
    module.__dict__["__builtins__"] = make_program_builtins(rules) if guards_on else builtins
    runner_module = sys.modules.get("__main__")  # Kept alive: the runner's own functions use its globals.
    sys.modules[module_name] = module

    status, code, detail, shadowed = "ok", 0, "", []
    try:
        compiled = compile(program, "<solution>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        compiled = None
        status, code, detail = "syntax_error", 1, _describe(exc)

    # The tests and the trusted context are parsed before the program runs, so a broken test file is reported
    # as such and not blamed on the program.
    tests_tree = None
    if compiled is not None and mode == "assert":
        try:
            import ast
            tests_src = spec["tests"]
            tests_tree = ast.parse(tests_src, filename="<tests>")
            tests_lines = tests_src.splitlines()
            loaded, defined = analyse_tests(tests_tree, module_name)
            context = spec.get("context")
            context_code = compile(context, "<context>", "exec", dont_inherit=True) if context else None
            guards = make_value_guards()
        except BaseException as exc:
            status, code, detail = "tests_error", 1, _describe(exc)
            compiled = None
            tests_tree = None

    if compiled is not None:
        try:
            exec(compiled, module.__dict__)
        except SystemExit as exc:
            if mode == "stdin":
                code, message = _exit_status(exc)
                if message:
                    stderr.write(message + "\n")
                if code != 0:
                    status, detail = "exit", message or f"exit code {code}"
            else:
                status, code, detail = "early_exit", 0, "the program raised SystemExit before the tests ran"
        except MemoryError:
            status, code, detail = "memory_error", 1, "MemoryError"
        except BaseException as exc:
            print_traceback(exc)
            status, code, detail = "exception", 1, _describe(exc)

    if tests_tree is not None and status == "ok":
        program_globals = module.__dict__
        try:
            namespace, shadowed, trusted_roots = build_test_namespace(
                program_globals, loaded, entry_point, context_code, guards, rules)
            sys.modules[module_name] = make_program_view(program_globals, entry_point, guards["_sbx_ret_"], rules)
            tests_code = compile(instrument_tests(tests_tree, trusted_roots | defined), "<tests>", "exec",
                                 dont_inherit=True)
        except NotPlainValue as exc:
            namespace = None
            status, code, detail = "not_plain", 1, _text(exc.args[0] if exc.args else "")
        except BaseException as exc:
            namespace = None
            status, code, detail = "tests_error", 1, "while preparing the tests: " + _describe(exc)
        if namespace is not None:
            try:
                exec(tests_code, namespace)
                if guards["hits"]() == 0:
                    status, code, detail = "no_asserts", 1, "the tests finished without running any assert"
            except AssertionError as exc:
                line = 0
                tb = exc.__traceback__
                while tb is not None:
                    if tb.tb_frame.f_code.co_filename == "<tests>":
                        line = tb.tb_lineno
                    tb = tb.tb_next
                source = tests_lines[line - 1].strip() if 0 < line <= len(tests_lines) else ""
                status, code, detail = "assert_failed", 1, f"line {line}: {source[:300]}"
                message = _text(exc, 300)
                if message:
                    detail += f" ({message})"
            except NotPlainValue as exc:
                status, code, detail = "not_plain", 1, _text(exc.args[0] if exc.args else "")
            except SystemExit:
                status, code, detail = "early_exit", 0, "SystemExit was raised while the tests were running"
            except MemoryError:
                status, code, detail = "memory_error", 1, "MemoryError"
            except BaseException as exc:
                print_traceback(exc)
                status, code, detail = "exception", 1, _describe(exc)
        if shadowed and status != "ok":
            detail += f" [the tests used the real {', '.join(shadowed)}, not the program's]"

    # Interpreter shutdown, done by hand: wait for non-daemon threads (the deep-recursion idiom runs the
    # whole solution in one), then flush whatever object the program left in sys.stdout. threading is only
    # loaded if the program imported it; the program cannot have replaced _shutdown, because both the static
    # policy and the runtime guards deny private module attributes.
    try:
        threading = sys.modules.get("threading")
        if threading is not None:
            threading._shutdown()
        for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
            # Like interpreter shutdown, skip streams the program already closed.
            if stream is not None and not getattr(stream, "closed", False):
                stream.flush()
    except BaseException as exc:
        if status == "ok":
            status, code, detail = "exception", 1, "while finishing: " + _describe(exc)

    del runner_module
    emit(status, code, detail, shadowed)
    # os._exit skips interpreter finalisation, so no program code (atexit hooks, __del__ methods) can run
    # after the verdict is written.
    os._exit(0)


if __name__ == "__main__":
    main()
