"""The allowlists and attribute rules shared by the static policy and the sandbox's runtime guards.

policy.py uses these tables to check a program's AST before it runs. runner.py imports this module inside the
sandbox, before the sandbox closes, and enforces the same tables at runtime for the dynamic paths a static
check cannot see (getattr with a computed name, imports, open). Keeping one copy means the two layers cannot
drift apart.

This module must stay cheap to import: every sandbox start loads it, and when the interpreter's files live on a
shared filesystem each extra module costs about a millisecond. So it imports only sys.
"""

import sys

# Modules a program may import. The value says which attributes the program may use:
#   None      -> any public attribute except those in DENIED_ATTRS (modules that only compute);
#   frozenset -> only the listed attributes (modules that also expose process, file or interpreter control).
ALLOWED_MODULES = {
    # Numbers.
    "math": None, "cmath": None, "decimal": None, "fractions": None, "numbers": None, "statistics": None,
    "random": None,
    # Data structures and algorithm helpers.
    "collections": None, "collections.abc": None, "heapq": None, "bisect": None, "array": None, "queue": None,
    "copy": None, "itertools": None, "functools": None, "operator": None,
    # Typing and class helpers.
    "typing": None, "dataclasses": None, "enum": None, "abc": None, "__future__": None,
    # Text, dates and clocks. The wall clock is enforced by the parent, so reading time is harmless.
    "re": None, "string": None, "textwrap": None, "unicodedata": None, "json": None, "datetime": None,
    "calendar": None, "time": None,
    # sys: the standard streams, recursion and int-size knobs, and exit. Exiting early cannot fake a pass
    # because the verdict comes from the harness, not from the exit code.
    "sys": frozenset({
        "stdin", "stdout", "stderr", "__stdin__", "__stdout__", "__stderr__", "setrecursionlimit",
        "getrecursionlimit", "set_int_max_str_digits", "get_int_max_str_digits", "maxsize", "maxunicode",
        "float_info", "int_info", "byteorder", "exit", "version_info", "getsizeof",
    }),
    # os: only what the fast-IO templates use (os.read(0, os.fstat(0).st_size), os.write(1, ...)), plus the
    # os.path checks some templates run before falling back to stdin.
    "os": frozenset({"read", "write", "fstat", "linesep", "path"}),
    "os.path": frozenset({"exists", "isfile", "isdir", "join", "basename", "dirname", "splitext"}),
    # io: in-memory buffers and the base classes the PyRival FastIO template subclasses, but no file opening.
    "io": frozenset({
        "BytesIO", "StringIO", "IOBase", "RawIOBase", "BufferedIOBase", "TextIOBase", "TextIOWrapper",
        "BufferedReader", "BufferedWriter", "DEFAULT_BUFFER_SIZE", "SEEK_SET", "SEEK_CUR", "SEEK_END",
        "UnsupportedOperation",
    }),
    # threading: the deep-recursion idiom (threading.stack_size(...) then Thread(target=main).start()) and the
    # basic locks, but no tracing hooks and no access to other threads.
    "threading": frozenset({
        "Thread", "stack_size", "Lock", "RLock", "Event", "Condition", "Semaphore", "BoundedSemaphore",
        "current_thread", "main_thread", "get_ident", "local",
    }),
    # hashlib: the hash constructors only. HumanEval/162 (md5 of a string) needs it, and hashing only computes.
    # file_digest is left out because it reads from file objects, which no task needs.
    "hashlib": frozenset({
        "md5", "sha1", "sha224", "sha256", "sha384", "sha512", "sha3_224", "sha3_256", "sha3_384", "sha3_512",
        "shake_128", "shake_256", "blake2b", "blake2s", "new", "algorithms_available", "algorithms_guaranteed",
    }),
}

# The names a program binds when it imports an allowed module ("os" for "import os.path"). In assert mode the
# tests always get the real module under these names, never an object the program bound to the same name.
MODULE_NAMES = frozenset(name.split(".")[0] for name in ALLOWED_MODULES)

# The only module attributes a program may assign. Fast-IO templates replace the standard streams.
SYS_WRITABLE = frozenset({"stdin", "stdout", "stderr"})

# Public attributes of otherwise allowed modules that are still denied.
DENIED_ATTRS = {
    # These evaluate strings (string annotations, forward references) with eval(), which would run code that
    # this check never parsed.
    "typing": frozenset({"get_type_hints", "ForwardRef", "get_overloads", "evaluate_forward_ref"}),
    "functools": frozenset({"singledispatch", "singledispatchmethod"}),
    # Formatter.format follows "{0.attr}" fields to arbitrary attributes, like str.format on a computed string.
    "string": frozenset({"Formatter"}),
    # make_dataclass builds attribute names from runtime strings, which defeats the attribute-name checks.
    "dataclasses": frozenset({"make_dataclass"}),
}

# Library functions that are allowed only when called directly with checkable arguments. Using them as values
# (passing them around) would let the arguments be computed at runtime.
CALL_CHECKED = frozenset({
    ("operator", "attrgetter"), ("operator", "methodcaller"),
    ("functools", "update_wrapper"), ("functools", "wraps"),
})

# Builtins that are never available to a program. They execute strings, expose namespaces or start a debugger.
DENIED_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__", "breakpoint", "globals", "vars", "locals", "help",
})

# Builtins that are allowed only as direct calls whose arguments the check can read.
CALL_ONLY_BUILTINS = frozenset({"getattr", "setattr", "delattr", "hasattr", "open"})

# Dunder attributes a program may read with obj.__x__ syntax: the data-model methods (so super().__init__()
# and explicit operator calls work) and harmless names. Anything else, such as __class__, __dict__,
# __globals__ or __subclasses__, is rejected, because those are the stepping stones of every known escape
# from restricted Python.
_OPERATORS = "add sub mul matmul truediv floordiv mod divmod pow lshift rshift and xor or".split()
SAFE_DUNDER_ATTRS = frozenset(
    {f"__{name}__" for name in (
        "init new del repr str bytes format lt le eq ne gt ge hash bool len length_hint getitem setitem "
        "delitem missing iter reversed contains next neg pos abs invert complex int float index round trunc "
        "floor ceil enter exit call name qualname doc module"
    ).split()}
    | {f"__{prefix}{op}__" for op in _OPERATORS for prefix in ("", "r", "i")}
)

# Bare dunder names a program may use. "if __name__ == '__main__':" must keep working; __builtins__,
# __import__, __loader__ and __spec__ must not.
SAFE_DUNDER_NAMES = frozenset({"__name__", "__doc__", "__debug__", "__all__", "__slots__", "__version__",
                               "__author__"})

# Attributes that lead from ordinary objects to frames, code objects and closure cells. With a frame a program
# could read the harness's locals, including the test code with the expected answers.
INTROSPECTION_ATTRS = frozenset({
    "gi_frame", "gi_code", "gi_yieldfrom", "gi_suspended", "cr_frame", "cr_code", "cr_await", "cr_origin",
    "ag_frame", "ag_code", "ag_await", "tb_frame", "tb_next", "tb_lasti", "tb_lineno",
    "f_back", "f_builtins", "f_code", "f_globals", "f_locals", "f_lasti", "f_lineno", "f_trace",
    "f_trace_lines", "f_trace_opcodes", "f_generator", "cell_contents",
    "co_code", "co_consts", "co_names", "co_varnames", "co_freevars", "co_cellvars", "co_filename", "co_name",
    "co_qualname", "co_lines", "co_positions", "co_linetable", "co_lnotab", "co_exceptiontable",
})

# Dunder names that are dangerous even as plain strings, because some library function might look them up with
# getattr. A string literal equal to one of these, or to an introspection attribute, is rejected.
DANGEROUS_DUNDERS = frozenset({
    "__globals__", "__builtins__", "__dict__", "__class__", "__bases__", "__base__", "__mro__",
    "__subclasses__", "__code__", "__closure__", "__func__", "__self__", "__traceback__", "__getattribute__",
    "__reduce__", "__reduce_ex__", "__import__", "__loader__", "__spec__", "__wrapped__", "__defaults__",
    "__kwdefaults__", "__objclass__", "__setattr__", "__delattr__", "__init_subclass__", "__dictoffset__",
})

# Names starting with this prefix are reserved for helpers the harness injects into the test namespace.
RESERVED_PREFIX = "_sbx"

# Real solutions are a few kilobytes. The cap keeps a pathological completion from spending seconds of
# parser time in the trusted parent.
MAX_SOURCE_CHARS = 200_000


def is_dunder(name):
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def canonical_module_name(module):
    """Returns the ALLOWED_MODULES key for a module object, or its real name if it has none.

    os.path is the one module whose real name (posixpath) differs from the name programs import it by.
    """
    name = getattr(module, "__name__", None)
    if name == "posixpath":
        return "os.path"
    return name


def _module_attr(module_name, attr):
    """Looks an attribute up on the real module, importing it on first use.

    Only names from ALLOWED_MODULES reach this function, and those are all standard-library modules, so
    importing them in the trusted parent (or in the runner, outside the program) is safe.
    """
    __import__(module_name)
    return getattr(sys.modules[module_name], attr, None)


def module_attr_problem(module_name, attr, store=False):
    """Returns why module_name.attr may not be used, or None if it may.

    This one function serves the static check (attribute syntax, from-imports, getattr with a literal name)
    and the runtime guards in runner.py (getattr, setattr and from-imports at runtime).
    """
    if module_name not in ALLOWED_MODULES:
        return f"module '{module_name}' is not allowed"
    allowed = ALLOWED_MODULES[module_name]
    if store:
        if module_name == "sys" and attr in SYS_WRITABLE:
            return None
        return f"assigning '{module_name}.{attr}' is not allowed"
    if isinstance(allowed, frozenset):
        if attr not in allowed:
            return f"'{module_name}.{attr}' is not in the allowlist for '{module_name}'"
        return None
    if attr.startswith("_"):
        return f"private attribute '{module_name}.{attr}' is not allowed"
    if attr in DENIED_ATTRS.get(module_name, ()):
        return f"'{module_name}.{attr}' is not allowed"
    if f"{module_name}.{attr}" in ALLOWED_MODULES:
        return None
    if isinstance(_module_attr(module_name, attr), type(sys)):
        # Many modules keep other modules as attributes (typing.sys, dataclasses.inspect). Following them
        # would bypass the import allowlist.
        return f"'{module_name}.{attr}' is a module that is not allowed"
    return None


def attribute_name_problem(attr, store=False):
    """Returns why reading (or, with store=True, assigning) obj.<attr> is not allowed, or None.

    Used for attributes of ordinary objects; module attributes go through module_attr_problem.
    """
    if is_dunder(attr):
        if store:
            return f"assigning dunder attribute '{attr}' is not allowed"
        if attr not in SAFE_DUNDER_ATTRS:
            return f"dunder attribute '{attr}' is not allowed"
    if attr in INTROSPECTION_ATTRS:
        return f"introspection attribute '{attr}' is not allowed"
    if attr in ("format", "format_map"):
        return f"'{attr}' reached through a computed name can follow fields to arbitrary attributes"
    return None
