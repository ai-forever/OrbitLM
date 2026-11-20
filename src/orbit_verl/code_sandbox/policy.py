"""Static AST policy for generated Python programs.

check_program(source) parses a program and returns (ok, reasons) without running it. It runs in the trusted
parent before any sandbox is started, so a rejected program costs almost nothing and gets reward 0. Every reason
starts with a rule id ("denied_import: line 3: ..."), so the reward code can count violations per rule.

The sandbox in runner.py is the security boundary. This policy is a second layer with a different job: it closes
the routes to reward hacking that happen inside the interpreter, where the sandbox sees nothing wrong. Examples are
reading the test code through frame objects, replacing a library function the tests call, or evaluating strings
that this check never saw. The rules are tuned so that ordinary competitive-programming and function-style
solutions pass.

The tables themselves live in rules.py, which runner.py also imports to enforce them at runtime.
"""

import ast
import importlib
import warnings

import _string

from .rules import (
    ALLOWED_MODULES, CALL_CHECKED, CALL_ONLY_BUILTINS, DANGEROUS_DUNDERS, DENIED_BUILTINS, INTROSPECTION_ATTRS,
    MAX_SOURCE_CHARS, RESERVED_PREFIX, SAFE_DUNDER_NAMES, attribute_name_problem, is_dunder, module_attr_problem,
)

# Generated programs often hold regexes such as "\d" in plain strings. Parsing one makes Python 3.12 print a
# SyntaxWarning (3.11: DeprecationWarning) to the trainer's stderr, once per program, which floods the logs.
# The filter is installed once at import, because warnings.catch_warnings is not thread-safe and a reward pool
# parses from many threads. It only matches the file name "<solution>" that check_program parses under, so no
# other warning of the trainer is hidden.
warnings.filterwarnings("ignore", category=SyntaxWarning, module="<solution>")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="<solution>")


def check_program(source):
    """Checks a program against the policy. Returns (ok, reasons); ok is True when reasons is empty."""
    if not isinstance(source, str):
        return False, ["syntax_error: the program is not a string"]
    if len(source) > MAX_SOURCE_CHARS:
        return False, [f"too_large: the program has {len(source)} characters (limit {MAX_SOURCE_CHARS})"]
    try:
        tree = ast.parse(source, filename="<solution>", mode="exec")
    except (SyntaxError, ValueError) as exc:
        return False, [f"syntax_error: {type(exc).__name__}: {exc}"]
    except (RecursionError, MemoryError):
        return False, ["syntax_error: the program is too deeply nested to parse"]
    checker = _Checker(tree)
    checker.run()
    return not checker.reasons, checker.reasons


class _Checker:
    """Walks the AST twice: first to learn what every name refers to, then to check every node."""

    def __init__(self, tree):
        self.tree = tree
        self.reasons = []
        self._seen = set()
        # Local name -> canonical module name, for every "import x" / "from x import submodule".
        self.module_aliases = {}
        # Local name -> (module, attribute), for every "from module import name".
        self.object_aliases = {}
        # Local name -> reason, for names a star import brings in that the program must not use.
        self.poisoned = {}
        # Every name the program binds anywhere. A denied builtin name that the program defines itself refers
        # to the program's object, so it is not reported. The runtime builtins catch the rare case where the
        # binding never executes.
        self.bound = set()

    def run(self):
        # The walk keeps an explicit stack, so deeply nested programs cannot exhaust Python's recursion limit.
        nodes = []
        stack = [(self.tree, None, "module")]
        while stack:
            node, parent, scope = stack.pop()
            nodes.append((node, parent, scope))
            child_scope = scope
            if isinstance(node, ast.ClassDef):
                child_scope = "class"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                child_scope = "function"
            for child in ast.iter_child_nodes(node):
                stack.append((child, node, child_scope))
        nodes.reverse()
        for node, _, _ in nodes:
            self._collect(node)
        for node, parent, scope in nodes:
            self._check(node, parent, scope)

    def _add(self, rule, node, message):
        line = getattr(node, "lineno", 0)
        reason = f"{rule}: line {line}: {message}"
        if reason not in self._seen:
            self._seen.add(reason)
            self.reasons.append(reason)

    # ----- pass 1: what names refer to -----

    def _collect(self, node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    self.module_aliases[alias.asname] = alias.name
                    self.bound.add(alias.asname)
                else:
                    top = alias.name.split(".")[0]
                    self.module_aliases[top] = top
                    self.bound.add(top)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or module not in ALLOWED_MODULES:
                return
            for alias in node.names:
                if alias.name == "*":
                    self._collect_star(module)
                    continue
                local = alias.asname or alias.name
                self.bound.add(local)
                if f"{module}.{alias.name}" in ALLOWED_MODULES:
                    self.module_aliases[local] = f"{module}.{alias.name}"
                else:
                    self.object_aliases[local] = (module, alias.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.bound.add(node.name)
        elif isinstance(node, ast.arg):
            self.bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            self.bound.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            self.bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            self.bound.add(node.rest)

    def _collect_star(self, module):
        if isinstance(ALLOWED_MODULES[module], frozenset):
            return  # Reported in pass 2; restricted modules cannot be star-imported.
        real = importlib.import_module(module)
        names = getattr(real, "__all__", None) or [n for n in dir(real) if not n.startswith("_")]
        for name in names:
            problem = module_attr_problem(module, name)
            if problem:
                self.poisoned[name] = problem
            elif (module, name) in CALL_CHECKED:
                self.object_aliases.setdefault(name, (module, name))

    # ----- pass 2: checks -----

    def _check(self, node, parent, scope):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES:
                    self._add("denied_import", node, f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            self._check_import_from(node)
        elif isinstance(node, ast.Name):
            self._check_name(node, parent, scope)
        elif isinstance(node, ast.Attribute):
            self._check_attribute(node, parent)
        elif isinstance(node, ast.Call):
            self._check_call(node)
        elif isinstance(node, ast.MatchClass):
            for attr in node.kwd_attrs:
                problem = attribute_name_problem(attr)
                if problem:
                    self._add("dunder_attr" if is_dunder(attr) else "introspection_attr", node, problem)
        elif isinstance(node, ast.keyword) and node.arg:
            if node.arg in INTROSPECTION_ATTRS or node.arg in DANGEROUS_DUNDERS:
                self._add("protected_name", node, f"keyword argument '{node.arg}' names a protected attribute")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in INTROSPECTION_ATTRS or node.value in DANGEROUS_DUNDERS:
                self._add("protected_name", node, f"string literal '{node.value}' names a protected attribute")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._check_reserved(node.name, node)
        elif isinstance(node, ast.arg):
            self._check_reserved(node.arg, node)
        elif isinstance(node, ast.alias) and node.asname:
            self._check_reserved(node.asname, parent)

    def _check_reserved(self, name, node):
        if name.startswith(RESERVED_PREFIX):
            self._add("reserved_name", node, f"names starting with '{RESERVED_PREFIX}' are reserved")

    def _check_import_from(self, node):
        module = node.module or ""
        if node.level:
            self._add("denied_import", node, "relative imports are not allowed")
            return
        if module not in ALLOWED_MODULES:
            self._add("denied_import", node, f"import from '{module}' is not allowed")
            return
        for alias in node.names:
            if alias.name == "*":
                if isinstance(ALLOWED_MODULES[module], frozenset):
                    self._add("star_import", node, f"'from {module} import *' is not allowed")
                continue
            problem = module_attr_problem(module, alias.name)
            if problem:
                self._add("denied_module_attr", node, problem)

    def _check_name(self, node, parent, scope):
        name = node.id
        load = isinstance(node.ctx, ast.Load)
        if name.startswith(RESERVED_PREFIX):
            self._add("reserved_name", node, f"names starting with '{RESERVED_PREFIX}' are reserved")
        if is_dunder(name):
            # Assigning a dunder inside a class body defines a special method (__hash__ = None, __lt__ = f),
            # which is allowed just like "def __lt__".
            class_store = scope == "class" and isinstance(node.ctx, ast.Store)
            if not class_store and name not in SAFE_DUNDER_NAMES:
                self._add("dunder_name", node, f"name '{name}' is not allowed")
        if name in INTROSPECTION_ATTRS:
            self._add("protected_name", node, f"name '{name}' matches a protected attribute")
        if not load:
            return
        is_called = isinstance(parent, ast.Call) and parent.func is node
        if name in DENIED_BUILTINS and name not in self.bound:
            self._add("denied_builtin", node, f"builtin '{name}' is not allowed")
        if name in CALL_ONLY_BUILTINS and name not in self.bound and not is_called:
            self._add("builtin_as_value", node, f"builtin '{name}' may only be called directly")
        if name in self.module_aliases and not (isinstance(parent, ast.Attribute) and parent.value is node):
            self._add("module_as_value", node,
                      f"module '{self.module_aliases[name]}' may only be used as 'module.attribute'")
        target = self.object_aliases.get(name)
        if target in CALL_CHECKED and not is_called:
            self._add("builtin_as_value", node, f"'{target[0]}.{target[1]}' may only be called directly")
        if name in self.poisoned and name not in self.bound:
            self._add("denied_module_attr", node, self.poisoned[name])

    def _resolve(self, expr):
        """Says what an expression refers to: ("module", name), ("object", module, attr) or None."""
        if isinstance(expr, ast.Name):
            if expr.id in self.module_aliases:
                return ("module", self.module_aliases[expr.id])
            if expr.id in self.object_aliases:
                return ("object",) + self.object_aliases[expr.id]
            return None
        if isinstance(expr, ast.Attribute):
            base = self._resolve(expr.value)
            if base and base[0] == "module":
                submodule = f"{base[1]}.{expr.attr}"
                if submodule in ALLOWED_MODULES:
                    return ("module", submodule)
                return ("object", base[1], expr.attr)
        return None

    def _check_attribute(self, node, parent):
        store = isinstance(node.ctx, (ast.Store, ast.Del))
        base = self._resolve(node.value)
        if base and base[0] == "module":
            problem = module_attr_problem(base[1], node.attr, store)
            if problem:
                rule = "module_attr_write" if store else "denied_module_attr"
                self._add(rule, node, problem)
            is_called = isinstance(parent, ast.Call) and parent.func is node
            if (base[1], node.attr) in CALL_CHECKED and not is_called:
                self._add("builtin_as_value", node, f"'{base[1]}.{node.attr}' may only be called directly")
            return
        if store and base and base[0] == "object":
            # Assigning attributes of imported classes or functions (Fraction.x = ..., Counter.y = ...) would
            # change library behaviour that the tests and the harness rely on.
            self._add("imported_object_write", node,
                      f"assigning attribute '{node.attr}' of imported '{base[1]}.{base[2]}' is not allowed")
        if node.attr in ("format", "format_map"):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                self._check_format_string(node.value.value, node)
            else:
                self._add("format_traversal", node,
                          f"'.{node.attr}' is only allowed on a string literal, because format fields like "
                          "'{0.attr}' can reach arbitrary attributes")
            return
        problem = attribute_name_problem(node.attr, store)
        if problem:
            rule = "introspection_attr" if node.attr in INTROSPECTION_ATTRS else "dunder_attr"
            self._add(rule, node, problem)

    def _check_format_string(self, fmt, node, depth=0):
        try:
            parsed = list(_string.formatter_parser(fmt))
        except ValueError:
            return  # An invalid format string fails at runtime without reaching any attribute.
        for _, field, spec, _ in parsed:
            if field:
                try:
                    _, rest = _string.formatter_field_name_split(field)
                    for is_attr, key in rest:
                        if is_attr and (key.startswith("_") or key in INTROSPECTION_ATTRS):
                            self._add("format_traversal", node,
                                      f"format field '{field}' reads a private or protected attribute")
                except ValueError:
                    pass
            if spec and depth < 3:
                self._check_format_string(spec, node, depth + 1)

    def _check_call(self, node):
        func = node.func
        if isinstance(func, ast.Name) and func.id not in self.bound:
            if func.id in ("getattr", "setattr", "delattr", "hasattr"):
                self._check_attr_call(func.id, node)
            elif func.id == "open":
                self._check_open(node)
        target = self._resolve(func)
        if not target or target[0] != "object":
            return
        key = (target[1], target[2])
        if key in (("operator", "attrgetter"), ("operator", "methodcaller")):
            names = node.args if key[1] == "attrgetter" else node.args[:1]
            if not names or node.keywords and key[1] == "attrgetter":
                self._add("dynamic_attr", node, f"'{key[1]}' needs literal attribute names")
            for arg in names:
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                    self._add("dynamic_attr", node, f"'{key[1]}' needs literal attribute names")
                    continue
                for part in arg.value.split("."):
                    problem = attribute_name_problem(part)
                    if problem:
                        self._add("dynamic_attr", node, problem)
        elif key in (("functools", "update_wrapper"), ("functools", "wraps")):
            # With custom 'assigned'/'updated' lists these copy attributes named by runtime strings from one
            # object to another, which would launder protected attributes.
            max_positional = 2 if key[1] == "update_wrapper" else 1
            custom = len(node.args) > max_positional or any(
                kw.arg in (None, "assigned", "updated") for kw in node.keywords)
            if custom:
                self._add("dynamic_attr", node, f"'functools.{key[1]}' with custom assigned/updated lists")

    def _check_attr_call(self, name, node):
        args = node.args
        if len(args) < 2 or any(isinstance(a, ast.Starred) for a in args) or node.keywords:
            self._add("dynamic_attr", node, f"'{name}' needs an object and a literal attribute name")
            return
        key = args[1]
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            if name != "hasattr":
                self._add("dynamic_attr", node, f"'{name}' with a computed attribute name is not allowed")
            return
        if name == "hasattr":
            return  # hasattr only answers yes or no, so any literal name is harmless.
        attr = key.value
        store = name in ("setattr", "delattr")
        base = self._resolve(args[0])
        if base and base[0] == "module":
            problem = module_attr_problem(base[1], attr, store)
            if problem:
                self._add("module_attr_write" if store else "denied_module_attr", node, problem)
            return
        if store and base and base[0] == "object":
            self._add("imported_object_write", node,
                      f"'{name}' on imported '{base[1]}.{base[2]}' is not allowed")
        problem = attribute_name_problem(attr, store)
        if problem:
            self._add("dynamic_attr", node, problem)

    def _check_open(self, node):
        # Reading is allowed: open(0) is the common way to read stdin, and the sandbox decides which files
        # exist for the program at all. Writing is not: the program has no business creating files.
        if any(isinstance(a, ast.Starred) for a in node.args):
            self._add("open_mode", node, "open() with *args is not allowed")
            return
        mode = node.args[1] if len(node.args) > 1 else None
        if len(node.args) > 7:
            self._add("open_mode", node, "open() with an opener is not allowed")
        for kw in node.keywords:
            if kw.arg is None:
                self._add("open_mode", node, "open() with **kwargs is not allowed")
            elif kw.arg == "mode":
                mode = kw.value
            elif kw.arg == "opener":
                self._add("open_mode", node, "open() with an opener is not allowed")
        if mode is None:
            return
        if not (isinstance(mode, ast.Constant) and isinstance(mode.value, str)):
            self._add("open_mode", node, "open() needs a literal mode")
        elif any(ch in mode.value for ch in "wax+"):
            self._add("open_mode", node, f"open() in mode '{mode.value}' is not allowed (read-only)")
