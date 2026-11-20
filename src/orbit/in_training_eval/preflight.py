"""The startup check of the evaluation environment, run with eval_python before the first training step."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from .benchmarks import BENCHMARKS, NLTK_PUNKT_CHECK
from .environment import build_eval_env, libstdcxx_preload
from .harness import LM_EVAL_VERSION, lm_eval_install_hint, verify_lm_eval_files
from .options import _has_tokenizer_files
from .processes import kill_process_group
from .settings import EvalSettings
from .suite import ITR_RENDER_PATH


# The checks that the preflight runs in the evaluation environment, besides imports.
MANAGER_CHECK = "+multiprocessing_manager"
CUDA_CHECK = "+cuda"
# The render check of a chat suite (itr_render.render_check). Its name carries its arguments as JSON
# after "=", and its result and errors are reported under the bare name.
CHAT_RENDER_CHECK = "+chat_render"
# Every other rank waits for the preflight inside a collective, so it must end well within the
# NCCL timeout, which is 10 minutes by default.
PREFLIGHT_TIMEOUT_SECONDS = 420.0


def preflight_modules(settings: EvalSettings, *, cuda: bool = False) -> list[str]:
    # torch comes first because it comes first in an evaluation process as well: run_lm_eval.py's
    # flash_attn guard and vLLM import it before lm-eval builds any task. The first library that
    # loads libstdc++ decides which copy the whole process uses (see build_eval_env). In this
    # order, the imports below get the same libstdc++ as in an evaluation. In another order, a
    # module such as sqlite3 could load the environment's copy first and pass here, while the
    # evaluation itself fails.
    modules = ["torch", "lm_eval", "lm_eval.evaluator"]
    # vLLM's CUDA platform module imports vLLM's compiled extensions when it loads. Importing it
    # checks that they match the evaluation environment's torch and CUDA libraries, which a plain
    # "import vllm" does not. The check goes through that module because the extensions' names
    # change between releases: vLLM 0.22 loads vllm._C, and 0.26 has no vllm._C
    # and loads vllm._C_stable_libtorch instead.
    modules += ["vllm", "vllm.platforms.cuda"]
    # lm-eval's vLLM backend imports several vLLM modules by name when it loads. Importing it here
    # shows that the pinned lm-eval and the environment's vLLM fit together.
    modules += ["lm_eval.models.vllm_causallms"]
    names = set(settings.suite.benchmarks)
    if "math" in names:
        modules += ["sympy", "math_verify"]
    if "ruler" in names:
        # RULER's prompt generator imports wonderwords and nltk. nltk imports sqlite3, the module
        # that needs the newer libstdc++, so sqlite3 is also imported on its own, and a failure
        # there is reported under its own name.
        modules += ["wonderwords", "sqlite3", "nltk"]
    if names & {"humaneval", "mbpp"}:
        # Both score through the code_eval metric of the evaluate package.
        modules += ["evaluate"]
    if names & {"humaneval", "cruxeval", "mbpp"}:
        # All three run every generated program in a child started through multiprocessing.Manager,
        # whose Unix socket lives under TMPDIR. A TMPDIR path longer than about 70 characters
        # overflows the socket path limit, and the manager dies. HumanEval then refuses to load,
        # MBPP's process fails when it scores its first program, and CRUXEval counts every program
        # as failed and silently scores 0.
        modules += [MANAGER_CHECK]
    if settings.suite.chat is not None:
        for name in settings.suite.benchmarks:
            modules += [module for module in BENCHMARKS[name].modules if module not in modules]
        # The chat prompts are rendered with the template of the exported tokenizer, which comes from
        # eval_tokenizer_source. The check renders probes with it and compares their ids with the
        # prompt contract (eval/in_training/README.md). It also compares the
        # vocabulary with the training tokenizer's. That comparison needs tokenizer files next to
        # tokenizer_config, so it is skipped when the config file is kept outside a tokenizer folder.
        train_folder = Path(settings.tokenizer_config).expanduser().resolve().parent
        spec = {
            "module_dir": str(ITR_RENDER_PATH.parent),
            "tokenizer": str(settings.tokenizer_source),
            "train_tokenizer": str(train_folder) if _has_tokenizer_files(train_folder) else None,
        }
        modules += [f"{CHAT_RENDER_CHECK}={json.dumps(spec, sort_keys=True)}"]
    if cuda:
        modules += [CUDA_CHECK]
    return modules


PREFLIGHT_CODE = r"""
import importlib, json, sys
report = {"python": sys.version.split()[0], "modules": {}, "errors": {}}
for name in sys.argv[1:]:
    try:
        if name == "+multiprocessing_manager":
            import multiprocessing, tempfile
            manager = multiprocessing.Manager()
            manager.shutdown()
            report["modules"][name] = {"version": None, "file": tempfile.gettempdir()}
            continue
        if name == "+cuda":
            import torch
            x = torch.ones(256, 256, device="cuda")
            float((x @ x).sum().item())
            report["modules"][name] = {"version": torch.__version__, "file": torch.cuda.get_device_name(0)}
            continue
        if name == "+nltk_punkt_tab":
            import nltk
            report["modules"][name] = {"version": None, "file": str(nltk.data.find("tokenizers/punkt_tab"))}
            continue
        if name.startswith("+chat_render="):
            spec = json.loads(name.split("=", 1)[1])
            sys.path.insert(0, spec["module_dir"])
            import itr_render
            result = itr_render.render_check(spec["tokenizer"], spec.get("train_tokenizer"))
            report["modules"]["+chat_render"] = {"version": None, "file": json.dumps(result, sort_keys=True)}
            continue
        module = importlib.import_module(name)
        report["modules"][name] = {"version": getattr(module, "__version__", None), "file": getattr(module, "__file__", None)}
    except BaseException as exc:
        report["errors"][name.split("=", 1)[0] if name.startswith("+") else name] = repr(exc)
# Where every lm_eval module that the imports above loaded comes from. A namespace package has no
# file, so its search path stands in for it.
report["lm_eval_modules"] = {}
for name, module in list(sys.modules.items()):
    if module is not None and (name == "lm_eval" or name.startswith("lm_eval.")):
        file = getattr(module, "__file__", None)
        report["lm_eval_modules"][name] = [file] if file else [str(entry) for entry in getattr(module, "__path__", [])]
# The libstdc++ that the imports above ended up with, for the startup log. Normally there is one
# copy, and it serves every library of the process.
report["libstdcxx"] = []
try:
    with open("/proc/self/maps") as maps:
        fields = [line.split(None, 5) for line in maps]
    report["libstdcxx"] = sorted({f[5].strip() for f in fields if len(f) == 6 and "/libstdc++.so" in f[5]})
except OSError:
    pass
print("PREFLIGHT_JSON " + json.dumps(report), flush=True)
"""


def probe_eval_environment(
    eval_python: Path, env: dict[str, str], modules: list[str], *, timeout: float = PREFLIGHT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Runs PREFLIGHT_CODE for the given modules with the evaluation interpreter, and returns its report.

    The report lists the version and file of every module that imported, and the error of every
    one that did not. Under "lm_eval_modules" it also lists the file of every lm_eval module that
    those imports loaded along the way, and under "libstdcxx" the libstdc++ files that the process
    loaded (empty without /proc). It raises only when the interpreter does not run or does not
    finish in time.
    """
    command = [str(eval_python), "-c", PREFLIGHT_CODE, *modules]
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_group(process)
        raise RuntimeError(
            f"in-training eval preflight: {eval_python} did not finish within {timeout:.0f} s."
        ) from None
    report = None
    for line in stdout.splitlines():
        if line.startswith("PREFLIGHT_JSON "):
            report = json.loads(line[len("PREFLIGHT_JSON "):])
    if report is None:
        raise RuntimeError(
            f"in-training eval preflight: {eval_python} did not run (rc={process.returncode}).\n"
            f"{stderr[-4000:]}"
        )
    return report


def lm_eval_problems(
    report: dict[str, Any], harness_path: Optional[Path], *, eval_python: Any = "python",
) -> list[str]:
    """What is wrong with the lm_eval that the evaluation processes import; empty when nothing is.

    The report is the preflight's (probe_eval_environment). Three things must hold:

    * lm_eval reports the pinned version.
    * lm_eval comes from eval_harness_path when that is set, and every lm_eval module that the
      preflight loaded comes from the same package folder. The second half matters in an
      environment that also has an editable install of another lm-eval checkout. Such an install
      adds an import hook, which serves any lm_eval module that the package found first does not
      have. Without this check, a package that lacks a module would silently run that module from
      the checkout.
    * The package's files are exactly those of the pinned release (verify_lm_eval_files). This is
      checked last, and only when the first two hold, because it reads every file.

    When all three hold, the number of verified files goes into the report for the startup log.
    """
    info = report["modules"].get("lm_eval")
    if info is None:
        return []  # The import failed, and the caller reports that error itself.
    hint = lm_eval_install_hint(eval_python)
    problems = []
    version = info.get("version")
    if version != LM_EVAL_VERSION:
        problems.append(
            f"lm_eval {info.get('file')} is version {version}, but the in-training evaluation runs on the "
            f"public lm-eval {LM_EVAL_VERSION}. {hint}"
        )
    lm_eval_file = info.get("file")
    if not lm_eval_file:
        problems.append("lm_eval was imported without an __init__.py, so the folder it comes from is unknown")
        return problems
    package = Path(lm_eval_file).resolve().parent
    if harness_path is not None and not package.is_relative_to(Path(harness_path).resolve()):
        problems.append(f"lm_eval resolves to {lm_eval_file}, not to eval_harness_path {harness_path}")
    outside = [
        f"{name} from {location}"
        for name, locations in sorted(report.get("lm_eval_modules", {}).items())
        for location in locations
        if not Path(location).resolve().is_relative_to(package)
    ]
    if outside:
        problems.append(
            f"{len(outside)} lm_eval modules come from outside the lm_eval package {package}, "
            f"for example {'; '.join(outside[:3])}"
        )
    if not problems:
        try:
            report["lm_eval_files_verified"] = verify_lm_eval_files(package.parent)
        except ValueError as exc:
            problems.append(f"{exc}. {hint}")
    return problems


def run_preflight(
    settings: EvalSettings,
    *,
    gpu: str = "",
    rank_cache: Optional[Path] = None,
    timeout: float = PREFLIGHT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Imports the evaluation modules in the evaluation environment and, given a GPU, uses it.

    It runs on local rank 0 of every node at startup. Any of these then stops the run before its
    first step, instead of failing silently at the first evaluation hours later: a missing
    package; an lm-eval other than the pinned release, from another folder than
    eval_harness_path, or with changed files (see lm_eval_problems); a CUDA library that does not
    match (for example one that LD_LIBRARY_PATH or LD_PRELOAD of the trainer's environment puts
    first); a package that needs a newer libstdc++ than the process gets (see preflight_modules); an
    interpreter that cannot start on this machine. The check gets the environment of an evaluation
    process and imports in the order an evaluation does, so that an import which would fail in
    an evaluation fails here first. Under "libstdcxx_preload", the report says whether the
    evaluation environment's libstdc++ is preloaded, and why (see libstdcxx_preload).
    """
    env = build_eval_env(
        dict(os.environ),
        eval_python=settings.eval_python,
        gpu=gpu,
        cache_dir=settings.cache_dir,
        rank_cache=rank_cache,
        harness_path=settings.harness_path,
        extra_pythonpath=settings.extra_pythonpath,
    )
    report = probe_eval_environment(
        settings.eval_python, env, preflight_modules(settings, cuda=bool(gpu)), timeout=timeout
    )
    # The decision that build_eval_env made above, asked again for the log. Asking again is cheap,
    # because each libstdc++ file is read only once.
    report["libstdcxx_preload"] = libstdcxx_preload(settings.eval_python, env.get("LD_LIBRARY_PATH", ""))[1]
    problems = []
    for name, error in report["errors"].items():
        if name == MANAGER_CHECK:
            problems.append(f"cannot start multiprocessing.Manager (is TMPDIR too long for a Unix socket?): {error}")
        elif name == CUDA_CHECK:
            problems.append(
                f"cannot run a CUDA matmul on GPU {gpu} (CUDA libraries that do not match the evaluation "
                f"environment, for example from LD_LIBRARY_PATH or LD_PRELOAD?): {error}"
            )
        elif name == NLTK_PUNKT_CHECK:
            problems.append(
                "nltk's punkt_tab data, which IFEval needs, is not in the offline cache; run 'train_eval "
                f"prefetch' for this suite: {error}"
            )
        elif name == "langdetect":
            problems.append(
                "cannot import langdetect, which IFEval's language checks need; install requirements/eval.txt "
                "into the evaluation environment, or eval/in_training/requirements-sft-eval.txt into the "
                f"folder of eval_extra_pythonpath: {error}"
            )
        elif name == CHAT_RENDER_CHECK:
            problems.append(
                f"the chat template of {settings.tokenizer_source} does not render the prompts of the "
                f"think/no_think switch (eval/in_training/README.md) or does not match the "
                f"training vocabulary: {error}"
            )
        else:
            problems.append(f"cannot import {name}: {error}")
    if report["errors"]:
        # A failed import can come from the libstdc++ that the process got, so the decision about
        # the preload goes into the message.
        problems.append(f"libstdc++: {report['libstdcxx_preload']}")
    problems += lm_eval_problems(report, settings.harness_path, eval_python=settings.eval_python)
    if problems:
        raise RuntimeError("in-training eval preflight failed:\n  " + "\n  ".join(problems))
    return report
