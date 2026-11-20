"""The environment and the command line of one evaluation process."""
from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import Optional

from .benchmarks import CHAT_MODEL
from .plan import EvalProcess
from .suite import Suite


# Variables that the launcher sets for the trainer. vLLM reads several of them and tries to join
# the trainer's process group, which hangs the engine, so none of them reaches an evaluation.
SCRUBBED_ENV_KEYS = {
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "NODE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
}
SCRUBBED_ENV_PREFIXES = ("TORCHELASTIC_", "GROUP_", "ROLE_", "PET_")


def is_scrubbed_env_key(key: str) -> bool:
    return key in SCRUBBED_ENV_KEYS or key.startswith(SCRUBBED_ENV_PREFIXES)


# Every evaluation process carries the UUID of its GPU in this variable, and lm-eval's code
# runners inherit it. Together with the session ids that each rank records, that is how a rank
# finds evaluation processes that an earlier, crashed trainer left behind on its GPU
# (kill_eval_processes).
EVAL_GPU_ENV = "ORBIT_EVAL_GPU"
# The trainer's PID. run_lm_eval.py watches it and kills its own process group when the trainer
# is gone, because a trainer killed with SIGKILL cannot clean up its evaluation processes.
EVAL_PARENT_ENV = "ORBIT_EVAL_PARENT_PID"


def compile_threads_per_rank(local_world_size: int) -> int:
    """CPU threads for torch.compile in one evaluation process.

    Inductor starts one compile worker per CPU by default. Every rank of a node starts its vLLM
    engine at the same moment, so the default would put eight times as many workers on the node
    as it has CPUs. The CPUs are shared out between the ranks of the node instead.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return max(1, cpus // max(1, int(local_world_size)))


def environment_libstdcxx(eval_python: Path) -> Optional[Path]:
    """The libstdc++ that the evaluation environment ships, or None when it has none of its own.

    A conda environment keeps it in <prefix>/lib, next to the C++ libraries that were built
    against it. The prefix comes from eval_python as given, and otherwise from the file it links
    to, because the python of a virtual environment links to the interpreter of the environment
    it was made from.
    """
    for python in (Path(eval_python), Path(eval_python).resolve()):
        candidate = python.parent.parent / "lib" / "libstdc++.so.6"
        if candidate.is_file():
            return candidate
    return None


# The folders where the dynamic linker finds the system's libstdc++ when no LD_LIBRARY_PATH entry
# has one. Debian and Ubuntu keep it in the multiarch folders and Red Hat in the lib64 folders.
# /lib and /usr/lib are the linker's last resort on every system.
SYSTEM_LIBRARY_DIRS = (
    "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/lib/aarch64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
    "/lib64", "/usr/lib64", "/usr/local/lib", "/lib", "/usr/lib",
)
# build_eval_env sets this variable when it leaves the environment's libstdc++ out of LD_PRELOAD,
# and the value says why. run_lm_eval.py reads it and then does not load that copy either, so the
# two places that load it never disagree.
EVAL_NO_LIBSTDCXX_ENV = "ORBIT_EVAL_NO_ENV_LIBSTDCXX"


def libstdcxx_versions(path: Path) -> dict[str, tuple[int, ...]]:
    """The newest GLIBCXX and CXXABI symbol versions that a libstdc++ file defines.

    The result looks like {"GLIBCXX": (3, 4, 34), "CXXABI": (1, 3, 15)}. A library built against
    some libstdc++ asks for the symbol versions it used, and a copy can serve the library only
    when it defines all of them. So the copy with the higher numbers serves more libraries. A
    missing or unreadable file gives an empty result.
    """
    try:
        stat = Path(path).stat()
    except OSError:
        return {}
    return dict(_read_libstdcxx_versions(str(path), stat.st_size, stat.st_mtime_ns))


# The environment's copy is about 20 MB, and reading it from a shared filesystem took 7 s.
# build_eval_env runs for every evaluation process, so each file is read once per trainer
# process. The size and the modification time are part of the cache key, so that a file that was
# replaced is read again.
@functools.lru_cache(maxsize=32)
def _read_libstdcxx_versions(path: str, size: int, mtime_ns: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return ()
    newest: dict[str, tuple[int, ...]] = {}
    # The version names sit in the file's string table, each one between two NUL bytes.
    for match in re.finditer(rb"(?<=\x00)(GLIBCXX|CXXABI)_(\d+(?:\.\d+)+)(?=\x00)", data):
        tag = match.group(1).decode()
        version = tuple(int(part) for part in match.group(2).split(b"."))
        newest[tag] = max(newest.get(tag, ()), version)
    return tuple(sorted(newest.items()))


def describe_libstdcxx_versions(versions: dict[str, tuple[int, ...]]) -> str:
    """The versions as the dynamic linker names them, for example "CXXABI_1.3.15, GLIBCXX_3.4.34"."""
    names = [f"{tag}_{'.'.join(str(part) for part in version)}" for tag, version in sorted(versions.items())]
    return ", ".join(names) or "no GLIBCXX or CXXABI version"


def libstdcxx_preload(eval_python: Path, ld_library_path: str) -> tuple[Optional[Path], str]:
    """Whether build_eval_env preloads the evaluation environment's libstdc++, and why.

    It returns the copy to preload, or None when nothing is preloaded, together with a sentence
    for the startup log.

    A newer libstdc++ serves the libraries that were built against an older one, but an older copy
    fails every library that needs a newer one. So the environment's copy is preloaded only when
    no other copy that the dynamic linker could pick instead is newer. The other copies are those
    in the folders on the LD_LIBRARY_PATH of the evaluation process and in the system's library
    folders. A copy counts as newer when it defines a newer GLIBCXX or a newer CXXABI version.
    When one is newer, the evaluation process keeps the copy that the linker finds, and the
    startup check shows whether that copy serves every import.
    """
    own = environment_libstdcxx(eval_python)
    if own is None:
        return None, f"nothing is preloaded, because the environment of {eval_python} has no lib/libstdc++.so.6"
    own_versions = libstdcxx_versions(own)
    described = f"the environment's {own} ({describe_libstdcxx_versions(own_versions)})"
    folders = [entry for entry in ld_library_path.split(":") if entry] + list(SYSTEM_LIBRARY_DIRS)
    for folder in folders:
        other = Path(folder) / "libstdc++.so.6"
        newer = {
            tag: version for tag, version in libstdcxx_versions(other).items()
            if version > own_versions.get(tag, ())
        }
        if newer:
            return None, (
                f"{described} is not preloaded, because {other} defines the newer "
                f"{describe_libstdcxx_versions(newer)}"
            )
    return own, f"{described} is preloaded"


def build_eval_env(
    base_env: dict[str, str],
    *,
    eval_python: Path,
    gpu: str,
    cache_dir: Optional[Path],
    harness_path: Optional[Path],
    extra_pythonpath: Optional[str],
    rank_cache: Optional[Path] = None,
    offline: bool = True,
    parent_pid: Optional[int] = None,
    gpu_uuid: Optional[str] = None,
    compile_threads: Optional[int] = None,
) -> dict[str, str]:
    """Environment of one evaluation process on one rank.

    ``cache_dir`` is the shared offline cache of datasets and metrics. ``rank_cache`` holds the
    compile caches of one rank of one run (see InTrainingEval.rank_cache).
    """
    env = {key: value for key, value in base_env.items() if not is_scrubbed_env_key(key)}
    pythonpath = [str(harness_path)] if harness_path is not None else []
    pythonpath += [entry for entry in (extra_pythonpath or "").split(os.pathsep) if entry]
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    # Tools that vLLM starts by name (for example ninja) must come from the evaluation environment.
    env["PATH"] = str(Path(eval_python).parent) + os.pathsep + env.get("PATH", "")
    # A process has one libstdc++ for all of its libraries: the first copy that any library loads.
    # An evaluation process loads torch first, and torch's libraries do not look in the
    # environment's lib folder, so they get the copy that LD_LIBRARY_PATH or the system offers.
    # When the evaluation environment ships a newer libstdc++ than the system, that copy is older
    # than the environment's and can lack CXXABI_1.3.15. RULER then fails, because nltk imports
    # sqlite3, and sqlite3's ICU library from the environment needs that version. Preloading the
    # environment's own copy makes it the first one in every evaluation process and in the
    # processes they start. The environment's C++ libraries were built against it.
    # libstdcxx_preload decides whether the copy is preloaded, because an older copy would do harm
    # where the system has a newer one.
    # LD_LIBRARY_PATH is left as it is, because the trainer's environment may point it at the GPU
    # driver's libraries, which vLLM needs.
    libstdcxx, reason = libstdcxx_preload(eval_python, env.get("LD_LIBRARY_PATH", ""))
    env.pop(EVAL_NO_LIBSTDCXX_ENV, None)
    if libstdcxx is None:
        env[EVAL_NO_LIBSTDCXX_ENV] = reason
    else:
        preload = env.get("LD_PRELOAD", "")
        # The dynamic linker accepts colons and spaces between the entries.
        if str(libstdcxx) not in preload.replace(":", " ").split():
            env["LD_PRELOAD"] = f"{preload}:{libstdcxx}" if preload.strip() else str(libstdcxx)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    if parent_pid is not None:
        env[EVAL_PARENT_ENV] = str(int(parent_pid))
    if gpu_uuid:
        env[EVAL_GPU_ENV] = gpu_uuid
    if compile_threads is not None:
        env["TORCHINDUCTOR_COMPILE_THREADS"] = str(int(compile_threads))
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        # Packages in the user's ~/.local would shadow those of the evaluation environment.
        PYTHONNOUSERSITE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_ALLOW_CODE_EVAL="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
        VLLM_NO_USAGE_STATS="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        # FlashInfer's sampler is compiled with nvcc on first use. vLLM's own sampler needs no
        # compiler, so an evaluation environment without the CUDA toolkit still works.
        VLLM_USE_FLASHINFER_SAMPLER="0",
        # DeepGEMM serves only FP8 layers, and OrbitLM evaluates in bf16. The copy bundled with
        # vLLM 0.22 may fail to load, and its warmup then kills the engine.
        VLLM_USE_DEEP_GEMM="0",
        # The GPU index in CUDA_VISIBLE_DEVICES is a physical index in PCI bus order (GpuHandle).
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        # With torch 2.10 and later, vLLM loads its AOT-compiled graph from the cache on every run
        # after the first and then mis-measures free memory (about 4.6 GiB of KV cache instead
        # of about 110 on an H200). AOT is off; the regular compile cache below still works.
        VLLM_USE_AOT_COMPILE="0",
    )
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        env.update(
            HF_HOME=str(cache_dir / "hf"),
            HF_DATASETS_CACHE=str(cache_dir / "hf" / "datasets"),
        )
        if (cache_dir / "nltk").is_dir():
            env["NLTK_DATA"] = str(cache_dir / "nltk")
    if rank_cache is not None:
        own = Path(rank_cache)
        env.update(
            # Compile caches and the copied remote model code are per rank. Many ranks share one
            # folder on a shared filesystem, and two processes writing the same cache file at once
            # can break it.
            HF_MODULES_CACHE=str(own / "hf_modules"),
            VLLM_CACHE_ROOT=str(own / "vllm"),
            TORCHINDUCTOR_CACHE_DIR=str(own / "inductor"),
            TRITON_CACHE_DIR=str(own / "triton"),
            FLASHINFER_WORKSPACE_BASE=str(own / "flashinfer"),
            XDG_CACHE_HOME=str(own / "xdg"),
        )
    if offline:
        env.update(
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_EVALUATE_OFFLINE="1",
        )
    return env


def prepare_rank_cache(cache_dir: Path, rank_cache: Path) -> None:
    """Creates the per-rank cache folders and links the shared evaluate metrics into them.

    HF_MODULES_CACHE is per rank, but the code_eval metric of HumanEval and MBPP was cached once
    by prefetch under the shared HF_HOME. evaluate looks for it under HF_MODULES_CACHE, so the
    rank's folder gets a link to the shared copy.
    """
    own = Path(rank_cache)
    for name in ("hf_modules", "vllm", "inductor", "triton", "flashinfer", "xdg"):
        (own / name).mkdir(parents=True, exist_ok=True)
    shared = Path(cache_dir) / "hf" / "modules" / "evaluate_modules"
    link = own / "hf_modules" / "evaluate_modules"
    if shared.is_dir() and not link.exists():
        try:
            link.symlink_to(shared, target_is_directory=True)
        except FileExistsError:
            pass


def build_lm_eval_command(
    process: EvalProcess,
    *,
    eval_python: Path,
    suite: Suite,
    model_dir: Path,
    out_dir: Path,
    max_model_len: int,
    gpu_memory_utilization: float,
    samples_path: Optional[Path],
) -> list[str]:
    """The command line of one evaluation process."""
    model_args = (
        f"pretrained={model_dir},dtype=bfloat16,trust_remote_code=True,"
        f"gpu_memory_utilization={gpu_memory_utilization:.3f},max_model_len={max_model_len},seed=1234"
    )
    if suite.vllm_model_args:
        model_args += "," + suite.vllm_model_args
    signature = list(process.signature)
    model_name = "vllm"
    if "--model_args" in signature:
        # A chat process: its own model arguments select the prompt mode of the model subclass, which
        # run_lm_eval.py registers under CHAT_MODEL. lm-eval would merge a second --model_args too, but
        # one merged string keeps command.json readable.
        position = signature.index("--model_args")
        model_args += "," + signature[position + 1]
        del signature[position:position + 2]
        model_name = CHAT_MODEL
    model = ["--model", model_name, "--model_args", model_args, "--batch_size", "auto"]
    if "--metadata" in signature:
        # RULER builds its prompts with the tokenizer named in the metadata.
        position = signature.index("--metadata") + 1
        metadata = json.loads(signature[position])
        metadata["pretrained"] = str(model_dir)
        signature[position] = json.dumps(metadata, sort_keys=True)
    command = [
        str(eval_python),
        str(suite.entry),
        *model,
        "--include_path",
        str(suite.tasks_dir),
        "--tasks",
        ",".join(process.tasks),
        *signature,
        "--output_path",
        str(out_dir),
        # CRUXEval, HumanEval and MBPP execute generated code; the flag is lm-eval's explicit consent.
        "--confirm_run_unsafe_code",
    ]
    if samples_path is not None:
        command += ["--samples", str(samples_path)]
    return command
