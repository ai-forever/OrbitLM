"""The GPU of a rank and the evaluation processes on it: running them, and killing them and their leftovers."""
from __future__ import annotations

import gc
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch

from .environment import EVAL_GPU_ENV
from .files import find_results_file, newest_mtime, process_outputs_ready


class GpuHandle:
    """The trainer's own GPU, as the evaluation processes of this rank see it.

    On a CPU-only machine (the unit tests) every method is a harmless no-op.
    """

    def __init__(self, device: Optional[torch.device]):
        usable = device is not None and device.type == "cuda" and torch.cuda.is_available()
        self.device = device if usable else None
        self._uuid: Optional[str] = None

    def _index(self) -> int:
        assert self.device is not None
        return self.device.index if self.device.index is not None else torch.cuda.current_device()

    def uuid(self) -> str:
        """The GPU's UUID in nvidia-smi's form ("GPU-..."), or "" when it is unknown."""
        if self.device is None:
            return ""
        if self._uuid is None:
            try:
                uuid = str(torch.cuda.get_device_properties(self._index()).uuid)
                self._uuid = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
            except Exception:
                self._uuid = ""
        return self._uuid

    def trainer_needs(self) -> int:
        """Bytes the trainer will reserve again at its peak, beyond what it holds right now."""
        if self.device is None:
            return 0
        return max(0, int(torch.cuda.max_memory_reserved(self.device)) - int(torch.cuda.memory_reserved(self.device)))

    def visible_id(self) -> str:
        """The CUDA_VISIBLE_DEVICES value that shows the child exactly this GPU.

        vLLM accepts only numeric indices there, not GPU UUIDs. The trainer itself may run under
        a CUDA_VISIBLE_DEVICES that renumbers the GPUs, so its device index is not enough: the
        GPU's UUID is looked up in nvidia-smi, which lists the physical indices in PCI bus order.
        The child gets CUDA_DEVICE_ORDER=PCI_BUS_ID, so CUDA counts in the same order.
        """
        if self.device is None:
            return ""
        index = self._index()
        uuid = self.uuid()
        if uuid:
            try:
                listing = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=60,
                ).stdout
                for line in listing.splitlines():
                    physical, _, listed_uuid = (part.strip() for part in line.partition(","))
                    if listed_uuid == uuid and physical.isdigit():
                        return physical
            except (OSError, subprocess.TimeoutExpired):
                pass
        parent = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
        if index < len(parent) and parent[index].isdigit():
            return parent[index]
        return str(index)

    def release_cached_memory(self) -> None:
        if self.device is None:
            return
        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()

    def memory(self) -> Optional[tuple[int, int]]:
        """(free, total) bytes on the whole GPU, counting every process on it."""
        if self.device is None:
            return None
        free, total = torch.cuda.mem_get_info(self.device)
        return int(free), int(total)

    def wait_until_free(self, target_free: int, *, timeout: float, poll: float = 5.0) -> bool:
        if self.device is None:
            return True
        end = time.monotonic() + timeout
        while True:
            free, _ = self.memory()
            if free >= target_free:
                return True
            if time.monotonic() >= end:
                return False
            time.sleep(poll)


def kill_process_group(process: subprocess.Popen, *, grace: float = 10.0) -> None:
    """Stops every process of a child's session, including vLLM's EngineCore.

    The EngineCore is a grandchild that can outlive lm-eval and keep almost the whole GPU, so the
    whole process group is signalled, first gently and then with SIGKILL.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        end = time.monotonic() + grace
        while time.monotonic() < end:
            # An exited child stays a zombie member of its group until it is reaped, and the
            # group would look alive for the whole grace period.
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except (ProcessLookupError, PermissionError):
                break
            time.sleep(0.5)
        else:
            continue
        break
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def _process_ids(proc_root: Path) -> list[int]:
    try:
        return [int(entry.name) for entry in Path(proc_root).iterdir() if entry.name.isdigit()]
    except OSError:
        return []


def _session_of(pid: int, proc_root: Path) -> Optional[int]:
    """The session id of a live process from /proc/<pid>/stat, or None (gone or a zombie)."""
    try:
        stat = (Path(proc_root) / str(pid) / "stat").read_text()
    except OSError:
        return None
    # The command name in parentheses may contain spaces, so the fields are counted after it.
    fields = stat.rsplit(")", 1)[1].split()
    if fields[0] == "Z":
        return None
    return int(fields[3])


def _looks_like_eval(pid: int, gpu_uuid: str, proc_root: Path) -> bool:
    """Whether a process is one of the evaluation's: the wrapper, vLLM's engine, or tagged."""
    try:
        cmdline = (Path(proc_root) / str(pid) / "cmdline").read_bytes()
    except OSError:
        return False
    if b"run_lm_eval" in cmdline or cmdline.startswith(b"VLLM::"):
        return True
    if gpu_uuid:
        try:
            environ = (Path(proc_root) / str(pid) / "environ").read_bytes()
        except OSError:
            return False
        return f"{EVAL_GPU_ENV}={gpu_uuid}".encode() in environ.split(b"\0")
    return False


def find_eval_processes(gpu_uuid: str, proc_root: Path = Path("/proc")) -> list[int]:
    """PIDs of evaluation processes for this GPU, found by the EVAL_GPU_ENV tag in their environment.

    Only the user's own processes can be read, which are the only ones the trainer could have
    started. The trainer process itself never carries the tag. vLLM's EngineCore is not found
    this way: it sets its process title, which overwrites the memory that /proc shows as its
    environment. find_eval_sessions covers it through its session.
    """
    if not gpu_uuid:
        return []
    needle = f"{EVAL_GPU_ENV}={gpu_uuid}".encode()
    own = os.getpid()
    found: list[int] = []
    for pid in _process_ids(proc_root):
        if pid == own:
            continue
        try:
            environ = (Path(proc_root) / str(pid) / "environ").read_bytes()
        except OSError:
            continue
        if needle in environ.split(b"\0"):
            found.append(pid)
    return sorted(found)


def find_eval_sessions(gpu_uuid: str, sessions: Iterable[int] = (), proc_root: Path = Path("/proc")) -> list[int]:
    """Every live process of the evaluation sessions on this GPU.

    The trainer starts each evaluation process as the leader of a new session, and its children
    (vLLM's EngineCore, the code runners of HumanEval, MBPP and CRUXEval) stay in that session. A
    session counts when it is listed in ``sessions`` (recorded by the trainer when it started
    the process) or holds a tagged process, and when one of its members looks like an
    evaluation process, which guards against a recorded session id that was reused since. The
    trainer's own session is never searched, but tagged processes always count.
    """
    tagged = find_eval_processes(gpu_uuid, proc_root)
    candidates = set(int(sid) for sid in sessions)
    for pid in tagged:
        sid = _session_of(pid, proc_root)
        if sid is not None:
            candidates.add(sid)
    try:
        candidates.discard(os.getsid(0))
    except OSError:
        pass
    if not candidates:
        return tagged
    members: dict[int, list[int]] = {}
    own = os.getpid()
    for pid in _process_ids(proc_root):
        if pid == own:
            continue
        sid = _session_of(pid, proc_root)
        if sid in candidates:
            members.setdefault(sid, []).append(pid)
    found = set(tagged)
    for pids in members.values():
        if any(_looks_like_eval(pid, gpu_uuid, proc_root) for pid in pids):
            found.update(pids)
    return sorted(found)


def kill_eval_processes(gpu_uuid: str, sessions: Iterable[int] = (), proc_root: Path = Path("/proc")) -> list[int]:
    """SIGKILLs every process of the evaluation sessions on this GPU and returns their PIDs.

    A rank calls this only when none of its own evaluation processes should be running: at
    startup, before its first process, and when the GPU memory did not come back after one.
    Anything found then is a leftover, for example vLLM's EngineCore from a trainer that was
    killed with SIGKILL while its wrapper was killed too.
    """
    pids = find_eval_sessions(gpu_uuid, sessions, proc_root)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return pids


def run_eval_process(
    command: list[str],
    env: dict[str, str],
    proc_dir: Path,
    *,
    timeout: float,
    deadline: float,
    poll_seconds: float,
    quiet_seconds: float,
    should_stop: Callable[[], bool] = lambda: False,
    on_start: Optional[Callable[[int], Any]] = None,
    chat: bool = False,
) -> dict[str, Any]:
    """Runs one lm-eval process in its own session and always cleans it up.

    The own session lets the whole process group, vLLM's EngineCore included, be killed at once. It
    also means that the launcher's kill does not reach the group when the trainer dies, which is why
    run_lm_eval.py watches the trainer's PID itself (EVAL_PARENT_ENV). ``on_start`` receives the PID,
    which is also the session id of everything the process starts.

    lm-eval sometimes writes its results and then hangs in vLLM's shutdown. Once the outputs are
    complete and nothing in the folder has changed for ``quiet_seconds``, the process is stopped and
    counted as done. A chat process writes its records after lm-eval returns, so its outputs are
    complete only when the records (or the records error) exist too.
    """
    started = time.monotonic()
    status = "failed"
    rc: Optional[int] = None
    with (Path(proc_dir) / "log.txt").open("ab") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(proc_dir), start_new_session=True,
        )
        if on_start is not None:
            on_start(process.pid)
        try:
            while True:
                rc = process.poll()
                if rc is not None:
                    status = "exited"
                    break
                now = time.monotonic()
                if process_outputs_ready(proc_dir, chat) and time.time() - newest_mtime(proc_dir) > quiet_seconds:
                    status = "exited"
                    break
                if now - started > timeout:
                    status = "timeout"
                    break
                if now >= deadline:
                    status = "deadline"
                    break
                if should_stop():
                    status = "stopped"
                    break
                time.sleep(poll_seconds)
        finally:
            kill_process_group(process)
    if status == "exited":
        status = "ok" if find_results_file(proc_dir) is not None else "failed"
    return {"status": status, "rc": rc, "seconds": time.monotonic() - started}
