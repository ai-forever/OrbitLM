"""Files that the ranks share during a pause: JSON files, the file barrier, heartbeats and process outputs."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def wait_for_files(
    paths: Iterable[Path],
    *,
    deadline: float,
    poll_seconds: float,
    should_stop: Optional[Callable[[], bool]] = None,
    given_up: Optional[Callable[[Path], bool]] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Polls until every path exists or the deadline passes, and returns the missing ones.

    This is the barrier of the evaluation. The ranks wait on files in the shared run folder, not
    inside an NCCL collective, because an evaluation can take longer than the NCCL timeout, and a
    collective that times out kills the whole job.

    ``given_up(path)`` says that the writer of a path is dead; such a path is no longer waited for
    and counts as missing. A filesystem error while checking a path also counts as "not there
    yet", so a hiccup of a shared filesystem never raises out of the barrier.
    """
    order = [Path(p) for p in paths]
    pending = list(order)
    abandoned: set[Path] = set()
    while True:
        still = []
        for path in pending:
            # os.path.exists never raises: any OSError (ESTALE, EIO, EACCES) reads as False.
            if os.path.exists(path):
                continue
            if given_up is not None and given_up(path):
                abandoned.add(path)
                continue
            still.append(path)
        pending = still
        stopping = should_stop is not None and should_stop()
        if not pending or clock() >= deadline or stopping:
            missing = set(pending) | abandoned
            return [path for path in order if path in missing]
        sleep(min(poll_seconds, max(0.0, deadline - clock())))


class Heartbeat:
    """A small file that one rank rewrites every few seconds while it evaluates.

    The other ranks read it to notice a rank that died during the pause, because a dead rank's
    file stops changing. The file keeps a fixed width and is rewritten in place, so a beat needs
    no new disk blocks and keeps working when the disk fills up during the evaluation.
    """

    WIDTH = 48

    def __init__(self, path: Path, interval: float):
        self.path = Path(path)
        self.interval = float(interval)
        self._count = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def beat(self) -> None:
        self._count += 1
        text = f"{self._count} {time.time():.1f}".ljust(self.WIDTH)[: self.WIDTH]
        try:
            try:
                with self.path.open("r+", encoding="ascii") as handle:
                    handle.write(text)
            except FileNotFoundError:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(text, encoding="ascii")
        except OSError:
            pass  # a missed beat is harmless; only a long silence counts

    def start(self) -> None:
        self.beat()

        def loop() -> None:
            while not self._stop.wait(self.interval):
                self.beat()

        self._thread = threading.Thread(target=loop, name="in_training_eval_heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 5.0)


class HeartbeatWatcher:
    """Tells whether a rank's heartbeat has stopped changing for longer than ``stale_seconds``.

    Only a heartbeat that was seen and then went silent counts as dead. A file that never appeared
    or cannot be read counts as alive, so a filesystem problem never makes a live rank look dead;
    the shared deadlines still bound every wait. Changes are timed with this process's own clock,
    so clock differences between nodes do not matter.
    """

    def __init__(self, folder: Path, stale_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.folder = Path(folder)
        self.stale_seconds = float(stale_seconds)
        self.clock = clock
        self._seen: dict[int, tuple[str, float]] = {}

    def dead(self, rank: int) -> bool:
        try:
            content = (self.folder / f"rank_{rank:05d}").read_text(encoding="ascii")
        except (OSError, ValueError):
            self._seen.pop(rank, None)
            return False
        now = self.clock()
        previous = self._seen.get(rank)
        if previous is None or previous[0] != content:
            self._seen[rank] = (content, now)
            return False
        return now - previous[1] > self.stale_seconds


def find_results_file(folder: Path) -> Optional[Path]:
    """The newest lm-eval results file under a process folder, if any."""
    candidates: list[tuple[float, Path]] = []
    try:
        for path in Path(folder).glob("**/results_*.json"):
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
    except OSError:
        return None
    return max(candidates)[1] if candidates else None


def newest_mtime(folder: Path) -> float:
    newest = 0.0
    try:
        for path in Path(folder).rglob("*"):
            try:
                if path.is_file():
                    newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return newest


# Files that the chat model subclass (eval/in_training/itr_chat.py) writes into lm-eval's output folder
# of a process, after lm-eval has finished: one compact line per response for the gates, or, when that
# fails, the error. The process counts as finished only when one of them exists.
CHAT_RECORDS_NAME = "itr_records.jsonl"
CHAT_RECORDS_ERROR_NAME = "itr_records_error.json"


def chat_records_path(proc_dir: Path) -> Path:
    return Path(proc_dir) / "lm_eval" / CHAT_RECORDS_NAME


def process_outputs_ready(proc_dir: Path, chat: bool) -> bool:
    """Whether a process has written everything it will write: its results, and for chat its records."""
    if find_results_file(proc_dir) is None:
        return False
    if not chat:
        return True
    return chat_records_path(proc_dir).exists() or (Path(proc_dir) / "lm_eval" / CHAT_RECORDS_ERROR_NAME).exists()
