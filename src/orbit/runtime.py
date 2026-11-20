"""Small runtime helpers of the trainer: tokenizer ids, seeds, signals, metric reporting, FA3 preflight."""
from __future__ import annotations

import json
import random
import signal
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .experiment_config import default_attn_kernel_path_from_env

FA3_KERNEL_IMPL = "kernels-community/vllm-flash-attn3"


class TokenizerConfig:
    """The vocabulary size and the special-token ids of a tokenizer_config.json."""

    def __init__(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        added = payload.get("added_tokens_decoder", {})
        token_to_id = {}
        for key, value in added.items():
            token_to_id[value.get("content")] = int(key)

        if "vocab_size" not in payload:
            # The model's embedding size is taken from here, so a guess would build a model
            # that cannot load the weights of an export.
            raise ValueError(
                f"{path} has no vocab_size. The OrbitLM tokenizers set it to 151936 "
                "(data_prep prepares them)."
            )
        self.vocab_size = int(payload["vocab_size"])
        self.bos_token = payload.get("bos_token")
        self.eos_token = payload.get("eos_token")
        self.pad_token = payload.get("pad_token")
        self.bos_id = token_to_id.get(self.bos_token)
        self.eos_id = token_to_id.get(self.eos_token)
        self.pad_id = token_to_id.get(self.pad_token)


def load_model_config(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return dict(payload)


def read_clearml_task_id(path: Path) -> Optional[str]:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return value or None


def write_clearml_task_id(path: Path, task_id: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{task_id}\n", encoding="utf-8")
    except Exception:
        pass


class GracefulStopController:
    """Capture termination signals and request orderly training stop."""

    def __init__(self, rank: int):
        self.rank = int(rank)
        self._stop_event = threading.Event()
        self._installed: list[tuple[int, Any]] = []

    def install(self) -> None:
        def _handler(signum, _frame):
            if not self._stop_event.is_set() and self.rank == 0:
                print(
                    f"received_signal={int(signum)} graceful_stop_requested=1",
                    flush=True,
                )
            self._stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)
                signal.signal(sig, _handler)
                self._installed.append((sig, prev))
            except Exception:
                continue

    def restore(self) -> None:
        for sig, prev in self._installed:
            try:
                signal.signal(sig, prev)
            except Exception:
                pass
        self._installed.clear()

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()


def clearml_report_scalar(logger, title: str, series: str, value, step: int):
    if logger is None or value is None:
        return False
    try:
        logger.report_scalar(title=title, series=series, value=float(value), iteration=int(step))
        return True
    except Exception as exc:
        if not getattr(clearml_report_scalar, "_warned", False):
            print(f"WARNING: ClearML report_scalar failed: {exc}", flush=True)
            clearml_report_scalar._warned = True
        return False


def tb_report_scalar(writer, tag: str, value, step: int):
    if writer is None or value is None:
        return
    try:
        writer.add_scalar(tag, float(value), int(step))
    except Exception:
        pass


def report_scalar(clearml_logger, tb_writer, title: str, series: str, value, step: int):
    clearml_report_scalar(clearml_logger, title, series, value, step)
    tb_report_scalar(tb_writer, f"{title}/{series}", value, step)


def set_global_random_seeds(base_seed: int, rank: int) -> int:
    seed = int(base_seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def format_count(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    abs_value = abs(value)
    for suffix, unit in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs_value >= unit:
            val = value / unit
            if abs(val) >= 100:
                return f"{val:.0f}{suffix}"
            if abs(val) >= 10:
                return f"{val:.1f}{suffix}"
            return f"{val:.2f}{suffix}"
    return str(value)


def preflight_flash_attention(
    attn_impl: str,
    *,
    attn_kernel_path: Optional[Path] = None,
) -> None:
    """Loads the FA3 kernel once before the model is built, so a broken kernel stops the run early.

    The kernel comes from attn_kernel_path, else from ORBIT_ATTN_KERNEL_PATH, else from the HF Hub.
    eager and sdpa need no kernel.
    """
    if attn_impl != FA3_KERNEL_IMPL:
        return
    try:
        from kernels import get_kernel, get_local_kernel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"{FA3_KERNEL_IMPL} requested but kernels could not be imported: {exc}"
        ) from exc
    try:
        if attn_kernel_path is None:
            attn_kernel_path = default_attn_kernel_path_from_env()
        if attn_kernel_path is not None:
            kernel_repo_path = Path(attn_kernel_path).expanduser().resolve()
            if not kernel_repo_path.exists():
                raise RuntimeError(
                    f"Local attention kernel path does not exist: {kernel_repo_path}"
                )
            kernel = get_local_kernel(kernel_repo_path, "vllm_flash_attn3")
        else:
            kernel = get_kernel(FA3_KERNEL_IMPL)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"{FA3_KERNEL_IMPL} requested but kernel load failed: {exc}"
        ) from exc
    if not hasattr(kernel, "flash_attn_func"):
        raise RuntimeError(
            f"{FA3_KERNEL_IMPL} loaded, but flash_attn_func is missing."
        )
