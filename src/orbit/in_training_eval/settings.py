"""The evaluation settings of one run, and the hashes that decide what a later attempt may reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from .harness import LM_EVAL_VERSION
from .plan import CHAT_TIMEOUT_SECONDS_PER_TOKEN, EvalProcess, Plan
from .suite import Suite, load_itr_render, load_suite


@dataclass(frozen=True)
class EvalSettings:
    """The evaluation settings of one run, built from the trainer's eval_* options (from_args)."""

    every: int
    at_start: bool
    suite: Suite
    eval_python: Path
    harness_path: Optional[Path]
    extra_pythonpath: Optional[str]
    max_model_len: int
    gpu_memory_utilization: float
    timeout_minutes: float
    total_timeout_minutes: float
    weights: str
    keep_exports: bool
    cache_dir: Path
    tokenizer_source: Path
    tokenizer_config: Path
    # eval_at_end: also evaluate the last step of the run when the cadence does not reach it.
    at_end: bool = False
    # The sha256 of the chat template that the export carries, set for chat suites only.
    chat_template_sha256: Optional[str] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EvalSettings":
        settings = cls._from_args(args)
        if settings.suite.chat is None:
            return settings
        return replace(settings, chat_template_sha256=load_itr_render().template_sha256(settings.tokenizer_source))

    @classmethod
    def _from_args(cls, args: argparse.Namespace) -> "EvalSettings":
        return cls(
            every=int(args.eval_every or 0),
            at_start=bool(args.eval_at_start),
            suite=load_suite(args.eval_suite),
            eval_python=Path(args.eval_python),
            harness_path=Path(args.eval_harness_path) if args.eval_harness_path else None,
            extra_pythonpath=args.eval_extra_pythonpath,
            max_model_len=int(args.eval_max_model_len),
            gpu_memory_utilization=float(args.eval_gpu_memory_utilization),
            timeout_minutes=float(args.eval_timeout_minutes),
            total_timeout_minutes=float(args.eval_total_timeout_minutes),
            weights=str(args.eval_weights),
            keep_exports=bool(args.eval_keep_exports),
            cache_dir=Path(args.eval_cache_dir),
            tokenizer_source=Path(args.eval_tokenizer_source),
            tokenizer_config=Path(args.tokenizer_config),
            at_end=bool(getattr(args, "eval_at_end", False)),
        )

    def fingerprint(self, plan: Plan) -> str:
        """Hash of everything that decides what the process results of an evaluated step contain."""
        payload = {
            "suite": self.suite.digest,
            "plan": plan.to_dict(),
            "lm_eval": LM_EVAL_VERSION,
            "weights": self.weights,
            "max_model_len": self.max_model_len,
            "tokenizer_source": str(self.tokenizer_source),
        }
        payload.update(self._chat_payload())
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def export_key(self) -> str:
        """Hash of everything besides the step that decides what an export contains."""
        payload = {
            "weights": self.weights,
            "max_model_len": self.max_model_len,
            "tokenizer_source": str(self.tokenizer_source),
            "tokenizer_config": str(self.tokenizer_config),
        }
        payload.update(self._chat_payload())
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _chat_payload(self) -> dict[str, Any]:
        """What a chat suite adds to the fingerprint and the export key; nothing for a base suite.

        The chat template decides every chat prompt, and a template edited in place keeps its folder's
        path, so its sha256 is part of both keys. Base suites add nothing.
        """
        if self.suite.chat is None:
            return {}
        return {"chat_template_sha256": self.chat_template_sha256}

    def process_timeout(self, process: EvalProcess) -> float:
        """Wall-clock cap of one evaluation process, in seconds.

        It is three times the planned time of the process plus ten minutes, so a process that hangs
        is stopped long before the overall cap. It never exceeds eval_timeout_minutes per unit.

        A chat process may always run at least long enough for two responses at its budget, one after
        the other, at 30 ms per token, plus its startup. Its planned time comes from assumed response
        lengths, and an early SFT checkpoint that writes much longer responses must not lose its gates
        to a timeout.
        """
        planned = process.cost + self.suite.process_startup_seconds
        timeout = min(self.timeout_minutes * 60.0 * len(process.units), 3.0 * planned + 600.0)
        if process.budget:
            floor = self.suite.process_startup_seconds + 2.0 * process.budget * CHAT_TIMEOUT_SECONDS_PER_TOKEN
            timeout = max(timeout, floor)
        return timeout
