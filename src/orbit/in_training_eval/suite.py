"""Suite files (eval/in_training/suites/*.yaml): parsing, checking, and the repository files they name."""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..experiment_config import _load_yaml_text
from .benchmarks import BENCHMARKS, RULER_LENGTHS

# The repository root: this file is src/orbit/in_training_eval/suite.py.
REPO_ROOT = Path(__file__).resolve().parents[3]
# The suite that runs when a config sets no eval_suite, which is the 32K context stage's.
DEFAULT_SUITE_PATH = REPO_ROOT / "eval" / "in_training" / "suites" / "base.yaml"

SUITE_KEYS = {
    "tasks_dir",
    "entry",
    "process_startup_seconds",
    "vllm_model_args",
    "benchmarks",
    "chat",
}
BENCHMARK_KEYS = {"limit", "shards"}
RULER_KEYS = {"limit", "lengths", "num_samples"}
# A chat benchmark may set its own budget; it then runs in processes of its own, because lm-eval
# applies --gen_kwargs to every task of a process.
CHAT_BENCHMARK_KEYS = BENCHMARK_KEYS | {"max_gen_toks"}

# The keys of a suite's chat section. The modes hold the sampling preset of each prompt mode.
CHAT_SECTION_KEYS = {"think", "no_think", "no_prefill", "lead_eos", "detectors", "gates"}
# Sampling keys that lm-eval passes on to vLLM's SamplingParams, and the budget.
CHAT_PRESET_KEYS = {
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty",
    "max_gen_toks", "repetition_detection",
}
REPETITION_DETECTION_KEYS = {"min_pattern_size", "max_pattern_size", "min_count"}
# Thresholds of the detectors (eval/in_training/README.md). They are
# starting values that a calibration on a reference model may change, so a suite may override them.
DEFAULT_DETECTORS = {"loop_min_tokens": 512.0, "loop_rep16": 0.5, "short_think_tokens": 32.0}
# Thresholds of the gates; without them a suite reports the rates and bounds but no pass flags.
GATE_KEYS = {
    "g2_max", "g3_min", "g4_min", "g5_max", "g6_tags_max", "g6_markers_max", "g6_len_p50_max",
    "g7_math500_min", "g7_gsm8k_min", "g11_min",
}
# lm-eval cuts the start of a prompt that does not fit next to the budget in the model window, with
# only a log warning. A chat unit's budget plus this room must fit the window, or the run is refused.
CHAT_PROMPT_ROOM = 2048


@dataclass(frozen=True)
class BenchmarkSettings:
    """What the suite file says about one benchmark."""

    limit: Optional[int] = None
    shards: int = 1
    lengths: tuple[int, ...] = ()
    num_samples: int = 0
    max_gen_toks: Optional[int] = None


@dataclass(frozen=True)
class ChatSettings:
    """The chat section of a suite: how SFT checkpoints are prompted, sampled and gated.

    ``presets`` maps each prompt mode to the generation arguments that lm-eval receives through
    --gen_kwargs, with do_sample already decided. ``lead_eos`` says whether every prompt starts with
    one <|endoftext|>.
    """

    presets: dict[str, dict[str, Any]]
    lead_eos: bool = True
    detectors: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DETECTORS))
    gates: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Suite:
    """A parsed suite file. Paths are absolute, resolved against the suite file's folder."""

    path: Path
    tasks_dir: Path
    entry: Path
    benchmarks: dict[str, BenchmarkSettings]
    process_startup_seconds: float
    vllm_model_args: str
    digest: str
    chat: Optional[ChatSettings] = None

    @property
    def ruler_lengths(self) -> tuple[int, ...]:
        settings = self.benchmarks.get("ruler")
        return settings.lengths if settings is not None else ()


def _positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{what} must be a positive integer, got {value!r}.")
    return int(value)


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{what} must be a number, got {value!r}.")
    return float(value)


def _load_preset(path: Path, mode: str, raw: Any) -> dict[str, Any]:
    """One mode's sampling preset, checked, with do_sample decided from the temperature.

    lm-eval forces greedy decoding when a task says do_sample: false, even with a temperature from
    the command line, so do_sample always travels with the temperature. A seed is refused: lm-eval
    would give every request of a process that same seed, vLLM would then draw the same random
    numbers for every prompt, and the outcomes of different prompts would no longer be independent.
    """
    where = f"{path}: chat.{mode}"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping of sampling settings.")
    if "seed" in raw:
        raise ValueError(
            f"{where} sets a seed. lm-eval would pass it to every request, so every prompt would draw the "
            "same random numbers and the gates' bounds would not hold. Leave it out; the engine seed fixes "
            "the run."
        )
    unknown = sorted(set(raw) - CHAT_PRESET_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}; known keys are {sorted(CHAT_PRESET_KEYS)}.")
    if "max_gen_toks" not in raw:
        raise ValueError(f"{where} needs max_gen_toks; lm-eval's default of 256 tokens would cut the answers.")
    preset: dict[str, Any] = {"max_gen_toks": _positive_int(raw["max_gen_toks"], f"{where}.max_gen_toks")}
    for key in ("temperature", "top_p", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty"):
        if key in raw:
            preset[key] = _number(raw[key], f"{where}.{key}")
    if "top_k" in raw:
        preset["top_k"] = _positive_int(raw["top_k"], f"{where}.top_k")
    temperature = preset.get("temperature")
    if temperature is None or temperature < 0:
        raise ValueError(f"{where} needs a temperature of 0 or more.")
    preset["do_sample"] = temperature > 0
    detection = raw.get("repetition_detection")
    if detection is not None:
        if not isinstance(detection, dict) or set(detection) != REPETITION_DETECTION_KEYS:
            raise ValueError(f"{where}.repetition_detection must set exactly {sorted(REPETITION_DETECTION_KEYS)}.")
        values = {key: _positive_int(detection[key], f"{where}.repetition_detection.{key}") for key in detection}
        # vLLM's own rules (RepetitionDetectionParams), checked here so a bad value stops the run at
        # startup and not in every evaluation process.
        if values["min_pattern_size"] > values["max_pattern_size"] or values["min_count"] < 2:
            raise ValueError(f"{where}.repetition_detection needs min_pattern_size <= max_pattern_size and min_count >= 2.")
        preset["repetition_detection"] = values
    return preset


def _load_chat(path: Path, raw: Any, chat_benchmarks: list[str]) -> ChatSettings:
    """The chat section of a suite file, checked against the chat benchmarks it lists."""
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 'chat' must be a mapping.")
    unknown = sorted(set(raw) - CHAT_SECTION_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown chat keys {unknown}; known keys are {sorted(CHAT_SECTION_KEYS)}.")
    modes = {BENCHMARKS[name].chat_mode for name in chat_benchmarks}
    missing = sorted(mode for mode in modes if mode not in raw)
    if missing:
        raise ValueError(f"{path}: the chat benchmarks need sampling presets for {missing} in the chat section.")
    presets = {mode: _load_preset(path, mode, raw[mode]) for mode in sorted(modes)}
    lead_eos = raw.get("lead_eos", True)
    if not isinstance(lead_eos, bool):
        raise ValueError(f"{path}: chat.lead_eos must be true or false.")
    detectors = dict(DEFAULT_DETECTORS)
    raw_detectors = raw.get("detectors") or {}
    if not isinstance(raw_detectors, dict) or set(raw_detectors) - set(DEFAULT_DETECTORS):
        raise ValueError(f"{path}: chat.detectors may only set {sorted(DEFAULT_DETECTORS)}.")
    detectors.update({key: _number(value, f"{path}: chat.detectors.{key}") for key, value in raw_detectors.items()})
    raw_gates = raw.get("gates") or {}
    if not isinstance(raw_gates, dict) or set(raw_gates) - GATE_KEYS:
        raise ValueError(f"{path}: chat.gates may only set {sorted(GATE_KEYS)}.")
    gates = {key: _number(value, f"{path}: chat.gates.{key}") for key, value in raw_gates.items()}
    return ChatSettings(presets=presets, lead_eos=lead_eos, detectors=detectors, gates=gates)


def load_suite(path: Path) -> Suite:
    """Reads and checks a suite file. Every problem raises ValueError with the reason."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Evaluation suite file not found: {path}")
    text = path.read_text(encoding="utf-8")
    payload = _load_yaml_text(text, source=path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: the suite file must be a mapping.")
    unknown = sorted(set(payload) - SUITE_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown suite keys {unknown}; known keys are {sorted(SUITE_KEYS)}.")

    raw_benchmarks = payload.get("benchmarks")
    if not isinstance(raw_benchmarks, dict) or not raw_benchmarks:
        raise ValueError(f"{path}: 'benchmarks' must be a non-empty mapping.")
    benchmarks: dict[str, BenchmarkSettings] = {}
    for name, raw in raw_benchmarks.items():
        name = str(name)
        if name not in BENCHMARKS:
            raise ValueError(
                f"{path}: unknown benchmark {name!r}; known benchmarks are {sorted(BENCHMARKS)}."
            )
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: the settings of benchmark {name!r} must be a mapping.")
        if name == "ruler":
            allowed = RULER_KEYS
        elif BENCHMARKS[name].chat_mode is not None:
            allowed = CHAT_BENCHMARK_KEYS
        else:
            allowed = BENCHMARK_KEYS
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"{path}: unknown keys {unknown} for benchmark {name!r}; known keys are {sorted(allowed)}.")
        limit = raw.get("limit")
        if limit is not None:
            limit = _positive_int(limit, f"{path}: {name}.limit")
        if name == "ruler":
            lengths = raw.get("lengths")
            if not isinstance(lengths, (list, tuple)) or not lengths:
                raise ValueError(f"{path}: ruler.lengths must be a non-empty list of sequence lengths.")
            lengths = tuple(sorted({_positive_int(x, f"{path}: ruler.lengths entry") for x in lengths}))
            bad = [x for x in lengths if x not in RULER_LENGTHS]
            if bad:
                raise ValueError(f"{path}: ruler.lengths {bad} are not among the supported {list(RULER_LENGTHS)}.")
            num_samples = _positive_int(raw.get("num_samples", 100), f"{path}: ruler.num_samples")
            if num_samples > 500:
                raise ValueError(f"{path}: ruler.num_samples is at most 500, the size of the stock RULER sets.")
            benchmarks[name] = BenchmarkSettings(limit=limit, lengths=lengths, num_samples=num_samples)
            continue
        shards = _positive_int(raw.get("shards", 1), f"{path}: {name}.shards")
        if shards > 1 and not BENCHMARKS[name].shardable:
            raise ValueError(f"{path}: benchmark {name!r} cannot be sharded.")
        if limit is not None and shards > limit:
            raise ValueError(f"{path}: {name} has more shards ({shards}) than documents (limit {limit}).")
        max_gen_toks = raw.get("max_gen_toks")
        if max_gen_toks is not None:
            max_gen_toks = _positive_int(max_gen_toks, f"{path}: {name}.max_gen_toks")
        benchmarks[name] = BenchmarkSettings(limit=limit, shards=shards, max_gen_toks=max_gen_toks)

    chat_benchmarks = [name for name in benchmarks if BENCHMARKS[name].chat_mode]
    chat: Optional[ChatSettings] = None
    if payload.get("chat") is not None:
        if not chat_benchmarks:
            raise ValueError(f"{path}: the suite has a chat section but no chat benchmark.")
        chat = _load_chat(path, payload["chat"], chat_benchmarks)
    elif chat_benchmarks:
        raise ValueError(
            f"{path}: the chat benchmarks {chat_benchmarks} need a chat section with the sampling presets "
            "of their prompt modes."
        )

    def resolve(key: str, default: str) -> Path:
        value = payload.get(key) or default
        candidate = Path(os.path.expandvars(str(value))).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        return candidate.resolve()

    tasks_dir = resolve("tasks_dir", "../tasks")
    entry = resolve("entry", "../run_lm_eval.py")
    if not tasks_dir.is_dir():
        raise ValueError(f"{path}: tasks_dir {tasks_dir} is not a folder.")
    if not entry.is_file():
        raise ValueError(f"{path}: entry {entry} does not exist.")
    startup = float(payload.get("process_startup_seconds", 110.0))
    if startup < 0:
        raise ValueError(f"{path}: process_startup_seconds must not be negative.")
    return Suite(
        path=path,
        tasks_dir=tasks_dir,
        entry=entry,
        benchmarks=benchmarks,
        process_startup_seconds=startup,
        vllm_model_args=str(payload.get("vllm_model_args") or "").strip().strip(","),
        digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        chat=chat,
    )


ITR_RENDER_PATH = REPO_ROOT / "eval" / "in_training" / "itr_render.py"


@functools.lru_cache(maxsize=1)
def load_itr_render() -> Any:
    """eval/in_training/itr_render.py, the rendering and token-contract code of the chat evaluation.

    The trainer uses its pure helpers (the template of a folder and its sha256). The same file runs
    in the evaluation processes and in the startup check, so the template is read the same way in all
    three places. It imports transformers only inside the functions that need it.
    """
    spec = importlib.util.spec_from_file_location("orbit_itr_render", ITR_RENDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
