"""Work units and the plan: how a suite is cut into lm-eval processes and spread over the ranks.

Every rank computes the same plan on its own, so the plan depends only on the suite and the number
of ranks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from .benchmarks import BENCHMARKS, Benchmark, RULER_TASKS, ruler_seconds_per_sample
from .suite import BenchmarkSettings, ChatSettings, Suite


@dataclass(frozen=True)
class Unit:
    """The smallest piece of evaluation work: some lm-eval tasks, run by one rank.

    ``signature`` holds everything that lm-eval applies to a whole process (shot count, generation
    arguments, --limit, --metadata). Units share a process only when their signatures are equal,
    and a benchmark with ``own_process`` shares its processes with no other benchmark (process_key).
    ``samples`` is a (task, first, end) document range when the unit is one shard of a task.
    ``length`` is set for RULER units, whose metric is per sequence length.
    """

    name: str
    benchmark: str
    tasks: tuple[str, ...]
    signature: tuple[str, ...]
    docs: int
    cost: float
    samples: Optional[tuple[str, int, int]] = None
    length: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "benchmark": self.benchmark,
            "tasks": list(self.tasks),
            "signature": list(self.signature),
            "docs": self.docs,
            "cost": round(self.cost, 3),
            "samples": list(self.samples) if self.samples else None,
            "length": self.length,
        }


# Decode time of one OrbitLM response on an H100 under a full KV cache: 12-20 ms per token was measured
# with vLLM 0.22 (4 loops), and 25 ms leaves room for the larger KV of long think traces. The planner
# uses it for the tail of a chat unit, and the timeouts use a larger 30 ms.
CHAT_TAIL_SECONDS_PER_TOKEN = 0.025
CHAT_TIMEOUT_SECONDS_PER_TOKEN = 0.030


def chat_signature(benchmark: Benchmark, settings: BenchmarkSettings, chat: ChatSettings) -> tuple[str, ...]:
    """What lm-eval applies to a whole process of one chat benchmark.

    The model arguments select the prompt mode and the leading-token policy of the model subclass
    (vllm_itr_chat); build_lm_eval_command merges them into the trainer's own model arguments. The
    sampling preset goes in as JSON with sorted keys, so equal presets give equal signatures.
    --log_samples keeps the response texts, which the records of the gates are built from.
    """
    model_args = [f"itr_mode={benchmark.chat_mode}", f"itr_lead_eos={chat.lead_eos}"]
    preset = dict(chat.presets[benchmark.chat_mode])
    if settings.max_gen_toks is not None:
        preset["max_gen_toks"] = settings.max_gen_toks
    return (
        "--apply_chat_template",
        "--log_samples",
        "--model_args",
        ",".join(model_args),
        "--gen_kwargs",
        json.dumps(preset, sort_keys=True),
    )


def signature_budget(signature: tuple[str, ...]) -> int:
    """The max_gen_toks of a chat signature, or 0 for any other signature.

    Only chat signatures carry their own --model_args, and only they pass --gen_kwargs as JSON; a base
    benchmark such as MMLU-Pro passes "max_gen_toks=512" and keeps the base timeouts.
    """
    if "--model_args" not in signature or "--gen_kwargs" not in signature:
        return 0
    payload = json.loads(signature[signature.index("--gen_kwargs") + 1])
    return int(payload.get("max_gen_toks") or 0)


def chat_unit_tail(signature: tuple[str, ...]) -> float:
    """The wall time of one response at the budget, which no sharding shortens.

    A rank's think process ends with its longest response. Adding this tail to the cost of every chat
    unit keeps the planner from putting a second process behind a think process, which would start
    only after that tail.
    """
    return signature_budget(signature) * CHAT_TAIL_SECONDS_PER_TOKEN


def expand_units(suite: Suite) -> list[Unit]:
    """Turns the suite into work units small enough to balance across ranks."""
    units: list[Unit] = []
    for name, settings in suite.benchmarks.items():
        benchmark = BENCHMARKS[name]
        limit_args = ("--limit", str(settings.limit)) if settings.limit is not None else ()
        if benchmark.chat_mode is not None:
            units += _chat_units(suite, name, benchmark, settings, limit_args)
            continue
        if name == "ruler":
            for length in settings.lengths:
                # One process builds and evaluates one length, so the metadata carries one length.
                metadata = json.dumps(
                    {"max_seq_lengths": [length], "num_samples": settings.num_samples},
                    sort_keys=True,
                )
                docs = settings.num_samples
                if settings.limit is not None:
                    docs = min(docs, settings.limit)
                for task in RULER_TASKS:
                    units.append(
                        Unit(
                            name=f"ruler/{task}@{length}",
                            benchmark=name,
                            tasks=(task,),
                            signature=(*limit_args, "--metadata", metadata),
                            docs=docs,
                            cost=docs * ruler_seconds_per_sample(length),
                            length=length,
                        )
                    )
            continue
        for task, total_docs in benchmark.subtasks:
            docs = min(total_docs, settings.limit) if settings.limit is not None else total_docs
            if settings.shards <= 1:
                units.append(
                    Unit(
                        name=f"{name}/{task}",
                        benchmark=name,
                        tasks=(task,),
                        signature=(*benchmark.lm_eval_args, *limit_args),
                        docs=docs,
                        cost=docs * benchmark.sec_per_doc,
                    )
                )
                continue
            # Shards are passed to lm-eval as document lists (--samples), which lm-eval refuses to
            # combine with --limit. A limit therefore becomes part of the document ranges: the
            # shards split the first `limit` documents.
            for shard in range(settings.shards):
                first = shard * docs // settings.shards
                end = (shard + 1) * docs // settings.shards
                units.append(
                    Unit(
                        name=f"{name}/{task}#{shard + 1}of{settings.shards}",
                        benchmark=name,
                        tasks=(task,),
                        signature=benchmark.lm_eval_args,
                        docs=end - first,
                        cost=(end - first) * benchmark.sec_per_doc,
                        samples=(task, first, end),
                    )
                )
    return units


def _chat_units(suite: Suite, name: str, benchmark: Benchmark, settings: BenchmarkSettings,
                limit_args: tuple[str, ...]) -> list[Unit]:
    """The units of one chat benchmark, sharded like the base benchmarks, with a tail-aware cost."""
    assert suite.chat is not None
    signature = chat_signature(benchmark, settings, suite.chat)
    tail = chat_unit_tail(signature)
    units = []
    for task, total_docs in benchmark.subtasks:
        docs = min(total_docs, settings.limit) if settings.limit is not None else total_docs
        per_doc = benchmark.sec_per_doc
        if settings.shards <= 1:
            units.append(Unit(name=f"{name}/{task}", benchmark=name, tasks=(task,),
                              signature=(*signature, *limit_args), docs=docs, cost=docs * per_doc + tail))
            continue
        for shard in range(settings.shards):
            first = shard * docs // settings.shards
            end = (shard + 1) * docs // settings.shards
            units.append(Unit(name=f"{name}/{task}#{shard + 1}of{settings.shards}", benchmark=name, tasks=(task,),
                              signature=signature, docs=end - first, cost=(end - first) * per_doc + tail,
                              samples=(task, first, end)))
    return units


@dataclass(frozen=True)
class EvalProcess:
    """One lm-eval subprocess on one rank: units with the same process_key, run together."""

    rank: int
    index: int
    units: tuple[Unit, ...]

    @property
    def name(self) -> str:
        return f"r{self.rank:05d}_p{self.index:02d}"

    @property
    def signature(self) -> tuple[str, ...]:
        return self.units[0].signature

    @property
    def tasks(self) -> list[str]:
        ordered: list[str] = []
        for unit in self.units:
            for task in unit.tasks:
                if task not in ordered:
                    ordered.append(task)
        return ordered

    @property
    def samples(self) -> Optional[dict[str, list[int]]]:
        """Document indices per sharded task, in the form of lm-eval's --samples file."""
        merged: dict[str, set[int]] = {}
        for unit in self.units:
            if unit.samples is not None:
                task, first, end = unit.samples
                merged.setdefault(task, set()).update(range(first, end))
        if not merged:
            return None
        return {task: sorted(indices) for task, indices in sorted(merged.items())}

    @property
    def cost(self) -> float:
        return sum(unit.cost for unit in self.units)

    @property
    def is_chat(self) -> bool:
        """Whether the process runs chat benchmarks, which also write the records of the gates."""
        return BENCHMARKS[self.units[0].benchmark].chat_mode is not None

    @property
    def budget(self) -> int:
        """The generation budget of a chat process, 0 for a base process."""
        return signature_budget(self.signature)


def process_key(unit: Unit) -> tuple[tuple[str, ...], str]:
    """What two units must have in common to run in one process.

    The first part is the signature, because lm-eval applies it to every task of the process. The
    second part is the benchmark's name when the benchmark has own_process, and empty otherwise.
    So such a benchmark never shares a process with another benchmark, and the other benchmarks
    group exactly as they would without it.
    """
    owner = unit.benchmark if BENCHMARKS[unit.benchmark].own_process else ""
    return unit.signature, owner


def assign_units(units: list[Unit], world_size: int, startup_seconds: float) -> list[list[Unit]]:
    """Assigns units to ranks with longest-processing-time-first, aware of process startup.

    Units are taken from the most to the least expensive and each goes to the rank where it
    would finish earliest. A rank that has no process with the unit's process_key yet would have
    to start one, so the vLLM startup time is added to its cost there; that keeps small units
    together on ranks that already run their kind of process. Ties go to the lowest rank, so
    every rank computes the same plan on its own.
    """
    if world_size < 1:
        raise ValueError("world_size must be at least 1.")
    loads = [0.0] * world_size
    keys: list[set[tuple[tuple[str, ...], str]]] = [set() for _ in range(world_size)]
    assigned: list[list[Unit]] = [[] for _ in range(world_size)]
    for unit in sorted(units, key=lambda item: (-item.cost, item.name)):
        def finish_time(rank: int) -> tuple[float, int]:
            startup = 0.0 if process_key(unit) in keys[rank] else startup_seconds
            return (loads[rank] + unit.cost + startup, rank)

        best = min(range(world_size), key=finish_time)
        loads[best] = finish_time(best)[0]
        keys[best].add(process_key(unit))
        assigned[best].append(unit)
    return assigned


def group_processes(rank: int, units: list[Unit]) -> list[EvalProcess]:
    """Groups one rank's units into processes, one per distinct process_key.

    The processes are ordered by signature, and a benchmark's own process comes right after the
    shared process with the same signature. A rank without such a benchmark therefore numbers its
    processes as if own_process did not exist, which keeps the golden snapshot of the base suites
    valid (tests/train/base_golden.py).
    """
    groups: dict[tuple[tuple[str, ...], str], list[Unit]] = {}
    for unit in units:
        groups.setdefault(process_key(unit), []).append(unit)
    ordered = sorted(groups.items(), key=lambda item: (json.dumps(list(item[0][0])), item[0][1]))
    return [
        EvalProcess(rank=rank, index=index, units=tuple(sorted(group, key=lambda u: u.name)))
        for index, (_, group) in enumerate(ordered)
    ]


@dataclass(frozen=True)
class Plan:
    """The full assignment of work: processes per rank, and the estimated time per rank."""

    world_size: int
    units: tuple[Unit, ...]
    processes: tuple[tuple[EvalProcess, ...], ...]
    estimated_seconds: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "units": [unit.to_dict() for unit in self.units],
            "ranks": {
                str(rank): [
                    {"name": proc.name, "units": [unit.name for unit in proc.units], "cost": round(proc.cost, 3)}
                    for proc in procs
                ]
                for rank, procs in enumerate(self.processes)
                if procs
            },
            "estimated_seconds": [round(x, 1) for x in self.estimated_seconds],
        }

    @property
    def makespan_seconds(self) -> float:
        return max(self.estimated_seconds) if self.estimated_seconds else 0.0


def build_plan(suite: Suite, world_size: int) -> Plan:
    units = expand_units(suite)
    assigned = assign_units(units, world_size, suite.process_startup_seconds)
    processes = tuple(tuple(group_processes(rank, rank_units)) for rank, rank_units in enumerate(assigned))
    estimated = tuple(
        sum(proc.cost + suite.process_startup_seconds for proc in procs) for procs in processes
    )
    return Plan(world_size=world_size, units=tuple(units), processes=processes, estimated_seconds=estimated)
