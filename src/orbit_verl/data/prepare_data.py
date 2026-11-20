"""Convert a question file into verl's train.parquet and val.parquet, with one prompt mode for every row.

CLI::

    $PY -m orbit_verl.data.prepare_data --src FILE --prompt-mode MODE --tokenizer DIR \\
        --max-prompt-length N --out-dir DIR [--runs-root DIR] [--val-src FILE | --val-size 64] [--limit N] ...

The pass-rate tool (``scripts/verl/data/passrate``) builds its val-only datasets with it. The RL and
distillation recipes mix prompt modes and use ``orbit_verl.data.distill_data`` instead.

What the converter does, in order:

0. Refuses a prompt mode whose markup the tokenizer does not encode as single tokens
   (``prompts.missing_tokens``).
1. Reads (question, gold answer) pairs from a local jsonl or parquet file with ``sources.read_rows``.
2. Drops rows with an empty question or answer, and repeated questions (the first copy stays).
3. Renders each question with the prompt mode and measures its length with
   ``prompts.prompt_token_ids``, the same function the rollout uses. Rows longer than
   ``max_prompt_length`` tokens are dropped, because the rollout would otherwise cut their prompt.
4. Shuffles with ``seed``, holds out the first ``val_size`` rows as validation (unless a separate
   ``val_src`` is given), and keeps the first ``limit`` of the remaining rows for training.
5. Writes both parquet files with one fixed schema, then ``manifest.json`` last. The launcher
   treats the manifest as the marker that the directory is complete.

The parquet files hold prompt text, not token ids, but a dataset still belongs to one tokenizer:
the length filter of step 3 counted that tokenizer's tokens. The manifest records the tokenizer's
fingerprint (``tokenizer_sha256``) and its family (``tokenizer_family``, the model preset), and the
launcher refuses the dataset for a model dir with another tokenizer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from orbit_verl import common
from orbit_verl.data import sources
from orbit_verl.data.prompts import (
    DATA_SOURCES,
    PROMPT_MODES,
    PromptMode,
    get_mode,
    load_template,
    missing_tokens,
    prompt_token_ids,
    render_messages,
)

TRAIN_FILE = "train.parquet"
VAL_FILE = "val.parquet"


def parquet_schema():
    """The one schema of every file this tool writes, so files of different datasets can be mixed."""
    import pyarrow as pa

    return pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("agent_name", pa.string()),
            ("ability", pa.string()),
            ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            (
                "extra_info",
                pa.struct(
                    [
                        ("index", pa.int64()),
                        ("split", pa.string()),
                        ("prompt_mode", pa.string()),
                        ("question", pa.string()),
                        ("prompt_tokens", pa.int64()),
                    ]
                ),
            ),
        ]
    )


def _load_tokenizer(tokenizer_dir: str):
    # The export dirs carry remote model code. trust_remote_code=True lets transformers read their
    # config without asking on stdin, which would stall a non-interactive run. verl loads the same
    # dirs with trust_remote_code too.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)


def tokenizer_family(tokenizer_dir: str | os.PathLike) -> str | None:
    """The tokenizer family of a tokenizer dir: the model preset name, such as qwen3.

    A prepared model dir names its preset in its manifest. Any other dir with a config.json is
    matched to a preset by its vocabulary size. A bare tokenizer dir has neither and gives None;
    the dataset is then still tied to its tokenizer by ``tokenizer_sha256``.
    """
    from orbit_verl.model import preset_for_vocab

    root = Path(tokenizer_dir)
    try:
        manifest = common.read_manifest(root / common.MODEL_MANIFEST, common.MODEL_MANIFEST_KIND)
    except (FileNotFoundError, ValueError):
        manifest = {}
    if manifest.get("preset"):
        return str(manifest["preset"])
    try:
        return preset_for_vocab(int(json.loads((root / "config.json").read_text(encoding="utf-8"))["vocab_size"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _resolve_data_source(data_source: str | None, mode: PromptMode) -> str:
    """The parquet ``data_source``, which selects the reward of every row: the given one, else the mode's default."""
    if data_source is None:
        data_source = mode.default_data_source
    if data_source not in DATA_SOURCES:
        raise ValueError(f"Unknown data_source {data_source!r}; known sources: {sorted(DATA_SOURCES)}.")
    return data_source


def _filter_rows(
    rows: list[tuple[int, str, str]],
    mode: PromptMode,
    tokenizer: Any,
    max_prompt_length: int,
    *,
    drop_duplicates: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    """Apply steps 2 and 3 of the module docstring to one source. Returns kept rows and drop counts."""
    dropped: Counter = Counter({"empty": 0, "duplicate": 0, "overlong": 0})
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for number, question, answer in rows:
        if not question.strip() or not answer.strip():
            dropped["empty"] += 1
            continue
        if drop_duplicates:
            # Some math files list a problem twice, once with a worked solution and once with the
            # bare answer. Without this step a held-out problem could also sit in the train split.
            key = question.strip()
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)
        messages = render_messages(mode, question)
        n_tokens = len(prompt_token_ids(tokenizer, messages))
        if n_tokens > max_prompt_length:
            dropped["overlong"] += 1
            continue
        kept.append(
            {"number": number, "question": question, "answer": answer, "messages": messages, "n_tokens": n_tokens}
        )
    return kept, dropped


def _records(rows: list[dict[str, Any]], split: str, mode: PromptMode, data_source: str) -> list[dict[str, Any]]:
    return [
        {
            "data_source": data_source,
            "prompt": row["messages"],
            "agent_name": mode.agent_loop,
            "ability": "code" if data_source == "orbit/code" else "math",
            # The gold answer is stored exactly as the file has it. The MATH reward accepts a bare
            # answer or a solution with \boxed{}, and a code row holds its tests as JSON.
            "reward_model": {"style": "rule", "ground_truth": row["answer"]},
            "extra_info": {
                # The row number in the source file, so a sample can be traced back to its origin.
                "index": int(row["number"]),
                "split": split,
                "prompt_mode": mode.name,
                "question": row["question"],
                "prompt_tokens": int(row["n_tokens"]),
            },
        }
        for row in rows
    ]


def _percentile(sorted_values: list[int], q: float) -> int:
    """Nearest-rank percentile of an already sorted list, 0 for an empty list."""
    if not sorted_values:
        return 0
    rank = max(1, math.ceil(q * len(sorted_values)))
    return int(sorted_values[min(rank, len(sorted_values)) - 1])


def _check_out_dir(out_dir: Path, force: bool) -> None:
    """Refuse to overwrite anything but a directory this tool wrote itself."""
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise FileExistsError(f"{out_dir} exists and is not a directory.")
    if not any(out_dir.iterdir()):
        return
    manifest = out_dir / common.DATA_MANIFEST
    if not force:
        raise FileExistsError(f"{out_dir} is not empty. Pass --force to replace a dataset this tool wrote.")
    try:
        common.read_manifest(manifest, common.DATA_MANIFEST_KIND)
    except (FileNotFoundError, ValueError) as exc:
        raise FileExistsError(
            f"{out_dir} has no valid {common.DATA_MANIFEST} of kind {common.DATA_MANIFEST_KIND} ({exc}). "
            "--force replaces only directories this tool wrote; remove it by hand if it is safe to."
        ) from None


def convert(
    src: str,
    out_dir: str,
    *,
    prompt_mode: str,
    tokenizer_dir: str,
    max_prompt_length: int,
    data_source: str | None = None,
    question_field: str = "instruction",
    answer_field: str = "response",
    val_size: int = 64,
    val_src: str | None = None,
    limit: int | None = None,
    seed: int = 0,
    force: bool = False,
    drop_duplicates: bool = True,
    runs_root: str | None = None,
) -> dict[str, Any]:
    """Build ``out_dir`` with train.parquet, val.parquet and manifest.json. Returns the manifest.

    ``src`` and ``val_src`` are local jsonl or parquet files (``sources``), whose questions and gold
    answers are in the columns ``question_field`` and ``answer_field``.

    ``runs_root`` places the caches under ``<runs_root>/cache`` (``common.apply_cache_env``). It
    may be None when ``ORBIT_VERL_RUNS`` is set or the caller has applied the cache env already,
    as the launcher does. Without any of them the tokenizer's remote code would be cached in the
    home directory, so the call raises ``common.RunsRootError``.
    """
    mode = get_mode(prompt_mode)
    data_source = _resolve_data_source(data_source, mode)
    if max_prompt_length < 1:
        raise ValueError("max_prompt_length must be positive.")
    if val_src is None and val_size < 1:
        # verl builds a validation dataloader even with validation turned off, and an empty
        # val.parquet makes that dataloader fail at start-up.
        raise ValueError("val_size must be at least 1 when no --val-src is given.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive.")
    out = Path(out_dir).resolve()
    _check_out_dir(out, force)

    common.apply_cache_env(None if runs_root is None else str(runs_root))
    import pyarrow as pa
    import pyarrow.parquet as pq

    tokenizer = _load_tokenizer(tokenizer_dir)
    load_template(mode)  # Fails early if the template file is wrong.
    missing = missing_tokens(mode, tokenizer)
    if missing:
        raise ValueError(
            f"Prompt mode {mode.name!r} needs {missing} as single tokens, but the tokenizer in "
            f"{tokenizer_dir} splits them into pieces. The prompt modes fit the Qwen3 tokenizer of the "
            "release SFT exports."
        )

    options = dict(drop_duplicates=drop_duplicates)
    raw_rows, source_record = sources.read_rows(src, question_field, answer_field)
    train_rows, dropped = _filter_rows(raw_rows, mode, tokenizer, max_prompt_length, **options)
    random.Random(seed).shuffle(train_rows)
    val_record = None
    if val_src is None:
        val_rows, train_rows = train_rows[:val_size], train_rows[val_size:]
    else:
        raw_val, val_record = sources.read_rows(val_src, question_field, answer_field)
        val_rows, val_dropped = _filter_rows(raw_val, mode, tokenizer, max_prompt_length, **options)
        dropped.update(val_dropped)
    if limit is not None:
        train_rows = train_rows[:limit]
    if not train_rows or not val_rows:
        raise ValueError(
            f"Nothing left to write: {len(train_rows)} train and {len(val_rows)} val rows after filtering "
            f"(dropped {dict(dropped)})."
        )

    lengths = sorted(row["n_tokens"] for row in train_rows + val_rows)
    payload = {
        "src": sources.canonical_src(src),
        "val_src": sources.canonical_src(val_src) if val_src else None,
        "source": source_record,
        "val_source": val_record,
        "prompt_mode": mode.name,
        "data_source": data_source,
        "agent_name": mode.agent_loop,
        "max_prompt_length": int(max_prompt_length),
        "tokenizer_dir": str(Path(tokenizer_dir).resolve()),
        "tokenizer_sha256": common.tokenizer_sha256(tokenizer_dir),
        "tokenizer_family": tokenizer_family(tokenizer_dir),
        "rows": {"train": len(train_rows), "val": len(val_rows)},
        "dropped": dict(dropped),
        "prompt_tokens": {
            "max": int(lengths[-1]),
            "p50": _percentile(lengths, 0.50),
            "p99": _percentile(lengths, 0.99),
        },
        "seed": int(seed),
        "limit": limit,
        "val_size": int(val_size) if val_src is None else None,
        "drop_duplicates": bool(drop_duplicates),
        "question_field": question_field,
        "answer_field": answer_field,
    }

    # Everything is written to a staging directory next to the target and moved into place at the
    # end. An interrupted run then leaves no half-written dataset under the target name.
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.staging-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        schema = parquet_schema()
        for name, split, rows in ((TRAIN_FILE, "train", train_rows), (VAL_FILE, "val", val_rows)):
            table = pa.Table.from_pylist(_records(rows, split, mode, data_source), schema=schema)
            pq.write_table(table, staging / name)
        manifest = common.write_manifest(staging / common.DATA_MANIFEST, common.DATA_MANIFEST_KIND, payload)
        if out.exists():
            # _check_out_dir has already confirmed that this is empty or one of our datasets.
            shutil.rmtree(out)
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orbit_verl.data.prepare_data",
        description="Convert a jsonl or parquet file into verl train/val parquet files with one prompt mode.",
    )
    parser.add_argument("--src", required=True, help="A local .jsonl or .parquet file.")
    parser.add_argument("--prompt-mode", required=True, choices=sorted(PROMPT_MODES))
    parser.add_argument("--tokenizer", required=True, help="Tokenizer dir, normally the prepared model dir.")
    parser.add_argument(
        "--max-prompt-length",
        required=True,
        type=int,
        help="Drop prompts longer than this many tokens. Must equal data.max_prompt_length of the run.",
    )
    parser.add_argument("--out-dir", required=True, help="Output dir for train.parquet, val.parquet, manifest.json.")
    parser.add_argument(
        "--runs-root",
        help=f"Root of the runtime outputs; every cache goes under <runs-root>/cache. Default: ${common.RUNS_ROOT_ENV}.",
    )
    parser.add_argument(
        "--data-source",
        choices=sorted(DATA_SOURCES),
        help="Default: the prompt mode's default.",
    )
    parser.add_argument("--question-field", default="instruction", help="Question field of the files.")
    parser.add_argument("--answer-field", default="response", help="Gold-answer field of the files.")
    parser.add_argument("--val-size", type=int, default=64, help="Rows held out from --src (ignored with --val-src).")
    parser.add_argument("--val-src", help="Separate validation file, written like --src.")
    parser.add_argument("--limit", type=int, help="Keep the first N train rows after filtering and shuffling.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-duplicates", action="store_true", help="Keep repeated questions.")
    parser.add_argument("--force", action="store_true", help="Replace an output dir that this tool wrote.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # The tool has no default runs root: every person writes into their own one.
    try:
        runs_root = common.resolve_runs_root(args.runs_root)
    except common.RunsRootError:
        print(f"prepare_data: error: {common.runs_root_help(('--runs-root PATH',))}", file=sys.stderr)
        return 2
    try:
        manifest = convert(
            args.src,
            args.out_dir,
            prompt_mode=args.prompt_mode,
            tokenizer_dir=args.tokenizer,
            max_prompt_length=args.max_prompt_length,
            data_source=args.data_source,
            question_field=args.question_field,
            answer_field=args.answer_field,
            val_size=args.val_size,
            val_src=args.val_src,
            limit=args.limit,
            seed=args.seed,
            force=args.force,
            drop_duplicates=not args.keep_duplicates,
            runs_root=runs_root,
        )
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        print(f"prepare_data: error: {exc}", file=sys.stderr)
        return 2
    rows, dropped, tokens = manifest["rows"], manifest["dropped"], manifest["prompt_tokens"]
    print(f"prepare_data: wrote {Path(args.out_dir).resolve()}")
    print(f"  source {manifest['src']}, tokenizer family {manifest['tokenizer_family'] or 'unknown'}")
    print(f"  prompt mode {manifest['prompt_mode']}, data source {manifest['data_source']}, agent {manifest['agent_name']}")
    print(f"  rows: train {rows['train']}, val {rows['val']}")
    print("  dropped: " + ", ".join(f"{key} {value}" for key, value in sorted(dropped.items())))
    print(
        f"  prompt tokens: max {tokens['max']}, p50 {tokens['p50']}, p99 {tokens['p99']} "
        f"(limit {manifest['max_prompt_length']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
