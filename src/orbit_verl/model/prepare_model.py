"""Build a model dir that verl (FSDP actor and ref) and vLLM can load, from a OrbitLM HF export.

The HF exports of the trainer hold model code written for transformers 4.57 and may be read-only.
A prepared dir reuses the export's weights and tokenizer through symlinks, and adds what verl needs:

- the transformers-5 port of the model code (``modelling_ouro.py``, ``configuration_ouro.py``);
- ``config.json`` from the export with ``early_exit_threshold: null`` and an ``auto_map`` that points
  at the ported code, so the plain forward is the fixed-depth path that vLLM also implements;
- ``embedding_padding_idx`` in that ``config.json``: the token-embedding row that training kept at
  zero and frozen, or null when training froze none (see ``detect_embedding_padding``);
- ``generation_config.json`` with the stop tokens of the tokenizer family (the preset);
- ``orbit_prepare.json``, the manifest the launcher checks, written last.

The SFT chat export that the recipes start from (SFT v5 at step 7000) builds its embedding
without ``padding_idx`` and has no zero row. The detection therefore records it as ``release_qwen3``
with no frozen row, and the launcher prepares its dir without any manual step. Other exports may
bundle code that passes ``padding_idx=151643`` although that row is not zero in the weights. The
detection refuses them (``detect_embedding_padding``). For those, and only for those,
``--embedding-padding-idx none`` prepares the dir by hand, with the lineage recorded as unknown. The
release configs accept both lineages, and both freeze no row.

Usage::

    $PY -m orbit_verl.model.prepare_model --export DIR --name sft_v5_step7000 --check
    $PY -m orbit_verl.model.prepare_model --out-dir DIR --check      # check an existing dir only

``--name`` puts the dir under ``<runs root>/models``. The runs root comes from ``--runs-root`` or
from ``$ORBIT_VERL_RUNS``; it also holds the caches, so the tool needs it even with ``--out-dir``.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orbit_verl import common
from orbit_verl.model import MODEL_CODE_DIR, MODEL_CODE_FILES, PRESETS, preset_for_vocab

logger = logging.getLogger(__name__)

# Lineage labels written into the manifest. A lineage is the training line an export comes from,
# and it decides whether one token-embedding row has to stay frozen during RL.
# - release_qwen3: the release line on the 151936-token Qwen3 vocabulary, as the detection finds it.
#   It was trained without padding_idx, because its pad token is also the eos and document
#   separator, so no row is frozen.
# - unknown: any other combination, including a value chosen by hand with --embedding-padding-idx.
#   An SFT export whose bundled code names a padding row that is not zero records this lineage
#   once it is prepared by hand (see the module docstring).
LINEAGE_RELEASE_QWEN3 = "release_qwen3"
LINEAGE_UNKNOWN = "unknown"
LINEAGES = (LINEAGE_RELEASE_QWEN3, LINEAGE_UNKNOWN)

# The safetensors key of the token embedding in every OrbitLM export.
EMBED_WEIGHT = "model.embed_tokens.weight"

# Appended to every detection failure, so the message says how to go on.
PADDING_OVERRIDE_HINT = (
    "If you know which row training froze, pass --embedding-padding-idx N, or "
    "--embedding-padding-idx none when it froze no row."
)


class PaddingDetectionError(ValueError):
    """The export's training code and its weights disagree about the frozen embedding row."""


@dataclass(frozen=True)
class EmbeddingPadding:
    """Which token-embedding row a prepared dir freezes, and why.

    ``source`` is "detected" when the value comes from the export's code and weights, and
    "override" when the caller chose it. ``evidence`` is what the detection saw; it goes into the
    manifest as ``embedding_padding_evidence``.
    """

    embedding_padding_idx: int | None
    lineage: str
    source: str
    evidence: dict[str, Any] = field(default_factory=dict)

# Tokenizer files that are linked into the prepared dir when the tokenizer dir has them.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "added_tokens.json",
)

# auto_map that makes AutoConfig and AutoModelForCausalLM load the ported code from the model dir.
AUTO_MAP = {
    "AutoConfig": "configuration_ouro.OuroConfig",
    "AutoModelForCausalLM": "modelling_ouro.OuroForCausalLM",
}

def _weight_files(export_dir: Path) -> list[str]:
    """Names of the safetensors files of an export, including the shard index when there is one."""
    index = export_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        if not shards:
            raise ValueError(f"{index} lists no weight files.")
        names = [index.name, *shards]
    elif (export_dir / "model.safetensors").exists():
        names = ["model.safetensors"]
    else:
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {export_dir}. "
            "Only safetensors exports are supported."
        )
    missing = [name for name in names if not (export_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Weight files listed for {export_dir} are missing: {missing}")
    return names


def _check_token_ids(ids: dict[str, Any], vocab_size: int) -> None:
    """Fail when a token id is not an int inside the vocabulary."""
    for key, value in ids.items():
        values = value if isinstance(value, list) else [value]
        for token_id in values:
            if token_id is None and key == "bos_token_id":
                continue
            if not isinstance(token_id, int) or isinstance(token_id, bool) or not 0 <= token_id < vocab_size:
                raise ValueError(f"{key}={value!r} is not a token id below vocab_size {vocab_size}.")


def _resolve_token_ids(
    preset: str,
    vocab_size: int,
    eos_token_ids: list[int] | None,
    pad_token_id: int | None,
    bos_token_id: int | str | None,
) -> tuple[str, dict[str, Any]]:
    """Pick the preset and apply the explicit overrides. Returns (preset name, token ids)."""
    name = preset_for_vocab(vocab_size) if preset == "auto" else preset
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; use auto or one of {sorted(PRESETS)}.")
    values = PRESETS[name]
    # A preset's ids name tokens of one tokenizer family. Applying them to a model with another
    # vocabulary would stop generation on unrelated tokens, so that is only allowed when the
    # caller replaces the eos and pad ids anyway.
    if values["vocab_size"] != vocab_size and (eos_token_ids is None or pad_token_id is None):
        raise ValueError(
            f"Preset {name!r} is for vocab_size {values['vocab_size']}, but the export has "
            f"vocab_size {vocab_size}. Use --preset auto, or pass --eos-ids and --pad-id explicitly."
        )
    ids = {
        "eos_token_ids": list(values["eos_token_ids"]) if eos_token_ids is None else [int(i) for i in eos_token_ids],
        "pad_token_id": values["pad_token_id"] if pad_token_id is None else int(pad_token_id),
        "bos_token_id": values["bos_token_id"],
    }
    # The string "none" clears the bos id. None keeps the preset's value.
    if isinstance(bos_token_id, str):
        ids["bos_token_id"] = None if bos_token_id.strip().lower() == "none" else int(bos_token_id)
    elif bos_token_id is not None:
        ids["bos_token_id"] = int(bos_token_id)
    if not ids["eos_token_ids"]:
        raise ValueError("eos_token_ids must not be empty.")
    _check_token_ids(ids, vocab_size)
    return name, ids


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    """Copy ``src`` to ``dst``, or make ``dst`` a symlink to it.

    Copies follow symlinks in the export, so a materialized dir does not depend on the export.
    """
    if copy:
        shutil.copyfile(src, dst)
    else:
        # Absolute targets keep the links valid wherever the prepared dir is read from.
        os.symlink(os.path.abspath(src), dst)


def _has_our_manifest(path: Path) -> bool:
    """True when ``path`` holds a manifest written by this tool, of any manifest version."""
    try:
        data = json.loads((path / common.MODEL_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("kind") == common.MODEL_MANIFEST_KIND


def _call_name(call: ast.Call) -> str:
    """The called name without its module, so nn.Embedding and torch.nn.Embedding both give "Embedding"."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _resolve_padding_expr(expr: ast.expr, assigned: dict[str, list[ast.expr]], depth: int = 0) -> tuple[str, Any]:
    """Follow a padding_idx expression back to a literal or a config attribute.

    ``assigned`` maps each assignment target inside OuroModel.__init__ to the values assigned to it.
    That lets ``self.padding_idx``, assigned once from ``config.pad_token_id``, resolve to that
    config attribute. Returns a (kind, value) pair as described in ``_code_padding``.
    """
    if isinstance(expr, ast.Constant):
        if expr.value is None:
            return "none", None
        if isinstance(expr.value, int) and not isinstance(expr.value, bool):
            return "literal", expr.value
        return "unknown", f"padding_idx is the constant {ast.unparse(expr)}"
    if isinstance(expr, ast.Attribute) and ast.unparse(expr.value) in ("config", "self.config"):
        return "config", expr.attr
    if isinstance(expr, (ast.Attribute, ast.Name)) and depth < 3:
        values = assigned.get(ast.unparse(expr), [])
        if len(values) == 1:
            return _resolve_padding_expr(values[0], assigned, depth + 1)
        return "unknown", f"{ast.unparse(expr)} is assigned {len(values)} times in OuroModel.__init__, expected once"
    return "unknown", f"padding_idx is {ast.unparse(expr)}, which is neither a literal nor a config attribute"


def _code_padding(path: Path) -> tuple[str, Any, str]:
    """Read which padding_idx the export's own model code passes to the token embedding.

    The file is parsed and never imported. Export code targets another transformers version, and
    importing it could run anything. Only the call that builds ``self.embed_tokens`` counts: the
    release line still assigns ``self.padding_idx`` but no longer passes it to the embedding, so
    searching the file for "padding_idx" would give the wrong answer.

    Returns ``(kind, value, expression)``, where expression is the source text of the argument:
    - ``("none", None, ...)``: no padding_idx argument, or a literal None;
    - ``("literal", N, ...)``: padding_idx is the int N written in the code;
    - ``("config", attr, ...)``: padding_idx is ``config.<attr>``, directly or through one
      assignment such as ``self.padding_idx = config.pad_token_id``;
    - ``("unknown", reason, ...)``: the code matches none of these patterns.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return "unknown", f"cannot read {path.name}: {exc}", ""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return "unknown", f"{path.name} does not parse: {exc}", ""
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "OuroModel"]
    if len(classes) != 1:
        return "unknown", f"{path.name} defines OuroModel {len(classes)} times, expected once", ""
    inits = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
    if len(inits) != 1:
        return "unknown", "OuroModel has no __init__", ""

    assigned: dict[str, list[ast.expr]] = {}
    for node in ast.walk(inits[0]):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            assigned.setdefault(ast.unparse(target), []).append(value)

    builds = assigned.get("self.embed_tokens", [])
    if len(builds) != 1:
        return "unknown", f"OuroModel.__init__ assigns self.embed_tokens {len(builds)} times, expected once", ""
    call = builds[0]
    if not isinstance(call, ast.Call) or _call_name(call) != "Embedding":
        return "unknown", f"self.embed_tokens is built by {ast.unparse(call)}, not by an Embedding call", ""
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
        return "unknown", f"{ast.unparse(call)} passes *args or **kwargs", ast.unparse(call)

    # nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, ...): the third positional
    # argument or the padding_idx keyword.
    keyword = [kw.value for kw in call.keywords if kw.arg == "padding_idx"]
    if keyword:
        expr = keyword[0]
    elif len(call.args) >= 3:
        expr = call.args[2]
    else:
        return "none", None, ""
    kind, value = _resolve_padding_expr(expr, assigned)
    return kind, value, ast.unparse(expr)


def _index_or_text(value: Any) -> Any:
    """Normalise a recorded token id for comparison: None, an int, or the value's text when it is neither.

    export_info.json may hold ids as ints or as strings such as "2".
    """
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("", "none", "null"):
            return None
        if text.lstrip("-").isdigit():
            return int(text)
    return repr(value)


def _export_info_value(export: Path, attr: str) -> tuple[bool, Any]:
    """The training-time model config value of ``attr``, as recorded in the export's export_info.json.

    Returns (False, None) when the file or the key is absent. The exporter copies the model config
    of the training run into checkpoint_metadata.model_config_overrides, so this value is a check on
    config.json that does not depend on how the export's config.json was written.
    """
    try:
        info = json.loads((export / "export_info.json").read_text(encoding="utf-8"))
        recorded = info["checkpoint_metadata"]["model_config_overrides"]
    except (OSError, ValueError, KeyError, TypeError):
        return False, None
    if not isinstance(recorded, dict) or attr not in recorded:
        return False, None
    return True, _index_or_text(recorded[attr])


def _embedding_file(export: Path, weight_names: list[str]) -> str:
    """The safetensors file of the export that holds the token embedding."""
    if "model.safetensors.index.json" in weight_names:
        index = json.loads((export / "model.safetensors.index.json").read_text(encoding="utf-8"))
        name = index.get("weight_map", {}).get(EMBED_WEIGHT)
        if name is None:
            raise KeyError(f"{EMBED_WEIGHT} is not in the weight_map of model.safetensors.index.json")
        return name
    return "model.safetensors"


def _scan_embedding(
    export: Path, weight_names: list[str], row: int | None, chunk_rows: int = 8192
) -> tuple[list[int], list[int], float | None]:
    """Find the all-zero rows of the token embedding in the export's safetensors.

    The tensor is read in slices of ``chunk_rows`` rows, so the 151936-row release embedding never
    sits in memory as a whole; that scan takes about two seconds on a CPU. Returns the zero rows,
    the tensor shape, and the L2 norm of ``row`` (None when ``row`` is None or out of range).
    """
    # torch is imported here so that importing this module stays cheap for the launcher.
    import torch
    from safetensors import safe_open

    path = export / _embedding_file(export, weight_names)
    with safe_open(str(path), framework="pt") as handle:
        if EMBED_WEIGHT not in handle.keys():
            raise KeyError(f"{EMBED_WEIGHT} is not in {path.name}")
        tensor = handle.get_slice(EMBED_WEIGHT)
        shape = [int(n) for n in tensor.get_shape()]
        if len(shape) != 2:
            raise ValueError(f"{EMBED_WEIGHT} has shape {shape}, expected two dimensions")
        zero_rows: list[int] = []
        for start in range(0, shape[0], chunk_rows):
            block = tensor[start:start + chunk_rows]
            zero_rows.extend((torch.nonzero((block == 0).all(dim=1)).flatten() + start).tolist())
        norm = None
        if row is not None and 0 <= row < shape[0]:
            norm = float(tensor[row:row + 1].float().norm())
    return zero_rows, shape, norm


def _padding_evidence(
    export: Path, config: dict[str, Any], weight_names: list[str]
) -> tuple[dict[str, Any], int | None, str | None, list[int] | None]:
    """Collect what the export says about its frozen embedding row and check that it agrees.

    Returns ``(evidence, row, problem, zero_rows)``. ``row`` is the row the export's code freezes,
    or None. ``problem`` is None when the code, config.json, export_info.json and the weights all
    agree, and otherwise a sentence that says what disagrees. ``zero_rows`` is the full list of
    all-zero embedding rows, or None when the weights could not be read.
    """
    vocab_size = int(config["vocab_size"])
    code_path = export / "modelling_ouro.py"
    kind, value, expr = _code_padding(code_path)
    evidence: dict[str, Any] = {
        "code_file_sha256": common.sha256_file(code_path) if code_path.is_file() else None,
        "padding_expr": expr or None,
    }
    row: int | None = None
    problem: str | None = None
    no_row_reason = "the export's modelling_ouro.py builds the token embedding without padding_idx"

    if kind == "unknown":
        evidence["padding_from"] = "unknown"
        problem = f"Cannot tell which padding_idx the export's modelling_ouro.py gives the token embedding: {value}."
    elif kind == "none":
        evidence["padding_from"] = "none"
    elif kind == "literal":
        evidence["padding_from"] = "literal"
        row = value
    else:
        attr = value
        evidence["padding_from"] = f"config.{attr}"
        raw = config.get(attr)
        evidence["config_value"] = raw
        if raw is None:
            # The code passes padding_idx=None then, which freezes no row.
            no_row_reason = f"the export's code takes padding_idx from config.{attr}, which config.json leaves null"
        elif isinstance(raw, bool) or not isinstance(raw, int):
            problem = f"The export's code takes padding_idx from config.{attr}, but config.json has {attr}={raw!r}."
        else:
            row = raw
        recorded, info_value = _export_info_value(export, attr)
        if recorded:
            evidence["export_info_value"] = info_value
            if problem is None and info_value != _index_or_text(raw):
                problem = (
                    f"config.json has {attr}={raw!r}, but export_info.json records {info_value!r} as the "
                    "value the model was trained with. The export's config.json was changed after training."
                )
    evidence["candidate_row"] = row
    if problem is None and row is not None and not 0 <= row < vocab_size:
        problem = f"The export's code freezes row {row}, which is outside vocab_size {vocab_size}."

    try:
        zero_rows, shape, norm = _scan_embedding(export, weight_names, row)
    except (OSError, KeyError, ValueError) as exc:
        evidence["scan_error"] = str(exc)
        return evidence, row, problem or f"Cannot read the token embedding from the export's weights: {exc}.", None
    evidence["embedding_shape"] = shape
    evidence["zero_rows"] = zero_rows[:16]
    evidence["zero_rows_count"] = len(zero_rows)
    if row is not None:
        evidence["candidate_row_norm"] = norm

    if problem is None and shape[0] != vocab_size:
        problem = f"The token embedding has {shape[0]} rows, but config.json says vocab_size {vocab_size}."
    if problem is None and row is None and zero_rows:
        problem = (
            f"{no_row_reason[0].upper()}{no_row_reason[1:]}, but embedding rows {zero_rows[:10]} are all zero "
            f"({len(zero_rows)} in total). Only a frozen padding row stays exactly zero, so the bundled "
            "code is probably not the code the model was trained with."
        )
    if problem is None and row is not None and row not in set(zero_rows):
        problem = (
            f"The export's code freezes embedding row {row} (padding_idx from {evidence['padding_from']}), "
            f"but that row is not zero in the weights (norm {norm:.4g}). Training would have kept a frozen "
            "row at zero, so the bundled code or config.json is not what the model was trained with."
        )
    if kind == "unknown":
        problem = f"{problem} All-zero embedding rows in the weights: {zero_rows[:10]} ({len(zero_rows)} in total)."
    return evidence, row, problem, zero_rows


def _lineage(row: int | None, vocab_size: int) -> str:
    """The lineage label for a frozen row and a vocabulary size (see LINEAGES)."""
    if row is None and vocab_size == PRESETS["qwen3"]["vocab_size"]:
        return LINEAGE_RELEASE_QWEN3
    return LINEAGE_UNKNOWN


def parse_padding_choice(value: Any) -> int | str | None:
    """Turn a --embedding-padding-idx value into "auto", None (freeze no row) or a row index.

    None and "auto" both mean "detect it from the export". The string "none" means "freeze no row".
    """
    if value is None:
        return "auto"
    if isinstance(value, bool):
        raise ValueError(f"embedding_padding_idx must be 'auto', 'none' or a row index, got {value!r}.")
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    if text == "none":
        return None
    if text.isdigit():
        return int(text)
    raise ValueError(f"embedding_padding_idx must be 'auto', 'none' or a row index, got {value!r}.")


def detect_embedding_padding(
    export_dir: str | os.PathLike,
    *,
    override: Any = "auto",
    weight_names: list[str] | None = None,
) -> EmbeddingPadding:
    """Decide which token-embedding row the prepared dir freezes, following the model's training.

    Two independent sources must agree:
    - the export's own ``modelling_ouro.py``, parsed (never imported): does the call that builds
      ``embed_tokens`` pass a padding_idx, and which config value or literal does it name?
    - the weights: a row frozen as padding_idx stays exactly zero through training, and no other
      row of a trained embedding is exactly zero.
    When the code takes the row from config.json, the training-time value in export_info.json must
    match it as well.

    Any disagreement raises ``PaddingDetectionError``. The caller can then decide by hand with
    ``override`` (a row index, or "none"); the evidence is still collected and recorded then.
    """
    export = Path(export_dir).expanduser().resolve()
    config = json.loads((export / "config.json").read_text(encoding="utf-8"))
    vocab_size = int(config["vocab_size"])
    choice = parse_padding_choice(override)
    if weight_names is None:
        weight_names = _weight_files(export)
    evidence, detected, problem, zero_rows = _padding_evidence(export, config, weight_names)

    if choice == "auto":
        if problem is not None:
            raise PaddingDetectionError(f"{export}: {problem} {PADDING_OVERRIDE_HINT}")
        lineage = _lineage(detected, vocab_size)
        if detected is not None and len(zero_rows) > 1:
            logger.warning(
                "%s: the code freezes embedding row %d, and %d other rows are all zero as well (%s). Only "
                "row %d is frozen.", export, detected, len(zero_rows) - 1,
                [r for r in zero_rows if r != detected][:10], detected,
            )
        if lineage == LINEAGE_UNKNOWN:
            logger.warning(
                "%s: frozen embedding row %s with vocab_size %d does not match the release Qwen3 line; "
                "the lineage is recorded as unknown.", export, detected, vocab_size,
            )
        return EmbeddingPadding(detected, lineage, "detected", evidence)

    if choice is not None and not 0 <= choice < vocab_size:
        raise ValueError(f"--embedding-padding-idx {choice} is not a row below vocab_size {vocab_size}.")
    if problem is not None:
        evidence["detection_error"] = problem
    if choice is not None and zero_rows is not None and choice not in set(zero_rows):
        # A frozen row that training kept trainable is never right: for a release model that row can
        # be the eos and document separator (151643), which must stay trainable. Refuse instead of
        # warning, so a wrong override cannot reach training.
        raise PaddingDetectionError(
            f"{export}: --embedding-padding-idx {choice} names a row that is not zero in the weights, "
            "so training never froze it. Freezing it in RL would hold a trained row fixed. "
            f"Use --embedding-padding-idx none or one of the zero rows {sorted(zero_rows)[:10]}."
        )
    agrees = problem is None and choice == detected
    lineage = _lineage(choice, vocab_size) if agrees else LINEAGE_UNKNOWN
    return EmbeddingPadding(choice, lineage, "override", evidence)


def prepare_model_dir(
    export_dir: str,
    out_dir: str,
    *,
    preset: str = "auto",
    eos_token_ids: list[int] | None = None,
    pad_token_id: int | None = None,
    bos_token_id: int | str | None = None,
    max_position_embeddings: int | None = None,
    tokenizer_dir: str | None = None,
    materialize_weights: bool = False,
    link_code: bool = False,
    force: bool = False,
    embedding_padding_idx: int | str | None = "auto",
) -> dict:
    """Build a verl-ready model dir at ``out_dir`` from the HF export ``export_dir``.

    Token-id arguments left at None take the preset's values. ``bos_token_id="none"`` clears the
    preset's bos id. Returns the manifest that was written.

    ``embedding_padding_idx`` is the token-embedding row that stays frozen. "auto" (or None)
    detects it from the export with ``detect_embedding_padding`` and refuses an export whose code
    and weights disagree. A row index or "none" (freeze no row) decides it by hand.

    The dir is built in a temporary sibling and renamed into place at the end, so a crash never
    leaves a half-built dir behind under the final name. With ``force``, an existing dir is
    replaced only if it holds this tool's manifest; any other existing path is left alone.
    """
    export = Path(export_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().absolute()
    tok_dir = Path(tokenizer_dir).expanduser().resolve() if tokenizer_dir else export

    config_path = export / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} does not exist; --export must be an HF export dir.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "ouro":
        raise ValueError(f"{config_path} has model_type {config.get('model_type')!r}, expected 'ouro'.")
    vocab_size = int(config["vocab_size"])
    preset_name, ids = _resolve_token_ids(preset, vocab_size, eos_token_ids, pad_token_id, bos_token_id)
    weight_names = _weight_files(export)
    tokenizer_names = [name for name in TOKENIZER_FILES if (tok_dir / name).exists()]
    # Fails early when the tokenizer dir has neither tokenizer.json nor vocab.json + merges.txt.
    common.tokenizer_sha256(tok_dir)
    for name in MODEL_CODE_FILES:
        if not (MODEL_CODE_DIR / name).exists():
            raise FileNotFoundError(f"Ported model code {MODEL_CODE_DIR / name} is missing.")

    if out.is_symlink() or (out.exists() and not out.is_dir()):
        raise FileExistsError(f"{out} exists and is not a directory; refusing to touch it.")
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} already exists. Pass --force to rebuild it.")
        if not _has_our_manifest(out):
            raise FileExistsError(
                f"{out} exists but has no {common.MODEL_MANIFEST} written by this tool. "
                "--force only replaces dirs this tool prepared; remove or rename it by hand."
            )

    # Decided before anything is written, so a refused export leaves nothing behind.
    padding = detect_embedding_padding(export, override=embedding_padding_idx, weight_names=weight_names)

    # The ported RoPE falls back to OuroConfig's default of 10000 when config.json carries no
    # rope_theta. The release stages train with other values (1e6 at 16K, 4e6 at 32K), so a missing
    # value would silently give the model the wrong positions. Refuse such an export instead.
    rope_parameters = config.get("rope_parameters") or {}
    if config.get("rope_theta") is None and rope_parameters.get("rope_theta") is None:
        raise ValueError(
            f"{export}/config.json has no rope_theta. The model would silently get RoPE theta 10000; "
            "re-export it with the rope_theta it was trained with."
        )

    overrides: dict[str, Any] = {
        "early_exit_threshold": None,
        "auto_map": dict(AUTO_MAP),
        "embedding_padding_idx": padding.embedding_padding_idx,
    }
    export_max_positions = config.get("max_position_embeddings")
    if max_position_embeddings is not None:
        max_position_embeddings = int(max_position_embeddings)
        if max_position_embeddings < 1:
            raise ValueError("max_position_embeddings must be positive.")
        # The export's config normally states the length the model was trained at. A 16K-stage
        # export may still say 4096, the value of the stage it started from, which is why a larger
        # value can be passed here. It is only right for a model that really trained that long.
        if isinstance(export_max_positions, int) and max_position_embeddings > export_max_positions:
            logger.warning(
                "max_position_embeddings=%d is above the export's own %d. That is only correct for a "
                "model trained at that length, such as the 16K stage; for any other model, quality past "
                "%d positions is unknown.",
                max_position_embeddings, export_max_positions, export_max_positions,
            )
        overrides["max_position_embeddings"] = max_position_embeddings
    new_config = {**config, **overrides}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        for name in weight_names:
            _link_or_copy(export / name, tmp / name, copy=materialize_weights)
        for name in tokenizer_names:
            _link_or_copy(tok_dir / name, tmp / name, copy=False)
        for name in MODEL_CODE_FILES:
            _link_or_copy(MODEL_CODE_DIR / name, tmp / name, copy=not link_code)

        (tmp / "config.json").write_text(json.dumps(new_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Only token ids go into generation_config.json. vLLM also takes default sampling
        # settings (temperature, top_p) from this file, and those must come from the verl config.
        generation = {"eos_token_id": ids["eos_token_ids"], "pad_token_id": ids["pad_token_id"]}
        if ids["bos_token_id"] is not None:
            generation["bos_token_id"] = ids["bos_token_id"]
        (tmp / "generation_config.json").write_text(json.dumps(generation, indent=2) + "\n", encoding="utf-8")

        payload = {
            "export_dir": str(export),
            "tokenizer_dir": str(tok_dir),
            "preset": preset_name,
            "eos_token_ids": ids["eos_token_ids"],
            "pad_token_id": ids["pad_token_id"],
            "bos_token_id": ids["bos_token_id"],
            "vocab_size": vocab_size,
            "total_ut_steps": int(config.get("total_ut_steps", 4)),
            "max_position_embeddings": int(new_config["max_position_embeddings"]),
            "export_max_position_embeddings": export_max_positions,
            "rope_theta": config.get("rope_theta"),
            "rope_scaling": config.get("rope_scaling"),
            "lineage": padding.lineage,
            "embedding_padding_idx": padding.embedding_padding_idx,
            "embedding_padding_source": padding.source,
            "embedding_padding_evidence": padding.evidence,
            "config_overrides": overrides,
            "weights": "copy" if materialize_weights else "symlink",
            "code": "symlink" if link_code else "copy",
            "code_sha256": {name: common.sha256_file(MODEL_CODE_DIR / name) for name in MODEL_CODE_FILES},
            "tokenizer_sha256": common.tokenizer_sha256(tmp),
        }
        # The manifest is the last file written: the launcher treats its presence as "complete".
        manifest = common.write_manifest(tmp / common.MODEL_MANIFEST, common.MODEL_MANIFEST_KIND, payload)

        if out.exists():
            old = out.parent / f".{out.name}.old-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            os.rename(out, old)
            try:
                os.rename(tmp, out)
            except OSError:
                os.rename(old, out)
                raise
            # rmtree unlinks symlinks without following them, so the export files stay untouched.
            shutil.rmtree(old, ignore_errors=True)
            if old.exists():
                logger.warning("Could not fully remove the replaced dir %s; delete it by hand.", old)
        else:
            os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


def _eos_list(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(v) for v in value] if isinstance(value, (list, tuple)) else [int(value)]


def recorded_padding_problems(model_dir: str | os.PathLike, manifest: dict[str, Any]) -> list[str]:
    """Problems with the frozen-row record of a prepared dir; an empty list means it is usable.

    A dir whose config.json or manifest lacks the frozen-row record was not written by this version
    of the tool and has to be rebuilt. This uses only the standard library, so the launcher's
    pre-flight can call it without importing torch.
    """
    root = Path(model_dir)
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"cannot read {root / 'config.json'}: {exc}"]
    missing = [] if "embedding_padding_idx" in config else ["config.json key embedding_padding_idx"]
    missing += [f"manifest key {key}" for key in ("embedding_padding_idx", "lineage") if key not in manifest]
    if missing:
        return [
            f"the dir was prepared without a record of the frozen embedding row (no {', '.join(missing)}). "
            "Rebuild it with prepare_model --force."
        ]
    problems = []
    if config["embedding_padding_idx"] != manifest["embedding_padding_idx"]:
        problems.append(
            f"config.json has embedding_padding_idx={config['embedding_padding_idx']!r}, but the manifest "
            f"recorded {manifest['embedding_padding_idx']!r}. Rebuild the dir with prepare_model --force "
            "instead of editing config.json."
        )
    if manifest["lineage"] not in LINEAGES:
        problems.append(f"the manifest has lineage {manifest['lineage']!r}, expected one of {list(LINEAGES)}")
    return problems


def check_model_dir(model_dir: str, *, load_weights: bool = False, vllm_view: bool = True) -> dict:
    """Check that a prepared dir loads the way verl and vLLM will load it. Raises on problems.

    Always checked: the manifest, broken symlinks, the code copies, the recorded frozen embedding
    row, the config class and its early_exit_threshold, the tokenizer, and the generation stop
    tokens. With ``vllm_view`` the stop tokens and RoPE settings are also read through vLLM's own
    helpers when vllm is importable. With ``load_weights`` the model is loaded on CPU in fp32: no
    checkpoint key may be missing, unexpected or mismatched, the frozen row must be the recorded
    one and exactly zero, and an 8-token forward on the final loop must be finite.

    Prints a short summary and returns it as a dict.
    """
    root = Path(model_dir).expanduser().resolve()
    manifest = common.read_manifest(root / common.MODEL_MANIFEST, common.MODEL_MANIFEST_KIND)
    problems: list[str] = []
    notes: list[str] = []
    summary: dict[str, Any] = {
        "model_dir": str(root),
        "preset": manifest["preset"],
        "lineage": manifest.get("lineage", "missing"),
        "embedding_padding_idx": manifest.get("embedding_padding_idx", "missing"),
    }
    padding_problems = recorded_padding_problems(root, manifest)
    problems.extend(padding_problems)

    broken = sorted(p.name for p in root.iterdir() if p.is_symlink() and not p.exists())
    if broken:
        problems.append(f"broken symlinks (export moved or unreadable?): {broken}")
    for name, sha in manifest["code_sha256"].items():
        if common.sha256_file(root / name) != sha:
            problems.append(f"{name} differs from the copy recorded in the manifest")
        elif (MODEL_CODE_DIR / name).exists() and common.sha256_file(MODEL_CODE_DIR / name) != sha:
            # Not an error: the dir is consistent, it just holds an older copy of the port.
            notes.append(f"{name} is older than {MODEL_CODE_DIR / name}; re-run prepare_model with --force")

    # transformers reads its module cache location at import time.
    common.apply_cache_env()
    import transformers
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig

    config = AutoConfig.from_pretrained(str(root), trust_remote_code=True)
    summary["config_class"] = type(config).__name__
    summary["early_exit_threshold"] = getattr(config, "early_exit_threshold", "missing")
    summary["vocab_size"] = config.vocab_size
    summary["total_ut_steps"] = config.total_ut_steps
    summary["max_position_embeddings"] = config.max_position_embeddings
    summary["rope_theta"] = getattr(config, "rope_theta", "missing")
    # The config class must hand the recorded row to the model. An old configuration_ouro.py copy
    # would still read the key as a plain attribute, so this compares the value, not the class.
    if not padding_problems and getattr(config, "embedding_padding_idx", "missing") != manifest["embedding_padding_idx"]:
        problems.append(
            f"AutoConfig gives embedding_padding_idx={getattr(config, 'embedding_padding_idx', 'missing')!r}, "
            f"the manifest recorded {manifest['embedding_padding_idx']!r}"
        )
    if manifest.get("rope_theta") is not None and summary["rope_theta"] != manifest["rope_theta"]:
        problems.append(f"config rope_theta={summary['rope_theta']} but the export had {manifest['rope_theta']}")
    if summary["config_class"] != "OuroConfig":
        problems.append(f"AutoConfig returned {summary['config_class']}, expected OuroConfig")
    if summary["early_exit_threshold"] is not None:
        problems.append(f"early_exit_threshold is {summary['early_exit_threshold']!r}, expected None")
    for key in ("vocab_size", "total_ut_steps", "max_position_embeddings"):
        if summary[key] != manifest[key]:
            problems.append(f"config {key}={summary[key]} but the manifest says {manifest[key]}")

    tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True)
    summary["tokenizer_len"] = len(tokenizer)
    if len(tokenizer) > config.vocab_size:
        problems.append(f"tokenizer has {len(tokenizer)} tokens, more than vocab_size {config.vocab_size}")
    if common.tokenizer_sha256(root) != manifest["tokenizer_sha256"]:
        problems.append("tokenizer files changed since the dir was prepared")

    generation = GenerationConfig.from_pretrained(str(root))
    summary["generation_eos"] = _eos_list(generation.eos_token_id)
    summary["generation_pad"] = generation.pad_token_id
    if summary["generation_eos"] != manifest["eos_token_ids"]:
        problems.append(f"generation eos {summary['generation_eos']} != manifest {manifest['eos_token_ids']}")
    if generation.pad_token_id != manifest["pad_token_id"]:
        problems.append(f"generation pad {generation.pad_token_id} != manifest {manifest['pad_token_id']}")

    summary["vllm_eos"] = "skipped"
    if vllm_view:
        try:
            from vllm.transformers_utils.config import try_get_generation_config
        except Exception as exc:  # vllm is optional here; a CPU-only env may not have it.
            summary["vllm_eos"] = f"skipped ({type(exc).__name__})"
        else:
            vllm_generation = try_get_generation_config(str(root), trust_remote_code=True)
            vllm_eos = _eos_list(None if vllm_generation is None else vllm_generation.eos_token_id)
            summary["vllm_eos"] = vllm_eos
            if set(vllm_eos) != set(manifest["eos_token_ids"]):
                problems.append(f"vLLM sees eos {vllm_eos}, manifest says {manifest['eos_token_ids']}")
            # vLLM builds RoPE from its own reading of config.json (rope_parameters on recent
            # versions). A 16K export may carry a different rope_theta, and vLLM has to see the
            # same value as the HF model, or rollouts and training would use different positions.
            from vllm.transformers_utils.config import get_config as vllm_get_config

            try:
                vllm_config = vllm_get_config(str(root), trust_remote_code=True)
            except Exception as exc:  # The run would fail the same way, so this is a problem.
                problems.append(f"vLLM cannot read the config: {type(exc).__name__}: {exc}")
            else:
                rope_parameters = getattr(vllm_config, "rope_parameters", None) or {}
                vllm_theta = rope_parameters.get("rope_theta", getattr(vllm_config, "rope_theta", None))
                summary["vllm_rope_theta"] = vllm_theta
                summary["vllm_max_position_embeddings"] = vllm_config.max_position_embeddings
                if vllm_theta is None or float(vllm_theta) != float(config.rope_theta):
                    problems.append(f"vLLM sees rope_theta {vllm_theta}, the HF config has {config.rope_theta}")
                if vllm_config.max_position_embeddings != manifest["max_position_embeddings"]:
                    problems.append(
                        f"vLLM sees max_position_embeddings {vllm_config.max_position_embeddings}, "
                        f"the manifest says {manifest['max_position_embeddings']}"
                    )

    if load_weights:
        import torch
        from transformers import AutoModelForCausalLM

        # transformers 5 renamed torch_dtype to dtype; 4.57 still expects the old name.
        dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        model, info = AutoModelForCausalLM.from_pretrained(
            str(root),
            trust_remote_code=True,
            attn_implementation="sdpa",
            output_loading_info=True,
            **{dtype_key: torch.float32},
        )
        model.eval()
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
            keys = sorted(str(k) for k in info.get(key, ()) or ())
            summary[key] = len(keys)
            if keys:
                problems.append(f"{len(keys)} {key}: {keys[:5]}")
        if not config.tie_word_embeddings and (
            model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
        ):
            problems.append("lm_head is tied to embed_tokens although tie_word_embeddings is false")
        embedding = model.model.embed_tokens
        summary["loaded_padding_idx"] = embedding.padding_idx
        if not padding_problems:
            if embedding.padding_idx != manifest["embedding_padding_idx"]:
                problems.append(
                    f"the loaded model freezes embedding row {embedding.padding_idx}, the manifest "
                    f"recorded {manifest['embedding_padding_idx']}"
                )
            elif embedding.padding_idx is not None:
                summary["padding_row_zero"] = bool(torch.count_nonzero(embedding.weight[embedding.padding_idx]) == 0)
                if not summary["padding_row_zero"]:
                    problems.append(f"the frozen embedding row {embedding.padding_idx} is not zero in the weights")
        ids = tokenizer.encode("Question: What is 7 plus 5?\nAnswer: 12", add_special_tokens=False)
        ids = (ids * 8)[:8]
        with torch.no_grad():
            logits = model(
                input_ids=torch.tensor([ids]), use_cache=False, exit_at_step=int(config.total_ut_steps) - 1
            ).logits
        summary["forward_shape"] = list(logits.shape)
        summary["forward_finite"] = bool(torch.isfinite(logits).all())
        if not summary["forward_finite"]:
            problems.append("the 8-token forward produced non-finite logits")
        if list(logits.shape) != [1, 8, config.vocab_size]:
            problems.append(f"logits shape {list(logits.shape)}, expected [1, 8, {config.vocab_size}]")
        del model

    summary["notes"] = notes
    summary["problems"] = problems
    print(f"[check] {root}")
    for key, value in summary.items():
        if key not in ("model_dir", "notes", "problems"):
            print(f"[check]   {key}: {value}")
    for note in notes:
        print(f"[check]   note: {note}")
    for problem in problems:
        print(f"[check]   PROBLEM: {problem}")
    print(f"[check] {'FAILED' if problems else 'OK'}")
    if problems:
        raise RuntimeError(f"Model dir {root} failed the check: " + "; ".join(problems))
    return summary


def _parse_ids(text: str) -> list[int]:
    return [int(part) for part in text.replace(" ", "").split(",") if part]


def _padding_arg(text: str) -> int | str:
    """Validate --embedding-padding-idx and keep "none" as a string.

    prepare_model_dir reads a Python None as "detect", so returning None here would quietly turn
    "--embedding-padding-idx none" into automatic detection.
    """
    choice = parse_padding_choice(text)
    return "none" if choice is None else choice


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orbit_verl.model.prepare_model",
        description="Build a verl-ready OrbitLM model dir from an HF export.",
    )
    parser.add_argument("--export", help="HF export dir (config.json, safetensors, tokenizer). "
                        "Leave out together with --check/--check-load to check an existing dir.")
    parser.add_argument("--name", help="output dir name under --out-root")
    parser.add_argument("--runs-root", help=f"root of all outputs and caches (default: ${common.RUNS_ROOT_ENV})")
    parser.add_argument("--out-root", help="default: <runs root>/models")
    parser.add_argument("--out-dir", help="explicit output dir; overrides --out-root/--name")
    parser.add_argument("--preset", default="auto", choices=["auto", *PRESETS],
                        help="stop-token preset; auto picks it by vocab_size")
    parser.add_argument("--eos-ids", type=_parse_ids, help="comma-separated eos ids, e.g. 151645,151643")
    parser.add_argument("--pad-id", type=int)
    parser.add_argument("--bos-id", help="bos id, or 'none' to leave it unset")
    parser.add_argument("--max-position-embeddings", type=int,
                        help="override the export's value; only right for a model trained at that length")
    parser.add_argument("--embedding-padding-idx", type=_padding_arg, default="auto",
                        metavar="auto|none|N",
                        help="the token-embedding row that stays frozen. auto (default) detects it from "
                        "the export's own model code and weights and refuses on any contradiction; "
                        "none freezes no row; N freezes row N")
    parser.add_argument("--tokenizer-dir", help="default: the export dir")
    parser.add_argument("--materialize-weights", action="store_true",
                        help="copy safetensors instead of symlinking (for nodes that cannot read the export)")
    parser.add_argument("--link-code", action="store_true",
                        help="symlink the ported .py files instead of copying them (development only)")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing output dir, only if it holds this tool's manifest")
    parser.add_argument("--check", action="store_true", help="after building, load config, tokenizer, generation config")
    parser.add_argument("--check-load", action="store_true", help="--check plus loading the weights on CPU")
    parser.add_argument("--no-vllm-check", action="store_true", help="skip reading the stop tokens through vLLM")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="[prepare] %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        runs_root = common.resolve_runs_root(args.runs_root)
    except common.RunsRootError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # check_model_dir sets the caches again without a root; it then finds this one.
    os.environ[common.RUNS_ROOT_ENV] = runs_root
    # Set the cache variables before anything below imports transformers or vllm.
    common.apply_cache_env(runs_root)
    if args.out_dir:
        out_dir = args.out_dir
    elif args.name:
        out_dir = os.path.join(args.out_root or os.path.join(runs_root, "models"), args.name)
    else:
        print("error: pass --name or --out-dir", file=sys.stderr)
        return 2
    checking = args.check or args.check_load
    if args.export is None and not checking:
        print("error: --export is required unless only --check/--check-load is wanted", file=sys.stderr)
        return 2

    if args.export is not None:
        try:
            manifest = prepare_model_dir(
                args.export,
                out_dir,
                preset=args.preset,
                eos_token_ids=args.eos_ids,
                pad_token_id=args.pad_id,
                bos_token_id=args.bos_id,
                max_position_embeddings=args.max_position_embeddings,
                tokenizer_dir=args.tokenizer_dir,
                materialize_weights=args.materialize_weights,
                link_code=args.link_code,
                force=args.force,
                embedding_padding_idx=args.embedding_padding_idx,
            )
        except (FileExistsError, FileNotFoundError, ValueError) as exc:
            # PaddingDetectionError is a ValueError, so a refused export ends up here too.
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"[prepare] wrote {out_dir}")
        for key in ("preset", "eos_token_ids", "pad_token_id", "bos_token_id", "vocab_size",
                    "max_position_embeddings", "rope_theta", "lineage", "embedding_padding_idx",
                    "embedding_padding_source", "weights", "code"):
            print(f"[prepare]   {key}: {manifest[key]}")

    if checking:
        try:
            check_model_dir(out_dir, load_weights=args.check_load, vllm_view=not args.no_vllm_check)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
