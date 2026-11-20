"""Pre-flight checks of a composed config, and building the prepared model dir and dataset.

A run needs two prepared directories: the model dir (``orbit.model_dir``), built by
``orbit_verl.model.prepare_model``, and the dataset dir (``orbit.data_dir``), built by
``orbit_verl.data.prepare_data``. Each tool writes a manifest last, so a directory with a valid
manifest is complete. The checks here read those manifests and compare them with the config, so
that a run never starts on a stale or half-written directory.

A distillation config builds its dataset with ``orbit_verl.data.distill_data`` instead, which
gives every row its own prompt mode. It says so with ``orbit.prepare.data.builder: distill``, or
with the prompt mode "mixed" (``data_builder``). Its dataset is then checked against the keys of
that tool: ``pool``, ``gold``, ``val``, ``think_share``, ``seed`` and ``limit`` under
``orbit.prepare.data``.

Every problem carries the exact command that fixes it. The launcher prints those commands and
exits with code 2 instead of failing an hour later inside a Ray worker.

The checks use only the standard-library part of the model tool (``recorded_padding_problems``).
The tools' heavy work, which pulls in transformers, torch and pandas, runs only inside
``run_prepare``.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from orbit_verl import common
from orbit_verl.model import MODEL_CODE_DIR, MODEL_CODE_FILES
from orbit_verl.rltt.drift import THRESHOLD_KEY, check_threshold

# override_config keys that would change which embedding row is frozen behind the model dir's back.
# The prepared model dir records the row that the model's training kept at zero and frozen (none for
# the release models). pad_token_id does not select that row, and keeping it in a config would only
# mislead the reader. check_settings also refuses the
# nested override_config.model_config block, which verl would apply in place of these keys.
_PADDING_OVERRIDES = {
    "embedding_padding_idx": "it would replace the row that the model dir recorded from the model's "
                             "training code. Rebuild the model dir with prepare_model "
                             "--embedding-padding-idx N|none instead",
    "pad_token_id": "it no longer selects the frozen embedding row; the model dir decides. Remove it",
}


@dataclass(frozen=True)
class Problem:
    """One reason why the run cannot start.

    ``kind`` is ``model``, ``data``, ``lineage``, ``config``, ``lengths``, ``plugin``,
    ``sandbox``, ``clearml``, ``resume`` or ``gpus``. ``buildable`` is true when ``--prepare`` can fix the problem by building
    a missing directory. A directory that exists but disagrees with the config needs the printed
    command with ``--force``, or the launcher's ``--force-prepare``. ``--prepare`` never does that on
    its own, because it deletes the old directory.
    """

    kind: str
    message: str
    commands: tuple[str, ...] = field(default=())
    buildable: bool = False


def _get(cfg: Any, dotted: str, default: Any = None) -> Any:
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, dotted, default=default)
    return default if value is None else value


def _is_missing(cfg: Any, dotted: str) -> bool:
    """True when ``dotted`` still holds the mandatory marker ``???``, as in a template config."""
    from omegaconf import OmegaConf

    parent, _, key = dotted.rpartition(".")
    node = OmegaConf.select(cfg, parent) if parent else cfg
    return node is not None and OmegaConf.is_missing(node, key)


def _same_path(a: Any, b: Any) -> bool:
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


def _tool(module: str, args: list[str], cfg: Any, python: str | None = None) -> str:
    """A copy-pasteable command line that runs one of our tools from any directory.

    Both tools keep their caches under the runs root, so the command passes it with --runs-root.
    It then also works in a shell where ORBIT_VERL_RUNS is not exported.
    """
    python = python or sys.executable
    runs_root = _get(cfg, "orbit.runs_root")
    if runs_root is not None:
        args = [*args, "--runs-root", str(runs_root)]
    return shlex.join([f"PYTHONPATH={os.path.join(str(_get(cfg, 'orbit.repo_root')), 'src')}", python, "-m",
                       module, *args])


def model_command(cfg: Any, *, force: bool = False, python: str | None = None) -> str:
    """The prepare_model command for this config's ``orbit.prepare.model`` block."""
    model = _get(cfg, "orbit.prepare.model")
    args = ["--export", str(model.export), "--out-dir", str(_get(cfg, "orbit.model_dir")),
            "--preset", str(model.get("preset") or "auto")]
    if model.get("max_position_embeddings") is not None:
        args += ["--max-position-embeddings", str(model.max_position_embeddings)]
    if model.get("materialize_weights"):
        args.append("--materialize-weights")
    if force:
        args.append("--force")
    return _tool("orbit_verl.model.prepare_model", args, cfg, python)


DEFAULT_BUILDER = "prepare_data"
DISTILL_BUILDER = "distill"
DATA_BUILDERS = (DEFAULT_BUILDER, DISTILL_BUILDER)
# The prompt mode that a distillation dataset records. Only the distill builder can build it.
MIXED_PROMPT_MODE = "mixed"


def data_builder(cfg: Any) -> str:
    """The tool that builds this config's dataset: ``prepare_data`` or ``distill``.

    ``orbit.prepare.data.builder`` decides when it is set. When it is not set, a prompt mode of
    "mixed" selects the distill builder, because no single prompt mode describes such a dataset and
    prepare_data cannot build it. Every other config keeps prepare_data. An unknown name is returned
    as it is, and ``check_data`` reports it.
    """
    builder = _get(cfg, "orbit.prepare.data.builder")
    if builder is not None:
        return str(builder)
    if str(_get(cfg, "orbit.prepare.data.prompt_mode")) == MIXED_PROMPT_MODE:
        return DISTILL_BUILDER
    return DEFAULT_BUILDER


def _spec_list(value: Any) -> list[str]:
    """A config value that holds input specs, as a list of strings. A single string counts as one spec."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _distill_settings(cfg: Any) -> dict[str, Any]:
    """The arguments of ``distill_data.convert`` that come from ``orbit.prepare.data``.

    A pool marked ``???`` counts as unset, like every value that ``_get`` reads.
    """
    limit = _get(cfg, "orbit.prepare.data.limit")
    pool = _get(cfg, "orbit.prepare.data.pool")
    return {
        "pool": None if pool is None else str(pool),
        "gold": _spec_list(_get(cfg, "orbit.prepare.data.gold")),
        "val": _spec_list(_get(cfg, "orbit.prepare.data.val")),
        "think_share": float(_get(cfg, "orbit.prepare.data.think_share", 0.5)),
        "seed": int(_get(cfg, "orbit.prepare.data.seed", 0)),
        "limit": None if limit is None else int(limit),
    }


def distill_command(cfg: Any, *, force: bool = False, python: str | None = None) -> str:
    """The distill_data command for this config's ``orbit.prepare.data`` block."""
    settings = _distill_settings(cfg)
    args = []
    if settings["pool"] is not None:
        args += ["--pool", settings["pool"]]
    for spec in settings["gold"]:
        args += ["--gold", spec]
    for spec in settings["val"]:
        args += ["--val", spec]
    args += ["--tokenizer", str(_get(cfg, "orbit.model_dir")),
             "--max-prompt-length", str(_get(cfg, "data.max_prompt_length")),
             "--out-dir", str(_get(cfg, "orbit.data_dir")),
             "--think-share", str(settings["think_share"]), "--seed", str(settings["seed"])]
    if settings["limit"] is not None:
        args += ["--limit", str(settings["limit"])]
    if force:
        args.append("--force")
    return _tool("orbit_verl.data.distill_data", args, cfg, python)


def data_command(cfg: Any, *, force: bool = False, python: str | None = None) -> str:
    """The command that builds this config's dataset: prepare_data, or distill_data for a distillation config."""
    if data_builder(cfg) == DISTILL_BUILDER:
        return distill_command(cfg, force=force, python=python)
    data = _get(cfg, "orbit.prepare.data")
    args = ["--src", str(data.src), "--prompt-mode", str(data.prompt_mode),
            "--tokenizer", str(_get(cfg, "orbit.model_dir")),
            "--max-prompt-length", str(_get(cfg, "data.max_prompt_length")),
            "--out-dir", str(_get(cfg, "orbit.data_dir"))]
    if data.get("data_source") is not None:
        args += ["--data-source", str(data.data_source)]
    args += ["--question-field", str(data.get("question_field") or "instruction"),
             "--answer-field", str(data.get("answer_field") or "response"),
             "--val-size", str(data.get("val_size", 64))]
    if data.get("val_src") is not None:
        args += ["--val-src", str(data.val_src)]
    if data.get("limit") is not None:
        args += ["--limit", str(data.limit)]
    if force:
        args.append("--force")
    return _tool("orbit_verl.data.prepare_data", args, cfg, python)


def _stale_model_code(model_dir: str) -> list[str]:
    """Names of the model code files in ``model_dir`` whose content differs from the repo's port.

    The prepared dir holds its own copy of the model code, and verl and vLLM load that copy
    through ``trust_remote_code``. When the port in ``orbit_verl/model`` changes later, an old
    dir would keep running the old code without any warning. The files themselves are compared,
    not the hashes in the manifest, because the copy in the dir is what actually runs.
    """
    return [name for name in MODEL_CODE_FILES
            if common.sha256_file(os.path.join(model_dir, name)) != common.sha256_file(MODEL_CODE_DIR / name)]


def _padding_problem(cfg: Any, model_dir: str, manifest: Mapping[str, Any]) -> Problem | None:
    """Refuse a model dir that does not record its frozen embedding row, or records two different ones.

    The rules live in prepare_model, next to the code that writes the record, and its
    ``check_model_dir`` applies the same function. It reads only config.json and the manifest, so
    the pre-flight stays cheap. ``--prepare`` does not rebuild a refused dir on its own, because
    rebuilding replaces the old dir.
    """
    from orbit_verl.model.prepare_model import recorded_padding_problems

    found = recorded_padding_problems(model_dir, dict(manifest))
    if not found:
        return None
    return Problem("model", f"Model dir {model_dir}: " + " ".join(found) + " Rebuild command:",
                   (model_command(cfg, force=True),))


# How a problem message names the flag that rebuilds a prepared dir which does not match the config.
FORCE_PREPARE_HINT = "rebuild it with --force-prepare, which replaces the old dir"


def check_model(cfg: Any) -> list[Problem]:
    """Problems with ``orbit.model_dir``; an empty list means it is ready."""
    if _is_missing(cfg, "orbit.prepare.model.export"):
        # The experiment configs name no checkpoint. The model comes from --export.
        return [Problem("model", "No model is set. Pass the HF export dir of the checkpoint to train "
                        "(config.json, *.safetensors, modelling_ouro.py) to the launcher:",
                        ("--export /path/to/the/hf/export",))]
    model_dir = str(_get(cfg, "orbit.model_dir"))
    if not os.path.isdir(model_dir):
        export = str(_get(cfg, "orbit.prepare.model.export"))
        if not os.path.isfile(os.path.join(export, "config.json")):
            # Building would fail on the missing export, so the problem is not buildable.
            return [Problem("model", f"Model dir {model_dir} does not exist, and it cannot be built: {export} "
                            "has no config.json, so it is not an HF export dir. Pass the export dir of a "
                            "checkpoint with --export PATH.")]
        return [Problem("model", f"Model dir {model_dir} does not exist.", (model_command(cfg),), True)]
    missing = [name for name in ("config.json", "generation_config.json", *MODEL_CODE_FILES)
               if not os.path.isfile(os.path.join(model_dir, name))]
    try:
        manifest = common.read_manifest(os.path.join(model_dir, common.MODEL_MANIFEST),
                                        common.MODEL_MANIFEST_KIND)
    except FileNotFoundError:
        # The tool builds in a temporary dir and renames it at the end, so a dir without a
        # manifest was not made by it, and the tool refuses to replace it even with --force.
        return [Problem("model", f"{model_dir} exists but has no {common.MODEL_MANIFEST}, so "
                        "prepare_model did not build it. Remove or rename it by hand, then run:",
                        (model_command(cfg),))]
    except ValueError as exc:
        return [Problem("model", f"Model dir {model_dir} has an unusable manifest: {exc}",
                        (model_command(cfg, force=True),))]
    if missing:
        return [Problem("model", f"Model dir {model_dir} has a manifest but lacks {', '.join(missing)}.",
                        (model_command(cfg, force=True),))]
    padding = _padding_problem(cfg, model_dir, manifest)
    if padding is not None:
        return [padding]
    model = _get(cfg, "orbit.prepare.model")
    mismatches = []
    if manifest.get("export_dir") and not _same_path(manifest["export_dir"], model.export):
        # Two exports whose paths end in the same two parts get the same derived name. This is
        # where that collision is caught, before the run trains the other checkpoint.
        mismatches.append(f"it was built from {manifest['export_dir']}, not from {model.export}")
    preset = str(model.get("preset") or "auto")
    if preset != "auto" and manifest.get("preset") not in (None, preset):
        mismatches.append(f"it has preset {manifest['preset']}, the config asks for {preset}")
    if model.get("materialize_weights") and manifest.get("weights") == "symlink":
        mismatches.append("its weights are symlinks, the config asks for copies (materialize_weights)")
    want_len = model.get("max_position_embeddings")
    if want_len is not None and manifest.get("max_position_embeddings") not in (None, want_len):
        mismatches.append(f"it has max_position_embeddings={manifest['max_position_embeddings']}, "
                          f"the config asks for {want_len}")
    if mismatches:
        return [Problem("model", f"Model dir {model_dir} does not match the config: " + "; ".join(mismatches)
                        + ". Give this model a dir of its own with --model-name NAME, or "
                        + FORCE_PREPARE_HINT + ". The command that rebuilds it:",
                        (model_command(cfg, force=True),))]
    stale = _stale_model_code(model_dir)
    if stale:
        return [Problem("model", f"Model dir {model_dir} holds a copy of {', '.join(stale)} that differs "
                        f"from the one in {MODEL_CODE_DIR}. verl and vLLM load the copy in the dir, so the run "
                        "would not use the current model code. Rebuild the dir to copy the current code in:",
                        (model_command(cfg, force=True),))]
    return []


def _data_config_problem(cfg: Any) -> Problem | None:
    """A ``orbit.prepare.data`` block that no tool can build from or that the run cannot use, or None.

    Without this check a distillation config without a pool would print a build command that fails,
    and a run that routes rows to teachers by a column that its dataset lacks would fail only in
    the first rollout.
    """
    from orbit_verl.data import distill_data

    builder = data_builder(cfg)
    if builder not in DATA_BUILDERS:
        return Problem("config", f"orbit.prepare.data.builder={builder} is not a known dataset builder. Use "
                       f"{DISTILL_BUILDER} for a distillation dataset, or leave it null for prepare_data.",
                       (f"orbit.prepare.data.builder={DISTILL_BUILDER}",))
    if builder != DISTILL_BUILDER:
        # verl routes rows by distillation.teacher_key only when it runs more than one teacher. Each
        # row then needs that column, and verl would fail on the first rollout without it.
        routed = (bool(_get(cfg, "distillation.enabled", False))
                  and len(_get(cfg, "distillation.teacher_models", {})) > 1
                  and str(_get(cfg, "distillation.teacher_key", "")) == distill_data.TEACHER_ROUTE)
        if routed:
            return Problem("config", f"distillation.teacher_key is {distill_data.TEACHER_ROUTE}, a column that "
                           "only the distill builder writes, but this config builds its dataset with prepare_data.",
                           (f"orbit.prepare.data.builder={DISTILL_BUILDER}", "distillation.teacher_key=data_source"))
        return None
    settings = _distill_settings(cfg)
    if settings["pool"] is None and not settings["gold"]:
        return Problem("config", "The distill builder needs a prompt pool (orbit.prepare.data.pool), gold "
                       "inputs (orbit.prepare.data.gold), or both.",
                       ("orbit.prepare.data.pool=/path/to/pool.jsonl",))
    if not settings["val"]:
        return Problem("config", "The distill builder needs at least one validation set in "
                       "orbit.prepare.data.val, written PATH:DATA_SOURCE:PROMPT_MODE[:LIMIT]. verl fails at "
                       "start-up on an empty val.parquet.",
                       ("orbit.prepare.data.val=['/path/to/math500.jsonl:orbit/math#think:sft_think']",))
    try:
        for spec in settings["gold"]:
            distill_data.parse_gold_spec(spec)
        for spec in settings["val"]:
            distill_data.parse_val_spec(spec)
    except ValueError as exc:
        return Problem("config", f"orbit.prepare.data: {exc}")
    return None


def _distill_mismatches(cfg: Any, manifest: Mapping[str, Any]) -> list[str]:
    """How a dataset differs from the distill settings of the config; empty when it matches.

    The specs are compared in their canonical form, so a relative and an absolute path to the same
    file are equal. Their order counts, because it decides the order in which the seeded sampling
    reads the inputs.
    """
    from orbit_verl.data import distill_data
    from orbit_verl.data.sources import same_source

    if manifest.get("builder") != DISTILL_BUILDER:
        return [f"it was built by prepare_data with prompt_mode={manifest.get('prompt_mode')}, but the config "
                "builds a mixed distillation dataset with distill_data"]
    settings = _distill_settings(cfg)
    mismatches = []
    if not same_source(manifest.get("pool"), settings["pool"]):
        mismatches.append(f"it was built from the pool {manifest.get('pool')}, the config names {settings['pool']}")
    wanted = {
        "gold": [distill_data.parse_gold_spec(spec).canonical() for spec in settings["gold"]],
        "val": [distill_data.parse_val_spec(spec).canonical() for spec in settings["val"]],
        "think_share": settings["think_share"],
        "seed": settings["seed"],
        "limit": settings["limit"],
    }
    for key, want in wanted.items():
        if manifest.get(key) != want:
            mismatches.append(f"it has {key}={manifest.get(key)}, the config has {want}")
    # A manifest without the field was written before the format rules had a version, under version 1.
    if manifest.get("format_version", 1) != distill_data.FORMAT_VERSION:
        mismatches.append(f"its prompt formats are version {manifest.get('format_version', 1)}, the data tool now "
                          f"writes version {distill_data.FORMAT_VERSION} (distill_data.FORMAT_VERSION)")
    return mismatches


def _prepare_data_mismatches(cfg: Any, manifest: Mapping[str, Any]) -> list[str]:
    """How a prepare_data dataset differs from the config's ``orbit.prepare.data`` block; empty when it matches."""
    from orbit_verl.data.sources import same_source

    mismatches = []
    data = _get(cfg, "orbit.prepare.data")
    # A source is a local file path, and same_source compares the resolved paths.
    if manifest.get("src") and not same_source(manifest["src"], str(data.src)):
        mismatches.append(f"it was built from {manifest['src']}, the config names {data.src}")
    val_src = data.get("val_src")
    if "val_src" in manifest and (manifest["val_src"] is None) != (val_src is None):
        mismatches.append(f"it has val_src={manifest['val_src']}, the config has {val_src}")
    elif manifest.get("val_src") and not same_source(manifest["val_src"], str(val_src)):
        mismatches.append(f"it has val_src={manifest['val_src']}, the config has {val_src}")
    # Settings that change which rows end up in the parquet files. A key that the manifest does
    # not record is not compared, and data_source is compared only when the config sets it.
    # prepare_data ignores val_size when a separate val_src is given and records None, so the
    # config's val_size only matters without val_src.
    wanted = {"prompt_mode": str(data.prompt_mode), "limit": data.get("limit"),
              "val_size": data.get("val_size") if val_src is None else None,
              "question_field": data.get("question_field"), "answer_field": data.get("answer_field")}
    if data.get("data_source") is not None:
        wanted["data_source"] = str(data.data_source)
    for key, want in wanted.items():
        if key in manifest and manifest[key] != want:
            mismatches.append(f"it has {key}={manifest[key]}, the config has {want}")
    return mismatches


def check_data(cfg: Any, *, model_ok: bool = True) -> list[Problem]:
    """Problems with ``orbit.data_dir``; an empty list means it is ready.

    The tokenizer comparison needs a ready model dir, so it is skipped when ``model_ok`` is false.
    """
    config_problem = _data_config_problem(cfg)
    if config_problem is not None:
        return [config_problem]
    data_dir = str(_get(cfg, "orbit.data_dir"))
    if not os.path.isdir(data_dir):
        return [Problem("data", f"Dataset dir {data_dir} does not exist.", (data_command(cfg),), True)]
    missing = [name for name in ("train.parquet", "val.parquet")
               if not os.path.isfile(os.path.join(data_dir, name))]
    try:
        manifest = common.read_manifest(os.path.join(data_dir, common.DATA_MANIFEST),
                                        common.DATA_MANIFEST_KIND)
    except FileNotFoundError:
        tool = "distill_data" if data_builder(cfg) == DISTILL_BUILDER else "prepare_data"
        return [Problem("data", f"{data_dir} exists but has no {common.DATA_MANIFEST}, so "
                        f"{tool} did not finish it. Remove or rename it by hand, then run:",
                        (data_command(cfg),))]
    except ValueError as exc:
        return [Problem("data", f"Dataset dir {data_dir} has an unusable manifest: {exc}",
                        (data_command(cfg, force=True),))]
    if missing:
        return [Problem("data", f"Dataset dir {data_dir} has a manifest but lacks {', '.join(missing)}.",
                        (data_command(cfg, force=True),))]

    mismatches = []
    want_len = int(_get(cfg, "data.max_prompt_length"))
    if manifest.get("max_prompt_length") != want_len:
        mismatches.append(f"it was filtered at max_prompt_length={manifest.get('max_prompt_length')}, "
                          f"the config has data.max_prompt_length={want_len}")
    if data_builder(cfg) == DISTILL_BUILDER:
        mismatches += _distill_mismatches(cfg, manifest)
    elif manifest.get("builder") == DISTILL_BUILDER:
        # A distillation dataset has no single src or prompt mode, so the comparisons below would
        # only list symptoms of this one difference.
        mismatches.append("it is a mixed distillation dataset built by distill_data, but the config builds "
                          "with prepare_data (set orbit.prepare.data.builder=distill to use it)")
    else:
        mismatches += _prepare_data_mismatches(cfg, manifest)
    if model_ok:
        model_dir = str(_get(cfg, "orbit.model_dir"))
        try:
            model_sha = common.tokenizer_sha256(model_dir)
        except FileNotFoundError as exc:
            return [Problem("model", f"Cannot fingerprint the tokenizer of {model_dir}: {exc}",
                            (model_command(cfg, force=True),))]
        if manifest.get("tokenizer_sha256") != model_sha:
            # Rebuilding in place would break the runs of the other model family that use this
            # dataset, so the message points at a separate name per tokenizer family.
            mismatches.append(
                f"its tokenizer differs from the one in {model_dir}. A dataset belongs to the tokenizer "
                "its length filter used, so each tokenizer family needs its own dataset name; the "
                "experiment configs end it with __${orbit.prepare.model.preset}")
    if mismatches:
        return [Problem("data", f"Dataset dir {data_dir} does not match the config: " + "; ".join(mismatches)
                        + ". Give the dataset a new orbit.prepare.data.name, or " + FORCE_PREPARE_HINT
                        + ". The command that rebuilds it:",
                        (data_command(cfg, force=True),))]
    return []


def check_prepared(cfg: Any) -> list[Problem]:
    """Problems with the prepared model dir and dataset, model first because data depends on it."""
    model = check_model(cfg)
    return model + check_data(cfg, model_ok=not model)


def export_family(export: str) -> str | None:
    """The tokenizer family (preset name, such as ``qwen3``) of an export, from its vocab size.

    None when the export's config.json cannot be read or its vocabulary matches no preset.
    """
    from orbit_verl.model import preset_for_vocab

    try:
        with open(os.path.join(export, "config.json"), encoding="utf-8") as handle:
            return preset_for_vocab(int(json.load(handle)["vocab_size"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def prepared_family(model_dir: str) -> str | None:
    """The preset that a prepared model dir's manifest records, or None without a usable manifest."""
    try:
        manifest = common.read_manifest(os.path.join(model_dir, common.MODEL_MANIFEST), common.MODEL_MANIFEST_KIND)
    except (OSError, ValueError):
        return None
    return str(manifest["preset"]) if manifest.get("preset") else None


def _own_manifest(cfg: Any) -> dict[str, Any]:
    """The manifest of the run's model dir, or ``{}`` when the dir does not describe this model.

    Some checks predict the run's model from the prepared dir before check_model has looked at
    it. A dir of the same name that was built from another export describes another model, so
    its manifest must not be read for this one. check_model reports that dir separately. A
    manifest without ``export_dir`` is accepted, as check_model accepts it.
    """
    if _is_missing(cfg, "orbit.prepare.model.export"):
        return {}
    try:
        manifest = common.read_manifest(os.path.join(str(_get(cfg, "orbit.model_dir")), common.MODEL_MANIFEST),
                                        common.MODEL_MANIFEST_KIND)
    except (OSError, ValueError):
        return {}
    recorded = manifest.get("export_dir")
    if recorded and not _same_path(recorded, _get(cfg, "orbit.prepare.model.export")):
        return {}
    return dict(manifest)


def model_lineage(cfg: Any) -> tuple[str | None, str]:
    """The lineage of the run's model, and where it was read from.

    A prepared dir records the lineage in its manifest; prepare_model detected it from the export's
    own model code and weights. Before the dir exists, the export's vocabulary size predicts it.
    ``(None, reason)`` means nothing says it yet.

    The manifest counts only when the dir was built from the config's export (``_own_manifest``).
    A dir of the same name that was built from another export would otherwise let a config pass
    for a model it does not fit, or refuse it for the wrong reason.
    """
    from orbit_verl.model import prepare_model as tool

    if _is_missing(cfg, "orbit.prepare.model.export"):
        return None, "no model is set"
    manifest = _own_manifest(cfg)
    if manifest.get("lineage"):
        return str(manifest["lineage"]), f"recorded in {_get(cfg, 'orbit.model_dir')}"
    export = str(_get(cfg, "orbit.prepare.model.export"))
    lineage = {"qwen3": tool.LINEAGE_RELEASE_QWEN3}.get(export_family(export))
    if lineage is None:
        return None, f"the vocabulary of {export} matches no known lineage"
    return lineage, f"predicted from the vocabulary size of {export}"


def check_lineage(cfg: Any) -> list[Problem]:
    """Refuse a model whose lineage the config says it does not fit.

    A config that only makes sense for some lineages lists them in ``orbit.fits.lineages`` and says
    why in ``orbit.fits.why``. The check runs before anything is prepared, so a mismatch does not
    first build a model dir and a dataset that the run cannot use.
    """
    wanted = _get(cfg, "orbit.fits.lineages")
    if wanted is None:
        return []
    wanted = [str(wanted)] if isinstance(wanted, str) else [str(item) for item in wanted]
    lineage, source = model_lineage(cfg)
    if lineage is None or lineage in wanted:
        return []
    why = _get(cfg, "orbit.fits.why")
    return [Problem(
        "lineage", f"This config fits only {' or '.join(wanted)} models (orbit.fits.lineages), but the "
        f"model is {lineage} ({source})." + (f" {str(why).strip()}" if why else ""))]


def check_fits(cfg: Any) -> list[Problem]:
    """Problems from the config's ``orbit.fits`` block: a model of a lineage the config does not fit."""
    return check_lineage(cfg)


def check_settings(cfg: Any, environ: Mapping[str, str] | None = None) -> list[Problem]:
    """Config combinations that cannot work, found before any process starts."""
    environ = os.environ if environ is None else environ
    problems = []
    loss_mode = str(_get(cfg, "actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla"))
    if loss_mode.startswith("rltt_") and not bool(_get(cfg, "orbit.plugin", True)):
        problems.append(Problem(
            "plugin", f"loss_mode={loss_mode} needs the OrbitLM engine, but orbit.plugin is false. "
            "Without the plugin verl would train plain final-loop GRPO under an RLTT name.",
            ("orbit.plugin=true",)))
    try:
        check_threshold(_get(cfg, THRESHOLD_KEY))
    except ValueError as exc:
        # The trainer builds the alarm only after Ray and the models are up, so a bad value would
        # otherwise surface minutes into the launch.
        problems.append(Problem("config", str(exc), (f"{THRESHOLD_KEY}=0.3",)))
    loggers = _get(cfg, "trainer.logger", [])
    loggers = [loggers] if isinstance(loggers, str) else list(loggers)
    if "clearml" in loggers and not environ.get("CLEARML_API_ACCESS_KEY"):
        problems.append(Problem(
            "clearml", "trainer.logger has clearml, but CLEARML_API_ACCESS_KEY is not set. Point "
            "orbit.clearml_env at a dotenv file with the CLEARML_* credentials, or drop clearml "
            "from trainer.logger.", ("orbit.clearml_env=/path/to/clearml.env",)))
    override = _get(cfg, "actor_rollout_ref.model.override_config", {})
    for key, why in _PADDING_OVERRIDES.items():
        if key in override:
            problems.append(Problem(
                "config", f"actor_rollout_ref.model.override_config.{key} is set: {why}.",
                (f"~actor_rollout_ref.model.override_config.{key}",)))
    if "model_config" in override:
        # verl's HFModelConfig applies the nested block in place of the keys next to it. That is
        # why the whole block is refused: checking its keys would still drop the sibling keys.
        problems.append(Problem(
            "config", "actor_rollout_ref.model.override_config.model_config is set. When this block "
            "exists, verl applies it instead of the keys next to it. Its pad_token_id or "
            "embedding_padding_idx would then get past the check above, and the keys that this config "
            "sets directly under override_config, such as early_exit_threshold, would be ignored. Put "
            "the keys directly under override_config instead.",
            ("~actor_rollout_ref.model.override_config.model_config",)))
    problems += _model_path_problems(cfg)
    return problems + check_lengths(cfg)


# verl's model keys that name a directory to load from, with what verl loads from each. Both fall
# back to actor_rollout_ref.model.path when they are null.
_MODEL_SIDE_PATHS = {
    "hf_config_path": "config and model code",
    "tokenizer_path": "tokenizer",
}


def _model_path_problems(cfg: Any) -> list[Problem]:
    """Refuse a verl model path that points somewhere other than ``orbit.model_dir``.

    The checks above look at ``orbit.model_dir``: its manifest, its recorded frozen embedding row,
    its copy of the model code and its tokenizer. verl never reads that key. It loads the weights
    from ``actor_rollout_ref.model.path``, the config and model code from ``hf_config_path``, and
    the tokenizer from ``tokenizer_path``. The base config ties all three to ``orbit.model_dir``.
    An override that points one of them elsewhere would train a model that no check has seen, for
    example an old prepared dir that freezes the eos row of a release model.
    """
    model_dir = str(_get(cfg, "orbit.model_dir"))
    problems = []
    path = _get(cfg, "actor_rollout_ref.model.path")
    if path is None or not _same_path(path, model_dir):
        problems.append(Problem(
            "config", f"actor_rollout_ref.model.path is {path}, but orbit.model_dir is {model_dir}. "
            "verl would train the model in the first dir, while the checks looked at the second. To "
            "train another model, change orbit.prepare.model.export and orbit.prepare.model.name "
            "instead.",
            (f"actor_rollout_ref.model.path={model_dir}",)))
    for key, what in _MODEL_SIDE_PATHS.items():
        value = _get(cfg, f"actor_rollout_ref.model.{key}")
        if value is not None and not _same_path(value, model_dir):
            problems.append(Problem(
                "config", f"actor_rollout_ref.model.{key} is {value}. It must be null or orbit.model_dir "
                f"({model_dir}), because verl takes the model's {what} from it and the checks looked "
                "only at orbit.model_dir.",
                (f"actor_rollout_ref.model.{key}=null",)))
    return problems


def _context_length(cfg: Any) -> tuple[int | None, str]:
    """The max_position_embeddings the run's model dir has or will have, and where it comes from.

    The config's own ``orbit.prepare.model.max_position_embeddings`` wins, because check_model
    refuses a dir that differs from it. Without it, the prepared dir's manifest is used when the
    dir was built from this export, and otherwise the export's config.json. None means that nothing
    readable says it.
    """
    wanted = _get(cfg, "orbit.prepare.model.max_position_embeddings")
    if wanted is not None:
        return int(wanted), "orbit.prepare.model.max_position_embeddings"
    manifest = _own_manifest(cfg)
    if manifest.get("max_position_embeddings") is not None:
        return int(manifest["max_position_embeddings"]), f"the manifest of {_get(cfg, 'orbit.model_dir')}"
    if _is_missing(cfg, "orbit.prepare.model.export"):
        return None, "unknown"
    path = os.path.join(str(_get(cfg, "orbit.prepare.model.export")), "config.json")
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle).get("max_position_embeddings")
    except (OSError, ValueError, AttributeError):
        return None, "unknown"
    return (int(value), path) if value is not None else (None, "unknown")


def check_lengths(cfg: Any) -> list[Problem]:
    """Refuse prompt and response lengths that verl would cut silently or reject deep in a worker.

    A whole sequence is data.max_prompt_length + data.max_response_length tokens. It has to fit:

    - into the rollout context, which is rollout.max_model_len or else the model dir's
      max_position_embeddings. verl gives vLLM ``max_tokens = context - prompt length``, so a
      response limit beyond the context is silently never reached;
    - into every dynamic-batch token budget (``*_max_token_len_per_gpu``), because verl asserts that
      one micro-batch holds at least one whole sequence. That assert fires in a Ray worker after
      the rollout of the first step.

    verl itself refuses a rollout.max_model_len above max_position_embeddings at start-up; that is
    checked here too, so the error comes with the fix.
    """
    prompt = int(_get(cfg, "data.max_prompt_length"))
    total = prompt + int(_get(cfg, "data.max_response_length"))
    sum_text = f"data.max_prompt_length + data.max_response_length = {total}"
    problems = []
    positions, source = _context_length(cfg)
    max_model_len = _get(cfg, "actor_rollout_ref.rollout.max_model_len")
    if max_model_len is not None and positions is not None and int(max_model_len) > positions:
        problems.append(Problem(
            "lengths", f"actor_rollout_ref.rollout.max_model_len={max_model_len} is above the model's "
            f"max_position_embeddings {positions} ({source}); verl refuses that.",
            ("actor_rollout_ref.rollout.max_model_len=null",)))
    context = int(max_model_len) if max_model_len is not None else positions
    if context is not None and total > context:
        where = "actor_rollout_ref.rollout.max_model_len" if max_model_len is not None else source
        problems.append(Problem(
            "lengths", f"{sum_text} does not fit into the rollout context of {context} tokens ({where}). "
            "verl would shorten every long response without saying so. Lower the response length, or, "
            "for a model trained at a longer context, prepare it with a larger max_position_embeddings:",
            (f"data.max_response_length={max(1, context - prompt)}",
             "orbit.prepare.model.max_position_embeddings=<the trained context length>")))
    budgets = (
        ("actor_rollout_ref.actor.ppo_max_token_len_per_gpu", "actor_rollout_ref.actor.use_dynamic_bsz"),
        ("actor_rollout_ref.ref.log_prob_max_token_len_per_gpu", "actor_rollout_ref.ref.log_prob_use_dynamic_bsz"),
        ("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
         "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz"),
    )
    for key, switch in budgets:
        value = _get(cfg, key)
        if value is not None and bool(_get(cfg, switch, False)) and int(value) < total:
            problems.append(Problem(
                "lengths", f"{key}={value} is below {sum_text}. verl needs one whole sequence to fit "
                "into a micro-batch and would stop with an assertion after the first rollout.",
                (f"{key}={total}",)))
    return problems


def check_plugin_import() -> list[Problem]:
    """Import the verl plugin in this process, so a broken plugin fails before Ray starts.

    In the Ray workers verl imports the plugin by itself (VERL_USE_EXTERNAL_MODULES). A failure
    there shows up only deep in a worker log, which is why the driver tries the import first.
    """
    import importlib
    import traceback

    try:
        importlib.import_module("orbit_verl.plugin")
    except Exception as exc:  # noqa: BLE001 - any import failure means the run cannot use the plugin
        last = traceback.format_exception_only(type(exc), exc)[-1].strip()
        return [Problem("plugin", f"orbit_verl.plugin does not import: {last}")]
    return []


def has_code_rows(cfg: Any) -> bool:
    """Whether the run's train or validation parquet holds rows that the code reward scores.

    Only the ``data_source`` column is read. A file that does not exist or cannot be read is skipped.
    The dataset check has already reported a missing file, and verl fails at start-up on a file
    that it cannot read, so neither case can lose code rows quietly.
    """
    import pyarrow.parquet as pq

    from orbit_verl.data.prompts import reward_of

    for key in ("data.train_files", "data.val_files"):
        for path in _spec_list(_get(cfg, key)):
            try:
                column = pq.read_table(path, columns=["data_source"]).column("data_source")
            except (OSError, ValueError):  # pyarrow's read errors are OSError or ArrowInvalid, a ValueError
                continue
            if any(reward_of(str(source)) == "code" for source in column.unique().to_pylist()):
                return True
    return False


def check_code_sandbox(cfg: Any) -> list[Problem]:
    """Run the code sandbox's self-check in this process when the run has code rows.

    Each reward worker runs the same check before its first code reward and raises when it fails.
    verl's V1 trainer catches that error and leaves the prompt's group out of the batch, so a run on
    a machine where the sandbox does not hold would go on without its code rows instead of stopping.
    This check stops such a launch before Ray starts. Its result describes the machine that runs it,
    so the launcher calls it only where training runs, never in a preparation, which may run on
    another machine. The reward workers of a one-node run share this machine; a multi-node run
    checks only its head node.
    """
    if not has_code_rows(cfg):
        return []
    from orbit_verl.data.reward import check_sandbox

    try:
        check_sandbox()
    except Exception as exc:  # noqa: BLE001 - a check that cannot even run proves nothing about the sandbox
        return [Problem("sandbox", f"{type(exc).__name__}: {exc} The launch stops here, because verl would drop "
                        "every code prompt during training and go on with the other rows.")]
    return []


# Settings whose change makes an existing checkpoint belong to a different experiment. Besides the
# model, the data and the loss, they hold the recipe keys that two runs on the same dataset may differ
# in: the learning rate, the KL, the loss aggregation, the rollout correction, the KV cache type, the
# batch, the response length and the reward of a cut response. A key with sub-keys is compared whole.
_RESUME_KEYS = ("orbit.model_dir", "orbit.data_dir", "orbit.prompt_mode",
                "actor_rollout_ref.actor.policy_loss.loss_mode", "algorithm.adv_estimator",
                "actor_rollout_ref.actor.optim.lr", "actor_rollout_ref.actor.kl_loss_coef",
                "actor_rollout_ref.actor.loss_agg_mode", "algorithm.rollout_correction",
                "actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype", "data.train_batch_size",
                "data.max_response_length", "reward.custom_reward_function.reward_kwargs.truncated_score")

# What OmegaConf.select returns for a key that the saved config does not have at all.
_ABSENT = object()


def _plain(value: Any) -> Any:
    """A config value as plain Python data, with sub-configs turned into dicts and lists."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _comparable(value: Any) -> str:
    """Text that is the same for two equal settings.

    Every number becomes a float, so the 0 of a command line equals the 0.0 of a YAML file. Booleans
    stay booleans, although Python counts them as numbers.
    """
    def canonical(item: Any) -> Any:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item)
        if isinstance(item, dict):
            return {str(key): canonical(sub) for key, sub in item.items()}
        if isinstance(item, (list, tuple)):
            return [canonical(sub) for sub in item]
        return item

    return json.dumps(canonical(_plain(value)), sort_keys=True, default=str)


def check_resume(cfg: Any) -> list[Problem]:
    """Refuse to resume a checkpoint that was trained with different key settings.

    With ``trainer.resume_mode=auto`` verl continues from the newest checkpoint in the run dir, with
    no questions asked. A second experiment that reuses a run name, for example GRPO after RLTT,
    would silently continue the first one. The resolved config of the earlier launch is saved in
    ``run_dir/config.yaml``, so the two can be compared before training.

    A key that the saved config lacks is not compared, because a launcher without that key in
    ``_RESUME_KEYS`` may have written the file. A key that it has with the value null is compared
    like any other value.
    """
    from omegaconf import OmegaConf

    if str(_get(cfg, "trainer.resume_mode", "auto")) == "disable":
        return []
    run_dir = str(_get(cfg, "orbit.run_dir"))
    ckpt_dir = str(_get(cfg, "trainer.default_local_dir"))
    saved = os.path.join(run_dir, "config.yaml")
    if not (os.path.isfile(os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt"))
            and os.path.isfile(saved)):
        return []
    try:
        old = OmegaConf.load(saved)
    except Exception as exc:  # noqa: BLE001 - an unreadable file is reported, not raised
        return [Problem("resume", f"Cannot read {saved} to compare it with this run: {exc}")]
    changed = []
    for key in _RESUME_KEYS:
        before = OmegaConf.select(old, key, default=_ABSENT)
        if before is _ABSENT:
            continue
        now = OmegaConf.select(cfg, key, default=None)
        if _comparable(before) != _comparable(now):
            changed.append(f"{key}: {_plain(before)} -> {_plain(now)}")
    if changed:
        return [Problem("resume", f"{run_dir} holds checkpoints of a run with other settings ("
                        + "; ".join(changed) + "). Choose a new orbit.run_name for this experiment.",
                        (f"orbit.run_name={_get(cfg, 'orbit.run_name')}_2",))]
    return []


# The file in the checkpoint dir that names the newest checkpoint, as verl's trainer writes it.
CHECKPOINT_TRACKER = "latest_checkpointed_iteration.txt"
# verl logs every validation metric under a key with this prefix: val-core/<data source>/<name>.
VAL_METRIC_PREFIX = "val-core/"


def resumed_step(cfg: Any) -> int | None:
    """The global step whose checkpoint verl will load at start-up, or None for a run from scratch.

    This follows verl's own ``_load_checkpoint``. With ``trainer.resume_mode`` auto, the step is read
    from the tracker file in the checkpoint dir, and its ``global_step_N`` dir has to exist. With
    resume_path, the step is the number after ``global_step_`` in ``trainer.resume_from_path``.
    """
    mode = str(_get(cfg, "trainer.resume_mode", "auto"))
    if mode == "resume_path":
        path = str(_get(cfg, "trainer.resume_from_path", ""))
        tail = path.split("global_step_")[-1]
        return int(tail) if "global_step_" in path and tail.isdigit() else None
    if mode != "auto":
        return None
    ckpt_dir = str(_get(cfg, "trainer.default_local_dir"))
    try:
        with open(os.path.join(ckpt_dir, CHECKPOINT_TRACKER), encoding="utf-8") as handle:
            step = int(handle.read().strip())
    except (OSError, ValueError):
        return None
    return step if os.path.isdir(os.path.join(ckpt_dir, f"global_step_{step}")) else None


def validated_steps(run_dir: str) -> set[int]:
    """The steps whose validation metrics any launch of the run has logged.

    verl's file logger writes one JSON line per step, ``{"step": N, "data": {...}}``, to
    metrics.jsonl. The launcher renames the file of an earlier launch to metrics.<time>.jsonl before
    verl truncates it, so every one of these files is read. A line that does not parse is skipped,
    such as the last line of a launch that was killed while writing it.
    """
    import glob

    steps: set[int] = set()
    for path in glob.glob(os.path.join(glob.escape(run_dir), "metrics*.jsonl")):
        try:
            with open(path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            step = record.get("step") if isinstance(record, dict) else None
            data = record.get("data") if isinstance(record, dict) else None
            if isinstance(step, int) and isinstance(data, dict) and any(
                    str(key).startswith(VAL_METRIC_PREFIX) for key in data):
                steps.add(step)
    return steps


def resume_validation(cfg: Any) -> str | None:
    """Skip the validation before training when it would repeat one that is already logged.

    verl validates before its first step whenever ``trainer.val_before_train`` is true, also when it
    resumes from a checkpoint. On a resumed job that repeats a validation of about 30 minutes.

    verl saves the checkpoint of a step before it validates that step, though. A job that was stopped
    during the validation has left a checkpoint whose validation was never logged, and skipping the
    validation on resume would lose that evaluation for good. So the validation is turned off only
    when the metrics files already hold validation metrics of the step that the run resumes from. In
    every other case it runs, and verl logs it under that step.

    The decision is written into ``cfg``, so the run's saved config.yaml shows it. Returns a sentence
    for the log, or None when the run starts from scratch or the config never validates first.
    """
    if not bool(_get(cfg, "trainer.val_before_train", True)):
        return None
    step = resumed_step(cfg)
    if step is None:
        return None
    if step in validated_steps(str(_get(cfg, "orbit.run_dir"))):
        cfg.trainer.val_before_train = False
        return (f"resuming from global_step_{step}, whose validation is already logged: "
                "trainer.val_before_train=false")
    return (f"resuming from global_step_{step}, whose validation is not logged: it is validated "
            "before the first step")


def _check_command(cfg: Any, python: str | None = None) -> str:
    """The prepare_model command that re-runs the check of an existing model dir."""
    return _tool("orbit_verl.model.prepare_model", ["--out-dir", str(_get(cfg, "orbit.model_dir")), "--check"],
                 cfg, python)


def _build_model(cfg: Any, force: bool, log: Callable[[str], None]) -> list[Problem]:
    """Run prepare_model for the config, then prepare_model's own check of the new dir.

    A refusal of the tool (an existing foreign dir, a missing export, a padding row it cannot
    decide) becomes a Problem with the command to run by hand, instead of a traceback.
    """
    from orbit_verl.model import prepare_model as tool

    model = _get(cfg, "orbit.prepare.model")
    model_dir = str(_get(cfg, "orbit.model_dir"))
    log(f"preparing model dir {model_dir} from {model.export}")
    try:
        tool.prepare_model_dir(
            str(model.export), model_dir,
            preset=str(model.get("preset") or "auto"),
            max_position_embeddings=model.get("max_position_embeddings"),
            materialize_weights=bool(model.get("materialize_weights", False)),
            force=force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        command = model_command(cfg, force=force)
        # prepare_model refuses when the export's training code and its weights disagree about
        # the frozen embedding row. Only a person can decide that case, with this flag.
        if isinstance(exc, tool.PaddingDetectionError):
            command += " --embedding-padding-idx <N|none>"
        return [Problem("model", f"prepare_model refused to build {model_dir}: {exc}", (command,))]
    # The tool's own check loads the new dir the way verl does: config class, tokenizer, stop
    # tokens and the recorded padding row. vLLM's view is skipped, because importing vLLM here
    # would cost more than the rest of the preparation.
    try:
        tool.check_model_dir(model_dir, load_weights=False, vllm_view=False)
    except Exception as exc:  # noqa: BLE001 - any failure of the check means the dir is not usable
        return [Problem("model", f"The new model dir failed prepare_model's check: {exc}",
                        (_check_command(cfg),))]
    return []


def run_prepare(cfg: Any, force: bool = False, log: Callable[[str], None] = print) -> list[Problem]:
    """Build the model dir and the dataset that the config needs, then return what is still wrong.

    Without ``force`` only missing or unfinished directories are built. A directory that exists but
    disagrees with the config is left alone and reported, because rebuilding it deletes the old one.
    With ``force`` (the launcher's ``--force-prepare``) such a directory is rebuilt as well. A
    directory that already matches the config is never rebuilt.
    """
    common.apply_cache_env(str(_get(cfg, "orbit.runs_root")))

    model_problems = check_model(cfg)
    if _is_missing(cfg, "orbit.prepare.model.export"):
        return model_problems + check_data(cfg, model_ok=False)
    if any(p.buildable for p in model_problems) or (force and model_problems):
        failed = _build_model(cfg, force, log)
        if failed:
            return failed

    if check_model(cfg):
        return check_prepared(cfg)

    data_problems = check_data(cfg)
    if any(p.buildable for p in data_problems) or (force and data_problems):
        failed = _build_data(cfg, force, log)
        if failed:
            return failed
    return check_prepared(cfg)


def _build_data(cfg: Any, force: bool, log: Callable[[str], None]) -> list[Problem]:
    """Run the config's dataset builder, prepare_data or distill_data, for ``orbit.data_dir``.

    A refusal of the tool becomes a Problem with the command to run by hand, instead of a traceback.
    A ``orbit.prepare.data`` block that no tool can build from is reported and nothing is built.
    That matters with ``force``, which builds whenever any data problem exists.
    """
    config_problem = _data_config_problem(cfg)
    if config_problem is not None:
        return [config_problem]
    data_dir = str(_get(cfg, "orbit.data_dir"))
    runs_root = str(_get(cfg, "orbit.runs_root"))
    if data_builder(cfg) == DISTILL_BUILDER:
        from orbit_verl.data import distill_data

        settings = _distill_settings(cfg)
        log(f"preparing distillation dataset {data_dir} from the pool {settings['pool']}, "
            f"{len(settings['gold'])} gold and {len(settings['val'])} validation inputs")
        try:
            distill_data.convert(
                settings["pool"], data_dir,
                tokenizer_dir=str(_get(cfg, "orbit.model_dir")),
                max_prompt_length=int(_get(cfg, "data.max_prompt_length")),
                val=settings["val"],
                gold=settings["gold"],
                think_share=settings["think_share"],
                seed=settings["seed"],
                limit=settings["limit"],
                force=force,
                runs_root=runs_root,
            )
        except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
            return [Problem("data", f"distill_data refused to build {data_dir}: {exc}",
                            (data_command(cfg, force=force),))]
        return []

    from orbit_verl.data.prepare_data import convert

    data = _get(cfg, "orbit.prepare.data")
    log(f"preparing dataset {data_dir} from {data.src}")
    try:
        convert(
            str(data.src), data_dir,
            prompt_mode=str(data.prompt_mode),
            tokenizer_dir=str(_get(cfg, "orbit.model_dir")),
            max_prompt_length=int(_get(cfg, "data.max_prompt_length")),
            data_source=data.get("data_source"),
            question_field=str(data.get("question_field") or "instruction"),
            answer_field=str(data.get("answer_field") or "response"),
            val_size=int(data.get("val_size", 64)),
            val_src=None if data.get("val_src") is None else str(data.val_src),
            limit=None if data.get("limit") is None else int(data.limit),
            force=force,
            runs_root=runs_root,
        )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        return [Problem("data", f"prepare_data refused to build {data_dir}: {exc}",
                        (data_command(cfg, force=force),))]
    return []
