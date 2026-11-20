"""The draft model dir for self-speculative decoding: the same checkpoint, run with one loop.

    python inference/make_draft.py MODEL_DIR DRAFT_DIR

A OrbitLM checkpoint run with one loop is a cheap draft for the same checkpoint run with all its loops,
because both share every weight. vLLM's draft_model method needs the draft as a model dir of its own,
and it does not apply the target's hf_overrides to the draft. So the draft dir holds a config.json that
says total_ut_steps=1 and, for every other file of the model dir, a symlink to that file. The weights
are never copied.

generate_vllm.py --speculative calls make_draft itself. This command prepares the dir once in a place
of your choice, and generate_vllm.py --speculative --draft-dir DRAFT_DIR then reuses it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# The draft always runs one loop. The speed and quality numbers in README.md are for this draft.
DRAFT_LOOPS = 1
CONFIG_NAME = "config.json"


class DraftDirError(ValueError):
    """The draft dir cannot be created or reused."""


def read_config(model_dir: Path) -> dict[str, Any]:
    path = Path(model_dir) / CONFIG_NAME
    if not path.is_file():
        raise DraftDirError(f"{model_dir} has no {CONFIG_NAME}, so it is not an HF export")
    return json.loads(path.read_text(encoding="utf-8"))


def draft_config(config: dict[str, Any]) -> dict[str, Any]:
    """The model's config with total_ut_steps set to the draft's one loop and nothing else changed."""
    return {**config, "total_ut_steps": DRAFT_LOOPS}


def default_draft_dir(model_dir: Path) -> Path:
    """A draft dir under the system temp dir, named after the model dir, so that a second run reuses it.

    Creating a draft dir takes a moment, because it holds only links. A temp dir therefore costs nothing
    when it is cleaned up, and it keeps the draft out of model dirs that are often shared or read-only.
    """
    digest = hashlib.sha256(str(Path(model_dir).resolve()).encode("utf-8")).hexdigest()[:12]
    user = os.getuid() if hasattr(os, "getuid") else "user"
    return Path(tempfile.gettempdir()) / f"orbit_drafts_{user}" / f"{digest}_loops{DRAFT_LOOPS}"


def make_draft(model_dir: Path, draft_dir: Path) -> Path:
    """Create the draft dir, or check and reuse one that a previous run created. Returns draft_dir.

    A dir that holds anything else (a copied file, a link to another model's file, another config) is
    refused rather than overwritten, because it may belong to someone else.
    """
    model_dir = Path(model_dir).resolve()
    draft_dir = Path(draft_dir)
    config = read_config(model_dir)
    total = int(config.get("total_ut_steps") or 1)
    if total <= DRAFT_LOOPS:
        raise DraftDirError(
            f"the draft runs {DRAFT_LOOPS} loop and must run fewer loops than the model, "
            f"but {model_dir} has total_ut_steps={total}"
        )
    resolved = draft_dir.resolve()
    if resolved == model_dir or model_dir in resolved.parents:
        raise DraftDirError("the draft dir must not be the model dir or lie inside it")
    wanted_config = draft_config(config)
    if draft_dir.exists() and not draft_dir.is_dir():
        raise DraftDirError(f"{draft_dir} exists and is not a directory")
    draft_dir.mkdir(parents=True, exist_ok=True)

    sources = {entry.name: entry for entry in model_dir.iterdir() if entry.name != CONFIG_NAME}
    for name in os.listdir(draft_dir):
        if name not in (CONFIG_NAME, CONFIG_NAME + ".partial") and name not in sources:
            raise DraftDirError(f"{draft_dir} holds {name}, which the model dir {model_dir} does not have")
    for name, source in sorted(sources.items()):
        link = draft_dir / name
        # The link points at the real file, so that a chain of links in the model dir does not matter.
        target = source.resolve()
        if link.is_symlink():
            if link.resolve() != target:
                raise DraftDirError(f"{link} points to {link.resolve()}, not to the model's {target}")
            continue
        if link.exists():
            raise DraftDirError(f"{link} is a copy, not a link; a draft dir links to the model's files")
        link.symlink_to(target)

    config_path = draft_dir / CONFIG_NAME
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != wanted_config:
            raise DraftDirError(f"{config_path} is not the model's config with total_ut_steps={DRAFT_LOOPS}")
    else:
        # The config is written last and renamed into place, so a dir without it is an unfinished one
        # that the next call completes.
        partial = draft_dir / (CONFIG_NAME + ".partial")
        partial.write_text(json.dumps(wanted_config, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, config_path)
    return draft_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="HF export dir of the checkpoint")
    parser.add_argument("draft_dir", type=Path, help="where the draft dir goes; an existing draft dir of the same model is reused")
    args = parser.parse_args(argv)
    try:
        path = make_draft(args.model_dir, args.draft_dir)
    except DraftDirError as exc:
        parser.error(str(exc))
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
