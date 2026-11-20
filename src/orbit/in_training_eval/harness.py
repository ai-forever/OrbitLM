"""The pinned lm-eval release: its version, its install command, and the checks of an install folder."""
from __future__ import annotations

import base64
import csv
import hashlib
from pathlib import Path
from typing import Any, Optional

# The public lm-eval release that the evaluation runs on (PyPI lm_eval). The pin is exact because
# the local task files, the CRUXEval scorer and run_lm_eval.py were checked against this release,
# and lm-eval changes prompts, scoring and its vLLM backend between releases. Moving to another
# release means re-checking those, then changing this string. The startup check compares it with
# the version of the lm_eval that the evaluation processes import.
LM_EVAL_VERSION = "0.4.13"
# RULER's haystack generator in lm-eval imports wonderwords, which only lm-eval's ruler extra
# requires, so the evaluation environment may lack it. The install command puts this release into
# the same folder as lm-eval.
WONDERWORDS_VERSION = "3.0.1"


def lm_eval_install_command(folder: Any, python: Any = "python") -> str:
    """The command that installs the pinned lm-eval into a folder for eval_harness_path.

    The folder must be new and empty. pip install --target does not replace a package folder that
    already exists in the target, yet it still adds the new release's metadata next to it. Run
    over a source checkout or an older install, the command would therefore leave the old code
    under the pinned release's version. verify_lm_eval_files refuses such a folder.
    """
    return (
        f"{python} -m pip install --no-deps --no-compile --target {folder} "
        f"lm_eval=={LM_EVAL_VERSION} wonderwords=={WONDERWORDS_VERSION}"
    )


def lm_eval_install_hint(python: Any) -> str:
    """The sentence that tells how to make a valid eval_harness_path, for the error messages."""
    return (
        f"Install it into a new, empty folder with '{lm_eval_install_command('<new folder>', python)}' "
        "and set eval_harness_path to that folder."
    )


def lm_eval_metadata_problem(folder: Path) -> Optional[str]:
    """Why folder does not look like a pip install of the pinned lm-eval, or None when it does.

    lm-eval reads its own version from the metadata folder that pip writes next to the package.
    So the folder must hold exactly one such metadata folder, the pinned release's, with its RECORD
    file. A bare source checkout has none. With two of them, the reported version depends on which
    one Python finds first. This function only looks at names; verify_lm_eval_files compares the
    files themselves.
    """
    if not (folder / "lm_eval").is_dir():
        return f"{folder} has no lm_eval package folder"
    found = sorted(path.name for path in folder.glob("lm_eval-*.dist-info"))
    if found != [f"lm_eval-{LM_EVAL_VERSION}.dist-info"]:
        return (
            f"{folder} does not hold the metadata of lm-eval {LM_EVAL_VERSION} alone "
            f"(found: {found or 'no lm_eval metadata'})"
        )
    if not (folder / found[0] / "RECORD").is_file():
        return f"{folder / found[0]} has no RECORD file"
    return None


def verify_lm_eval_files(folder: Path) -> int:
    """Checks that folder/lm_eval holds exactly the files of the pinned release, and returns their number.

    The version that lm-eval reports comes from its metadata alone, so it says nothing about the
    code. pip writes the sha256 of every installed file into the RECORD file of that metadata.
    Here every file of the package must still have its recorded hash, and the package must hold
    no other files besides Python's bytecode caches. That catches a source checkout which pip only
    labelled with the pinned version (see lm_eval_install_command), and files edited or added in
    place. For 0.4.13 it reads about 15,600 files (12 MB), which took 9 to 21 s on a shared
    filesystem, so it runs once per node in the preflight and in prefetch, not on every rank. Every
    problem raises ValueError with the reason.
    """
    problem = lm_eval_metadata_problem(folder)
    if problem:
        raise ValueError(problem)
    record = folder / f"lm_eval-{LM_EVAL_VERSION}.dist-info" / "RECORD"
    expected: dict[str, str] = {}
    for row in csv.reader(record.read_text(encoding="utf-8").splitlines()):
        # Bytecode files are left out. pip records the ones it compiles, but Python rewrites them
        # whenever a source file looks newer, for example after a copy that did not keep the
        # modification times. Python checks a bytecode file against its source before using it.
        if len(row) >= 2 and row[0].startswith("lm_eval/") and row[1] and "__pycache__" not in row[0].split("/"):
            expected[row[0]] = row[1]
    if not expected:
        raise ValueError(f"{record} lists no files of the lm_eval package")
    changed = []
    for relative, recorded in expected.items():
        algorithm, _, digest = recorded.partition("=")
        try:
            data = (folder / relative).read_bytes()
        except FileNotFoundError:
            changed.append(f"{relative} (missing)")
            continue
        try:
            actual = base64.urlsafe_b64encode(hashlib.new(algorithm, data).digest()).rstrip(b"=").decode("ascii")
        except ValueError:
            raise ValueError(f"{record} uses the unknown hash {algorithm!r}") from None
        if actual != digest:
            changed.append(relative)
    added = []
    for path in sorted((folder / "lm_eval").rglob("*")):
        relative = path.relative_to(folder).as_posix()
        if path.is_file() and relative not in expected and "__pycache__" not in relative.split("/"):
            added.append(relative)
    if changed or added:
        details = []
        if changed:
            details.append(f"{len(changed)} files differ from the release, for example {', '.join(changed[:3])}")
        if added:
            details.append(f"{len(added)} files are not part of the release, for example {', '.join(added[:3])}")
        raise ValueError(
            f"{folder / 'lm_eval'} is not lm-eval {LM_EVAL_VERSION} as pip installed it: " + "; ".join(details)
        )
    return len(expected)


def harness_folder_problem(folder: Path, python: Any = "python") -> Optional[str]:
    """Why a folder cannot be eval_harness_path, or None when it can.

    The folder must hold the lm_eval package next to the metadata of the pinned release alone,
    which is what pip install --target leaves in a new folder. Every rank runs this check at
    startup, so it only looks at folder names. The preflight then compares the files themselves,
    once per node (lm_eval_problems).
    """
    problem = lm_eval_metadata_problem(folder)
    if problem is None:
        return None
    return f"eval_harness_path: {problem}. {lm_eval_install_hint(python)}"
