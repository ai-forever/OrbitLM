"""Checks that the dense path (document masking off) matches a reference trainer.

The reference is another copy of the package, normally a checkout of the commit
before the change under test:

    git worktree add ../orbit_reference <commit>
    python tests/train/check_dense_unchanged.py \\
        --reference-package ../orbit_reference/src/orbit \\
        --hf-dir <hf export> --streams-root <streams root>

Both packages are imported side by side: this checkout under its own name and the
reference under the alias orbit_reference. The same weights and one real 16K
sample with 28-36 EOS go through both models, and the per-step hidden states and
the final logits are compared. The check passes when every difference is exactly
zero, because both models then run the same computation on the same inputs.

It needs the same GPU, FA3 kernel and data as groups a-d of
test_document_masking.py, and it is skipped with a message on a machine without
a suitable GPU. The exit status is 0 on a pass or a skip and 1 on a difference.
On a Hopper GPU, a FA3 kernel that cannot be loaded stops the script with the
loader's error, which also gives status 1.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_document_masking as t  # noqa: E402  (also puts this checkout's src/ on sys.path)

REFERENCE_ALIAS = "orbit_reference"


def import_reference_package(package_dir: Path):
    """Imports the package in package_dir under REFERENCE_ALIAS.

    Returns the reference's modelling_ouro and configuration_ouro modules.
    Importing the package under another name keeps it apart from this
    checkout's orbit, which is already imported.
    """
    package_dir = package_dir.expanduser().resolve()
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        raise FileNotFoundError(f"{package_dir} is not a package directory (no __init__.py)")
    spec = importlib.util.spec_from_file_location(
        REFERENCE_ALIAS, init_file, submodule_search_locations=[str(package_dir)]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[REFERENCE_ALIAS] = package
    spec.loader.exec_module(package)
    return (
        importlib.import_module(f"{REFERENCE_ALIAS}.modelling_ouro"),
        importlib.import_module(f"{REFERENCE_ALIAS}.configuration_ouro"),
    )


def load_reference_model(package_dir: Path, hf_dir: Path, kernel_path, rope_theta: float, device):
    modelling, configuration = import_reference_package(package_dir)
    config = configuration.OuroConfig.from_pretrained(str(hf_dir))
    config.rope_theta = float(rope_theta)
    config.use_cache = False
    model = modelling.OuroForCausalLM.from_pretrained(
        str(hf_dir), config=config, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.config._attn_implementation = t.FA3_IMPL
    model.config._attn_kernel_repo_path = kernel_path
    return model.to(device).eval()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--reference-package",
        type=Path,
        required=True,
        help="src/orbit directory of the reference checkout.",
    )
    p.add_argument(
        "--hf-dir",
        type=Path,
        default=t.env_path("ORBIT_TEST_HF_DIR"),
        help="HF export (save_pretrained directory) of the model. Default: $ORBIT_TEST_HF_DIR.",
    )
    p.add_argument(
        "--streams-root",
        type=Path,
        default=t.env_path("ORBIT_TEST_STREAMS_ROOT"),
        help="Directory with the nemotron_cc MDS stream. Default: $ORBIT_TEST_STREAMS_ROOT.",
    )
    p.add_argument(
        "--kernel-path",
        type=str,
        default=os.environ.get("ORBIT_ATTN_KERNEL_PATH"),
        help="Local kernels-community/vllm-flash-attn3 build. Default: $ORBIT_ATTN_KERNEL_PATH.",
    )
    p.add_argument("--rope-theta", type=float, default=1.0e6)
    p.add_argument("--seq-len", type=int, default=16384)
    args = p.parse_args()

    skip_reason = t.gpu_skip_reason()
    if skip_reason is not None:
        print(f"SKIPPED: {skip_reason}", flush=True)
        return 0
    if args.hf_dir is None:
        p.error("--hf-dir or ORBIT_TEST_HF_DIR is required")
    if args.streams_root is None:
        p.error("--streams-root or ORBIT_TEST_STREAMS_ROOT is required")

    device = torch.device("cuda")
    new_model = t.load_model(args.hf_dir, args.kernel_path, args.rope_theta, device).eval()
    reference_model = load_reference_model(
        args.reference_package, args.hf_dir, args.kernel_path, args.rope_theta, device
    )

    sample = t.find_samples_with_eos(args.streams_root, "nemotron_cc", 28, 36, count=1, scan=3000)[0]
    ids = torch.tensor(sample[: args.seq_len][None], device=device)
    h_new = t.loop_hidden(new_model, ids)
    h_reference = t.loop_hidden(reference_model, ids)
    report = {
        "reference_package": str(args.reference_package),
        "eos_in_input": int((ids == t.EOS_ID).sum()),
        "hidden_max_abs_diff_per_step": t.hidden_max_diff(h_new, h_reference),
        "logits": t.logits_diff(new_model, h_new[-1], h_reference[-1]),
    }
    report["pass"] = (
        all(diff == 0.0 for diff in report["hidden_max_abs_diff_per_step"])
        and report["logits"]["max_abs_diff"] == 0.0
    )
    print(json.dumps(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
