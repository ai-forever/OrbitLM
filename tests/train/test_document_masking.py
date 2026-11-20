"""Checks for the opt-in document masking in OuroModel.

The script runs these groups and prints one JSON report at the end:
  unit - CPU checks of the segment offsets that build_document_cu_seqlens builds
         from EOS positions;
  a    - a batch with no interior EOS gives the same logits masked and dense;
  b    - with [docA EOS docB], docB's logits under masking match [EOS docB] run
         alone, and differ without masking;
  c    - backward runs with masking, gradients are finite, the loss matches dense
         on a batch with no interior EOS, and no gradient crosses a document;
  d    - forward+backward time at seq 16384, batch 1/2/4, masked vs dense, on real
         samples. This group only measures and has no pass/fail criterion.

Only the unit group runs on CPU, and pytest collects it as test_segment_offsets.
Groups a-d need a Hopper GPU, the FA3 kernel, an HF export of the model and a
directory of MDS streams. On a machine without a suitable GPU they are skipped
with a message. On a Hopper GPU, a FA3 kernel that cannot be loaded is not a
skip but a failure, reported as the check "kernel". tests/train/README.md lists the
requirements and the environment variables.

    python tests/train/test_document_masking.py --groups unit
    python tests/train/test_document_masking.py --hf-dir <hf export> --streams-root <streams> --out report.json

The exit status is 0 when every group that ran passed (or was skipped for lack
of a suitable GPU). It is 1 when any group failed or the FA3 kernel could not
be loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# The tests import the trainer package from the src/ directory of this checkout.
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from orbit import modelling_ouro  # noqa: E402
from orbit.configuration_ouro import OuroConfig  # noqa: E402
from orbit.data import build_xy_from_sample, tokens_bytes_to_ids  # noqa: E402
from orbit.modelling_ouro import (  # noqa: E402
    OuroForCausalLM,
    build_document_cu_seqlens,
)
from orbit.train import OrbitLMTrainingModule  # noqa: E402

# <|endoftext|> of the Qwen3 tokenizer, which separates documents in the OrbitLM data.
EOS_ID = 151643
FA3_IMPL = "kernels-community/vllm-flash-attn3"
GROUPS = ("unit", "a", "b", "c", "d")


def env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value) if value else None


def gpu_skip_reason() -> Optional[str]:
    """Explains why the GPU checks cannot run on this machine, or returns None."""
    if not torch.cuda.is_available():
        return "no CUDA GPU is available"
    major, minor = torch.cuda.get_device_capability(0)
    if major != 9:
        return (
            "the FA3 kernel needs a Hopper GPU (compute capability 9.x), "
            f"found {torch.cuda.get_device_name(0)} with {major}.{minor}"
        )
    return None


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def read_stream_samples(
    streams_root: Path, stream: str, count: int, start: int = 0
) -> list[np.ndarray]:
    """Reads `count` token samples from one MDS stream without copying or caching it.

    The streaming library's own MDS reader decodes the shards in place, and the
    trainer's tokens_bytes_to_ids turns the bytes into ids (the streams hold uint32
    tokens, which is dtype code 1 in data.py).
    """
    from streaming.base.format import reader_from_json

    stream_dir = streams_root / stream
    index = json.loads((stream_dir / "index.json").read_text())
    samples: list[np.ndarray] = []
    skipped = 0
    for shard_info in index["shards"]:
        reader = reader_from_json(str(stream_dir), None, shard_info)
        for idx in range(reader.samples):
            if skipped < start:
                skipped += 1
                continue
            ids = tokens_bytes_to_ids(reader.get_item(idx), override_dtype_code=1)
            samples.append(ids)
            if len(samples) >= count:
                return samples
    return samples


def strip_eos(ids: np.ndarray) -> np.ndarray:
    return ids[ids != EOS_ID]


def eos_count(ids: np.ndarray) -> int:
    return int((ids == EOS_ID).sum())


def find_samples_with_eos(
    streams_root: Path, stream: str, lo: int, hi: int, count: int, scan: int
) -> list[np.ndarray]:
    """Picks the first `count` samples whose EOS count lies in [lo, hi].

    The ctx16k streams are much sparser in EOS than the 4K data (nemotron_cc has
    about 3 EOS per 4K tokens on average), so a test that needs the 4K density
    of 7-9 EOS per 4K tokens has to select such samples explicitly.
    """
    picked = [
        s for s in read_stream_samples(streams_root, stream, scan) if lo <= eos_count(s) <= hi
    ]
    if len(picked) < count:
        raise ValueError(f"{stream}: only {len(picked)} samples with {lo}..{hi} EOS in {scan}")
    return picked[:count]


def eos_free_rows(samples: list[np.ndarray], rows: int, length: int) -> list[np.ndarray]:
    """Cuts `rows` rows of real text with every EOS removed.

    Stripping its EOS tokens makes a sample shorter than a full row, so the
    stripped samples are joined first and then cut into rows.
    """
    stream = np.concatenate([strip_eos(s) for s in samples])
    if len(stream) < rows * length:
        raise ValueError(f"need {rows * length} EOS-free tokens, have {len(stream)}")
    return [stream[i * length : (i + 1) * length] for i in range(rows)]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def load_model(hf_dir: Path, kernel_path: str | None, rope_theta: float, device: torch.device):
    config = OuroConfig.from_pretrained(str(hf_dir))
    config.rope_theta = float(rope_theta)
    config.use_cache = False
    model, info = OuroForCausalLM.from_pretrained(
        str(hf_dir),
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        output_loading_info=True,
    )
    missing = [k for k in info.get("missing_keys", []) if "rotary_emb" not in k]
    if missing or info.get("unexpected_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"HF load mismatch: {info}")
    # Same order as train.build_base_model: build with eager, then switch the
    # config to the FA3 kernels path that training uses.
    model.config._attn_implementation = FA3_IMPL
    model.config._attn_kernel_repo_path = kernel_path
    return model.to(device)


def set_masking(model: OuroForCausalLM, enabled: bool) -> None:
    model.model.set_document_masking(EOS_ID if enabled else None)


@torch.no_grad()
def loop_hidden(model, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None):
    """Returns the per-recurrent-step hidden states (after the final norm)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            return_orbit_aux=True,
            compute_logits=False,
        )
    return [h.detach() for h in out.loop_hidden_states]


@torch.no_grad()
def logits_diff(model, h1: torch.Tensor, h2: torch.Tensor, chunk: int = 512) -> dict:
    """Max and mean |logits1 - logits2| over all positions, computed in chunks.

    Full logits at vocab 151936 and 16K positions would take ~10 GB per copy,
    so the comparison walks the sequence chunk by chunk.
    """
    assert h1.shape == h2.shape, (h1.shape, h2.shape)
    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    max_logit = 0.0
    argmax_agree = 0
    positions = 0
    for start in range(0, h1.shape[1], chunk):
        l1 = model.lm_head(h1[:, start : start + chunk]).float()
        l2 = model.lm_head(h2[:, start : start + chunk]).float()
        d = (l1 - l2).abs()
        max_abs = max(max_abs, float(d.max()))
        sum_abs += float(d.sum())
        count += d.numel()
        max_logit = max(max_logit, float(l1.abs().max()))
        argmax_agree += int((l1.argmax(-1) == l2.argmax(-1)).sum())
        positions += l1.shape[0] * l1.shape[1]
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": sum_abs / max(count, 1),
        "max_abs_logit": max_logit,
        "argmax_agreement": argmax_agree / max(positions, 1),
    }


def hidden_max_diff(hs1: list[torch.Tensor], hs2: list[torch.Tensor]) -> list[float]:
    return [float((a.float() - b.float()).abs().max()) for a, b in zip(hs1, hs2)]


# ---------------------------------------------------------------------------
# unit: segment offsets (CPU)
# ---------------------------------------------------------------------------


def check_segment_offsets() -> dict:
    E = EOS_ID
    cases = {
        # No EOS: one segment per sample.
        "no_eos": ([[1, 2, 3, 4], [5, 6, 7, 8]], [0, 4, 8], 4),
        # EOS in the middle opens a new segment that starts AT the EOS.
        "mid_eos": ([[1, 2, E, 4, 5]], [0, 2, 5], 3),
        # EOS at position 0 coincides with the sample start (no empty segment).
        "eos_first": ([[E, 1, 2, 3]], [0, 4], 4),
        # EOS as the last token becomes a one-token segment.
        "eos_last": ([[1, 2, 3, E]], [0, 3, 4], 3),
        # Consecutive EOS tokens give one-token segments each.
        "double_eos": ([[1, E, E, 2]], [0, 1, 2, 4], 2),
        # Sample boundaries always split, even when row 1 starts mid-document.
        "two_rows": ([[1, E, 2, 3], [4, 5, E, 6]], [0, 1, 4, 6, 8], 3),
    }
    results = {}
    ok = True
    for name, (rows, want_cu, want_max) in cases.items():
        cu, mx = build_document_cu_seqlens(torch.tensor(rows), E)
        passed = cu.tolist() == want_cu and mx == want_max and cu.dtype == torch.int32
        ok &= passed
        results[name] = {"cu_seqlens": cu.tolist(), "max_seqlen": mx, "pass": passed}
    # The input_ids tensor must not be modified by the helper.
    src = torch.tensor([[1, E, 3]])
    build_document_cu_seqlens(src, E)
    untouched = src.tolist() == [[1, E, 3]]
    ok &= untouched
    results["input_untouched"] = untouched
    results["pass"] = ok
    return results


def test_segment_offsets() -> None:
    result = check_segment_offsets()
    assert result["pass"], json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# a) no interior EOS: masked == dense
# ---------------------------------------------------------------------------


def check_no_interior_eos(model, samples: list[np.ndarray], device, seq_len: int) -> dict:
    rows = eos_free_rows(samples, 2, seq_len)
    ids = torch.tensor(np.stack(rows), device=device)
    assert int((ids == EOS_ID).sum()) == 0
    set_masking(model, False)
    dense = loop_hidden(model, ids)
    set_masking(model, True)
    masked = loop_hidden(model, ids)

    # Same batch with an EOS as the first token of each row. It sits on the
    # sample start, so the segments are still whole rows.
    ids_bos = ids.clone()
    ids_bos[:, 0] = EOS_ID
    set_masking(model, False)
    dense_bos = loop_hidden(model, ids_bos)
    set_masking(model, True)
    masked_bos = loop_hidden(model, ids_bos)
    set_masking(model, False)

    res = {
        "shape": list(ids.shape),
        "hidden_max_abs_diff_per_step": hidden_max_diff(dense, masked),
        "logits": logits_diff(model, dense[-1], masked[-1]),
        "eos_at_pos0_hidden_max_abs_diff_per_step": hidden_max_diff(dense_bos, masked_bos),
        "eos_at_pos0_logits": logits_diff(model, dense_bos[-1], masked_bos[-1]),
    }
    res["pass"] = (
        res["logits"]["max_abs_diff"] <= 0.05 and res["eos_at_pos0_logits"]["max_abs_diff"] <= 0.05
    )
    return res


# ---------------------------------------------------------------------------
# b) isolation: [docA EOS docB] masked == [EOS docB] alone
# ---------------------------------------------------------------------------


def check_document_isolation(
    model, samples: list[np.ndarray], rich_sample: np.ndarray, device, seq_len: int, len_a: int
) -> dict:
    doc_a = strip_eos(samples[0])[:len_a]
    len_b = seq_len - len_a - 1
    doc_b = strip_eos(samples[1])[:len_b]
    row0 = np.concatenate([doc_a, [EOS_ID], doc_b])
    # Row 1 is a window of a real sample with its natural EOS separators, so the
    # batch also checks several boundaries and the boundary between the two rows.
    # The window starts 200 tokens before the first EOS, so row 1 begins in the
    # middle of a document, as rows of packed data do.
    first_eos = int(np.flatnonzero(rich_sample == EOS_ID)[0])
    start = max(0, first_eos - 200)
    row1 = rich_sample[start : start + seq_len]
    ids = torch.tensor(np.stack([row0, row1]), device=device)

    set_masking(model, True)
    masked = loop_hidden(model, ids)
    set_masking(model, False)
    dense = loop_hidden(model, ids)

    solo_ids = torch.tensor(np.concatenate([[EOS_ID], doc_b])[None], device=device)
    solo = loop_hidden(model, solo_ids)
    # The same solo run at the positions docB occupies inside row 0. This removes
    # the RoPE rounding difference and isolates the masking itself.
    shifted_pos = torch.arange(len_a, seq_len, device=device)[None]
    solo_shifted = loop_hidden(model, solo_ids, position_ids=shifted_pos)

    b_slice = slice(len_a, seq_len)
    masked_b = [h[0:1, b_slice] for h in masked]
    dense_b = [h[0:1, b_slice] for h in dense]

    res = {
        "len_docA": int(len_a),
        "len_EOS_docB": int(seq_len - len_a),
        "masked_vs_solo_default_positions": logits_diff(model, masked_b[-1], solo[-1]),
        "masked_vs_solo_same_positions": logits_diff(model, masked_b[-1], solo_shifted[-1]),
        "masked_vs_solo_same_positions_hidden_per_step": hidden_max_diff(masked_b, solo_shifted),
        "dense_vs_solo_same_positions": logits_diff(model, dense_b[-1], solo_shifted[-1]),
        # Noise floor: the same solo input at two position offsets, no masking involved.
        "rope_offset_noise_floor": logits_diff(model, solo[-1], solo_shifted[-1]),
    }
    # docA's last token must still predict from docA only; under causal attention
    # that holds with or without masking, so masked and dense must agree there.
    res["docA_masked_vs_dense"] = logits_diff(
        model, masked[-1][0:1, :len_a], dense[-1][0:1, :len_a]
    )

    # Every natural document in row 1, run alone at its own positions.
    cu, _ = build_document_cu_seqlens(ids[1:2], EOS_ID)
    bounds = cu.tolist()
    seg_max = 0.0
    seg_leak_max = 0.0
    seg_lengths = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        seg_lengths.append(e - s)
        seg_ids = ids[1:2, s:e]
        seg_pos = torch.arange(s, e, device=device)[None]
        seg_solo = loop_hidden(model, seg_ids, position_ids=seg_pos)
        seg_max = max(seg_max, logits_diff(model, masked[-1][1:2, s:e], seg_solo[-1])["max_abs_diff"])
        if s > 0:
            seg_leak_max = max(
                seg_leak_max, logits_diff(model, dense[-1][1:2, s:e], seg_solo[-1])["max_abs_diff"]
            )
    res["row1_natural_segments"] = {
        "eos_count": int((ids[1] == EOS_ID).sum()),
        "segment_lengths": seg_lengths,
        "masked_vs_solo_max_abs_diff": seg_max,
        "dense_vs_solo_max_abs_diff_excl_first": seg_leak_max,
    }
    iso = res["masked_vs_solo_same_positions"]["max_abs_diff"]
    leak = res["dense_vs_solo_same_positions"]["max_abs_diff"]
    res["pass"] = (
        iso <= 0.05
        and seg_max <= 0.05
        and res["row1_natural_segments"]["eos_count"] >= 3
        and seg_leak_max > 20.0 * max(seg_max, 1e-3)
        and res["masked_vs_solo_default_positions"]["max_abs_diff"]
        <= max(0.05, 2.0 * res["rope_offset_noise_floor"]["max_abs_diff"])
        and leak > 20.0 * max(iso, 1e-3)
        and res["docA_masked_vs_dense"]["max_abs_diff"] <= 0.05
    )
    return res


# ---------------------------------------------------------------------------
# c) backward: finite grads with masking; masked loss == dense loss without EOS
# ---------------------------------------------------------------------------


def make_xy(ids_rows: list[np.ndarray], seq_len: int, device):
    xs, ys = [], []
    for ids in ids_rows:
        x, y = build_xy_from_sample({}, ids, block_size=seq_len, pad_id=EOS_ID, eos_id=EOS_ID)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def train_step(module: OrbitLMTrainingModule, x, y):
    module.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        metrics = module(input_ids=x, labels=y, need_diagnostics=True)
    loss = metrics["loss"]
    loss.backward()
    return metrics


def grad_summary(module) -> dict:
    total_sq = 0.0
    nonfinite = []
    n = 0
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        n += 1
        g = p.grad.float()
        if not bool(torch.isfinite(g).all()):
            nonfinite.append(name)
        total_sq += float(g.pow(2).sum())
    return {"params_with_grad": n, "nonfinite_params": nonfinite, "grad_norm": math.sqrt(total_sq)}


def flat_grads(module) -> torch.Tensor:
    return torch.cat([p.grad.float().reshape(-1) for p in module.parameters() if p.grad is not None])


def embedding_grad_isolation(model, doc_a: np.ndarray, doc_b: np.ndarray, device, masked: bool) -> float:
    """Largest |d loss(docB) / d embedding(docA)| over docA's positions.

    The loss is the cross-entropy of docB's positions only, so with document
    masking no gradient may reach docA's token embeddings through any of the
    four recurrent steps; the value must be exactly zero. Without masking it
    is not.
    """
    ids = torch.tensor(np.concatenate([doc_a, [EOS_ID], doc_b])[None], device=device)
    len_a = len(doc_a)
    captured = {}

    def keep_embedding_output(_module, _inputs, output):
        output.retain_grad()
        captured["emb"] = output

    handle = model.model.embed_tokens.register_forward_hook(keep_embedding_output)
    try:
        set_masking(model, masked)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, use_cache=False, return_orbit_aux=True, compute_logits=False)
            hidden_b = out.loop_hidden_states[-1][:, len_a:-1]
            logits_b = model.lm_head(hidden_b).float()
            loss = torch.nn.functional.cross_entropy(
                logits_b.reshape(-1, logits_b.shape[-1]), ids[:, len_a + 1 :].reshape(-1)
            )
        loss.backward()
        grad_a = captured["emb"].grad[:, :len_a]
        return float(grad_a.float().abs().max())
    finally:
        handle.remove()
        set_masking(model, False)
        model.zero_grad(set_to_none=True)


def check_backward(module, samples: list[np.ndarray], rich_sample: np.ndarray, device, seq_len: int) -> dict:
    model = module.model
    res = {}

    # c1: a real sample with many interior EOS, masked, forward+backward.
    x, y = make_xy([rich_sample[: seq_len + 1]], seq_len, device)
    set_masking(model, True)
    m = train_step(module, x, y)
    res["real_sample_masked"] = {
        "eos_in_input": int((x == EOS_ID).sum()),
        "loss": float(m["loss"]),
        "loss_lm": float(m["loss_lm"]),
        **grad_summary(module),
    }
    set_masking(model, False)
    m_dense = train_step(module, x, y)
    res["real_sample_dense_for_reference"] = {
        "loss": float(m_dense["loss"]),
        "loss_lm": float(m_dense["loss_lm"]),
        "grad_norm": grad_summary(module)["grad_norm"],
    }

    # c2: no interior EOS, masked vs dense loss and gradients. The dense pass
    # runs twice: FA3's backward accumulates dQ with atomics, so two identical
    # dense passes already differ slightly, and that is the noise floor for the
    # masked-vs-dense gradient difference.
    rows = eos_free_rows(samples, 2, seq_len + 1)
    x, y = make_xy(rows, seq_len, device)
    assert int((x == EOS_ID).sum()) == 0
    set_masking(model, False)
    md = train_step(module, x, y)
    gd = flat_grads(module)
    train_step(module, x, y)
    gd2 = flat_grads(module)
    set_masking(model, True)
    mm = train_step(module, x, y)
    gm = flat_grads(module)
    set_masking(model, False)

    def rel(a, b):
        return float((a - b).norm() / a.norm().clamp_min(1e-12))

    res["no_eos_batch"] = {
        "shape": list(x.shape),
        "loss_dense": float(md["loss"]),
        "loss_masked": float(mm["loss"]),
        "loss_abs_diff": abs(float(md["loss"]) - float(mm["loss"])),
        "grad_cosine_masked_vs_dense": float(torch.nn.functional.cosine_similarity(gd, gm, dim=0)),
        "grad_rel_l2_diff_masked_vs_dense": rel(gd, gm),
        "grad_rel_l2_diff_dense_vs_dense_noise_floor": rel(gd, gd2),
        **{f"masked_{k}": v for k, v in grad_summary(module).items()},
    }
    del gd, gd2, gm
    module.zero_grad(set_to_none=True)

    # c3: backward isolation through the embeddings, at 4K for speed.
    doc_a = strip_eos(samples[0])[:1500]
    doc_b = strip_eos(samples[1])[:2595]
    res["embedding_grad_isolation"] = {
        "masked_max_abs_grad_on_docA": embedding_grad_isolation(model, doc_a, doc_b, device, True),
        "dense_max_abs_grad_on_docA": embedding_grad_isolation(model, doc_a, doc_b, device, False),
    }

    nb = res["no_eos_batch"]
    iso = res["embedding_grad_isolation"]
    res["pass"] = (
        math.isfinite(res["real_sample_masked"]["loss"])
        and not res["real_sample_masked"]["nonfinite_params"]
        and nb["loss_abs_diff"] <= 1e-3
        and nb["grad_cosine_masked_vs_dense"] >= 0.999
        and nb["grad_rel_l2_diff_masked_vs_dense"]
        <= max(0.01, 3.0 * nb["grad_rel_l2_diff_dense_vs_dense_noise_floor"])
        and iso["masked_max_abs_grad_on_docA"] == 0.0
        and iso["dense_max_abs_grad_on_docA"] > 0.0
    )
    module.zero_grad(set_to_none=True)
    return res


# ---------------------------------------------------------------------------
# d) speed at seq 16384
# ---------------------------------------------------------------------------


def time_train_step(module, x, y, masked: bool, warmup: int, iters: int) -> dict:
    set_masking(module.model, masked)
    for _ in range(warmup):
        train_step(module, x, y)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train_step(module, x, y)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    module.zero_grad(set_to_none=True)
    set_masking(module.model, False)
    mean = sum(times) / len(times)
    return {
        "mean_s": mean,
        "min_s": min(times),
        "max_s": max(times),
        "tokens_per_s": x.numel() / mean,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


# Speed-test sample sets: (stream, how to pick, argument). "offset" takes
# consecutive samples after skipping the head of the stream, where the ctx16k
# builder put its longest documents. "eos" picks samples with the 4K-data
# density of 7-9 EOS per 4K tokens (28-36 per 16K sample).
SPEED_SETS = {
    "nemotron_cc": ("nemotron_cc", "offset", 1000),
    "ru_wiki": ("ru_wiki", "offset", 100),
    "nemotron_cc_7to9_per_4k": ("nemotron_cc", "eos", (28, 36)),
    "opc": ("opc", "offset", 100),
}


def measure_speed(module, stream_samples: dict[str, list[np.ndarray]], device, seq_len: int,
                  batches: list[int], warmup: int, iters: int) -> dict:
    res = {}
    for stream, samples in stream_samples.items():
        for bs in batches:
            rows = [s[: seq_len + 1] for s in samples[:bs]]
            x, y = make_xy(rows, seq_len, device)
            cu, mx = build_document_cu_seqlens(x, EOS_ID)
            entry = {
                "eos_per_row": [int((r == EOS_ID).sum()) for r in x],
                "segments": len(cu) - 1,
                "max_segment": mx,
            }
            try:
                entry["dense"] = time_train_step(module, x, y, False, warmup, iters)
                entry["masked"] = time_train_step(module, x, y, True, warmup, iters)
                entry["masked_speedup"] = entry["dense"]["mean_s"] / entry["masked"]["mean_s"]
            except torch.cuda.OutOfMemoryError as exc:
                entry["error"] = f"OOM: {str(exc)[:200]}"
                module.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            res[f"{stream}_bs{bs}"] = entry
            print(f"[d] {stream} bs={bs}: {json.dumps(entry)}", flush=True)
    return res


# ---------------------------------------------------------------------------


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    p = argparse.ArgumentParser(description="Checks for document masking in OuroModel.")
    p.add_argument(
        "--hf-dir",
        type=Path,
        default=env_path("ORBIT_TEST_HF_DIR"),
        help="HF export (save_pretrained directory) of the model. Default: $ORBIT_TEST_HF_DIR.",
    )
    p.add_argument(
        "--streams-root",
        type=Path,
        default=env_path("ORBIT_TEST_STREAMS_ROOT"),
        help=(
            "Directory with 16,385-token MDS streams (nemotron_cc, plus ru_wiki and opc "
            "for group d). Default: $ORBIT_TEST_STREAMS_ROOT."
        ),
    )
    p.add_argument(
        "--kernel-path",
        type=str,
        default=os.environ.get("ORBIT_ATTN_KERNEL_PATH"),
        help=(
            "Local kernels-community/vllm-flash-attn3 build. Default: $ORBIT_ATTN_KERNEL_PATH; "
            "without it the kernel is downloaded from the HF Hub."
        ),
    )
    p.add_argument("--rope-theta", type=float, default=1.0e6)
    p.add_argument("--groups", type=str, default=",".join(GROUPS))
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--b-seq-len", type=int, default=4096)
    p.add_argument("--b-len-a", type=int, default=1500)
    p.add_argument("--speed-batches", type=str, default="1,2,4")
    p.add_argument("--speed-sets", type=str, default=",".join(SPEED_SETS))
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--out", type=Path, default=None, help="Also write the JSON report to this file.")
    return p, p.parse_args()


def run_gpu_groups(args, gpu_groups: list[str], report: dict, dump) -> None:
    """Runs the requested GPU groups and stores their results in `report`.

    The FA3 kernel is loaded before any group runs. If it cannot be loaded, the
    error is stored as a failed "kernel" check and no group runs. On a Hopper GPU
    the kernel should always load, so this means the setup is broken.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = load_model(args.hf_dir, args.kernel_path, args.rope_theta, device)
    # The kernel is loaded here rather than inside the first forward pass, so a
    # missing or incompatible kernel build shows up as one clear failure.
    try:
        kernel = modelling_ouro._get_kernels_flash_attn3(model.model.layers[0].self_attn)
    except Exception as exc:
        report["kernel"] = {
            "pass": False,
            "error": repr(exc),
            "hint": (
                "set --kernel-path or ORBIT_ATTN_KERNEL_PATH to a build that matches "
                "this torch version"
            ),
        }
        print(f"FAILED: the FA3 kernel could not be loaded ({exc!r})", flush=True)
        return
    report["kernel_module"] = getattr(kernel, "__file__", None)
    report["model"] = {
        "hf_dir": str(args.hf_dir),
        "rope_theta": model.config.rope_theta,
        "total_ut_steps": model.config.total_ut_steps,
        "layers": model.config.num_hidden_layers,
        "heads": model.config.num_attention_heads,
        "kv_heads": model.config.num_key_value_heads,
    }
    nem = read_stream_samples(args.streams_root, "nemotron_cc", 7)
    rich = find_samples_with_eos(args.streams_root, "nemotron_cc", 28, 36, count=4, scan=3000)
    report["data"] = {
        "nemotron_cc_sample_len": [int(len(s)) for s in nem],
        "nemotron_cc_eos_per_sample": [eos_count(s) for s in nem],
        "nemotron_cc_rich_eos_per_sample": [eos_count(s) for s in rich],
    }

    if "a" in gpu_groups:
        model.eval()
        report["a"] = check_no_interior_eos(model, nem[0:3], device, args.seq_len)
        print("[a]", json.dumps(report["a"]), flush=True)
        dump()
    if "b" in gpu_groups:
        model.eval()
        report["b"] = check_document_isolation(
            model, nem[2:4], rich[0], device, args.b_seq_len, args.b_len_a
        )
        print("[b]", json.dumps(report["b"]), flush=True)
        dump()
    if "c" in gpu_groups or "d" in gpu_groups:
        model.train()
        # Same checkpointing mode as train.maybe_enable_gradient_checkpointing.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        module = OrbitLMTrainingModule(
            model, beta=0.15, exit_mode="learned", pad_token_id=EOS_ID, lm_chunk_size=128
        )
        if "c" in gpu_groups:
            report["c"] = check_backward(module, nem[4:7], rich[1], device, args.seq_len)
            print("[c]", json.dumps(report["c"]), flush=True)
            dump()
        if "d" in gpu_groups:
            batches = [int(b) for b in args.speed_batches.split(",") if b]
            stream_samples = {}
            for label in [s for s in args.speed_sets.split(",") if s]:
                stream, how, arg = SPEED_SETS[label]
                if how == "offset":
                    picked = read_stream_samples(args.streams_root, stream, max(batches), start=arg)
                elif len(rich) >= max(batches):
                    picked = rich
                else:
                    picked = find_samples_with_eos(
                        args.streams_root, stream, arg[0], arg[1], count=max(batches), scan=3000
                    )
                stream_samples[label] = picked
                report["data"][f"speed_{label}_eos_per_sample"] = [eos_count(s) for s in picked]
            report["d"] = measure_speed(
                module, stream_samples, device, args.seq_len, batches, args.warmup, args.iters
            )
            dump()


def main() -> int:
    parser, args = parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        parser.error(f"unknown groups {unknown}; choose from {','.join(GROUPS)}")

    report: dict = {"args": {k: str(v) for k, v in vars(args).items()}}
    report["env"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    def dump() -> None:
        if args.out is not None:
            args.out.write_text(json.dumps(report, indent=2))

    if "unit" in groups:
        report["unit"] = check_segment_offsets()
        print("[unit]", json.dumps(report["unit"]), flush=True)
        dump()

    gpu_groups = [g for g in groups if g != "unit"]
    if gpu_groups:
        skip_reason = gpu_skip_reason()
        if skip_reason is not None:
            report["skipped"] = {"groups": gpu_groups, "reason": skip_reason}
            print(f"SKIPPED groups {','.join(gpu_groups)}: {skip_reason}", flush=True)
        else:
            if args.hf_dir is None:
                parser.error("groups a-d need --hf-dir or ORBIT_TEST_HF_DIR")
            if args.streams_root is None:
                parser.error("groups a-d need --streams-root or ORBIT_TEST_STREAMS_ROOT")
            run_gpu_groups(args, gpu_groups, report, dump)

    # The "kernel" entry exists only when the FA3 kernel failed to load, and then
    # it must fail the run like any failed group.
    checks = ["kernel", *groups]
    report["pass"] = {
        name: report[name]["pass"]
        for name in checks
        if isinstance(report.get(name), dict) and "pass" in report[name]
    }
    dump()
    print(json.dumps(report, indent=2))
    return 0 if all(report["pass"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
