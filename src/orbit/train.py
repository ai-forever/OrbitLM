"""The OrbitLM trainer: one script for every training stage of the release, chosen by its config.

    torchrun --nproc_per_node 8 -m orbit.train --config configs/train/pretrain/stage1_4k.yaml

The model is a looped transformer (Ouro architecture): one stack of 24 layers runs
total_ut_steps times, and a learned exit gate gives every token a probability of stopping after
each loop. The loss is the LM loss of every loop weighted by that probability, minus beta times
the entropy of the exit distribution (losses.py).

The config, not a mode switch, decides the stage:

- 4K pretraining: seq_len 4096, random init (or init_from_hf of the previous 4K stage), plain
  causal attention over packed documents, labels on every token.
- Context extension (16K and 32K): seq_len 16384 or 32768, init_from_hf of the previous stage,
  an explicit rope_theta, and document_masking: true, which stops attention at every
  <|endoftext|> so packed documents cannot see each other.
- SFT: seq_len 32768, init_from_hf of the 32K stage, document masking, and labels_premasked: true,
  which reads the per-token loss mask of the SFT shards so only the responses are trained.

Every stage mixes its MDS streams with one Mosaic StreamingDataset (stage1_streams and
stage1_dataset_proportions), trains with FSDP and fp32 master weights in bf16 compute, and
writes rank checkpoints that resume exactly on the same world size (checkpointing.py).
orbit.export turns a checkpoint into the HF folder that the next stage loads.

main() runs the phases in order: parse the config, resolve the init source, set up the process
group, start the run (run folder, logging, in-training evaluation, seeds), build the data, build
and wrap the model, build the optimizer, restore a checkpoint or start fresh, run the step loop,
and tear down. The package README describes the config keys.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    _PACKAGE_ROOT = Path(__file__).resolve().parent
    _REPO_ROOT = _PACKAGE_ROOT.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    __package__ = _PACKAGE_ROOT.name

from .configuration_ouro import OuroConfig
from .modelling_ouro import OuroForCausalLM

from .checkpointing import (
    build_checkpoint_metadata_extra,
    build_local_dataloader_state,
    checkpoint_dir_for_step,
    infer_resume_model_config_overrides,
    load_rank_checkpoint,
    resolve_run_dir_from_ref,
    restore_local_dataloader_state,
    capture_rng_state,
    restore_rng_state,
    save_rank_checkpoint,
    write_checkpoint_metadata,
    write_latest_pointer,
)
from .experiment_config import load_experiment_config
from .data import (
    MdsTokenIterableDataset,
    build_stage1_mixture_dataset,
    decode_sample_dtype_code,
    build_streaming_mixture_dataset,
)
from .losses import (
    build_exit_pdf,
    compute_weighted_lm_and_exit_loss,
    summarize_exit_distribution,
    summarize_threshold_exit_counts,
    summarize_gate_probabilities,
)
from .optional import SummaryWriter, Task, clean_stale_shared_memory, load_dotenv
from .runtime import (
    FA3_KERNEL_IMPL,
    GracefulStopController,
    TokenizerConfig,
    format_count,
    load_model_config,
    preflight_flash_attention,
    read_clearml_task_id,
    report_scalar,
    set_global_random_seeds,
    write_clearml_task_id,
)
from .fsdp_utils import (
    build_parallel_model,
    clip_grad_norm,
    distributed_enabled,
    get_fsdp_use_orig_params,
    is_fsdp_model,
    maybe_no_sync,
    unwrap_model,
)
from .train_eval import add_eval_arguments, build_in_training_eval, validate_eval_arguments


# The trainer has one data-parallel mode, FSDP. The value is recorded in every checkpoint.
PARALLELISM = "fsdp"
# The only exit mode: the loss weights every loop by the learned exit distribution.
EXIT_MODE = "learned"

# Options that earlier versions of the trainer had. A config may still name one with the value
# listed here, which is the behaviour this trainer always has. Any other value is refused, because
# the option would otherwise be ignored without notice and the run would train differently.
# None in a tuple means the key may be null; ANY_VALUE marks options that never changed training.
ANY_VALUE = object()
REMOVED_CONFIG_KEYS: dict[str, Any] = {
    "init_from_fast": (None,),
    "init_from_legacy": (None,),
    "exit_mode": (None, EXIT_MODE),
    "parallelism": (None, PARALLELISM),
    "resume_validation": (None, "minimal"),
    "weight_decay_mode": (None, "groups"),
    "zero_grad_set_to_none": (None, True),
    "detect_anomaly": (None, False),
    "ema_decay": (None, 0, 0.0),
    "ema_interval": ANY_VALUE,
    "embedding_injection_mode": (None, "none"),
    "embedding_injection_s0_mode": ANY_VALUE,
    "embedding_injection_projection_mode": ANY_VALUE,
}


@dataclass
class DistContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


@dataclass
class ResumeState:
    epoch: int = 0
    global_step: int = 0
    micro_step: int = 0
    micro_in_step: int = 0
    grad_accum: int = 1
    local_samples_seen: int = 0
    tokens_seen: int = 0
    max_steps: Optional[int] = None
    # Counted since this process started; it is not part of the checkpoint.
    nonfinite_grad_skips: int = 0

    @classmethod
    def from_payload(cls, payload: Optional[dict[str, Any]]) -> "ResumeState":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            epoch=int(payload.get("epoch") or 0),
            global_step=int(payload.get("global_step") or 0),
            micro_step=int(payload.get("micro_step") or 0),
            micro_in_step=int(payload.get("micro_in_step") or 0),
            grad_accum=int(payload.get("grad_accum") or 1),
            local_samples_seen=int(payload.get("local_samples_seen") or 0),
            tokens_seen=int(payload.get("tokens_seen") or 0),
            max_steps=(
                int(payload["max_steps"])
                if payload.get("max_steps") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "micro_step": int(self.micro_step),
            "micro_in_step": int(self.micro_in_step),
            "grad_accum": int(self.grad_accum),
            "local_samples_seen": int(self.local_samples_seen),
            "tokens_seen": int(self.tokens_seen),
            "max_steps": None if self.max_steps is None else int(self.max_steps),
        }


class WarmupLRScheduler:
    """Constant LR, or a linear warmup followed by a cosine decay to min_lr (warmup_cosine)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler_type: str,
        base_lr: float,
        warmup_steps: int = 0,
        warmup_start_lr: Optional[float] = None,
        warmup_start_factor: float = 0.1,
        warmup_start_step: int = 0,
        cosine_steps: Optional[int] = None,
        min_lr: Optional[float] = None,
        min_lr_factor: float = 0.1,
    ):
        self.optimizer = optimizer
        self.scheduler_type = str(scheduler_type)
        self.base_lr = float(base_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.warmup_start_lr = (
            float(warmup_start_lr)
            if warmup_start_lr is not None
            else self.base_lr * float(warmup_start_factor)
        )
        self.warmup_start_factor = float(warmup_start_factor)
        self.warmup_start_step = int(warmup_start_step)
        self.cosine_steps = (
            None if cosine_steps is None else max(1, int(cosine_steps))
        )
        self.min_lr = (
            float(min_lr)
            if min_lr is not None
            else self.base_lr * float(min_lr_factor)
        )
        self.min_lr_factor = float(min_lr_factor)
        self.last_global_step: Optional[int] = None

    def set_warmup_start_step(self, step: int) -> None:
        self.warmup_start_step = int(step)

    def set_cosine_steps(self, steps: int) -> None:
        self.cosine_steps = max(1, int(steps))

    def get_lr(self, global_step: int) -> float:
        if self.scheduler_type == "constant":
            return self.base_lr
        if self.scheduler_type != "warmup_cosine":
            raise RuntimeError(f"Unsupported lr_scheduler={self.scheduler_type!r}")
        if self.warmup_steps > 0:
            warmup_progress = (int(global_step) - int(self.warmup_start_step)) / float(
                max(1, self.warmup_steps)
            )
            warmup_progress = min(1.0, max(0.0, warmup_progress))
            if warmup_progress < 1.0:
                return self.warmup_start_lr + (
                    self.base_lr - self.warmup_start_lr
                ) * warmup_progress
        cosine_steps = max(1, int(self.cosine_steps or 1))
        cosine_start_step = int(self.warmup_start_step) + int(self.warmup_steps)
        cosine_progress = (int(global_step) - cosine_start_step) / float(cosine_steps)
        cosine_progress = min(1.0, max(0.0, cosine_progress))
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_scale

    def step(self, global_step: int) -> float:
        lr = self.get_lr(int(global_step))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_global_step = int(global_step)
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler_type": self.scheduler_type,
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "warmup_start_lr": self.warmup_start_lr,
            "warmup_start_factor": self.warmup_start_factor,
            "warmup_start_step": self.warmup_start_step,
            "cosine_steps": self.cosine_steps,
            "min_lr": self.min_lr,
            "min_lr_factor": self.min_lr_factor,
            "last_global_step": self.last_global_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self.scheduler_type = str(state.get("scheduler_type", self.scheduler_type))
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.warmup_start_lr = float(state.get("warmup_start_lr", self.warmup_start_lr))
        self.warmup_start_factor = float(
            state.get("warmup_start_factor", self.warmup_start_factor)
        )
        self.warmup_start_step = int(
            state.get("warmup_start_step", self.warmup_start_step)
        )
        cosine_steps = state.get("cosine_steps", self.cosine_steps)
        self.cosine_steps = None if cosine_steps is None else max(1, int(cosine_steps))
        self.min_lr = float(state.get("min_lr", self.min_lr))
        self.min_lr_factor = float(state.get("min_lr_factor", self.min_lr_factor))
        last_global_step = state.get("last_global_step")
        self.last_global_step = (
            None if last_global_step is None else int(last_global_step)
        )


EXIT_TOKEN_COUNT_THRESHOLDS = (0.7, 0.9)
PRIMARY_EXIT_TOKEN_COUNT_THRESHOLD = EXIT_TOKEN_COUNT_THRESHOLDS[-1]


def threshold_tag(threshold: float) -> str:
    return f"thr{str(float(threshold)).replace('.', 'p')}"


class OrbitLMTrainingModule(nn.Module):
    """Wraps OuroForCausalLM and returns the training loss and its diagnostics for one batch."""

    def __init__(
        self,
        model: OuroForCausalLM,
        *,
        beta: float,
        exit_mode: str,
        pad_token_id: Optional[int],
        lm_chunk_size: int,
    ):
        super().__init__()
        self.model = model
        self.beta = float(beta)
        self.exit_mode = str(exit_mode)
        if self.exit_mode != EXIT_MODE:
            raise ValueError(f"Unsupported exit_mode={self.exit_mode!r}; only {EXIT_MODE!r} is implemented.")
        # Recorded in checkpoints and compared on resume. The forward pass never masks by value.
        self.pad_token_id = pad_token_id
        self.lm_chunk_size = int(max(0, lm_chunk_size))
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        need_diagnostics: bool = False,
    ) -> dict[str, Any]:
        # The labels arrive already masked: data.build_xy_from_sample sets prompt and padding
        # positions to -100 by their position in the sample. They must not be masked again here
        # by comparing them with the pad id. The pad id equals the EOS id for this tokenizer, so
        # such masking would also drop the supervised EOS targets, and the model would never
        # learn to end a document.
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
            return_orbit_aux=True,
            compute_logits=False,
        )
        hidden_states_list = list(outputs.loop_hidden_states or ())
        gate_list = list(outputs.loop_gate_logits or ())
        if not hidden_states_list:
            raise RuntimeError("OrbitLM training path expected non-empty hidden_states_list.")

        valid = labels != -100
        stacked_exit_pdf = build_exit_pdf(gate_list)
        if stacked_exit_pdf is None:
            raise RuntimeError("The learned-exit loss needs the exit gate; the model config sets use_exit_gate false.")
        exit_tok_counts_by_threshold: dict[str, torch.Tensor] = {}
        for exit_threshold in EXIT_TOKEN_COUNT_THRESHOLDS:
            counts = summarize_threshold_exit_counts(
                stacked_exit_pdf,
                valid,
                threshold=exit_threshold,
            )
            if counts is not None:
                exit_tok_counts_by_threshold[threshold_tag(exit_threshold)] = counts.detach()
        (
            lm_loss,
            exit_loss,
            exit_entropy,
            per_ut_lm_losses,
        ) = compute_weighted_lm_and_exit_loss(
            hidden_states_list=hidden_states_list,
            stacked_exit_pdf=stacked_exit_pdf,
            labels=labels,
            valid_mask=valid,
            loss_fn=self.loss_fn,
            lm_head=self.model.lm_head,
            beta=self.beta,
            need_per_ut_lm_losses=need_diagnostics,
            lm_chunk_size=self.lm_chunk_size,
        )
        # The unweighted LM loss of the last loop, logged as loss_lm_final.
        final_loop_lm_loss = (
            per_ut_lm_losses[-1]
            if per_ut_lm_losses
            else None
        )
        if need_diagnostics:
            gate_lambda_means = summarize_gate_probabilities(gate_list, valid)
            exit_probs_mean = summarize_exit_distribution(stacked_exit_pdf, valid)
        else:
            gate_lambda_means = None
            exit_probs_mean = None

        total_loss = lm_loss + exit_loss
        metrics: dict[str, Any] = {
            "loss": total_loss,
            "loss_total": total_loss.detach(),
            "loss_lm": lm_loss.detach(),
            "loss_exit": exit_loss.detach(),
        }
        if need_diagnostics and final_loop_lm_loss is not None:
            metrics["loss_lm_final"] = final_loop_lm_loss.detach()
        if exit_entropy is not None:
            metrics["exit_entropy"] = exit_entropy.detach()
        if per_ut_lm_losses is not None:
            metrics["loss_lm_ut"] = torch.stack([x.detach() for x in per_ut_lm_losses])
        if gate_lambda_means is not None:
            metrics["gate_lambda"] = torch.stack([x.detach() for x in gate_lambda_means])
        if exit_probs_mean is not None:
            metrics["exit_prob"] = exit_probs_mean.detach()
        for tag, counts in exit_tok_counts_by_threshold.items():
            metrics[f"exit_tok_count_{tag}"] = counts
        primary_threshold_tag = threshold_tag(PRIMARY_EXIT_TOKEN_COUNT_THRESHOLD)
        if primary_threshold_tag in exit_tok_counts_by_threshold:
            metrics["exit_tok_count_thr"] = exit_tok_counts_by_threshold[primary_threshold_tag]
        return metrics


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def check_removed_config_keys(config: dict[str, Any], error) -> None:
    for key, allowed in REMOVED_CONFIG_KEYS.items():
        if key not in config or allowed is ANY_VALUE:
            continue
        if config[key] not in allowed:
            error(
                f"The config sets {key}: {config[key]!r}, but this trainer has no such option; "
                f"it always behaves as {key}: {allowed[-1]!r}. Remove the key."
            )


def check_config_choices(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuses a value of a choice option that is not one of its choices.

    argparse checks the choices only of values typed on the command line. A config value becomes
    the option's default, and argparse never checks a default. Without this check, a config that
    names a value this trainer no longer implements, such as an older mixing mode, would load
    without an error and the run would silently train with something else.
    """
    for action in parser._actions:
        if not action.choices:
            continue
        value = getattr(args, action.dest, None)
        if value not in action.choices:
            accepted = ", ".join(repr(choice) for choice in action.choices)
            parser.error(
                f"The config sets {action.dest}: {value!r}, which this trainer does not implement; "
                f"it accepts only {accepted}."
            )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML experiment config (configs/train/). Command-line options override its values.",
    )
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    # Without --config the parser below still builds, so that --help works; it then refuses to run.
    config_defaults = load_experiment_config(bootstrap_args.config) if bootstrap_args.config else {}

    def d(key: str, fallback: Any = None) -> Any:
        return config_defaults.get(key, fallback)

    parser = argparse.ArgumentParser(
        description="OrbitLM trainer.",
        parents=[bootstrap],
    )
    check_removed_config_keys(config_defaults, parser.error)
    parser.add_argument("--save-dir", type=Path, default=d("save_dir"))
    parser.add_argument("--run-name", type=str, default=d("run_name"))
    parser.add_argument("--resume-from", type=Path, default=d("resume_from"))
    parser.add_argument(
        "--init-from-hf",
        type=Path,
        default=d("init_from_hf"),
        help="Model-only warmstart from a HuggingFace save_pretrained directory.",
    )
    parser.add_argument("--model-config", type=Path, default=d("model_config"))
    parser.add_argument("--tokenizer-config", type=Path, default=d("tokenizer_config"))
    parser.add_argument("--stage1-root", type=str, default=d("stage1_root"))
    parser.add_argument("--stage1-epoch-samples", type=int, default=d("stage1_epoch_samples"))
    # The token dtype of every sample, which overrides the dtype label stored in each sample.
    # A wrong label would make the loader read the token bytes at the wrong width and produce ids
    # outside the vocabulary.
    parser.add_argument("--stage1-token-dtype", type=str, default=d("stage1_token_dtype"))
    # SFT shards carry a per-token loss_mask, so the labels arrive already masked by position
    # (response-only loss). This makes every stream's samples require that column.
    parser.add_argument(
        "--labels-premasked",
        action="store_true",
        default=bool(d("labels_premasked", False)),
    )
    parser.add_argument("--batch-size", type=int, default=d("batch_size"))
    parser.add_argument("--grad-accum", type=int, default=d("grad_accum"))
    parser.add_argument("--max-steps", type=int, default=d("max_steps"))
    parser.add_argument("--epochs", type=int, default=d("epochs", 1))
    parser.add_argument("--num-workers", type=int, default=d("num_workers", 0))
    parser.add_argument("--prefetch-factor", type=int, default=d("prefetch_factor", 2))
    parser.add_argument("--loader-timeout", type=float, default=d("loader_timeout", 0.0))
    parser.add_argument("--predownload", type=int, default=d("predownload"))
    parser.add_argument("--cache-limit", type=str, default=d("cache_limit"))
    parser.add_argument("--seed", type=int, default=d("seed", 42))
    parser.add_argument(
        "--model-init-seed",
        type=int,
        default=d("model_init_seed"),
        help="Optional torch seed applied immediately before model construction.",
    )
    parser.add_argument("--stage1-shuffle-seed", type=int, default=d("stage1_shuffle_seed"))
    parser.add_argument(
        "--stage1-shuffle-block-size",
        type=int,
        default=d("stage1_shuffle_block_size", 4194304),
    )
    parser.add_argument(
        "--stage1-num-canonical-nodes",
        type=int,
        default=d("stage1_num_canonical_nodes"),
    )
    # Mosaic random mixing is the only mode of this trainer and every release config names it.
    # Configs of earlier trainers that leave the key out were mixed by another mode, so such a
    # config reads its data in a different order here (README.md).
    parser.add_argument(
        "--stage1-mixing-mode",
        choices=["mosaic_random"],
        default=d("stage1_mixing_mode", "mosaic_random"),
        help="All streams are mixed by one Mosaic StreamingDataset, which samples each stream "
             "by its proportion and repeats a stream that runs out.",
    )
    parser.add_argument("--lr", type=float, default=d("lr"))
    parser.add_argument(
        "--lr-scheduler",
        choices=["constant", "warmup_cosine"],
        default=d("lr_scheduler", "constant"),
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=d("lr_warmup_steps", 0),
    )
    parser.add_argument(
        "--lr-warmup-start-lr",
        type=float,
        default=d("lr_warmup_start_lr"),
    )
    parser.add_argument(
        "--lr-warmup-start-factor",
        type=float,
        default=d("lr_warmup_start_factor", 0.1),
    )
    parser.add_argument(
        "--lr-warmup-start-step",
        default=d("lr_warmup_start_step", "auto"),
        help="Global step where warmup starts. Use 'auto' to start at the resumed step (0 for a new run).",
    )
    parser.add_argument(
        "--lr-cosine-steps",
        default=d("lr_cosine_steps", "auto"),
        help="Number of optimizer steps for cosine decay after warmup. Use 'auto' to end at max_steps.",
    )
    parser.add_argument("--lr-min", type=float, default=d("lr_min"))
    parser.add_argument(
        "--lr-min-factor",
        type=float,
        default=d("lr_min_factor", 0.1),
    )
    parser.add_argument("--weight-decay", type=float, default=d("weight_decay", 0.1))
    parser.add_argument("--adam-beta1", type=float, default=d("adam_beta1", 0.9))
    parser.add_argument("--adam-beta2", type=float, default=d("adam_beta2", 0.95))
    parser.add_argument("--max-grad-norm", type=float, default=d("max_grad_norm", 1.0))
    # Step-level NaN/Inf guard: on a non-finite gradient the optimizer step is
    # skipped (weights preserved) instead of corrupting the model. Abort once the
    # number of skipped steps exceeds this budget; <0 disables the abort.
    parser.add_argument(
        "--max-nonfinite-grad-skips",
        type=int,
        default=d("max_nonfinite_grad_skips", 20),
    )
    # With detect_nan on, the first microbatch of each rank that gives a non-finite gradient is
    # saved as a .pt file in this folder, so it can be inspected offline. An empty value saves nothing.
    parser.add_argument(
        "--dump-nonfinite-batch",
        type=str,
        default=d("dump_nonfinite_batch", ""),
    )
    parser.add_argument("--beta", type=float, default=d("beta"))
    parser.add_argument("--block-size", type=int, default=d("seq_len", d("block_size")))
    parser.add_argument("--total-ut-steps", type=int, default=d("total_ut_steps"))
    parser.add_argument("--rope-theta", type=float, default=d("rope_theta", 1.0e4))
    parser.add_argument("--tokens-per-step", type=int, default=d("tokens_per_step"))
    parser.add_argument("--token-budget", type=int, default=d("token_budget"))
    parser.add_argument(
        "--attn-impl",
        choices=[FA3_KERNEL_IMPL, "sdpa", "eager"],
        default=d("attn_impl", FA3_KERNEL_IMPL),
    )
    # A local build of the FA3 kernel. Without it, ORBIT_ATTN_KERNEL_PATH names one, and without
    # that the kernel is downloaded from the HF Hub on every rank.
    parser.add_argument("--attn-kernel-path", type=Path, default=d("attn_kernel_path"))
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default=d("dtype", "bf16"))
    # The precision of fp32 matmuls on tensor-core GPUs. "high" and "medium" run them through
    # TF32, which shortens the mantissa of the multiplied values but keeps fp32 accumulation, so
    # training with fp32 master weights stays stable. TF32 is about 7x faster than "highest",
    # which keeps exact IEEE fp32 matmuls. The setting does not touch bf16 matmuls.
    parser.add_argument(
        "--float32-matmul-precision",
        choices=["highest", "high", "medium"],
        default=d("float32_matmul_precision", "high"),
    )
    parser.add_argument("--fsdp-sharding-strategy", choices=["shard_grad_op", "full_shard", "no_shard"], default=d("fsdp_sharding_strategy", "shard_grad_op"))
    parser.add_argument(
        "--fsdp-reduce-dtype",
        choices=["auto", "bf16", "fp32"],
        default=d("fsdp_reduce_dtype", "fp32"),
        help="FSDP gradient reduction dtype. 'auto' follows the BF16 parameter dtype.",
    )
    parser.add_argument(
        "--fsdp-buffer-dtype",
        choices=["auto", "bf16", "fp32"],
        default=d("fsdp_buffer_dtype", "fp32"),
        help="FSDP floating-point buffer dtype. 'auto' follows the BF16 parameter dtype.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable activation gradient checkpointing.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable activation gradient checkpointing.",
    )
    parser.set_defaults(gradient_checkpointing=bool(d("gradient_checkpointing", True)))
    # Document masking makes attention stop at every EOS, so packed documents
    # cannot see each other. The EOS opens the next document and serves as its
    # BOS. It needs attn_impl kernels-community/vllm-flash-attn3. It is off by
    # default, which keeps plain causal attention over the whole sample.
    parser.add_argument(
        "--document-masking",
        dest="document_masking",
        action="store_true",
        help=(
            "Restrict attention to the current document, splitting samples at every EOS "
            "(needs attn_impl kernels-community/vllm-flash-attn3)."
        ),
    )
    parser.add_argument(
        "--no-document-masking",
        dest="document_masking",
        action="store_false",
        help="Use plain causal attention over the whole sample (default).",
    )
    parser.set_defaults(document_masking=bool(d("document_masking", False)))
    parser.add_argument("--log-every", type=int, default=d("log_every", 1))
    parser.add_argument("--save-every", type=int, default=d("save_every", 5000))
    parser.add_argument("--lm-loss-chunk-size", type=int, default=d("lm_loss_chunk_size", 256))
    parser.add_argument(
        "--stage1-shuffle",
        dest="stage1_shuffle",
        action="store_true",
        help="Enable deterministic StreamingDataset shuffling.",
    )
    parser.add_argument(
        "--no-stage1-shuffle",
        dest="stage1_shuffle",
        action="store_false",
        help="Disable StreamingDataset shuffling.",
    )
    parser.add_argument(
        "--strict-dataloader-determinism",
        dest="strict_dataloader_determinism",
        action="store_true",
        help="Recorded in the data-loader state of checkpoints and compared on resume.",
    )
    parser.add_argument(
        "--no-strict-dataloader-determinism",
        dest="strict_dataloader_determinism",
        action="store_false",
    )
    # ClearML logging is off unless clearml_enabled is true. Then clearml_env names a dotenv file
    # with the credentials and clearml_project the project of the task. TensorBoard always logs.
    parser.add_argument(
        "--clearml",
        dest="clearml_enabled",
        action="store_true",
        help="Log the run to ClearML.",
    )
    parser.add_argument(
        "--no-clearml",
        dest="clearml_enabled",
        action="store_false",
        help="Do not initialize ClearML and do not create a task (default).",
    )
    parser.set_defaults(clearml_enabled=bool(d("clearml_enabled", False)))
    parser.add_argument("--clearml-env", type=Path, default=d("clearml_env"))
    parser.add_argument("--clearml-project", type=str, default=d("clearml_project"))
    parser.add_argument("--clearml-task", type=str, default=None)
    parser.add_argument("--tensorboard-logdir", type=Path, default=None)
    parser.add_argument("--tensorboard-root", type=Path, default=d("tensorboard_root"))
    parser.add_argument(
        "--auto-resume",
        dest="auto_resume",
        action="store_true",
        help="If run_dir/latest.txt exists, exact-resume it instead of starting from init_from_hf.",
    )
    parser.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
        help="Disable automatic exact resume from run_dir/latest.txt.",
    )
    parser.add_argument(
        "--clearml-resume-task",
        dest="clearml_resume_task",
        action="store_true",
        help="Resume ClearML task id from run_dir/clearml_task_id.txt when available.",
    )
    parser.add_argument(
        "--no-clearml-resume-task",
        dest="clearml_resume_task",
        action="store_false",
        help="Always create a new ClearML task for this run.",
    )
    parser.add_argument(
        "--use-fused-adamw",
        dest="use_fused_adamw",
        action="store_true",
        help="Enable fused AdamW when supported by the torch build.",
    )
    parser.add_argument(
        "--no-use-fused-adamw",
        dest="use_fused_adamw",
        action="store_false",
        help="Disable fused AdamW.",
    )
    parser.set_defaults(use_fused_adamw=bool(d("use_fused_adamw", True)))
    parser.add_argument(
        "--detect-nan",
        action="store_true",
        default=bool(d("detect_nan", False)),
        help="Per-micro-step logging of the first non-finite loss / grad (with param name) to localize NaN/Inf.",
    )
    parser.set_defaults(clearml_resume_task=bool(d("clearml_resume_task", True)))
    parser.set_defaults(auto_resume=bool(d("auto_resume", False)))
    parser.set_defaults(
        stage1_shuffle=bool(d("stage1_shuffle", True)),
        strict_dataloader_determinism=bool(d("strict_dataloader_determinism", True)),
    )
    add_eval_arguments(parser, d)
    args = parser.parse_args(argv)
    check_config_choices(parser, args)
    if args.config is None:
        parser.error("--config is required.")
    args._config_keys = tuple(config_defaults.keys())
    args.config = Path(args.config).expanduser().resolve()
    args.stage1_streams = config_defaults.get("stage1_streams") or []
    args.stage1_dataset_proportions = config_defaults.get("stage1_dataset_proportions") or {}
    if not args.stage1_streams:
        parser.error("Experiment config must define stage1_streams.")
    if not args.stage1_dataset_proportions:
        parser.error("Experiment config must define stage1_dataset_proportions.")
    if args.save_dir is None:
        parser.error("Experiment config or CLI must define --save-dir.")
    if args.model_config is None:
        parser.error("Experiment config or CLI must define --model-config.")
    if args.tokenizer_config is None:
        parser.error("Experiment config or CLI must define --tokenizer-config.")
    if args.stage1_root is None:
        parser.error("Experiment config or CLI must define --stage1-root.")
    if args.lr is None:
        parser.error("Experiment config or CLI must define --lr.")
    if args.lr_warmup_steps is None:
        args.lr_warmup_steps = 0
    if args.lr_warmup_steps < 0:
        parser.error("--lr-warmup-steps must be non-negative.")
    if args.lr_warmup_start_factor is None:
        args.lr_warmup_start_factor = 0.1
    if args.lr_warmup_start_factor < 0.0:
        parser.error("--lr-warmup-start-factor must be non-negative.")
    if args.lr_min_factor is None:
        args.lr_min_factor = 0.1
    if args.lr_min_factor < 0.0:
        parser.error("--lr-min-factor must be non-negative.")
    if args.lr_min is not None and args.lr_min < 0.0:
        parser.error("--lr-min must be non-negative.")
    if args.beta is None:
        parser.error("Experiment config or CLI must define --beta.")
    if args.block_size is None:
        parser.error("Experiment config or CLI must define --block-size or seq_len.")
    if args.total_ut_steps is None:
        parser.error("Experiment config or CLI must define --total-ut-steps.")
    if args.tokens_per_step is None:
        parser.error("Experiment config or CLI must define --tokens-per-step.")
    if args.token_budget is None:
        parser.error("Experiment config or CLI must define --token-budget.")
    if args.clearml_enabled and not args.clearml_project:
        parser.error("clearml_enabled needs clearml_project.")
    validate_eval_arguments(args, parser.error)
    return args


# ---------------------------------------------------------------------------
# Process group, logging and small helpers
# ---------------------------------------------------------------------------


def setup_distributed() -> DistContext:
    use_dist = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not torch.cuda.is_available():
        raise RuntimeError("orbit requires CUDA.")
    if use_dist:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend="nccl",
                    device_id=torch.device("cuda", local_rank),
                )
            except TypeError:
                dist.init_process_group(backend="nccl")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        torch.cuda.set_device(local_rank)
    return DistContext(
        enabled=use_dist,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def rank0_print(ctx: DistContext, message: str) -> None:
    if ctx.rank == 0:
        print(message, flush=True)


def format_rank_timing_summaries(
    metric_names: list[str],
    local_values: list[float],
    ctx: DistContext,
    *,
    topk: int = 6,
) -> list[str]:
    if len(metric_names) != len(local_values):
        raise ValueError("metric_names and local_values must have the same length.")

    gathered_rows: list[list[float]]
    if ctx.enabled:
        local_tensor = torch.tensor(
            [float(x) for x in local_values],
            device=ctx.device,
            dtype=torch.float64,
        )
        gathered_tensors = [
            torch.zeros_like(local_tensor) for _ in range(ctx.world_size)
        ]
        dist.all_gather(gathered_tensors, local_tensor)
        gathered_rows = [
            [float(x) for x in tensor.detach().cpu().tolist()]
            for tensor in gathered_tensors
        ]
    else:
        gathered_rows = [[float(x) for x in local_values]]

    if ctx.rank != 0:
        return []

    summaries: list[str] = []
    world_size = len(gathered_rows)
    for metric_idx, metric_name in enumerate(metric_names):
        values = [row[metric_idx] for row in gathered_rows]
        if not values:
            continue
        min_value = min(values)
        max_value = max(values)
        mean_value = sum(values) / float(len(values))
        slowest = sorted(
            enumerate(values),
            key=lambda item: item[1],
            reverse=True,
        )[: max(1, int(topk))]
        slowest_str = " ".join(
            f"r{rank}:{value:.3f}s" for rank, value in slowest
        )
        summaries.append(
            "first_step_rank_skew "
            f"{metric_name} mean={mean_value:.3f}s "
            f"min={min_value:.3f}s max={max_value:.3f}s "
            f"slowest={slowest_str} world_size={world_size}"
        )
    return summaries


def broadcast_string(value: Optional[str], ctx: DistContext) -> Optional[str]:
    if not ctx.enabled:
        return value
    objects = [value]
    dist.broadcast_object_list(objects, src=0, device=ctx.device)
    return objects[0]


def broadcast_bool(value: bool, ctx: DistContext) -> bool:
    if not ctx.enabled:
        return bool(value)
    objects = [bool(value) if ctx.rank == 0 else None]
    dist.broadcast_object_list(objects, src=0, device=ctx.device)
    return bool(objects[0])


def build_seed_state(args: argparse.Namespace, ctx: DistContext) -> dict[str, Any]:
    return {
        "base_seed": int(args.seed),
        "rank_seed": int(args.seed) + int(ctx.rank),
        "model_init_seed": (
            None
            if args.model_init_seed is None
            else int(args.model_init_seed)
        ),
        "world_size": int(ctx.world_size),
        "seed_scheme": "base_plus_rank",
    }


def infer_stage1_num_canonical_nodes(
    args: argparse.Namespace,
    ctx: DistContext,
) -> int:
    if getattr(args, "stage1_num_canonical_nodes", None) is not None:
        return max(1, int(args.stage1_num_canonical_nodes))
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("NPROC_PER_NODE")
    try:
        ranks_per_node = (
            max(1, int(local_world_size))
            if local_world_size is not None
            else max(1, int(torch.cuda.device_count()))
        )
    except Exception:
        ranks_per_node = 1
    if not ctx.enabled:
        return 1
    if ctx.world_size % ranks_per_node != 0:
        return max(1, int(ctx.world_size))
    return max(1, int(ctx.world_size // ranks_per_node))


def resolve_stage1_shuffle_seed(args: argparse.Namespace) -> int:
    stage1_shuffle_seed = getattr(args, "stage1_shuffle_seed", None)
    if stage1_shuffle_seed is None:
        return int(args.seed)
    return int(stage1_shuffle_seed)


def stage1_shuffle_block_size(args: argparse.Namespace) -> Optional[int]:
    if args.stage1_shuffle and args.stage1_shuffle_block_size is not None:
        return int(args.stage1_shuffle_block_size)
    return None


def build_dataloader_repro_config(args: argparse.Namespace, ctx: DistContext) -> dict[str, Any]:
    """The data-loader settings that a checkpoint records and an exact resume must match."""
    stage1_dataset_proportions = {}
    for key, spec in (getattr(args, "stage1_dataset_proportions", {}) or {}).items():
        if isinstance(spec, dict):
            stage1_dataset_proportions[str(key)] = float(spec.get("proportion", 0.0))
        else:
            stage1_dataset_proportions[str(key)] = float(spec)
    return {
        "batch_size": int(args.batch_size),
        "block_size": int(args.block_size),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor) if args.num_workers > 0 else None,
        "persistent_workers": bool(args.num_workers > 0),
        "loader_timeout": float(args.loader_timeout) if args.num_workers > 0 else 0.0,
        "strict_dataloader_determinism": bool(args.strict_dataloader_determinism),
        "stage1_epoch_samples": int(args.stage1_epoch_samples),
        # The Mosaic dataset batches by args.batch_size; the key name is kept so that the
        # checkpoints of earlier runs still compare equal.
        "stage1_component_batch_size": int(args.batch_size),
        "stage1_shuffle": bool(args.stage1_shuffle),
        "stage1_shuffle_seed": int(resolve_stage1_shuffle_seed(args)),
        "stage1_shuffle_block_size": stage1_shuffle_block_size(args),
        "stage1_sampling_method": "balanced",
        "stage1_batching_method": "random",
        "stage1_mixing_mode": str(args.stage1_mixing_mode),
        "stage1_iterator_init_mode": "mosaic_streaming_dataset",
        "stage1_mixing_world_size": int(ctx.world_size),
        "stage1_partition_algo": "orig",
        "stage1_num_canonical_nodes": int(infer_stage1_num_canonical_nodes(args, ctx)),
        "stage1_streams": [
            (str(name), str(path))
            for name, path in (getattr(args, "stage1_streams", []) or [])
        ],
        "stage1_dataset_proportions": {
            key: stage1_dataset_proportions[key]
            for key in sorted(stage1_dataset_proportions)
        },
    }


def cli_option_was_explicitly_set(*option_names: str, argv: Optional[list[str]] = None) -> bool:
    argsv = list(sys.argv[1:] if argv is None else argv)
    normalized = tuple(str(name) for name in option_names)
    return any(
        any(item == name or str(item).startswith(f"{name}=") for name in normalized)
        for item in argsv
    )


def config_key_was_provided(args: argparse.Namespace, key: str) -> bool:
    return str(key) in set(getattr(args, "_config_keys", ()) or ())


def grad_accum_was_explicitly_set(argv: Optional[list[str]] = None) -> bool:
    return cli_option_was_explicitly_set("--grad-accum", argv=argv)


def validate_exact_resume_reproducibility(args: argparse.Namespace) -> None:
    if int(args.num_workers) != 0:
        raise RuntimeError(
            "Exact checkpoint reproducibility currently requires --num-workers 0. "
            "Worker prefetch / queue state is not checkpointed."
        )


def validate_resume_seed_state(
    payload: Optional[dict[str, Any]],
    *,
    args: argparse.Namespace,
    ctx: DistContext,
) -> None:
    if not isinstance(payload, dict):
        return
    seed_state = payload.get("seed_state")
    if not isinstance(seed_state, dict):
        return
    saved_base_seed = seed_state.get("base_seed")
    if saved_base_seed is not None and int(saved_base_seed) != int(args.seed):
        raise RuntimeError(
            f"Checkpoint base seed mismatch: checkpoint={saved_base_seed} runtime={args.seed}"
        )
    saved_rank_seed = seed_state.get("rank_seed")
    expected_rank_seed = int(args.seed) + int(ctx.rank)
    if saved_rank_seed is not None and int(saved_rank_seed) != expected_rank_seed:
        raise RuntimeError(
            f"Checkpoint rank seed mismatch on rank {ctx.rank}: "
            f"checkpoint={saved_rank_seed} runtime={expected_rank_seed}"
        )
    if "model_init_seed" in seed_state:
        saved_model_init_seed = seed_state.get("model_init_seed")
        runtime_model_init_seed = getattr(args, "model_init_seed", None)
        saved_normalized = (
            None
            if saved_model_init_seed is None
            else int(saved_model_init_seed)
        )
        runtime_normalized = (
            None
            if runtime_model_init_seed is None
            else int(runtime_model_init_seed)
        )
        if saved_normalized != runtime_normalized:
            raise RuntimeError(
                "Checkpoint model-init seed mismatch: "
                f"checkpoint={saved_normalized} runtime={runtime_normalized}"
            )


def prepare_run_dir(args: argparse.Namespace, ctx: DistContext) -> Path:
    if args.resume_from is not None:
        run_dir = resolve_run_dir_from_ref(args.resume_from)
        run_dir_str = str(run_dir)
    else:
        run_name = args.run_name or f"run_{uuid.uuid4().hex}"
        run_dir_str = str((args.save_dir / run_name).resolve()) if ctx.rank == 0 else None
        run_dir_str = broadcast_string(run_dir_str, ctx)
        if run_dir_str is None:
            raise RuntimeError("Failed to broadcast run dir.")
        run_dir = Path(run_dir_str)
        if ctx.rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
    if ctx.enabled:
        dist.barrier(device_ids=[ctx.local_rank])
    return Path(run_dir_str)


def apply_auto_resume_policy(args: argparse.Namespace) -> None:
    """Resumes the run's own latest checkpoint when it exists, instead of the init_from_hf start.

    A job that is restarted with the same config therefore continues where it stopped.
    """
    resume_from_cli = cli_option_was_explicitly_set("--resume-from")
    init_from_hf_cli = cli_option_was_explicitly_set("--init-from-hf")
    if (
        resume_from_cli
        and args.resume_from is not None
        and args.init_from_hf is not None
        and not init_from_hf_cli
    ):
        args.init_from_hf = None

    if args.resume_from is not None or not bool(getattr(args, "auto_resume", False)):
        return
    if args.save_dir is None or not args.run_name:
        return

    latest_ref = (Path(args.save_dir) / str(args.run_name) / "latest.txt").resolve()
    if not latest_ref.exists():
        return

    args.resume_from = latest_ref
    args.init_from_hf = None
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"auto_resume_from={latest_ref}", flush=True)


def _serialize_args_for_logging(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _json_log_normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_log_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_log_normalize(item) for item in value]
    return value


def format_json_for_log(value: Any) -> str:
    return json.dumps(
        _json_log_normalize(value),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )


def append_jsonl_trace(run_dir: Path, filename: str, record: dict[str, Any]) -> None:
    trace_path = run_dir / filename
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def append_step_trace(run_dir: Path, record: dict[str, Any]) -> None:
    append_jsonl_trace(run_dir, "step_trace.jsonl", record)


def append_checkpoint_trace(run_dir: Path, record: dict[str, Any]) -> None:
    append_jsonl_trace(run_dir, "checkpoint_trace.jsonl", record)


def tensor_sha256_cpu(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().to(device="cpu", copy=False).contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(cpu_tensor.dtype).encode("utf-8"))
    hasher.update(b"|")
    hasher.update(repr(tuple(int(x) for x in cpu_tensor.shape)).encode("utf-8"))
    hasher.update(b"|")
    try:
        hasher.update(cpu_tensor.numpy().tobytes())
    except Exception:
        hasher.update(repr(cpu_tensor.tolist()).encode("utf-8"))
    return hasher.hexdigest()


def init_clearml(
    args: argparse.Namespace,
    *,
    ctx: DistContext,
    run_dir: Path,
):
    if ctx.rank != 0:
        return None
    if not args.clearml_enabled:
        # Returning None is the same state every non-zero rank already runs in,
        # so all ClearML reporting downstream becomes a no-op.
        print("clearml=disabled (clearml_enabled=false); no ClearML task is created", flush=True)
        return None
    if Task is None:
        raise RuntimeError("ClearML is required but not installed.")
    if args.clearml_env is not None:
        if load_dotenv is None:
            raise RuntimeError("python-dotenv is required but not installed.")
        env_path = Path(args.clearml_env)
        if not env_path.exists():
            print(f"WARNING: ClearML env file not found: {env_path}", flush=True)
        loaded = load_dotenv(env_path)
        if not loaded:
            print(f"WARNING: Failed loading ClearML env from: {env_path}", flush=True)

    clearml_task_name = args.clearml_task or args.run_name or run_dir.name
    clearml_task_id_file = run_dir / "clearml_task_id.txt"
    init_sig = inspect.signature(Task.init)
    init_params = set(init_sig.parameters.keys())
    init_kwargs = {
        "project_name": args.clearml_project,
        "task_name": clearml_task_name,
    }
    if "auto_connect_frameworks" in init_params:
        init_kwargs["auto_connect_frameworks"] = False
    if "auto_connect_arg_parser" in init_params:
        init_kwargs["auto_connect_arg_parser"] = False
    if "auto_connect_streams" in init_params:
        init_kwargs["auto_connect_streams"] = {
            "stdout": True,
            "stderr": True,
            "logging": False,
        }

    clearml_task = None
    clearml_resume_init_mode = "new_task"
    resume_clearml_task_id = None
    if args.clearml_resume_task:
        resume_clearml_task_id = read_clearml_task_id(clearml_task_id_file)
    else:
        os.environ.pop("CLEARML_TASK_ID", None)

    if resume_clearml_task_id:
        try:
            resume_kwargs = dict(init_kwargs)
            if "reuse_last_task_id" in init_params:
                resume_kwargs["reuse_last_task_id"] = resume_clearml_task_id
                if "continue_last_task" in init_params:
                    resume_kwargs["continue_last_task"] = 0
                    clearml_resume_init_mode = "reuse_last_task_id+continue_last_task=0"
                else:
                    clearml_resume_init_mode = "reuse_last_task_id"
            elif "task_id" in init_params:
                resume_kwargs["task_id"] = resume_clearml_task_id
                if "continue_last_task" in init_params:
                    resume_kwargs["continue_last_task"] = 0
                    clearml_resume_init_mode = "task_id+continue_last_task=0"
                else:
                    clearml_resume_init_mode = "task_id"
            elif "continue_last_task" in init_params:
                resume_kwargs["continue_last_task"] = resume_clearml_task_id
                clearml_resume_init_mode = "continue_last_task=<task_id>"
            else:
                os.environ["CLEARML_TASK_ID"] = resume_clearml_task_id
                clearml_resume_init_mode = "env_CLEARML_TASK_ID"
            clearml_task = Task.init(**resume_kwargs)
        except Exception as exc:
            if os.environ.get("CLEARML_TASK_ID") == resume_clearml_task_id:
                os.environ.pop("CLEARML_TASK_ID", None)
            print(
                f"WARNING: ClearML resume failed for task_id={resume_clearml_task_id}: {exc!r}. Starting a new task.",
                flush=True,
            )

    if clearml_task is None:
        clearml_task = Task.init(**init_kwargs)

    if resume_clearml_task_id:
        try:
            set_initial_iteration = getattr(clearml_task, "set_initial_iteration", None)
            if callable(set_initial_iteration):
                set_initial_iteration(0)
        except Exception as exc:
            print(f"WARNING: clearml_set_initial_iteration_failed: {exc!r}", flush=True)
        print(f"clearml_resume_init_mode={clearml_resume_init_mode}", flush=True)

    clearml_logger = clearml_task.get_logger()
    clearml_task_id = getattr(clearml_task, "id", "n/a")
    if clearml_task_id != "n/a":
        write_clearml_task_id(clearml_task_id_file, str(clearml_task_id))

    try:
        clearml_task.connect(_serialize_args_for_logging(args))
    except Exception:
        pass
    try:
        clearml_task.add_tags(["orbit", PARALLELISM, clearml_task_name])
    except Exception:
        pass

    print(
        f"clearml_init project={args.clearml_project} task={clearml_task_name} task_id={clearml_task_id}",
        flush=True,
    )
    print(
        f"clearml_resume_mode={'enabled' if args.clearml_resume_task else 'disabled'}",
        flush=True,
    )
    return clearml_logger


def init_tensorboard(
    args: argparse.Namespace,
    *,
    ctx: DistContext,
    run_dir: Path,
):
    if ctx.rank != 0:
        return None
    logdir = args.tensorboard_logdir
    if logdir is None:
        if args.tensorboard_root is None:
            raise RuntimeError("tensorboard_root must be configured when tensorboard_logdir is not set.")
        logdir = Path(args.tensorboard_root) / run_dir.name
    if SummaryWriter is None:
        raise RuntimeError("TensorBoard requested, but torch.utils.tensorboard is unavailable.")
    writer = SummaryWriter(log_dir=str(logdir))
    print(f"tensorboard_logdir={logdir}", flush=True)
    return writer


def make_autocast_context(dtype_name: str):
    if dtype_name == "fp32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def first_nonfinite_named_grad(model: torch.nn.Module) -> Optional[tuple[str, str, float]]:
    """First parameter (by name) whose .grad has a NaN/Inf, with its grad abs-max.

    Scans local (per-rank / per-shard) grads; run before grad clipping so the
    original culprit is found (clipping by a NaN norm makes every grad NaN).
    """
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if not bool(torch.isfinite(grad).all()):
            kind = "nan" if bool(torch.isnan(grad).any()) else "inf"
            finite = grad[torch.isfinite(grad)]
            absmax = float(finite.abs().max()) if finite.numel() else float("nan")
            return name, kind, absmax
    return None


def first_nonfinite_named_param(model: torch.nn.Module) -> Optional[tuple[str, str]]:
    """First parameter (by name) whose value is NaN/Inf (post optimizer step check)."""
    for name, param in model.named_parameters():
        if param.is_floating_point() and not bool(torch.isfinite(param).all()):
            kind = "nan" if bool(torch.isnan(param).any()) else "inf"
            return name, kind
    return None


def local_grads_have_nonfinite(model: torch.nn.Module) -> bool:
    """Cheap local scan: True if any (per-rank/shard) grad holds a NaN/Inf."""
    for param in model.parameters():
        grad = param.grad
        if grad is not None and not bool(torch.isfinite(grad).all()):
            return True
    return False


def all_reduce_flag(flag: bool, ctx: DistContext) -> bool:
    """Collective OR of a boolean across ranks so every rank agrees to skip.

    Keeps the optimizer-step decision identical on all ranks; a divergent
    decision under FSDP would desync parameters and hang the job.
    """
    if not ctx.enabled:
        return bool(flag)
    t = torch.tensor([1 if flag else 0], device=ctx.device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(int(t.item()))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def maybe_enable_gradient_checkpointing(model: OuroForCausalLM, enabled: bool, ctx: DistContext) -> None:
    if not enabled:
        rank0_print(ctx, "gradient_checkpointing=disabled")
        return
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise RuntimeError("Model does not expose gradient_checkpointing_enable().")
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    rank0_print(ctx, "gradient_checkpointing=enabled")


def load_hf_config_overrides(hf_model_dir: Optional[Path]) -> Optional[dict[str, Any]]:
    if hf_model_dir is None:
        return None
    config_path = Path(hf_model_dir) / "config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def normalize_hf_state_key(key: str) -> str:
    if key.startswith("model.model.") or key.startswith("model.lm_head."):
        return key[len("model.") :]
    if key.startswith("module."):
        return key[len("module.") :]
    return key


def discover_hf_weight_files(hf_model_dir: Path) -> list[Path]:
    hf_model_dir = Path(hf_model_dir)
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = hf_model_dir / index_name
        if not index_path.exists():
            continue
        with index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Invalid HF weight index: {index_path}")
        filenames = sorted(set(str(name) for name in weight_map.values()))
        return [hf_model_dir / filename for filename in filenames]

    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = hf_model_dir / filename
        if path.exists():
            return [path]

    candidates = sorted(hf_model_dir.glob("model-*-of-*.safetensors"))
    if candidates:
        return candidates
    candidates = sorted(hf_model_dir.glob("pytorch_model-*-of-*.bin"))
    if candidates:
        return candidates
    raise FileNotFoundError(f"No HF model weights found under {hf_model_dir}")


def load_hf_weight_file(path: Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except Exception as exc:
            raise RuntimeError(
                f"safetensors is required to load HF safetensors shard: {path}"
            ) from exc
        return dict(load_safetensors_file(str(path), device="cpu"))

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported HF weight payload type at {path}: {type(payload).__name__}")
    return payload


def load_hf_warmstart(
    model: OuroForCausalLM,
    hf_model_dir: Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    hf_model_dir = Path(hf_model_dir)
    if not hf_model_dir.exists():
        raise FileNotFoundError(f"HF warmstart directory not found: {hf_model_dir}")

    weight_files = discover_hf_weight_files(hf_model_dir)
    expected_state = model.state_dict()
    expected_keys = set(str(key) for key in expected_state.keys())
    loaded_keys: set[str] = set()
    unexpected_keys: list[str] = []
    mismatched_shapes: list[str] = []

    for weight_file in weight_files:
        raw_state = load_hf_weight_file(weight_file)
        shard_state: dict[str, torch.Tensor] = {}
        for raw_key, value in raw_state.items():
            key = normalize_hf_state_key(str(raw_key))
            if key not in expected_keys:
                unexpected_keys.append(key)
                continue
            if not torch.is_tensor(value):
                unexpected_keys.append(key)
                continue
            expected_shape = tuple(int(x) for x in expected_state[key].shape)
            actual_shape = tuple(int(x) for x in value.shape)
            if actual_shape != expected_shape:
                mismatched_shapes.append(f"{key}: checkpoint={actual_shape} runtime={expected_shape}")
                continue
            shard_state[key] = value
            loaded_keys.add(key)
        if shard_state:
            incompatible = model.load_state_dict(shard_state, strict=False)
            load_unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
            unexpected_keys.extend(str(key) for key in load_unexpected)
        del raw_state
        del shard_state

    missing_keys = sorted(expected_keys.difference(loaded_keys))
    if strict and (missing_keys or unexpected_keys or mismatched_shapes):
        raise RuntimeError(
            "HF warmstart state mismatch: "
            f"missing_keys={missing_keys[:16]} "
            f"unexpected_keys={sorted(set(unexpected_keys))[:16]} "
            f"mismatched_shapes={mismatched_shapes[:16]}"
        )

    return {
        "hf_model_dir": str(hf_model_dir.resolve()),
        "weight_files": [str(path.name) for path in weight_files],
        "loaded_keys": int(len(loaded_keys)),
        "missing_keys": missing_keys,
        "unexpected_keys": sorted(set(unexpected_keys)),
        "mismatched_shapes": mismatched_shapes,
    }


def build_base_model(
    args: argparse.Namespace,
    *,
    tokenizer_cfg: TokenizerConfig,
    device: torch.device,
    architecture_overrides: Optional[dict[str, Any]] = None,
    init_on_cpu: bool = False,
) -> OuroForCausalLM:
    """Builds OuroForCausalLM from the model config, overlaid with the init source's config.

    architecture_overrides is the config.json of init_from_hf or the config that a resumed
    checkpoint recorded, so the model always gets the architecture of the weights it loads.
    rope_theta from the experiment config wins over the init source's value, because a context
    extension stage changes it.
    """
    model_cfg = load_model_config(args.model_config)
    config_kwargs = dict(model_cfg)
    if architecture_overrides:
        config_kwargs.update(copy.deepcopy(architecture_overrides))

    saved_vocab_size = config_kwargs.get("vocab_size")
    if saved_vocab_size is not None and int(saved_vocab_size) != int(tokenizer_cfg.vocab_size):
        raise RuntimeError(
            "Tokenizer vocab_size mismatch: "
            f"checkpoint/config={saved_vocab_size} runtime={tokenizer_cfg.vocab_size}"
        )
    config_kwargs.setdefault("hidden_size", 2048)
    config_kwargs.setdefault("intermediate_size", 5632)
    config_kwargs.setdefault("num_hidden_layers", 24)
    config_kwargs.setdefault("num_attention_heads", 16)
    config_kwargs.setdefault("num_key_value_heads", int(config_kwargs["num_attention_heads"]))
    configured_max_position = int(config_kwargs.get("max_position_embeddings") or 0)
    config_kwargs["max_position_embeddings"] = max(
        configured_max_position,
        int(args.block_size),
    )
    config_kwargs.setdefault("attention_dropout", 0.0)
    config_kwargs.setdefault("rms_norm_eps", 1e-6)
    config_kwargs.setdefault("tie_word_embeddings", False)
    config_kwargs.setdefault("use_sliding_window", False)
    config_kwargs.setdefault("sliding_window", 4096)
    config_kwargs.setdefault("max_window_layers", int(config_kwargs["num_hidden_layers"]))
    config_kwargs.setdefault("early_exit_threshold", 1.0)
    if (
        config_key_was_provided(args, "rope_theta")
        or cli_option_was_explicitly_set("--rope-theta")
        or "rope_theta" not in config_kwargs
    ):
        config_kwargs["rope_theta"] = float(args.rope_theta)
    config_kwargs["total_ut_steps"] = int(args.total_ut_steps)
    config_kwargs["use_exit_gate"] = True
    # The runtime tokenizer ids are authoritative for training, so a config.json that carries
    # other bos/eos/pad ids still loads.
    config_kwargs["vocab_size"] = int(tokenizer_cfg.vocab_size)
    config_kwargs["use_cache"] = False
    config_kwargs["pad_token_id"] = tokenizer_cfg.pad_id
    config_kwargs["bos_token_id"] = tokenizer_cfg.bos_id
    config_kwargs["eos_token_id"] = tokenizer_cfg.eos_id

    config = OuroConfig(**config_kwargs)
    requested_attn_impl = args.attn_impl
    # transformers checks the attention implementation when the model is built, and it does not
    # know the kernels FA3 name, so the model is built with eager and switched afterwards.
    init_attn_impl = (
        "eager"
        if requested_attn_impl == FA3_KERNEL_IMPL
        else requested_attn_impl
    )
    config._attn_implementation = init_attn_impl
    config._attn_kernel_repo_path = (
        str(Path(args.attn_kernel_path).expanduser().resolve())
        if args.attn_kernel_path is not None
        else None
    )
    if args.attn_impl == FA3_KERNEL_IMPL and args.dtype != "fp32":
        config.dtype = torch.bfloat16
    model = OuroForCausalLM(config)
    model.config._attn_implementation = requested_attn_impl
    model.config._attn_kernel_repo_path = config._attn_kernel_repo_path
    if init_on_cpu:
        # FSDP mixed precision casts only compute parameters. Keeping the source
        # module in FP32 preserves FP32 sharded master parameters for AdamW.
        model = model.to(dtype=torch.float32)
    else:
        if args.dtype == "bf16":
            model = model.to(dtype=torch.bfloat16)
        model = model.to(device)
    return model


def build_optimizer_param_groups(
    model: torch.nn.Module,
    *,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Weight decay on the attention and MLP projection weights only.

    Embeddings, the LM head, norms and the exit gate get no weight decay.
    """
    decay_parameter_suffixes = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    )

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen_param_ids: set[int] = set()

    for param_name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen_param_ids:
            continue
        seen_param_ids.add(id(param))
        if param_name.endswith(decay_parameter_suffixes):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    expected_params = [param for param in model.parameters() if param.requires_grad]
    expected_param_ids = {id(param) for param in expected_params}
    if seen_param_ids != expected_param_ids:
        missing_count = len(expected_param_ids.difference(seen_param_ids))
        extra_count = len(seen_param_ids.difference(expected_param_ids))
        raise RuntimeError(
            "Failed to classify every trainable parameter for weight decay: "
            f"missing={missing_count} extra={extra_count}"
        )

    if not decay_params or not no_decay_params:
        raise RuntimeError(
            "Expected non-empty decay and no-decay optimizer parameter groups, got "
            f"decay={len(decay_params)} no_decay={len(no_decay_params)}"
        )

    groups = [
        {
            "params": decay_params,
            "weight_decay": float(weight_decay),
            "group_name": "decay",
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "group_name": "no_decay",
        },
    ]
    summary = {
        "decay_tensors": len(decay_params),
        "decay_numel": sum(int(param.numel()) for param in decay_params),
        "no_decay_tensors": len(no_decay_params),
        "no_decay_numel": sum(int(param.numel()) for param in no_decay_params),
    }
    return groups, summary


def create_optimizer(
    args: argparse.Namespace,
    model: torch.nn.Module,
    *,
    param_groups: list[dict[str, Any]],
) -> torch.optim.Optimizer:
    grouped_param_ids = {
        id(param)
        for group in param_groups
        for param in group["params"]
        if param.requires_grad
    }
    model_param_ids = {
        id(param)
        for param in model.parameters()
        if param.requires_grad
    }
    if grouped_param_ids != model_param_ids:
        fsdp_use_orig_params = get_fsdp_use_orig_params(model)
        raise RuntimeError(
            "Optimizer parameter groups no longer match the wrapped model parameters. "
            "FSDP weight-decay grouping requires use_orig_params=True; "
            f"fsdp_use_orig_params={fsdp_use_orig_params!r}."
        )

    adamw_kwargs = {
        "lr": args.lr,
        "betas": (float(args.adam_beta1), float(args.adam_beta2)),
        "foreach": False,
    }
    if args.use_fused_adamw and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        adamw_kwargs["fused"] = True
    return torch.optim.AdamW(param_groups, **adamw_kwargs)


def lr_warmup_start_step_is_auto(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "auto"}


def lr_cosine_steps_is_auto(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "auto"}


def parse_lr_warmup_start_step(value: Any, *, fallback: int = 0) -> int:
    if lr_warmup_start_step_is_auto(value):
        return int(fallback)
    return int(value)


def parse_lr_cosine_steps(value: Any) -> Optional[int]:
    if lr_cosine_steps_is_auto(value):
        return None
    return max(1, int(value))


def create_lr_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    *,
    warmup_start_step: int = 0,
) -> WarmupLRScheduler:
    return WarmupLRScheduler(
        optimizer,
        scheduler_type=args.lr_scheduler,
        base_lr=float(args.lr),
        warmup_steps=int(args.lr_warmup_steps or 0),
        warmup_start_lr=args.lr_warmup_start_lr,
        warmup_start_factor=float(args.lr_warmup_start_factor),
        warmup_start_step=int(warmup_start_step),
        cosine_steps=parse_lr_cosine_steps(args.lr_cosine_steps),
        min_lr=args.lr_min,
        min_lr_factor=float(args.lr_min_factor),
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_train_loader(
    args: argparse.Namespace,
    *,
    ctx: DistContext,
    tokenizer_cfg: TokenizerConfig,
) -> tuple[DataLoader, Any, int]:
    """The training DataLoader over one Mosaic StreamingDataset that mixes every config stream.

    Returns the loader, the dataset (whose position the checkpoints record) and the number of
    batches of one epoch on this rank.
    """
    if clean_stale_shared_memory is not None:
        clean_stale_shared_memory()
    stage1_partition_algo = "orig"
    stage1_num_canonical_nodes = infer_stage1_num_canonical_nodes(args, ctx)
    stage1_shuffle_seed = resolve_stage1_shuffle_seed(args)
    shuffle_block_size = stage1_shuffle_block_size(args)
    if ctx.rank == 0:
        print(
            "streaming_resume_mode "
            f"shuffle={int(bool(args.stage1_shuffle))} "
            f"shuffle_seed={stage1_shuffle_seed} "
            f"shuffle_block_size={shuffle_block_size} "
            "sampling_method=balanced "
            "batching_method=random "
            f"mixing_mode={args.stage1_mixing_mode} "
            "iterator_init_mode=mosaic_streaming_dataset "
            f"mixing_world_size={ctx.world_size} "
            f"partition_algo={stage1_partition_algo} "
            f"num_canonical_nodes={stage1_num_canonical_nodes} "
            f"dataset_batch_size={int(args.batch_size)}",
            flush=True,
        )
    stage1_root = Path(args.stage1_root)
    stage1_entries = [(key, path) for key, path in args.stage1_streams]
    streams, _, _ = build_stage1_mixture_dataset(
        stage1_root,
        stage1_entries,
        token_budget=int(args.token_budget),
        total_samples=args.stage1_epoch_samples,
        dataset_specs=args.stage1_dataset_proportions,
        log_fn=print if ctx.rank == 0 else None,
    )
    stage1_dataset = build_streaming_mixture_dataset(
        streams,
        seed=stage1_shuffle_seed,
        shuffle=bool(args.stage1_shuffle),
        shuffle_block_size=shuffle_block_size,
        batch_size=int(args.batch_size),
        predownload=args.predownload,
        cache_limit=args.cache_limit,
        # Mosaic derives the epoch size from the streams. Passing a target epoch size here
        # would make it allocate shuffle arrays for billions of sample ids.
        epoch_size=None,
        partition_algo=stage1_partition_algo,
        num_canonical_nodes=stage1_num_canonical_nodes,
    )
    # StreamingDataset.length is already the rank-local epoch length.
    global_epoch_samples = int(getattr(stage1_dataset, "epoch_size", 0) or 0)
    local_cycle_samples = max(1, int(len(stage1_dataset)))
    if ctx.rank == 0:
        print(
            "streaming_mosaic_random "
            f"streams={len(streams)} "
            f"global_epoch_samples={global_epoch_samples} "
            f"local_samples={local_cycle_samples}",
            flush=True,
        )
    token_dtype_code = (
        decode_sample_dtype_code(args.stage1_token_dtype)
        if getattr(args, "stage1_token_dtype", None)
        else None
    )
    if token_dtype_code is not None:
        rank0_print(
            ctx,
            f"stage1_token_dtype_override={args.stage1_token_dtype} "
            f"(code={token_dtype_code}; ignoring per-shard dtype label)",
        )
    token_dataset = MdsTokenIterableDataset(
        stage1_dataset,
        args.block_size,
        tokenizer_cfg.pad_id,
        tokenizer_cfg.eos_id,
        token_dtype_code=token_dtype_code,
        require_loss_mask=bool(getattr(args, "labels_premasked", False)),
    )
    loader = DataLoader(
        token_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        timeout=args.loader_timeout if args.num_workers > 0 else 0.0,
        worker_init_fn=None,
    )
    steps_per_epoch = int(local_cycle_samples // args.batch_size)
    return loader, token_dataset, steps_per_epoch


def compute_tokens_per_micro(args: argparse.Namespace, ctx: DistContext) -> int:
    data_parallel_partitions = ctx.world_size if ctx.enabled else 1
    return int(args.batch_size * args.block_size * data_parallel_partitions)


def determine_grad_accum(args: argparse.Namespace, tokens_per_micro: int) -> int:
    if args.grad_accum is not None:
        return max(1, int(args.grad_accum))
    return max(1, int(math.ceil(float(args.tokens_per_step) / float(tokens_per_micro))))


def determine_max_steps(
    args: argparse.Namespace,
    *,
    tokens_per_step: int,
    resumed_state: Optional[ResumeState],
) -> int:
    if args.max_steps is not None:
        return int(args.max_steps)
    if resumed_state is not None and resumed_state.max_steps is not None:
        return int(resumed_state.max_steps)
    return max(1, int(math.ceil(float(int(args.token_budget)) / float(tokens_per_step))))


def restore_local_dataset_state_from_payload(
    dataset: Any,
    *,
    payload: Optional[dict[str, Any]],
    ctx: DistContext,
    expected_loader_config: Optional[dict[str, Any]],
) -> bool:
    if not isinstance(payload, dict):
        return False
    wrapped = {"states": [None] * ctx.world_size}
    wrapped["states"][ctx.rank] = payload
    return restore_local_dataloader_state(
        dataset,
        wrapped,
        rank=ctx.rank,
        world_size=ctx.world_size,
        loader_active=True,
        expected_loader_config=expected_loader_config,
    )


# ---------------------------------------------------------------------------
# Phases of main()
# ---------------------------------------------------------------------------


@dataclass
class RunServices:
    """The run folder and the services that live for the whole run."""

    run_dir: Path
    clearml_logger: Any
    tb_writer: Any
    in_training_eval: Any
    loader_repro_config: dict[str, Any]


@dataclass
class TrainingData:
    tokenizer_cfg: TokenizerConfig
    loader: DataLoader
    dataset: Any
    steps_per_epoch: int
    tokens_per_micro: int
    grad_accum: int
    tokens_per_step: int


@dataclass
class LoopProgress:
    """Counters and timings that the step loop carries from one batch to the next."""

    last_log_time: float
    local_sample_counter: int
    first_loader_iter_init_time: Optional[float] = None
    first_batch_next_time: Optional[float] = None
    first_step_diagnostics_emitted: bool = False
    nonfinite_batch_dumped: bool = False


@dataclass
class TrainingLoop:
    """Everything the step loop and its checkpoint and logging hooks use."""

    args: argparse.Namespace
    ctx: DistContext
    run_dir: Path
    clearml_logger: Any
    tb_writer: Any
    loader_repro_config: dict[str, Any]
    tokenizer_cfg: TokenizerConfig
    loader: DataLoader
    dataset: Any
    wrapped_model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: WarmupLRScheduler
    state: ResumeState
    max_steps: int
    tokens_per_step: int
    stop_controller: GracefulStopController
    norm_diag_model: Any
    progress: LoopProgress


@dataclass
class StepAccumulator:
    """What one optimizer step collects over its grad_accum microbatches."""

    step_started: float
    step_log_due: bool
    step_load_time: float = 0.0
    step_forward_time: float = 0.0
    step_backward_time: float = 0.0
    step_optimizer_time: float = 0.0
    accum_metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics_micro_count: int = 0
    loss_lm_final_rank0: Optional[float] = None
    step_effective_tokens_local: int = 0
    batch_probe_hasher: Any = None
    batch_probe_input_prefix: Optional[list[int]] = None
    batch_probe_label_prefix: Optional[list[int]] = None


def resolve_init_source(args: argparse.Namespace) -> None:
    """Chooses between resuming the run and starting from init_from_hf (or random weights)."""
    apply_auto_resume_policy(args)
    if args.attn_kernel_path is not None and args.attn_impl != FA3_KERNEL_IMPL:
        raise RuntimeError(
            f"--attn-kernel-path is only supported with --attn-impl {FA3_KERNEL_IMPL}"
        )
    if args.resume_from is not None and args.init_from_hf is not None:
        raise RuntimeError("--resume-from, --init-from-hf are mutually exclusive.")


def configure_matmul_precision(args: argparse.Namespace, ctx: DistContext) -> None:
    # The comment at --float32-matmul-precision in parse_args explains the choice. The setting
    # applies to the whole process.
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    allow_tf32 = args.float32_matmul_precision != "highest"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    rank0_print(
        ctx,
        f"float32_matmul_precision={args.float32_matmul_precision} "
        f"allow_tf32={allow_tf32}",
    )


def start_run(args: argparse.Namespace, ctx: DistContext) -> RunServices:
    """Creates the run folder, checks the attention kernel, starts logging and the evaluator, seeds."""
    run_dir = prepare_run_dir(args, ctx)
    args.run_name = args.run_name or run_dir.name
    args.clearml_task = args.clearml_task or args.run_name
    validate_exact_resume_reproducibility(args)

    preflight_flash_attention(
        args.attn_impl,
        attn_kernel_path=args.attn_kernel_path,
    )

    rank0_print(ctx, f"orbit parallelism={PARALLELISM} world_size={ctx.world_size}")
    rank0_print(ctx, f"save_run_dir={run_dir}")
    if args.attn_impl == FA3_KERNEL_IMPL:
        rank0_print(ctx, f"flash-attn available: using {args.attn_impl} backend.")
    if args.attn_kernel_path is not None:
        rank0_print(
            ctx,
            f"attn_kernel_path={Path(args.attn_kernel_path).expanduser().resolve()}",
        )
    clearml_logger = init_clearml(args, ctx=ctx, run_dir=run_dir)
    tb_writer = init_tensorboard(args, ctx=ctx, run_dir=run_dir)
    # None unless eval_every, eval_at_start or eval_at_end is set. Local rank 0 of every node
    # checks the evaluation environment here, so a broken one stops the run before the model is
    # built.
    in_training_eval = build_in_training_eval(
        args,
        rank=ctx.rank,
        local_rank=ctx.local_rank,
        world_size=ctx.world_size,
        distributed=ctx.enabled,
        device=ctx.device,
        run_dir=run_dir,
    )
    set_global_random_seeds(args.seed, ctx.rank)
    loader_repro_config = build_dataloader_repro_config(args, ctx)
    rank0_print(ctx, f"seed_base={int(args.seed)} seed_rank0={int(args.seed)}")
    if ctx.rank == 0:
        print(
            "exact_resume_contract "
            f"num_workers={args.num_workers} "
            f"batch_size={args.batch_size} "
            f"block_size={args.block_size}",
            flush=True,
        )
    return RunServices(
        run_dir=run_dir,
        clearml_logger=clearml_logger,
        tb_writer=tb_writer,
        in_training_eval=in_training_eval,
        loader_repro_config=loader_repro_config,
    )


def build_training_data(args: argparse.Namespace, ctx: DistContext) -> TrainingData:
    tokenizer_cfg = TokenizerConfig(args.tokenizer_config)
    loader, dataset, steps_per_epoch = build_train_loader(args, ctx=ctx, tokenizer_cfg=tokenizer_cfg)

    tokens_per_micro = compute_tokens_per_micro(args, ctx)
    grad_accum = determine_grad_accum(args, tokens_per_micro)
    tokens_per_step = tokens_per_micro * grad_accum
    rank0_print(
        ctx,
        f"tokens_per_micro={format_count(tokens_per_micro)} grad_accum={grad_accum} "
        f"tokens_per_step={format_count(tokens_per_step)} lm_loss_chunk_size={args.lm_loss_chunk_size}",
    )
    return TrainingData(
        tokenizer_cfg=tokenizer_cfg,
        loader=loader,
        dataset=dataset,
        steps_per_epoch=steps_per_epoch,
        tokens_per_micro=tokens_per_micro,
        grad_accum=grad_accum,
        tokens_per_step=tokens_per_step,
    )


def print_architecture_overrides(ctx: DistContext, label: str, overrides: Optional[dict[str, Any]]) -> None:
    if ctx.rank != 0 or overrides is None:
        return
    print(
        f"{label} "
        f"hidden={overrides.get('hidden_size')} "
        f"layers={overrides.get('num_hidden_layers')} "
        f"heads={overrides.get('num_attention_heads')} "
        f"kv_heads={overrides.get('num_key_value_heads')} "
        f"intermediate={overrides.get('intermediate_size')} "
        f"max_pos={overrides.get('max_position_embeddings')}",
        flush=True,
    )


def enable_document_masking(args: argparse.Namespace, base_model: OuroForCausalLM, tokenizer_cfg: TokenizerConfig) -> None:
    # Only the FA3 kernel path has a varlen implementation. The model would
    # also refuse at its first forward pass, but checking here stops the run
    # before the weights are loaded and wrapped for FSDP.
    if args.attn_impl != FA3_KERNEL_IMPL:
        raise RuntimeError(
            f"document_masking requires attn_impl {FA3_KERNEL_IMPL}, "
            f"got {args.attn_impl!r}."
        )
    if tokenizer_cfg.eos_id is None:
        raise RuntimeError("document_masking needs an EOS id in the tokenizer config.")
    # The tokenizer config's EOS is used as the document boundary. Chat-style
    # tokenizer configs set EOS to the turn terminator <|im_end|>, which also
    # ends every user turn inside one conversation. Masking on it would stop
    # the answer from seeing its own prompt, so such a config is refused.
    if tokenizer_cfg.eos_token == "<|im_end|>":
        raise RuntimeError(
            "document_masking needs a tokenizer config whose eos_token ends a whole "
            "document (<|endoftext|>), not the chat turn terminator <|im_end|>."
        )
    base_model.model.set_document_masking(int(tokenizer_cfg.eos_id))


def build_training_model(
    args: argparse.Namespace,
    ctx: DistContext,
    tokenizer_cfg: TokenizerConfig,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Builds the model with the architecture of its init source, loads init_from_hf, wraps it in FSDP.

    Returns the wrapped training module and the weight-decay parameter groups of the optimizer.
    """
    resume_architecture_overrides: Optional[dict[str, Any]] = None
    if args.resume_from is not None:
        resume_architecture_overrides = infer_resume_model_config_overrides(
            args.resume_from,
            rank=ctx.rank,
        )
        print_architecture_overrides(ctx, "resume_architecture_overrides", resume_architecture_overrides)

    hf_init_architecture_overrides: Optional[dict[str, Any]] = None
    if args.init_from_hf is not None:
        hf_init_architecture_overrides = load_hf_config_overrides(args.init_from_hf)
        print_architecture_overrides(ctx, "hf_init_architecture_overrides", hf_init_architecture_overrides)

    raw_model_config = load_model_config(args.model_config)
    rank0_print(ctx, f"model_config_path={Path(args.model_config).resolve()}")
    rank0_print(ctx, "model_config_raw=\n" + format_json_for_log(raw_model_config))

    if args.model_init_seed is not None:
        torch.manual_seed(int(args.model_init_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.model_init_seed))
        rank0_print(ctx, f"model_init_seed={int(args.model_init_seed)}")

    # With several processes the model is built on the CPU in fp32 and FSDP moves its shards to
    # the GPU. A single process builds it directly on its GPU in the training dtype.
    init_model_on_cpu = bool(ctx.enabled)
    if init_model_on_cpu:
        rank0_print(ctx, "model_init_device=cpu_for_fsdp")
    base_model = build_base_model(
        args,
        tokenizer_cfg=tokenizer_cfg,
        device=ctx.device,
        architecture_overrides=(
            resume_architecture_overrides
            or hf_init_architecture_overrides
        ),
        init_on_cpu=init_model_on_cpu,
    )
    rank0_print(ctx, "model_config_final=\n" + format_json_for_log(base_model.config.to_dict()))
    if args.document_masking:
        enable_document_masking(args, base_model, tokenizer_cfg)
    rank0_print(
        ctx,
        f"document_masking={int(bool(args.document_masking))} "
        f"boundary_token_id={base_model.model.document_masking_token_id}",
    )
    # Only rank 0 reads the export; FSDP's sync_module_states copies its weights to the others.
    if args.init_from_hf is not None and (not ctx.enabled or ctx.rank == 0):
        hf_started = time.monotonic()
        hf_load_info = load_hf_warmstart(
            base_model,
            args.init_from_hf,
            strict=True,
        )
        rank0_print(
            ctx,
            "hf_model_warmstart "
            f"source={hf_load_info['hf_model_dir']} "
            f"shards={len(hf_load_info['weight_files'])} "
            f"loaded_keys={hf_load_info['loaded_keys']} "
            f"time={time.monotonic() - hf_started:.3f}s",
        )
    maybe_enable_gradient_checkpointing(base_model, args.gradient_checkpointing, ctx)
    if args.labels_premasked:
        rank0_print(
            ctx,
            "labels_premasked=1 (positional loss_mask from shards; "
            "pad-value label masking disabled so supervised eos targets survive)",
        )
    train_module = OrbitLMTrainingModule(
        base_model,
        beta=args.beta,
        exit_mode=EXIT_MODE,
        # None records that premasked streams already carry -100 on prompt and padding
        # positions, so real <|im_end|> targets in responses stay supervised (pad == eos).
        pad_token_id=None if args.labels_premasked else tokenizer_cfg.pad_id,
        lm_chunk_size=args.lm_loss_chunk_size,
    )
    optimizer_param_groups, optimizer_group_summary = build_optimizer_param_groups(
        train_module,
        weight_decay=args.weight_decay,
    )
    rank0_print(
        ctx,
        "optimizer_param_groups "
        "weight_decay_mode=groups "
        f"decay_tensors={optimizer_group_summary['decay_tensors']} "
        f"decay_numel={optimizer_group_summary['decay_numel']} "
        f"decay_weight_decay={args.weight_decay} "
        f"no_decay_tensors={optimizer_group_summary['no_decay_tensors']} "
        f"no_decay_numel={optimizer_group_summary['no_decay_numel']} "
        "no_decay_weight_decay=0.0",
    )
    rank0_print(ctx, f"exit_mode={EXIT_MODE} use_exit_gate=1")
    if not init_model_on_cpu:
        train_module = train_module.to(ctx.device)

    # A new run (random init or init_from_hf) copies rank 0's weights to every rank. A resumed
    # run loads each rank's own shard from its checkpoint.
    sync_module_states = bool(ctx.enabled and args.resume_from is None)
    rank0_print(ctx, f"sync_module_states={int(sync_module_states)}")
    rank0_print(
        ctx,
        "fsdp_mixed_precision "
        f"param_dtype={args.dtype} "
        f"reduce_dtype={args.fsdp_reduce_dtype} "
        f"buffer_dtype={args.fsdp_buffer_dtype}",
    )
    rank0_print(
        ctx,
        f"model_parameter_dtype_before_wrap={next(train_module.parameters()).dtype}",
    )

    wrapped_model = build_parallel_model(
        train_module,
        local_rank=ctx.local_rank,
        bf16=(args.dtype == "bf16"),
        sync_module_states=sync_module_states,
        fsdp_sharding_strategy=args.fsdp_sharding_strategy,
        fsdp_reduce_dtype=args.fsdp_reduce_dtype,
        fsdp_buffer_dtype=args.fsdp_buffer_dtype,
    )
    if is_fsdp_model(wrapped_model):
        fsdp_use_orig_params_after_wrap = get_fsdp_use_orig_params(wrapped_model)
        if not fsdp_use_orig_params_after_wrap:
            raise RuntimeError(
                "Correct optimizer weight-decay grouping requires "
                "FSDP(use_orig_params=True)."
            )
        trainable_param_dtypes = {
            param.dtype
            for param in wrapped_model.parameters()
            if param.requires_grad and param.is_floating_point()
        }
        if trainable_param_dtypes != {torch.float32}:
            raise RuntimeError(
                "FSDP master parameters must remain FP32 outside forward/backward, "
                f"got dtypes={sorted(str(dtype) for dtype in trainable_param_dtypes)}"
            )
        rank0_print(ctx, "fsdp_master_parameter_dtype=fp32")
    return wrapped_model, optimizer_param_groups


def build_optimizer_and_scheduler(
    args: argparse.Namespace,
    ctx: DistContext,
    wrapped_model: nn.Module,
    optimizer_param_groups: list[dict[str, Any]],
) -> tuple[torch.optim.Optimizer, WarmupLRScheduler]:
    optimizer = create_optimizer(
        args,
        wrapped_model,
        param_groups=optimizer_param_groups,
    )
    if getattr(args, "detect_nan", False):
        rank0_print(ctx, "detect_nan=enabled (per-micro non-finite loss/grad logging with param names)")
    scheduler = create_lr_scheduler(
        args,
        optimizer,
        warmup_start_step=(
            0
            if lr_warmup_start_step_is_auto(args.lr_warmup_start_step)
            else parse_lr_warmup_start_step(args.lr_warmup_start_step)
        ),
    )
    return optimizer, scheduler


def restore_checkpoint(
    args: argparse.Namespace,
    ctx: DistContext,
    data: TrainingData,
    run: RunServices,
    wrapped_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupLRScheduler,
) -> tuple[Optional[ResumeState], bool]:
    """Loads the resume checkpoint into the model, optimizer, scheduler, data position and RNG.

    Returns the restored training state (None for a new run) and whether the scheduler state came
    from the checkpoint. A resume keeps the checkpoint's grad_accum, which updates data.
    """
    if args.resume_from is None:
        return None, False
    resume_load_timings: dict[str, float] = {}
    payload = load_rank_checkpoint(
        args.resume_from,
        model=wrapped_model,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=ctx.rank,
        world_size=ctx.world_size,
        timings_out=resume_load_timings,
    )
    scheduler_loaded_from_checkpoint = isinstance(payload, dict) and "scheduler" in payload
    if ctx.rank == 0 and resume_load_timings:
        rank0_print(
            ctx,
            "resume_timing "
            f"checkpoint_read_time={resume_load_timings.get('checkpoint_read_time_s', 0.0):.3f}s "
            f"state_apply_time={resume_load_timings.get('state_apply_time_s', 0.0):.3f}s "
            f"resume_total_time={resume_load_timings.get('resume_total_time_s', 0.0):.3f}s",
        )
    validate_resume_seed_state(payload, args=args, ctx=ctx)
    resume_state = ResumeState.from_payload(payload.get("training_state"))
    saved_rng_state = payload.get("rng_state")
    restored = restore_local_dataset_state_from_payload(
        data.dataset,
        payload=payload.get("dataloader_state"),
        ctx=ctx,
        expected_loader_config=run.loader_repro_config,
    )
    if not restored:
        raise RuntimeError(
            "Exact resume requires successful dataloader state restore. "
            "Checkpoint payload did not restore the local dataset state."
        )
    # Restore RNG after dataloader state so any internal dataset bookkeeping
    # during load_state_dict() cannot perturb the resumed random stream.
    restore_rng_state(saved_rng_state)
    dataloader_state_payload = payload.get("dataloader_state")
    if isinstance(dataloader_state_payload, dict):
        capture_mode = dataloader_state_payload.get("capture_mode")
        version = dataloader_state_payload.get("version")
        if capture_mode is not None or version is not None:
            rank0_print(
                ctx,
                "resume_dataloader_state "
                f"version={version} capture_mode={capture_mode}",
            )
    rank0_print(
        ctx,
        f"resume_from={args.resume_from} restored_dataloader_state={restored}",
    )
    saved_grad_accum = int(resume_state.grad_accum)
    if grad_accum_was_explicitly_set():
        if int(args.grad_accum) != saved_grad_accum:
            raise RuntimeError(
                "Exact resume requires the same grad_accum as the checkpoint: "
                f"checkpoint={saved_grad_accum} runtime={int(args.grad_accum)}"
            )
    else:
        data.grad_accum = saved_grad_accum
        args.grad_accum = saved_grad_accum
    data.tokens_per_step = data.tokens_per_micro * data.grad_accum
    rank0_print(ctx, f"resume_grad_accum={saved_grad_accum}")
    return resume_state, scheduler_loaded_from_checkpoint


def plan_training_length(
    args: argparse.Namespace,
    ctx: DistContext,
    data: TrainingData,
    resume_state: Optional[ResumeState],
    scheduler: WarmupLRScheduler,
    scheduler_loaded_from_checkpoint: bool,
) -> tuple[ResumeState, int]:
    """Fixes max_steps and the number of epochs, and points the LR schedule at them."""
    max_steps = determine_max_steps(
        args,
        tokens_per_step=data.tokens_per_step,
        resumed_state=resume_state,
    )
    # steps_per_epoch counts dataloader batches. global_step advances once
    # per grad_accum batches, and a trailing partial accumulation does not
    # count. Comparing max_steps to the batch count schedules a single epoch
    # whenever grad_accum > 1 and the batch count lies between max_steps and
    # max_steps * grad_accum.
    optimizer_steps_per_epoch = max(1, int(data.steps_per_epoch) // max(1, int(data.grad_accum)))
    args.epochs = max(
        int(args.epochs),
        int(math.ceil(float(max_steps) / float(optimizer_steps_per_epoch))),
    )
    rank0_print(
        ctx,
        f"max_steps={max_steps} epochs={args.epochs} "
        f"steps_per_epoch={data.steps_per_epoch} optimizer_steps_per_epoch={optimizer_steps_per_epoch}",
    )

    state = resume_state or ResumeState()
    state.grad_accum = data.grad_accum
    state.max_steps = max_steps

    if lr_warmup_start_step_is_auto(args.lr_warmup_start_step) and not scheduler_loaded_from_checkpoint:
        scheduler.set_warmup_start_step(state.global_step)
    if (
        args.lr_scheduler == "warmup_cosine"
        and lr_cosine_steps_is_auto(args.lr_cosine_steps)
        and not scheduler_loaded_from_checkpoint
    ):
        scheduler.set_cosine_steps(
            max(
                1,
                int(max_steps)
                - int(scheduler.warmup_start_step)
                - int(scheduler.warmup_steps),
            )
        )
    current_lr = scheduler.step(state.global_step)
    rank0_print(
        ctx,
        "lr_scheduler={} base_lr={} current_lr={} warmup_steps={} "
        "warmup_start_lr={} warmup_start_step={} cosine_steps={} min_lr={}".format(
            args.lr_scheduler,
            float(args.lr),
            float(current_lr),
            int(args.lr_warmup_steps or 0),
            float(scheduler.warmup_start_lr),
            int(scheduler.warmup_start_step),
            scheduler.cosine_steps,
            float(scheduler.min_lr),
        ),
    )
    rank0_print(
        ctx,
        "loss_logging=global_rank_mean "
        "rank0_diagnostics=loss_total_rank0,loss_lm_rank0,loss_exit_rank0,"
        "loss_lm_final "
        "loss_lm_final_definition=rank0_current_batch_final_ut_ce_all_shifted_targets",
    )
    rank0_print(
        ctx,
        "loss_objective "
        f"training={EXIT_MODE} "
        "clearml_loss_lm=optimized_lm_component "
        "clearml_loss_lm_final=rank0_final_ut_ce "
        "clearml_loss_total=optimized_total",
    )
    return state, max_steps


def start_in_training_eval(loop: TrainingLoop, in_training_eval: Any) -> None:
    """Hands the model to the in-training evaluation and runs its evaluation before step 1."""
    if in_training_eval is not None:
        in_training_eval.attach(
            model=loop.wrapped_model,
            clearml_logger=loop.clearml_logger,
            tb_writer=loop.tb_writer,
            stop_requested=loop.stop_controller.stop_requested,
        )
        # A run resumed at its last step catches up on that step's evaluation here when a crash
        # interrupted it, because the training loop below does not run again.
        in_training_eval.maybe_run(
            loop.state.global_step, at_start=True, last=loop.state.global_step >= loop.max_steps
        )


# ---------------------------------------------------------------------------
# The step loop
# ---------------------------------------------------------------------------


def begin_optimizer_step(loop: TrainingLoop) -> StepAccumulator:
    """Starts the accumulator of a new optimizer step and decides whether it is logged."""
    ctx, args, state = loop.ctx, loop.args, loop.state
    step_started = time.monotonic()
    if ctx.rank == 0:
        step_log_due = (
            ((state.global_step + 1) % args.log_every == 0)
            or ((time.monotonic() - loop.progress.last_log_time) >= 30.0)
        )
    else:
        step_log_due = False
    step_log_due = broadcast_bool(step_log_due, ctx)
    if loop.norm_diag_model is not None:
        loop.norm_diag_model.set_norm_diagnostics(bool(step_log_due))
        if step_log_due:
            loop.norm_diag_model.reset_norm_diagnostics()
    return StepAccumulator(
        step_started=step_started,
        step_log_due=step_log_due,
        batch_probe_hasher=hashlib.sha256() if ctx.rank == 0 else None,
    )


def dump_nonfinite_batch(
    loop: TrainingLoop,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    first_param: str,
    kind: str,
) -> None:
    """Saves the microbatch that first gave a non-finite gradient on this rank."""
    ctx, state = loop.ctx, loop.state
    dump_dir = Path(loop.args.dump_nonfinite_batch)
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = (
        dump_dir
        / f"nonfinite_batch_rank{ctx.rank}"
        f"_step{state.global_step}"
        f"_micro{state.micro_in_step}.pt"
    )
    torch.save(
        {
            "input_ids": input_ids.detach().cpu(),
            "labels": labels.detach().cpu(),
            "rank": int(ctx.rank),
            "global_step": int(state.global_step),
            "micro_in_step": int(state.micro_in_step),
            "first_param": first_param,
            "kind": kind,
        },
        dump_path,
    )
    loop.progress.nonfinite_batch_dumped = True
    print(
        f"NONFINITE_BATCH_DUMPED rank={ctx.rank} path={dump_path}",
        flush=True,
    )


def run_micro_step(
    loop: TrainingLoop,
    acc: StepAccumulator,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    load_time: float,
) -> bool:
    """Runs forward and backward on one microbatch and adds its metrics to the step.

    Returns True when this microbatch completes the step's gradient accumulation.
    """
    ctx, args, state = loop.ctx, loop.args, loop.state
    wrapped_model = loop.wrapped_model
    acc.step_load_time += load_time
    if acc.batch_probe_hasher is not None:
        acc.batch_probe_hasher.update(tensor_sha256_cpu(input_ids).encode("utf-8"))
        acc.batch_probe_hasher.update(b"|")
        acc.batch_probe_hasher.update(tensor_sha256_cpu(labels).encode("utf-8"))
        if acc.batch_probe_input_prefix is None and input_ids.numel() > 0:
            acc.batch_probe_input_prefix = (
                input_ids[0, : min(8, int(input_ids.shape[1]))].detach().cpu().tolist()
            )
        if acc.batch_probe_label_prefix is None and labels.numel() > 0:
            acc.batch_probe_label_prefix = (
                labels[0, : min(8, int(labels.shape[1]))].detach().cpu().tolist()
            )
    input_ids = input_ids.to(ctx.device, non_blocking=True)
    labels = labels.to(ctx.device, non_blocking=True)
    if loop.tokenizer_cfg.pad_id is None:
        acc.step_effective_tokens_local += int(labels.numel())
    else:
        acc.step_effective_tokens_local += int(labels.ne(loop.tokenizer_cfg.pad_id).sum().item())
    loop.progress.local_sample_counter += int(input_ids.shape[0])

    step_will_happen = (state.micro_in_step + 1) >= state.grad_accum
    if ctx.rank == 0:
        micro_log_due = (
            ((state.micro_step + 1) % args.log_every == 0)
            or ((time.monotonic() - loop.progress.last_log_time) >= 30.0)
        )
    else:
        micro_log_due = False
    micro_log_due = broadcast_bool(micro_log_due, ctx)

    sync_ctx = maybe_no_sync(
        wrapped_model,
        enabled=ctx.enabled and not step_will_happen,
    )
    with sync_ctx:
        forward_started = time.monotonic()
        with make_autocast_context(args.dtype):
            metrics = wrapped_model(
                input_ids=input_ids,
                labels=labels,
                need_diagnostics=bool(acc.step_log_due or micro_log_due),
            )
            loss = metrics["loss"] / state.grad_accum
        acc.step_forward_time += time.monotonic() - forward_started
        if args.detect_nan and not bool(torch.isfinite(loss.detach())):
            lm = metrics.get("loss_lm")
            ex = metrics.get("loss_exit")
            print(
                f"NONFINITE_LOSS rank={ctx.rank} global_step={state.global_step} "
                f"micro_in_step={state.micro_in_step} "
                f"loss={float(loss.detach().float().item()):.6e} "
                f"loss_lm={float(lm) if torch.is_tensor(lm) else lm} "
                f"loss_exit={float(ex) if torch.is_tensor(ex) else ex} "
                "(forward already non-finite -> look in the forward/model, not backward)",
                flush=True,
            )
        backward_started = time.monotonic()
        loss.backward()
        acc.step_backward_time += time.monotonic() - backward_started
        if args.detect_nan:
            bad_grad = first_nonfinite_named_grad(wrapped_model)
            if bad_grad is not None:
                gname, gkind, gabsmax = bad_grad
                print(
                    f"NONFINITE_GRAD rank={ctx.rank} global_step={state.global_step} "
                    f"micro_in_step={state.micro_in_step} first_param={gname} kind={gkind} "
                    f"finite_grad_absmax={gabsmax:.3e}",
                    flush=True,
                )
                if args.dump_nonfinite_batch and not loop.progress.nonfinite_batch_dumped:
                    dump_nonfinite_batch(loop, input_ids, labels, gname, gkind)

    if ctx.rank == 0:
        final_loop_metric = metrics.get("loss_lm_final")
        if torch.is_tensor(final_loop_metric):
            acc.loss_lm_final_rank0 = float(
                final_loop_metric.detach().float().item()
            )
    for key, value in metrics.items():
        if key in {"loss", "loss_lm_final"}:
            continue
        if torch.is_tensor(value):
            acc.accum_metrics[key] = acc.accum_metrics.get(key, 0.0) + value.detach()
    if acc.step_log_due:
        acc.diagnostics_micro_count += 1

    state.micro_step += 1
    state.micro_in_step += 1
    if (
        ctx.rank == 0
        and micro_log_due
        and acc.loss_lm_final_rank0 is not None
    ):
        report_scalar(
            loop.clearml_logger,
            loop.tb_writer,
            "loss_lm_final",
            "micro",
            acc.loss_lm_final_rank0,
            state.micro_step,
        )
        loop.progress.last_log_time = time.monotonic()
    return step_will_happen


def apply_optimizer_step(loop: TrainingLoop, acc: StepAccumulator) -> Optional[float]:
    """Clips the gradients and steps the optimizer, or skips the step on a non-finite gradient.

    Returns the total gradient norm before clipping.
    """
    ctx, args, state = loop.ctx, loop.args, loop.state
    wrapped_model, optimizer = loop.wrapped_model, loop.optimizer
    optimizer_started = time.monotonic()
    total_grad_norm = clip_grad_norm(wrapped_model, args.max_grad_norm)
    total_grad_norm_value = (
        None
        if total_grad_norm is None
        else float(
            total_grad_norm.detach().float().item()
            if torch.is_tensor(total_grad_norm)
            else total_grad_norm
        )
    )
    # There is no loss scaler in bf16, so a single non-finite gradient would be written
    # straight into the weights by optimizer.step(). The step is skipped instead.
    #
    # FSDP.clip_grad_norm_ returns the norm over all ranks, so every rank sees the same value.
    # The flag is still all-reduced, because with clipping off (max_grad_norm <= 0) each rank
    # checks only its own gradients.
    grad_is_nonfinite = (
        total_grad_norm_value is not None
        and not math.isfinite(total_grad_norm_value)
    )
    if total_grad_norm_value is None:
        grad_is_nonfinite = local_grads_have_nonfinite(wrapped_model)
    grad_is_nonfinite = all_reduce_flag(grad_is_nonfinite, ctx)

    if grad_is_nonfinite:
        state.nonfinite_grad_skips += 1
        # This line is printed whether or not detect_nan is on. Rank 0 names the first
        # non-finite parameter so the source can be traced.
        bad = (
            first_nonfinite_named_grad(wrapped_model)
            if ctx.rank == 0
            else None
        )
        first_param = bad[0] if bad is not None else "n/a"
        probe_hash = (
            acc.batch_probe_hasher.hexdigest()[:12]
            if acc.batch_probe_hasher is not None
            else "n/a"
        )
        print(
            f"SKIP_OPTIMIZER_STEP rank={ctx.rank} "
            f"global_step={state.global_step} "
            f"micro_in_step={state.micro_in_step} "
            f"total_grad_norm={total_grad_norm_value} "
            f"first_nonfinite_param={first_param} "
            f"batch_probe_hash={probe_hash} "
            f"skips_total={state.nonfinite_grad_skips}/"
            f"{args.max_nonfinite_grad_skips} "
            "(optimizer step SKIPPED, grads zeroed, weights preserved)",
            flush=True,
        )
        optimizer.zero_grad(set_to_none=True)
        if (
            args.max_nonfinite_grad_skips >= 0
            and state.nonfinite_grad_skips > args.max_nonfinite_grad_skips
        ):
            raise RuntimeError(
                f"Aborting: {state.nonfinite_grad_skips} non-finite gradient "
                f"steps exceeded --max-nonfinite-grad-skips="
                f"{args.max_nonfinite_grad_skips}. Find the numerical source first: "
                "detect_nan with dump_nonfinite_batch saves the failing batch."
            )
    else:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if args.detect_nan:
            bad_param = first_nonfinite_named_param(wrapped_model)
            if bad_param is not None:
                pname, pkind = bad_param
                print(
                    f"NONFINITE_PARAM rank={ctx.rank} "
                    f"global_step={state.global_step} "
                    f"first_param={pname} kind={pkind} "
                    "(weights corrupted after optimizer step)",
                    flush=True,
                )
    acc.step_optimizer_time += time.monotonic() - optimizer_started
    return total_grad_norm_value


def reduce_exit_token_counts(loop: TrainingLoop, acc: StepAccumulator) -> dict[str, torch.Tensor]:
    exit_tok_counts_step_by_tag: dict[str, torch.Tensor] = {}
    for exit_threshold in EXIT_TOKEN_COUNT_THRESHOLDS:
        tag = threshold_tag(exit_threshold)
        metric_key = f"exit_tok_count_{tag}"
        if metric_key not in acc.accum_metrics:
            continue
        counts_tensor = acc.accum_metrics[metric_key].detach().clone().float()
        if loop.ctx.enabled:
            dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        exit_tok_counts_step_by_tag[tag] = counts_tensor
    return exit_tok_counts_step_by_tag


def reduce_logged_metrics(loop: TrainingLoop, acc: StepAccumulator) -> dict[str, torch.Tensor]:
    """Averages the logged metrics over ranks and returns rank 0's own loss values first."""
    ctx = loop.ctx
    rank0_loss_metric_snapshots: dict[str, torch.Tensor] = {}
    if ctx.rank == 0 and acc.step_log_due:
        for metric_key in ("loss_total", "loss_lm", "loss_exit"):
            metric_value = acc.accum_metrics.get(metric_key)
            if torch.is_tensor(metric_value):
                rank0_loss_metric_snapshots[metric_key] = (
                    metric_value.detach().clone()
                )
    if acc.step_log_due and ctx.enabled:
        for metric_key in (
            "loss_total",
            "loss_lm",
            "loss_exit",
            "loss_lm_ut",
            "gate_lambda",
            "exit_prob",
            "exit_entropy",
        ):
            metric_value = acc.accum_metrics.get(metric_key)
            if not torch.is_tensor(metric_value):
                continue
            reduced_value = metric_value.detach().clone().float()
            dist.all_reduce(reduced_value, op=dist.ReduceOp.SUM)
            acc.accum_metrics[metric_key] = reduced_value / float(ctx.world_size)
    return rank0_loss_metric_snapshots


def log_optimizer_step(
    loop: TrainingLoop,
    acc: StepAccumulator,
    *,
    epoch_idx: int,
    step_time: float,
    total_grad_norm_value: Optional[float],
    scheduled_tokens_step: int,
    effective_tokens_step: int,
    exit_tok_counts_step_by_tag: dict[str, torch.Tensor],
    rank0_loss_metric_snapshots: dict[str, torch.Tensor],
    norm_diag_values: dict[str, float],
) -> None:
    """Prints the step line, reports the scalars and appends the step trace (rank 0 only)."""
    ctx, state = loop.ctx, loop.state
    clearml_logger, tb_writer = loop.clearml_logger, loop.tb_writer
    accum_metrics = acc.accum_metrics
    loss_lm_final_rank0 = acc.loss_lm_final_rank0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    loss_total = float((accum_metrics.get("loss_total") / state.grad_accum).item())
    loss_lm = float((accum_metrics.get("loss_lm") / state.grad_accum).item())
    loss_exit = float((accum_metrics.get("loss_exit") / state.grad_accum).item())
    loss_total_rank0 = None
    loss_lm_rank0 = None
    loss_exit_rank0 = None
    if "loss_total" in rank0_loss_metric_snapshots:
        loss_total_rank0 = float(
            (
                rank0_loss_metric_snapshots["loss_total"]
                / state.grad_accum
            ).item()
        )
    if "loss_lm" in rank0_loss_metric_snapshots:
        loss_lm_rank0 = float(
            (
                rank0_loss_metric_snapshots["loss_lm"]
                / state.grad_accum
            ).item()
        )
    if "loss_exit" in rank0_loss_metric_snapshots:
        loss_exit_rank0 = float(
            (
                rank0_loss_metric_snapshots["loss_exit"]
                / state.grad_accum
            ).item()
        )
    diagnostics_divisor = float(max(1, acc.diagnostics_micro_count))
    per_ut_lm_losses = None
    if "loss_lm_ut" in accum_metrics:
        per_ut_lm_losses = (
            (accum_metrics["loss_lm_ut"] / diagnostics_divisor).detach().float().cpu().tolist()
        )
    gate_lambda_means = None
    if "gate_lambda" in accum_metrics:
        gate_lambda_means = (
            (accum_metrics["gate_lambda"] / diagnostics_divisor).detach().float().cpu().tolist()
        )
    exit_probs_mean = None
    if "exit_prob" in accum_metrics:
        exit_probs_mean = (
            (accum_metrics["exit_prob"] / diagnostics_divisor).detach().float().cpu().tolist()
        )
    throughput_tok_s = float(effective_tokens_step) / max(step_time, 1e-12)
    throughput_tok_gpu_s = throughput_tok_s / max(1, ctx.world_size)
    throughput_sched_tok_s = float(scheduled_tokens_step) / max(step_time, 1e-12)
    throughput_sched_tok_gpu_s = throughput_sched_tok_s / max(1, ctx.world_size)
    current_lr = float(loop.optimizer.param_groups[0]["lr"])
    batch_probe_hash = (
        acc.batch_probe_hasher.hexdigest()
        if acc.batch_probe_hasher is not None
        else None
    )
    message = (
        f"{timestamp}\n"
        f"epoch={epoch_idx} step={state.global_step} loss={loss_total:.4f} "
        f"loss_lm={loss_lm:.4f} exit_loss={loss_exit:.4f} "
        f"load_time={acc.step_load_time:.3f}s step_time={step_time:.3f}s "
        f"scheduled_tokens={scheduled_tokens_step} "
        f"effective_tokens={effective_tokens_step} "
        f"throughput_tok_s={throughput_tok_s:.1f} "
        f"throughput_tok_gpu_s={throughput_tok_gpu_s:.1f} "
        f"throughput_sched_tok_s={throughput_sched_tok_s:.1f} "
        f"throughput_sched_tok_gpu_s={throughput_sched_tok_gpu_s:.1f} "
        f"lr={current_lr:.8g}"
    )
    if loss_lm_final_rank0 is not None:
        message += (
            f" loss_lm_final="
            f"{loss_lm_final_rank0:.4f}"
        )
    if total_grad_norm_value is not None:
        message += f" grad_norm={total_grad_norm_value:.4f}"
    if batch_probe_hash is not None:
        message += f" batch_probe_hash={batch_probe_hash[:12]}"
    if "exit_entropy" in accum_metrics:
        exit_entropy = float((accum_metrics["exit_entropy"] / diagnostics_divisor).item())
        message += f" exit_entropy={exit_entropy:.4f}"
    exit_tok_counts_lists_by_tag: dict[str, list[float]] = {}
    exit_tok_percent_lists_by_tag: dict[str, list[float]] = {}
    for exit_threshold in EXIT_TOKEN_COUNT_THRESHOLDS:
        tag = threshold_tag(exit_threshold)
        counts_tensor = exit_tok_counts_step_by_tag.get(tag)
        if counts_tensor is None:
            continue
        counts_list = [float(x) for x in counts_tensor.cpu().tolist()]
        denom = max(float(effective_tokens_step), 1.0)
        percents_list = [(count / denom) * 100.0 for count in counts_list]
        exit_tok_counts_lists_by_tag[tag] = counts_list
        exit_tok_percent_lists_by_tag[tag] = percents_list
        message += f" exit@{exit_threshold:.1f}=" + " ".join(
            f"t{idx}:{int(round(count))}({percent:.1f}%)"
            for idx, (count, percent) in enumerate(zip(counts_list, percents_list))
        )
    print(message, flush=True)

    step = state.global_step
    scalars: list[tuple[str, str, Any]] = [
        ("loss_total", "step", loss_total),
        ("loss_lm", "step", loss_lm),
        ("loss_exit", "step", loss_exit),
    ]
    if loss_lm_final_rank0 is not None:
        scalars.append(("loss_lm_final", "step", loss_lm_final_rank0))
    if total_grad_norm_value is not None:
        scalars.append(("grad_norm", "total", total_grad_norm_value))
    for diag_name, diag_value in norm_diag_values.items():
        if "/" in diag_name:
            diag_title, diag_series = diag_name.split("/", 1)
        else:
            diag_title, diag_series = diag_name, "step"
        scalars.append((diag_title, diag_series, float(diag_value)))
    if ctx.enabled and loss_total_rank0 is not None:
        scalars.append(("loss_total_rank0", "step", loss_total_rank0))
    if ctx.enabled and loss_lm_rank0 is not None:
        scalars.append(("loss_lm_rank0", "step", loss_lm_rank0))
    if ctx.enabled and loss_exit_rank0 is not None:
        scalars.append(("loss_exit_rank0", "step", loss_exit_rank0))
    scalars += [
        ("timing", "load_time_step", acc.step_load_time),
        ("timing", "step_time", step_time),
        ("throughput", "tok_s", throughput_tok_s),
        ("throughput", "tok_gpu_s", throughput_tok_gpu_s),
        ("throughput", "sched_tok_s", throughput_sched_tok_s),
        ("throughput", "sched_tok_gpu_s", throughput_sched_tok_gpu_s),
        ("optim", "grad_accum", state.grad_accum),
        ("optim", "lr", loop.optimizer.param_groups[0]["lr"]),
        ("tokens", "effective_step", effective_tokens_step),
        ("tokens", "scheduled_step", scheduled_tokens_step),
    ]
    if "exit_entropy" in accum_metrics:
        scalars.append(("exit_entropy", "step", exit_entropy))
    if per_ut_lm_losses is not None:
        scalars += [("loss_lm_ut", f"ut{i}_step", v) for i, v in enumerate(per_ut_lm_losses)]
    if gate_lambda_means is not None:
        scalars += [("gate_lambda", f"ut{i}_step", v) for i, v in enumerate(gate_lambda_means)]
    if exit_probs_mean is not None:
        scalars += [("exit_prob", f"ut{i}_step", v) for i, v in enumerate(exit_probs_mean)]
    for tag, counts_list in exit_tok_counts_lists_by_tag.items():
        percents_list = exit_tok_percent_lists_by_tag[tag]
        scalars += [("tokens", f"exit_{tag}_t{i}_step", count) for i, count in enumerate(counts_list)]
        scalars += [("tokens", f"exit_{tag}_t{i}_percent", percent) for i, percent in enumerate(percents_list)]
    for title, series, value in scalars:
        report_scalar(clearml_logger, tb_writer, title, series, value, step)

    trace_record: dict[str, Any] = {
        "epoch": int(epoch_idx),
        "global_step": int(state.global_step),
        "grad_accum": int(state.grad_accum),
        "loss_total": float(loss_total),
        "loss_lm": float(loss_lm),
        "loss_exit": float(loss_exit),
        "exit_entropy": (
            None if "exit_entropy" not in accum_metrics else float(exit_entropy)
        ),
        "load_time_step": float(acc.step_load_time),
        "step_time": float(step_time),
        "scheduled_tokens_step": int(scheduled_tokens_step),
        "effective_tokens_step": int(effective_tokens_step),
        "throughput_tok_s": float(throughput_tok_s),
        "throughput_tok_gpu_s": float(throughput_tok_gpu_s),
        "throughput_sched_tok_s": float(throughput_sched_tok_s),
        "throughput_sched_tok_gpu_s": float(throughput_sched_tok_gpu_s),
        "lr": float(current_lr),
    }
    if loss_lm_final_rank0 is not None:
        trace_record["loss_lm_final"] = float(
            loss_lm_final_rank0
        )
    if total_grad_norm_value is not None:
        trace_record["grad_norm_total"] = float(total_grad_norm_value)
    if ctx.enabled:
        if loss_total_rank0 is not None:
            trace_record["loss_total_rank0"] = float(loss_total_rank0)
        if loss_lm_rank0 is not None:
            trace_record["loss_lm_rank0"] = float(loss_lm_rank0)
        if loss_exit_rank0 is not None:
            trace_record["loss_exit_rank0"] = float(loss_exit_rank0)
    if batch_probe_hash is not None:
        trace_record["batch_probe_hash"] = str(batch_probe_hash)
    if acc.batch_probe_input_prefix is not None:
        trace_record["batch_probe_input_prefix"] = [
            int(x) for x in acc.batch_probe_input_prefix
        ]
    if acc.batch_probe_label_prefix is not None:
        trace_record["batch_probe_label_prefix"] = [
            int(x) for x in acc.batch_probe_label_prefix
        ]
    if per_ut_lm_losses is not None:
        for ut_idx, ut_loss in enumerate(per_ut_lm_losses):
            trace_record[f"loss_lm_ut_t{ut_idx}"] = float(ut_loss)
    if gate_lambda_means is not None:
        for ut_idx, ut_lambda in enumerate(gate_lambda_means):
            trace_record[f"gate_lambda_t{ut_idx}"] = float(ut_lambda)
    if exit_probs_mean is not None:
        for ut_idx, ut_prob in enumerate(exit_probs_mean):
            trace_record[f"exit_prob_t{ut_idx}"] = float(ut_prob)
    if exit_tok_counts_lists_by_tag:
        trace_record["exit_token_threshold_primary"] = float(
            PRIMARY_EXIT_TOKEN_COUNT_THRESHOLD
        )
        trace_record["exit_token_thresholds"] = [
            float(thr) for thr in EXIT_TOKEN_COUNT_THRESHOLDS
        ]
        for tag, counts_list in exit_tok_counts_lists_by_tag.items():
            percents_list = exit_tok_percent_lists_by_tag[tag]
            for step_idx, count in enumerate(counts_list):
                trace_record[f"exit_{tag}_tok_count_t{step_idx}"] = float(count)
            for step_idx, percent in enumerate(percents_list):
                trace_record[f"exit_{tag}_tok_percent_t{step_idx}"] = float(percent)
    append_step_trace(
        loop.run_dir,
        trace_record,
    )
    loop.progress.last_log_time = time.monotonic()


CHECKPOINT_TIMING_KEYS = (
    "state_export_time_s",
    "payload_build_time_s",
    "serialize_time_s",
    "replace_time_s",
    "save_rank_total_time_s",
)


def save_training_checkpoint(loop: TrainingLoop, epoch_idx: int) -> None:
    """Writes every rank's checkpoint of the current step, then meta.json and latest.txt."""
    ctx, args, state = loop.ctx, loop.args, loop.state
    checkpoint_started = time.monotonic()
    checkpoint_dir = checkpoint_dir_for_step(loop.run_dir, state.global_step)
    dataloader_state_started = time.monotonic()
    local_dataloader_state = build_local_dataloader_state(
        loop.dataset,
        samples_seen=state.local_samples_seen,
        rank=ctx.rank,
        world_size=ctx.world_size,
        loader_active=True,
        loader_config=loop.loader_repro_config,
    )
    dataloader_state_time = time.monotonic() - dataloader_state_started
    checkpoint_save_timings: dict[str, float] = {}
    save_rank_checkpoint(
        checkpoint_dir,
        model=loop.wrapped_model,
        optimizer=loop.optimizer,
        scheduler=loop.scheduler,
        rank=ctx.rank,
        world_size=ctx.world_size,
        parallelism=PARALLELISM,
        training_state=state.to_dict(),
        rng_state=capture_rng_state(),
        seed_state=build_seed_state(args, ctx),
        dataloader_state=local_dataloader_state,
        timings_out=checkpoint_save_timings,
    )
    barrier_after_save_time = 0.0
    if ctx.enabled:
        barrier_after_save_started = time.monotonic()
        dist.barrier(device_ids=[ctx.local_rank])
        barrier_after_save_time = time.monotonic() - barrier_after_save_started
    checkpoint_metadata_time = 0.0
    if ctx.rank == 0:
        meta_extra = build_checkpoint_metadata_extra(loop.wrapped_model)
        checkpoint_metadata_started = time.monotonic()
        write_checkpoint_metadata(
            checkpoint_dir,
            parallelism=PARALLELISM,
            world_size=ctx.world_size,
            training_state=state.to_dict(),
            extra=meta_extra,
        )
        write_latest_pointer(loop.run_dir, checkpoint_dir)
        checkpoint_metadata_time = time.monotonic() - checkpoint_metadata_started
    barrier_after_meta_time = 0.0
    if ctx.enabled:
        barrier_after_meta_started = time.monotonic()
        dist.barrier(device_ids=[ctx.local_rank])
        barrier_after_meta_time = time.monotonic() - barrier_after_meta_started
    checkpoint_total_time = time.monotonic() - checkpoint_started
    if ctx.rank != 0:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rank_checkpoint_bytes = checkpoint_save_timings.get("rank_checkpoint_bytes", 0.0)
    rank_checkpoint_gib = float(rank_checkpoint_bytes) / float(1024 ** 3)
    checkpoint_message = (
        f"{timestamp}\n"
        f"checkpoint step={state.global_step} "
        f"dataloader_state_time={dataloader_state_time:.3f}s "
        f"state_export_time={checkpoint_save_timings.get('state_export_time_s', 0.0):.3f}s "
        f"payload_build_time={checkpoint_save_timings.get('payload_build_time_s', 0.0):.3f}s "
        f"serialize_time={checkpoint_save_timings.get('serialize_time_s', 0.0):.3f}s "
        f"replace_time={checkpoint_save_timings.get('replace_time_s', 0.0):.3f}s "
        f"save_rank_total_time={checkpoint_save_timings.get('save_rank_total_time_s', 0.0):.3f}s "
        f"barrier_after_save_time={barrier_after_save_time:.3f}s "
        f"metadata_time={checkpoint_metadata_time:.3f}s "
        f"barrier_after_meta_time={barrier_after_meta_time:.3f}s "
        f"checkpoint_total_time={checkpoint_total_time:.3f}s "
        f"rank_checkpoint_gib={rank_checkpoint_gib:.3f}"
    )
    print(checkpoint_message, flush=True)
    append_checkpoint_trace(
        loop.run_dir,
        {
            "epoch": int(epoch_idx),
            "global_step": int(state.global_step),
            "dataloader_state_time_s": float(dataloader_state_time),
            "barrier_after_save_time_s": float(barrier_after_save_time),
            "metadata_time_s": float(checkpoint_metadata_time),
            "barrier_after_meta_time_s": float(barrier_after_meta_time),
            "checkpoint_total_time_s": float(checkpoint_total_time),
            **{
                key: float(value)
                for key, value in checkpoint_save_timings.items()
            },
        },
    )
    step = state.global_step
    scalars: list[tuple[str, str, Any]] = [("timing", "checkpoint_dataloader_state_s", dataloader_state_time)]
    for metric_name in CHECKPOINT_TIMING_KEYS:
        metric_value = checkpoint_save_timings.get(metric_name)
        if metric_value is not None:
            scalars.append(("timing", f"checkpoint_{metric_name}", metric_value))
    scalars += [
        ("timing", "checkpoint_barrier_after_save_s", barrier_after_save_time),
        ("timing", "checkpoint_metadata_s", checkpoint_metadata_time),
        ("timing", "checkpoint_barrier_after_meta_s", barrier_after_meta_time),
        ("timing", "checkpoint_total_s", checkpoint_total_time),
        ("storage", "checkpoint_rank_bytes", rank_checkpoint_bytes),
    ]
    for title, series, value in scalars:
        report_scalar(loop.clearml_logger, loop.tb_writer, title, series, value, step)


def finish_optimizer_step(
    loop: TrainingLoop,
    acc: StepAccumulator,
    *,
    epoch_idx: int,
    total_grad_norm_value: Optional[float],
) -> None:
    """Advances the step counters and the LR, then logs the step when it is due."""
    ctx, state = loop.ctx, loop.state
    progress = loop.progress
    norm_diag_values: dict[str, float] = {}
    if acc.step_log_due and loop.norm_diag_model is not None:
        norm_diag_values = loop.norm_diag_model.pop_norm_diagnostics()

    state.global_step += 1
    loop.scheduler.step(state.global_step)
    state.micro_in_step = 0
    state.local_samples_seen = int(progress.local_sample_counter)
    state.tokens_seen += int(loop.tokens_per_step)

    step_time = time.monotonic() - acc.step_started
    scheduled_tokens_step = int(loop.tokens_per_step)
    effective_tokens_step = int(acc.step_effective_tokens_local)
    if ctx.enabled:
        effective_tokens_tensor = torch.tensor(
            [effective_tokens_step],
            device=ctx.device,
            dtype=torch.long,
        )
        dist.all_reduce(effective_tokens_tensor, op=dist.ReduceOp.SUM)
        effective_tokens_step = int(effective_tokens_tensor.item())
    if not progress.first_step_diagnostics_emitted:
        for summary_line in format_rank_timing_summaries(
            [
                "loader_iter_init",
                "first_batch_next",
                "step_load_total",
                "step_forward_total",
                "step_backward_total",
                "step_optimizer_total",
                "step_total",
            ],
            [
                float(progress.first_loader_iter_init_time or 0.0),
                float(progress.first_batch_next_time or 0.0),
                float(acc.step_load_time),
                float(acc.step_forward_time),
                float(acc.step_backward_time),
                float(acc.step_optimizer_time),
                float(step_time),
            ],
            ctx,
        ):
            rank0_print(ctx, summary_line)
        progress.first_step_diagnostics_emitted = True
    exit_tok_counts_step_by_tag = reduce_exit_token_counts(loop, acc)
    rank0_loss_metric_snapshots = reduce_logged_metrics(loop, acc)
    if ctx.rank == 0 and acc.step_log_due:
        log_optimizer_step(
            loop,
            acc,
            epoch_idx=epoch_idx,
            step_time=step_time,
            total_grad_norm_value=total_grad_norm_value,
            scheduled_tokens_step=scheduled_tokens_step,
            effective_tokens_step=effective_tokens_step,
            exit_tok_counts_step_by_tag=exit_tok_counts_step_by_tag,
            rank0_loss_metric_snapshots=rank0_loss_metric_snapshots,
            norm_diag_values=norm_diag_values,
        )


def run_training_steps(loop: TrainingLoop, in_training_eval: Any) -> None:
    """Trains until max_steps or a stop signal, saving checkpoints and running the evaluation."""
    args, state, progress = loop.args, loop.state, loop.progress
    stop_controller = loop.stop_controller
    acc: Optional[StepAccumulator] = None
    for epoch_idx in range(state.epoch, args.epochs):
        if state.global_step >= loop.max_steps:
            break
        iterator_init_started = time.monotonic()
        iterator = iter(loop.loader)
        iterator_init_time = time.monotonic() - iterator_init_started
        if progress.first_loader_iter_init_time is None:
            progress.first_loader_iter_init_time = float(iterator_init_time)
        while state.global_step < loop.max_steps:
            load_started = time.monotonic()
            try:
                input_ids, labels = next(iterator)
            except StopIteration:
                break
            load_time = time.monotonic() - load_started
            if progress.first_batch_next_time is None:
                progress.first_batch_next_time = float(load_time)

            if stop_controller.stop_requested():
                break

            if state.micro_in_step == 0:
                acc = begin_optimizer_step(loop)

            step_will_happen = run_micro_step(loop, acc, input_ids, labels, load_time)
            if not step_will_happen:
                continue

            total_grad_norm_value = apply_optimizer_step(loop, acc)
            finish_optimizer_step(
                loop,
                acc,
                epoch_idx=epoch_idx,
                total_grad_norm_value=total_grad_norm_value,
            )

            if (
                state.global_step % args.save_every == 0
                or state.global_step >= loop.max_steps
                or stop_controller.stop_requested()
            ):
                save_training_checkpoint(loop, epoch_idx)

            if in_training_eval is not None:
                # Runs after the checkpoint of this step (eval_every is a multiple of
                # save_every), so a crash or a stop during the evaluation loses no training.
                in_training_eval.maybe_run(state.global_step, last=state.global_step >= loop.max_steps)

            if stop_controller.stop_requested():
                break

        state.epoch = epoch_idx + 1
        if stop_controller.stop_requested():
            break


def finish_run(loop: TrainingLoop, start_global_step: int) -> None:
    """Closes the writers and the process group, then reports a run that ended too early."""
    state = loop.state
    early_exit_error: Optional[RuntimeError] = None
    if not loop.stop_controller.stop_requested() and state.global_step < loop.max_steps:
        completed_steps = int(state.global_step) - int(start_global_step)
        if completed_steps <= 0:
            early_exit_error = RuntimeError(
                "Training exited before the first optimizer step. "
                "The rank-local dataloader yielded no batches before max_steps was reached. "
                "Under torchrun this usually points to rank-local StreamingDataset partitioning "
                "or first-batch loader startup behavior, not model initialization."
            )
        else:
            early_exit_error = RuntimeError(
                "Training ended before reaching max_steps without a stop signal: "
                f"global_step={state.global_step} max_steps={loop.max_steps} "
                f"completed_steps_this_invocation={completed_steps}"
            )

    loop.stop_controller.restore()
    if loop.tb_writer is not None and loop.ctx.rank == 0:
        loop.tb_writer.flush()
        loop.tb_writer.close()
    if distributed_enabled():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()
    if early_exit_error is not None:
        raise early_exit_error


def main() -> None:
    args = parse_args()
    resolve_init_source(args)
    ctx = setup_distributed()
    configure_matmul_precision(args, ctx)
    run = start_run(args, ctx)
    data = build_training_data(args, ctx)
    wrapped_model, optimizer_param_groups = build_training_model(args, ctx, data.tokenizer_cfg)
    optimizer, scheduler = build_optimizer_and_scheduler(args, ctx, wrapped_model, optimizer_param_groups)
    resume_state, scheduler_loaded_from_checkpoint = restore_checkpoint(
        args, ctx, data, run, wrapped_model, optimizer, scheduler
    )
    state, max_steps = plan_training_length(
        args, ctx, data, resume_state, scheduler, scheduler_loaded_from_checkpoint
    )

    stop_controller = GracefulStopController(ctx.rank)
    stop_controller.install()
    # The decoder collects per-layer norm diagnostics on the steps that are logged.
    norm_diag_model = getattr(
        getattr(unwrap_model(wrapped_model), "model", None), "model", None
    )
    if not hasattr(norm_diag_model, "set_norm_diagnostics"):
        norm_diag_model = None
    loop = TrainingLoop(
        args=args,
        ctx=ctx,
        run_dir=run.run_dir,
        clearml_logger=run.clearml_logger,
        tb_writer=run.tb_writer,
        loader_repro_config=run.loader_repro_config,
        tokenizer_cfg=data.tokenizer_cfg,
        loader=data.loader,
        dataset=data.dataset,
        wrapped_model=wrapped_model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        max_steps=max_steps,
        tokens_per_step=data.tokens_per_step,
        stop_controller=stop_controller,
        norm_diag_model=norm_diag_model,
        progress=LoopProgress(
            last_log_time=time.monotonic(),
            local_sample_counter=int(state.local_samples_seen),
        ),
    )
    start_in_training_eval(loop, run.in_training_eval)
    optimizer.zero_grad(set_to_none=True)
    # The 30-second log cadence counts from here, after the evaluation before step 1.
    loop.progress.last_log_time = time.monotonic()
    start_global_step = int(state.global_step)
    run_training_steps(loop, run.in_training_eval)
    finish_run(loop, start_global_step)


if __name__ == "__main__":
    main()
