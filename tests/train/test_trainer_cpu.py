"""CPU checks of the trainer's building blocks on a tiny model: loss, LR schedule, optimizer
groups, checkpoint round trip and the HF export.

The GPU paths (FA3 attention, document masking, FSDP) are covered by test_document_masking.py
and the smoke configs in configs/train/smoke/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from orbit import export as orbit_export  # noqa: E402
from orbit import train  # noqa: E402
from orbit.checkpointing import (  # noqa: E402
    build_checkpoint_metadata_extra,
    capture_rng_state,
    checkpoint_dir_for_step,
    load_rank_checkpoint,
    save_rank_checkpoint,
    write_checkpoint_metadata,
    write_latest_pointer,
)
from orbit.configuration_ouro import OuroConfig  # noqa: E402
from orbit.modelling_ouro import OuroForCausalLM  # noqa: E402

VOCAB = 64


def tiny_config(**overrides) -> OuroConfig:
    values = dict(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
        rope_theta=1.0e6, total_ut_steps=4, early_exit_threshold=1.0, pad_token_id=0, eos_token_id=0,
    )
    values.update(overrides)
    config = OuroConfig(**values)
    config._attn_implementation = "eager"
    return config


def tiny_module(seed: int = 0) -> train.OrbitLMTrainingModule:
    torch.manual_seed(seed)
    model = OuroForCausalLM(tiny_config())
    return train.OrbitLMTrainingModule(model, beta=0.15, exit_mode="learned", pad_token_id=0, lm_chunk_size=8)


def batch(seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, VOCAB, (2, 17), generator=generator)
    labels = ids[:, 1:].clone()
    labels[0, -3:] = -100
    return ids[:, :-1], labels


def test_the_loss_is_the_exit_weighted_lm_loss_minus_beta_times_the_entropy():
    module = tiny_module()
    input_ids, labels = batch()
    metrics = module(input_ids=input_ids, labels=labels, need_diagnostics=True)
    assert torch.isfinite(metrics["loss"])
    assert torch.allclose(metrics["loss"].detach(), metrics["loss_lm"] + metrics["loss_exit"])
    assert torch.allclose(metrics["loss_exit"], -0.15 * metrics["exit_entropy"])
    assert metrics["loss_lm_ut"].shape == (4,)
    assert metrics["exit_prob"].shape == (4,)
    assert torch.allclose(metrics["exit_prob"].sum(), torch.tensor(1.0), atol=1e-5)
    # loss_lm_final is the plain cross entropy of the last loop.
    assert torch.allclose(metrics["loss_lm_final"], metrics["loss_lm_ut"][-1])

    # The same LM loss computed directly from the model's outputs.
    out = module.model(input_ids=input_ids, use_cache=False, return_orbit_aux=True, compute_logits=False)
    pdf = train.build_exit_pdf(list(out.loop_gate_logits))
    valid = labels != -100
    expected = 0.0
    for step, hidden in enumerate(out.loop_hidden_states):
        logits = module.model.lm_head(hidden).float()
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=-100, reduction="none"
        ).view_as(labels)
        expected = expected + (ce * pdf[..., step] * valid).sum()
    expected = expected / valid.sum()
    assert torch.allclose(metrics["loss_lm"], expected, rtol=1e-5, atol=1e-6)

    metrics["loss"].backward()
    grads = [p.grad for p in module.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_only_the_learned_exit_mode_exists():
    with pytest.raises(ValueError, match="fixed_last"):
        train.OrbitLMTrainingModule(OuroForCausalLM(tiny_config()), beta=0.1, exit_mode="fixed_last",
                                   pad_token_id=None, lm_chunk_size=0)


def test_an_export_config_with_embedding_injection_is_refused():
    assert tiny_config(embedding_injection_mode="none").embedding_injection_mode == "none"
    with pytest.raises(ValueError, match="embedding_injection_mode"):
        tiny_config(embedding_injection_mode="concat")


def test_weight_decay_goes_to_the_projection_weights_only():
    module = tiny_module()
    groups, summary = train.build_optimizer_param_groups(module, weight_decay=0.1)
    names = {id(p): n for n, p in module.named_parameters()}
    decay = {names[id(p)] for p in groups[0]["params"]}
    no_decay = {names[id(p)] for p in groups[1]["params"]}
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    assert all(n.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                           "gate_proj.weight", "up_proj.weight", "down_proj.weight")) for n in decay)
    assert "model.lm_head.weight" in no_decay and "model.model.embed_tokens.weight" in no_decay
    assert any("early_exit_gate" in n for n in no_decay)
    assert summary["decay_tensors"] + summary["no_decay_tensors"] == len(list(module.parameters()))


def test_warmup_cosine_schedule():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = train.WarmupLRScheduler(
        optimizer, scheduler_type="warmup_cosine", base_lr=1e-5, warmup_steps=100,
        warmup_start_factor=0.1, cosine_steps=6900, min_lr=1e-6,
    )
    assert scheduler.get_lr(0) == pytest.approx(1e-6)
    assert scheduler.get_lr(50) == pytest.approx(5.5e-6)
    assert scheduler.get_lr(100) == pytest.approx(1e-5)
    # Halfway through the cosine, the LR is halfway between lr and lr_min.
    assert scheduler.get_lr(3550) == pytest.approx(5.5e-6)
    assert scheduler.get_lr(7000) == pytest.approx(1e-6)
    assert scheduler.get_lr(8000) == pytest.approx(1e-6)
    constant = train.WarmupLRScheduler(optimizer, scheduler_type="constant", base_lr=2e-5)
    assert constant.get_lr(12345) == 2e-5


def test_max_steps_from_the_token_budget():
    args = type("Args", (), {"max_steps": None, "token_budget": 20_000_000_000})()
    # The 32K stage: 20B tokens at 4,194,304 tokens per step round up to 4,769 steps.
    assert train.determine_max_steps(args, tokens_per_step=4194304, resumed_state=None) == 4769
    resumed = train.ResumeState(max_steps=4769, global_step=100)
    args.max_steps = None
    assert train.determine_max_steps(args, tokens_per_step=1, resumed_state=resumed) == 4769


def test_resume_state_round_trip():
    state = train.ResumeState(epoch=1, global_step=7, micro_step=14, grad_accum=2, local_samples_seen=28,
                              tokens_seen=99, max_steps=10, nonfinite_grad_skips=3)
    restored = train.ResumeState.from_payload(state.to_dict())
    assert restored.to_dict() == state.to_dict()
    assert "nonfinite_grad_skips" not in state.to_dict()


def save_tiny_checkpoint(run_dir: Path, module, optimizer, scheduler, step: int) -> Path:
    checkpoint_dir = checkpoint_dir_for_step(run_dir, step)
    state = train.ResumeState(global_step=step, max_steps=10)
    save_rank_checkpoint(
        checkpoint_dir, model=module, optimizer=optimizer, scheduler=scheduler, rank=0, world_size=1,
        parallelism=train.PARALLELISM, training_state=state.to_dict(), rng_state=capture_rng_state(),
        seed_state={"base_seed": 1, "rank_seed": 1}, dataloader_state=None,
    )
    write_checkpoint_metadata(
        checkpoint_dir, parallelism=train.PARALLELISM, world_size=1, training_state=state.to_dict(),
        extra=build_checkpoint_metadata_extra(module),
    )
    write_latest_pointer(run_dir, checkpoint_dir)
    return checkpoint_dir


def test_checkpoint_round_trip_restores_model_optimizer_and_scheduler(tmp_path):
    module = tiny_module(seed=3)
    groups, _ = train.build_optimizer_param_groups(module, weight_decay=0.1)
    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    scheduler = train.WarmupLRScheduler(optimizer, scheduler_type="warmup_cosine", base_lr=1e-3,
                                        warmup_steps=2, cosine_steps=8)
    input_ids, labels = batch()
    module(input_ids=input_ids, labels=labels)["loss"].backward()
    optimizer.step()
    scheduler.step(1)
    save_tiny_checkpoint(tmp_path, module, optimizer, scheduler, step=1)

    fresh = tiny_module(seed=4)
    fresh_groups, _ = train.build_optimizer_param_groups(fresh, weight_decay=0.1)
    fresh_optimizer = torch.optim.AdamW(fresh_groups, lr=1e-3)
    fresh_scheduler = train.WarmupLRScheduler(fresh_optimizer, scheduler_type="warmup_cosine", base_lr=1e-3)
    payload = load_rank_checkpoint(tmp_path / "latest.txt", model=fresh, optimizer=fresh_optimizer,
                                   scheduler=fresh_scheduler, rank=0, world_size=1)
    assert payload["training_state"]["global_step"] == 1
    for (name, a), (_, b) in zip(module.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b), name
    assert fresh_scheduler.state_dict() == scheduler.state_dict()
    assert fresh_optimizer.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()

    other_beta = train.OrbitLMTrainingModule(OuroForCausalLM(tiny_config()), beta=0.3, exit_mode="learned",
                                            pad_token_id=0, lm_chunk_size=8)
    other_optimizer = torch.optim.AdamW(train.build_optimizer_param_groups(other_beta, weight_decay=0.1)[0])
    with pytest.raises(RuntimeError, match="beta mismatch"):
        load_rank_checkpoint(tmp_path, model=other_beta, optimizer=other_optimizer, scheduler=None,
                             rank=0, world_size=1)


def write_tokenizer_config(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "tokenizer_config.json"
    path.write_text(json.dumps({
        "vocab_size": VOCAB,
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "bos_token": None,
        "added_tokens_decoder": {"0": {"content": "<|endoftext|>"}, "1": {"content": "<|im_end|>"}},
    }))
    return path


def test_export_writes_a_folder_that_loads_with_the_trained_weights(tmp_path):
    module = tiny_module(seed=5)
    module.model.config.rope_theta = 4.0e6
    module.model.config.max_position_embeddings = 48
    # The local FA3 build that a run trains with must not travel into the export.
    module.model.config._attn_kernel_repo_path = str(tmp_path / "local_fa3_build")
    optimizer = torch.optim.AdamW(train.build_optimizer_param_groups(module, weight_decay=0.1)[0])
    run_dir = tmp_path / "run"
    save_tiny_checkpoint(run_dir, module, optimizer, None, step=3)
    model_config = tmp_path / "model_config.json"
    model_config.write_text(json.dumps({"hidden_size": 999, "rope_theta": 1.0e4}))
    tokenizer_config = write_tokenizer_config(tmp_path / "tokenizer")

    out = orbit_export.export_checkpoint(
        run_dir, tmp_path / "hf", model_config=model_config, tokenizer_dir=None,
        tokenizer_config=tokenizer_config,
    )
    saved = json.loads((out / "config.json").read_text())
    # The checkpoint's recorded config wins over the model config file.
    assert saved["rope_theta"] == 4.0e6 and saved["max_position_embeddings"] == 48
    assert saved["hidden_size"] == 32
    assert saved["auto_map"]["AutoModelForCausalLM"] == "modelling_ouro.OuroForCausalLM"
    assert saved.get("_attn_kernel_repo_path") is None
    for name in orbit_export.REMOTE_CODE_FILES:
        assert (out / name).is_file()
    assert (out / "train_tokenizer_config.json").is_file()
    info_text = (out / "export_info.json").read_text()
    info = json.loads(info_text)
    assert info["training_state"]["global_step"] == 3
    # The export is copied to other machines, so it names its source without the training machine's paths.
    assert info["source_checkpoint"] == "run/checkpoints/" + checkpoint_dir_for_step(run_dir, 3).name
    assert "_attn_kernel_repo_path" not in info_text and str(tmp_path) not in info_text
    loaded = OuroForCausalLM.from_pretrained(str(out), attn_implementation="eager")
    reference = module.model.state_dict()
    for key, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, reference[key].to(tensor.dtype)), key


def test_the_export_names_a_run_reached_through_a_symlink_by_the_symlink(tmp_path):
    # A run can be published under another name than the folder it trained in, through a symlink.
    run_dir = tmp_path / "first_run_name"
    step_dir = checkpoint_dir_for_step(run_dir, 3)
    step_dir.mkdir(parents=True)
    (step_dir / "rank_0000.pt").touch()
    (run_dir / "latest.txt").write_text(step_dir.name)
    link = tmp_path / "published_name"
    link.symlink_to(run_dir, target_is_directory=True)
    expected = f"published_name/checkpoints/{step_dir.name}"
    for ref in (link, link / "latest.txt", link / "checkpoints" / step_dir.name,
                link / "checkpoints" / step_dir.name / "rank_0000.pt"):
        assert orbit_export.source_checkpoint_name(ref, step_dir) == expected, ref


def test_the_export_command_needs_the_model_config_and_the_tokenizer(capsys):
    with pytest.raises(SystemExit):
        orbit_export.parse_args(["run_dir", "out"])
    args = orbit_export.parse_args(["run", "out", "--model-config", "m.json", "--tokenizer", "tok"])
    assert args.attn_impl == "eager" and args.chat_export is False


def test_tokenizer_config_without_vocab_size_is_refused(tmp_path):
    from orbit.runtime import TokenizerConfig

    path = tmp_path / "tokenizer_config.json"
    path.write_text(json.dumps({"eos_token": "<|endoftext|>", "added_tokens_decoder": {}}))
    with pytest.raises(ValueError, match="vocab_size"):
        TokenizerConfig(path)


def planned_schedule(monkeypatch, tmp_path, config: Path, *extra: str, world_size: int = 16):
    """Parses a release config and plans its length and LR schedule as a new run on world_size GPUs."""
    for name in ("ORBIT_DATA", "ORBIT_RUNS", "ORBIT_EXPORTS", "ORBIT_TOKENIZERS", "ORBIT_EVAL_CACHE",
                 "ORBIT_LM_EVAL"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("ORBIT_EVAL_PYTHON", sys.executable)
    # The evaluation settings are checked against a real lm-eval install, which is not part of this test.
    monkeypatch.setattr(train, "validate_eval_arguments", lambda args, error: None)
    args = train.parse_args(["--config", str(config), *extra])
    ctx = train.DistContext(enabled=True, rank=0, local_rank=0, world_size=world_size, device=torch.device("cpu"))
    tokens_per_micro = train.compute_tokens_per_micro(args, ctx)
    grad_accum = train.determine_grad_accum(args, tokens_per_micro)
    data = train.TrainingData(tokenizer_cfg=None, loader=None, dataset=None, steps_per_epoch=10**9,
                              tokens_per_micro=tokens_per_micro, grad_accum=grad_accum,
                              tokens_per_step=tokens_per_micro * grad_accum)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = train.create_lr_scheduler(args, optimizer)
    _, max_steps = train.plan_training_length(args, ctx, data, None, scheduler, False)
    return scheduler, max_steps, grad_accum


@pytest.mark.parametrize("world_size, expected_grad_accum", [(64, 1), (32, 2)])
def test_the_sft_v5_budget_is_7000_steps_with_the_cosine_over_all_of_them(monkeypatch, tmp_path, world_size,
                                                                            expected_grad_accum):
    """The header of sft_v5.yaml: a step is 64 sequences on 64 GPUs, or on 32 with grad_accum 2.

    The token budget gives 7000 steps on either GPU count, and the cosine ends at lr_min on the last one.
    """
    config = REPO_ROOT / "configs" / "train" / "sft" / "sft_v5.yaml"
    scheduler, max_steps, grad_accum = planned_schedule(monkeypatch, tmp_path, config, world_size=world_size)
    assert (max_steps, grad_accum, scheduler.cosine_steps) == (7000, expected_grad_accum, 6900)
    assert scheduler.get_lr(100) == pytest.approx(1e-5)
    assert scheduler.get_lr(7000) == pytest.approx(1e-6)
