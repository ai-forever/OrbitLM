"""CPU checks of the release configs (configs/train/) and of their loading.

Every release config must load through the trainer's own option parsing when its environment
variables are set, use only the documented variables, and keep the recipe facts the lineage
depends on. The smoke configs must differ from their parents only in the keys their headers list.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from orbit import train  # noqa: E402
from orbit.experiment_config import load_experiment_config  # noqa: E402

CONFIG_ROOT = REPO_ROOT / "configs" / "train"
RELEASE_CONFIGS = sorted((CONFIG_ROOT / "pretrain").glob("*.yaml")) + sorted((CONFIG_ROOT / "sft").glob("*.yaml"))
SMOKE_CONFIGS = sorted((CONFIG_ROOT / "smoke").glob("*.yaml"))
ENV_VARS = (
    "ORBIT_DATA",
    "ORBIT_RUNS",
    "ORBIT_EXPORTS",
    "ORBIT_TOKENIZERS",
    "ORBIT_EVAL_CACHE",
    "ORBIT_LM_EVAL",
    "ORBIT_EVAL_PYTHON",
)

# Each smoke config, its parent, and the keys it may change.
SMOKE_PARENTS = {
    "pretrain_4k.yaml": ("pretrain/stage1_4k.yaml", {
        "run_name", "auto_resume", "batch_size", "tokens_per_step", "stage1_streams",
        "stage1_dataset_proportions", "max_steps", "clearml_enabled",
    }),
    "context_16k.yaml": ("pretrain/stage3_16k.yaml", {
        "run_name", "auto_resume", "batch_size", "tokens_per_step", "stage1_streams",
        "stage1_dataset_proportions", "max_steps", "clearml_enabled",
    }),
    "sft.yaml": ("sft/sft_v5.yaml", {
        "run_name", "dump_nonfinite_batch", "auto_resume", "tokens_per_step",
        "stage1_streams", "stage1_dataset_proportions", "max_steps", "clearml_enabled",
        "eval_every", "eval_at_start", "eval_at_end", "eval_suite", "eval_python", "eval_harness_path",
        "eval_cache_dir", "eval_tokenizer_source", "eval_weights", "eval_gpu_memory_utilization",
    }),
}

# The run folder name of every release stage.
RUN_NAMES = {
    "stage1_4k.yaml": "orbit_stage1_4k",
    "stage2_4k.yaml": "orbit_stage2_4k",
    "stage3_16k.yaml": "orbit_stage3_16k",
    "stage4_32k.yaml": "orbit_stage4_32k",
    "sft_v5.yaml": "orbit_sft_v5",
}

# The recipe facts of every stage: values a cleanup must never change.
RECIPE = {
    "stage1_4k.yaml": dict(lr=3e-4, rope_theta=40000.0, block_size=4096, seed=44, token_budget=419430400000,
                           init_from_hf=None, document_masking=False, labels_premasked=False),
    "stage2_4k.yaml": dict(lr=1e-4, rope_theta=40000.0, block_size=4096, seed=44, token_budget=838860800000,
                           document_masking=False),
    "stage3_16k.yaml": dict(lr=2e-5, rope_theta=1e6, block_size=16384, stage1_shuffle_seed=16385,
                            token_budget=200000000000, document_masking=True, batch_size=4),
    "stage4_32k.yaml": dict(lr=1e-5, rope_theta=4e6, block_size=32768, seed=45, stage1_shuffle_seed=32769,
                            token_budget=20000000000, document_masking=True, eval_every=1000, save_every=500),
    "sft_v5.yaml": dict(lr=1e-5, lr_min=1e-6, rope_theta=4e6, block_size=32768, seed=46, stage1_shuffle_seed=47,
                        tokens_per_step=2097152, token_budget=14680064000, batch_size=1, labels_premasked=True,
                        document_masking=True, eval_every=1000, save_every=500),
}


@pytest.fixture
def release_env(tmp_path, monkeypatch):
    """Dummy values for every documented variable, all below one temporary folder."""
    roots = {}
    for name in ENV_VARS:
        path = tmp_path / name.lower()
        path.mkdir()
        roots[name] = path
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("ORBIT_EVAL_PYTHON", sys.executable)
    roots["ORBIT_EVAL_PYTHON"] = Path(sys.executable)
    monkeypatch.delenv("ORBIT_ATTN_KERNEL_PATH", raising=False)
    return roots


def parse(config: Path, monkeypatch, *extra: str):
    # The evaluation settings are checked against a real lm-eval install and cache at startup;
    # that check belongs to the evaluation's own tests, so it is left out here.
    monkeypatch.setattr(train, "validate_eval_arguments", lambda args, error: None)
    return train.parse_args(["--config", str(config), *extra])


@pytest.mark.parametrize("config", RELEASE_CONFIGS + SMOKE_CONFIGS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_config_loads_with_the_documented_variables(config, release_env, monkeypatch):
    args = parse(config, monkeypatch)
    assert args.save_dir == release_env["ORBIT_RUNS"].resolve()
    assert args.model_config == (REPO_ROOT / "src" / "orbit" / "ouro_1.4b_qwen3_config.json").resolve()
    assert args.model_config.is_file()
    assert Path(args.tokenizer_config).is_relative_to(release_env["ORBIT_TOKENIZERS"].resolve())
    assert Path(args.stage1_root).is_relative_to(release_env["ORBIT_DATA"])
    assert args.stage1_mixing_mode == "mosaic_random"
    assert args.attn_kernel_path is None  # no config pins a kernel build; ORBIT_ATTN_KERNEL_PATH may
    assert args.clearml_enabled is False
    assert args.tensorboard_root == (release_env["ORBIT_RUNS"] / "tensorboard").resolve()
    for name, path in args.stage1_streams:
        if Path(path).is_absolute():
            assert path.startswith(str(release_env["ORBIT_DATA"])), name
    if args.init_from_hf is not None:
        assert args.init_from_hf.is_relative_to(release_env["ORBIT_EXPORTS"].resolve())
    if args.eval_every:
        assert args.eval_cache_dir == release_env["ORBIT_EVAL_CACHE"].resolve()
        assert args.eval_harness_path == release_env["ORBIT_LM_EVAL"].resolve()
        assert args.eval_python == Path(sys.executable)
    if "eval_suite" in args._config_keys:
        assert args.eval_suite == (REPO_ROOT / "eval" / "in_training" / "suites" / "sft.yaml").resolve()
        assert args.eval_suite.is_file()
        # The SFT suite's extra packages come from the evaluation environment itself, so no release
        # config needs a separate folder for them.
        assert args.eval_extra_pythonpath is None


@pytest.mark.parametrize("config", RELEASE_CONFIGS + SMOKE_CONFIGS, ids=lambda p: p.name)
def test_configs_name_only_documented_variables(config):
    text = config.read_text()
    assert set(re.findall(r"\$\{([A-Za-z_0-9]+)\}", text)) <= set(ENV_VARS)


@pytest.mark.parametrize("config", RELEASE_CONFIGS, ids=lambda p: p.name)
def test_the_recipe_values_of_every_stage(config, release_env, monkeypatch):
    args = parse(config, monkeypatch)
    for key, expected in RECIPE[config.name].items():
        value = getattr(args, key)
        if isinstance(expected, float):
            assert value == pytest.approx(expected, rel=0, abs=0), key
        else:
            assert value == expected, key
    # Every 16K/32K and SFT stage must set rope_theta itself, or it falls back to the export's.
    assert "rope_theta" in args._config_keys
    assert [name for name, _ in args.stage1_streams] == list(args.stage1_dataset_proportions)
    assert args.run_name == RUN_NAMES[config.name]


def test_the_sft_stage_trains_with_the_tools_tokenizer_and_evaluates_with_the_think_tokenizer(release_env,
                                                                                               monkeypatch):
    """Stage 5 starts from the 32K export and renders its shards with the tools template.

    Its in-training evaluation renders the prompts with the think template. The chat suite's prompts
    have no system turn, and for such chats the two templates give the same text and token ids.
    """
    args = parse(CONFIG_ROOT / "sft" / "sft_v5.yaml", monkeypatch)
    tokenizers = release_env["ORBIT_TOKENIZERS"].resolve()
    assert Path(args.tokenizer_config).resolve() == tokenizers / "tokenizer_qwen3_4b_qwen35_tools" / "tokenizer_config.json"
    assert Path(args.eval_tokenizer_source).resolve() == tokenizers / "tokenizer_qwen3_4b_qwen35_think"
    assert args.init_from_hf.resolve() == release_env["ORBIT_EXPORTS"].resolve() / "stage4_32k_step4769"
    assert Path(args.stage1_root) == release_env["ORBIT_DATA"] / "sft" / "v5"


@pytest.mark.parametrize("config", [CONFIG_ROOT / "pretrain" / "stage1_4k.yaml", CONFIG_ROOT / "pretrain" / "stage2_4k.yaml",
                                    CONFIG_ROOT / "smoke" / "pretrain_4k.yaml"], ids=lambda p: p.name)
def test_the_4k_configs_read_the_english_and_russian_folders_of_pretrain_4k(config, release_env, monkeypatch):
    args = parse(config, monkeypatch)
    data = release_env["ORBIT_DATA"]
    assert Path(args.stage1_root) == data / "pretrain_4k" / "en"
    for name, path in args.stage1_streams:
        if name.startswith("ru_"):
            assert Path(path).parent == data / "pretrain_4k" / "ru", name
        else:
            assert not Path(path).is_absolute(), name


@pytest.mark.parametrize("smoke", SMOKE_CONFIGS, ids=lambda p: p.name)
def test_smoke_configs_differ_from_their_parent_only_in_the_listed_keys(smoke, release_env):
    parent_name, allowed = SMOKE_PARENTS[smoke.name]
    parent = load_experiment_config(CONFIG_ROOT / parent_name)
    child = load_experiment_config(smoke)
    changed = {
        key for key in set(parent) | set(child)
        if key != "config" and parent.get(key, "<absent>") != child.get(key, "<absent>")
    }
    assert changed <= allowed, sorted(changed - allowed)
    # The streams of a smoke config are a subset of the parent's, with the parent's paths and weights.
    parent_streams = dict(parent["stage1_streams"])
    for name, path in child["stage1_streams"]:
        assert parent_streams[name] == path
        assert child["stage1_dataset_proportions"][name] == parent["stage1_dataset_proportions"][name]
    assert child["max_steps"] <= 5
    assert child["clearml_enabled"] is False
    assert not any(key.startswith("eval_") for key in child)


def test_an_unset_variable_is_named_in_the_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ORBIT_DATA", raising=False)
    config = tmp_path / "run.yaml"
    config.write_text("stage1_root: ${ORBIT_DATA}/ctx16k/streams\n")
    with pytest.raises(KeyError, match=r"'stage1_root' uses the environment variable ORBIT_DATA"):
        load_experiment_config(config)


def test_variables_expand_in_nested_values_and_only_in_braces(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBIT_DATA", "/data")
    config = tmp_path / "run.yaml"
    config.write_text(
        "stage1_streams:\n  ru: ${ORBIT_DATA}/ru\n  en: en\n"
        "stage1_dataset_proportions:\n  ru: 1\n  en: 1\nrun_name: cost$5\n"
    )
    cfg = load_experiment_config(config)
    assert cfg["stage1_streams"] == [("ru", "/data/ru"), ("en", "en")]
    assert cfg["run_name"] == "cost$5"


@pytest.mark.parametrize(
    "line",
    ["init_from_legacy: /some/checkpoint", "exit_mode: fixed_last", "embedding_injection_mode: concat",
     "parallelism: ddp", "resume_validation: full", "weight_decay_mode: all", "ema_decay: 0.999"],
)
def test_a_removed_option_with_another_value_is_refused(line, release_env, monkeypatch, tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text((CONFIG_ROOT / "pretrain" / "stage1_4k.yaml").read_text() + line + "\n")
    with pytest.raises(SystemExit):
        parse(config, monkeypatch)


def config_with(tmp_path: Path, key: str, value: str) -> Path:
    """A copy of the stage-1 config in which one key has the given value."""
    lines = [line for line in (CONFIG_ROOT / "pretrain" / "stage1_4k.yaml").read_text().splitlines()
             if not line.startswith(f"{key}:")]
    config = tmp_path / "run.yaml"
    config.write_text("\n".join(lines + [f"{key}: {value}"]) + "\n")
    return config


@pytest.mark.parametrize(
    "key, value",
    [("stage1_mixing_mode", "global_indexed_wrr"), ("lr_scheduler", "warmup_constant"), ("dtype", "fp16"),
     ("attn_impl", "flash_attention_2"), ("eval_weights", "ema"), ("fsdp_sharding_strategy", "hybrid_shard")],
)
def test_a_config_value_outside_the_choices_of_its_option_is_refused(key, value, release_env, monkeypatch, tmp_path,
                                                                      capsys):
    with pytest.raises(SystemExit):
        parse(config_with(tmp_path, key, value), monkeypatch)
    assert f"The config sets {key}: {value!r}" in capsys.readouterr().err


def test_a_config_without_a_mixing_mode_gets_the_only_one(release_env, monkeypatch, tmp_path):
    lines = [line for line in (CONFIG_ROOT / "pretrain" / "stage1_4k.yaml").read_text().splitlines()
             if not line.startswith("stage1_mixing_mode:")]
    config = tmp_path / "run.yaml"
    config.write_text("\n".join(lines) + "\n")
    assert parse(config, monkeypatch).stage1_mixing_mode == "mosaic_random"


@pytest.mark.parametrize(
    "line",
    ["init_from_legacy: null", "exit_mode: learned", "embedding_injection_mode: none",
     "parallelism: fsdp", "resume_validation: minimal", "detect_anomaly: false"],
)
def test_a_removed_option_at_its_only_value_is_accepted(line, release_env, monkeypatch, tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text((CONFIG_ROOT / "pretrain" / "stage1_4k.yaml").read_text() + line + "\n")
    args = parse(config, monkeypatch)
    assert args.lr == pytest.approx(3e-4)


@pytest.mark.parametrize("config", sorted((REPO_ROOT / "tests" / "train" / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_the_end_to_end_test_configs_still_load(config, tmp_path, monkeypatch):
    """The GPU end-to-end configs leave the machine paths null; their commands pass them."""
    tokenizer = tmp_path / "tokenizer_config.json"
    tokenizer.write_text("{}")
    args = parse(
        config, monkeypatch, "--save-dir", str(tmp_path), "--tokenizer-config", str(tokenizer),
        "--stage1-root", str(tmp_path), "--tensorboard-root", str(tmp_path / "tb"),
    )
    assert args.document_masking is True
    assert args.model_config.is_file()
