"""CPU tests of the launcher and the configs.

Run with the verl python and ``PYTHONPATH=<repo>/src``. Nothing here starts Ray or touches a GPU:
every ``ray`` CLI call is replaced by a function that fails the test, and every run uses a temporary
runs root. The exports are fake dirs that hold only a config.json, which is all the launcher reads
from an export before preparing it.

Most tests run the RL recipe ``rl_sft_mix`` with the single-mode dataset builder that the pass-rate
tool uses (``SINGLE_MODE``). Its prepared dataset is one prepare_data dir, which is simple to fake,
and the launcher logic under test is the same for both builders.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from orbit_verl import common
from orbit_verl.launch import cli, cluster
from orbit_verl.launch import config as lconfig
from orbit_verl.launch import env as lenv
from orbit_verl.launch import prepare as lprepare
from orbit_verl.model import MODEL_CODE_DIR, MODEL_CODE_FILES, PRESETS

REPO = Path(__file__).resolve().parents[2]
LAUNCH_SCRIPT = REPO / "scripts" / "verl" / "launch.py"
CONFIG = "rl_sft_mix"
# The overrides of the pass-rate tool: the RL recipe with one prompt mode, built by prepare_data from
# one file. The file does not need to exist, because no test here builds the dataset.
SRC = "/data/rl/math_keep.jsonl"
SINGLE_MODE = ["orbit.prepare.data.builder=prepare_data", "orbit.prompt_mode=sft_think",
               f"orbit.prepare.data.src={SRC}", f"orbit.prepare.data.val_src={SRC}",
               "orbit.prepare.data.gold=null", "orbit.prepare.data.val=null"]
# The dataset dir of rl_sft_mix for the release tokenizer.
DATA_NAME = "rl_sft_mix__qwen3"
# The gold rows and validation sets of the recipe, for tests that compose it with its own builder.
RECIPE_DATA = ['orbit.prepare.data.gold=["/data/rl/math_keep.jsonl:orbit/math:math"]',
               'orbit.prepare.data.val=["/data/rl/math500.jsonl:orbit/math#think:sft_think"]']
# The lineage that prepare_model records for an export of each tokenizer family, with its frozen row.
FAMILY_LINEAGE = {"qwen3": ("release_qwen3", None)}
# Overrides that keep a dry run independent of the RLTT plugin, which another part of the code base
# provides; the plugin import has its own check.
NO_PLUGIN = ["orbit.plugin=false", "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"]


@pytest.fixture
def process_state():
    """Restore os.environ and the working directory after a test that runs the launcher in-process."""
    saved, cwd = dict(os.environ), os.getcwd()
    yield
    os.environ.clear()
    os.environ.update(saved)
    os.chdir(cwd)


@pytest.fixture
def no_ray(monkeypatch):
    """Fail the test if anything tries to run a ray CLI command."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"a test tried to run {args[0] if args else kwargs}")
    monkeypatch.setattr(cluster.subprocess, "call", refuse)


def _set_env(monkeypatch, **values):
    for key in list(os.environ):
        if lenv.is_launcher_var(key):
            monkeypatch.delenv(key)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _export(root: Path, family: str = "qwen3", run: str = "orbit_run_a", step: str = "step_00000100",
            max_positions: int = 32768) -> Path:
    """A fake HF export: only the config.json that the launcher reads before preparing."""
    path = root / "exports" / run / step
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({
        "model_type": "ouro", "vocab_size": PRESETS[family]["vocab_size"],
        "max_position_embeddings": max_positions, "rope_theta": 40000.0}))
    return path


def _compose(config: str, tmp_path: Path, *overrides: str, export: Path | None = None,
             model_name: str | None = None, runs: Path | None = None, mode=()):
    """Compose a config the way the launcher does, including the values it derives afterwards."""
    user = list(overrides)
    chosen = lconfig.model_overrides(None if export is None else str(export), model_name, user)
    cfg = lconfig.compose_config(config, [*user, *chosen], repo_root=tmp_path, mode=mode)
    lconfig.finish_config(cfg, config, runs_root=str(runs or tmp_path / "runs_root"), environ={})
    return cfg


# --------------------------------------------------------------------------- runs root and names


def test_runs_root_order_is_cli_then_config_then_env(tmp_path):
    env = {common.RUNS_ROOT_ENV: str(tmp_path / "env")}
    assert common.resolve_runs_root_with_source("/cli", "/cfg", env) == ("/cli", "--runs-root")
    assert common.resolve_runs_root_with_source(None, "/cfg", env) == ("/cfg", "orbit.runs_root")
    assert common.resolve_runs_root_with_source(None, None, env) == (str(tmp_path / "env"), common.RUNS_ROOT_ENV)
    # Empty values count as unset; a relative or ~ path becomes absolute.
    assert common.resolve_runs_root("", "", {common.RUNS_ROOT_ENV: "rel"}) == os.path.abspath("rel")
    assert common.resolve_runs_root("~/r", None, {}) == os.path.expanduser("~/r")


def test_no_runs_root_is_refused_with_the_export_line():
    with pytest.raises(common.RunsRootError) as info:
        common.resolve_runs_root(None, None, {})
    assert f"export {common.RUNS_ROOT_ENV}=/path/to/your/verl_runs" in str(info.value)
    assert isinstance(info.value, ValueError)
    with pytest.raises(common.RunsRootError):
        common.cache_env("")


def test_there_is_no_default_runs_root():
    # Without a default, a run never writes into a directory that nobody chose for it.
    assert not hasattr(common, "DEFAULT_RUNS_ROOT")


def test_apply_cache_env_takes_the_root_from_the_env(monkeypatch, tmp_path):
    for key in (*common.CACHE_ENV_KEYS, common.RUNS_ROOT_ENV):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(common.RunsRootError):
        common.apply_cache_env()
    monkeypatch.setenv(common.RUNS_ROOT_ENV, str(tmp_path))
    applied = common.apply_cache_env()
    assert applied["HF_MODULES_CACHE"].startswith(f"{tmp_path}/cache/")
    # Once the caches are set, a call without a root has nothing to decide.
    monkeypatch.delenv(common.RUNS_ROOT_ENV)
    assert common.apply_cache_env() == applied


def test_node_env_replaces_values_that_an_import_set(tmp_path):
    """Only the caller's own cache settings survive; a value set later by a library is replaced."""
    environ = {"HF_MODULES_CACHE": "/mine/hf_modules"}
    keep = lenv.caller_set_node_env(environ)
    assert keep == {"HF_MODULES_CACHE"}
    # This is what `import verl` does through torch's inductor.
    environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor_someone"
    applied = lenv.apply_node_env(str(tmp_path), environ, keep=keep)
    assert applied["HF_MODULES_CACHE"] == "/mine/hf_modules"
    assert applied["TORCHINDUCTOR_CACHE_DIR"].startswith(f"{tmp_path}/cache/")
    assert applied["TORCHINDUCTOR_CACHE_DIR"].endswith("/inductor")


def test_launcher_moves_the_inductor_cache_that_composing_set(tmp_path, process_state, no_ray, monkeypatch,
                                                              capsys):
    """The launcher's node env ends up under the runs root even though composing imports verl."""
    runs, export, _ = _smoke_prepared(tmp_path)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    real_compose = lconfig.compose_config

    def compose_like_a_first_verl_import(*args, **kwargs):
        cfg = real_compose(*args, **kwargs)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor_someone"
        return cfg

    monkeypatch.setattr(cli, "compose_config", compose_like_a_first_verl_import)
    rc = cli.main(_dry_run(CONFIG, export, runs, *SINGLE_MODE, *NO_PLUGIN))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"TORCHINDUCTOR_CACHE_DIR={runs}/cache/" in out
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"].startswith(f"{runs}/cache/")


def test_model_name_for_export(tmp_path):
    export = _export(tmp_path, run="orbit_sft v5", step="step_00007000")
    assert common.model_name_for_export(export) == "orbit_sft_v5__step_00007000"
    assert common.model_name_for_export(f"{export}/") == "orbit_sft_v5__step_00007000"
    # A link is named after the step it points to, so "latest" never becomes a dir name.
    link = tmp_path / "exports" / "orbit_sft v5" / "latest"
    link.symlink_to(export)
    assert common.model_name_for_export(link) == "orbit_sft_v5__step_00007000"
    assert common.default_run_name("rl_sft_mix", "m") == "rl_sft_mix__m"
    assert common.default_run_name(str(tmp_path / "my_exp.yaml"), "m") == "my_exp__m"
    with pytest.raises(ValueError):
        common.safe_name("///")


def test_config_values_without_hydra(tmp_path):
    assert common.override_value(["a=1", "+orbit.runs_root='/x y'", "orbit.runs_root=/z"], "orbit.runs_root") == "/z"
    path = tmp_path / "e.yaml"
    path.write_text("orbit:\n  runs_root: /r  # comment\n  run_name: ???\n  prepare:\n    x: 1\n"
                    "data:\n  run_name: no\n")
    assert common.yaml_orbit_value(path, "runs_root") == "/r"
    assert common.yaml_orbit_value(path, "run_name") is None
    assert common.yaml_orbit_value(REPO / "configs" / "verl" / "orbit_base.yaml", "runs_root") is None


# --------------------------------------------------------------------------- env


def test_strip_launcher_env_removes_every_launcher_var():
    env = {name: "1" for name in lenv.LAUNCHER_ENV_VARS}
    env.update({"TORCHELASTIC_RUN_ID": "x", "TORCHELASTIC_USE_AGENT_STORE": "True",
                "PATH": "/bin", "NCCL_DEBUG": "INFO", "MY_RANK": "3"})
    assert lenv.strip_launcher_env(env) == {"PATH": "/bin", "NCCL_DEBUG": "INFO", "MY_RANK": "3"}

    removed = lenv.remove_launcher_env(env)
    assert set(removed) == set(lenv.LAUNCHER_ENV_VARS) | {"TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE"}
    assert env == {"PATH": "/bin", "NCCL_DEBUG": "INFO", "MY_RANK": "3"}


def test_read_topology():
    assert lenv.read_topology({}) == (0, 1, None)
    assert lenv.read_topology({}).mode == "local"
    assert lenv.read_topology({"WORLD_SIZE": "1", "RANK": "0"}).mode == "local"
    head = lenv.read_topology({"WORLD_SIZE": "3", "RANK": "0", "MASTER_ADDR": "h"})
    assert head.mode == "head" and head.master_addr == "h"
    rank, world, master = lenv.read_topology({"WORLD_SIZE": "3", "RANK": "2", "MASTER_ADDR": "h"})
    assert (rank, world, master) == (2, 3, "h")
    with pytest.raises(ValueError):
        lenv.read_topology({"WORLD_SIZE": "2", "RANK": "2"})


def _orbit_cfg(**orbit):
    base = {"run_dir": "/r/runs/x", "runs_root": "/r", "plugin": True,
            "rltt": {"progressive_alpha": 2.0, "chunk_size": 256}, "env": {}}
    base.update(orbit)
    return {"orbit": base}


def test_job_env_is_strings_only_without_secrets_or_cache_vars():
    cfg = _orbit_cfg(env={"NCCL_DEBUG": "INFO", "SOME_INT": 3})
    env = lenv.job_env(cfg, "/repo", base_env={"PYTHONPATH": "/other:/repo/src", "CLEARML_API_ACCESS_KEY": "s"})
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
    assert env["PYTHONPATH"] == "/repo/src:/other"
    assert env["VERL_USE_EXTERNAL_MODULES"] == "orbit_verl.plugin"
    assert env["VERL_ENGINE_VENDOR"] == "orbit"
    assert env["VERL_FILE_LOGGER_PATH"] == "/r/runs/x/metrics.jsonl"
    assert env["ORBIT_RLTT_PROGRESSIVE_ALPHA"] == "2.0" and env["ORBIT_RLTT_CHUNK_SIZE"] == "256"
    assert env["ORBIT_RLTT_TRAIN_EXIT_GATE"] == "false"
    # Every Ray process sees the resolved runs root.
    assert env[common.RUNS_ROOT_ENV] == "/r"
    assert env["SOME_INT"] == "3"
    assert not [k for k in env if k.startswith("CLEARML_")]
    assert not set(env) & set(common.CACHE_ENV_KEYS)


def test_job_env_without_plugin_and_forbidden_user_keys():
    env = lenv.job_env(_orbit_cfg(plugin=False), "/repo", base_env={})
    assert "VERL_USE_EXTERNAL_MODULES" not in env and "VERL_ENGINE_VENDOR" not in env
    for bad in ("CLEARML_API_SECRET_KEY", "RANK", "TORCHELASTIC_RUN_ID", common.RUNS_ROOT_ENV):
        with pytest.raises(ValueError):
            lenv.job_env(_orbit_cfg(env={bad: "x"}), "/repo", base_env={})


def test_job_env_passes_train_exit_gate_as_a_strict_boolean():
    rltt = {"progressive_alpha": 1.0, "chunk_size": 512}
    env = lenv.job_env(_orbit_cfg(rltt={**rltt, "train_exit_gate": True}), "/repo", base_env={})
    assert env["ORBIT_RLTT_TRAIN_EXIT_GATE"] == "true"
    # A string would be true in Python whatever it says, so only a YAML boolean is accepted.
    for bad in ("no", "false", 1):
        with pytest.raises(ValueError, match="train_exit_gate"):
            lenv.job_env(_orbit_cfg(rltt={**rltt, "train_exit_gate": bad}), "/repo", base_env={})


def test_load_dotenv(tmp_path):
    path = tmp_path / "clearml.env"
    path.write_text(
        "# credentials\n"
        "\n"
        "CLEARML_API_ACCESS_KEY=abc\n"
        "export CLEARML_API_SECRET_KEY='s3cr$t #x'\n"
        'CLEARML_API_HOST="https://api.example"  \n'
        "PLAIN=value # trailing comment\n"
        "EMPTY=\n"
    )
    assert lenv.load_dotenv(path) == {
        "CLEARML_API_ACCESS_KEY": "abc",
        "CLEARML_API_SECRET_KEY": "s3cr$t #x",
        "CLEARML_API_HOST": "https://api.example",
        "PLAIN": "value",
        "EMPTY": "",
    }
    path.write_text("not a pair\n")
    with pytest.raises(ValueError):
        lenv.load_dotenv(path)


def test_redact_hides_credentials():
    out = lenv.redact({"CLEARML_API_HOST": "h", "WANDB_API_KEY": "k", "NCCL_DEBUG": "INFO"})
    assert out == {"CLEARML_API_HOST": "***", "WANDB_API_KEY": "***", "NCCL_DEBUG": "INFO"}


# --------------------------------------------------------------------------- config


def test_override_order_puts_mode_overrides_last(tmp_path):
    mode = lconfig.mode_overrides(nnodes=1, gpus_per_node=2)
    order = lconfig.build_overrides(tmp_path, lconfig.CONFIG_DIR, ["trainer.nnodes=7", "x=1"], mode)
    assert order[0] == "hydra.searchpath=[pkg://verl.trainer.config]"
    assert order[1] == f"orbit.repo_root={tmp_path.resolve()}"
    assert order[2:4] == ["trainer.nnodes=7", "x=1"]
    assert order[4:] == mode

    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "trainer.nnodes=7", "trainer.n_gpus_per_node=5",
                   "trainer.total_training_steps=5", export=_export(tmp_path), mode=mode)
    assert (cfg.trainer.nnodes, cfg.trainer.n_gpus_per_node) == (1, 2)   # the mode overrides win
    assert cfg.trainer.total_training_steps == 5                        # other user overrides apply


def test_mode_overrides_for_a_cluster_drop_num_cpus():
    assert "ray_kwargs.ray_init.num_cpus=4" in lconfig.mode_overrides(nnodes=1, gpus_per_node=1, num_cpus=4)
    joined = lconfig.mode_overrides(nnodes=2, gpus_per_node=8, ray_address="10.0.0.1:6379", num_cpus=4)
    assert "ray_kwargs.ray_init.address=10.0.0.1:6379" in joined
    assert not [o for o in joined if "num_cpus" in o]


@pytest.mark.parametrize("nnodes,gpus", [(1, 1), (4, 8)])
def test_the_rl_recipe_composes_and_validates(nnodes, gpus, tmp_path):
    from orbit_verl.launch.train import validate

    export = _export(tmp_path)
    model = common.model_name_for_export(export)
    cfg = _compose(CONFIG, tmp_path, *RECIPE_DATA, export=export,
                   mode=lconfig.mode_overrides(nnodes=nnodes, gpus_per_node=gpus))
    job = lenv.job_env(cfg, str(cfg.orbit.repo_root), base_env={})
    lconfig.put_runtime_env(cfg, job)
    validate(cfg)

    root = str(tmp_path / "runs_root")
    run = f"{CONFIG}__{model}"
    assert cfg.orbit.runs_root == root
    assert cfg.trainer.experiment_name == cfg.orbit.run_name == run
    assert cfg.actor_rollout_ref.model.path == f"{root}/models/{model}"
    # The dataset belongs to the tokenizer family, which the launcher read from the export.
    assert cfg.orbit.prepare.model.preset == "qwen3"
    assert cfg.data.train_files == f"{root}/data/{DATA_NAME}/train.parquet"
    assert cfg.data.val_files == f"{root}/data/{DATA_NAME}/val.parquet"
    assert cfg.actor_rollout_ref.rollout.agent.default_agent_loop == "orbit_sft_think"
    assert cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path == \
        f"{tmp_path.resolve()}/configs/verl/agent_loops.yaml"
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "rltt_exit"
    # The model dir decides which embedding row is frozen; no experiment may override it.
    override = cfg.actor_rollout_ref.model.override_config
    assert "pad_token_id" not in override and "embedding_padding_idx" not in override
    assert job["ORBIT_RLTT_TRAIN_EXIT_GATE"] == "false"
    # The rollout and the agent loop read their limit from the rollout config, which verl derives.
    assert cfg.actor_rollout_ref.rollout.response_length == cfg.data.max_response_length == 16384
    assert cfg.data.max_prompt_length + cfg.data.max_response_length <= 32768
    assert cfg.trainer.default_local_dir == f"{root}/runs/{run}/ckpt"
    assert list(cfg.orbit.fits.lineages) == ["release_qwen3", "unknown"]
    # No experiment config names a checkpoint: the model comes from --export.
    plain = lconfig.compose_config(CONFIG, repo_root=tmp_path)
    assert lconfig._value(plain, "orbit.prepare.model.export") is None
    assert lconfig._value(plain, "orbit.runs_root") is None

    runtime = dict(cfg.ray_kwargs.ray_init.runtime_env.env_vars)
    assert runtime == job
    assert not set(runtime) & set(common.CACHE_ENV_KEYS)
    assert not [k for k in runtime if k.startswith("CLEARML_")]


def test_finish_config_keeps_given_names_and_reads_the_preset_of_a_prepared_dir(tmp_path):
    export = _export(tmp_path)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "orbit.run_name=my_run", export=export, model_name="rel")
    assert (cfg.orbit.run_name, cfg.orbit.prepare.model.name) == ("my_run", "rel")
    assert cfg.orbit.prepare.model.preset == "qwen3"

    # An export that a training node cannot read: the prepared dir's manifest names the family.
    runs = tmp_path / "runs_root"
    _fake_prepared(runs, "rel", DATA_NAME, max_prompt_length=2048, prompt_mode="sft_think",
                   export="/unreadable/step_1", src=SRC)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=Path("/unreadable/step_1"), model_name="rel")
    assert cfg.orbit.prepare.model.preset == "qwen3"
    assert cfg.data.train_files == f"{runs}/data/{DATA_NAME}/train.parquet"

    # Without a readable export or a prepared dir, auto stays auto, and an explicit preset is never replaced.
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=Path("/unreadable/step_2"), model_name="other")
    assert cfg.orbit.prepare.model.preset == "auto"
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "orbit.prepare.model.preset=qwen3",
                   export=Path("/unreadable/step_2"), model_name="other")
    assert cfg.orbit.prepare.model.preset == "qwen3"


def test_finish_config_without_a_model_has_no_run_name(tmp_path):
    cfg = lconfig.compose_config(CONFIG, repo_root=tmp_path)
    with pytest.raises(ValueError, match="--export PATH"):
        lconfig.finish_config(cfg, CONFIG, runs_root=str(tmp_path), environ={})
    cfg = lconfig.compose_config(CONFIG, repo_root=tmp_path)
    with pytest.raises(common.RunsRootError):
        lconfig.finish_config(cfg, CONFIG, environ={})


def test_model_overrides_follow_the_options(tmp_path):
    export = _export(tmp_path)
    derived = common.model_name_for_export(export)
    assert lconfig.model_overrides(str(export), None) == [
        f"orbit.prepare.model.export={export}", f"orbit.prepare.model.name={derived}"]
    assert lconfig.model_overrides(str(export), "mine")[1] == "orbit.prepare.model.name=mine"
    # A name given as an override is kept, so the derived name is not added.
    assert lconfig.model_overrides(str(export), None, ["orbit.prepare.model.name=x"]) == [
        f"orbit.prepare.model.export={export}"]
    assert lconfig.model_overrides(None, None) == []


def test_config_path_outside_the_repo_finds_the_base(tmp_path):
    path = tmp_path / "my_exp.yaml"
    path.write_text(
        "defaults:\n  - orbit_base\n  - _self_\n"
        "orbit:\n  run_name: my_exp\n  prompt_mode: sft_think_code\n  prepare:\n"
        "    model:\n      export: /x\n      name: m\n"
        "    data:\n      src: /y.jsonl\n      name: d\n"
    )
    cfg = lconfig.compose_config(str(path), repo_root=tmp_path)
    lconfig.finish_config(cfg, str(path), runs_root=str(tmp_path), environ={})
    assert cfg.trainer.experiment_name == "my_exp"
    assert cfg.actor_rollout_ref.rollout.agent.default_agent_loop == "orbit_sft_think_code"


def test_unknown_config_lists_the_available_ones():
    with pytest.raises(FileNotFoundError, match="rl_sft_mix"):
        lconfig.resolve_config_ref("no_such_experiment")
    # The two recipes and the distillation configs they inherit, and nothing else.
    assert set(lconfig.available_configs()) == {"rl_sft_mix", "distill_sft", "distill_rl_sft", "distill_rl_sft_16k"}


def test_user_env_reaches_the_job_env(tmp_path):
    # orbit.env starts empty, so a new key needs hydra's "+" prefix.
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "+orbit.env.NCCL_DEBUG=INFO", export=_export(tmp_path))
    assert lenv.job_env(cfg, str(tmp_path), base_env={})["NCCL_DEBUG"] == "INFO"


def test_put_runtime_env_rejects_non_strings(tmp_path):
    cfg = lconfig.compose_config(CONFIG, repo_root=tmp_path)
    with pytest.raises(TypeError):
        lconfig.put_runtime_env(cfg, {"N": 3})
    lconfig.put_runtime_env(cfg, {"LITERAL": "${not_an_interpolation}"})
    assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.LITERAL == "${not_an_interpolation}"


# --------------------------------------------------------------------------- GPUs and the Ray CLI


def test_gpu_count_from_cuda_visible_devices():
    assert cli.count_visible_devices("0,1,2") == 3
    assert cli.count_visible_devices("") == 0
    assert cli.count_visible_devices("0,-1,2") == 1
    assert cli.count_visible_devices("GPU-aaa,GPU-bbb") == 2
    assert cli.detect_gpus(None, {"CUDA_VISIBLE_DEVICES": "4,5"}) == (2, "CUDA_VISIBLE_DEVICES")
    assert cli.detect_gpus(3, {"CUDA_VISIBLE_DEVICES": "4,5"}) == (3, "--gpus-per-node")


def test_ray_cli_uses_this_python_and_strips_launcher_env(monkeypatch):
    seen = {}

    def fake_call(cmd, env):
        seen.update(cmd=cmd, env=env)
        return 0

    monkeypatch.setattr(cluster.subprocess, "call", fake_call)
    assert cluster.ray_cli(["status"], env={"RANK": "1", "TORCHELASTIC_X": "1", "PATH": "/bin"},
                           log=lambda msg: None) == 0
    assert seen["cmd"] == [sys.executable, "-m", "ray.scripts.scripts", "status"]
    assert seen["env"] == {"PATH": "/bin"}
    head = cluster.head_start_args(ip="1.2.3.4", port=6379, gpus=8, num_cpus=16)
    assert head[:2] == ["start", "--head"]
    assert {"--include-dashboard=false", "--disable-usage-stats", "--num-gpus=8", "--num-cpus=16"} <= set(head)


def _fake_worker_loop(monkeypatch, *, head_up_polls: int):
    """Run the worker loop with a fake clock: the head answers ``head_up_polls`` polls, then stops."""
    clock = {"now": 0.0, "polls": 0, "stopped": 0}

    def fake_sleep(seconds):
        clock["now"] += seconds

    def fake_port_open(host, port, timeout=5.0):
        clock["polls"] += 1
        # The first poll is the join check before `ray start`, so the head answers once more
        # than head_up_polls.
        return clock["polls"] <= head_up_polls + 1

    def fake_ray_cli(args, **kwargs):
        clock["stopped"] += args[0] == "stop"
        return 0

    monkeypatch.setattr(cluster.time, "sleep", fake_sleep)
    monkeypatch.setattr(cluster.time, "time", lambda: clock["now"])
    monkeypatch.setattr(cluster, "port_open", fake_port_open)
    monkeypatch.setattr(cluster, "ray_cli", fake_ray_cli)
    return clock


def test_worker_has_no_lifetime_limit_by_default(monkeypatch):
    args, _ = cli.parse_args(["--config", "x"])
    assert args.max_lifetime == 0
    # Three days of a healthy head: the worker stays the whole time and leaves only after the
    # head's port stops answering three times in a row.
    up = 3 * 24 * 3600 // cluster.HEAD_POLL_SECONDS
    clock = _fake_worker_loop(monkeypatch, head_up_polls=up)
    logs = []
    rc = cluster.run_worker_until_head_gone(head="127.0.0.1", port=1, gpus=8, num_cpus=None,
                                            join_timeout=60, max_lifetime=args.max_lifetime, log=logs.append)
    assert rc == 0
    assert clock["now"] == (up + cluster.HEAD_MISSES_TO_LEAVE) * cluster.HEAD_POLL_SECONDS
    assert clock["stopped"] == 1 and "the head is gone" in logs[-1]


def test_worker_lifetime_limit_when_given(monkeypatch):
    clock = _fake_worker_loop(monkeypatch, head_up_polls=10**6)
    logs = []
    rc = cluster.run_worker_until_head_gone(head="127.0.0.1", port=1, gpus=8, num_cpus=None,
                                            join_timeout=60, max_lifetime=100, log=logs.append)
    assert rc == 0
    assert clock["now"] == 100 and clock["stopped"] == 1 and "max lifetime of 100 s" in logs[-1]


def test_importing_the_package_leaves_the_bytecode_setting_alone(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
    env["PYTHONPATH"] = str(REPO / "src")
    # The .pyc files of this check go to tmp_path, not into the repo.
    env["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")
    out = subprocess.run([sys.executable, "-c", "import sys; a = sys.dont_write_bytecode; import orbit_verl; "
                          "print(a, sys.dont_write_bytecode)"],
                         capture_output=True, text=True, check=True, env=env).stdout
    assert out.split() == ["False", "False"]


def test_unknown_option_is_an_error():
    with pytest.raises(SystemExit):
        cli.parse_args(["--config", "x", "--bogus"])
    args, overrides = cli.parse_args(["--config", "x", "a=1", "--dry-run", "--", "+b=2"])
    assert args.dry_run and overrides == ["a=1", "+b=2"]


@pytest.mark.parametrize("argv", [
    ["--export", "/e", "orbit.prepare.model.export=/f"],
    ["--model-name", "a", "orbit.prepare.model.name=b"],
    ["--model-name", "a/b"],
    ["--no-prepare", "--prepare"],
    ["--no-prepare", "--force-prepare"],
])
def test_contradicting_options_are_errors(argv):
    with pytest.raises(SystemExit):
        cli.parse_args(["--config", "x", *argv])


def test_launch_script_needs_no_pythonpath(tmp_path):
    # Run from another directory with no PYTHONPATH: the script finds the package by itself.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([sys.executable, str(LAUNCH_SCRIPT), "--help"], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert "--export PATH" in out.stdout and "--runs-root PATH" in out.stdout


# --------------------------------------------------------------------------- pre-flight checks


def _fake_prepared(runs_root: Path, model: str, data: str, *, max_prompt_length: int, prompt_mode: str,
                   export: str, src: str, model_extra: dict | None = None, data_extra: dict | None = None,
                   preset: str = "qwen3", lineage: str | None = None) -> None:
    """A model dir and a dataset dir that pass the launcher's checks, written without any model.

    The frozen embedding row and the lineage follow the preset, as prepare_model records them: no
    frozen row and the lineage release_qwen3 for the release models. ``lineage`` replaces the
    lineage, as for an export prepared by hand with --embedding-padding-idx none (lineage unknown).
    """
    recorded, padding = FAMILY_LINEAGE[preset]
    lineage = lineage or recorded
    model_dir = runs_root / "models" / model
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"embedding_padding_idx": padding}))
    (model_dir / "generation_config.json").write_text("{}")
    (model_dir / "tokenizer.json").write_text(json.dumps({"fake": "tokenizer", "lineage": lineage}))
    for name in MODEL_CODE_FILES:
        shutil.copyfile(MODEL_CODE_DIR / name, model_dir / name)
    common.write_manifest(model_dir / common.MODEL_MANIFEST, common.MODEL_MANIFEST_KIND,
                          {"export_dir": str(export), "embedding_padding_idx": padding, "lineage": lineage,
                           "preset": preset, **(model_extra or {})})
    data_dir = runs_root / "data" / data
    data_dir.mkdir(parents=True)
    for name in ("train.parquet", "val.parquet"):
        (data_dir / name).write_bytes(b"")
    common.write_manifest(data_dir / common.DATA_MANIFEST, common.DATA_MANIFEST_KIND, {
        "max_prompt_length": max_prompt_length, "prompt_mode": prompt_mode, "src": src, "val_src": src,
        "tokenizer_sha256": common.tokenizer_sha256(model_dir), **(data_extra or {})})


def _smoke_prepared(tmp_path: Path, family: str = "qwen3", **kwargs) -> tuple[Path, Path, str]:
    """A fake export and the prepared dirs that rl_sft_mix with ``SINGLE_MODE`` wants for it."""
    runs = tmp_path / "runs_root"
    export = _export(tmp_path, family)
    model = common.model_name_for_export(export)
    _fake_prepared(runs, model, DATA_NAME, max_prompt_length=2048, prompt_mode="sft_think", export=str(export),
                   src=SRC, preset=family, **kwargs)
    return runs, export, model


def _dry_run(config: str, export: Path | None, runs: Path, *extra: str) -> list[str]:
    argv = ["--config", config, "--dry-run", "--gpus-per-node", "1", "--runs-root", str(runs)]
    if export is not None:
        argv += ["--export", str(export)]
    return [*argv, *extra]


def test_dry_run_with_missing_dirs_exits_2_with_prepare_commands(tmp_path, process_state, no_ray, capsys):
    runs = tmp_path / "runs_root"
    export = _export(tmp_path)
    model = common.model_name_for_export(export)
    rc = cli.main(_dry_run(CONFIG, export, runs, *SINGLE_MODE, *NO_PLUGIN))
    out = capsys.readouterr().out
    assert rc == 2
    assert f"-m orbit_verl.model.prepare_model --export {export}" in out
    assert f"--out-dir {runs}/models/{model} --preset qwen3 --runs-root {runs}" in out
    assert f"-m orbit_verl.data.prepare_data --src {SRC} --prompt-mode sft_think" in out
    assert "--max-prompt-length 2048" in out and f"--val-src {SRC}" in out
    assert f"data/{DATA_NAME} " in out
    # The one-go command runs the entry script with the same model and root spelled out.
    assert (f"{LAUNCH_SCRIPT} --config {CONFIG} --export {export} --model-name {model} "
            f"--runs-root {runs} --prepare-only") in out
    assert f"run name:    {CONFIG}__{model} (<config>__<model name>)" in out
    assert "(from --runs-root)" in out and "derived from --export" in out
    assert not runs.exists()   # a dry run creates nothing


@pytest.mark.parametrize("lineage", ["release_qwen3", "unknown"])
def test_dry_run_with_prepared_dirs_passes(lineage, tmp_path, process_state, no_ray, capsys):
    # The recipes' start export prepares as release_qwen3. An export that bundles a padding row that is
    # not zero is prepared by hand with --embedding-padding-idx none, which records the lineage unknown.
    runs, export, _ = _smoke_prepared(tmp_path, lineage=lineage)
    rc = cli.main(_dry_run(CONFIG, export, runs, *SINGLE_MODE, *NO_PLUGIN))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "checks:      all passed" in out
    assert not (runs / "runs").exists()


def test_launcher_refuses_without_a_runs_root(tmp_path, process_state, no_ray, monkeypatch, capsys):
    monkeypatch.delenv(common.RUNS_ROOT_ENV, raising=False)
    rc = cli.main(["--config", CONFIG, "--dry-run", "--gpus-per-node", "1", "--export",
                   str(_export(tmp_path)), *SINGLE_MODE])
    out = capsys.readouterr().out
    assert rc == 2 and f"export {common.RUNS_ROOT_ENV}=/path/to/your/verl_runs" in out


def test_runs_root_from_the_env_and_cli_over_config(tmp_path, process_state, no_ray, monkeypatch, capsys):
    runs, export, _ = _smoke_prepared(tmp_path)
    monkeypatch.setenv(common.RUNS_ROOT_ENV, str(runs))
    rc = cli.main(["--config", CONFIG, "--dry-run", "--gpus-per-node", "1", "--export", str(export),
                   *SINGLE_MODE, *NO_PLUGIN])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"runs root:   {runs} (from {common.RUNS_ROOT_ENV})" in out
    rc = cli.main(["--config", CONFIG, "--dry-run", "--gpus-per-node", "1", "--export", str(export),
                   "--runs-root", str(runs), f"orbit.runs_root={tmp_path / 'elsewhere'}", *SINGLE_MODE,
                   *NO_PLUGIN])
    out = capsys.readouterr().out
    assert rc == 0 and f"runs root:   {runs} (from --runs-root)" in out


def test_launch_without_a_model_says_to_pass_export(tmp_path, process_state, no_ray, capsys):
    rc = cli.main(["--config", CONFIG, "--dry-run", "--gpus-per-node", "1", "--runs-root", str(tmp_path),
                   *SINGLE_MODE])
    assert rc == 2 and "--export PATH" in capsys.readouterr().out


def test_export_without_config_json_is_not_buildable(tmp_path):
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=tmp_path / "not_an_export" / "step_1")
    problems = lprepare.check_model(cfg)
    assert [p.kind for p in problems] == ["model"] and not problems[0].buildable
    assert "has no config.json" in problems[0].message


def test_prepare_only_needs_no_gpu(tmp_path, process_state, no_ray, monkeypatch, capsys):
    runs, export, _ = _smoke_prepared(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rc = cli.main(["--config", CONFIG, "--prepare-only", "--export", str(export), "--runs-root", str(runs),
                   *SINGLE_MODE])
    assert rc == 0, capsys.readouterr().out
    assert not (runs / "runs").exists()


def _record_prepare(monkeypatch) -> list[bool]:
    """Replace run_prepare with a recorder that returns no problems; returns the forces it saw."""
    calls: list[bool] = []

    def fake(cfg, force=False, log=print):
        calls.append(force)
        return []

    monkeypatch.setattr(lprepare, "run_prepare", fake)
    return calls


def test_export_prepares_in_local_mode_only(tmp_path, process_state, no_ray, monkeypatch, capsys):
    runs, export, _ = _smoke_prepared(tmp_path)
    calls = _record_prepare(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert cli.main(["--config", CONFIG, "--prepare-only", "--export", str(export),
                     "--runs-root", str(runs), *SINGLE_MODE]) == 0
    assert cli.main(["--config", CONFIG, "--prepare-only", "--force-prepare", "--export", str(export),
                     "--runs-root", str(runs), *SINGLE_MODE]) == 0
    assert calls == [False, True]
    # A dry run and a --no-prepare launch never prepare.
    cli.main(_dry_run(CONFIG, export, runs, *SINGLE_MODE, *NO_PLUGIN))
    cli.main(_dry_run(CONFIG, export, runs, "--no-prepare", *SINGLE_MODE, *NO_PLUGIN))
    assert calls == [False, True]
    capsys.readouterr()


def test_job_without_prepared_dirs_prints_the_prepare_command(tmp_path, process_state, no_ray, monkeypatch,
                                                              capsys):
    runs = tmp_path / "runs_root"
    export = _export(tmp_path)
    calls = _record_prepare(monkeypatch)
    rc = cli.main(["--config", CONFIG, "--no-prepare", "--gpus-per-node", "1", "--export", str(export),
                   "--runs-root", str(runs), *SINGLE_MODE, *NO_PLUGIN])
    out = capsys.readouterr().out
    assert rc == 2 and calls == []
    assert "This launch never prepares" in out
    assert f"{LAUNCH_SCRIPT} --config {CONFIG} --export {export}" in out and "--prepare-only" in out
    assert DATA_NAME in out


def test_multi_node_launch_refuses_to_prepare(tmp_path, process_state, no_ray, monkeypatch, capsys):
    _set_env(monkeypatch, WORLD_SIZE="2", RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
    rc = cli.main(["--config", CONFIG, "--prepare", "--gpus-per-node", "8", "--export", str(_export(tmp_path)),
                   "--runs-root", str(tmp_path), *SINGLE_MODE])
    assert rc == 2 and "never prepares" in capsys.readouterr().out


def test_dir_built_from_another_export_is_refused_until_renamed_or_forced(tmp_path, process_state, no_ray,
                                                                         monkeypatch, capsys):
    from orbit_verl.model import prepare_model as tool

    runs, old_export, model = _smoke_prepared(tmp_path)
    # Another checkpoint whose path ends in the same two parts, so it derives the same name.
    other = _export(tmp_path / "copy")
    assert common.model_name_for_export(other) == model
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=other)
    problems = lprepare.check_model(cfg)
    assert [p.kind for p in problems] == ["model"] and not problems[0].buildable
    assert f"built from {old_export}, not from {other}" in problems[0].message
    assert "--model-name NAME" in problems[0].message and "--force-prepare" in problems[0].message

    # --export alone leaves the dir alone and reports it; --force-prepare rebuilds it.
    built = []
    monkeypatch.setattr(tool, "prepare_model_dir", lambda export, out_dir, **kw: built.append(kw["force"]))
    monkeypatch.setattr(tool, "check_model_dir", lambda *args, **kw: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    argv = ["--config", CONFIG, "--prepare-only", "--export", str(other), "--runs-root", str(runs), *SINGLE_MODE]
    assert cli.main(argv) == 2 and built == []
    assert "not from" in capsys.readouterr().out
    cli.main([*argv, "--force-prepare"])
    assert built == [True]
    # A new name needs no force: it is simply a new dir.
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=other, model_name="copy_of_run_a")
    assert [p.buildable for p in lprepare.check_model(cfg)] == [True]
    capsys.readouterr()


def test_dataset_for_another_prompt_length_is_rejected(tmp_path, process_state, no_ray, capsys):
    runs, export, model = _smoke_prepared(tmp_path)
    manifest = runs / "data" / DATA_NAME / common.DATA_MANIFEST
    data = common.read_manifest(manifest, common.DATA_MANIFEST_KIND)
    common.write_manifest(manifest, common.DATA_MANIFEST_KIND, {**data, "max_prompt_length": 1024})
    rc = cli.main(_dry_run(CONFIG, export, runs, *SINGLE_MODE, *NO_PLUGIN))
    out = capsys.readouterr().out
    assert rc == 2
    assert "max_prompt_length=1024" in out and "--force" in out and "--force-prepare" in out


def test_prepared_dirs_built_with_other_settings_are_rejected(tmp_path):
    runs, export, model = _smoke_prepared(tmp_path, model_extra={"weights": "symlink"},
                                          data_extra={"limit": 100, "val_size": None})
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=export)
    problems = lprepare.check_prepared(cfg)
    assert [p.kind for p in problems] == ["data"] and "limit=100" in problems[0].message
    assert problems[0].commands[0].endswith(f"--val-src {SRC} --force --runs-root {runs}")

    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "orbit.prepare.model.materialize_weights=true", export=export)
    problems = lprepare.check_model(cfg)
    assert [p.kind for p in problems] == ["model"] and "symlinks" in problems[0].message
    assert "--materialize-weights --force" in problems[0].commands[0]


def test_dataset_from_another_source_is_rejected(tmp_path):
    runs, export, _ = _smoke_prepared(tmp_path)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=export)
    assert lprepare.check_data(cfg) == []
    # Sources compare by their resolved path, so another spelling of the same file is the same source.
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "orbit.prepare.data.src=/data/rl/../rl/math_keep.jsonl",
                   export=export)
    assert lprepare.check_data(cfg) == []
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "orbit.prepare.data.src=/some/file.jsonl", export=export)
    problems = lprepare.check_data(cfg)
    assert [p.kind for p in problems] == ["data"] and f"built from {SRC}" in problems[0].message


def test_dataset_of_the_distill_builder_is_rejected_for_a_single_mode_config(tmp_path):
    runs, export, _ = _smoke_prepared(tmp_path, data_extra={"builder": "distill", "prompt_mode": "mixed"})
    problems = lprepare.check_data(_compose(CONFIG, tmp_path, *SINGLE_MODE, export=export))
    assert [p.kind for p in problems] == ["data"] and "mixed distillation dataset" in problems[0].message
    # The recipe's own builder refuses the single-mode dataset the other way round.
    runs, export, _ = _smoke_prepared(tmp_path / "other")
    cfg = _compose(CONFIG, tmp_path / "other", *RECIPE_DATA, export=export, runs=runs)
    problems = lprepare.check_data(cfg)
    assert [p.kind for p in problems] == ["data"] and "built by prepare_data" in problems[0].message


def test_model_dir_with_outdated_model_code_is_rejected(tmp_path):
    runs, export, model = _smoke_prepared(tmp_path)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=export)
    assert lprepare.check_model(cfg) == []

    copy = runs / "models" / model / "modelling_ouro.py"
    copy.write_text(copy.read_text() + "\n# an older port\n")
    problems = lprepare.check_prepared(cfg)
    assert [p.kind for p in problems] == ["model"]
    assert "modelling_ouro.py" in problems[0].message and "configuration_ouro.py" not in problems[0].message
    assert problems[0].commands[0].endswith(f"--preset qwen3 --force --runs-root {runs}")
    assert not problems[0].buildable

    copy.unlink()
    problems = lprepare.check_model(cfg)
    assert [p.kind for p in problems] == ["model"] and "lacks modelling_ouro.py" in problems[0].message


def test_model_dir_without_a_record_of_the_padding_row_is_rejected(tmp_path):
    runs, export, model = _smoke_prepared(tmp_path)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=export)
    model_dir = runs / "models" / model

    # A config.json without the key, as a dir that this tool did not write would have.
    (model_dir / "config.json").write_text(json.dumps({"pad_token_id": 151643, "vocab_size": 151936}))
    problems = lprepare.check_prepared(cfg)
    assert [p.kind for p in problems] == ["model"] and not problems[0].buildable
    assert "embedding_padding_idx" in problems[0].message
    assert "--preset qwen3 --force" in problems[0].commands[0]

    # The key in config.json alone is not enough: the manifest must record the lineage too.
    (model_dir / "config.json").write_text(json.dumps({"embedding_padding_idx": None}))
    manifest = common.read_manifest(model_dir / common.MODEL_MANIFEST, common.MODEL_MANIFEST_KIND)
    del manifest["lineage"]
    common.write_manifest(model_dir / common.MODEL_MANIFEST, common.MODEL_MANIFEST_KIND, manifest)
    problems = lprepare.check_model(cfg)
    assert [p.kind for p in problems] == ["model"] and "lineage" in problems[0].message


def test_model_dir_whose_config_disagrees_with_its_manifest_is_rejected(tmp_path):
    runs, export, model = _smoke_prepared(tmp_path)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=export)
    (runs / "models" / model / "config.json").write_text(json.dumps({"embedding_padding_idx": 151643}))
    problems = lprepare.check_model(cfg)
    assert [p.kind for p in problems] == ["model"]
    assert "embedding_padding_idx=151643" in problems[0].message and "recorded None" in problems[0].message
    assert "--force" in problems[0].commands[0]


def test_dataset_of_another_tokenizer_is_rejected(tmp_path):
    runs, export, model = _smoke_prepared(tmp_path)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=export)
    assert lprepare.check_prepared(cfg) == []
    # The model pointed at a dataset whose length filter counted other tokens.
    other = runs / "data" / "rl_sft_mix__other"
    shutil.copytree(runs / "data" / DATA_NAME, other)
    manifest = common.read_manifest(other / common.DATA_MANIFEST, common.DATA_MANIFEST_KIND)
    common.write_manifest(other / common.DATA_MANIFEST, common.DATA_MANIFEST_KIND,
                          {**manifest, "tokenizer_sha256": "another tokenizer"})
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "orbit.prepare.data.name=rl_sft_mix__other", export=export)
    problems = lprepare.check_prepared(cfg)
    assert [p.kind for p in problems] == ["data"]
    assert "tokenizer differs" in problems[0].message and "__${orbit.prepare.model.preset}" in problems[0].message


# A config may name the lineages it fits; these overrides narrow rl_sft_mix to the lineage that the
# SFT exports record, so a release export predicted as release_qwen3 does not fit.
ONLY_UNKNOWN = ["orbit.fits.lineages=[unknown]", "orbit.fits.why=only_sft_exports"]


def test_config_for_another_lineage_is_refused_before_preparing(tmp_path, process_state, no_ray, monkeypatch,
                                                                capsys):
    calls = _record_prepare(monkeypatch)
    export = _export(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rc = cli.main(["--config", CONFIG, "--prepare-only", "--export", str(export), "--runs-root", str(tmp_path),
                   *SINGLE_MODE, *ONLY_UNKNOWN])
    out = capsys.readouterr().out
    assert rc == 2 and calls == []
    assert "PROBLEM (lineage)" in out and "the model is release_qwen3 (predicted from the vocabulary size" in out
    assert "only_sft_exports" in out
    # Nothing else is reported: building the missing dirs would not help.
    assert "PROBLEM (model)" not in out and "PROBLEM (data)" not in out


def test_lineage_comes_from_the_manifest_once_prepared(tmp_path):
    runs, export, model = _smoke_prepared(tmp_path, lineage="unknown")
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, *ONLY_UNKNOWN, export=export)
    assert lprepare.model_lineage(cfg) == ("unknown", f"recorded in {runs}/models/{model}")
    assert lprepare.check_lineage(cfg) == [] and lprepare.check_fits(cfg) == []
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "orbit.fits.lineages=[release_qwen3]", export=export)
    assert [p.kind for p in lprepare.check_lineage(cfg)] == ["lineage"]


def test_a_dir_built_from_another_export_does_not_decide_the_lineage(tmp_path, process_state, no_ray, monkeypatch,
                                                                     capsys):
    # A dir of lineage unknown, with a shorter context, sits under the name that another export is
    # pointed at. Its manifest describes its own export, so the new export is judged by its own
    # vocabulary, and it is refused before anything is built or replaced.
    runs = tmp_path / "runs_root"
    sft = _export(tmp_path, run="orbit_sft", step="step_00005000")
    _fake_prepared(runs, "shared_name", DATA_NAME, max_prompt_length=2048, prompt_mode="sft_think",
                   export=str(sft), src=SRC, lineage="unknown", model_extra={"max_position_embeddings": 16384})
    release = _export(tmp_path, run="orbit_release", step="step_00007000")
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, *ONLY_UNKNOWN, export=release, model_name="shared_name",
                   runs=runs)
    assert lprepare.model_lineage(cfg) == ("release_qwen3", f"predicted from the vocabulary size of {release}")
    assert lprepare._context_length(cfg) == (32768, str(release / "config.json"))

    calls = _record_prepare(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rc = cli.main(["--config", CONFIG, "--prepare-only", "--force-prepare", "--export", str(release),
                   "--model-name", "shared_name", "--runs-root", str(runs), *SINGLE_MODE, *ONLY_UNKNOWN, *NO_PLUGIN])
    out = capsys.readouterr().out
    assert rc == 2 and calls == []
    assert "PROBLEM (lineage)" in out and "the model is release_qwen3 (predicted" in out

    # Without the narrowed lineages the export fits, and check_model reports the dir of the other export.
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=release, model_name="shared_name", runs=runs)
    assert lprepare.check_fits(cfg) == []
    problems = lprepare.check_model(cfg)
    assert [p.kind for p in problems] == ["model"]
    assert f"built from {sft}, not from {release}" in problems[0].message


def test_prepare_only_runs_the_checks_that_need_no_gpu(tmp_path, process_state, no_ray, monkeypatch, capsys):
    # A multi-node run prepares with --prepare-only first. A response length beyond the model's
    # context would stop the training launch later, so the preparation must refuse it already.
    runs, export, _ = _smoke_prepared(tmp_path)
    calls = _record_prepare(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rc = cli.main(["--config", CONFIG, "--prepare-only", "--export", str(export), "--runs-root", str(runs),
                   "data.max_response_length=99999", *SINGLE_MODE, *NO_PLUGIN])
    out = capsys.readouterr().out
    assert rc == 2 and calls == [False]
    assert "PROBLEM (lengths)" in out
    # The GPU count is the training launch's business: a CPU-only shell is not refused for it.
    assert "PROBLEM (gpus)" not in out


@pytest.mark.parametrize("key", ["pad_token_id", "embedding_padding_idx"])
def test_padding_overrides_are_rejected(key, tmp_path):
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, f"+actor_rollout_ref.model.override_config.{key}=2",
                   export=_export(tmp_path))
    problems = [p for p in lprepare.check_settings(cfg, environ={}) if p.kind == "config"]
    assert len(problems) == 1 and key in problems[0].message
    assert problems[0].commands == (f"~actor_rollout_ref.model.override_config.{key}",)


def test_nested_model_config_override_is_rejected(tmp_path):
    # verl applies override_config.model_config in place of the top-level keys, so this would
    # freeze the eos row of a release model if the launcher let it through.
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE,
                   "+actor_rollout_ref.model.override_config.model_config.embedding_padding_idx=151643",
                   export=_export(tmp_path))
    problems = [p for p in lprepare.check_settings(cfg, environ={}) if p.kind == "config"]
    assert len(problems) == 1 and "model_config" in problems[0].message
    assert problems[0].commands == ("~actor_rollout_ref.model.override_config.model_config",)


@pytest.mark.parametrize("key,fix", [("path", None), ("hf_config_path", "null"), ("tokenizer_path", "null")])
def test_model_paths_away_from_the_model_dir_are_rejected(key, fix, tmp_path):
    other = tmp_path / "other_model"
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, f"actor_rollout_ref.model.{key}={other}", export=_export(tmp_path))
    problems = [p for p in lprepare.check_settings(cfg, environ={}) if p.kind == "config"]
    assert len(problems) == 1 and f"actor_rollout_ref.model.{key} is {other}" in problems[0].message
    fix = fix or cfg.orbit.model_dir
    assert problems[0].commands == (f"actor_rollout_ref.model.{key}={fix}",)


@pytest.mark.parametrize("key", ["hf_config_path", "tokenizer_path"])
def test_model_side_path_equal_to_the_model_dir_passes(key, tmp_path):
    export = _export(tmp_path)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=export)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, f"actor_rollout_ref.model.{key}={cfg.orbit.model_dir}",
                   export=export)
    assert [p for p in lprepare.check_settings(cfg, environ={}) if p.kind == "config"] == []


@pytest.mark.parametrize("override,fix", [
    ("+actor_rollout_ref.model.override_config.model_config.embedding_padding_idx=151643",
     "~actor_rollout_ref.model.override_config.model_config"),
    ("actor_rollout_ref.model.path={other}", "actor_rollout_ref.model.path={model_dir}"),
    ("actor_rollout_ref.model.hf_config_path={other}", "actor_rollout_ref.model.hf_config_path=null"),
])
def test_dry_run_refuses_overrides_that_bypass_the_model_dir(override, fix, tmp_path, process_state, no_ray,
                                                              capsys):
    # The release dir records no frozen row. Each override would make verl freeze one anyway, or
    # load a model the checks never saw, so the dry run must stop with the fix instead of passing.
    runs, export, model = _smoke_prepared(tmp_path)
    other, model_dir = runs / "models" / "some_other_model", runs / "models" / model
    rc = cli.main(_dry_run(CONFIG, export, runs, *SINGLE_MODE, *NO_PLUGIN, override.format(other=other)))
    out = capsys.readouterr().out
    assert rc == 2, out
    assert fix.format(model_dir=model_dir) in out


def test_a_context_override_reaches_the_prepare_command(tmp_path):
    export = _export(tmp_path, run="orbit_16k", step="step_00001000", max_positions=16384)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "orbit.prepare.model.max_position_embeddings=32768",
                   export=export)
    problems = lprepare.check_model(cfg)
    assert [p.kind for p in problems] == ["model"] and problems[0].buildable
    assert f"--export {export}" in problems[0].commands[0]
    assert "--preset qwen3 --max-position-embeddings 32768" in problems[0].commands[0]
    assert lprepare.check_lengths(cfg) == []


def test_the_recipes_pass_the_length_check(tmp_path):
    cfg = _compose(CONFIG, tmp_path, *RECIPE_DATA, export=_export(tmp_path))
    assert lprepare.check_lengths(cfg) == []


def test_lengths_that_verl_would_cut_or_reject_are_refused(tmp_path):
    export = _export(tmp_path)

    # Token budgets above any length below, so that each case reports only the problem it is about.
    budgets = [f"{key}=32768" for key in ("actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
                                           "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
                                           "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu")]

    def problems(*extra):
        cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "orbit.prepare.model.max_position_embeddings=20480",
                       *budgets, *extra, export=export)
        return lprepare.check_lengths(cfg)

    assert problems() == []
    # 2048 + 20000 tokens do not fit into 20480 positions; the fix keeps the prompt length.
    found = problems("data.max_response_length=20000")
    assert [p.kind for p in found] == ["lengths"] and found[0].commands[0] == "data.max_response_length=18432"
    found = problems("actor_rollout_ref.rollout.max_model_len=1024")
    assert len(found) == 1 and "actor_rollout_ref.rollout.max_model_len" in found[0].message
    found = problems("actor_rollout_ref.rollout.max_model_len=32768")
    assert len(found) == 1 and found[0].commands == ("actor_rollout_ref.rollout.max_model_len=null",)
    found = problems("actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024")
    assert len(found) == 1 and found[0].commands == ("actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432",)
    # Without dynamic batching verl does not use the token budgets, so they are not checked.
    assert problems("actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024",
                    "actor_rollout_ref.actor.use_dynamic_bsz=false") == []


def test_run_prepare_reports_refusals_instead_of_raising(tmp_path, process_state, monkeypatch):
    from orbit_verl.model import prepare_model as tool

    runs = tmp_path / "runs_root"
    export = _export(tmp_path)
    model = common.model_name_for_export(export)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=export)

    def refuse(*args, **kwargs):
        raise tool.PaddingDetectionError("the code freezes row 151643, but that row is not zero")

    monkeypatch.setattr(tool, "prepare_model_dir", refuse)
    problems = lprepare.run_prepare(cfg, log=lambda msg: None)
    assert [p.kind for p in problems] == ["model"] and "that row is not zero" in problems[0].message
    assert problems[0].commands[0].endswith(f"--preset qwen3 --runs-root {runs} --embedding-padding-idx <N|none>")

    # A dir that builds but fails prepare_model's own check is reported with the check command.
    def build(export_dir, out_dir, **kwargs):
        _fake_prepared(runs, model, DATA_NAME, max_prompt_length=2048, prompt_mode="sft_think",
                       export=str(export), src=SRC)

    def failing_check(model_dir, **kwargs):
        raise RuntimeError("config embedding_padding_idx=2 but the manifest says None")

    monkeypatch.setattr(tool, "prepare_model_dir", build)
    monkeypatch.setattr(tool, "check_model_dir", failing_check)
    problems = lprepare.run_prepare(cfg, log=lambda msg: None)
    assert [p.kind for p in problems] == ["model"] and "failed prepare_model's check" in problems[0].message
    assert problems[0].commands[0].endswith(f"--out-dir {runs}/models/{model} --check --runs-root {runs}")


def test_run_prepare_hands_the_runs_root_to_the_data_tool(tmp_path, process_state, monkeypatch):
    from orbit_verl.data import prepare_data

    runs, export, model = _smoke_prepared(tmp_path)
    shutil.rmtree(runs / "data")
    seen = {}

    def fake_convert(src, out_dir, **kwargs):
        seen.update(src=src, out_dir=out_dir, **kwargs)
        raise ValueError("stop here")

    monkeypatch.setattr(prepare_data, "convert", fake_convert)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=export)
    problems = lprepare.run_prepare(cfg, log=lambda msg: None)
    assert [p.kind for p in problems] == ["data"] and "stop here" in problems[0].message
    assert seen["src"] == SRC and seen["val_src"] == SRC and seen["runs_root"] == str(runs)
    assert seen["out_dir"] == f"{runs}/data/{DATA_NAME}"
    assert seen["tokenizer_dir"] == f"{runs}/models/{model}" and seen["prompt_mode"] == "sft_think"


def test_rltt_loss_without_plugin_is_rejected(tmp_path):
    export = _export(tmp_path)
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "orbit.plugin=false", export=export)
    kinds = [p.kind for p in lprepare.check_settings(cfg, environ={})]
    assert kinds == ["plugin"]
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, "trainer.logger=[console,clearml]", export=export)
    assert [p.kind for p in lprepare.check_settings(cfg, environ={})] == ["clearml"]
    assert lprepare.check_settings(cfg, environ={"CLEARML_API_ACCESS_KEY": "k"}) == []


@pytest.mark.parametrize("value", ["0", "-0.1", "true", "abc", "null"])
def test_a_bad_exit_drift_threshold_is_refused_before_launch(value, tmp_path):
    cfg = _compose(CONFIG, tmp_path, *SINGLE_MODE, f"orbit.rltt.exit_drift_alarm={value}", export=_export(tmp_path))
    problems = lprepare.check_settings(cfg, environ={})
    assert [p.kind for p in problems] == ["config"] and "orbit.rltt.exit_drift_alarm" in problems[0].message
    assert problems[0].commands == ("orbit.rltt.exit_drift_alarm=0.3",)


def test_the_rl_recipe_resolves_to_its_learning_rate_schedule(tmp_path):
    """The values of the RL recipe as verl will read them, with verl's own lr scheduler."""
    from types import SimpleNamespace

    import torch
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    cfg = _compose(CONFIG, tmp_path, *RECIPE_DATA, export=_export(tmp_path))
    actor, rollout = cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout
    # k3 KL on the final loop with coefficient 1e-3, gradient clip 0.1, per-response loss mean.
    assert (actor.use_kl_loss, actor.kl_loss_type, actor.kl_loss_coef) == (True, "low_var_kl", 1e-3)
    assert actor.optim.clip_grad == 0.1
    assert actor.loss_agg_mode == "seq-mean-token-mean"
    # 64 prompts x 8 completions, all of them in one optimizer update per step.
    assert (cfg.data.train_batch_size, rollout.n, actor.ppo_mini_batch_size, actor.ppo_epochs) == (64, 8, 64, 1)
    # 16K response tokens everywhere the rollout and the training read a response limit.
    assert cfg.data.max_response_length == rollout.response_length == 16384
    assert cfg.data.max_prompt_length + rollout.response_length <= actor.ppo_max_token_len_per_gpu
    assert lprepare.check_lengths(cfg) == []
    # Every checkpoint is kept, one every 5 steps.
    assert (cfg.trainer.max_actor_ckpt_to_keep, cfg.trainer.save_freq) == (None, 5)
    # The gate is not trained by the RL loss, and the drift alarm watches it closely.
    assert (cfg.orbit.rltt.train_exit_gate, cfg.orbit.rltt.exit_drift_alarm) == (False, 0.05)

    # verl's own scheduler on these optimizer settings: a warmup of 10 steps, then the lr stays at
    # its peak until the end.
    optim = omega_conf_to_dataclass(actor.optim)
    optim.total_training_steps = cfg.trainer.total_training_steps
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=optim.lr)
    scheduler = FSDPEngine._build_lr_scheduler(SimpleNamespace(optimizer_config=optim, rank=1), optimizer)
    lrs = []
    for _ in range(cfg.trainer.total_training_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    warmup = 10
    assert lrs[0] == 0.0 and all(lrs[i] < lrs[i + 1] for i in range(warmup))
    assert all(lr == pytest.approx(1e-6) for lr in lrs[warmup:])


def test_resume_into_a_run_with_another_loss_is_rejected(tmp_path):
    export = _export(tmp_path)
    old = _compose(CONFIG, tmp_path, *SINGLE_MODE, export=export)
    run_dir = Path(old.orbit.run_dir)
    (run_dir / "ckpt").mkdir(parents=True)
    (run_dir / "ckpt" / "latest_checkpointed_iteration.txt").write_text("20")
    (run_dir / "config.yaml").write_text(lconfig.to_yaml(old))
    assert lprepare.check_resume(old) == []
    new = _compose(CONFIG, tmp_path, *SINGLE_MODE, *NO_PLUGIN, export=export)
    problems = lprepare.check_resume(new)
    assert [p.kind for p in problems] == ["resume"] and "rltt_exit -> vanilla" in problems[0].message


# --------------------------------------------------------------------------- simulated topologies


def test_local_plan_strips_torchelastic_vars(tmp_path, process_state, no_ray, monkeypatch, capsys):
    _set_env(monkeypatch, WORLD_SIZE="1", RANK="0", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1",
             MASTER_PORT="29500", TORCHELASTIC_RUN_ID="x")
    cli.main(["--config", CONFIG, "--dry-run", "--gpus-per-node", "8", "--export", str(_export(tmp_path)),
              "--runs-root", str(tmp_path), *SINGLE_MODE, *NO_PLUGIN])
    out = capsys.readouterr().out
    assert "mode:        local" in out
    assert "removed launcher vars: LOCAL_RANK, MASTER_ADDR, MASTER_PORT, RANK, TORCHELASTIC_RUN_ID, WORLD_SIZE" in out
    assert "  trainer.nnodes=1\n" in out and "  trainer.n_gpus_per_node=8\n" in out
    assert "  ray_kwargs.ray_init.address=local\n" in out
    assert not [k for k in os.environ if lenv.is_launcher_var(k)]


def test_head_plan(tmp_path, process_state, no_ray, monkeypatch, capsys):
    _set_env(monkeypatch, WORLD_SIZE="3", RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
    cli.main(["--config", CONFIG, "--dry-run", "--gpus-per-node", "8", "--export", str(_export(tmp_path)),
              "--runs-root", str(tmp_path), *SINGLE_MODE, *NO_PLUGIN])
    out = capsys.readouterr().out
    ip = cluster.own_ip(None)
    assert "mode:        head" in out
    assert f"-m ray.scripts.scripts start --head --port=6379 --node-ip-address={ip} --num-gpus=8" in out
    assert "--include-dashboard=false" in out
    assert f"wait up to 900 s for 3 nodes / 24 GPUs at {ip}:6379" in out
    assert "  trainer.nnodes=3\n" in out and f"  ray_kwargs.ray_init.address={ip}:6379\n" in out
    assert "prepare:     never" in out


def test_worker_plan(tmp_path, process_state, no_ray, monkeypatch, capsys):
    _set_env(monkeypatch, WORLD_SIZE="3", RANK="2", MASTER_ADDR="127.0.0.1", MASTER_PORT="29500")
    rc = cli.main(["--config", CONFIG, "--dry-run", "--gpus-per-node", "8", "--export", str(_export(tmp_path)),
                   "--runs-root", str(tmp_path), *SINGLE_MODE])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mode:        worker" in out
    assert "-m ray.scripts.scripts start --address=127.0.0.1:6379" in out
    assert "no training on this node" in out
    assert "leave after 3 misses)" in out   # no lifetime limit unless --max-lifetime is given
    assert "checks:      skipped on worker nodes" in out


def test_worker_without_master_addr_is_a_config_error(tmp_path, process_state, no_ray, monkeypatch):
    _set_env(monkeypatch, WORLD_SIZE="2", RANK="1")
    assert cli.main(["--config", CONFIG, "--dry-run", "--gpus-per-node", "8", *SINGLE_MODE]) == 2


# --------------------------------------------------------------------------- entry script


def test_launch_script_dry_run_end_to_end(tmp_path):
    runs, export, model = _smoke_prepared(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not lenv.is_launcher_var(k)}
    env[common.RUNS_ROOT_ENV] = str(runs)
    proc = subprocess.run([sys.executable, str(LAUNCH_SCRIPT), "--config", CONFIG, "--export", str(export),
                           "--dry-run", "--gpus-per-node", "1", *SINGLE_MODE, *NO_PLUGIN],
                          capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert f"run dir:     {runs}/runs/{CONFIG}__{model}" in proc.stdout


def test_tools_refuse_without_a_runs_root(tmp_path, monkeypatch):
    from orbit_verl.model import parity_logprobs, prepare_model

    monkeypatch.delenv(common.RUNS_ROOT_ENV, raising=False)
    assert prepare_model.main(["--out-dir", str(tmp_path / "m"), "--check"]) == 2
    assert parity_logprobs.main(["--phase", "hf", "--model-dir", str(tmp_path), "--out-dir", str(tmp_path)]) == 2


# --------------------------------------------------------------------------- stopping a run


_SIGINT_CHILD = """
import signal, sys, time
# What a launcher started with nohup or "&" from a script inherits: an ignored SIGINT.
signal.signal(signal.SIGINT, signal.SIG_IGN)
from orbit_verl.launch import cli

def fake_run_mode(cfg, plan, args, log):
    print("training", flush=True)
    try:
        time.sleep(120)
    except KeyboardInterrupt:
        return 7
    return 0

cli.run_checks = lambda *args, **kwargs: []
cli._run_mode = fake_run_mode
sys.exit(cli.main(sys.argv[1:]))
"""


def test_launcher_stops_on_sigint_even_when_started_with_sigint_ignored(tmp_path):
    # "kill -INT" sends SIGINT. The launcher must turn it into KeyboardInterrupt even
    # when its parent ignored SIGINT, or the run keeps training until the SIGTERM fallback.
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    for key in [k for k in env if lenv.is_launcher_var(k)]:
        del env[key]
    proc = subprocess.Popen(
        [sys.executable, "-c", _SIGINT_CHILD, "--config", CONFIG, "--gpus-per-node", "1",
         "--export", str(_export(tmp_path)), "--runs-root", str(tmp_path / "runs_root"), "--no-prepare",
         *SINGLE_MODE, *NO_PLUGIN],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(tmp_path),
    )
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if line.strip() == "training":
            break
    else:
        proc.wait(timeout=60)
        pytest.fail("the launcher never reached training:\n" + "".join(lines[-30:]))
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=60) == 7
