"""Unit tests of the shared pieces of the long-context data build, on small synthetic data.

They need numpy and mosaicml-streaming but no real data, and run in a few seconds:
  python -m pytest data_prep/long_context/tests
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_ru_open  # noqa: E402
import documents  # noqa: E402
import mds  # noqa: E402
import mix  # noqa: E402
import pack  # noqa: E402
import paths  # noqa: E402
import run_tasks  # noqa: E402
import stage_16k  # noqa: E402
import stage_32k  # noqa: E402

EOS = documents.EOS


def make_doc(rng, length):
    """A document of `length` tokens plus EOS, with ids that are not control tokens."""
    return np.append(rng.integers(0, 1000, length, dtype=np.uint32), np.uint32(EOS))


@pytest.mark.parametrize("block_tokens", [101, stage_16k.BLOCK_TOKENS, stage_32k.BLOCK_TOKENS])
def test_packer_fills_every_block_exactly(block_tokens):
    rng = np.random.default_rng(0)
    blocks = []
    packer = mds.Packer(blocks.append, block_tokens, buffer_docs=64)
    tokens_in = 0
    for _ in range(500):
        doc = make_doc(rng, int(rng.integers(1, 3 * block_tokens)))
        tokens_in += len(doc)
        packer.add(doc)
    packer.flush()
    assert blocks and all(len(b) == block_tokens for b in blocks)
    # Every token goes into a block, except the last partial block that flush drops.
    assert tokens_in == len(blocks) * block_tokens + packer.stats["dropped_tail_tokens"]
    # A block cuts at most one document that is not a full-block window.
    assert packer.stats["cut_docs"] <= len(blocks) - packer.stats["window_blocks"]


def filter_args(**overrides):
    values = dict(length_filter_only=False, allow_think=False, truncate_doc_tokens=0, min_doc_tokens=0,
                  max_doc_tokens=0)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_filter_document_rules():
    rng = np.random.default_rng(1)
    think = np.array(documents.THINK_TOKEN_IDS, np.uint32)
    with_think = np.concatenate([think[:1], make_doc(rng, 10)[:-1], think[1:], [EOS]]).astype(np.uint32)
    stats = {k: 0 for k in ("special_tokens", "truncated", "too_short", "too_long", "duplicates")}

    assert pack.filter_document(with_think, filter_args(), set(), stats) is None
    assert stats["special_tokens"] == 1
    assert pack.filter_document(with_think, filter_args(allow_think=True), set(), stats) is not None
    assert pack.filter_document(with_think, filter_args(length_filter_only=True), set(), stats) is not None

    long_doc = make_doc(rng, 50)
    cut = pack.filter_document(long_doc, filter_args(truncate_doc_tokens=20), set(), stats)
    assert len(cut) == 21 and cut[-1] == EOS and np.array_equal(cut[:20], long_doc[:20])
    assert pack.filter_document(long_doc, filter_args(min_doc_tokens=51), set(), stats) is None
    assert pack.filter_document(long_doc, filter_args(max_doc_tokens=49), set(), stats) is None

    seen = set()
    assert pack.filter_document(long_doc, filter_args(), seen, stats) is not None
    assert pack.filter_document(long_doc.copy(), filter_args(), seen, stats) is None
    assert stats["duplicates"] == 1
    assert pack.filter_document(long_doc.copy(), filter_args(length_filter_only=True), seen, stats) is not None


def test_sample_runs_is_seeded_and_keeps_source_order():
    runs = [("root", None, (i,), 10) for i in range(100)]
    sample = pack.sample_runs(runs, 250, seed=16384)
    assert sample == pack.sample_runs(runs, 250, seed=16384)
    assert len(sample) == 25 and sample == sorted(sample, key=lambda r: r[2])
    assert sample != pack.sample_runs(runs, 250, seed=1)


def test_stream_round_trip_and_merge(tmp_path):
    """Blocks written by StreamWriter read back as the same tokens, and merge_parts joins the parts."""
    rng = np.random.default_rng(2)
    block_tokens = 33
    parts, written = [], []
    for p in range(2):
        part_dir = str(tmp_path / f"part_{p:02d}")
        writer = mds.StreamWriter(part_dir, block_tokens)
        for _ in range(5):
            block = rng.integers(0, 1000, block_tokens, dtype=np.uint32)
            writer(block)
            written.append(block)
        writer.close()
        parts.append(part_dir)
    root, shards = mds.read_shards(parts[0])
    assert np.array_equal(mds.read_shard_tokens(os.path.join(root, shards[0]["raw_data"]["basename"])),
                          np.concatenate(written[:5]))
    assert mds.merge_parts(str(tmp_path), parts) == 10
    assert sorted(os.listdir(tmp_path)) == ["index.json", "shard.00000000.mds", "shard.00000001.mds"]


def test_split_runs_ends_a_run_at_every_partial_shard():
    shards = [{"samples": n} for n in (4, 4, 2, 4, 1, 4, 4)]
    assert mds.split_runs(shards) == [[0, 1, 2], [3, 4], [5, 6]]


@pytest.mark.parametrize("stage", [stage_16k, stage_32k])
def test_stage_definitions_are_consistent(stage):
    assert stage.BLOCK_TOKENS == stage.SEQ_LEN + 1
    assert set(stage.MIX) == set(stage.STREAMS)
    names = {t["name"] for t in run_tasks.stage_tasks(stage)}
    for stream in stage.STREAMS:
        assert {f"pack_{stream}", f"merge_{stream}", f"check_{stream}"} <= names
    for needs in stage.STREAM_NEEDS.values():
        assert set(needs) <= names


def test_32k_mix_shares():
    total = sum(stage_32k.MIX.values())
    long_share = sum(stage_32k.MIX[n] for n in stage_32k.LONG_STREAMS) / total
    stem_share = sum(stage_32k.MIX[n] for n in stage_32k.STEM_SHARES) / total
    assert long_share == pytest.approx(0.40, abs=1e-3)
    assert stem_share == pytest.approx(0.35, abs=1e-3)


def test_ru_sft_store_renders_the_role_files(tmp_path, monkeypatch, tokenizer_root):
    """The ru_sft store reads the chat roles of the user's folder and skips the benchmark roles."""
    from tokenizers import Tokenizer

    monkeypatch.setattr(documents, "_tokenizer",
                        Tokenizer.from_file(str(tokenizer_root / "tokenizer_qwen3_orbit" / "tokenizer.json")))
    monkeypatch.setattr(documents, "_added_token_strings", None)
    folder = tmp_path / "russian_sft"
    folder.mkdir()
    monkeypatch.setattr(paths, "RUSSIAN_SFT", str(folder))

    def write(role, chats):
        lines = [json.dumps({"messages": m}, ensure_ascii=False) for m in chats]
        (folder / f"{role}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sum_chat = [{"role": "system", "content": "Отвечай кратко."},
                {"role": "user", "content": "Сколько будет 2+2?"},
                {"role": "assistant", "content": "<think>\nСложим.\n</think>\n\n4"}]
    write("math", [sum_chat, sum_chat, [{"role": "assistant", "content": "Привет"}]])
    write("dialog", [[{"role": "user", "content": "Привет!"}, {"role": "assistant", "content": "Здравствуйте!"}]])
    write("tasks", [[{"role": "user", "content": "Вопрос"}, {"role": "assistant", "content": "Ответ"}]])

    out = tmp_path / "out"
    args = argparse.Namespace(part=0, nparts=1, limit=0, out=str(out), overwrite=False, twix_hashes=None)
    build_ru_open.build_ru_sft(args)

    manifest = json.loads((out / "ru_sft.manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"] == ["math.jsonl", "dialog.jsonl"]
    assert "tasks.jsonl" not in manifest["missing_role_files"]
    assert "science.jsonl" in manifest["missing_role_files"]
    assert manifest["docs_written"] == 2
    assert manifest["drops"] == {"bad_roles": 1, "duplicate": 1}
    store = out / "ru_sft"
    tok = documents.tokenizer()
    texts = [tok.decode(doc[:-1].tolist(), skip_special_tokens=False)
             for name in documents.docstore_files(str(store))
             for doc in documents.iter_docstore_file(str(store), name)]
    assert texts[0] == "System: Отвечай кратко.\n\nUser: Сколько будет 2+2?\n\nAssistant: <think>\nСложим.\n</think>\n\n4"
    assert texts[1] == "User: Привет!\n\nAssistant: Здравствуйте!"


def test_config_path_keeps_the_data_root_unexpanded():
    assert paths.config_path(os.path.join(paths.DATA, "ctx32k", "streams")) == "${ORBIT_DATA}/ctx32k/streams"
    assert paths.config_path(paths.DATA) == "${ORBIT_DATA}"
    # A sibling directory whose name only starts like the data root is not under it.
    sibling = os.path.abspath(paths.DATA) + "_other"
    assert paths.config_path(sibling) == sibling


def test_mix_writes_the_data_block_with_an_unexpanded_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", str(tmp_path))
    root = tmp_path / "ctx32k"
    monkeypatch.setattr(stage_32k, "ROOT", str(root))
    stream = root / "streams" / "s"
    stream.mkdir(parents=True)
    (stream / "index.json").write_text(json.dumps({"shards": [{"samples": 10}]}))
    shares = tmp_path / "shares.json"
    shares.write_text(json.dumps({"s": 1.0}))
    out = tmp_path / "data.yaml"
    monkeypatch.setattr(sys, "argv", ["mix.py", "--stage", "32k", "--mix", str(shares), "--out", str(out)])
    mix.main()
    lines = out.read_text().splitlines()
    assert lines[0] == "stage1_root: ${ORBIT_DATA}/ctx32k/streams"
    assert "stage1_mixing_mode: mosaic_random" in lines
    assert lines[-1] == "  s: 1.0"


def test_dependents_follow_needs_through_other_tasks():
    tasks = [{"name": "build"}, {"name": "pack", "needs": ["build"]}, {"name": "merge", "needs": ["pack"]},
             {"name": "check", "needs": ["merge"]}, {"name": "other"}]
    assert run_tasks.dependents(tasks, {"pack"}) == {"merge", "check"}
    assert run_tasks.dependents(tasks, {"build"}) == {"pack", "merge", "check"}
    assert run_tasks.dependents(tasks, {"check", "other"}) == set()


# One step of the task chain below. It records that it ran, or fails when a fail_<name> file
# exists in the stage root.
STEP_SCRIPT = """
import os, sys
name, root = sys.argv[1], sys.argv[2]
if os.path.exists(os.path.join(root, "fail_" + name)):
    sys.exit(3)
with open(os.path.join(root, "ran.txt"), "a") as f:
    f.write(name + "\\n")
"""


def chain_tasks(tmp_path):
    """Returns pack (two parts), merge and check of one stream, linked by needs as in stage_tasks."""
    script = tmp_path / "step.py"
    script.write_text(STEP_SCRIPT)

    def task(name, **fields):
        return dict(name=name, cmd=f"{{py}} {script} {name} {{root}}", **fields)

    return [task("pack_x", nparts=2), task("merge_x", needs=["pack_x"]), task("check_x", needs=["merge_x"])]


def run_runner(tmp_path, monkeypatch, tasks):
    """Runs run_tasks.main on a task list in a scratch stage root and returns its exit code.

    The stage's checks of the machine (paths, free disk, packages) are turned off,
    and the runner's one-second poll is shortened so the test stays fast.
    """
    root = tmp_path / "root"
    monkeypatch.setattr(stage_32k, "ROOT", str(root))
    monkeypatch.setattr(stage_32k, "RUN_LOG_DIR", str(root / "logs"))
    monkeypatch.setattr(stage_32k, "PREFLIGHT_PATHS", ())
    monkeypatch.setattr(stage_32k, "MIN_FREE_TB", 0)
    monkeypatch.setattr(run_tasks, "REQUIRED_MODULES", "os")
    real_sleep = time.sleep
    monkeypatch.setattr(run_tasks.time, "sleep", lambda seconds: real_sleep(0.02))
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps({"tasks": tasks}))
    monkeypatch.setattr(sys, "argv", ["run_tasks.py", "--stage", "32k", "--tasks", str(tasks_file), "--workers", "2"])
    with pytest.raises(SystemExit) as exit_info:
        run_tasks.main()
    return exit_info.value.code


def old_markers(tmp_path, *names):
    """Leaves finished-part markers of an earlier build for these one-part tasks."""
    logs = tmp_path / "root" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for name in names:
        (logs / f"{name}.p000.done").write_text("an earlier build\n")
    return logs


def test_runner_creates_a_missing_stage_root(tmp_path, monkeypatch):
    """On a fresh machine nothing has created the 32K root yet, and the runner creates it."""
    root = tmp_path / "data" / "ctx32k"
    monkeypatch.setattr(stage_32k, "ROOT", str(root))
    monkeypatch.setattr(stage_32k, "RUN_LOG_DIR", str(root / "logs"))
    monkeypatch.setattr(stage_32k, "PREFLIGHT_PATHS", ())
    monkeypatch.setattr(stage_32k, "MIN_FREE_TB", 0)
    monkeypatch.setattr(run_tasks, "REQUIRED_MODULES", "os")
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps({"tasks": chain_tasks(tmp_path)}))
    monkeypatch.setattr(sys, "argv", ["run_tasks.py", "--stage", "32k", "--tasks", str(tasks_file), "--dry-run"])
    run_tasks.main()
    assert root.is_dir()


def test_runner_reruns_merge_and_check_after_a_new_pack(tmp_path, monkeypatch):
    old_markers(tmp_path, "merge_x", "check_x")
    assert run_runner(tmp_path, monkeypatch, chain_tasks(tmp_path)) == 0
    assert (tmp_path / "root" / "ran.txt").read_text().split() == ["pack_x", "pack_x", "merge_x", "check_x"]


def test_runner_keeps_old_markers_when_the_new_pack_fails(tmp_path, monkeypatch):
    logs = old_markers(tmp_path, "merge_x", "check_x")
    (tmp_path / "root" / "fail_pack_x").write_text("")
    assert run_runner(tmp_path, monkeypatch, chain_tasks(tmp_path)) == 1
    assert not (tmp_path / "root" / "ran.txt").exists()
    assert (logs / "merge_x.p000.done").exists() and (logs / "check_x.p000.done").exists()


def test_runner_restart_after_a_new_pack_still_merges_and_checks(tmp_path, monkeypatch):
    """A stop between the new pack and the merge must not leave the old merge marker in force."""
    logs = old_markers(tmp_path, "merge_x", "check_x")
    (tmp_path / "root" / "fail_merge_x").write_text("")
    assert run_runner(tmp_path, monkeypatch, chain_tasks(tmp_path)) == 1
    assert not (logs / "merge_x.p000.done").exists() and not (logs / "check_x.p000.done").exists()

    (tmp_path / "root" / "fail_merge_x").unlink()
    (tmp_path / "root" / "ran.txt").unlink()
    assert run_runner(tmp_path, monkeypatch, chain_tasks(tmp_path)) == 0
    assert (tmp_path / "root" / "ran.txt").read_text().split() == ["merge_x", "check_x"]
