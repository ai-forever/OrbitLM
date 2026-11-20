"""CPU tests of the SFT (chat) evaluation of the in-training callback.

Run from the repository root with the trainer's environment:

    PYTHONPATH=src python -m pytest tests/train/test_train_eval_chat.py

The chat evaluation adds suite keys, benchmarks, a model subclass for lm-eval (eval/in_training/
itr_chat.py), the rendering and token-contract helpers (itr_render.py), the prompts, scorers and
detectors (tasks/chat_local.py) and the gates. These tests check each of them on CPU, with stand-ins
for lm-eval and vLLM where the code imports them. They also check that the chat code leaves the base
suites unchanged: tests/train/base_golden.py snapshots their plans, commands, timeouts, fingerprints,
export keys and reported scalar names. Without the benchmarks that base_golden.ADDED_SINCE_REFERENCE
lists, the reference snapshot of the original trainer before the release cleanup must still match;
with them, the snapshot that base_golden.py last wrote must match. The render tests need a Qwen3
tokenizer folder in ORBIT_TEST_TOKENIZER_DIR and are skipped without one. The math scorer test
needs math_verify.
"""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "train"))

import base_golden  # noqa: E402
from orbit import in_training_eval as te  # noqa: E402
# The modules whose own names some tests replace or read, because the code under test looks them up there.
from orbit.in_training_eval import evaluator as te_evaluator  # noqa: E402
from orbit.in_training_eval import preflight as te_preflight  # noqa: E402
from orbit.in_training_eval import scoring as te_scoring  # noqa: E402
from orbit.runtime import report_scalar  # noqa: E402
import test_train_eval as base_tests  # noqa: E402
from test_train_eval import (  # noqa: E402
    BASE_SUITE,
    FakeClearMLLogger,
    ParserError,
    fake_export_recorder,
    make_args,
    make_settings,
    raise_error,
    write_suite,
)

# The fixture with a complete, valid set of evaluation settings, shared with the base tests.
eval_env = base_tests.eval_env

IN_TRAINING = REPO_ROOT / "eval" / "in_training"
SUITES = IN_TRAINING / "suites"
TASKS_DIR = IN_TRAINING / "tasks"
SFT_SUITE = SUITES / "sft.yaml"
CHAT_SUITES = ("sft.yaml", "sft_smoke.yaml")

THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_gen_toks": 16384}
CHAT_SECTION = (
    "chat:\n"
    "  think: {temperature: 0.6, top_p: 0.95, top_k: 20, max_gen_toks: 16384,\n"
    "          repetition_detection: {min_pattern_size: 64, max_pattern_size: 128, min_count: 16}}\n"
    "  no_think: {temperature: 0.7, top_p: 0.8, top_k: 20, presence_penalty: 1.5, max_gen_toks: 4096}\n"
    "  no_prefill: {temperature: 0.6, top_p: 0.95, top_k: 20, max_gen_toks: 64}\n"
)


def load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chat_local = load_module(TASKS_DIR / "chat_local.py", "test_chat_local")
itr_render = load_module(IN_TRAINING / "itr_render.py", "test_itr_render")


def chat_suite(tmp_path: Path, benchmarks: str, chat: str = CHAT_SECTION, entry: Path | None = None) -> te.Suite:
    return te.load_suite(write_suite(tmp_path, chat + "benchmarks:\n" + benchmarks, entry=entry))


# ---------------------------------------------------------------------------
# The base suites do not move
# ---------------------------------------------------------------------------


def test_base_and_smoke_suites_without_the_added_benchmarks_match_the_reference_snapshot():
    """Everything the golden snapshot holds for base.yaml and smoke.yaml without MBPP, except the fingerprint.

    The fingerprint hashes the suite file among other inputs, and the suite file with MBPP differs
    from the one the reference was taken with. The next test compares it with base_golden.json.
    """
    reference = json.loads(base_golden.REFERENCE_PATH.read_text())
    current = base_golden.all_snapshots(te, without=base_golden.ADDED_SINCE_REFERENCE)
    assert set(current) == set(reference)
    assert base_golden.differing_parts(current, reference, skip=base_golden.NOT_IN_REFERENCE) == []
    assert base_golden.NOT_IN_REFERENCE == ("fingerprint",)


def test_base_and_smoke_suites_match_their_golden_snapshot():
    """The same parts and the fingerprint of base.yaml and smoke.yaml as they are, MBPP included."""
    golden = json.loads(base_golden.GOLDEN_PATH.read_text())
    current = base_golden.all_snapshots(te)
    assert set(current) == set(golden)
    for key, parts in golden.items():
        for part, value in parts.items():
            assert current[key][part] == value, f"{key}: {part} changed"


def test_base_preflight_imports_are_unchanged(tmp_path):
    settings = make_settings(te.load_suite(BASE_SUITE), tmp_path)
    assert te.preflight_modules(settings) == [
        "torch", "lm_eval", "lm_eval.evaluator", "vllm", "vllm.platforms.cuda", "lm_eval.models.vllm_causallms",
        "sympy", "math_verify", "wonderwords", "sqlite3", "nltk", "evaluate", "+multiprocessing_manager",
    ]


def test_base_summaries_and_exports_get_no_chat_keys(tmp_path):
    scores = te.Scores(benchmarks={"mmlu": 0.5})
    assert set(scores.to_dict()) == {"benchmarks", "parts", "ruler", "samples", "incomplete"}
    settings = make_settings(te.load_suite(BASE_SUITE), tmp_path)
    assert settings._chat_payload() == {}
    evaluator = te.InTrainingEval(settings, rank=0, local_rank=0, world_size=1, distributed=False, device=None,
                                  run_dir=tmp_path / "run")
    assert evaluator._chat_export_metadata() == {}


# ---------------------------------------------------------------------------
# Suite files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CHAT_SUITES)
def test_repository_chat_suites_load(name):
    suite = te.load_suite(SUITES / name)
    assert suite.chat is not None
    think = suite.chat.presets["think"]
    assert think["do_sample"] is True and think["temperature"] == 0.6 and "seed" not in think
    assert think["repetition_detection"] == {"min_pattern_size": 64, "max_pattern_size": 128, "min_count": 16}
    units = te.expand_units(suite)
    for unit in units:
        for task in unit.tasks:
            if task.startswith("itr_"):
                assert (suite.tasks_dir / f"{task}.yaml").is_file(), task
    # Every chat unit leaves 2,048 tokens of prompt room in the 32K window of the SFT runs.
    assert max(te.signature_budget(unit.signature) for unit in units) + te.CHAT_PROMPT_ROOM <= 32768


def test_sft_suite_sets_the_planned_protocol():
    suite = te.load_suite(SFT_SUITE)
    assert suite.chat.lead_eos is True
    assert suite.chat.presets["no_think"] == {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_gen_toks": 4096, "do_sample": True,
    }
    assert suite.chat.presets["no_prefill"]["max_gen_toks"] == 64
    assert suite.chat.gates["g2_max"] == 0.005 and suite.chat.gates["g11_min"] == 0.98
    assert suite.benchmarks["chatprobe_think"].max_gen_toks == 8192
    assert {"mmlu", "hellaswag", "humaneval", "mbpp"} <= set(suite.benchmarks)


@pytest.mark.parametrize(
    "chat, benchmarks, message",
    [
        (CHAT_SECTION.replace("max_gen_toks: 4096}", "max_gen_toks: 4096, seed: 1}"), "  math500_nothink: {}\n",
         "sets a seed"),
        (CHAT_SECTION.replace("max_gen_toks: 4096}", "max_gen_toks: 4096, beam: 2}"), "  math500_nothink: {}\n",
         "unknown keys"),
        (CHAT_SECTION.replace(", max_gen_toks: 4096}", "}"), "  math500_nothink: {}\n", "needs max_gen_toks"),
        ("", "  math500_nothink: {}\n", "need a chat section"),
        (CHAT_SECTION, "  mmlu: {}\n", "no chat benchmark"),
        ("chat:\n  think: {temperature: 0.6, max_gen_toks: 128}\n", "  math500_nothink: {}\n", r"presets for \['no_think'\]"),
        (CHAT_SECTION, "  mmlu: {max_gen_toks: 512}\n  math500_think: {}\n", "unknown keys"),
        (CHAT_SECTION.replace("min_count: 16", "min_count: 1"), "  math500_think: {}\n", "min_count >= 2"),
        (CHAT_SECTION + "  lead: true\n", "  math500_think: {}\n", "unknown chat keys"),
        (CHAT_SECTION + "  gates: {g99: 1}\n", "  math500_think: {}\n", "chat.gates may only set"),
        (CHAT_SECTION + "  per_prompt_seed: 7\n", "  math500_think: {}\n", "unknown chat keys"),
        (CHAT_SECTION, "  think_open: {shards: 2}\n", "cannot be sharded"),
    ],
)
def test_bad_chat_suites_are_refused(tmp_path, chat, benchmarks, message):
    with pytest.raises(ValueError, match=message):
        chat_suite(tmp_path, benchmarks, chat)


# ---------------------------------------------------------------------------
# Units, plan and commands
# ---------------------------------------------------------------------------


def test_chat_units_carry_mode_sampling_and_budget_in_their_signature(tmp_path):
    suite = chat_suite(tmp_path, "  math500_think: {shards: 2}\n  math500_nothink: {limit: 3}\n"
                                 "  chatprobe_think: {max_gen_toks: 8192}\n  mmlu: {limit: 2}\n")
    units = {unit.name: unit for unit in te.expand_units(suite)}
    think = units["math500_think/itr_chat_math500#1of2"]
    assert think.signature[:3] == ("--apply_chat_template", "--log_samples", "--model_args")
    assert think.signature[3] == "itr_mode=think,itr_lead_eos=True"
    preset = json.loads(think.signature[think.signature.index("--gen_kwargs") + 1])
    assert preset == {**THINK, "do_sample": True,
                      "repetition_detection": {"min_pattern_size": 64, "max_pattern_size": 128, "min_count": 16}}
    nothink = units["math500_nothink/itr_chat_math500"]
    assert "itr_mode=no_think" in nothink.signature[3] and nothink.signature[-2:] == ("--limit", "3")
    probe = units["chatprobe_think/itr_chat_probe"]
    assert te.signature_budget(probe.signature) == 8192
    # The tail of one response at the budget is part of every chat unit's cost.
    assert think.cost == pytest.approx(250 * 5.0 + 16384 * te.CHAT_TAIL_SECONDS_PER_TOKEN)
    assert units["mmlu/mmlu_anatomy"].signature == ("--num_fewshot", "5", "--limit", "2")


def test_chat_and_continuity_processes_get_their_own_commands(tmp_path):
    suite = chat_suite(tmp_path, "  math500_think: {}\n  hellaswag: {shards: 2, limit: 4}\n")
    plan = te.build_plan(suite, 1)
    processes = {process.units[0].benchmark: process for process in plan.processes[0]}
    model_dir = tmp_path / "model"
    chat = te.build_lm_eval_command(
        processes["math500_think"], eval_python=Path("/py"), suite=suite, model_dir=model_dir, out_dir=tmp_path / "o",
        max_model_len=32768, gpu_memory_utilization=0.7, samples_path=None,
    )
    assert chat[chat.index("--model") + 1] == te.CHAT_MODEL
    model_args = chat[chat.index("--model_args") + 1]
    assert model_args.startswith(f"pretrained={model_dir},dtype=bfloat16,")
    assert model_args.endswith(",seed=1234,itr_mode=think,itr_lead_eos=True")
    assert chat.count("--model_args") == 1
    assert "--apply_chat_template" in chat and "--log_samples" in chat
    assert json.loads(chat[chat.index("--gen_kwargs") + 1])["do_sample"] is True
    base = te.build_lm_eval_command(
        processes["hellaswag"], eval_python=Path("/py"), suite=suite, model_dir=model_dir, out_dir=tmp_path / "o",
        max_model_len=32768, gpu_memory_utilization=0.7, samples_path=tmp_path / "s.json",
    )
    assert base[base.index("--model") + 1] == "vllm"
    assert "--apply_chat_template" not in base and "--log_samples" not in base and "--gen_kwargs" not in base
    assert processes["math500_think"].is_chat and not processes["hellaswag"].is_chat


def test_no_think_rank_runs_a_second_process_in_the_sft_pause():
    """At 64 GPUs every think process of sft.yaml has a rank of its own (the planner's tail cost)."""
    suite = te.load_suite(SFT_SUITE)
    plan = te.build_plan(suite, 64)
    for processes in plan.processes:
        modes = [te.BENCHMARKS[process.units[0].benchmark].chat_mode for process in processes]
        if "think" in modes:
            assert len(processes) == 1, [process.name for process in processes]
    assert plan.makespan_seconds < 15 * 60


def test_chat_process_timeouts_cover_two_responses_at_the_budget(tmp_path):
    suite = te.load_suite(SFT_SUITE)
    settings = make_settings(suite, tmp_path, timeout_minutes=30.0)
    plan = te.build_plan(suite, 64)
    for processes in plan.processes:
        for process in processes:
            timeout = settings.process_timeout(process)
            if process.budget:
                assert timeout >= suite.process_startup_seconds + 2 * process.budget * te.CHAT_TIMEOUT_SECONDS_PER_TOKEN
            else:
                assert timeout == min(30 * 60 * len(process.units), 3 * (process.cost + suite.process_startup_seconds) + 600)


def write_tokenizer(folder: Path, template: str | None = "{{ messages }}") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tokenizer.json").write_text("{}")
    (folder / "tokenizer_config.json").write_text(json.dumps({
        "eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>", "bos_token": None,
        "added_tokens_decoder": {"151643": {"content": "<|endoftext|>"}, "151645": {"content": "<|im_end|>"}},
    }))
    if template is not None:
        (folder / "chat_template.jinja").write_text(template)
    return folder


def test_chat_suites_need_a_template_and_room_for_the_prompt(eval_env, tmp_path):
    suite = write_suite(tmp_path / "s", CHAT_SECTION + "benchmarks:\n  math500_think: {}\n")
    with pytest.raises(ParserError, match="has no chat template"):
        te.validate_eval_arguments(make_args(eval_env, eval_suite=suite, block_size=32768), raise_error)
    tokenizer = write_tokenizer(tmp_path / "tok")
    args = make_args(eval_env, eval_suite=suite, block_size=16384, eval_tokenizer_source=tokenizer)
    with pytest.raises(ParserError, match="leaves less than 2048 tokens"):
        te.validate_eval_arguments(args, raise_error)
    args = make_args(eval_env, eval_suite=suite, block_size=32768, eval_tokenizer_source=tokenizer)
    te.validate_eval_arguments(args, raise_error)
    settings = te.EvalSettings.from_args(args)
    assert settings.chat_template_sha256 == itr_render.template_sha256(tokenizer)


def test_the_template_is_part_of_the_keys_of_chat_suites_only(tmp_path):
    suite = chat_suite(tmp_path, "  math500_think: {}\n")
    one = make_settings(suite, tmp_path, chat_template_sha256="a")
    two = make_settings(suite, tmp_path, chat_template_sha256="b")
    plan = te.build_plan(suite, 1)
    assert one.fingerprint(plan) != two.fingerprint(plan)
    assert one.export_key() != two.export_key()
    base = make_settings(te.load_suite(BASE_SUITE), tmp_path)
    assert base.fingerprint(te.build_plan(base.suite, 1)) == make_settings(
        te.load_suite(BASE_SUITE), tmp_path, chat_template_sha256="x").fingerprint(te.build_plan(base.suite, 1))


def test_chat_preflight_checks_the_imports_and_renders_the_template(tmp_path):
    train = write_tokenizer(tmp_path / "train")
    settings = make_settings(te.load_suite(SFT_SUITE), tmp_path, tokenizer_source=tmp_path / "ship",
                             tokenizer_config=train / "tokenizer_config.json")
    modules = te.preflight_modules(settings, cuda=True)
    assert modules[:3] == ["torch", "lm_eval", "lm_eval.evaluator"]
    assert {"sympy", "math_verify", "immutabledict", "langdetect", te.NLTK_PUNKT_CHECK} <= set(modules)
    assert modules.index(te.NLTK_PUNKT_CHECK) < modules.index("lm_eval.tasks.ifeval.instructions_util")
    render = [module for module in modules if module.startswith(te.CHAT_RENDER_CHECK + "=")]
    assert len(render) == 1 and modules[-1] == te.CUDA_CHECK
    spec = json.loads(render[0].split("=", 1)[1])
    assert spec["tokenizer"] == str(tmp_path / "ship") and spec["train_tokenizer"] == str(train.resolve())
    assert Path(spec["module_dir"]) == IN_TRAINING
    # A training config outside a tokenizer folder (the package's default) is not compared.
    package = make_settings(te.load_suite(SFT_SUITE), tmp_path, tokenizer_config=tmp_path / "pkg" / "tokenizer_config.json")
    render = [module for module in te.preflight_modules(package) if module.startswith(te.CHAT_RENDER_CHECK + "=")]
    assert json.loads(render[0].split("=", 1)[1])["train_tokenizer"] is None


# ---------------------------------------------------------------------------
# Clopper-Pearson bounds
# ---------------------------------------------------------------------------


def test_clopper_pearson_matches_known_intervals():
    lower, upper = te.clopper_pearson(0, 500)
    assert lower == 0.0 and upper == pytest.approx(1 - 0.025 ** (1 / 500), abs=1e-9)  # 0.00735
    lower, upper = te.clopper_pearson(500, 500)
    assert upper == 1.0 and lower == pytest.approx(0.025 ** (1 / 500), abs=1e-9)
    # A textbook interval: 3 successes in 10 trials.
    lower, upper = te.clopper_pearson(3, 10)
    assert lower == pytest.approx(0.06674, abs=1e-4) and upper == pytest.approx(0.65245, abs=1e-4)
    # Symmetry: failures and successes swap the bounds.
    a = te.clopper_pearson(37, 1319)
    b = te.clopper_pearson(1319 - 37, 1319)
    assert a[0] == pytest.approx(1 - b[1], abs=1e-9) and a[1] == pytest.approx(1 - b[0], abs=1e-9)


def test_what_the_pooled_samples_allow():
    # G2 over about 2,600 think responses passes with at most 5 failures; G11 needs 183 answers.
    assert te.clopper_pearson(5, 2579)[1] <= 0.005 < te.clopper_pearson(6, 2579)[1]
    assert te.smallest_n_that_can_pass(0.98) == 183
    assert te.clopper_pearson(183, 183)[0] >= 0.98 > te.clopper_pearson(182, 182)[0]


# ---------------------------------------------------------------------------
# Prompts, extraction and detectors (tasks/chat_local.py)
# ---------------------------------------------------------------------------


def blank(token: int) -> bool:
    return token in (198, 271, 220)


def test_the_answer_is_the_text_after_the_last_think_close():
    assert chat_local.split_answer("plan </think>\n\nThe answer is 4.", "think") == "\n\nThe answer is 4."
    assert chat_local.split_answer("a </think> b </think> c", "think") == " c"
    assert chat_local.split_answer("still thinking", "think") is None
    assert chat_local.split_answer("still thinking", "no_prefill") is None
    assert chat_local.split_answer("plain answer<|im_end|>", "no_think") == "plain answer"
    assert chat_local.split_answer("leak</think>answer", "no_think") == "answer"


def test_prompts_carry_the_format_instructions():
    assert chat_local.math_prompt({"problem": "1+1?"}).endswith("put your final answer within \\boxed{}.")
    assert chat_local.mgsm_ru_prompt({"question": "Сколько?"}).endswith("окончательный ответ в \\boxed{}.")
    prompt = chat_local.mmlu_redux_prompt({"question": "Q?", "choices": ["w", "x", "y", "z"]})
    assert "\nA. w\nB. x\nC. y\nD. z\n" in prompt and prompt.endswith('`"answer": "C"`.')


def test_mcq_letters_are_read_from_the_end():
    assert chat_local.extract_choice('so {"answer": "B"} or "answer": "D"') == ("D", True)
    assert chat_local.extract_choice("The answer is (C).") == ("C", False)
    assert chat_local.extract_choice("Reasons...\n\n(B)") == ("B", False)
    assert chat_local.extract_choice("B") == ("B", False)
    # The article "A" at the start of a sentence is not an answer.
    assert chat_local.extract_choice("A good question. I am not sure.") == (None, False)
    assert chat_local.extract_choice("I think the answer is a bit unclear.") == (None, False)
    assert chat_local.extract_choice("Answer: Avogadro's number.") == (None, False)
    assert chat_local.extract_choice(None) == (None, False)


@pytest.mark.parametrize("text, letter", [
    # Answer endings that Qwen3-1.7B wrote in no_think mode on MMLU-Redux in a GPU calibration run.
    ("answer: C", "C"),
    ('answer: "B"', "B"),
    ('`answer": "D"`', "D"),
    ("the correct answer is:\n\n```plaintext\nanswer: A\n```", "A"),
    ("Thus, the correct answer is:\n\n**answer**: \"B\"", "B"),
    ("### Final Answer:\n\n$$\n\\boxed{\\text{C}}\n$$", "C"),
    ("So the correct answer is:\n\n$$\n\\boxed{D}\n$$", "D"),
    ("**Answer:** `A`", "A"),
    ("### Answer: **B. 0.60c**", "B"),
    ("### ✅ Correct choice: **A. 1**", "A"),
    ("Option B is wrong, so the answer is C.\n\n### Final\n**C. False, True**", "C"),
    ("Statement 1 is **true**.\n\n### Correct Answer\n**D. True, False**", "D"),
])
def test_mcq_letters_in_the_forms_models_write(text, letter):
    assert chat_local.extract_choice(text) == (letter, False)


def test_math_answers_are_verified_on_the_answer_part():
    pytest.importorskip("math_verify")
    assert chat_local.math_correct("so \\boxed{18}.", "18") == 1
    assert chat_local.math_correct("so \\boxed{17}.", "18") == 0
    assert chat_local.math_correct(None, "18") == 0
    doc = {"itr_gold": "The result is $\\boxed{\\frac{1}{2}}$."}
    assert chat_local.score_response("itr_chat_math500", doc, "x</think>\n\nSo $\\boxed{\\frac{1}{2}}$.", "think") == 1.0
    # A number in the unclosed reasoning is never read as the answer.
    assert chat_local.score_response("itr_chat_math500", doc, "maybe \\boxed{1/2} ... still", "think") == 0.0


def test_token_facts_of_a_well_formed_think_response():
    reasoning = [1000 + i for i in range(40)]
    ids = [198] + reasoning + [198, chat_local.THINK_CLOSE, 271, 5, 6, chat_local.IM_END]
    facts = chat_local.token_facts(ids, "think", "stop", blank)
    assert facts["closed"] and facts["reasoning_len"] == 40  # the blank ends are trimmed
    assert facts["format_ok"] and not facts["stop_inside"]
    assert facts["answer_len"] == 3 and facts["answer_nonblank"]


@pytest.mark.parametrize(
    "ids, finish, expect",
    [
        # Closed after 5 tokens: a short think (G2) but a well-formed response (G4).
        ([1, 2, 3, 4, 5, chat_local.THINK_CLOSE, 271, 9, chat_local.IM_END], "stop",
         {"closed": True, "reasoning_len": 5, "format_ok": True, "stop_inside": False}),
        # Ended on <|endoftext|> inside the think block (G2), not well-formed.
        ([1, 2, 3, chat_local.END_OF_TEXT], "stop", {"closed": False, "stop_inside": True, "format_ok": False}),
        # Hit the budget inside the think block: truncated, not "stop inside".
        ([1, 2, 3, 4], "length", {"closed": False, "stop_inside": False, "format_ok": False}),
        # Two closes, and a re-opened block.
        ([1, chat_local.THINK_CLOSE, 2, chat_local.THINK_CLOSE, 3, chat_local.IM_END], "stop", {"format_ok": False}),
        ([1, chat_local.THINK_CLOSE, chat_local.THINK_OPEN, 2, chat_local.IM_END], "stop", {"format_ok": False}),
        # Ended on <|endoftext|> after the answer instead of <|im_end|>.
        ([1, chat_local.THINK_CLOSE, 2, chat_local.END_OF_TEXT], "stop", {"format_ok": False}),
        # Closed, then nothing but blanks.
        ([1, chat_local.THINK_CLOSE, 271, chat_local.IM_END], "stop", {"format_ok": False}),
    ],
)
def test_token_facts_flag_the_failure_modes(ids, finish, expect):
    facts = chat_local.token_facts(ids, "think", finish, blank)
    for key, value in expect.items():
        assert facts[key] == value, key


def test_token_facts_of_no_think_and_no_prefill_responses():
    ok = chat_local.token_facts([5, 6, chat_local.IM_END], "no_think", "stop", blank)
    assert ok["format_ok"] and ok["closed"] is None and ok["answer_len"] == 2
    leak = chat_local.token_facts([chat_local.THINK_OPEN, 5, chat_local.THINK_CLOSE, 6, chat_local.IM_END],
                                  "no_think", "stop", blank)
    assert not leak["format_ok"] and leak["n_open"] == 1 and leak["n_close"] == 1
    opens = [chat_local.THINK_OPEN, chat_local.NEWLINE] + list(range(1000, 1062))
    facts = chat_local.token_facts(opens, "no_prefill", "length", blank)
    assert facts["np_opened"] and facts["np_ok"] and facts["reasoning_len"] == 62
    early = [chat_local.THINK_OPEN, chat_local.NEWLINE] + list(range(1000, 1010)) + [chat_local.THINK_CLOSE] + list(range(50))
    assert not chat_local.token_facts(early, "no_prefill", "length", blank)["np_ok"]
    direct = list(range(1000, 1064))
    facts = chat_local.token_facts(direct, "no_prefill", "length", blank)
    assert not facts["np_opened"] and not facts["np_ok"]


def test_token_facts_use_the_ids_of_the_vocabulary_in_use():
    assert chat_local.vocabulary_ids(ToyTokenizer()) == chat_local.QWEN3_IDS
    # A vocabulary with other ids: <|endoftext|> 0, <|im_end|> 2, <think> 3, </think> 4.
    ouro = {"end_of_text": 0, "im_end": 2, "think_open": 3, "think_close": 4, "newline": 198}
    to_ouro = {chat_local.END_OF_TEXT: 0, chat_local.IM_END: 2, chat_local.THINK_OPEN: 3, chat_local.THINK_CLOSE: 4}
    cases = [
        ([1000, 1001, chat_local.THINK_CLOSE, 271, 1002, chat_local.IM_END], "think", "stop"),
        ([1000, 1001, chat_local.END_OF_TEXT], "think", "stop"),
        ([chat_local.THINK_OPEN, chat_local.NEWLINE] + list(range(1000, 1062)), "no_prefill", "length"),
    ]
    for ids, mode, finish in cases:
        qwen = chat_local.token_facts(ids, mode, finish, blank)
        mapped = [to_ouro.get(token, token) for token in ids]
        other = chat_local.token_facts(mapped, mode, finish, blank, ouro)
        assert {k: v for k, v in other.items() if k != "last_id"} == {k: v for k, v in qwen.items() if k != "last_id"}

    class SplitTokenizer(ToyTokenizer):
        def encode(self, text, add_special_tokens=True):
            return [1, 2] if text == "</think>" else super().encode(text)

    with pytest.raises(ValueError, match="not a single token"):
        chat_local.vocabulary_ids(SplitTokenizer())


def test_the_loop_detector_sees_late_loops_and_spares_clean_traces():
    rng = random.Random(0)
    clean = [rng.randrange(1000, 60000) for _ in range(16000)]
    late = clean[:14500] + [7, 8, 9, 10] * 20 + [rng.randrange(1000, 60000) for _ in range(3)] + list(range(500, 600)) * 13
    detectors = dict(te.DEFAULT_DETECTORS)
    clean_facts = chat_local.token_facts(clean, "think", "length", blank)
    late_facts = chat_local.token_facts(late, "think", "length", blank)
    assert clean_facts["rep16_tail"] == 0.0 and not te_scoring._is_loop({**clean_facts}, detectors)
    # The loop covers the last 1,300 tokens: the 1,024-token tail sees it, the 4,096 window does not.
    assert late_facts["rep16_tail"] >= 0.5 and te_scoring._is_loop(late_facts, detectors)
    assert late_facts["rep16_w4k"] < 0.5 and not te_scoring._is_loop_w4k(late_facts, detectors)
    assert te_scoring._is_loop({"finish": "repetition", "n_gen": 10}, detectors)
    assert not te_scoring._is_loop({**late_facts, "n_gen": 300}, detectors)  # too short to call


def test_markers_opener_and_language_of_the_answer():
    assert chat_local.count_markers("Wait, hmmm. Let me double-check. Let me verify. On second thought, ok.") == 5
    assert chat_local.count_markers("Хм, подожди. Давайте проверим. Стоп. перепроверю, погоди") == 6
    assert chat_local.count_markers("We waited for the train.") == 0
    facts = chat_local.text_facts("r</think>\n\nOkay, so the answer is four, because two plus two is four.", "think")
    assert facts["opener"] and facts["lang"] == "en"
    russian = "</think>Ответ: площадь равна $\\frac{a^2 \\sqrt{3}}{4}$, то есть \\boxed{\\frac{\\sqrt3}{4}} при a=1."
    assert chat_local.text_facts(russian, "think")["lang"] == "ru"
    assert chat_local.letter_counts("$x^2 + y^2$ ```python\nprint(1)\n``` see https://example.com") == (3, 0)
    assert chat_local.answer_language("\\boxed{42}") is None  # undetermined, not English
    assert chat_local.text_facts("thinking only", "think")["lang"] is None


def test_the_chat_probe_takes_200_english_and_every_russian_root_prompt():
    messages = [
        {"message_id": f"m{i:04d}", "parent_id": None, "role": "prompter", "lang": "en" if i % 3 else "ru", "text": f"t{i}"}
        for i in range(400)
    ] + [{"message_id": "z", "parent_id": "m0001", "role": "assistant", "lang": "en", "text": "reply"},
         {"message_id": "y", "parent_id": None, "role": "prompter", "lang": "es", "text": "hola"}]
    rows = chat_local.select_probe_prompts(list(reversed(messages)))
    english = [row for row in rows if row["itr_lang"] == "en"]
    russian = [row for row in rows if row["itr_lang"] == "ru"]
    assert len(english) == 200 and english[0]["source"] == "oasst2:m0001"
    assert len(russian) == 134 and all(row["itr_domain"] == "chat" for row in rows)
    trivial = chat_local.read_trivial_prompts()
    assert len(trivial) == 100
    assert sum(row["itr_lang"] == "ru" for row in trivial) == 50
    # The OASST2 validation split at the revision that chat_local.OASST2 pins has 275 English and 60
    # Russian root prompts.
    assert te.CHAT_TASK_DOCS["itr_chat_probe"] == 200 + 60 + len(trivial)


class FakeDataset:
    """The two datasets.Dataset methods that the process_docs functions use."""

    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, keep):
        return FakeDataset(row for row in self.rows if keep(row))

    def map(self, function):
        return FakeDataset(function(row) for row in self.rows)


def test_ifeval_keeps_every_prompt_including_the_langdetect_checks():
    rows = [
        {"instruction_id_list": ["punctuation:no_comma"]},
        {"instruction_id_list": ["language:response_language"]},
        {"instruction_id_list": ["detectable_format:title", "change_case:english_capital"]},
        {"instruction_id_list": ["change_case:english_lowercase"]},
    ]
    kept = chat_local.process_ifeval(FakeDataset(rows)).rows
    assert [row["instruction_id_list"] for row in kept] == [row["instruction_id_list"] for row in rows]
    assert all(row["itr_lang"] == "en" and row["itr_domain"] == "if" for row in kept)
    assert te.CHAT_TASK_DOCS["itr_chat_ifeval"] == 541
    gsm = chat_local.process_gsm8k(FakeDataset([{"question": "q", "answer": "work\n#### 1,234"}])).rows[0]
    assert gsm["itr_gold"] == "1234"
    mgsm = chat_local.process_mgsm_ru(FakeDataset([{"question": "q", "answer_number": 18.0}])).rows[0]
    assert mgsm["itr_gold"] == "18" and mgsm["itr_lang"] == "ru"


def test_ifeval_scoring_names_a_missing_langdetect_and_seeds_a_present_one(monkeypatch):
    # A None entry in sys.modules makes "import langdetect" fail, as it does without the package.
    monkeypatch.setitem(sys.modules, "langdetect", None)
    with pytest.raises(ImportError, match="requirements-sft-eval.txt"):
        chat_local.ifeval_scores({"instruction_id_list": [], "kwargs": [], "prompt": "p"}, "answer")
    factory = types.SimpleNamespace(seed=None)
    monkeypatch.setitem(sys.modules, "langdetect", types.SimpleNamespace(DetectorFactory=factory))
    chat_local._require_langdetect()
    assert factory.seed == 0


def test_ifeval_scoring_is_repeatable_and_keeps_the_callers_random_state(monkeypatch):
    # lm-eval's letter check draws a random letter for prompt 19 ("#"); the stand-in draws one too.
    utils = types.ModuleType("lm_eval.tasks.ifeval.utils")
    utils.process_results = lambda doc, results: {"prompt_level_strict_acc": random.random() < 0.5, "draw": random.random()}
    for name in ("lm_eval", "lm_eval.tasks", "lm_eval.tasks.ifeval"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "lm_eval.tasks.ifeval.utils", utils)
    monkeypatch.setattr(sys.modules["lm_eval.tasks.ifeval"], "utils", utils, raising=False)
    monkeypatch.setitem(sys.modules, "langdetect", types.SimpleNamespace(DetectorFactory=types.SimpleNamespace(seed=None)))
    random.seed(123)
    expected_next = random.Random(123).random()
    first = chat_local.ifeval_scores({}, "answer")
    assert random.random() == expected_next
    random.seed(456)
    assert chat_local.ifeval_scores({}, "answer") == first


def test_preflight_names_a_missing_langdetect(tmp_path, monkeypatch):
    settings = make_settings(te.load_suite(SFT_SUITE), tmp_path)
    report = {"modules": {}, "errors": {"langdetect": "ModuleNotFoundError('langdetect')"}, "libstdcxx": []}
    monkeypatch.setattr(te_preflight, "probe_eval_environment", lambda *args, **kwargs: copy.deepcopy(report))
    with pytest.raises(RuntimeError, match="cannot import langdetect, which IFEval's language checks need"):
        te.run_preflight(settings)


# ---------------------------------------------------------------------------
# Gates and diagnostics from records
# ---------------------------------------------------------------------------


def make_record(task: str, doc_id: int, *, mode: str = "think", lang: str = "en", domain: str = "math", **overrides) -> dict:
    record = {
        "task": task, "doc_id": doc_id, "repeat": 0, "mode": mode, "lang_expected": lang, "domain": domain,
        "finish": "stop", "stop_reason": 151645, "n_gen": 900, "last_id": 151645, "n_open": 0,
        "n_close": 0 if mode == "no_think" else 1, "rep16_tail": 0.0, "rep16_w4k": 0.0, "answer_len": 40,
        "answer_nonblank": True, "closed": None if mode == "no_think" else True,
        "reasoning_len": None if mode == "no_think" else 800, "format_ok": True, "stop_inside": False,
        "has_answer": True, "markers": 0, "opener": False, "ans_latin": 100, "ans_cyr": 0, "lang": lang, "correct": 1.0,
    }
    if mode == "no_prefill":
        record.update(np_opened=True, np_ok=True, correct=None)
    record.update(overrides)
    return record


# Large enough pools that clean records pass every bound: about 1,350 think and 1,100 no_think responses.
GATE_SUITE = (
    "  math500_think: {}\n  gsm8k_think: {limit: 600}\n  mgsm_ru_think: {}\n"
    "  math500_nothink: {}\n  gsm8k_nothink: {limit: 600}\n  think_open: {limit: 50}\n  mmlu: {limit: 2}\n"
)
GATES = (
    "  gates: {g2_max: 0.005, g3_min: 0.98, g4_min: 0.98, g5_max: 0.05, g6_tags_max: 0.005, g6_markers_max: 0.6,\n"
    "          g6_len_p50_max: 1000, g7_math500_min: 0.258, g7_gsm8k_min: 0.364, g11_min: 0.98}\n"
)


def records_for(process: te.EvalProcess, tweak=None) -> list[dict]:
    rows = []
    for unit in process.units:
        benchmark = te.BENCHMARKS[unit.benchmark]
        for task in unit.tasks:
            lang = "ru" if "mgsm_ru" in task else "en"
            for doc_id in range(unit.docs):
                record = make_record(task, doc_id, mode=benchmark.chat_mode, lang=lang)
                rows.append(tweak(unit.benchmark, doc_id, record) if tweak else record)
    return rows


def gate_inputs(tmp_path, tweak=None, drop=None):
    suite = chat_suite(tmp_path, GATE_SUITE, CHAT_SECTION + GATES)
    plan = te.build_plan(suite, 2)
    pairs = []
    for processes in plan.processes:
        for process in processes:
            if process.is_chat:
                missing = drop is not None and process.units[0].benchmark == drop
                pairs.append((process, None if missing else records_for(process, tweak)))
    return suite, pairs


def test_gates_pass_on_clean_records(tmp_path):
    suite, pairs = gate_inputs(tmp_path)
    result = te.score_records(pairs, suite.chat)
    gates = result.gates
    assert gates["g2"]["pass"] == 1.0 and gates["g2"]["n"] == 1350.0 and gates["g2"]["rate"] == 0.0
    assert gates["g3"]["pass"] == 1.0 and gates["g3"]["n"] == 250.0
    assert gates["g4"]["pass"] == 1.0 and gates["g4"]["think_rate"] == 1.0 and gates["g4"]["nothink_rate"] == 1.0
    assert gates["g5"]["pass"] == 1.0
    assert gates["g6"]["pass"] == 1.0 and gates["g6"]["len_p50"] == 900.0
    assert gates["g7"]["pass"] == 1.0 and gates["g7"]["math500_rate"] == 1.0
    assert gates["g11"]["pass"] == 1.0 and gates["g11"]["n"] == 250.0 and gates["g11"]["n_ru_prompts"] == 250.0
    assert not result.notes
    assert gates["g2_cells"] == {"en_math": 0.0, "ru_math": 0.0}
    diagnostics = result.diagnostics
    assert diagnostics["math500_think"]["acc_strict"] == 1.0 and diagnostics["math500_think"]["loop"] == 0.0
    assert diagnostics["math500_nothink"]["markers"] == 0.0
    assert diagnostics["np_open"]["opens"] == 1.0
    assert not result.missing_units


def test_gates_catch_short_thinks_loops_and_leaks(tmp_path):
    def tweak(benchmark, doc_id, record):
        if benchmark == "math500_think" and doc_id < 6:
            record.update(reasoning_len=10)  # short think
        if benchmark == "math500_think" and 10 <= doc_id < 30:
            record.update(finish="length", n_gen=16384, rep16_tail=0.9, closed=False, has_answer=False, format_ok=False)
        if benchmark == "math500_nothink" and doc_id < 3:
            record.update(n_open=1, markers=2, format_ok=False)
        if benchmark == "mgsm_ru_think" and doc_id < 20:
            record.update(lang="en")
        if benchmark == "think_open" and doc_id < 10:
            record.update(np_ok=False)
        return record

    suite, pairs = gate_inputs(tmp_path, tweak)
    gates = te.score_records(pairs, suite.chat).gates
    assert gates["g2"]["count"] == 6.0 and gates["g2"]["pass"] == 0.0
    assert gates["g5"]["count"] == 20.0 and gates["g5"]["pass"] == 0.0 and gates["g5"]["loop_or_trunc_rate"] == 0.04
    assert gates["g6"]["pass"] == 0.0 and gates["g6"]["count"] == 3.0
    assert gates["g11"]["pass"] == 0.0 and gates["g11"]["rate"] == pytest.approx(230 / 250)
    assert gates["g3"]["pass"] == 0.0
    # Truncated or looping responses score 0 in acc_strict, and G7 uses it.
    assert gates["g7"]["math500_rate"] == 1.0


def test_missing_or_incomplete_records_withhold_their_gates(tmp_path):
    suite, pairs = gate_inputs(tmp_path, drop="math500_nothink")
    result = te.score_records(pairs, suite.chat)
    assert result.missing_units and "math500_nothink" not in result.diagnostics
    assert "g6" not in result.gates and "g11" not in result.gates and "g7" in result.gates  # gsm8k alone
    assert "g2" in result.gates  # the think pool is complete
    suite, pairs = gate_inputs(tmp_path / "again")
    process, rows = pairs[0]
    pairs[0] = (process, rows[:-1])
    result = te.score_records(pairs, suite.chat)
    assert any("records instead of" in note for note in result.notes)


def test_g11_reports_insufficient_instead_of_failing_on_few_answers(tmp_path):
    suite = chat_suite(tmp_path, "  mgsm_ru_think: {limit: 50}\n", CHAT_SECTION + GATES)
    pairs = [(process, records_for(process)) for processes in te.build_plan(suite, 1).processes for process in processes]
    gate = te.score_records(pairs, suite.chat).gates["g11"]
    assert gate["insufficient"] == 1.0 and "pass" not in gate and gate["rate"] == 1.0


def test_chat_scalars_have_their_own_titles():
    scores = te.Scores(
        benchmarks={"mmlu": 0.53, "math500_think": 0.4, "math500_nothink": 0.2},
        diagnostics={"math500_think": {"trunc": 0.1}},
        gates={"g2": {"rate": 0.001, "ub95": 0.004, "pass": 1.0}, "g2_cells": {"ru_chat": 0.0}},
    )
    logger = FakeClearMLLogger()
    te.report_eval_scalars(report_scalar, logger, None, scores=scores, step=1000, weights="raw",
                           timings={}, failures={"records_missing": 0})
    reported = {(title, series): value for title, series, value, _ in logger.scalars}
    assert reported[("eval_mmlu", "raw")] == 0.53
    assert reported[("eval_chat_math500", "think_raw")] == 0.4
    assert reported[("eval_chat_math500", "nothink_raw")] == 0.2
    assert reported[("eval_chat_trunc", "math500_think_raw")] == 0.1
    assert reported[("eval_gate_g2", "ub95_raw")] == 0.004 and reported[("eval_gate_g2", "pass_raw")] == 1.0
    assert reported[("eval_gate_g2_cells", "ru_chat_raw")] == 0.0
    assert reported[("eval_failures", "records_missing")] == 0.0
    stored = scores.to_dict()
    again = te.Scores(**{key: dict(stored.get(key) or {}) for key in
                         ("benchmarks", "parts", "ruler", "samples", "incomplete", "diagnostics", "gates")})
    assert again.gates == scores.gates and again.diagnostics == scores.diagnostics


# ---------------------------------------------------------------------------
# Rendering and the token contract with a real tokenizer (itr_render.py)
# ---------------------------------------------------------------------------

# The generation part of the stock Qwen3.5-4B template, which thinks by default.
QWEN35_STYLE_TEMPLATE = (
    "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}"
)
# The generation part of the stock Qwen3 template: think mode has no prefill.
QWEN3_STYLE_TEMPLATE = QWEN35_STYLE_TEMPLATE.replace("{{- '<think>\\n' }}", "")

needs_tokenizer = pytest.mark.skipif(not os.environ.get("ORBIT_TEST_TOKENIZER_DIR"), reason="needs ORBIT_TEST_TOKENIZER_DIR")
# The files a tokenizer folder may hold. A test copies the ones that exist and then writes its own
# chat template.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                   "added_tokens.json", "chat_template.jinja", "tokenizer.model")


def tokenizer_with_template(tmp_path: Path, template: str) -> Path:
    pytest.importorskip("transformers")
    source = Path(os.environ["ORBIT_TEST_TOKENIZER_DIR"])
    folder = tmp_path / "tokenizer"
    folder.mkdir()
    for name in TOKENIZER_FILES:
        if (source / name).is_file() and name != "chat_template.jinja":
            shutil.copy2(source / name, folder / name)
    config = json.loads((folder / "tokenizer_config.json").read_text())
    config.pop("chat_template", None)
    (folder / "tokenizer_config.json").write_text(json.dumps(config))
    (folder / "chat_template.jinja").write_text(template)
    return folder


@needs_tokenizer
def test_the_qwen35_style_template_renders_the_contract_ids(tmp_path):
    from transformers import AutoTokenizer

    folder = tokenizer_with_template(tmp_path, QWEN35_STYLE_TEMPLATE)
    report = itr_render.render_check(str(folder), str(folder))
    assert report["probes"] == 12 and report["template_sha256"] == itr_render.template_sha256(folder)
    tokenizer = AutoTokenizer.from_pretrained(folder)
    contract = itr_render.expected_ids(tokenizer)
    assert contract["tails"] == itr_render.CONTRACT_TAIL_IDS and contract["lead"] == 151643
    messages = [{"role": "user", "content": "Hi"}]
    for mode, tail in itr_render.CONTRACT_TAIL_IDS.items():
        with_lead = tokenizer.encode(itr_render.render(tokenizer, messages, mode, True), add_special_tokens=False)
        assert with_lead[:2] == [151643, 151644] and with_lead.count(151643) == 1 and with_lead[-len(tail):] == tail
        without = tokenizer.encode(itr_render.render(tokenizer, messages, mode, False), add_special_tokens=False)
        assert without[0] == 151644 and 151643 not in without
    # A template that writes the leading token itself still gives exactly one.
    lead_template = "{{- '<|endoftext|>' }}" + QWEN35_STYLE_TEMPLATE
    assert itr_render.render(tokenizer, messages, "think", True, chat_template=lead_template).count("<|endoftext|>") == 1


@needs_tokenizer
def test_a_template_without_the_think_prefill_stops_the_run(tmp_path):
    folder = tokenizer_with_template(tmp_path, QWEN3_STYLE_TEMPLATE)
    with pytest.raises(itr_render.ContractError, match="no prefill to remove|must end with"):
        itr_render.render_check(str(folder))


def test_contract_problems_name_the_wrong_part():
    contract = {"tails": {"think": [1, 2, 3], "no_think": [1, 2, 4, 5], "no_prefill": [1, 2]}, "lead": 9, "im_start": 1}
    assert itr_render.contract_problem([9, 1, 7, 1, 2, 3], "think", True, contract) is None
    assert "must start with [9, 1]" in itr_render.contract_problem([9, 9, 1, 7, 1, 2, 3], "think", True, contract)
    assert "no leading token" in itr_render.contract_problem([9, 1, 7, 1, 2, 3], "think", False, contract)
    assert "must end with [1, 2, 4, 5]" in itr_render.contract_problem([9, 1, 7, 1, 2, 3], "no_think", True, contract)


# ---------------------------------------------------------------------------
# The model subclass with stand-ins for lm-eval and vLLM (itr_chat.py)
# ---------------------------------------------------------------------------

SPECIALS = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645, "<think>": 151667, "</think>": 151668}


class ToyTokenizer:
    """Specials keep their Qwen ids; every other character is one token (1000 + its code point)."""

    def _pieces(self, text: str):
        position = 0
        while position < len(text):
            for special, token in SPECIALS.items():
                if text.startswith(special, position):
                    yield token
                    position += len(special)
                    break
            else:
                yield 198 if text[position] == "\n" else 1000 + ord(text[position])
                position += 1

    def encode(self, text, add_special_tokens=True):
        return list(self._pieces(text))

    def decode(self, ids):
        reverse = {token: special for special, token in SPECIALS.items()}
        return "".join(reverse.get(t, "\n" if t == 198 else chr(t - 1000) if t >= 1000 else "?") for t in ids)

    def convert_tokens_to_ids(self, token):
        return SPECIALS[token]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None,
                            chat_template=None, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n")
        return text


STUB_VLLM_MODULE = r'''
import copy
from dataclasses import dataclass


class SamplingParams:
    def __init__(self, max_tokens=16, **kwargs):
        self.max_tokens = max_tokens
        self.seed = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def clone(self):
        return copy.deepcopy(self)


@dataclass
class RepetitionDetectionParams:
    max_pattern_size: int = 0
    min_pattern_size: int = 0
    min_count: int = 0
'''

STUB_BACKEND = r'''
from vllm import SamplingParams


class VLLM:
    """The parts of lm-eval 0.4.13's vLLM backend that the subclass relies on.

    generate_until encodes the contexts, sorts the requests by length and context (stable, as lm-eval's
    Collator does), asks _model_generate for the outputs, and returns the texts in the original order.
    """

    def __init__(self, pretrained=None, stub_tokenizer=None, stub_engine=None, **kwargs):
        self.tokenizer = stub_tokenizer
        self.engine = stub_engine
        self.hf_chat_template = None
        self.chat_template_args = {}
        self.max_gen_toks = 256
        self.kwargs = kwargs

    def tok_encode(self, strings):
        if isinstance(strings, str):
            return self.tokenizer.encode(strings, add_special_tokens=False)
        return [self.tokenizer.encode(s, add_special_tokens=False) for s in strings]

    @staticmethod
    def modify_gen_kwargs(gen_kwargs, eos=None, default_max_gen_toks=256):
        kwargs = dict(gen_kwargs)
        until = kwargs.pop("until", []) or []
        max_gen_toks = int(kwargs.pop("max_gen_toks", default_max_gen_toks))
        kwargs.pop("do_sample", None)
        return kwargs, until + [eos], max_gen_toks

    def generate_until(self, requests, disable_tqdm=False):
        contexts = [r.args[0] for r in requests]
        encoded = self.tok_encode(contexts)
        order = sorted(range(len(requests)), key=lambda i: (-len(encoded[i]), contexts[i]))
        params = []
        for i in order:
            kwargs, until, max_gen_toks = self.modify_gen_kwargs(requests[i].args[1], eos="<|im_end|>")
            params.append(SamplingParams(max_tokens=max_gen_toks, stop=until, **kwargs))
        outputs = self._model_generate([encoded[i] for i in order], generate=True, sampling_params=params)
        texts = [None] * len(requests)
        for i, output in zip(order, outputs):
            texts[i] = output.outputs[0].text
        return texts

    def _model_generate(self, requests, generate=False, sampling_params=None):
        return self.engine(requests, sampling_params)

    def loglikelihood(self, requests, disable_tqdm=False):
        raise NotImplementedError
'''


class Completion:
    def __init__(self, token_ids, text, finish_reason="stop", stop_reason=151645, logprobs=None):
        self.token_ids, self.text, self.finish_reason, self.stop_reason, self.logprobs = (
            token_ids, text, finish_reason, stop_reason, logprobs)


class Output:
    def __init__(self, completion):
        self.outputs = [completion]


class Request:
    def __init__(self, task, doc_id, context, gen_kwargs=None):
        self.task_name, self.doc_id = task, doc_id
        self.args = (context, gen_kwargs if gen_kwargs is not None else {"until": [], "max_gen_toks": 64})


@pytest.fixture
def itr_chat(monkeypatch, tmp_path):
    """itr_chat.py loaded against stand-ins of lm-eval and vLLM."""
    def module(name, source=""):
        stub = types.ModuleType(name)
        # Registered before its code runs, because dataclasses look their module up while they are built.
        monkeypatch.setitem(sys.modules, name, stub)
        exec(source, stub.__dict__)
        return stub

    module("vllm", STUB_VLLM_MODULE)
    sys.modules["vllm"].sampling_params = module("vllm.sampling_params", "from vllm import RepetitionDetectionParams")
    for name in ("lm_eval", "lm_eval.api", "lm_eval.models"):
        module(name)
    registered = {}
    module("lm_eval.api.registry").register_model = lambda name: (lambda cls: registered.setdefault(name, cls))
    module("lm_eval.utils").hash_string = lambda text: "h" + str(len(text)) + text[-8:]
    module("lm_eval.models.vllm_causallms", STUB_BACKEND)
    monkeypatch.syspath_prepend(str(IN_TRAINING))
    monkeypatch.setenv("ORBIT_ITR_TASKS", str(TASKS_DIR))
    monkeypatch.setenv("ORBIT_ITR_OUT", str(tmp_path / "out"))
    monkeypatch.delenv(chat_local.MODE_ENV, raising=False)
    loaded = load_module(IN_TRAINING / "itr_chat.py", "test_itr_chat")
    loaded.registered = registered
    return loaded


def toy_prompt(text: str, mode: str = "think") -> str:
    return itr_render.render(ToyTokenizer(), [{"role": "user", "content": text}], mode, True)


def test_the_subclass_keys_every_response_and_writes_the_sidecar(itr_chat, tmp_path):
    seen = {}

    def engine(prompts, params):
        seen["params"] = params
        outputs = []
        for prompt in prompts:
            # The response echoes the last prompt character, so the order of the outputs can be checked.
            marker = 1000 + ord("a") + (prompt[-6] - 1000) % 3
            ids = [marker] * 40 + [151668, 1000 + ord("x"), 151645]
            outputs.append(Output(Completion(ids, "y" * 40 + "</think>x")))
        return outputs

    model = itr_chat.VLLMITRChat(stub_tokenizer=ToyTokenizer(), stub_engine=engine, itr_mode="think",
                                 itr_lead_eos="True")
    assert itr_chat.registered["vllm_itr_chat"] is itr_chat.VLLMITRChat
    assert os.environ[chat_local.MODE_ENV] == "think"
    long_prompt, short_prompt = toy_prompt("a much longer question"), toy_prompt("q?")
    requests = [Request("t", 0, short_prompt), Request("t", 1, long_prompt), Request("t", 0, short_prompt)]
    texts = model.generate_until(requests)
    assert texts == ["y" * 40 + "</think>x"] * 3
    lines = [json.loads(line) for line in (tmp_path / "out" / itr_chat.SIDECAR_NAME).read_text().splitlines()]
    keys = sorted((line["doc_index"], line["repeat"]) for line in lines)
    assert keys == [(0, 0), (0, 1), (1, 0)]
    assert all(line["closed"] and line["reasoning_len"] == 40 and line["format_ok"] for line in lines)
    # Every request reaches vLLM with the sampling parameters that lm-eval built, and no seed of its own.
    assert len(seen["params"]) == 3 and all(params.seed is None for params in seen["params"])


def test_the_subclass_refuses_prompts_that_break_the_contract(itr_chat):
    def engine(prompts, params):
        raise AssertionError("never reached")

    model = itr_chat.VLLMITRChat(stub_tokenizer=ToyTokenizer(), stub_engine=engine, itr_mode="no_think")
    # The subclass raises its own copy of itr_render's ContractError, a ValueError.
    with pytest.raises(ValueError, match="no_think prompt must end with"):
        model.generate_until([Request("t", 0, toy_prompt("q?", "think"))])


def test_the_no_prefill_prompt_drops_the_prefill(itr_chat):
    model = itr_chat.VLLMITRChat(stub_tokenizer=ToyTokenizer(), stub_engine=None, itr_mode="no_prefill")
    text = model.apply_chat_template([{"role": "user", "content": "hi"}])
    assert text.startswith("<|endoftext|><|im_start|>user") and text.endswith("<|im_start|>assistant\n")


def test_repetition_detection_becomes_vllms_parameter_object(itr_chat):
    kwargs, until, budget = itr_chat.VLLMITRChat.modify_gen_kwargs(
        {"until": [], "max_gen_toks": 100, "temperature": 0.6, "do_sample": True,
         "repetition_detection": {"min_pattern_size": 64, "max_pattern_size": 128, "min_count": 16}},
        eos="<|im_end|>",
    )
    assert budget == 100 and kwargs["temperature"] == 0.6
    assert kwargs["repetition_detection"].max_pattern_size == 128


def write_samples(out: Path, task: str, docs: list[tuple[int, dict, list[str], str]]) -> None:
    folder = out / "model"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"samples_{task}_2026-01-01T12-00-00.123456.jsonl").open("w") as handle:
        for doc_id, doc, texts, prompt_hash in docs:
            handle.write(json.dumps({"doc_id": doc_id, "doc": doc, "resps": [texts], "prompt_hash": prompt_hash,
                                     "filter": "none"}) + "\n")


def test_records_join_samples_and_sidecar_through_the_shard_indices(itr_chat, tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    facts = chat_local.token_facts([1] * 40 + [151668, 5, 151645], "think", "stop", blank)
    sidecar = [
        {"task": "itr_chat_mgsm_ru", "doc_index": 0, "repeat": 0, "prompt_hash": "p10", "gen_ids": [1], **facts},
        {"task": "itr_chat_mgsm_ru", "doc_index": 1, "repeat": 0, "prompt_hash": "p11", "gen_ids": [1], **facts},
    ]
    (out / itr_chat.SIDECAR_NAME).write_text("".join(json.dumps(line) + "\n" for line in sidecar))
    doc = {"itr_lang": "ru", "itr_domain": "math", "itr_gold": "18"}
    write_samples(out, "itr_chat_mgsm_ru", [
        (10, doc, ["рассуждение</think>Ответ: восемнадцать яиц, то есть \\boxed{18}."], "p10"),
        (11, doc, ["рассуждение</think>The answer is \\boxed{17} eggs for the farmers market."], "p11"),
    ])
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps({"itr_chat_mgsm_ru": [10, 11]}))
    monkeypatch.setattr(chat_local, "math_correct", lambda answer, gold: int(answer is not None and gold in answer))
    monkeypatch.setattr(itr_chat, "chat_local", lambda: chat_local)
    assert itr_chat.write_records(out, samples, "think") == 2
    records = [json.loads(line) for line in (out / itr_chat.RECORDS_NAME).read_text().splitlines()]
    assert [(r["doc_id"], r["lang"], r["correct"]) for r in records] == [(10, "ru", 1.0), (11, "en", 0.0)]
    assert all("gen_ids" not in r and r["lang_expected"] == "ru" and r["closed"] for r in records)
    # A sidecar entry of another prompt is refused, and finish leaves the error instead of records.
    sidecar[1]["prompt_hash"] = "other"
    (out / itr_chat.SIDECAR_NAME).write_text("".join(json.dumps(line) + "\n" for line in sidecar))
    (out / itr_chat.RECORDS_NAME).unlink()
    itr_chat.finish(["--output_path", str(out), "--samples", str(samples)])
    assert not (out / itr_chat.RECORDS_NAME).exists()
    assert "another prompt" in json.loads((out / itr_chat.RECORDS_ERROR_NAME).read_text())["error"]
    # With a consistent sidecar, finish writes the records and compresses the big files.
    sidecar[1]["prompt_hash"] = "p11"
    (out / itr_chat.SIDECAR_NAME).write_text("".join(json.dumps(line) + "\n" for line in sidecar))
    itr_chat.finish(["--output_path", str(out), "--samples", str(samples)])
    assert (out / itr_chat.RECORDS_NAME).is_file()
    assert (out / (itr_chat.SIDECAR_NAME + ".gz")).is_file() and not (out / itr_chat.SIDECAR_NAME).exists()
    assert len(list(out.glob("model/samples_*.jsonl.gz"))) == 1
    with gzip.open(out / (itr_chat.SIDECAR_NAME + ".gz"), "rt") as handle:
        assert len(handle.read().splitlines()) == 2


# ---------------------------------------------------------------------------
# The wrapper on the chat path
# ---------------------------------------------------------------------------

STUB_CHAT_MAIN = r'''
import json, os, sys

def cli_evaluate():
    loaded = sorted(name for name in sys.modules if name in ("itr_chat", "itr_render"))
    with open(os.environ["STUB_OUT"], "w") as handle:
        json.dump({"loaded": loaded, "out_env": os.environ.get("ORBIT_ITR_OUT")}, handle)
'''


def test_the_wrapper_loads_the_subclass_only_for_the_chat_model_and_keeps_the_exit_code(tmp_path):
    stub = tmp_path / "stub"
    for package in ("lm_eval", "lm_eval/api", "lm_eval/models", "vllm"):
        (stub / package).mkdir(parents=True)
        (stub / package / "__init__.py").write_text("")
    (stub / "lm_eval" / "__main__.py").write_text(STUB_CHAT_MAIN)
    (stub / "lm_eval" / "api" / "registry.py").write_text("def register_model(name):\n    return lambda cls: cls\n")
    (stub / "lm_eval" / "utils.py").write_text("def hash_string(text):\n    return text\n")
    (stub / "lm_eval" / "models" / "vllm_causallms.py").write_text(STUB_BACKEND)
    (stub / "vllm" / "__init__.py").write_text(STUB_VLLM_MODULE)
    (stub / "vllm" / "sampling_params.py").write_text("from vllm import RepetitionDetectionParams\n")
    out = tmp_path / "result.json"
    env = {**os.environ, "PYTHONPATH": str(stub), "STUB_OUT": str(out), "PYTHONDONTWRITEBYTECODE": "1",
           "ORBIT_EVAL_HIDDEN_MODULES": "flash_attn"}
    env.pop(te.EVAL_PARENT_ENV, None)
    entry = IN_TRAINING / "run_lm_eval.py"
    lm_out = tmp_path / "lm_eval"
    arguments = ["--model", te.CHAT_MODEL, "--include_path", str(TASKS_DIR), "--output_path", str(lm_out)]
    result = subprocess.run([sys.executable, str(entry), *arguments], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    seen = json.loads(out.read_text())
    assert seen["loaded"] == ["itr_chat", "itr_render"] and seen["out_env"] == str(lm_out.resolve())
    # lm-eval wrote no samples here, so the records fail, and the exit code stays 0.
    assert "no samples found" in json.loads((lm_out / "itr_records_error.json").read_text())["error"]
    result = subprocess.run([sys.executable, str(entry), "--model", "vllm"], env=env, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0 and json.loads(out.read_text())["loaded"] == []


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_shipping_specials_rewrite_only_the_special_tokens(tmp_path):
    path = tmp_path / "tokenizer_config.json"
    path.write_text(json.dumps({"eos_token": "<|endoftext|>", "bos_token": "<|endoftext|>", "model_max_length": 7,
                                "added_tokens_decoder": {"151645": {"content": "<|im_end|>"}}}))
    te.set_shipping_specials(path)
    config = json.loads(path.read_text())
    assert config["eos_token"] == "<|im_end|>" and config["pad_token"] == "<|endoftext|>"
    assert config["bos_token"] is None and config["add_bos_token"] is False and config["model_max_length"] == 7


def test_chat_export_ships_im_end_everywhere_and_keeps_the_token_ids(tmp_path):
    folder = write_tokenizer(tmp_path / "export")
    (folder / "special_tokens_map.json").write_text(json.dumps({
        "eos_token": {"content": "<|endoftext|>", "lstrip": False, "normalized": False},
        "pad_token": {"content": "<|endoftext|>", "lstrip": False, "normalized": False},
    }))
    (folder / "config.json").write_text(json.dumps({"eos_token_id": 151643, "pad_token_id": 151643, "vocab_size": 151936}))
    tokenizer_json = (folder / "tokenizer.json").read_bytes()

    te.ship_chat_export(folder)

    assert json.loads((folder / "tokenizer_config.json").read_text())["eos_token"] == "<|im_end|>"
    special = json.loads((folder / "special_tokens_map.json").read_text())
    assert special["eos_token"] == {"content": "<|im_end|>", "lstrip": False, "normalized": False}
    assert special["pad_token"]["content"] == "<|endoftext|>"
    config = json.loads((folder / "config.json").read_text())
    assert config["eos_token_id"] == 151645 and config["pad_token_id"] == 151643 and config["vocab_size"] == 151936
    # Stop ids only: sampling fields here would become vLLM's default sampling parameters.
    assert json.loads((folder / "generation_config.json").read_text()) == {"eos_token_id": [151645, 151643], "pad_token_id": 151643}
    assert (folder / "tokenizer.json").read_bytes() == tokenizer_json


def test_chat_export_refuses_a_folder_without_a_chat_template(tmp_path):
    folder = write_tokenizer(tmp_path / "base", template=None)
    (folder / "config.json").write_text(json.dumps({"eos_token_id": 151643}))
    with pytest.raises(ValueError, match="not a chat export"):
        te.ship_chat_export(folder)
    assert json.loads((folder / "config.json").read_text())["eos_token_id"] == 151643


# ---------------------------------------------------------------------------
# Whole evaluations with a stand-in for lm-eval
# ---------------------------------------------------------------------------

FAKE_CHAT_LM_EVAL = r'''
import argparse, json, os, sys
parser = argparse.ArgumentParser()
parser.add_argument("--tasks"); parser.add_argument("--output_path"); parser.add_argument("--samples")
parser.add_argument("--limit"); parser.add_argument("--model"); parser.add_argument("--model_args")
args, _ = parser.parse_known_args()
samples = json.load(open(args.samples)) if args.samples else {}
results, counts, records = {}, {}, []
for task in args.tasks.split(","):
    n = len(samples[task]) if task in samples else int(args.limit or 10)
    results[task] = {"acc,none": 0.5, "prompt_level_strict_acc,none": 0.25, "acc_norm,none": 0.75}
    counts[task] = {"original": n, "effective": n}
    mode = [p.split("=")[1] for p in (args.model_args or "").split(",") if p.startswith("itr_mode=")]
    for doc_id in range(n):
        records.append({"task": task, "doc_id": doc_id, "repeat": 0, "mode": mode[0] if mode else None,
                        "lang_expected": "en", "domain": "math", "finish": "stop", "n_gen": 500, "n_open": 0,
                        "n_close": 1, "rep16_tail": 0.0, "rep16_w4k": 0.0, "closed": True, "reasoning_len": 400,
                        "format_ok": True, "stop_inside": False, "has_answer": True, "answer_len": 9, "markers": 0,
                        "opener": False, "lang": "en", "correct": 1.0, "np_ok": True})
os.makedirs(os.path.join(args.output_path, "model"), exist_ok=True)
json.dump({"results": results, "n-samples": counts}, open(os.path.join(args.output_path, "model", "results_x.json"), "w"))
if args.model == "vllm_itr_chat" and not os.environ.get("FAKE_NO_RECORDS"):
    with open(os.path.join(args.output_path, "itr_records.jsonl"), "w") as handle:
        handle.write("".join(json.dumps(r) + "\n" for r in records))
'''

SMALL_CHAT_SUITE = CHAT_SECTION + "benchmarks:\n  math500_think: {limit: 3}\n  math500_nothink: {limit: 3}\n  mmlu: {limit: 2}\n"


def chat_evaluator(tmp_path, monkeypatch):
    entry = tmp_path / "fake_lm_eval.py"
    entry.write_text(FAKE_CHAT_LM_EVAL)
    suite = te.load_suite(write_suite(tmp_path, "process_startup_seconds: 1\n" + SMALL_CHAT_SUITE, entry=entry))
    (tmp_path / "cache").mkdir(exist_ok=True)
    exports: list = []
    seen_metadata: list = []
    recorder = fake_export_recorder(exports)

    def export(state, *, out_dir, metadata=None, **kwargs):
        seen_metadata.append(dict(metadata or {}))
        return recorder(state, out_dir=out_dir, metadata=metadata, **kwargs)

    monkeypatch.setattr(te_evaluator, "write_eval_export", export)
    monkeypatch.setattr(te_evaluator, "_live_model_config", lambda model: object())
    settings = make_settings(suite, tmp_path, chat_template_sha256="abc")
    evaluator = te.InTrainingEval(settings, rank=0, local_rank=0, world_size=1, distributed=False, device=None,
                                  run_dir=tmp_path / "run", poll_seconds=0.05)
    evaluator.WRITE_RETRY_SECONDS = 0.0
    logger = FakeClearMLLogger()
    evaluator.attach(model=torch.nn.Linear(4, 4), clearml_logger=logger, tb_writer=None, stop_requested=lambda: False)
    return evaluator, logger, seen_metadata


def test_a_chat_evaluation_reports_accuracy_diagnostics_and_gates(tmp_path, monkeypatch):
    evaluator, logger, metadata = chat_evaluator(tmp_path, monkeypatch)
    evaluator.maybe_run(10)
    summary = json.loads((evaluator.step_dir(10) / "summary.json").read_text())
    assert summary["status"] == "ok" and summary["failures"]["records_missing"] == 0
    assert summary["chat_template_sha256"] == "abc" and summary["chat_notes"] == []
    assert summary["scores"]["gates"]["g4"]["think_rate"] == 1.0
    assert set(summary["scores"]["diagnostics"]) == {"math500_think", "math500_nothink"}
    reported = {(title, series) for title, series, _, _ in logger.scalars}
    assert {("eval_chat_math500", "think_raw"), ("eval_chat_math500", "nothink_raw"), ("eval_mmlu", "raw"),
            ("eval_gate_g2", "rate_raw"), ("eval_chat_trunc", "math500_think_raw")} <= reported
    assert metadata == [{"step": 10, "export_key": evaluator.export_key, "chat_template_sha256": "abc"}]
    commands = sorted((evaluator.step_dir(10) / "procs").glob("*/command.json"))
    models = sorted(json.loads(path.read_text())[3] for path in commands)
    assert models == ["vllm", "vllm_itr_chat", "vllm_itr_chat"]


def test_a_chat_process_without_records_keeps_its_accuracy_and_withholds_its_gates(tmp_path, monkeypatch):
    evaluator, logger, _ = chat_evaluator(tmp_path, monkeypatch)
    evaluator.QUIET_SECONDS = 0.2
    monkeypatch.setenv("FAKE_NO_RECORDS", "1")
    evaluator.maybe_run(10)
    summary = json.loads((evaluator.step_dir(10) / "summary.json").read_text())
    assert summary["failures"]["records_missing"] == 2
    assert "math500_think" in summary["scores"]["benchmarks"]
    assert not summary["scores"].get("gates")
    # A process that left no records is not reused by a later attempt.
    process = evaluator.plan.processes[0][0]
    assert not te.process_outputs_ready(evaluator.step_dir(10) / "procs" / process.name, True)


# ---------------------------------------------------------------------------
# The repository's chat task files
# ---------------------------------------------------------------------------


def test_chat_task_files_use_no_stop_strings_and_no_shots():
    yaml = pytest.importorskip("yaml")

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))
    for path in sorted(TASKS_DIR.glob("itr_chat_*.yaml")):
        task = yaml.load(path.read_text(), Loader=Loader)
        assert task["num_fewshot"] == 0, path.name
        assert "in_training_eval" not in task["tag"], path.name
        if task["output_type"] == "generate_until":
            assert task["generation_kwargs"]["until"] == [], path.name
            assert "do_sample" not in task["generation_kwargs"] and "temperature" not in task["generation_kwargs"]
        for key in ("process_docs", "doc_to_text", "process_results", "custom_dataset"):
            if key in task:
                assert hasattr(chat_local, task[key].split(".")[1]), (path.name, key)
    registered = {task for benchmark in te.BENCHMARKS.values() if benchmark.chat_mode for task, _ in benchmark.subtasks}
    assert registered == {path.stem for path in TASKS_DIR.glob("itr_chat_*.yaml")}
    assert set(chat_local.TASK_KINDS) == registered


def test_the_smoke_suite_runs_every_benchmark_of_sft_with_its_sampling():
    """sft_smoke.yaml is the quick GPU check of sft.yaml, so it must exercise exactly the same paths.

    It carries sft.yaml's chat section, so a change of a preset must reach it, and it runs every
    benchmark of sft.yaml on a few documents.
    """
    sft = te.load_suite(SUITES / "sft.yaml")
    smoke = te.load_suite(SUITES / "sft_smoke.yaml")
    assert copy.deepcopy(smoke.chat) == copy.deepcopy(sft.chat)
    assert list(smoke.benchmarks) == list(sft.benchmarks)
    assert all(settings.limit is not None for settings in smoke.benchmarks.values())
