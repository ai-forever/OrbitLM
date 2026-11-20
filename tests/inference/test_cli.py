"""Argument checks of both generators, the vLLM engine arguments, and a whole run with a scripted engine."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import textwrap

import pytest

from inference import REPO_ROOT, cli
from inference import generate_hf, generate_vllm
from inference.budget import Completion
from inference.prompts import PromptError
from orbit_verl.data.response_info import STOP_THINKING_TEXT, STOP_THINKING_TEXT_RU

CONFIG = {"model_type": "ouro", "total_ut_steps": 4, "max_position_embeddings": 32768}


@pytest.fixture
def model_dir(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    return path


def vllm_args(model_dir, *extra):
    """Parse and check vLLM arguments the way generate_vllm.main does, up to the engine start."""
    args = generate_vllm.build_parser().parse_args([str(model_dir), "--prompt", "hi", *extra])
    cli.check_shared_args(args)
    generate_vllm.check_engine_args(args)
    cli.resolve_model_args(args, cli.read_model_config(args.model))
    generate_vllm.check_speculative_loops(args)
    return args


def test_a_prompt_source_is_required_and_unique(model_dir):
    parser = generate_vllm.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([str(model_dir)])
    with pytest.raises(SystemExit):
        parser.parse_args([str(model_dir), "--prompt", "a", "--prompts", "p.jsonl"])


@pytest.mark.parametrize("extra, message", [
    (["--budget-forcing", "--mode", "no_think"], "needs --mode think"),
    (["--think-budget", "100"], "only with --budget-forcing"),
    (["--answer-budget", "100"], "only with --budget-forcing"),
    (["--budget-forcing", "--max-new-tokens", "100"], "not --max-new-tokens"),
    (["--budget-forcing", "--think-budget", "0"], "must be positive"),
    (["--temperature", "-1"], "--temperature"),
    (["--top-p", "0"], "--top-p"),
    (["--top-k", "-1"], "--top-k"),
    (["--loops", "0"], "starts at 1"),
    (["--loops", "5"], "from 1 to the model's total_ut_steps, 4"),
    (["--budget-forcing", "--think-budget", "30000", "--answer-budget", "2768"], "no room for the prompt"),
    (["--draft-dir", "/tmp/x"], "only with --speculative"),
    (["--num-speculative-tokens", "2"], "only with --speculative"),
    (["--speculative", "--num-speculative-tokens", "0"], "must be positive"),
    (["--speculative", "--loops", "1"], "at least 2 loops"),
    (["--tensor-parallel-size", "0"], "--tensor-parallel-size"),
    (["--gpu-memory-utilization", "1.5"], "--gpu-memory-utilization"),
])
def test_bad_arguments_are_refused(model_dir, extra, message):
    with pytest.raises(cli.UsageError, match=message):
        vllm_args(model_dir, *extra)


def test_main_turns_a_usage_error_into_a_parser_error(tmp_path):
    with pytest.raises(SystemExit) as raised:
        generate_vllm.main([str(tmp_path), "--prompt", "hi"])
    assert raised.value.code == 2


def test_defaults_come_from_the_model_config(model_dir):
    args = vllm_args(model_dir)
    assert (args.loops, args.model_loops, args.max_model_len, args.max_new_tokens) == (4, 4, 32768, cli.DEFAULT_MAX_NEW_TOKENS)
    assert (args.think_budget, args.answer_budget) == (None, None)
    args = vllm_args(model_dir, "--budget-forcing")
    assert (args.think_budget, args.answer_budget, args.max_new_tokens) == (14336, 2048, None)
    assert vllm_args(model_dir, "--speculative").num_speculative_tokens == 4


def test_one_loop_gets_a_note(model_dir):
    args = generate_vllm.build_parser().parse_args([str(model_dir), "--prompt", "hi", "--loops", "1"])
    notes = cli.resolve_model_args(args, CONFIG)
    assert any("two loops" in note for note in notes)


def test_engine_kwargs_default_to_fp8_flash_attention_and_all_loops(model_dir):
    kwargs = generate_vllm.engine_kwargs(vllm_args(model_dir))
    assert kwargs["kv_cache_dtype"] == "fp8" and kwargs["attention_backend"] == "FLASH_ATTN"
    assert kwargs["dtype"] == "bfloat16" and kwargs["max_model_len"] == 32768
    assert "hf_overrides" not in kwargs and "speculative_config" not in kwargs


def test_engine_kwargs_with_fewer_loops_bf16_and_vllm_attention(model_dir):
    kwargs = generate_vllm.engine_kwargs(vllm_args(model_dir, "--loops", "3", "--kv-cache-dtype", "auto",
                                                   "--attention-backend", "auto"))
    assert kwargs["hf_overrides"] == {"total_ut_steps": 3}
    assert kwargs["kv_cache_dtype"] == "auto" and "attention_backend" not in kwargs


def test_engine_kwargs_with_the_one_loop_draft(model_dir, tmp_path):
    kwargs = generate_vllm.engine_kwargs(vllm_args(model_dir, "--speculative"), tmp_path / "draft")
    assert kwargs["speculative_config"] == {
        "method": "draft_model", "model": str(tmp_path / "draft"), "num_speculative_tokens": 4,
        "draft_sample_method": "greedy", "attention_backend": "FLASH_ATTN",
    }
    assert kwargs["async_scheduling"] is True


def test_sampling_settings_of_both_engines(model_dir):
    greedy = vllm_args(model_dir, "--temperature", "0")
    assert generate_vllm.sampling_kwargs(greedy, (1, 2)) == {
        "temperature": 0.0, "top_p": 1.0, "stop_token_ids": [1, 2], "detokenize": False}
    assert generate_hf.generate_kwargs(cli.sampling_settings(greedy)) == {
        "do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    sampled = vllm_args(model_dir)
    assert generate_vllm.sampling_kwargs(sampled, (1,))["top_k"] == 20
    assert generate_hf.generate_kwargs(cli.sampling_settings(sampled)) == {
        "do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    no_top_k = vllm_args(model_dir, "--top-k", "0")
    assert "top_k" not in generate_vllm.sampling_kwargs(no_top_k, (1,))
    assert generate_hf.generate_kwargs(cli.sampling_settings(no_top_k))["top_k"] == 0


@pytest.mark.parametrize("version, ok", [
    ("4.55.0", False), ("4.56.0", True), ("4.57.6", True), ("5.1.0", True), ("5.2.0", True),
    ("5.10.4", True), ("5.23.0", True), ("5.24.0", False), ("5.24.0.dev0", False), ("6.0.0", False),
])
def test_transformers_range_of_the_hf_path(version, ok):
    assert (generate_hf.transformers_problem(version) is None) == ok


def test_a_whole_run_with_budget_forcing(model_dir, tmp_path, tokenizer, scripted_engine, special):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
        {"id": "en", "prompt": "What is 6 times 7?"},
        {"id": "ru", "prompt": "Сколько будет 6 умножить на 7?"},
        {"id": "short", "prompt": "Say hi."},
    ]) + "\n", encoding="utf-8")
    output = tmp_path / "out" / "answers.jsonl"
    args = generate_vllm.build_parser().parse_args([
        str(model_dir), "--prompts", str(prompts), "--output", str(output),
        "--budget-forcing", "--think-budget", "4", "--answer-budget", "3",
    ])
    cli.check_shared_args(args)
    cli.resolve_model_args(args, CONFIG)
    end = special["<|im_end|>"]

    def script(prompt, limit, call):
        if call == 0 and tokenizer.decode(prompt).endswith("Say hi.<|im_end|>\n<|im_start|>assistant\n<think>\n"):
            return Completion((*tokenizer.encode("ok</think>hi"), end), "stop")
        if call == 0:
            return Completion(tuple(tokenizer.encode("a" * limit)), "length")
        return Completion((*tokenizer.encode("42"), end), "stop")

    engine = scripted_engine(script)
    started = []

    def start_engine(run_args, run_tokenizer, end_ids):
        started.append(tuple(end_ids))
        return engine

    assert cli.run(args, start_engine, tokenizer_loader=lambda path: tokenizer) == 0
    assert started == [(special["<|im_end|>"], special["<|endoftext|>"])]
    records = {r["id"]: r for r in map(json.loads, output.read_text(encoding="utf-8").splitlines())}
    assert [records[k]["lang"] for k in ("en", "ru", "short")] == ["en", "ru", "en"]
    for key, sentence in (("en", STOP_THINKING_TEXT), ("ru", STOP_THINKING_TEXT_RU)):
        record = records[key]
        assert record["forced_close"] and record["think"] == "closed"
        assert (record["thinking"], record["answer"]) == ("aaaa", "42")
        assert record["text"] == "aaaa" + sentence + "42<|im_end|>"
        assert record["finish_reason"] == "stop" and record["completion_tokens"] == 7
    short = records["short"]
    assert not short["forced_close"] and (short["thinking"], short["answer"]) == ("ok", "hi")
    assert len(engine.calls) == 2 and len(engine.calls[1][0]) == 2


def parsed_args(model_dir, *extra):
    args = generate_vllm.build_parser().parse_args([str(model_dir), *extra])
    cli.check_shared_args(args)
    cli.resolve_model_args(args, CONFIG)
    return args


def test_a_prompt_that_fills_the_context_fails_before_the_engine_starts(model_dir, tokenizer):
    args = parsed_args(model_dir, "--prompt", "short", "--prompt", "x" * 100, "--max-model-len", "64")
    started = []
    with pytest.raises(PromptError, match="prompt 1 has .* tokens, which fills the context of 64"):
        cli.run(args, lambda *a: started.append(a), tokenizer_loader=lambda path: tokenizer,
                records_stream=io.StringIO())
    assert started == []


def test_without_output_the_records_go_to_the_records_stream(model_dir, tokenizer, scripted_engine, special):
    args = parsed_args(model_dir, "--prompt", "Say hi.", "--mode", "no_think", "--max-new-tokens", "8")
    engine = scripted_engine(lambda prompt, limit, call: Completion((*tokenizer.encode("hi"), special["<|im_end|>"]), "stop"))
    stream = io.StringIO()
    assert cli.run(args, lambda *a: engine, tokenizer_loader=lambda path: tokenizer, records_stream=stream) == 0
    [record] = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert (record["answer"], record["think"], record["finish_reason"]) == ("hi", "none", "stop")


def test_reserve_stdout_leaves_standard_output_to_the_records():
    # The check needs a process of its own, because it repoints file descriptor 1 for good. The child
    # process started after the call stands for vLLM's engine process, which inherits the descriptor.
    code = textwrap.dedent("""
        import subprocess
        import sys
        from inference import cli

        stream = cli.reserve_stdout()
        print("noise from python")
        subprocess.run([sys.executable, "-c", "print('noise from a child process')"], check=True)
        cli.write_records([{"id": "0"}], None, stream)
    """)
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"id": "0"}\n'
    assert "noise from python" in result.stderr and "noise from a child process" in result.stderr
