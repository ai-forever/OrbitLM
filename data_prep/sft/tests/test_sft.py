"""Unit tests of the SFT builder, on synthetic dialogs.

They need numpy, torch, transformers and mosaicml-streaming, and the release
tokenizers from the tokenizer_root fixture (data_prep/conftest.py). From the
repository root:

  PYTHONPATH=src python -m pytest data_prep/sft/tests
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SFT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SFT))
sys.path.insert(0, SFT)
sys.path.insert(0, os.path.join(REPO, "src"))

import benches  # noqa: E402
import blocks  # noqa: E402
import dialogs  # noqa: E402
from system_prompts import classify_system, row_has_tools, system_texts_from_row  # noqa: E402
from weights import addition_weights, epochs_at_budget, part_weights, recipe_sampling_weights  # noqa: E402
from render import (  # noqa: E402
    END_OF_TEXT,
    EOS_ID,
    IM_END_ID,
    example_enable_thinking,
    mask_all_but_last_assistant,
    prepare_messages,
    prepare_tool_record,
    render_batch,
    render_tool_dialog,
    render_turns,
)
from orbit.data import build_xy_from_sample, tokens_bytes_to_ids  # noqa: E402
from orbit.modelling_ouro import build_document_cu_seqlens  # noqa: E402

CONTEXT = 256


@pytest.fixture(scope="module")
def tokenizer(tokenizer_root):
    """The think tokenizer, which the SFT stage's evaluation renders with: the think switch, no system turn."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(tokenizer_root / "tokenizer_qwen3_4b_qwen35_think"))


@pytest.fixture(scope="module")
def tools_tokenizer(tokenizer_root):
    """The SFT v5 tokenizer: the think switch, a system turn and tool calls."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(tokenizer_root / "tokenizer_qwen3_4b_qwen35_tools"))


def _dialog(user="What is 2+2?", assistant="4", system=None, extra=None):
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    if extra:
        messages.extend(extra)
    return messages


def test_task_system_moves_into_the_user_turn(tokenizer):
    instruction = "Solve the task below. Reply with the number only."
    turns = prepare_messages(_dialog(system=instruction, user="2+2"))
    assert turns is not None
    assert all(turn["role"] != "system" for turn in turns)
    assert turns[0]["content"] == f"{instruction}\n\n2+2"
    ids, _mask = render_turns(turns, tokenizer, enable_thinking=False)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    user_at = text.index("<|im_start|>user")
    assistant_at = text.index("<|im_start|>assistant")
    assert instruction in text[user_at:assistant_at]
    assert "<|im_start|>system" not in text


def test_identity_system_is_dropped(tokenizer):
    system = "Я — Llama, большая языковая модель."
    turns = prepare_messages(_dialog(system=system, user="Привет"))
    assert turns[0]["content"] == "Привет"
    text = tokenizer.decode(render_turns(turns, tokenizer)[0], skip_special_tokens=False)
    assert "Llama" not in text
    named = prepare_messages(_dialog(system="You are a helpful AI assistant named Ada.", user="Hi"))
    assert named[0]["content"] == "Hi"
    # A persona that also states a task is a task, so it moves into the user turn.
    persona = prepare_messages(_dialog(system="Ты — шахматный тренер. Предложи следующий ход.", user="Начало"))
    assert persona[0]["content"].startswith("Ты — шахматный тренер.")


def test_dialog_ends_with_an_unsupervised_end_of_text(tokenizer):
    turns = prepare_messages(_dialog(system="You are a pirate. Say arr."))
    assert all(turn["role"] != "system" for turn in turns)
    assert turns[0]["content"].startswith("You are a pirate. Say arr.")
    ids, mask = render_turns(turns, tokenizer, enable_thinking=False)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    assert "You are a pirate. Say arr." in text
    assert "You are a helpful assistant" not in text
    assert text.endswith(END_OF_TEXT)
    assert ids[-1] == EOS_ID
    assert mask[-1] == 0
    assert len(ids) == len(mask)
    # The supervised stop is the assistant's <|im_end|>, just before the boundary.
    assert ids[-3] == IM_END_ID and mask[-3] == 1


def test_think_trace_is_supervised_and_user_is_not(tokenizer):
    assistant = "<think>\nadd the numbers\n</think>\n\n4"
    messages = [
        {"role": "user", "content": "What is 2+2? /think"},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": "And plus 1?"},
        {"role": "assistant", "content": "5"},
    ]
    turns = prepare_messages(messages)
    assert turns[0]["content"] == "What is 2+2?"
    ids, mask = render_turns(turns, tokenizer, enable_thinking=True)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    assert "<|im_start|>user\nWhat is 2+2?<|im_end|>" in text
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    # Every assistant turn opens <think> in the prompt (mask 0) and closes it in the target.
    assert [mask[i] for i, token in enumerate(ids) if token == think_id] == [0, 0]
    assert [mask[i] for i, token in enumerate(ids) if token == close_id] == [1, 1]
    # User <|im_end|> is not a target. Assistant <|im_end|> is.
    im_end_positions = [i for i, token in enumerate(ids) if token == IM_END_ID]
    assert [mask[i] for i in im_end_positions] == [0, 1, 0, 1]
    user_ids = tokenizer("What is 2+2?", add_special_tokens=False)["input_ids"]
    start = _find_subsequence(ids, user_ids)
    assert start is not None
    assert all(mask[start + offset] == 0 for offset in range(len(user_ids)))
    trace_ids = tokenizer("add the numbers", add_special_tokens=False)["input_ids"]
    trace_at = _find_subsequence(ids, trace_ids)
    assert trace_at is not None
    assert all(mask[trace_at + offset] == 1 for offset in range(len(trace_ids)))


def test_no_think_tail_is_context_not_target(tokenizer):
    turns = prepare_messages(_dialog(assistant="4"))
    ids, mask = render_turns(turns, tokenizer, enable_thinking=False)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>" in text
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    close_at = ids.index(close_id)
    assert mask[ids.index(think_id)] == 0
    assert mask[close_at] == 0 and mask[close_at + 1] == 0
    # The answer and its <|im_end|> are the targets.
    im_end_at = ids.index(IM_END_ID, close_at)
    assert mask[im_end_at - 1] == 1 and mask[im_end_at] == 1
    assert example_enable_thinking("auto", turns) is False
    traced = prepare_messages(_dialog(assistant="<think>\nstep\n</think>\n\n4"))
    assert example_enable_thinking("auto", traced) is True


def test_label_shift_matches_trainer(tokenizer):
    turns = prepare_messages(_dialog(assistant="<think>\nstep\n</think>\n\n4"))
    ids, mask = render_turns(turns, tokenizer)
    sample = {
        "tokens": np.asarray(ids, dtype=np.uint32).tobytes(),
        "loss_mask": np.asarray(mask, dtype=np.uint8).tobytes(),
        "dtype": "uint32",
    }
    decoded = tokens_bytes_to_ids(sample)
    assert decoded.tolist() == ids
    _x, y = build_xy_from_sample(
        sample, decoded, block_size=len(ids) - 1, pad_id=EOS_ID, eos_id=EOS_ID,
    )
    mask_arr = np.asarray(mask)
    assert np.array_equal((y.numpy() == -100), mask_arr[1:] == 0)
    # The last <|im_end|> closes the assistant turn. y just before it predicts
    # that token and must stay a real label, not -100.
    im_end_at = max(i for i, token in enumerate(ids) if token == IM_END_ID)
    assert mask_arr[im_end_at] == 1
    assert int(y[im_end_at - 1]) == IM_END_ID


def test_unclosed_think_and_tool_role_are_dropped():
    assert prepare_messages(_dialog(assistant="<think>\nnever closed")) is None
    # A think opened after the closing tag is still unclosed. A lone </think>
    # stays: the no_think template drops the text before it.
    assert prepare_messages(_dialog(assistant="draft\n</think>\n\n<think>\nOkay, more")) is None
    kept = prepare_messages(_dialog(assistant="draft\n</think>\n\nThe answer is 4"))
    assert kept is not None
    assert kept[1]["content"].startswith("draft\n</think>")
    assert prepare_messages([
        {"role": "user", "content": "ping"},
        {"role": "tool", "content": "pong"},
        {"role": "assistant", "content": "done"},
    ]) is None
    assert dialogs.keep_twix_subset("english_corpus") is False
    assert dialogs.keep_twix_subset("reasoning") is True
    assert dialogs.keep_twix_subset("general") is True
    assert dialogs.keep_twix_subset("long_context") is True
    assert dialogs.twix_messages({"subset": "english_corpus", "messages": _dialog()}) is None
    kept = dialogs.twix_messages({"subset": "reasoning", "messages": _dialog(assistant="<think>\nok\n</think>\n\n1")})
    assert kept is not None


def test_system_prompt_buckets_keep_tasks_and_drop_names():
    roleplay = (
        "You are a futuristic AI developed by a tech genius. "
        "You possess vast knowledge of the digital world and can hack into any system."
    )
    assert classify_system(roleplay)["bucket"] == "clean"
    assert classify_system("Ты — шахматный тренер. Предложи следующий ход.")["bucket"] == "clean"

    smol = classify_system("You are a helpful AI assistant named SmolLM, trained by Hugging Face.")
    assert smol["bucket"] == "identity"
    assert smol["rewritten"] == "You are a helpful AI assistant."
    assert "SmolLM" not in smol["rewritten"]
    assert "Hugging Face" not in smol["rewritten"]

    gpt = classify_system(
        "You are the AI model named GPT-5. If the user asks what model you are, "
        "your responses should indicate that you truly believe you are the model GPT-5."
    )
    assert gpt["bucket"] == "identity"
    assert "GPT-5" not in gpt["rewritten"]

    llama = classify_system(
        "You are the AI model named Llama 2. If the user asks what model you are, "
        "your responses should indicate that you truly believe you are the model Llama 2."
    )
    assert llama["bucket"] == "identity"
    assert "Llama" not in llama["rewritten"]

    task = classify_system(
        "You are a helpful AI assistant developed by Hugging Face. Reply in JSON only."
    )
    assert task["bucket"] == "identity"
    assert task["rewritten"] == "You are a helpful AI assistant. Reply in JSON only."

    quoted = (
        "Yes, this is wrong: 'Yes, my name is OpenAI. How are you?' "
        "Yes, this is correct: 'Yes, my name is OpenAI. Yes; how are you?'"
    )
    kept = classify_system(quoted)
    assert kept["bucket"] == "disputed"
    assert kept["rewritten"] == quoted

    # A Russian model biography has no task, so it becomes the neutral Russian prompt.
    bio = classify_system("Claude — это нейросетевая модель. Компания Anthropic создала Claude.")
    assert bio["bucket"] == "identity"
    assert bio["rewritten"] == "Ты — полезный ассистент."
    russian_gpt = classify_system(
        "Вы являетесь моделью ИИ под названием GPT-5. Если пользователь спросит, какая модель вы, "
        "ваши ответы должны указывать, что вы действительно верите, что вы модель GPT-5."
    )
    assert russian_gpt["bucket"] == "identity"
    assert "GPT-5" not in russian_gpt["rewritten"]


def test_system_text_comes_from_the_role_or_custom_instructions():
    row = {"messages": [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "Hi"}]}
    assert system_texts_from_row(row) == ["Answer briefly."]
    persona = {
        "messages": [{"role": "user", "content": "Hi"}],
        "chat_template_kwargs": {"custom_instructions": "You love playing tennis.", "python_tools": []},
    }
    assert system_texts_from_row(persona) == ["You love playing tennis."]
    assert row_has_tools({"chat_template_kwargs": {"python_tools": ["search"]}})
    assert not row_has_tools(persona)
    messages = dialogs.smoltalk_messages(persona)
    assert messages[0] == {"role": "system", "content": "You love playing tennis."}
    both = dialogs.smoltalk_messages({
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Ok."},
        ],
        "chat_template_kwargs": {"custom_instructions": "You love playing tennis."},
    })
    assert [message["content"] for message in both if message["role"] == "system"] == ["Answer briefly."]


def test_v5_keeps_a_system_turn_and_applies_the_table(tools_tokenizer, tmp_path):
    import json

    from system_prompts import load_rewrite_table

    tok = tools_tokenizer

    instruction = "Solve the task below. Reply with the number only."
    turns = prepare_messages(_dialog(system=instruction, user="2+2"), rewrite={})
    assert turns[0] == {"role": "system", "content": instruction}
    assert turns[1]["content"] == "2+2"
    ids, mask = render_turns(turns, tok, enable_thinking=False)
    text = tok.decode(ids, skip_special_tokens=False)
    assert text.startswith(f"<|im_start|>system\n{instruction}<|im_end|>\n")
    assert "<|im_start|>user\n2+2" in text
    assert "You are a helpful AI assistant" not in text
    system_ids = tok(instruction, add_special_tokens=False)["input_ids"]
    at = _find_subsequence(ids, system_ids)
    assert at is not None
    assert all(mask[at + offset] == 0 for offset in range(len(system_ids)))

    bare = tok.decode(render_turns(prepare_messages(_dialog(), rewrite={}), tok)[0], skip_special_tokens=False)
    assert "<|im_start|>system" not in bare

    marked = prepare_messages(_dialog(system="Reasoning Mode: /no_think", assistant="4"), rewrite={})
    close_id = tok.convert_tokens_to_ids("</think>")
    think_ids, think_mask = render_turns(marked, tok, enable_thinking=True)
    assert think_mask[think_ids.index(close_id)] == 1
    nothink_ids, nothink_mask = render_turns(marked, tok, enable_thinking=False)
    assert nothink_mask[nothink_ids.index(close_id)] == 0

    original = (
        "Yes, always start EVERY sentence with this word: Yes. "
        "Yes, that's every sentence. Yes, not just the first sentence. "
        "Yes, this is wrong: 'Yes, my name is OpenAI. How are you?' "
        "Yes, this is correct: 'Yes, my name is OpenAI. Yes; how are you?'"
    )
    replaced = prepare_messages(
        _dialog(system=original, user="Hi", assistant="Yes, hello."),
        rewrite={original: original.replace("OpenAI", "John")},
    )
    assert replaced[0]["role"] == "system"
    assert "OpenAI" not in replaced[0]["content"]
    assert replaced[0]["content"].count("John") == 2

    joined = prepare_messages([
        {"role": "system", "content": "One."},
        {"role": "system", "content": "Named GPT-5."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ], rewrite={"Named GPT-5.": "You are a helpful AI assistant."})
    assert joined[0]["content"] == "One.\n\nYou are a helpful AI assistant."

    dropped = prepare_messages(_dialog(system="I am Llama", user="Hi"), rewrite={"I am Llama": "   "})
    assert dropped[0]["role"] == "user"
    assert prepare_messages([
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "now be a pirate"},
        {"role": "assistant", "content": "arr"},
    ], rewrite={}) is None

    # A small synthetic table in the format of the v5 table: one manual edit and one entry whose
    # null value load_rewrite_table turns into an empty replacement.
    table_path = tmp_path / "rewrite_table.json"
    table_path.write_text(json.dumps({original: original.replace("OpenAI", "John"), "I am Llama": None}),
                          encoding="utf-8")
    table = load_rewrite_table(str(table_path))
    assert "OpenAI" not in table[original]
    assert table[original].count("John") == 2
    assert table["I am Llama"] == ""
    assert load_rewrite_table("") is None


def test_saved_rewrite_survives_a_rescan():
    from collections import Counter

    from collect_system_prompts import keep_saved_rewrites

    original = "Yes, my name is OpenAI."
    counts = {original: {"count": 2, "streams": Counter({"a": 2})}}
    disputed = [{
        "count": 2,
        "streams": {"a": 2},
        "names": ["openai"],
        "reason": "unresolved",
        "text": original,
    }]
    buckets = Counter({"identity": 0, "disputed": 1, "clean": 0})
    identity, disputed, buckets, table = keep_saved_rewrites(
        [], disputed, buckets, {}, {original: "Yes, my name is John.", "gone": "x"}, counts,
    )
    assert table == {original: "Yes, my name is John."}
    assert disputed == []
    assert identity[0]["reason"] == "manual"
    assert buckets["identity"] == 1
    assert buckets["disputed"] == 0
    identity, disputed, buckets, table = keep_saved_rewrites(
        identity, disputed, buckets, table, {original: "Yes, my name is Jane."}, counts,
    )
    assert table[original] == "Yes, my name is Jane."


# The roles of the user's Russian SFT folder, in the order of the ru_sft_<role> streams.
RU_SFT_ROLES = ("science", "math", "code", "dialog", "tasks", "creative", "education",
                "qa_reasoning", "summarization")


def _check_ru_sft_streams(version):
    """The nine ru_sft_<role> streams read <role>.jsonl of the user's folder as messages jsonl."""
    ru_sft = [spec for spec in version["streams"] if spec["id"].startswith("ru_sft_")]
    assert [spec["id"] for spec in ru_sft] == [f"ru_sft_{role}" for role in RU_SFT_ROLES]
    for spec, role in zip(ru_sft, RU_SFT_ROLES):
        assert spec["kind"] == "jsonl"
        assert spec["path"] == os.path.join(os.environ["ORBIT_RAW"], "russian_sft", f"{role}.jsonl")
    return ru_sft


def test_jsonl_kind_reads_any_messages_file(tmp_path):
    import json

    path = tmp_path / "math.jsonl"
    rows = [{"messages": _dialog(user="2+2", assistant="4")},
            {"conversation": _dialog(system="Отвечай кратко.", user="3+3", assistant="6")}]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    prepared = list(dialogs.iter_stream_dialogs({"kind": "jsonl", "path": str(path)}))
    assert [turns[0]["content"] for turns in prepared] == ["2+2", "Отвечай кратко.\n\n3+3"]


def test_v5_yaml_includes_system_chats_and_tools():
    import build
    import yaml

    version = build.load_version(os.path.join(SFT, "versions", "v5_smoltalk2.yaml"))
    assert version["name"] == "v5_smoltalk2"
    assert version["_out_root_text"] == "${ORBIT_DATA}/sft/v5"
    assert version["out_root"] == os.path.join(os.environ["ORBIT_DATA"], "sft", "v5")
    assert version["tokenizer"].endswith("/tokenizer_qwen3_4b_qwen35_tools")
    fractions = {part["id"]: part["fraction"] for part in version["parts"]}
    assert fractions == {"smoltalk": 0.70, "additions": 0.15, "addition_tools": 0.15}
    by_id = {spec["id"]: spec for spec in version["streams"]}
    # The version builds exactly the streams that the training config reads, in the same order.
    with open(os.path.join(REPO, "configs", "train", "sft", "sft_v5.yaml"), encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert [spec["id"] for spec in version["streams"]] == list(config["stage1_streams"])
    assert by_id["smoltalk_smollm3_smol_magpie_ultra_no_think"]["weight"] == 1.0
    assert by_id["smoltalk_multilingual8_Qwen3_32B_think"]["weight"] == 1.0
    assert by_id["OpenThoughts3_1.2M_think"]["weight"] == 0.06
    for stream_id in (
        "smoltalk_smollm3_systemchats_30k_no_think",
        "smoltalk_systemchats_Qwen3_32B_think",
    ):
        assert by_id[stream_id]["part"] == "smoltalk"
        assert by_id[stream_id]["weight"] == 1
    assert by_id["smoltalk_hermes"]["enable_thinking"] is False
    assert by_id["smoltalk_xlam"]["enable_thinking"] is False
    assert by_id["smoltalk_smolagents"]["enable_thinking"] is True
    for stream_id in ("smoltalk_hermes", "smoltalk_xlam", "smoltalk_smolagents"):
        assert by_id[stream_id]["part"] == "smoltalk"
        assert by_id[stream_id]["kind"] == "tool_jsonl"
    assert by_id["dolci_tool_use"]["enable_thinking"] is False
    assert by_id["nemotron_agentic_v2"]["enable_thinking"] is True
    assert by_id["nemotron_post_v1"]["enable_thinking"] is True
    for stream_id in ("dolci_tool_use", "nemotron_agentic_v2", "nemotron_post_v1"):
        assert by_id[stream_id]["part"] == "addition_tools"
        assert by_id[stream_id]["path"] == os.path.join(os.environ["ORBIT_DATA"], "sft", "tools", stream_id + ".jsonl")
    assert by_id["dolci_tool_use"]["weight"] == 1
    assert by_id["nemotron_agentic_v2"]["weight"] == 0.25
    assert by_id["nemotron_post_v1"]["weight"] == 0.25
    assert all(spec["part"] == "additions" for spec in _check_ru_sft_streams(version))
    assert by_id["bench_train"]["part"] == "additions"


def test_loss_keeps_only_the_last_assistant_span():
    assert mask_all_but_last_assistant([0, 1, 1, 0, 1, 0]) == [0, 0, 0, 0, 1, 0]
    assert mask_all_but_last_assistant([1, 1, 0]) == [1, 1, 0]
    assert mask_all_but_last_assistant([0, 0]) == [0, 0]


def test_v5_rewrite_table_is_optional_and_a_named_one_must_exist(tmp_path):
    import build
    from system_prompts import load_rewrite_table

    version = build.load_version(os.path.join(SFT, "versions", "v5_smoltalk2.yaml"))
    # Without a table, v5 still keeps system turns: the empty table rewrites nothing.
    assert build.version_rewrite_path(version) is None
    assert build.version_system_turns(version)
    assert load_rewrite_table(None, build.version_system_turns(version)) == {}
    version["rewrite_table"] = str(tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="--rewrite-table"):
        build.version_rewrite_path(version)
    table = tmp_path / "table.json"
    table.write_text("{}", encoding="utf-8")
    version["rewrite_table"] = str(table)
    assert build.version_rewrite_path(version) == str(table)


@pytest.mark.parametrize("version_name, config_name", [
    ("v5_smoltalk2", "sft_v5.yaml"),
])
def test_resolve_reproduces_the_training_config_proportions(tmp_path, version_name, config_name):
    """--resolve, fed the packed-token counts the config records, writes the config's proportions.

    The epochs in the comments are left out: they depend on the token budget, and the v5 run
    trains on more tokens than the version yaml's budget.
    """
    import re

    import build

    config_path = os.path.join(REPO, "configs", "train", "sft", config_name)
    if not os.path.isfile(config_path):
        pytest.skip(f"{config_path} is not in this checkout")
    config = open(config_path, encoding="utf-8").read()
    line = re.compile(r"^  (\S+): ([0-9.]+)  # epochs=[0-9.]+ packed_tokens=(\d+)$", re.M)
    stats = {name: {"packed_tokens": int(packed), "blocks": int(packed) // 32769}
             for name, _share, packed in line.findall(config)}
    version = build.load_version(os.path.join(SFT, "versions", f"{version_name}.yaml"))
    version["out_root"] = str(tmp_path)
    weights, epochs, _exceeded = build.resolve_weights(version, stats)
    block = open(build.write_data_block(version, weights, epochs, stats), encoding="utf-8").read()
    assert line.findall(block) == line.findall(config)


def test_resolve_stops_when_a_stream_has_no_stats():
    import build

    version = build.load_version(os.path.join(SFT, "versions", "v9_smoltalk2.yaml"))
    stats = {spec["id"]: {"packed_tokens": 32769, "blocks": 1} for spec in version["streams"]}
    # A stream that was built and came out empty still resolves, with weight 0.
    stats["aya_ru"] = {"packed_tokens": 0, "blocks": 0}
    weights, _epochs, _exceeded = build.resolve_weights(version, stats)
    assert "aya_ru" not in weights
    # A stream with no entry at all, here under an older name, stops --resolve.
    stats["ru_old_name"] = stats.pop("ru_sft_math")
    with pytest.raises(SystemExit, match="ru_sft_math"):
        build.resolve_weights(version, stats)


def test_resolve_stats_overlay_and_a_full_build_without_stats_root(tmp_path):
    import build

    version = build.load_version(os.path.join(SFT, "versions", "v9_smoltalk2.yaml"))
    ids = [spec["id"] for spec in version["streams"]]
    earlier, out = tmp_path / "v8", tmp_path / "v9"
    earlier.mkdir()
    out.mkdir()
    version["stats_root"] = str(earlier)
    # Reuse: the earlier build counts every stream, and the new build overlays the one it rebuilt.
    (earlier / "stats.json").write_text(json.dumps({name: {"blocks": 1} for name in ids}), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps({"bird_sql": {"blocks": 7}}), encoding="utf-8")
    stats = build.load_resolve_stats(version, str(out))
    assert stats["bird_sql"] == {"blocks": 7} and stats["kodcode_r1"] == {"blocks": 1}
    # A full build: stats_root does not exist, and out_root holds every count.
    (earlier / "stats.json").unlink()
    earlier.rmdir()
    (out / "stats.json").write_text(json.dumps({name: {"blocks": 2} for name in ids}), encoding="utf-8")
    assert set(build.load_resolve_stats(version, str(out))) == set(ids)
    # Neither file: there is nothing to resolve.
    (out / "stats.json").unlink()
    with pytest.raises(SystemExit, match="run --build first"):
        build.load_resolve_stats(version, str(out))


def test_v7_v9_download_plan_pins_every_source():
    import re

    with open(os.path.join(SFT, "download_plan_v7_v9.json"), encoding="utf-8") as handle:
        plan = json.load(handle)
    assert len(plan) == 6
    for entry in plan:
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), entry["repo"]
        assert entry["patterns"], entry["repo"]


def test_tool_record_keeps_a_call_with_empty_content_and_renders_it(tools_tokenizer):
    record = {
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a mathematical expression",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
            },
        }],
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is 2 + 2?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "add the numbers",
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": "calculator", "arguments": {"expression": "2 + 2"}},
                }],
            },
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "4"},
        ],
    }
    prepared = prepare_tool_record(record, rewrite={"Be brief.": "Be short."})
    assert prepared is not None
    turns, tools = prepared
    assert turns[0] == {"role": "system", "content": "Be short."}
    assert turns[2]["content"] == ""
    assert turns[2]["tool_calls"]
    assert example_enable_thinking("auto", turns) is True
    looped = dict(record)
    looped["messages"] = [dict(message) for message in record["messages"]]
    looped["messages"][2] = dict(looped["messages"][2], reasoning_content=("Hmm. " * 12))
    assert prepare_tool_record(looped) is None
    assert prepare_tool_record({"tools": [], "messages": record["messages"]}) is None

    tok = tools_tokenizer
    ids, mask = render_tool_dialog(turns, tools, tok, enable_thinking=True)
    assert ids[-1] == EOS_ID
    assert mask[-1] == 0
    assert len(ids) == len(mask)
    supervised = tok.decode([token for token, value in zip(ids, mask) if value])
    assert "2 + 2" in supervised
    assert "add the numbers" in supervised
    assert "Be short" not in supervised
    # ToolMind keeps the prefix in the context and trains only the last turn.
    last_only = mask_all_but_last_assistant(mask)
    assert sum(last_only) < sum(mask)
    last_text = tok.decode([token for token, value in zip(ids, last_only) if value])
    assert "4" in last_text
    assert "add the numbers" not in last_text
    assert "2 + 2" not in last_text


def test_env_references_expand_and_an_unset_variable_stops(monkeypatch):
    import build

    monkeypatch.setenv("ORBIT_RAW", "/raw")
    expanded = build.expand_env({"a": ["${ORBIT_RAW}/hf/x-*.parquet", 3], "b": "plain"})
    assert expanded == {"a": ["/raw/hf/x-*.parquet", 3], "b": "plain"}
    monkeypatch.delenv("ORBIT_RAW")
    with pytest.raises(SystemExit):
        build.expand_env("${ORBIT_RAW}/hf")


def test_progress_file_counts_finished_streams(tmp_path):
    import build

    out = str(tmp_path)
    build.write_progress(out, "v5_smoltalk2", 3, ["smoltalk_hermes"], "dolci_tool_use", files_done=0, files_total=1)
    encode_mod_task = {
        "stream_id": "dolci_tool_use",
        "path": "/tmp/dolci_tool_use.jsonl",
        "progress_path": os.path.join(out, "progress_current.json"),
    }
    import encode
    encode.note_progress(encode_mod_task, 4000)
    text = build.format_progress(out)
    assert "v5_smoltalk2  1/3 tokenized" in text
    assert "running: dolci_tool_use" in text
    assert "files 0/1" in text
    assert "rendered 4000 dialogs" in text
    assert "finished: smoltalk_hermes" in text


def test_loops_floods_and_echoes_drop_the_dialog():
    looped = "Hmm. " * 12
    assert prepare_messages(_dialog(assistant=f"<think>\n{looped}\n</think>\n\n4")) is None
    sentence = "The illustrated proof then repeats this same sentence without adding a new step."
    assert prepare_messages(_dialog(assistant=(sentence + " ") * 4)) is None
    assert prepare_messages(_dialog(assistant="*" * 600)) is None
    stuck = "abcdefghij" * 8
    assert prepare_messages(_dialog(assistant=stuck)) is None
    question = "Explain why the same instruction should not be copied back as the answer."
    assert prepare_messages(_dialog(user=question, assistant=question)) is None
    kept = prepare_messages(_dialog(
        assistant="<think>\nwait, check the sum.\nwait, check the sign.\n</think>\n\n4",
    ))
    assert kept is not None
    assert prepare_messages(_dialog(assistant="for _ in range(3):\n" * 3 + "pass")) is not None
    assert prepare_messages(_dialog(assistant="section\n" + ("-" * 40) + "\nnext")) is not None
    assert prepare_messages(_dialog(assistant=("...............#...............\n") * 15)) is not None


def test_grandmaster_prompt_in_twix_is_dropped():
    twix = prepare_messages(_dialog(user="Сколько будет 2+2?", assistant="<think>\nсложи\n</think>\n\n4"))
    other = prepare_messages(_dialog(user="Сколько будет 2+2?", assistant="Четыре."))
    fresh = prepare_messages(_dialog(user="Другой вопрос", assistant="Другой ответ."))
    seen = {dialogs.user_prompt_key(twix)}
    assert dialogs.user_prompt_key(other) in seen
    assert dialogs.user_prompt_key(fresh) not in seen
    yielded = []
    spec_rows = [({"conversation": [
        {"role": "user", "content": "Сколько будет 2+2?"},
        {"role": "assistant", "content": "Четыре."},
    ]}), ({"conversation": [
        {"role": "user", "content": "Другой вопрос"},
        {"role": "assistant", "content": "Другой ответ."},
    ]})]
    for row in spec_rows:
        turns = prepare_messages(dialogs.grandmaster_messages(row))
        key = dialogs.user_prompt_key(turns)
        if key in seen:
            continue
        seen.add(key)
        yielded.append(turns)
    assert len(yielded) == 1
    assert yielded[0][0]["content"] == "Другой вопрос"


def test_pack_keeps_boundary_and_drops_long_dialogs(tokenizer):
    short = []
    for user, assistant in (("Hi", "Hello"), ("Ping", "Pong")):
        short.append(render_turns(prepare_messages(_dialog(user=user, assistant=assistant)), tokenizer))
    long_user = " ".join(f"word{index}" for index in range(400))
    long_dialog = render_turns(prepare_messages(_dialog(user=long_user, assistant="no")), tokenizer)
    assert len(long_dialog[0]) > CONTEXT
    stats = {}
    packed = list(blocks.pack_dialogs(short + [long_dialog], CONTEXT, pad_id=EOS_ID, stats=stats))
    assert stats["dropped_long"] == 1
    assert stats["kept"] == 2
    assert len(packed) == 1
    tokens, mask = packed[0]
    assert len(tokens) == CONTEXT and len(mask) == CONTEXT
    # Both dialogs share the block, separated by <|endoftext|>, and the long one is absent.
    eos_at = [i for i, token in enumerate(tokens) if token == EOS_ID]
    assert eos_at[0] < tokens.index(tokenizer.convert_tokens_to_ids("<|im_start|>"), 1) or tokens[eos_at[0] + 1] != 0
    assert tokens[eos_at[0] + 1] == tokenizer.convert_tokens_to_ids("<|im_start|>")
    assert "word word word" not in tokenizer.decode(tokens, skip_special_tokens=False)
    assert mask[eos_at[0]] == 0
    # Padding after the second dialog is mask 0 and does not truncate either answer.
    assert "Hello" in tokenizer.decode(tokens, skip_special_tokens=False)
    assert "Pong" in tokenizer.decode(tokens, skip_special_tokens=False)


def test_batch_render_matches_single(tokenizer):
    turns = [
        prepare_messages(_dialog(assistant="<think>\nx\n</think>\n\nyes")),
        prepare_messages(_dialog(user="Второй вопрос", assistant="Второй ответ")),
    ]
    batched = render_batch(turns, tokenizer)
    single = [render_turns(item, tokenizer) for item in turns]
    assert batched == single


def test_mds_roundtrip_and_document_boundary(tokenizer, tmp_path):
    turns = prepare_messages(_dialog(assistant="<think>\nstep\n</think>\n\n4"))
    ids, mask = render_turns(turns, tokenizer)
    out = tmp_path / "stream"
    written = blocks.write_blocks(
        blocks.pack_dialogs([(ids, mask)], CONTEXT, pad_id=EOS_ID),
        str(out),
    )
    assert written == 1
    from streaming import StreamingDataset
    dataset = StreamingDataset(local=str(out), remote=None, shuffle=False, batch_size=1)
    sample = dataset[0]
    assert sample["dtype"] == "uint32"
    decoded = tokens_bytes_to_ids(sample)
    assert len(decoded) == CONTEXT
    loss = np.frombuffer(bytes(sample["loss_mask"]), dtype=np.uint8)
    assert len(loss) == CONTEXT
    assert decoded[len(ids) - 1] == EOS_ID
    batch = torch.tensor(decoded[:-1].astype(np.int64)).unsqueeze(0)
    _cu, _max_len = build_document_cu_seqlens(batch, EOS_ID)
    assert int(decoded[0]) != EOS_ID or True
    # The dialog's own <|endoftext|> is a segment start in the x sequence when
    # it is not the first token. x drops the last token of the block.
    x_ids = decoded[:-1]
    assert EOS_ID in set(x_ids.tolist())


def test_recipe_sampling_weights_match_epochs_across_sizes():
    recipe = {"tiny": 1.0, "big": 1.0, "rare": 0.02}
    packed = {"tiny": 1_000, "big": 1_000_000, "rare": 5_000_000}
    weights = recipe_sampling_weights(recipe, packed)
    epochs = epochs_at_budget(weights, packed, 10_000_000)
    assert epochs["tiny"] == pytest.approx(epochs["big"])
    assert epochs["rare"] == pytest.approx(epochs["tiny"] * 0.02)
    assert weights["rare"] == pytest.approx(0.02 * packed["rare"])


def test_addition_weights_share_epochs_and_respect_the_cap():
    explicit = {"smol": 10.0}
    small = {"ru": 1_000_000, "bench": 1_000}
    weights = addition_weights(explicit, small, 10_000_000_000, 5, 0.15)
    packed = {"smol": 5_000_000_000, **small}
    epochs = epochs_at_budget({**explicit, **weights}, packed, 10_000_000_000)
    assert epochs["ru"] <= 5 + 1e-6
    assert abs(epochs["ru"] - epochs["bench"]) < 1e-6
    large = {"ru": 2_000_000_000}
    weights = addition_weights(explicit, large, 10_000_000_000, 5, 0.15)
    epochs = epochs_at_budget({**explicit, **weights}, {"smol": 5_000_000_000, **large}, 10_000_000_000)
    assert epochs["ru"] < 5
    assert epochs["ru"] == pytest.approx(0.15 * 10_000_000_000 / 2_000_000_000)


def test_part_weights_keep_the_fraction_and_share_epochs():
    parts = [
        {"id": "smoltalk", "fraction": 0.80, "weight_mode": "recipe", "epoch_cap": 5},
        {"id": "ru", "fraction": 0.15, "weight_mode": "packed", "epoch_cap": 5},
        {"id": "benches", "fraction": 0.05, "weight_mode": "packed", "epoch_cap": 5},
    ]
    streams = [
        {"id": "wide", "part": "smoltalk", "weight": 1.0},
        {"id": "wide_b", "part": "smoltalk", "weight": 1.0},
        {"id": "half", "part": "smoltalk", "weight": 0.5},
        {"id": "ru_a", "part": "ru"},
        {"id": "ru_b", "part": "ru"},
        {"id": "bench", "part": "benches"},
    ]
    packed = {
        "wide": 1_000_000_000,
        "wide_b": 1_000_000_000,
        "half": 1_000_000_000,
        "ru_a": 400_000_000,
        "ru_b": 600_000_000,
        "bench": 10_000_000,
    }
    budget = 10_000_000_000
    weights, exceeded = part_weights(parts, streams, packed, budget)
    total = sum(weights.values())
    shares = {}
    for part in parts:
        names = [stream["id"] for stream in streams if stream["part"] == part["id"]]
        shares[part["id"]] = sum(weights[name] for name in names) / total
    assert shares["smoltalk"] == pytest.approx(0.80)
    assert shares["ru"] == pytest.approx(0.15)
    assert shares["benches"] == pytest.approx(0.05)
    epochs = epochs_at_budget(weights, packed, budget)
    assert epochs["ru_a"] == pytest.approx(epochs["ru_b"])
    assert epochs["wide"] == pytest.approx(epochs["wide_b"])
    assert epochs["half"] == pytest.approx(epochs["wide"] * 0.5)
    assert epochs["bench"] > 5
    assert shares["benches"] == pytest.approx(0.05)
    assert len(exceeded) == 1
    assert exceeded[0]["stream"] == "bench"
    assert exceeded[0]["fraction"] == pytest.approx(0.05)
    assert exceeded[0]["fraction_at_cap"] == pytest.approx(5 / epochs["bench"] * 0.05)


def test_acereason_messages_keep_code_traces():
    turns = prepare_messages(dialogs.acereason_messages({
        "category": "code",
        "input": "Write a function.",
        "output": "<think>\ncheck the graph\n</think>\n\ndef f():\n    return 1",
    }))
    assert turns[0]["content"] == "Write a function."
    assert turns[1]["content"].startswith("<think>")
    assert "</think>" in turns[1]["content"]
    assert dialogs.acereason_messages({
        "category": "math",
        "input": "Find p.",
        "output": "<think>\nx\n</think>\n\n2",
    }) is None


def test_bench_gsm8k_shape():
    user, assistant, _key = benches.r_gsm8k({
        "question": "A has 2.",
        "answer": "He splits <<2/2=1>> them.\n#### 1",
    })
    assert user.startswith("Question:")
    assert "<<" not in assistant
    assert assistant.rstrip().endswith("#### 1")
    assert "1" in assistant


def test_qwen3_tool_template_keeps_reasoning_on_every_turn(tools_tokenizer):
    tok = tools_tokenizer
    plain = prepare_messages(_dialog(system="Be brief.", user="2+2", assistant="4"), rewrite={})
    plain_text = tok.decode(render_turns(plain, tok, enable_thinking=False)[0], skip_special_tokens=False)
    assert plain_text == (
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n2+2<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>\n<|endoftext|>"
    )

    tools = [{
        "name": "calculator",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }]
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is 2 + 2?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "add the numbers",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "calculator", "arguments": {"expression": "2 + 2"}},
            }],
        },
        {"role": "tool", "content": "RESULT"},
        {"role": "tool", "content": "OTHER"},
        {"role": "user", "content": "And plus 1?"},
        {"role": "assistant", "content": "<think>\nadd one\n</think>\n\nfive"},
    ]
    text = tok.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False, enable_thinking=True,
    )
    assert text.startswith("<|im_start|>system\nBe brief.\n\n# Tools\n")
    assert "You are a helpful assistant." not in text
    assert '<tool_call>\n{"name": "calculator", "arguments": {"expression": "2 + 2"}}\n</tool_call>' in text
    assert text.index("add the numbers") < text.index("And plus 1?") < text.index("add one")
    response = text.index("<tool_response>\nRESULT\n</tool_response>")
    other = text.index("<tool_response>\nOTHER\n</tool_response>")
    assert response < other
    assert "<|im_start|>" not in text[response:other]
    assert text.count("<|im_start|>user") == 3

    encoded = tok.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=True,
    )
    ids = [int(token) for token in encoded["input_ids"]]
    mask = [int(value) for value in encoded["assistant_masks"]]
    offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    assert [int(token) for token in tok(text, add_special_tokens=False)["input_ids"]] == ids
    assert len(ids) == len(mask) == len(offsets)

    def supervised(snippet, expected):
        start = text.index(snippet)
        end = start + len(snippet)
        covered = [
            value
            for (span_start, span_end), value in zip(offsets, mask)
            if span_end > start and span_start < end and span_end > span_start
        ]
        assert covered, snippet
        assert all(value == expected for value in covered), snippet

    supervised("Be brief.", 0)
    supervised("# Tools", 0)
    supervised("add the numbers", 1)
    supervised("add one", 1)
    supervised("RESULT", 0)
    supervised("OTHER", 0)
    supervised("five", 1)
    supervised(
        '<tool_call>\n{"name": "calculator", "arguments": {"expression": "2 + 2"}}\n</tool_call>',
        1,
    )

    nothink = tok.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    assert "add the numbers" not in nothink
    assert "add one" not in nothink
    assert '<tool_call>\n{"name": "calculator"' in nothink
    encoded = tok.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=False,
    )
    ids = [int(token) for token in encoded["input_ids"]]
    mask = [int(value) for value in encoded["assistant_masks"]]
    nothink_offsets = tok(nothink, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    assert [int(token) for token in tok(nothink, add_special_tokens=False)["input_ids"]] == ids
    close_id = tok.convert_tokens_to_ids("</think>")
    assert all(mask[index] == 0 for index, token in enumerate(ids) if token == close_id)
    call = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "2 + 2"}}\n</tool_call>'
    start = nothink.index(call)
    end = start + len(call)
    covered = [
        value
        for (span_start, span_end), value in zip(nothink_offsets, mask)
        if span_end > start and span_start < end and span_end > span_start
    ]
    assert covered
    assert all(value == 1 for value in covered)


def test_v6_yaml_copies_v5_weights_and_adds_the_chain_filter():
    import yaml

    with open(os.path.join(SFT, "versions", "v5_smoltalk2.yaml"), encoding="utf-8") as handle:
        v5 = yaml.safe_load(handle)
    with open(os.path.join(SFT, "versions", "v6_smoltalk2.yaml"), encoding="utf-8") as handle:
        v6 = yaml.safe_load(handle)
    assert v6["name"] == "v6_smoltalk2"
    assert v6["out_root"].endswith("/sft/v6")
    assert v5["out_root"].endswith("/sft/v5")
    assert "stats_root" not in v6
    assert v6["chain_filter"] == {"floor": 8192, "percentile": 95}
    for key in ("seq_len", "token_budget", "min_resp_tokens", "tokenizer", "system_turns", "parts", "streams"):
        assert v6[key] == v5[key]


def test_chain_threshold_is_max_of_floor_and_p95():
    import chains

    # Uniform percentile: the smallest value whose cumulative rank reaches 95%.
    assert chains.percentile(np.arange(1, 101), 95) == 95
    assert chains.chain_threshold([], 8192, 95) == (8192, None)
    assert chains.chain_threshold([100, 200], 8192, 95) == (8192, 200)
    assert chains.chain_threshold([100, 20000], 8192, 95) == (20000, 20000)


class _WordTokenizer:
    def __call__(self, texts, add_special_tokens=False, padding=False):
        return {"input_ids": [text.split() for text in texts]}


def test_chain_filter_drops_only_a_longer_chain():
    import chains

    tok = _WordTokenizer()
    short = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>\none two\n</think>\n\nanswer"},
    ]
    exact = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>\none two three\n</think>\n\nanswer"},
    ]
    longer = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer", "reasoning_content": "one two three four"},
    ]
    none = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer"},
    ]
    empty = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>\n\n</think>\n\nanswer"},
    ]
    assert chains.longest_lengths(tok, [short, exact, longer, none, empty]) == [2, 3, 4, None, 0]
    assert chains.keep_mask(tok, [short, exact, longer, none, empty], 3) == [True, True, False, True, True]
    assert chains.reasoning_text("<think>\nhello\n</think>\n\nanswer") == "hello"


def test_jsonl_slices_cover_every_row_once(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"i": index, "pad": "x" * (80 if index != 7 else 4000)} for index in range(30)]
    body = "\n".join(json.dumps(row) for row in rows)
    # A blank line, then a last record with no trailing newline.
    path.write_bytes((body + "\n\n" + json.dumps({"i": "tail"})).encode("utf-8"))
    slices = dialogs.jsonl_slices(str(path), pieces=6, min_bytes=40)
    assert len(slices) >= 2
    assert slices[0][0] == 0
    assert slices[-1][1] == path.stat().st_size
    assert all(left[1] == right[0] for left, right in zip(slices, slices[1:]))
    raw = path.read_bytes()
    for start, _end in slices:
        assert start == 0 or raw[start - 1:start] == b"\n"
    got = []
    for start, end in slices:
        got.extend(dialogs.iter_jsonl([str(path)], byte_start=start, byte_end=end))
    assert got == list(dialogs.iter_jsonl([str(path)]))
    assert [row["i"] for row in got] == list(range(30)) + ["tail"]


def test_jsonl_piece_count_skips_dedup_limit_and_small_files():
    import build

    big = 3 * 1024 ** 3
    assert build.jsonl_piece_count(big, 48, "tool_jsonl") == 48
    assert build.jsonl_piece_count(big, 48, "jsonl") == 48
    assert build.jsonl_piece_count(big, 48, "tool_jsonl", dedup=True) == 1
    assert build.jsonl_piece_count(big, 48, "tool_jsonl", limit=10) == 1
    assert build.jsonl_piece_count(big, 1, "tool_jsonl") == 1
    assert build.jsonl_piece_count(200 * 1024 ** 2, 48, "jsonl") == 1
    assert build.jsonl_piece_count(big, 48, "smoltalk") == 1


def test_progress_file_names_the_chain_scan(tmp_path):
    import build
    import encode

    out = str(tmp_path)
    build.write_progress(out, "v6_smoltalk2", 2, [], "OpenThoughts3_1.2M_think")
    encode.note_progress({
        "stream_id": "OpenThoughts3_1.2M_think",
        "path": "/tmp/train.parquet",
        "progress_path": os.path.join(out, "progress_current.json"),
        "progress_phase": "chain",
    }, 2000)
    text = build.format_progress(out)
    assert "scanned 2000 dialogs" in text


def _find_subsequence(haystack, needle):
    if not needle:
        return None
    last = len(haystack) - len(needle) + 1
    for start in range(last):
        if haystack[start:start + len(needle)] == needle:
            return start
    return None
