"""Stored shape of tool-calling dialogs, on synthetic rows."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SFT = os.path.dirname(HERE)
sys.path.insert(0, SFT)

from tool_records import Skip, build_record  # noqa: E402


DOLCI_SYSTEM = (
    "You are a helpful function-calling AI assistant. "
    "You are provided with function signatures within <functions></functions> XML tags. "
    "Output any function calls within <function_calls></function_calls> XML tags."
)


def test_hermes_python_calls_and_wrapped_results_become_json():
    tools = """<tools>
[{'type': 'function', 'function': {'name': 'get_camera_live_feed', 'description': 'Live feed.', 'parameters': {'type': 'object', 'properties': {'camera_id': {'type': 'string'}}, 'required': ['camera_id']}}}]
</tools>"""
    messages = [
        {"role": "user", "content": "Show the front door."},
        {"role": "assistant", "content": (
            "<tool_call>\n"
            "{'arguments': {'camera_id': 'front_door', 'stream_quality': '1080p'}, 'name': 'get_camera_live_feed'}\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "{'arguments': {'camera_id': 'front_door', 'duration': 30}, 'name': 'record_camera_feed'}\n"
            "</tool_call>"
        )},
        {"role": "tool", "content": (
            "<tool_response>\n{\"name\": \"get_camera_live_feed\"}\n</tool_response>\n"
            "<tool_response>\n{\"name\": \"record_camera_feed\"}\n</tool_response>\n"
        )},
        {"role": "assistant", "content": "The feed is up."},
    ]
    record = build_record("smoltalk_hermes", [tools], messages)
    assert record["tools"][0]["function"]["name"] == "get_camera_live_feed"
    call = record["messages"][1]
    assert call["tool_calls"][0]["function"]["arguments"] == {
        "camera_id": "front_door",
        "stream_quality": "1080p",
    }
    assert call["content"] == ""
    assert [turn["role"] for turn in record["messages"]] == ["user", "assistant", "tool", "tool", "assistant"]
    assert record["messages"][2]["content"] == '{"name": "get_camera_live_feed"}'
    assert "<tool_response>" not in record["messages"][2]["content"]


def test_hermes_keeps_the_task_text_and_drops_the_foreign_call_recipe():
    blob = (
        "You are an expert structured information extraction AI model. "
        "Extract questions.\n"
        "<tools>\n"
        '[{"type": "function", "function": {"name": "ExpertQAExtractor", "description": "Extract.", '
        '"parameters": {"type": "object", "properties": {}}}}]\n'
        "</tools>\n"
        "For each extraction function call return a json object followed by a <tool_call> tag with "
        "{'arguments': <args-dict>, 'name': <function-name>}\n"
        "</tool_call>"
    )
    messages = [
        {"role": "user", "content": "Document text."},
        {"role": "assistant", "content": "<tool_call>\n{'arguments': {'open_ended_questions': ['Q']}, 'name': 'ExpertQAExtractor'}\n</tool_call>"},
    ]
    record = build_record("smoltalk_hermes", [blob], messages)
    assert record["messages"][0] == {
        "role": "system",
        "content": "You are an expert structured information extraction AI model. Extract questions.",
    }
    assert "args-dict" not in record["messages"][0]["content"]
    assert record["tools"][0]["function"]["name"] == "ExpertQAExtractor"
    assert record["messages"][2]["tool_calls"][0]["function"]["name"] == "ExpertQAExtractor"


def test_hermes_schema_without_tags_and_name_nested_in_arguments():
    blob = (
        "You are an expert structured information extraction AI model. Extract questions. "
        '{"type": "function", "function": {"name": "ExpertQAExtractor", "description": "Extract.", '
        '"parameters": {"type": "object", "properties": {}}}}\n'
        "For each extraction function call return a json object."
    )
    messages = [
        {"role": "user", "content": "Document text."},
        {"role": "assistant", "content": (
            '<tool_call>\\n{"arguments": {"queries": ["What is grading?"], "name": "ExpertQAExtractor"}}\\n</tool_call>'
        )},
    ]
    record = build_record("smoltalk_hermes", [blob], messages)
    assert record["messages"][0]["role"] == "system"
    assert record["messages"][0]["content"].startswith("You are an expert")
    call = record["messages"][2]["tool_calls"][0]["function"]
    assert call["name"] == "ExpertQAExtractor"
    assert call["arguments"] == {"queries": ["What is grading?"]}


def test_xlam_json_call_has_no_tool_result():
    tools = '<tools>\n[{"type": "function", "function": {"name": "live_giveaways_by_type", "description": "Giveaways.", "parameters": {"type": "object", "properties": {}}}}]\n</tools>'
    messages = [
        {"role": "user", "content": "Where are the beta giveaways?"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}\n</tool_call>'},
    ]
    record = build_record("smoltalk_xlam", [tools], messages)
    assert record["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"type": "beta"}
    assert all(turn["role"] != "tool" for turn in record["messages"])


def test_smolagents_keeps_the_chain_and_the_observation():
    tools = (
        '{"type": "function", "function": {"name": "web_search", "description": "Search.", "parameters": {"type": "object", "properties": {}}}}\n'
        '{"type": "function", "function": {"name": "final_answer", "description": "Answer.", "parameters": {"type": "object", "properties": {}}}}'
    )
    messages = [
        {"role": "user", "content": "Which year?"},
        {"role": "assistant", "content": (
            "<think>I need the year.</think>\n"
            "<tool_call>{'name': 'web_search', 'arguments': {'query': 'independence year'}}</tool_call>"
        )},
        {"role": "tool", "content": "Observation:\nThe Poisonwood Bible"},
        {"role": "assistant", "content": "<think>It is 1960.</think>\n<tool_call>{'name': 'final_answer', 'arguments': {'answer': '1960'}}</tool_call>"},
    ]
    record = build_record("smoltalk_smolagents", [tools], messages)
    first = record["messages"][1]
    assert first["reasoning_content"] == "I need the year."
    assert "<think>" not in first["content"]
    assert first["tool_calls"][0]["function"]["name"] == "web_search"
    assert record["messages"][2]["content"].startswith("Observation:")
    assert len(record["tools"]) == 2


def test_dolci_calls_parse_and_the_protocol_system_is_dropped():
    tools = json.dumps([{
        "type": "function",
        "function": {
            "name": "predict_forest_area",
            "description": "Predict.",
            "parameters": {"type": "object", "properties": {}},
        },
    }])
    messages = [
        {"role": "system", "content": DOLCI_SYSTEM, "functions": tools},
        {"role": "user", "content": "Compare Brazil and Canada."},
        {"role": "assistant", "content": "", "function_calls": (
            'predict_forest_area(location="Brazil", num_years=20, include_humans=true)\n'
            'api.v1_ownersearch(search_keyword=null, postcode="3000")'
        )},
        {"role": "environment", "content": '{"area": 10}'},
        {"role": "assistant", "content": "Brazil has more forest."},
    ]
    record = build_record("dolci_tool_use", tools, messages)
    assert record["messages"][0]["role"] == "user"
    calls = record["messages"][1]["tool_calls"]
    assert calls[0]["function"]["arguments"]["include_humans"] is True
    assert calls[1]["function"]["arguments"]["search_keyword"] is None
    assert record["messages"][2] == {"role": "tool", "content": '{"area": 10}'}


def test_invalid_escape_in_a_call_is_kept():
    tools = [{"name": "match", "description": "Match.", "parameters": {"type": "object", "properties": {}}}]
    messages = [
        {"role": "user", "content": "Find it."},
        {"role": "assistant", "content": "", "function_calls": r'match(pattern="\d+")'},
    ]
    record = build_record("dolci_tool_use", tools, messages)
    assert record["messages"][1]["tool_calls"][0]["function"]["arguments"]["pattern"] == r"\d+"


def test_non_call_payload_is_skipped_instead_of_crashing():
    tools = [{"name": "match", "description": "Match.", "parameters": {}}]
    messages = [
        {"role": "user", "content": "Find it."},
        {"role": "assistant", "content": "No.", "function_calls": "true"},
    ]
    try:
        build_record("dolci_tool_use", tools, messages)
    except Skip as exc:
        assert exc.reason == "bad_call"
    else:
        raise AssertionError("expected Skip")


def test_unparsed_call_is_skipped():
    tools = [{"name": "match", "description": "Match.", "parameters": {}}]
    messages = [
        {"role": "user", "content": "Find it."},
        {"role": "assistant", "content": "", "function_calls": "this is not a call"},
    ]
    try:
        build_record("dolci_tool_use", tools, messages)
    except Skip as exc:
        assert exc.reason == "bad_call"
    else:
        raise AssertionError("expected Skip")


def test_nemotron_string_arguments_and_think_in_content():
    tools = [{
        "type": "function",
        "function": {"name": "check_valid_vin", "description": "Check a VIN.", "parameters": {"type": "object"}},
    }]
    messages = [
        {"role": "user", "content": "Check this VIN."},
        {
            "role": "assistant",
            "content": "<think>\nCheck the VIN first.\n</think>",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "check_valid_vin", "arguments": '{"vin": "1FMXK92W8YPA12345"}'},
            }],
        },
        {"role": "tool", "content": '{"valid": true}'},
        {"role": "assistant", "content": "The VIN is valid."},
    ]
    record = build_record("nemotron_post_v1", tools, messages)
    call = record["messages"][1]
    assert call["reasoning_content"] == "Check the VIN first."
    assert call["content"] == ""
    assert call["tool_calls"][0]["function"]["arguments"] == {"vin": "1FMXK92W8YPA12345"}
    assert "<tool_call>" not in json.dumps(record)


def test_toolmind_double_wrapped_function_schema():
    tools = [{
        "type": "function",
        "function": {
            "type": "function",
            "function": {
                "name": "getscreen",
                "description": "Capture the screen.",
                "parameters": {"type": "dict", "properties": {"phone_id": {"type": "str"}}},
            },
        },
    }]
    messages = [
        {"role": "user", "content": "Take a screenshot."},
        {"role": "assistant", "content": "Done."},
    ]
    record = build_record("When2Call", tools, messages)
    assert record["tools"][0]["function"]["name"] == "getscreen"
