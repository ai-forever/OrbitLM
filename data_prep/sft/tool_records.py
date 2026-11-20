"""Bring tool-calling dialogs to one stored shape.

The chat template renders this shape. A tool schema is an OpenAI function
object. An assistant call stores arguments as an object, not as a Python
call and not as a quoted JSON string. A tool result is raw text, without a
<tool_response> wrapper. A reasoning chain lives in reasoning_content, and
the assistant content is only the visible answer.

Dolci's system text tells the model to use <functions> and <function_calls>.
That contradicts the Qwen3 tool header, so that system turn is dropped.
"""
import ast
import json
import re

_JSON_WORDS = {"true": True, "false": False, "null": None}
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_BAD_ESCAPE = re.compile(r"\\(?![\\'\"abfnrtv0-7xuUN])")


class Skip(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def parse_tool_schemas(raw):
    """Return a list of {type, function: {name, description, parameters}}."""
    items = _schema_items(raw)
    tools = []
    for item in items:
        if isinstance(item, str):
            item = _loads_schema_text(item)
            if isinstance(item, list):
                tools.extend(normalize_tool(piece) for piece in item)
                continue
        tools.append(normalize_tool(item))
    if not tools:
        raise Skip("no_tools")
    return tools


def normalize_tool(item):
    if not isinstance(item, dict):
        raise Skip("bad_tools")
    function = item.get("function") if isinstance(item.get("function"), dict) else item
    # ToolMind sometimes wraps the function object twice.
    if not isinstance(function.get("name"), str):
        nested = function.get("function")
        if isinstance(nested, dict):
            function = nested
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise Skip("bad_tools")
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    description = function.get("description") or ""
    if not isinstance(description, str):
        description = str(description)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def build_record(source, tools, messages):
    """Normalize one dialog. Raises Skip when the row cannot be stored."""
    preamble, tools = _preamble_and_schema(tools)
    schemas = parse_tool_schemas(tools)
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        raise Skip("bad_messages")
    turns = []
    for message in messages:
        turns.extend(convert_message(message))
    if preamble:
        turns.insert(0, {"role": "system", "content": preamble})
    if turns and turns[0]["role"] == "system" and _is_foreign_protocol(turns[0]["content"]):
        turns = turns[1:]
    turns = [turn for turn in turns if not (turn["role"] == "system" and not turn["content"].strip())]
    if not any(turn["role"] == "user" for turn in turns):
        raise Skip("empty_dialog")
    if not any(turn["role"] == "assistant" for turn in turns):
        raise Skip("empty_dialog")
    return {"source": source, "tools": schemas, "messages": turns}


def convert_message(message):
    if not isinstance(message, dict):
        raise Skip("bad_messages")
    role = message.get("role")
    if role == "environment":
        role = "tool"
    if role not in {"system", "user", "assistant", "tool"}:
        raise Skip("bad_role")
    content = message.get("content")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = str(content)
    if role == "tool":
        return [{"role": "tool", "content": part} for part in _split_tool_responses(content)]
    if role != "assistant":
        return [{"role": role, "content": content}]

    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = ""
    extracted, content = _split_think(content)
    if not reasoning.strip():
        reasoning = extracted
    structured = message.get("tool_calls")
    if structured is None:
        structured = message.get("function_calls")
    if _present(structured):
        calls = _parse_structured_calls(structured)
        content = _TOOL_CALL.sub("", content).strip()
    else:
        calls, content = _extract_tagged_calls(content)
    turn = {"role": "assistant", "content": content.strip()}
    if reasoning.strip():
        turn["reasoning_content"] = reasoning.strip()
    if calls:
        turn["tool_calls"] = [
            {"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
            for call in calls
        ]
    if not turn["content"] and "tool_calls" not in turn:
        raise Skip("empty_assistant")
    return [turn]


def _is_foreign_protocol(text):
    return "<function_calls>" in text and "<functions>" in text


def _present(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "null", "none"}
    if isinstance(value, list):
        return len(value) > 0
    return True


def _split_think(content):
    parts = [part.strip() for part in _THINK.findall(content) if part.strip()]
    return "\n\n".join(parts), _THINK.sub("", content).strip()


def _split_tool_responses(content):
    parts = _TOOL_RESPONSE.findall(content)
    if parts:
        return [part.strip() for part in parts]
    return [content.strip()]


def _extract_tagged_calls(content):
    calls = []

    def take(match):
        calls.extend(_parse_call_text(match.group(1)))
        return ""

    content = _TOOL_CALL.sub(take, content).strip()
    return calls, content


def _parse_structured_calls(value):
    if isinstance(value, str):
        return _parse_call_text(value)
    if isinstance(value, dict):
        return _payload_to_calls(value)
    if isinstance(value, list):
        calls = []
        for item in value:
            if isinstance(item, str):
                calls.extend(_parse_call_text(item))
            else:
                calls.extend(_payload_to_calls(item))
        return calls
    raise Skip("bad_call")


def _parse_call_text(text):
    text = text.strip()
    while text.startswith("\\n"):
        text = text[2:].lstrip()
    while text.endswith("\\n"):
        text = text[:-2].rstrip()
    if text.lower() in {"", "null", "none"}:
        return []
    for loader in (_json_value, _python_value):
        try:
            value = loader(text)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        try:
            return _payload_to_calls(value)
        except ValueError:
            continue
    try:
        tree = _parse_module(text)
    except SyntaxError as exc:
        raise Skip("bad_call") from exc
    calls = []
    if not tree.body:
        raise Skip("bad_call")
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise Skip("bad_call")
        call = statement.value
        if call.args:
            raise Skip("bad_call")
        name = _call_name(call.func)
        if not name:
            raise Skip("bad_call")
        arguments = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                raise Skip("bad_call")
            try:
                arguments[keyword.arg] = _eval(keyword.value)
            except ValueError as exc:
                raise Skip("bad_call") from exc
        calls.append(_call(name, arguments))
    return calls


def _payload_to_calls(value):
    if isinstance(value, list):
        calls = []
        for item in value:
            calls.extend(_payload_to_calls(item))
        return calls
    if not isinstance(value, dict):
        raise ValueError("not a tool call")
    function = value.get("function")
    if isinstance(function, dict) and "name" in function and "name" not in value:
        return _payload_to_calls(function)
    arguments = value.get("arguments")
    if "name" not in value and isinstance(arguments, dict) and isinstance(arguments.get("name"), str):
        arguments = dict(arguments)
        value = {"name": arguments.pop("name"), "arguments": arguments}
    if "name" not in value:
        raise ValueError("not a tool call")
    name = value["name"]
    if not isinstance(name, str) or not name.strip():
        raise Skip("bad_call")
    arguments = value.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise Skip("bad_call") from exc
    if not isinstance(arguments, dict):
        raise Skip("bad_call")
    return [_call(name, arguments)]


def _call(name, arguments):
    try:
        json.dumps(arguments)
    except TypeError as exc:
        raise Skip("bad_call") from exc
    return {"name": name, "arguments": arguments}


def _json_value(text):
    return json.loads(text)


def _python_value(text):
    return _eval(ast.parse(text, mode="eval").body)


def _parse_module(text):
    try:
        return ast.parse(text)
    except SyntaxError:
        return ast.parse(_BAD_ESCAPE.sub(r"\\\\", text))


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        if parent is None:
            return None
        return parent + "." + node.attr
    return None


def _eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in _JSON_WORDS:
        return _JSON_WORDS[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        return -value
    if isinstance(node, ast.List):
        return [_eval(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_eval(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_eval(key): _eval(value) for key, value in zip(node.keys, node.values)}
    raise ValueError(type(node).__name__)


def _preamble_and_schema(raw):
    """Split an instruction that has a <tools> block in the middle.

    Hermes sometimes stores the task text, the schema, and a call-format
    reminder in one string. The reminder describes Hermes' own dict layout,
    so it is dropped. The task text becomes a system turn.
    """
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        raw = raw[0]
    if not isinstance(raw, str):
        return "", raw
    end = raw.rfind("</tools>")
    if end != -1:
        start = raw.rfind("<tools>", 0, end)
        if start != -1:
            return raw[:start].strip(), raw[start:end + len("</tools>")]
    for marker in ('[{"type":', "{'type':", '{"type":'):
        start = raw.find(marker)
        if start != -1:
            return raw[:start].strip(), raw[start:]
    return "", raw


def _schema_items(raw):
    if raw is None:
        raise Skip("no_tools")
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], str):
            return _schema_items(raw[0])
        return raw
    if isinstance(raw, dict):
        if "tools" in raw and isinstance(raw["tools"], list):
            return raw["tools"]
        return [raw]
    if isinstance(raw, str):
        return _loads_schema_text(raw)
    raise Skip("bad_tools")


def _loads_schema_text(text):
    body = text.strip()
    if body.startswith("<tools>") and body.endswith("</tools>"):
        body = body[len("<tools>"):-len("</tools>")].strip()
    if not body:
        raise Skip("no_tools")
    try:
        value = json.loads(body)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        pass
    try:
        value = ast.literal_eval(body)
        return value if isinstance(value, list) else [value]
    except (SyntaxError, ValueError):
        pass
    decoder = json.JSONDecoder()
    index = 0
    while index < len(body) and body[index].isspace():
        index += 1
    try:
        value, end = decoder.raw_decode(body, index)
    except json.JSONDecodeError:
        value = None
    else:
        rest = body[end:].strip()
        if isinstance(value, (dict, list)) and (not rest or rest.startswith("For each") or rest.startswith("<tool_call>")):
            return value if isinstance(value, list) else [value]
    decoder = json.JSONDecoder()
    items = []
    index = 0
    length = len(body)
    while index < length:
        while index < length and body[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            value, end = decoder.raw_decode(body, index)
        except json.JSONDecodeError as exc:
            raise Skip("bad_tools") from exc
        items.append(value)
        index = end
    if not items:
        raise Skip("bad_tools")
    return items
