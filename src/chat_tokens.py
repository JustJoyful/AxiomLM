"""Count native-chat prompt tokens for the validated Qwen3.5 Ollama renderer.

Ollama 0.40.4 serves the configured model with ``RENDERER qwen3.5`` (Go-side),
not a Modelfile template. This module reproduces that rendering for the message
shapes the agent sends and counts it with the GGUF tokenizer already validated
for answers. It is only trusted for the recorded model digest/runtime pair, and
only after ``scripts/probe_phase3_runtime.py`` matches served prompt_eval_count.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from src import config

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language "
    "BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with "
    "your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


def _value(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


# Ollama decodes tools into Go ``api.Tool`` structs before rendering, so only these
# keys survive, in struct-field order; other JSON Schema keywords are dropped.
_PARAMETER_KEYS = ("$defs", "items", "required")
_PROPERTY_KEYS = ("anyOf", "type", "items", "description", "enum", "properties")


def _go_property(field: dict) -> dict:
    out = {}
    for key in _PROPERTY_KEYS:
        value = field.get(key)
        if value in (None, "", []):
            continue
        if key == "properties":
            value = {name: _go_property(child) for name, child in value.items()}
        elif key == "anyOf":
            value = [_go_property(child) for child in value]
        out[key] = value
    return out


def served_tool(tool: dict) -> dict:
    """The tool definition as Ollama re-serializes it for the qwen3.5 renderer."""
    function = tool["function"]
    parameters = function.get("parameters") or {}
    rendered = {"type": parameters.get("type", "object")}
    rendered.update({k: parameters[k] for k in _PARAMETER_KEYS if parameters.get(k)})
    rendered["properties"] = {
        name: _go_property(field)
        for name, field in (parameters.get("properties") or {}).items()
    }
    body = {"name": function["name"]}
    if function.get("description"):
        body["description"] = function["description"]
    body["parameters"] = rendered
    return {"type": "function", "function": body}


def render_tools(tools: list[dict]) -> str:
    lines = "".join("\n" + _value(served_tool(tool)) for tool in tools)
    return (
        "# Tools\n\nYou have access to the following functions:\n\n<tools>"
        + lines
        + "\n</tools>"
        + TOOL_INSTRUCTIONS
    )


def _tool_call(call: dict) -> str:
    function = call["function"]
    out = "<tool_call>\n<function=" + function["name"] + ">\n"
    for key, value in (function.get("arguments") or {}).items():
        out += "<parameter=" + key + ">\n" + _value(value) + "\n</parameter>\n"
    return out + "</function>\n</tool_call>"


def render_chat(messages: list[dict], tools: list[dict] | None, *, think: bool) -> str:
    """Render messages plus the assistant generation prefix."""
    system = ""
    if messages and messages[0]["role"] == "system":
        system, messages = messages[0]["content"], messages[1:]
    out = ""
    if tools:
        out += IM_START + "system\n" + render_tools(tools)
        if system:
            out += "\n\n" + system.strip()
        out += IM_END + "\n"
    elif system:
        out += IM_START + "system\n" + system.strip() + IM_END + "\n"
    last_user = max(
        (i for i, m in enumerate(messages) if m["role"] == "user"), default=-1
    )
    for index, message in enumerate(messages):
        role, content = message["role"], (message.get("content") or "").strip()
        if role == "user":
            out += IM_START + "user\n" + content + IM_END + "\n"
        elif role == "assistant":
            out += IM_START + "assistant\n"
            reasoning = (message.get("thinking") or "").strip()
            if index > last_user and reasoning:
                out += "<think>\n" + reasoning + "\n</think>\n\n"
            out += content
            for number, call in enumerate(message.get("tool_calls") or []):
                prefix = (
                    "\n\n" if (number == 0 and content) else ("\n" if number else "")
                )
                out += prefix + _tool_call(call)
            out += IM_END + "\n"
        elif role == "tool":
            previous = messages[index - 1]["role"] if index else None
            following = (
                messages[index + 1]["role"] if index + 1 < len(messages) else None
            )
            if previous != "tool":
                out += IM_START + "user"
            out += "\n<tool_response>\n" + content + "\n</tool_response>"
            if following != "tool":
                out += IM_END + "\n"
        else:
            raise ValueError(f"Unsupported chat role: {role}")
    out += IM_START + "assistant\n"
    out += "<think>\n" if think else "<think>\n\n</think>\n\n"
    return out


class ChatTokenCounter:
    def __init__(self, encode: Callable[[str], int]):
        self._encode = encode

    def count(self, messages: list[dict], tools: list[dict] | None, *, think: bool):
        return self._encode(render_chat(messages, tools, think=think))


def load_chat_counter(model: str) -> ChatTokenCounter:
    """Reuse the digest/runtime-validated GGUF tokenizer; unvalidated pairs raise."""
    from src.answer_tokens import load_answer_counter

    if model != config.OLLAMA_MODEL:
        raise ValueError(
            "Chat token accounting is validated only for the configured model"
        )
    tokenizer = load_answer_counter(model).tokenizer
    return ChatTokenCounter(
        lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    )
