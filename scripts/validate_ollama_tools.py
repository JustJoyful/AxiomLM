"""Run and record a real JSON-schema tool round-trip against local Ollama."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from src import config
from src.llm import OllamaClient

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}


def _runtime_identity() -> dict[str, Any]:
    with httpx.Client(timeout=10, trust_env=False) as client:
        version_response = client.get(f"{config.OLLAMA_URL}/api/version")
        version_response.raise_for_status()
        tags_response = client.get(f"{config.OLLAMA_URL}/api/tags")
        tags_response.raise_for_status()
    model = next(
        (
            row
            for row in tags_response.json().get("models", [])
            if row.get("name") == config.OLLAMA_MODEL
        ),
        None,
    )
    if model is None:
        raise RuntimeError(
            f"Configured model {config.OLLAMA_MODEL!r} is not installed."
        )
    return {
        "ollama_version": version_response.json().get("version"),
        "model": model.get("name"),
        "digest": model.get("digest"),
        "details": model.get("details", {}),
    }


def _tool_arguments(value: Any) -> dict[str, int]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or set(value) != {"a", "b"}:
        raise ValueError("Tool arguments must contain exactly integer fields a and b.")
    if any(type(value[key]) is not int for key in ("a", "b")):
        raise ValueError("Tool arguments a and b must be integers.")
    return {"a": value["a"], "b": value["b"]}


def run_diagnostic() -> dict[str, Any]:
    client = OllamaClient()
    report: dict[str, Any] = {
        "model": config.OLLAMA_MODEL,
        "tool_schema": ADD_TOOL,
        "passed": False,
        "steps": [],
    }
    try:
        report.update(_runtime_identity())
        messages = [
            {"role": "system", "content": "Use the add tool to calculate the answer."},
            {"role": "user", "content": "What is 2 + 3? Use the tool."},
        ]
        first = client.chat(messages, tools=[ADD_TOOL], think=False)
        assistant = first.get("message")
        if not isinstance(assistant, dict):
            raise ValueError(
                "Ollama's first response did not contain a message object."
            )
        tool_calls = assistant.get("tool_calls") or []
        if len(tool_calls) != 1:
            raise ValueError(f"Expected exactly one tool call; got {len(tool_calls)}.")
        call = tool_calls[0].get("function", {})
        if call.get("name") != "add":
            raise ValueError(f"Unexpected tool name: {call.get('name')!r}.")
        arguments = _tool_arguments(call.get("arguments"))
        result = arguments["a"] + arguments["b"]
        report["steps"].append(
            {
                "request": messages,
                "tool_call": call,
                "validated_arguments": arguments,
                "tool_result": result,
            }
        )

        final_messages = messages + [
            assistant,
            {"role": "tool", "tool_name": "add", "content": str(result)},
        ]
        final = client.chat(final_messages, think=False)
        final_message = final.get("message", {})
        content = str(final_message.get("content", "")).strip()
        if not re.search(r"(?<!\d)5(?!\d)", content):
            raise ValueError(
                f"Final response did not confirm the calculated value 5: {content!r}."
            )
        report["steps"].append(
            {
                "request": final_messages,
                "final_message": final_message,
                "final_content": content,
            }
        )
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            client.unload_and_wait()
            report["unload"] = "all Ollama models unloaded and confirmed"
        except Exception as exc:
            report["unload"] = f"failed: {type(exc).__name__}: {exc}"
            report["passed"] = False
    return report


def main() -> int:
    report = run_diagnostic()
    config.TOOLS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.TOOLS_REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
