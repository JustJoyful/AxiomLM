"""Typed tool contracts and pre-dispatch validation.

Book, model, filesystem roots and budgets live in host-owned ``ToolContext``;
no schema lets the model choose them. Validation happens before any handler runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import jsonschema

from src import config

ToolName = Literal[
    "search_local_index",
    "web_search",
    "fetch_page",
    "execute_python",
    "generate_pdf",
    "verify_citation",
]
TOOL_NAMES: tuple[str, ...] = (
    "search_local_index",
    "web_search",
    "fetch_page",
    "execute_python",
    "generate_pdf",
    "verify_citation",
)


@dataclass(frozen=True)
class EvidenceRef:
    source_id: str
    kind: Literal["book", "web"]
    text: str
    book_stem: str | None = None
    physical_pages: tuple[int, ...] = ()
    chapter: str | None = None
    section: str | None = None
    url: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class ToolContext:
    request_id: str
    book_stem: str
    model: str
    web_enabled: bool
    stage: str
    top_k: int = config.TOP_K_RESULTS
    distance: float = config.DISTANCE_THRESHOLD
    allowed_tools: frozenset[str] = frozenset(TOOL_NAMES)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    status: Literal["ok", "error", "unavailable", "denied", "truncated"]
    data: dict = field(default_factory=dict)
    sources: tuple[EvidenceRef, ...] = ()


def _text(limit: int, description: str) -> dict:
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": limit,
        "description": description,
    }


SCHEMAS: dict[str, dict] = {
    "search_local_index": {
        "type": "object",
        "properties": {
            "query": _text(
                config.TOOL_QUERY_MAX_CHARS, "What to find in the selected book"
            ),
            "page_start": {
                "type": "integer",
                "minimum": 1,
                "description": "First physical PDF page (with page_end)",
            },
            "page_end": {
                "type": "integer",
                "minimum": 1,
                "description": "Last physical PDF page (with page_start)",
            },
        },
        "required": ["query"],
        "dependentRequired": {"page_start": ["page_end"], "page_end": ["page_start"]},
        "additionalProperties": False,
    },
    "web_search": {
        "type": "object",
        "properties": {"query": _text(config.TOOL_QUERY_MAX_CHARS, "Web search query")},
        "required": ["query"],
        "additionalProperties": False,
    },
    "fetch_page": {
        "type": "object",
        "properties": {"url": _text(2048, "Public http(s) URL from a search result")},
        "required": ["url"],
        "additionalProperties": False,
    },
    "execute_python": {
        "type": "object",
        "properties": {
            "code": _text(
                config.PYTHON_MAX_CODE_BYTES,
                "Python with sympy imported; assign the answer to `result`",
            )
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    "generate_pdf": {
        "type": "object",
        "properties": {
            "request": _text(config.TOOL_QUERY_MAX_CHARS, "Document request")
        },
        "required": ["request"],
        "additionalProperties": False,
    },
    "verify_citation": {
        "type": "object",
        "properties": {
            "claim": _text(config.TOOL_QUERY_MAX_CHARS, "Claim to check"),
            "source_id": _text(256, "Evidence source ID"),
        },
        "required": ["claim", "source_id"],
        "additionalProperties": False,
    },
}

DESCRIPTIONS = {
    "search_local_index": "Search the selected book for passages with chapter/page provenance.",
    "web_search": "Search the web for public sources (only when book evidence is insufficient).",
    "fetch_page": "Fetch readable text from one public web page returned by web_search.",
    "execute_python": "Compute with SymPy in an isolated sandbox; assign the answer to `result`.",
    "generate_pdf": "Generate a PDF document (unavailable until Phase 5).",
    "verify_citation": "Check a claim against one evidence source (unavailable until Phase 4).",
}

_VALIDATORS = {
    name: jsonschema.Draft202012Validator(schema) for name, schema in SCHEMAS.items()
}
for _schema in SCHEMAS.values():
    jsonschema.Draft202012Validator.check_schema(_schema)


def validate_call(name: object, arguments: object) -> dict:
    """Return validated arguments or raise ValueError; never reaches a handler."""
    if not isinstance(name, str) or name not in SCHEMAS:
        raise ValueError(f"Unknown tool: {name!r}")
    if isinstance(arguments, str):
        # Some runtimes deliver arguments as JSON text; decode strictly.
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("Tool arguments are not valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")  # noqa: TRY004
    try:
        size = len(json.dumps(arguments, ensure_ascii=False).encode())
    except (TypeError, ValueError) as exc:
        raise ValueError("Tool arguments are not JSON-serializable") from exc
    if size > config.TOOL_MAX_ARGUMENT_BYTES:
        raise ValueError("Tool arguments exceed the size limit")
    errors = sorted(_VALIDATORS[name].iter_errors(arguments), key=lambda e: e.path)
    if errors:
        raise ValueError(f"Invalid {name} arguments: {errors[0].message}")
    if (
        name == "search_local_index"
        and "page_start" in arguments
        and arguments["page_start"] > arguments["page_end"]
    ):
        raise ValueError("page_start must not exceed page_end")
    if name == "execute_python" and (
        len(arguments["code"].encode()) > config.PYTHON_MAX_CODE_BYTES
    ):
        raise ValueError("Code exceeds the sandbox size limit")
    return arguments


def definitions(names) -> list[dict]:
    """Ollama tool definitions for exactly the given (state-legal) tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": DESCRIPTIONS[name],
                "parameters": SCHEMAS[name],
            },
        }
        for name in names
    ]


def serialized_size(result: ToolResult) -> int:
    return len(tool_message(result).encode())


def tool_message(result: ToolResult) -> str:
    """The tool-role message content: status plus data, quoted as evidence."""
    return json.dumps(
        {"tool": result.name, "status": result.status, **result.data},
        ensure_ascii=False,
        separators=(",", ":"),
    )
