"""Validated dispatch of known tools under immutable, host-owned request context."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from src import config

from . import local
from .protocol import (
    ToolContext,
    ToolResult,
    serialized_size,
    validate_call,
)

Handler = Callable[[dict, ToolContext, str], Awaitable[ToolResult]]

UNAVAILABLE = {
    "generate_pdf": "PDF generation is unavailable until Phase 5 (Typst documents).",
    "verify_citation": "Citation verification is unavailable until Phase 4 (entailment scorer).",
}

PYTHON_STATUS = {
    "ok": "ok",
    "error": "error",
    "timeout": "error",
    "cancelled": "error",
    "worker_died": "error",
    "unavailable": "unavailable",
}


class ToolRegistry:
    def __init__(self, *, sandbox=None, search=None, web=None):
        self.sandbox = sandbox
        self._search = search
        self._web = web
        self.calls = 0

    async def _search_local(self, arguments, context, call_id):
        kwargs = {"search": self._search} if self._search else {}
        return await local.search_local_index(
            arguments, context, call_id=call_id, **kwargs
        )

    async def _python(self, arguments, context, call_id):
        if self.sandbox is None:
            return ToolResult(
                call_id,
                "execute_python",
                "unavailable",
                {"error": "Python sandbox is not configured"},
            )
        result = await self.sandbox.execute(arguments["code"])
        return ToolResult(
            call_id,
            "execute_python",
            PYTHON_STATUS[result.status],
            {
                "sandbox_status": result.status,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "result": result.value,
                "truncated": result.truncated,
            },
        )

    async def _web_tool(self, name, arguments, context, call_id):
        if not context.web_enabled:
            return ToolResult(
                call_id, name, "denied", {"error": "Web access is off (/web on)"}
            )
        if self._web is None:
            return ToolResult(
                call_id, name, "unavailable", {"error": "Web tools are not configured"}
            )
        handler = self._web.search if name == "web_search" else self._web.fetch
        value = arguments["query"] if name == "web_search" else arguments["url"]
        return await handler(value, context, call_id=call_id)

    async def dispatch(
        self, call_id: str, name: object, arguments: object, context: ToolContext
    ) -> ToolResult:
        tool = name if isinstance(name, str) else repr(name)
        try:
            arguments = validate_call(name, arguments)
        except ValueError as exc:
            return ToolResult(call_id, tool, "error", {"error": str(exc)})
        if name not in context.allowed_tools:
            return ToolResult(
                call_id,
                name,
                "denied",
                {"error": f"{name} is not permitted at stage {context.stage}"},
            )
        self.calls += 1
        if name in UNAVAILABLE:
            return ToolResult(
                call_id, name, "unavailable", {"error": UNAVAILABLE[name]}
            )
        if name == "search_local_index":
            result = await self._search_local(arguments, context, call_id)
        elif name == "execute_python":
            result = await self._python(arguments, context, call_id)
        else:
            result = await self._web_tool(name, arguments, context, call_id)
        if serialized_size(result) > config.TOOL_MAX_RESULT_BYTES:
            return ToolResult(
                call_id,
                name,
                "truncated",
                {"error": "Tool result exceeded the size limit and was withheld"},
            )
        return result
