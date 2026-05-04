#!/usr/bin/env python3
"""
validate_page_querying.py — Validate Phase 4/5 query parsing + citation behavior.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.page_querying import build_citation, extract_page_intent

OPTIONAL_TUI_VALIDATION_DEPS = {"httpx", "textual", "rich"}


def _validate_tui_followup_citation_wiring() -> bool:
    """Integration path: follow-up cache reuse + request-scoped citation append."""
    try:
        from src.tui import AxiomLMApp
        import src.tui as tui_module
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        missing_root = missing.split(".", 1)[0]
        if missing_root in OPTIONAL_TUI_VALIDATION_DEPS:
            print(f"SKIP: TUI wiring integration (missing optional dependency: {missing_root})")
            return False
        raise

    class _FakeStreamResponse:
        def __enter__(self) -> "_FakeStreamResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        def raise_for_status(self) -> None:
            return

        def iter_lines(self):
            yield json.dumps({"response": "Follow-up answer."})
            yield json.dumps(
                {
                    "done": True,
                    "eval_count": 8,
                    "eval_duration": 1_000_000_000,
                    "total_duration": 2_000_000_000,
                }
            )

    class _FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

        def stream(self, method: str, url: str, json: dict[str, object]) -> _FakeStreamResponse:  # noqa: ARG002
            return _FakeStreamResponse()

    app = AxiomLMApp()
    request_id = "followup-wire-test"
    request_book = "request_scoped_book"
    app.active_collection_name = "active_sidebar_book"
    app._request_books[request_id] = request_book
    app._last_chunks = ["[Intro · Basics · p.91]\nCached context block"]
    app._last_retrieved_pages = [91]

    captured: dict[str, object] = {}

    async def _capture_finalize_assistant_message(
        req_id: str,
        message: str,
        is_error: bool,
        telemetry_badge: str | None = None,
    ) -> None:
        captured["request_id"] = req_id
        captured["message"] = message
        captured["is_error"] = is_error
        captured["telemetry_badge"] = telemetry_badge

    def _call_immediately(fn, *args, **kwargs):  # noqa: ANN001
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        return result

    app._finalize_assistant_message = _capture_finalize_assistant_message  # type: ignore[method-assign]
    app.call_from_thread = _call_immediately  # type: ignore[method-assign]

    original_client = tui_module.Client
    tui_module.Client = _FakeClient
    try:
        app.execute_rag_query(
            query="Explain this further",
            history_str="",
            book_stem=request_book,
            model_name="fake-model",
            request_id=request_id,
        )
    finally:
        tui_module.Client = original_client

    assert app._last_retrieved_pages == []
    assert captured.get("request_id") == request_id
    assert captured.get("is_error") is False
    assert isinstance(captured.get("telemetry_badge"), str)

    final_message = captured.get("message")
    assert isinstance(final_message, str)
    assert final_message.endswith(f"[{request_book}]")
    assert f"[{request_book} · p." not in final_message
    assert "[active_sidebar_book" not in final_message
    return True


def main() -> int:
    # Phase 4: page intent extraction
    assert extract_page_intent("Explain page 75") == [75]
    assert extract_page_intent("What's on p.23?") == [23]
    assert extract_page_intent("Read pg90 now") == [90]
    assert extract_page_intent("pages 75-80 summary") == [75, 76, 77, 78, 79, 80]
    assert extract_page_intent("pages 80-75 summary") is None
    assert extract_page_intent("Regular question") is None

    # Phase 5: citation formatting from retrieved pages
    assert build_citation("thinkpython2", [75]) == "[thinkpython2 · p.75]"

    assert build_citation("mathsiia", [23, 24, 25]) == "[mathsiia · p.23-25]"

    assert build_citation("book", [2, 4, 5, 7]) == "[book · p.2,4-5,7]"

    assert build_citation("engineeringmath", []) == "[engineeringmath]"

    tui_wiring_ran = _validate_tui_followup_citation_wiring()
    if tui_wiring_ran:
        print("PASS: page intent parsing + citation formatting + TUI wiring")
    else:
        print("PASS: page intent parsing + citation formatting | SKIP: TUI wiring integration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
