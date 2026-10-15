"""
validate_page_querying.py — Validate Phase 4/5 query parsing + citation behavior.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.page_querying import build_citation, extract_page_intent

OPTIONAL_TUI_VALIDATION_DEPS = {"httpx", "textual", "rich"}


def _validate_tui_followup_citation_wiring() -> bool:
    """Fresh retrieval keeps request book/page provenance through async display."""
    try:
        from src import tui
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".", 1)[0] in OPTIONAL_TUI_VALIDATION_DEPS:
            print(f"SKIP: TUI wiring integration ({exc.name})")
            return False
        raise

    class Client:
        async def astream_generate(self, prompt, **_kwargs):
            assert prompt == "FRESH_CONTEXT"
            yield {"response": "Follow-up answer."}
            yield {"done": True}

        def unload_and_wait(self):
            return

    async def probe():
        app = tui.AxiomLMApp()
        app.active_collection_name = "other_active_book"
        app._last_retrieved_pages = [91]
        app._last_chunks = ["STALE CACHED CONTEXT"]
        app._discover_books = dict
        app._fetch_ollama_models = lambda: [tui.OLLAMA_MODEL]
        app._prepare_query = lambda *_: (
            "FRESH_CONTEXT",
            [{"physical_page": 7}],
            [7],
            ["fresh"],
        )
        async with app.run_test():
            await app._submit_question(
                "Explain this further", "request_scoped_book", tui.OLLAMA_MODEL
            )
            while app._active_query_request_id is not None:
                await asyncio.sleep(0.01)
            record = app.turns[-1].record
            assert record.phase == "DONE"
            assert record.raw_answer == "Follow-up answer."
            assert record.book == "request_scoped_book"
            assert [
                (source.book, source.physical_page) for source in record.sources
            ] == [("request_scoped_book", 7)]
            assert app._last_retrieved_pages == [7]
            assert "other_active_book" not in record.display_answer

    previous = tui.OllamaClient
    tui.OllamaClient = Client
    try:
        asyncio.run(probe())
    finally:
        tui.OllamaClient = previous
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
        print(
            "PASS: page intent parsing + citation formatting | SKIP: TUI wiring integration"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
