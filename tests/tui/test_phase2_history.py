import asyncio

import pytest


@pytest.mark.parametrize("command", ["clear", "new", "book_roundtrip", "ingest"])
async def test_rewrite_history_tracks_conversation_lifetime(
    app_factory, monkeypatch, command
):
    from src import tui
    from src.ui.stream import TurnRecord, TurnView

    app = app_factory()
    captured = []
    entered = asyncio.Event()

    async def capture(question, exchanges, book, **kwargs):
        captured.extend(exchanges)
        entered.set()
        raise RuntimeError("stop after capturing history")

    class Client:
        def unload_and_wait(self):
            pass

    monkeypatch.setattr(tui, "rewrite_followup", capture)
    monkeypatch.setattr(tui, "OllamaClient", Client)
    async with app.run_test():
        app.active_collection_name = "book"
        app._book_counts = {"book": 1, "other": 1}
        record = TurnRecord(
            "prior", "Explain interpreters.", "book", "local", phase="DONE"
        )
        if command != "ingest":
            record.set_answer("They execute programs.")
        app.turns.append(TurnView(record))
        app._conversation = (
            []
            if command == "ingest"
            else [
                {"role": "user", "content": record.question},
                {"role": "assistant", "content": record.copy_text},
            ]
        )
        if command == "book_roundtrip":
            await app.handle_command("/book", ["other"])
            await app.handle_command("/book", ["book"])
        elif command != "ingest":
            await app.handle_command("/" + command, [])
        await app._submit_question("What is it?", "book", "local")
        await asyncio.wait_for(entered.wait(), 1)
        expected = (
            [("Explain interpreters.", "They execute programs.", "book")]
            if command == "clear"
            else []
        )
        assert [(x.question, x.answer, x.book_stem) for x in captured] == expected


async def test_live_page_fallback_is_visible_after_real_preparation(
    app_factory, monkeypatch
):
    from types import SimpleNamespace

    from src import answer_tokens, hybrid_retrieval, index_store, tui

    app = app_factory()
    calls = []
    candidate = hybrid_retrieval.MergedCandidate(
        "fallback",
        "Evidence from another page.",
        {"book_stem": "book", "physical_page": 7, "physical_pages": "[7]"},
        1,
        None,
        0.1,
        None,
    )

    def retrieve(index, query, page_filter, *args, **kwargs):
        calls.append(page_filter)
        return [] if page_filter else [candidate]

    class Client:
        async def astream_generate(self, *args, **kwargs):
            yield {"response": "The retrieved evidence is on page 7.", "done": True}

        def unload_and_wait(self):
            pass

    monkeypatch.setattr(tui, "OllamaClient", Client)
    monkeypatch.setattr(
        index_store,
        "resolve_index",
        lambda book: SimpleNamespace(legacy=False, version_id="stage"),
    )
    monkeypatch.setattr(hybrid_retrieval, "retrieve_hybrid", retrieve)
    monkeypatch.setattr(
        answer_tokens,
        "load_answer_counter",
        lambda model: lambda prompt: len(prompt) // 4,
    )
    async with app.run_test():
        app.active_collection_name = "book"
        await app._submit_question("Explain page 999.", "book", "local")
        for _ in range(100):
            if app._active_query_request_id is None:
                break
            await asyncio.sleep(0.01)
        record = app.turns[-1].record
        assert record.phase == "DONE"
        assert calls == [(999, 999), None]
        assert any(
            "requested pages" in step and "fallback" in step for step in record.steps
        )
