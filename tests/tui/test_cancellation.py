import asyncio
import json
import time

import httpx
import pytest

from src.llm import OllamaClient


async def test_silent_read_cancel_closes_transport_under_200ms():
    entered = asyncio.Event()
    closed = asyncio.Event()
    requests = []

    class SilentStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"response":"partial "}\n'
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, stream=SilentStream())

    client = OllamaClient(async_transport=httpx.MockTransport(respond))
    chunks = []

    async def consume():
        async for chunk in client.astream_generate("prompt"):
            chunks.append(chunk["response"])

    task = asyncio.create_task(consume())
    await entered.wait()
    start = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert time.monotonic() - start < 0.2
    assert chunks == ["partial "]
    assert requests[0]["keep_alive"] == 0 and requests[0]["think"] is False
    assert requests[0]["stream"] is True


async def test_ui_cancel_keeps_partial_and_admission_until_confirmed_unload(
    app_factory, monkeypatch
):
    from threading import Event

    from src import tui
    from src.ui.prompt import QueryComposer

    app = app_factory()
    app.active_collection_name = "book"
    entered = asyncio.Event()
    closed = asyncio.Event()
    cleanup_entered = Event()
    finish_cleanup = Event()

    class Client:
        async def astream_generate(self, *_args, **_kwargs):
            assert app.turns[-1].question.is_attached
            assert app.turns[-1].record.question == "Explain evaluation"
            yield {"response": "Partial \\sum x"}
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        def unload_and_wait(self):
            cleanup_entered.set()
            assert finish_cleanup.wait(3)

    monkeypatch.setattr(tui, "OllamaClient", Client)
    monkeypatch.setattr(app, "_prepare_query", lambda *_: ("prompt", [], [], []))
    try:
        async with app.run_test():
            prompt = app.query_one(QueryComposer)
            await app.on_query_composer_submitted(
                QueryComposer.Submitted(prompt, "Explain evaluation")
            )
            await asyncio.wait_for(entered.wait(), 1)
            start = time.monotonic()
            await app._cancel_active_query()
            assert time.monotonic() - start < 0.2
            assert closed.is_set()
            record = app.turns[-1].record
            assert record.phase == "CANCELLED"
            assert record.raw_answer == "Partial \\sum x"
            assert record.copy_text == "Partial \\sum x"
            assert record.display_answer == "Partial ∑ x"
            assert app._workload_coordinator.active_kind == "generation"
            assert app._try_acquire_workload("ingest") is None
            await app._stream_query_token(record.request_id, "STALE")
            assert "STALE" not in app.turns[-1].answer.source
            finish_cleanup.set()
            for _ in range(100):
                if app._workload_coordinator.active_kind is None:
                    break
                await asyncio.sleep(0.01)
            assert app._workload_coordinator.active_kind is None
    finally:
        finish_cleanup.set()


async def test_cancelled_retrieval_thread_finishes_before_admission_reopens(
    app_factory, monkeypatch
):
    from threading import Event

    from src import tui
    from src.ui.prompt import QueryComposer

    app = app_factory()
    app.active_collection_name = "book"
    entered = Event()
    finish = Event()

    def retrieve(*_):
        entered.set()
        assert finish.wait(3)
        return "prompt", [], [], []

    class Client:
        def unload_and_wait(self):
            assert finish.is_set()

    monkeypatch.setattr(app, "_prepare_query", retrieve)
    monkeypatch.setattr(tui, "OllamaClient", Client)
    try:
        async with app.run_test():
            await app.on_query_composer_submitted(
                QueryComposer.Submitted(app.query_one(QueryComposer), "question")
            )
            assert await asyncio.to_thread(entered.wait, 1)
            await app._cancel_active_query()
            assert app._workload_coordinator.active_kind == "generation"
            finish.set()
            for _ in range(100):
                if app._workload_coordinator.active_kind is None:
                    break
                await asyncio.sleep(0.01)
            assert app._workload_coordinator.active_kind is None
    finally:
        finish.set()


async def test_cancel_during_rewrite_closes_request_and_keeps_gpu_admission(
    app_factory, monkeypatch
):
    from threading import Event

    from src import tui
    from src.ui.prompt import QueryComposer
    from src.ui.stream import TurnRecord, TurnView

    app = app_factory()
    app.active_collection_name = "book"
    record = TurnRecord("prior", "Explain interpreters.", "book", "local", phase="DONE")
    record.set_answer("They execute programs.")
    app.turns.append(TurnView(record))
    app._conversation = [
        {"role": "user", "content": record.question},
        {"role": "assistant", "content": record.copy_text},
    ]
    cancelled = TurnRecord(
        "cancelled", "PRIVATE CANCELLED", "book", "local", phase="CANCELLED"
    )
    app.turns.append(TurnView(cancelled))
    entered = asyncio.Event()
    closed = asyncio.Event()
    finish = Event()

    class Client:
        async def agenerate(self, prompt, **kwargs):
            assert (
                "Explain interpreters." in prompt and "PRIVATE CANCELLED" not in prompt
            )
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        def unload_and_wait(self):
            assert finish.wait(3)

    monkeypatch.setattr(tui, "OllamaClient", Client)
    monkeypatch.setattr(
        app,
        "_prepare_query",
        lambda *_: pytest.fail("retrieval ran after cancelled rewrite"),
    )
    try:
        async with app.run_test():
            await app.on_query_composer_submitted(
                QueryComposer.Submitted(app.query_one(QueryComposer), "What is it?")
            )
            await asyncio.wait_for(entered.wait(), 1)
            await app._cancel_active_query()
            assert (
                closed.is_set()
                and app._workload_coordinator.active_kind == "generation"
            )
            assert app._try_acquire_workload("ingest") is None
            finish.set()
            for _ in range(100):
                if app._workload_coordinator.active_kind is None:
                    break
                await asyncio.sleep(0.01)
            assert app._workload_coordinator.active_kind is None
            assert all(t.get("content") != "What is it?" for t in app._conversation)
    finally:
        finish.set()
