import asyncio

import pytest

from src.agent import AgentEvent
from src.tools.protocol import EvidenceRef

BOOK_REF = EvidenceRef("c1", "book", "text", "sicp", (225, 226), "Ch 2", "2.3.2")
WEB_REF = EvidenceRef("web:1", "web", "page", url="https://docs.org/a", title="Docs")


class Client:
    def __init__(self):
        self.unloads = 0

    async def astream_generate(self, *args, **kwargs):
        raise AssertionError("agent turns must not use the toolless generate path")
        yield  # pragma: no cover

    def unload_and_wait(self):
        self.unloads += 1


def agent_app(app_factory, monkeypatch, events, client):
    from src import config, tui

    monkeypatch.setattr(config, "AGENT_ROUTING_ENABLED", True)
    app = app_factory()
    app._discover_books = lambda: {"sicp": 1}
    app.active_collection_name = "sicp"
    monkeypatch.setattr(tui, "OllamaClient", lambda: client)

    async def counter(model, record):
        return object()

    monkeypatch.setattr(app, "_agent_counter", counter)
    monkeypatch.setattr(app, "_agent_events", lambda *args: events())
    return app


async def wait_idle(app):
    for _ in range(300):
        if app._active_query_request_id is None:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("query never finished")


async def test_agent_turn_renders_mixed_sources_steps_and_answer(
    app_factory, monkeypatch
):
    client = Client()

    async def events():
        yield AgentEvent("step", {"stage": "local_search"})
        yield AgentEvent("sources", {"sources": [BOOK_REF]})
        yield AgentEvent(
            "step", {"stage": "local_search_done", "status": "ok", "passages": 1}
        )
        yield AgentEvent(
            "step", {"stage": "verify_done", "status": "verified", "reason": "trusted"}
        )
        yield AgentEvent("sources", {"sources": [BOOK_REF, WEB_REF]})
        yield AgentEvent("token", {"text": "The derivative is 3x^2 + 2."})
        yield AgentEvent(
            "result",
            {
                "complete": True,
                "verification": {"status": "verified", "reason": "t"},
                "model_requests": 5,
                "tool_calls": 3,
            },
        )

    app = agent_app(app_factory, monkeypatch, events, client)
    async with app.run_test():
        await app.workers.wait_for_complete()
        await app._submit_question("Differentiate x^3 + 2x", "sicp", app.selected_model)
        await wait_idle(app)
        record = app.turns[-1].record
        assert record.phase == "DONE" and "3x" in record.copy_text
        labels = [s.label() for s in record.sources]
        assert "web: [Docs](https://docs.org/a)" in labels
        assert any("p. 225–226" in l or "p. 225-226" in l for l in labels)
        assert any(step.startswith("↳ verify verified") for step in record.steps)
        assert client.unloads == 1 and app._workload_coordinator.active_kind is None
        assert app._conversation[-2]["content"] == "Differentiate x^3 + 2x"


async def test_cancel_during_sandbox_stage_closes_agent_and_keeps_history_clean(
    app_factory, monkeypatch
):
    client = Client()
    entered, closed = asyncio.Event(), asyncio.Event()

    async def events():
        try:
            yield AgentEvent("step", {"stage": "compute"})
            entered.set()
            await asyncio.Event().wait()  # a long sandbox call
        finally:
            closed.set()

    app = agent_app(app_factory, monkeypatch, events, client)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await app._submit_question("Solve it", "sicp", app.selected_model)
        await asyncio.wait_for(entered.wait(), 2)
        assert app._workload_coordinator.active_kind == "generation"
        assert app._try_acquire_workload("ingest") is None
        await pilot.press("escape")
        await asyncio.wait_for(closed.wait(), 2)
        await wait_idle(app)
        assert app.turns[-1].record.phase == "CANCELLED"
        assert client.unloads == 1 and app._workload_coordinator.active_kind is None
        assert app._conversation == []


async def test_incomplete_agent_answer_is_a_failed_completion(app_factory, monkeypatch):
    async def events():
        yield AgentEvent("token", {"text": "partial"})
        yield AgentEvent(
            "result",
            {
                "complete": False,
                "verification": None,
                "model_requests": 3,
                "tool_calls": 1,
            },
        )

    app = agent_app(app_factory, monkeypatch, events, Client())
    async with app.run_test():
        await app.workers.wait_for_complete()
        await app._submit_question("q", "sicp", app.selected_model)
        await wait_idle(app)
        assert app.turns[-1].record.phase != "DONE" and app._conversation == []


async def test_agent_limit_error_is_explicit(app_factory, monkeypatch):
    async def events():
        yield AgentEvent("error", {"kind": "limit", "message": "Tool call cap reached"})

    app = agent_app(app_factory, monkeypatch, events, Client())
    async with app.run_test():
        await app.workers.wait_for_complete()
        await app._submit_question("q", "sicp", app.selected_model)
        await wait_idle(app)
        view = app.turns[-1]
        assert "Tool call cap reached" in str(view.step_line.render())
        assert app._conversation == []


async def test_unvalidated_model_falls_back_to_phase2_path(app_factory, monkeypatch):
    from src import config, tui

    monkeypatch.setattr(config, "AGENT_ROUTING_ENABLED", True)
    app = app_factory()
    used = []

    class Legacy(Client):
        async def astream_generate(self, *args, **kwargs):
            used.append("generate")
            yield {"response": "ok", "done": True}

    monkeypatch.setattr(tui, "OllamaClient", Legacy)
    monkeypatch.setattr(app, "_prepare_query", lambda *_: ("prompt", [], [], []))
    from src import chat_tokens

    monkeypatch.setattr(
        chat_tokens,
        "load_chat_counter",
        lambda model: (_ for _ in ()).throw(ValueError("unvalidated")),
    )
    async with app.run_test():
        await app.workers.wait_for_complete()
        await app._submit_question("q", "sicp", "qwen3:4b")
        await wait_idle(app)
        assert used == ["generate"] and app.turns[-1].record.phase == "DONE"
        assert any("Phase 2 path" in step for step in app.turns[-1].record.steps)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/web show", "web evidence off"),
        ("/web on", "web evidence on"),
        ("/thinking show", "on for the solve computation step, off for Q&A"),
        ("/thinking on", "not changed"),
    ],
)
async def test_web_and_thinking_commands(app_factory, command, expected):
    from src.ui.commands import dispatch_command

    app = app_factory()
    lines = []

    async def write(text):
        lines.append(text)

    app.write_line = write
    async with app.run_test():
        await dispatch_command(app, command)
    assert expected in lines[-1]
    if command == "/web on":
        assert app.web_enabled is True


async def test_web_permission_is_locked_while_busy(app_factory):
    from src.ui.commands import dispatch_command

    app = app_factory()
    lines = []

    async def write(text):
        lines.append(text)

    app.write_line = write
    async with app.run_test():
        app.queued_query = ("q", "sicp", "m")
        await dispatch_command(app, "/web on")
    assert app.web_enabled is False and "locked" in lines[-1]
