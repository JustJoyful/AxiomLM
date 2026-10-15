import pytest

from src.ui.render import SafeAnswerBuffer


@pytest.mark.parametrize("interrupt", ["escape", "quit"])
async def test_real_enter_dispatch_starts_backend_and_keeps_escape_and_quit_live(
    app_factory, monkeypatch, interrupt
):
    import asyncio

    from src import tui
    from src.ui.prompt import QueryComposer

    entered = asyncio.Event()
    closed = asyncio.Event()
    callbacks = []
    app = app_factory()
    app._discover_books = lambda: {"git_book": 1}
    original_after_refresh = app.call_after_refresh

    def capture(callback, *args, **kwargs):
        callbacks.append(callback)
        return original_after_refresh(callback, *args, **kwargs)

    monkeypatch.setattr(app, "call_after_refresh", capture)
    monkeypatch.setattr(app, "_prepare_query", lambda *_: ("prompt", [], [], []))

    class Client:
        async def astream_generate(self, *_args, **_kwargs):
            assert any(
                "What is Git?" in line.text
                for line in app.screen._compositor.render_strips()
            )
            entered.set()
            try:
                yield {"response": "Git is "}
                await asyncio.Event().wait()
            finally:
                closed.set()

        def unload_and_wait(self):
            return

    monkeypatch.setattr(tui, "OllamaClient", Client)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        prompt = app.query_one(QueryComposer)
        prompt.load_text("What is Git?")
        pressed = asyncio.create_task(pilot.press("enter"))
        try:
            await asyncio.wait_for(entered.wait(), 0.5)
            await pressed
            if interrupt == "escape":
                await pilot.press("escape")
                await asyncio.wait_for(closed.wait(), 0.5)
                for _ in range(100):
                    if app._active_query_request_id is None:
                        break
                    await asyncio.sleep(0.01)
                assert app.turns[-1].record.phase == "CANCELLED"
            prompt.load_text("/quit")
            await asyncio.wait_for(pilot.press("enter"), 1)
            assert app._exit
            assert closed.is_set()
        finally:
            # Release the broken implementation's self-wait so a RED test can exit.
            for callback in callbacks:
                if isinstance(getattr(callback, "__self__", None), asyncio.Event):
                    callback()
            await asyncio.wait_for(pressed, 2)


@pytest.mark.parametrize("interrupt", ["escape", "quit"])
async def test_cancel_before_first_query_paint_releases_admission(
    app_factory, monkeypatch, interrupt
):
    import asyncio

    from src import tui
    from src.ui.prompt import QueryComposer

    app = app_factory()
    app._discover_books = lambda: {"git_book": 1}
    original_after_refresh = app.call_after_refresh
    held = []

    def hold_query_paint(callback, *args, **kwargs):
        if isinstance(getattr(callback, "__self__", None), asyncio.Event):
            held.append(callback)
            return True
        return original_after_refresh(callback, *args, **kwargs)

    monkeypatch.setattr(app, "call_after_refresh", hold_query_paint)
    monkeypatch.setattr(
        app, "_prepare_query", lambda *_: pytest.fail("retrieval started before paint")
    )

    class Client:
        def unload_and_wait(self):
            return

    monkeypatch.setattr(tui, "OllamaClient", Client)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        prompt = app.query_one(QueryComposer)
        prompt.load_text("What is Git?")
        pressed = asyncio.create_task(pilot.press("enter"))
        try:
            await asyncio.wait_for(pressed, 0.5)
            assert held and app.turns[-1].record.phase == "SUBMITTED"
            if interrupt == "escape":
                await pilot.press("escape")
            else:
                prompt.load_text("/quit")
                await asyncio.wait_for(pilot.press("enter"), 1)
                assert app._exit
            for _ in range(100):
                if app._active_query_request_id is None:
                    break
                await asyncio.sleep(0.01)
            assert app._workload_coordinator.active_kind is None
        finally:
            for callback in held:
                callback()
            await asyncio.wait_for(pressed, 2)


ABSTENTION = (
    "The indexed material does not provide enough evidence to answer this question."
)


@pytest.mark.parametrize("split", range(len(ABSTENTION) + 1))
def test_abstention_prefix_splits_never_flash_continuation(split):
    buffer = SafeAnswerBuffer()
    first = buffer.feed(ABSTENTION[:split])
    assert not first or first == ABSTENTION
    second = buffer.feed(ABSTENTION[split:] + " Forbidden continuation.")
    assert first + second == ABSTENTION
    assert buffer.finish() == ABSTENTION
    assert buffer.raw == ABSTENTION + " Forbidden continuation."


def test_partial_citation_tokens_and_definition_lines_are_buffered():
    buffer = SafeAnswerBuffer()
    visible = ""
    for fragment in [
        "An answer[",
        "^1",
        "]\n[",
        "^1]: ",
        "Invented citation\n",
        "Next sentence.",
    ]:
        visible += buffer.feed(fragment)
        assert "[^" not in visible and "Invented citation" not in visible
    final = buffer.finish()
    assert final == "An answer\nNext sentence."


def test_non_abstention_diverges_and_keeps_exact_model_output():
    buffer = SafeAnswerBuffer()
    assert buffer.feed("The indexed ") == ""
    assert (
        buffer.feed("material describes evaluation.")
        == "The indexed material describes evaluation."
    )
    assert buffer.finish() == "The indexed material describes evaluation."


async def test_one_queued_question_waits_for_done_and_cleanup(app_factory, monkeypatch):
    import asyncio
    from threading import Event

    from src import tui
    from src.ui.prompt import QueryComposer

    app = app_factory()
    app.active_collection_name = "book"
    entered = asyncio.Event()
    finish_response = asyncio.Event()
    finish_cleanup = Event()
    started = []
    unload_calls = []

    class Client:
        async def astream_generate(self, prompt, **kwargs):
            started.append((prompt, kwargs["model"]))
            if len(started) == 1:
                entered.set()
                yield {"response": "First answer."}
                await finish_response.wait()
            else:
                yield {"response": "Second answer."}
            yield {"done": True}

        def unload_and_wait(self):
            unload_calls.append(True)
            if len(unload_calls) == 1:
                assert finish_cleanup.wait(3)

    monkeypatch.setattr(tui, "OllamaClient", Client)
    monkeypatch.setattr(app, "_prepare_query", lambda query, *_: (query, [], [], []))
    try:
        async with app.run_test():
            prompt = app.query_one(QueryComposer)
            await app.on_query_composer_submitted(
                QueryComposer.Submitted(prompt, "first")
            )
            await entered.wait()
            await app.on_query_composer_submitted(
                QueryComposer.Submitted(prompt, "second")
            )
            await app.on_query_composer_submitted(
                QueryComposer.Submitted(prompt, "third")
            )
            assert app.queued_query == ("second", "book", tui.OLLAMA_MODEL)
            assert prompt.text == "third"
            finish_response.set()
            await asyncio.sleep(0.1)
            assert len(started) == 1
            assert app._try_acquire_workload("ingest") is None
            finish_cleanup.set()
            for _ in range(100):
                if len(started) == 2 and app._workload_coordinator.active_kind is None:
                    break
                await asyncio.sleep(0.01)
            assert [query for query, _model in started] == ["first", "second"]
            assert app.queued_query is None
            assert app._workload_coordinator.active_kind is None
    finally:
        finish_cleanup.set()
        finish_response.set()


def test_model_numeric_citations_do_not_flash_but_code_indexes_survive():
    buffer = SafeAnswerBuffer()
    visible = ""
    for fragment in [
        "Claim [",
        "1",
        "].\n[",
        "1]: invented\n",
        "Literal `a[1]` and ",
        "```py\na[1]\n```",
    ]:
        visible += buffer.feed(fragment)
        assert "Claim [1]" not in visible and "invented" not in visible
    assert "Claim." in buffer.finish()
    assert "`a[1]`" in buffer.finish()
    assert "a[1]\n```" in buffer.finish()
