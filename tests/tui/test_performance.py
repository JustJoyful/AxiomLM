import asyncio
import time
from itertools import pairwise
from threading import Event

from src import tui
from src.ui.render import StudyMarkdown
from src.ui.stream import StudyStream, TurnRecord, TurnView


async def test_startup_paints_without_waiting_for_discovery(app_factory, monkeypatch):
    app = app_factory()
    gate = Event()
    monkeypatch.setattr(app, "_discover_books", lambda: gate.wait(2) or {})
    monkeypatch.setattr(app, "_fetch_ollama_models", lambda: gate.wait(2) or [])
    started = time.monotonic()
    try:
        async with app.run_test(size=(80, 24)):
            elapsed = time.monotonic() - started
            assert elapsed < 0.2
            assert app.theme == "ansi-dark"
            assert app.query_one("#home").display
            assert not app.motion.running
            gate.set()
    finally:
        gate.set()


async def test_two_thousand_tokens_keep_prompt_responsive_and_flush_at_30hz(
    app_factory, monkeypatch
):
    first = asyncio.Event()
    finished = asyncio.Event()
    append_times = []
    render_times = []
    original_append = StudyMarkdown.append

    async def append(widget, source):
        append_times.append(time.monotonic())
        await original_append(widget, source)

    monkeypatch.setattr(StudyMarkdown, "append", append)

    class Client:
        async def astream_generate(self, *_args, **_kwargs):
            first.set()
            for _ in range(2000):
                yield {"response": "token "}
                await asyncio.sleep(0.0005)
            yield {"done": True}
            finished.set()

        def unload_and_wait(self):
            return

    monkeypatch.setattr(tui, "OllamaClient", Client)
    app = app_factory()
    app.active_collection_name = "book"
    monkeypatch.setattr(app, "_prepare_query", lambda *_: ("prompt", [], [], []))
    async with app.run_test(size=(100, 30)) as pilot:
        prompt = app.query_one(tui.QueryComposer)
        await app.on_query_composer_submitted(
            tui.QueryComposer.Submitted(prompt, "stream performance")
        )
        await first.wait()
        await pilot.press("d", "r", "a", "f", "t")
        assert prompt.text == "draft"
        while app._active_query_request_id is not None:
            await asyncio.sleep(0.01)
        assert app.turns[-1].record.raw_answer == "token " * 2000
        assert len(app.query(StudyMarkdown)) == 1
        assert all(b - a >= 1 / 30 - 0.002 for a, b in pairwise(append_times))
        assert len(append_times) < 200
        assert all(
            b - a >= 1 / 30 - 0.002 for a, b in pairwise(render_times)
        ), render_times


async def test_native_scroll_anchor_and_prompt_caret_survive_resize(app_factory):
    app = app_factory()
    async with app.run_test(size=(80, 24)) as pilot:
        app._hide_home()
        record = TurnRecord("r", "question", "book", "model", phase="DONE")
        record.set_answer("\n\n".join(f"Paragraph {n}" for n in range(80)))
        view = TurnView(record)
        stream = app.query_one(StudyStream)
        await stream.add_turn(view)
        await pilot.pause()
        app.action_page_up()
        await pilot.pause()
        before = stream.scroll_y
        prompt = app.query_one(tui.QueryComposer)
        prompt.load_text("first\nsecond")
        prompt.move_cursor((1, 3))
        await pilot.resize_terminal(60, 24)
        assert prompt.cursor_location == (1, 3)
        assert stream.scroll_y == before
        app.action_follow()
        await pilot.pause()
        assert stream.is_vertical_scroll_end


async def test_two_thousand_retained_blocks_drop_older_ones_behind_one_summary(
    app_factory,
):
    from textual.widgets import Static

    app = app_factory()
    async with app.run_test():
        app._hide_home()
        stream = app.query_one(StudyStream)
        blocks = [
            Static(f"notice {n}", classes="notice", markup=False) for n in range(2001)
        ]
        await stream.mount(*blocks, before=stream.spacer)
        await stream.trim_history()
        assert len(stream.query_children(".turn, .notice")) == 2000
        assert stream.dropped == 1
        assert stream.summary.display
        assert not blocks[0].is_attached


async def test_native_selection_spans_multiple_turns(app_factory):
    app = app_factory()
    async with app.run_test(size=(80, 30)) as pilot:
        app._hide_home()
        stream = app.query_one(StudyStream)
        views = []
        for number in (1, 2):
            record = TurnRecord(
                str(number), f"Question {number}", "book", "model", phase="DONE"
            )
            record.set_answer(f"Answer {number}")
            view = TurnView(record)
            await stream.add_turn(view)
            views.append(view)
        await pilot.pause()
        await pilot.mouse_down(views[0].question, offset=(0, 0))
        await pilot.hover(views[1].answer, offset=(7, 0))
        await pilot.mouse_up(views[1].answer, offset=(7, 0))
        selected = app.screen.get_selected_text()
        assert selected is not None
        assert (
            "Question 1" in selected
            and "Answer 1" in selected
            and "Question 2" in selected
        )


async def test_diagnostic_limits_paints_across_query_transitions():
    from scripts.validate_tui_phase1 import validate

    report = await validate()
    assert report["maximum_paint_hz"] <= 30.1
    assert report["idle_active_timers"] == 0
    assert report["echo_ms"] <= 100
    assert report["cancel_ui_ms"] <= 200
    assert report["client_close_ms"] <= 200


async def test_final_math_formatting_preserves_native_selection(app_factory):
    app = app_factory()
    async with app.run_test(size=(80, 30)) as pilot:
        app._hide_home()
        record = TurnRecord("r", "question", "book", "model", phase="STREAMING")
        record.set_answer("Selected paragraph.\n\nRaw \\sum x")
        view = TurnView(record)
        app.turns.append(view)
        await app.query_one(StudyStream).add_turn(view)
        await view.answer.update(record.copy_text)
        await pilot.pause()
        paragraph = view.answer.query("MarkdownParagraph").first()
        await pilot.mouse_down(paragraph, offset=(0, 0))
        await pilot.hover(paragraph, offset=(8, 0))
        await pilot.mouse_up(paragraph, offset=(8, 0))
        before = app.screen.get_selected_text()
        assert before == "Selected "
        await view.finalize("DONE")
        await pilot.pause()
        assert app.screen.get_selected_text() == before
        assert "\\sum" in view.answer.source
        app.screen.clear_selection()
        await pilot.pause()
        assert "∑" in view.answer.source


async def test_native_stream_flushes_stay_bounded_after_a_slow_append(
    app_factory, monkeypatch
):
    from textual.widgets import Markdown

    times = []
    original = Markdown.append

    async def slow_first(widget, fragment):
        if not times:
            await asyncio.sleep(0.06)
        times.append(time.monotonic())
        await original(widget, fragment)

    monkeypatch.setattr(Markdown, "append", slow_first)
    app = app_factory()
    async with app.run_test():
        view = TurnView(TurnRecord("r", "q", "book", "model"))
        await app.query_one(StudyStream).add_turn(view)
        updater = view.answer.get_stream(view.answer)
        await updater.write("first ")
        await asyncio.sleep(0.034)
        await updater.write("second ")
        await asyncio.sleep(0.034)
        await updater.write("third")
        await updater.stop()
        assert len(times) >= 2
        assert all(b - a >= 1 / 30 - 0.002 for a, b in pairwise(times)), times
