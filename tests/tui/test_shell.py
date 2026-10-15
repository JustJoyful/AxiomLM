from threading import Event

from rich.cells import cell_len
from textual.widgets import Button, Footer, Select

from src.ui.prompt import QueryComposer
from src.ui.stream import StudyStream


async def test_home_tagline_and_logo_share_center_after_resize(app_factory):
    app = app_factory()
    async with app.run_test(size=(100, 30)) as pilot:
        for width in [100, 80, 60, 59, 40, 101, 160]:
            await pilot.resize_terminal(width, 30)
            await pilot.pause()
            lines = [strip.text for strip in app.screen._compositor.render_strips()]
            logo_lines = [line for line in lines if "█" in line or "AXIOM" in line]
            tagline = next(line for line in lines if "Your textbooks." in line)
            logo_left = min(
                cell_len(line) - cell_len(line.lstrip()) for line in logo_lines
            )
            logo_right = max(cell_len(line.rstrip()) for line in logo_lines)
            tagline_left = cell_len(tagline) - cell_len(tagline.lstrip())
            tagline_right = cell_len(tagline.rstrip())
            assert abs(logo_left + logo_right - tagline_left - tagline_right) <= 1
            assert abs(tagline_left + tagline_right - width) <= 1, {
                name: app.query_one(name).region
                for name in ["#chat_scroll", "#home", "#logo", "#tagline"]
            }


async def test_shell_has_one_surface_and_unknown_commands_stay_inline(app_factory):
    app = app_factory()
    async with app.run_test(size=(100, 30)) as pilot:
        assert not list(app.query(Button))
        assert not list(app.query(Select))
        assert not list(app.query(Footer))
        assert len(app.query(StudyStream)) == 1
        prompt = app.query_one("#query_input", QueryComposer)
        prompt.load_text("/xyz")
        await pilot.press("enter")
        await pilot.pause()
        assert app.notices[-1] == "✗ unknown command /xyz · /help"
        assert not app._conversation


async def test_pending_discovery_keeps_prompt_usable(app_factory, monkeypatch):
    app = app_factory()
    done = Event()
    monkeypatch.setattr(app, "_fetch_ollama_models", lambda: done.wait(2) or [])
    try:
        async with app.run_test() as pilot:
            prompt = app.query_one(QueryComposer)
            prompt.focus()
            await pilot.press("a", "b", "c")
            assert prompt.text == "abc"
    finally:
        done.set()


async def test_commands_popup_tab_and_narrow_resize(app_factory):
    app = app_factory()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+p")
        assert app.query_one("#overlay").display
        await pilot.press("tab")
        assert app.query_one(QueryComposer).text.startswith("/")
        assert not app.notices or not app.notices[-1].startswith("Commands")
        await pilot.press("escape")
        await pilot.resize_terminal(39, 11)
        assert app.query_one("#resize_warning").display
        assert not app.query_one("#surface").display
        await pilot.resize_terminal(60, 24)
        assert app.query_one("#surface").display


async def test_home_stays_hidden_after_clear_and_new(app_factory):
    app = app_factory()
    async with app.run_test() as pilot:
        prompt = app.query_one(QueryComposer)
        prompt.load_text("question without an indexed book")
        await pilot.press("enter")
        assert not app.query_one("#home").display
        assert app.notices[-1] == "○ no book loaded · /parse <path>"
        await app.handle_command("/clear", [])
        await app.handle_command("/new", [])
        assert not app.query_one("#home").display
        assert app.active_collection_name is None


async def test_removal_requires_inline_yes_and_discards_its_evidence(
    app_factory, monkeypatch
):
    from src import tui

    removed = []
    monkeypatch.setattr(
        tui, "delete_collection", lambda book: removed.append(book) or True
    )
    app = app_factory()
    app._discover_books = lambda: {"book": 1}
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.handle_command("/remove", ["book"])
        assert not removed
        assert app.notices[-1] == "○ remove book? type y to confirm"
        await app.on_query_composer_submitted(
            QueryComposer.Submitted(app.query_one(QueryComposer), "y")
        )
        assert removed == ["book"]
        assert app.active_collection_name is None


async def test_busy_commands_keep_live_turn_and_captured_identity(app_factory):
    from src.ui.stream import TurnRecord, TurnView

    app = app_factory()
    async with app.run_test():
        record = TurnRecord("r", "q", "book", "model")
        view = TurnView(record)
        await app.query_one(StudyStream).add_turn(view)
        app.turns.append(view)
        token = app._try_acquire_workload("generation")
        for name, arguments in [
            ("/clear", []),
            ("/new", []),
            ("/book", ["other"]),
            ("/remove", ["book"]),
        ]:
            await app.handle_command(name, arguments)
            assert view.is_attached
            assert record.book == "book" and record.model == "model"
        app._release_workload(token)


async def test_theme_preview_escape_keeps_fence_raw_and_display(app_factory):
    from src.ui.stream import TurnRecord, TurnView

    app = app_factory()
    async with app.run_test() as pilot:
        record = TurnRecord("r", "q", "book", "model")
        record.set_answer("```python\nx = 'λ'")
        view = TurnView(record)
        await app.query_one(StudyStream).add_turn(view)
        before = app.theme
        await app.handle_command("/theme", [])
        await pilot.press("down", "down")
        await pilot.pause()
        assert app.theme == "catppuccin-mocha"
        await pilot.press("escape")
        await pilot.pause()
        assert app.theme == before
        assert record.raw_answer == "```python\nx = 'λ'"
        assert view.answer.source == "```python\nx = 'λ'"
        assert not app.settings_path.exists()


async def test_theme_preview_failed_save_restores_palette_file_and_partial_fence(
    app_factory, monkeypatch
):
    from src.ui import settings
    from src.ui.stream import TurnRecord, TurnView

    app = app_factory()
    app.settings_path.write_text('# keep this\ntheme = "terminal"\n')
    before_bytes = app.settings_path.read_bytes()
    async with app.run_test() as pilot:
        record = TurnRecord("r", "q", "book", "model")
        record.set_answer("```python\nx = 'λ'")
        view = TurnView(record)
        await app.query_one(StudyStream).add_turn(view)
        before = app.theme
        await app.handle_command("/theme", [])
        await pilot.press("down", "down")
        await pilot.pause()
        assert app.theme == "catppuccin-mocha"

        def fail(*_args):
            raise OSError("disk full")

        monkeypatch.setattr(settings.os, "replace", fail)
        await pilot.press("enter")
        await pilot.pause()
        assert app.theme == before
        assert app.settings_path.read_bytes() == before_bytes
        assert record.raw_answer == "```python\nx = 'λ'"
        assert view.answer.source == record.raw_answer
        assert app.notices[-1].startswith("✗ theme save failed:")


async def test_startup_errors_are_exact_inline_and_prompt_stays_usable(app_factory):
    app = app_factory()
    app._fetch_ollama_models = list
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        assert f"✗ model {app.selected_model} not found · /models" in app.notices
        app.selected_model = ""
        await app._run_startup_checks()
        assert "○ no model · /model <name>" in app.notices

        def offline():
            raise OSError("offline")

        app._fetch_ollama_models = offline
        await app._run_startup_checks()
        assert (
            "✗ can't reach Ollama at localhost:11434 · start it, then /retry"
            in app.notices
        )
        await pilot.press("d", "r", "a", "f", "t")
        assert app.query_one(QueryComposer).text == "draft"


async def test_only_overlays_have_widget_borders(app_factory):
    from src.ui.stream import TurnRecord, TurnView

    app = app_factory()
    async with app.run_test() as pilot:
        record = TurnRecord("r", "q", "book", "model")
        record.set_answer(
            "# Heading\n\n```py\nx=1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |"
        )
        await app.query_one(StudyStream).add_turn(TurnView(record))
        await pilot.pause()
        for widget in app.screen.walk_children():
            if widget.id != "overlay":
                assert not any(edge[0] for edge in widget.styles.border), widget


async def test_truecolor_capability_warning_is_once_and_rows_are_tagged(
    app_factory, monkeypatch
):
    monkeypatch.delenv("COLORTERM", raising=False)
    app = app_factory()
    async with app.run_test():
        await app.handle_command("/theme", [])
        option = app.overlay.get_option("paper")
        assert "needs truecolor" in str(option.prompt)
        await app._theme_notice("paper")
        await app._theme_notice("nord")
        assert app.notices.count("○ truecolor unavailable · try /theme terminal") == 1


async def test_incomplete_book_does_not_hide_healthy_books_and_can_be_removed(
    app_factory, monkeypatch
):
    from types import SimpleNamespace

    from src import tui

    removed = []
    monkeypatch.setattr(tui, "list_collections", lambda: ["broken", "valid"])
    monkeypatch.setattr(
        tui, "get_collection", lambda name: SimpleNamespace(name=name, count=lambda: 3)
    )

    def ready(collection):
        if collection.name == "broken":
            raise RuntimeError("Index is incomplete")

    monkeypatch.setattr(tui, "ensure_collection_ready", ready)
    monkeypatch.setattr(
        tui, "delete_collection", lambda name: removed.append(name) or True
    )
    app = app_factory()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        assert app.active_collection_name == "valid"
        await app.handle_command("/books", [])
        assert "valid (3)" in app.notices[-1]
        assert "broken" in app.notices[-1] and "incomplete" in app.notices[-1]
        if app.overlay:
            await app.close_overlay()
        await app.handle_command("/book", ["broken"])
        assert app.active_collection_name == "valid"
        await app.handle_command("/remove", ["broken"])
        await app.on_query_composer_submitted(
            QueryComposer.Submitted(app.query_one(QueryComposer), "y")
        )
        assert removed == ["broken"]
        assert app.active_collection_name == "valid"
        await pilot.press("d", "r", "a", "f", "t")
        assert app.query_one(QueryComposer).text == "draft"


async def test_books_popup_selects_a_book_on_enter(app_factory, monkeypatch):
    app = app_factory()
    monkeypatch.setattr(app, "_discover_books", lambda: {"first": 2, "second": 3})
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await app.handle_command("/books", [])
        assert app.overlay is not None and app.overlay.kind == "books"
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.active_collection_name == "second"
        assert app.overlay is None


async def test_enter_runs_highlighted_typed_command_completion(app_factory):
    app = app_factory()
    async with app.run_test() as pilot:
        await pilot.press("/", "m", "o")
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.notices[-1] == "○ usage: /model <name>"
        prompt = app.query_one(QueryComposer)
        assert prompt.text == ""
        prompt.load_text("/model missing")
        await pilot.press("enter")
        assert app.notices[-1] == "✗ model missing not found · /models"
