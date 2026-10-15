"""Characterize the native APIs the redesign depends on."""

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.theme import BUILTIN_THEMES
from textual.widgets import Markdown, TextArea


def test_terminal_theme_has_supported_ansi_route():
    assert {
        "ansi-dark",
        "ansi-light",
    } <= BUILTIN_THEMES.keys() or "textual-ansi" in BUILTIN_THEMES


def test_app_keeps_native_cross_container_selection(app_factory):
    app = app_factory()
    assert app.ALLOW_SELECT
    assert app.ENABLE_SELECT_AUTO_SCROLL


def test_partial_fence_survives_theme_switch_and_focus_override():
    class Probe(App):
        CSS = "TextArea { border: none; background: transparent; } TextArea:focus { border: none; }"

        def compose(self) -> ComposeResult:
            with Vertical():
                yield Markdown(id="answer")
            with Vertical():
                yield TextArea(id="prompt", highlight_cursor_line=False)

    async def probe():
        app = Probe()
        async with app.run_test() as pilot:
            answer = app.query_one(Markdown)
            stream = Markdown.get_stream(answer)
            await stream.write("```python\nx = 'λ'")
            await pilot.pause()
            app.theme = "textual-light"
            await stream.write("\n```\n")
            await pilot.pause()
            await stream.stop()
            await pilot.pause()
            assert answer.source == "```python\nx = 'λ'\n```\n"
            prompt = app.query_one(TextArea)
            prompt.focus()
            await pilot.pause()
            assert not any(edge[0] for edge in prompt.styles.border)

    asyncio.run(asyncio.wait_for(probe(), timeout=10))
