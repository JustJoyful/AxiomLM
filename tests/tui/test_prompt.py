from textual.app import App

from src.ui.prompt import QueryComposer


async def test_enter_sends_and_newline_fallbacks_preserve_draft():
    submitted = []

    class Probe(App):
        def compose(self):
            yield QueryComposer(id="query_input")

        def on_query_composer_submitted(self, event):
            submitted.append(event.value)

    app = Probe()
    async with app.run_test() as pilot:
        prompt = app.query_one(QueryComposer)
        prompt.focus()
        await pilot.press(
            "a", "alt+enter", "b", "ctrl+j", "c", "shift+enter", "d", "enter"
        )
        assert submitted == ["a\nb\nc\nd"]
        prompt.load_text("draft")
        prompt.move_cursor((0, 5))
        await pilot.press("up")
        assert prompt.text == "a\nb\nc\nd"
        prompt.move_cursor((3, 1))
        await pilot.press("down")
        assert prompt.text == "draft"


async def test_unicode_paste_resize_and_prompt_height(app_factory):
    from textual.events import Paste

    app = app_factory()
    value = "中文 العربية e\u0301 👩‍💻\n" * 10
    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one(QueryComposer)
        app.post_message(Paste(value))
        await pilot.pause()
        assert prompt.text == value
        assert 1 <= prompt.size.height <= 8
        cursor = prompt.cursor_location
        await pilot.resize_terminal(40, 20)
        assert prompt.text == value
        assert prompt.cursor_location == cursor
        assert 1 <= prompt.size.height <= 8
