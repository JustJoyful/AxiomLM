"""Native multiline editing with history at the first and last logical lines."""

from pathlib import Path
from typing import ClassVar

from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class QueryComposer(TextArea):
    BINDINGS: ClassVar = [
        Binding("enter", "submit_query", show=False, priority=True),
        Binding(
            "shift+enter,alt+enter,ctrl+j",
            "insert_soft_newline",
            show=False,
            priority=True,
        ),
        Binding("up", "history_up", show=False, priority=True),
        Binding("down", "history_down", show=False, priority=True),
        Binding("tab", "complete", show=False, priority=True),
    ]

    class Submitted(Message):
        def __init__(self, composer, value):
            super().__init__()
            self.composer, self.value = composer, value

    class CompletionRequested(Message):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor_blink = False
        self.query_history = []
        self.query_history_index = 0
        self.draft = ""

    def action_submit_query(self):
        value = self.text
        if value.strip():
            if not self.query_history or value != self.query_history[-1]:
                self.query_history.append(value)
            self.query_history_index = len(self.query_history)
            self.post_message(self.Submitted(self, value))

    def action_insert_soft_newline(self):
        self.insert("\n")

    def action_complete(self):
        self.post_message(self.CompletionRequested())

    def action_history_up(self):
        if getattr(self.app, "overlay", None):
            self.app.move_overlay(-1)
            return
        if self.cursor_location[0] != 0:
            self.action_cursor_up()
        elif self.query_history_index > 0:
            if self.query_history_index == len(self.query_history):
                self.draft = self.text
            self.query_history_index -= 1
            self.load_text(self.query_history[self.query_history_index])
            self.move_cursor((0, 0))

    def action_history_down(self):
        if getattr(self.app, "overlay", None):
            self.app.move_overlay(1)
            return
        if self.cursor_location[0] != self.document.line_count - 1:
            self.action_cursor_down()
        elif self.query_history_index < len(self.query_history):
            self.query_history_index += 1
            self.load_text(
                self.query_history[self.query_history_index]
                if self.query_history_index < len(self.query_history)
                else self.draft
            )
            self.move_cursor(self.document.end)


def complete_pdf_path(prefix: str) -> list[str]:
    expanded = Path(prefix).expanduser()
    directory = expanded if prefix.endswith("/") else expanded.parent
    stem = "" if prefix.endswith("/") else expanded.name
    try:
        matches = sorted(
            path
            for path in directory.iterdir()
            if path.name.startswith(stem)
            and (path.is_dir() or path.suffix.lower() == ".pdf")
        )
    except OSError:
        return []
    return [str(path) + ("/" if path.is_dir() else "") for path in matches]
