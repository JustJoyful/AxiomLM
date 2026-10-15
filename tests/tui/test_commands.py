import pytest

from src.ui.commands import COMMANDS, dispatch_command, parse_command
from src.ui.prompt import complete_pdf_path


def test_quoted_pdf_path_is_one_argument():
    assert parse_command('/parse "~/books/Linear Algebra.pdf"') == (
        "/parse",
        ["~/books/Linear Algebra.pdf"],
    )
    with pytest.raises(ValueError):
        parse_command('/parse "unfinished')


async def test_unknown_command_is_one_notice_and_never_a_question():
    class App:
        def __init__(self):
            self.notices = []

        async def write_line(self, text):
            self.notices.append(text)

        async def handle_command(self, *_):
            raise AssertionError("unknown commands must not dispatch")

    app = App()
    await dispatch_command(app, "/xyz")
    assert app.notices == ["✗ unknown command /xyz · /help"]
    assert "/cancel" in COMMANDS and "/delete" in COMMANDS


def test_pdf_completion_retains_directories_spaces_and_unicode(tmp_path):
    (tmp_path / "数学 notes.pdf").touch()
    (tmp_path / "skip.txt").touch()
    (tmp_path / "more books").mkdir()
    completions = complete_pdf_path(str(tmp_path) + "/")
    assert completions == [
        str(tmp_path / "more books") + "/",
        str(tmp_path / "数学 notes.pdf"),
    ]


def test_documented_commands_and_current_copy_backend_match_implementation():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    documented = set(re.findall(r"`(/[-\w]+)`", (root / "README.md").read_text()))
    documented.discard("/tmp")  # Filesystem path in the setup section.
    assert documented <= COMMANDS.keys()
    architecture = root / "ARCHITECTURE.md"
    if architecture.exists():
        current_table = architecture.read_text().split("## Hardware profiles", 1)[0]
        assert "pyperclip" not in current_table
