import pytest

from src import clipboard


def test_wayland_stdin_is_byte_exact_without_shell(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "test")
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(clipboard.subprocess, "run", run)
    raw = "λ = '\\frac{a}{b}'\n\n"
    ok, message = clipboard.copy_to_clipboard(raw)
    assert ok and message == f"copied {len(raw)} chars"
    assert calls[0][0] == ["/usr/bin/wl-copy"]
    assert calls[0][1]["input"] == raw.encode()
    assert not calls[0][1].get("shell", False)
    assert calls[0][1]["timeout"] == 1.0


def test_missing_wayland_tool_warns_and_uses_osc52(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("WAYLAND_DISPLAY", "test")
    monkeypatch.setattr(clipboard.shutil, "which", lambda _: None)
    copied = []

    class App:
        _driver = SimpleNamespace(is_headless=False)

        def copy_to_clipboard(self, text):
            copied.append(text)

    ok, message = clipboard.copy_text(App(), "\\sum x\n")
    assert ok and copied == ["\\sum x\n"]
    assert "install wl-clipboard" in message
    assert "OSC 52 copy requested · 7 chars · terminal support required" in message


def test_failed_all_backends_leaves_a_readable_failure(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(clipboard.shutil, "which", lambda _: None)
    assert clipboard.copy_to_clipboard("text")[0] is False
    assert clipboard.copy_to_clipboard("")[0] is False


def test_native_clipboard_is_used_when_osc52_is_unconfirmed(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: "/usr/bin/xclip" if name == "xclip" else None,
    )
    calls = []
    monkeypatch.setattr(
        clipboard.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs))
    )

    class App:
        _driver = None

        def copy_to_clipboard(self, _text):
            raise AssertionError("no terminal driver")

    ok, message = clipboard.copy_text(App(), "exact\n\n")
    assert ok and message == "copied 7 chars"
    assert calls[0][0] == ["/usr/bin/xclip", "-selection", "clipboard"]
    assert calls[0][1]["input"] == b"exact\n\n"


def test_no_driver_cannot_claim_an_osc52_copy(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(clipboard.shutil, "which", lambda _: None)

    class App:
        _driver = None

        def copy_to_clipboard(self, _text):
            pass

    ok, message = clipboard.copy_text(App(), "text")
    assert not ok and "clipboard unavailable" in message


def test_daemon_clipboard_does_not_hold_a_captured_stderr_pipe(monkeypatch, tmp_path):
    import os
    import signal
    import sys

    executable = tmp_path / "wl-copy"
    child_pid = tmp_path / "child.pid"
    executable.write_text(
        f"#!{sys.executable}\nimport os, sys, time\n"
        "sys.stdin.buffer.read()\npid = os.fork()\n"
        f"if pid:\n open({str(child_pid)!r}, 'w').write(str(pid))\n os._exit(0)\n"
        "time.sleep(10)\nos._exit(0)\n"
    )
    executable.chmod(0o700)
    monkeypatch.setenv("WAYLAND_DISPLAY", "test")
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: str(executable) if name == "wl-copy" else None,
    )
    try:
        ok, message = clipboard.copy_to_clipboard("synthetic clipboard text")
        assert ok and message.startswith("copied")
    finally:
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("key", ["ctrl+c", "ctrl+shift+c", "super+c"])
async def test_ctrl_c_copies_native_selection_instead_of_arming_quit(
    app_factory, monkeypatch, key
):
    from src.ui.stream import StudyStream, TurnRecord, TurnView

    app = app_factory()
    copied = []
    monkeypatch.setattr(
        clipboard.subprocess,
        "run",
        lambda args, **kwargs: copied.append(kwargs["input"]),
    )
    monkeypatch.setenv("WAYLAND_DISPLAY", "test")
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: "/usr/bin/wl-copy" if name == "wl-copy" else None,
    )
    async with app.run_test(size=(80, 30)) as pilot:
        app._hide_home()
        record = TurnRecord("copy", "Question", "book", "model", phase="DONE")
        record.set_answer("Selected answer")
        view = TurnView(record)
        await app.query_one(StudyStream).add_turn(view)
        await pilot.pause()
        await pilot.mouse_down(view.question, offset=(0, 0))
        await pilot.hover(view.answer, offset=(7, 0))
        await pilot.mouse_up(view.answer, offset=(7, 0))
        selected = app.screen.get_selected_text()
        assert selected and "Question" in selected
        await pilot.press(key)
        await pilot.pause()
        assert copied == [selected.encode()]
        assert app._last_interrupt == 0
        assert not app._exit


async def test_selected_prompt_copy_pastes_back_into_composer(app_factory, monkeypatch):
    from src.ui.prompt import QueryComposer

    monkeypatch.setenv("WAYLAND_DISPLAY", "test")
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: "/usr/bin/wl-copy" if name == "wl-copy" else None,
    )
    monkeypatch.setattr(clipboard.subprocess, "run", lambda *args, **kwargs: None)
    app = app_factory()
    async with app.run_test() as pilot:
        composer = app.query_one(QueryComposer)
        composer.load_text("copy this λ")
        composer.select_all()
        await pilot.press("ctrl+c")
        composer.move_cursor(composer.document.end)
        await pilot.press("ctrl+v")
        assert composer.text == "copy this λcopy this λ"
