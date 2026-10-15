"""Explicit clipboard transports; call subprocess work from a worker."""

import os
import shutil
import subprocess
import sys
from threading import get_ident

from src.config import UI_CLIPBOARD_TIMEOUT
from src.ui.render import extract_fenced_blocks


def _copy(text, app=None):
    if not text:
        return False, "○ nothing to copy"
    warning = ""
    if os.environ.get("WAYLAND_DISPLAY"):
        executable = shutil.which("wl-copy")
        if executable:
            try:
                subprocess.run(
                    [executable],
                    input=text.encode("utf-8"),
                    stdout=subprocess.DEVNULL,
                    # Clipboard daemons inherit pipes after the launching process exits.
                    stderr=subprocess.DEVNULL,
                    timeout=UI_CLIPBOARD_TIMEOUT,
                    check=True,
                )
                return True, f"copied {len(text)} chars"
            except (OSError, subprocess.SubprocessError):
                warning = "○ wl-copy failed · check the Wayland connection\n"
        else:
            warning = "○ wl-copy missing · install wl-clipboard\n"
    # OSC 52 has no acknowledgement; prefer an applicable native transport.
    for name, arguments in [
        ("xclip", ["-selection", "clipboard"]),
        ("xsel", ["--clipboard", "--input"]),
        ("pbcopy", []),
    ]:
        if name in {"xclip", "xsel"} and not os.environ.get("DISPLAY"):
            continue
        if name == "pbcopy" and sys.platform != "darwin":
            continue
        executable = shutil.which(name)
        if executable:
            try:
                subprocess.run(
                    [executable, *arguments],
                    input=text.encode("utf-8"),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=UI_CLIPBOARD_TIMEOUT,
                    check=True,
                )
                return True, warning + f"copied {len(text)} chars"
            except (OSError, subprocess.SubprocessError):
                continue
    driver = getattr(app, "_driver", None)
    if driver is not None and not driver.is_headless:
        try:
            if hasattr(app, "_thread_id") and app._thread_id != get_ident():
                app.call_from_thread(app.copy_to_clipboard, text)
            else:
                app.copy_to_clipboard(text)
            return (
                True,
                warning
                + f"○ OSC 52 copy requested · {len(text)} chars · terminal support required",
            )
        except (RuntimeError, OSError):
            pass
    return (
        False,
        warning + "✗ clipboard unavailable · install wl-clipboard or enable OSC 52",
    )


def copy_to_clipboard(text: str) -> tuple[bool, str]:
    return _copy(text)


def copy_text(app, text: str) -> tuple[bool, str]:
    return _copy(text, app)


def extract_code_blocks(text: str) -> str | None:
    blocks = extract_fenced_blocks(text)
    return "\n".join(block.source for block in blocks) if blocks else None
