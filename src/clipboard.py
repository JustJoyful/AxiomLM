"""
clipboard.py — Clipboard helpers for copying assistant responses.
"""

from __future__ import annotations

import re

import pyperclip


def copy_to_clipboard(text: str) -> tuple[bool, str]:
    """Copy text to the system clipboard and return status + UX message."""
    if not text.strip():
        return False, "Nothing to copy."

    try:
        pyperclip.copy(text)
    except pyperclip.PyperclipException:
        return False, "Clipboard backend unavailable on this system."
    except Exception:
        return False, "Copy failed."

    char_count = len(text)
    line_count = text.count("\n") + 1
    return True, f"Copied {char_count} chars ({line_count} lines)."


def extract_code_blocks(text: str) -> str | None:
    """Return combined fenced markdown code blocks, or None if absent."""
    blocks = re.findall(r"```(?:[\w+-]+)?\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        return None
    return "\n\n".join(block.rstrip() for block in blocks if block.strip()) or None
