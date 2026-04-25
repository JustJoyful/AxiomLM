"""
checkpoint.py — Per-page crash-recovery helper for OCR engines.

Checkpoints are stored as individual markdown files:
    CHECKPOINTS / book_stem / page_NNNN.md

This allows either OCR engine to resume from the last successfully
processed page after a VRAM OOM crash or manual interruption.
"""

import re
from pathlib import Path

from src.config import CHECKPOINTS


def _page_dir(book_stem: str) -> Path:
    return CHECKPOINTS / book_stem


def save(book_stem: str, page_idx: int, md_text: str) -> Path:
    """
    Write md_text as UTF-8 to CHECKPOINTS/book_stem/page_NNNN.md.
    Creates parent directories if they don't exist.
    Returns the written path.
    """
    out = _page_dir(book_stem) / f"page_{page_idx:04d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md_text, encoding="utf-8")
    return out


def load(book_stem: str, page_idx: int) -> str | None:
    """
    Return the checkpointed markdown for this page, or None if not found.
    Never raises — missing file returns None silently.
    """
    path = _page_dir(book_stem) / f"page_{page_idx:04d}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def last_completed(book_stem: str) -> int:
    """
    Scan CHECKPOINTS/book_stem/ for files matching page_NNNN.md.
    Return the highest page index found, or -1 if no checkpoints exist.
    """
    page_dir = _page_dir(book_stem)
    if not page_dir.exists():
        return -1

    pattern = re.compile(r"^page_(\d{4})\.md$")
    indices = [
        int(m.group(1))
        for f in page_dir.iterdir()
        if (m := pattern.match(f.name))
    ]
    return max(indices) if indices else -1


def all_pages(book_stem: str) -> list[Path]:
    """
    Return a sorted list of all checkpoint paths for this book.
    Returns an empty list if no checkpoints exist.
    """
    page_dir = _page_dir(book_stem)
    if not page_dir.exists():
        return []

    pattern = re.compile(r"^page_\d{4}\.md$")
    paths = sorted(f for f in page_dir.iterdir() if pattern.match(f.name))
    return paths
