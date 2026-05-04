"""
page_querying.py — Shared page-intent parsing and citation formatting helpers.
"""

from __future__ import annotations

import re


def extract_page_intent(query: str) -> list[int] | None:
    """
    Extract explicit page intent from query.

    Accepted forms (v1):
      - page N
      - p.N / pN
      - pg.N / pgN
      - pages N-M (inclusive range)
    """
    text = query.strip().lower()
    if not text:
        return None

    range_match = re.search(r"\bpages\s+([1-9]\d*)\s*-\s*([1-9]\d*)\b", text)
    if range_match:
        start_page = int(range_match.group(1))
        end_page = int(range_match.group(2))
        if end_page < start_page:
            return None
        return list(range(start_page, end_page + 1))

    single_patterns = (
        r"\bpage\s+([1-9]\d*)\b",
        r"\bp\.?\s*([1-9]\d*)\b",
        r"\bpg\.?\s*([1-9]\d*)\b",
    )
    for pattern in single_patterns:
        match = re.search(pattern, text)
        if match:
            return [int(match.group(1))]
    return None


def collapse_page_ranges(pages: list[int]) -> str:
    """Collapse sorted page list into compact range text."""
    if not pages:
        return ""

    ranges: list[str] = []
    start = pages[0]
    end = pages[0]
    for page in pages[1:]:
        if page == end + 1:
            end = page
            continue
        ranges.append(f"{start}-{end}" if start != end else str(start))
        start = page
        end = page
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


def build_citation(book_stem: str, retrieved_pages: list[int]) -> str:
    """Build citation string from retrieved physical pages."""
    pages = sorted({p for p in retrieved_pages if isinstance(p, int) and p >= 1})
    if not pages:
        return f"[{book_stem}]"
    if len(pages) == 1:
        return f"[{book_stem} · p.{pages[0]}]"
    return f"[{book_stem} · p.{collapse_page_ranges(pages)}]"
