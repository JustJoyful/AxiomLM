"""Book evidence through the unchanged Phase 2 retrieval path."""

from __future__ import annotations

import asyncio
import json

from src import config
from src.page_querying import physical_pages_for
from src.retrieval import search_book

from .protocol import EvidenceRef, ToolContext, ToolResult


def book_page_count(book_stem: str) -> int | None:
    """Physical page count from the frozen checkpoint layout (zero-based files)."""
    folder = config.PARSED_MD / "checkpoints" / book_stem
    if not folder.is_dir():
        return None
    return sum(1 for _ in folder.glob("page_[0-9][0-9][0-9][0-9].md")) or None


def evidence_for(candidate, book_stem: str) -> EvidenceRef:
    meta = candidate.metadata
    return EvidenceRef(
        source_id=candidate.chunk_id,
        kind="book",
        text=candidate.text,
        book_stem=book_stem,
        physical_pages=tuple(physical_pages_for(meta)),
        chapter=str(meta.get("chapter", meta.get("Header 1", "Unknown Chapter"))),
        section=str(meta.get("section", meta.get("Header 2", "Unknown Section"))),
    )


def passage_row(ref: EvidenceRef) -> dict:
    return {
        "source_id": ref.source_id,
        "chapter": ref.chapter,
        "section": ref.section,
        "pages": list(ref.physical_pages),
        "text": ref.text,
    }


async def search_local_index(
    arguments: dict, context: ToolContext, *, call_id: str, search=search_book
) -> ToolResult:
    page_range = None
    if "page_start" in arguments:
        page_range = (arguments["page_start"], arguments["page_end"])
        pages = book_page_count(context.book_stem)
        if pages is not None and page_range[1] > pages:
            return ToolResult(
                call_id,
                "search_local_index",
                "error",
                {"error": f"Book has {pages} physical pages; requested {page_range}"},
            )
    found = await asyncio.to_thread(
        search,
        arguments["query"],
        book_stem=context.book_stem,
        distance=context.distance,
        page_range=page_range,
    )
    refs, rows, omitted = [], [], 0
    base = {
        "query": arguments["query"],
        "retrieval_query": found.retrieval_query,
        "page_fallback": found.page_fallback,
        "index_version": found.index_version,
        "legacy_index": found.legacy,
    }
    # Whole passages only: an item that does not fit is omitted, never cut.
    budget = config.TOOL_MAX_RESULT_BYTES - len(json.dumps(base).encode()) - 256
    for candidate in found.candidates[: context.top_k]:
        ref = evidence_for(candidate, context.book_stem)
        row = passage_row(ref)
        size = len(json.dumps(row, ensure_ascii=False).encode())
        if size > budget:
            omitted += 1
            continue
        budget -= size
        refs.append(ref)
        rows.append(row)
    if not rows and omitted:
        return ToolResult(
            call_id,
            "search_local_index",
            "error",
            {**base, "error": "Retrieved passages exceed the tool result limit"},
        )
    return ToolResult(
        call_id,
        "search_local_index",
        "ok",
        {**base, "passages": rows, "omitted_passages": omitted},
        tuple(refs),
    )
