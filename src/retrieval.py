"""Request-local adapter over the unchanged Phase 2 retrieval path.

Ranking, index resolution, page fallback and the deterministic overview
normalization are exactly those of the TUI's original preparation step; this
module only gives the agent and the TUI one shared, testable boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.page_querying import extract_page_intent


@dataclass(frozen=True)
class BookSearchResult:
    candidates: tuple
    page_fallback: bool
    index_version: str
    legacy: bool
    retrieval_query: str
    overview: bool
    page_filter: tuple[int, int] | None


def search_book(
    query: str,
    *,
    book_stem: str,
    distance: float,
    page_range: tuple[int, int] | None = None,
    retrieval_query: str | None = None,
) -> BookSearchResult:
    """Retrieve hybrid candidates; explicit page ranges override parsed intent."""
    # Module-attribute lookups keep existing test seams and later patches effective.
    from src import hybrid_retrieval, index_store
    from src.query_rewriting import is_book_overview

    index = index_store.resolve_index(book_stem)
    effective = query if retrieval_query is None else retrieval_query
    if page_range is None:
        target = extract_page_intent(query)
        page_range = (min(target), max(target)) if target else None
    overview = page_range is None and is_book_overview(query)
    if overview:
        effective = config.BOOK_OVERVIEW_RETRIEVAL_QUERY

    def retrieve(page_filter):
        return hybrid_retrieval.retrieve_hybrid(
            index,
            effective,
            page_filter,
            config.HYBRID_DENSE_FETCH,
            config.HYBRID_BM25_FETCH,
            distance_threshold=distance,
        )

    candidates = retrieve(page_range)
    fallback = bool(page_range and not candidates)
    if fallback:
        candidates = retrieve(None)
    return BookSearchResult(
        tuple(candidates),
        fallback,
        index.version_id,
        index.legacy,
        effective,
        overview,
        page_range,
    )
