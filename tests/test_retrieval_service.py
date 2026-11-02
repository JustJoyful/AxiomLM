from types import SimpleNamespace

from src import config, hybrid_retrieval, index_store
from src.retrieval import search_book


def patch(monkeypatch, responses):
    calls = []
    monkeypatch.setattr(
        index_store,
        "resolve_index",
        lambda book: SimpleNamespace(version_id="v7", legacy=False, book_stem=book),
    )

    def retrieve(index, query, page_filter, dense, bm25, distance_threshold):
        calls.append((query, page_filter, distance_threshold))
        return responses.pop(0)

    monkeypatch.setattr(hybrid_retrieval, "retrieve_hybrid", retrieve)
    return calls


def test_page_intent_filters_then_falls_back_and_traces(monkeypatch):
    calls = patch(monkeypatch, [[], ["c1"]])
    found = search_book("explain page 12", book_stem="sicp", distance=1.5)
    assert calls == [("explain page 12", (12, 12), 1.5), ("explain page 12", None, 1.5)]
    assert found.page_fallback and found.candidates == ("c1",)
    assert found.index_version == "v7" and found.page_filter == (12, 12)


def test_explicit_page_range_overrides_and_rewrite_is_used(monkeypatch):
    calls = patch(monkeypatch, [["c"]])
    found = search_book(
        "it",
        book_stem="sicp",
        distance=1.0,
        page_range=(3, 4),
        retrieval_query="streams",
    )
    assert calls == [("streams", (3, 4), 1.0)] and not found.overview


def test_overview_keeps_original_wording_but_uses_preface_query(monkeypatch):
    for question in ("What this book about brocacho", "What's this textbook about?"):
        calls = patch(monkeypatch, [["c"]])
        found = search_book(question, book_stem="sicp", distance=1.5)
        assert found.overview
        assert calls[0][0] == config.BOOK_OVERVIEW_RETRIEVAL_QUERY


def test_qualified_and_page_questions_are_not_normalized(monkeypatch):
    calls = patch(monkeypatch, [["c"], ["c"]])
    assert not search_book(
        "What is this book's chapter 3 about recursion?", book_stem="sicp", distance=1.5
    ).overview
    assert not search_book(
        "What is page 4 about?", book_stem="sicp", distance=1.5
    ).overview
    assert calls[1][1] == (4, 4)


def test_empty_retrieval_still_traces_actual_query(monkeypatch):
    patch(monkeypatch, [[]])
    found = search_book("nothing here", book_stem="sicp", distance=1.5)
    assert found.candidates == () and found.retrieval_query == "nothing here"


def test_page_and_overview_queries_skip_gate(monkeypatch):
    monkeypatch.setattr(config, "ENTAILMENT_GATE_ENABLED", True)
    for question in ("explain page 12", "What is this book about?"):
        patch(monkeypatch, [["c"]])
        found = search_book(question, book_stem="sicp", distance=1.5)
        assert not found.gated and found.candidates == ("c",)


def test_gate_disabled_keeps_phase2_order(monkeypatch):
    patch(monkeypatch, [["z", "a"]])
    found = search_book("streams", book_stem="sicp", distance=1.5)
    assert not found.gated and found.dropped == 0 and found.candidates == ("z", "a")


def test_scorer_load_failure_is_visible(monkeypatch):
    import pytest
    from src import entailment

    patch(monkeypatch, [["c"]])
    monkeypatch.setattr(config, "ENTAILMENT_GATE_ENABLED", True)

    def fail():
        raise OSError("missing offline files")

    monkeypatch.setattr(entailment, "get_scorer", fail)
    with pytest.raises(
        RuntimeError, match="Entailment scorer unavailable: missing offline files"
    ):
        search_book("streams", book_stem="sicp", distance=1.5)


async def test_search_tool_reports_gate_counts(monkeypatch):
    from src.retrieval import BookSearchResult
    from src.tools.local import search_local_index
    from src.tools.protocol import ToolContext

    result = await search_local_index(
        {"query": "streams"},
        ToolContext("r", "sicp", "m", False, "local"),
        call_id="c",
        search=lambda *a, **kw: BookSearchResult(
            (), False, "v", False, "streams", False, None, True, 15, 0.1
        ),
    )
    assert result.data["gated"] and result.data["dropped"] == 15
