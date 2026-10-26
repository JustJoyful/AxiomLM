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
