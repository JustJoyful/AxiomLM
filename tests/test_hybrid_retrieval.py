import pytest

from src import hybrid_retrieval as hybrid


def hit(cid, backend, score=0.0, text=None, meta=None):
    return hybrid.SearchHit(
        cid, text or cid, meta or {"physical_page": 1}, backend, 1, score
    )


def test_merge_interleaves_without_comparing_scales_and_keeps_both_ranks():
    dense = [hit("a", "dense", 0.1), hit("b", "dense", 0.1), hit("c", "dense", 0.3)]
    sparse = [hit("b", "bm25", -10000), hit("d", "bm25", -10)]
    rows = hybrid.merge_ranked_candidates(dense, sparse, 3)
    assert [c.chunk_id for c in rows] == ["a", "b", "d"]
    assert rows[1].dense_rank == 2 and rows[1].bm25_rank == 1
    with pytest.raises(ValueError):
        hybrid.merge_ranked_candidates(dense, [hit("a", "bm25", text="conflict")])
    assert not hybrid.merge_ranked_candidates([], [])


@pytest.mark.parametrize(
    "cue",
    [
        "overview",
        "SUMMARIZE",
        "compare",
        "contrast",
        "relationship",
        "across chapters",
        "how do closures relate to objects",
        "what is this book about",
    ],
)
def test_broad_context_fits_without_cutting_passages(cue):
    rows = hybrid.merge_ranked_candidates(
        [hit(str(i), "dense", i, text="evidence") for i in range(12)], []
    )
    chosen = hybrid.select_context(
        cue, rows, 15, lambda cs: 10 + sum(len(c.text) for c in cs), 100
    )
    assert len(chosen) == 8 and all(c.text == "evidence" for c in chosen)
    assert len(hybrid.select_context(cue, rows[:3], 5, lambda cs: len(cs), 20)) == 3
    with pytest.raises(ValueError, match="budget"):
        hybrid.select_context(cue, rows, 8, lambda cs: 1000 * len(cs), 20)


def test_normal_question_obeys_k_and_long_history_is_counted():
    rows = hybrid.merge_ranked_candidates(
        [hit(str(i), "dense", i) for i in range(8)], []
    )
    assert (
        len(hybrid.select_context("Define closure", rows, 3, lambda cs: len(cs), 10))
        == 3
    )
    with pytest.raises(ValueError):
        hybrid.select_context("question", rows, 3, lambda cs: 1000 + len(cs), 10)


def test_page_sets_preserve_gaps_and_legacy_fallback():
    from src.page_querying import physical_pages_for

    assert physical_pages_for({"physical_pages": "[2,4]", "physical_page": 2}) == (2, 4)
    assert physical_pages_for({"page": 7}) == (7,)
    with pytest.raises(ValueError):
        physical_pages_for({"physical_pages": "[0]"})


def test_real_pair_filter_refill_and_index_policy_precede_each_embedding(
    tmp_path, monkeypatch
):
    from src import db, index_store, tui
    from src.late_chunking import EmbeddedChunk

    rows = [
        EmbeddedChunk(
            str(i),
            "arithmetic evidence " + str(i),
            {
                "book_stem": "book",
                "physical_page": i + 1,
                "physical_page_start": i + 1,
                "physical_page_end": i + 2,
                "physical_pages": f"[{i+1},{i+2}]",
            },
            [1.0] + [0.0] * 767,
        )
        for i in range(20)
    ]
    index_store.build_index_version("book", rows, root=tmp_path)
    legacy = index_store.client_for(tmp_path).create_collection(
        "oldbook", embedding_function=None
    )
    legacy.add(
        ids=["old"],
        documents=["old evidence"],
        embeddings=[[1.0] + [0.0] * 767],
        metadatas=[{"physical_page": 1}],
    )
    events = []
    operation = index_store.resolve_index

    def resolve(book):
        events.append(("resolve", book))
        return operation(book, root=tmp_path)

    def embed(query, *, prompt_policy):
        assert events[-1][0] == "resolve"
        events.append(("embed", prompt_policy))
        return [1.0] + [0.0] * 767

    monkeypatch.setattr(index_store, "resolve_index", resolve)
    monkeypatch.setattr(db, "embed_query", embed)
    app = tui.AxiomLMApp()
    for book in ["book", "oldbook", "book"]:
        prompt, metadata, pages, ids = app._prepare_query(
            "Define arithmetic",
            "history",
            book,
            3,
            1.5,
            "standalone arithmetic",
            answer_token_counter=lambda p: len(p) // 4,
        )
        assert "Current question: Define arithmetic" in prompt
        assert metadata and ids and pages
    assert [v for kind, v in events if kind == "embed"] == [
        "named_retrieval_query",
        "legacy_document_query",
        "named_retrieval_query",
    ]
    handle = operation("book", root=tmp_path)
    events.append(("resolve", "book"))
    selected = hybrid.retrieve_hybrid(
        handle, "arithmetic", (5, 5), 2, 2, candidate_limit=5
    )
    assert {c.chunk_id for c in selected} == {"3", "4"}
    events.append(("resolve", "book"))
    filled = hybrid.retrieve_hybrid(
        handle, "arithmetic", None, 2, 2, candidate_limit=15
    )
    assert len(filled) == 15
