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


def test_rrf_rewards_shared_hit_without_changing_sources():
    meta = {
        "physical_pages": "[2, 3]",
        "physical_page_start": 2,
        "physical_page_end": 3,
    }
    dense = [hit("a", "dense", 0.1), hit("b", "dense", 0.2, meta=meta)]
    sparse = [hit("b", "bm25", -10, meta=meta), hit("c", "bm25", -5)]
    result = hybrid.merge_ranked_candidates(dense, sparse, policy="rrf")
    assert [c.chunk_id for c in result] == ["b", "a", "c"]
    assert result[0].metadata is meta and result[0].text == "b"
    assert result[0].dense_rank == 2 and result[0].bm25_rank == 1


def test_rrf_ties_use_chunk_id():
    result = hybrid.merge_ranked_candidates(
        [hit("z", "dense")], [hit("a", "bm25")], policy="rrf"
    )
    assert [c.chunk_id for c in result] == ["a", "z"]


def test_conflicting_shared_ids_still_fail_for_rrf():
    with pytest.raises(ValueError, match="Conflicting"):
        hybrid.merge_ranked_candidates(
            [hit("a", "dense")], [hit("a", "bm25", text="changed")], policy="rrf"
        )


def test_unknown_policy_and_invalid_rrf_constant_fail(monkeypatch):
    with pytest.raises(ValueError, match="policy"):
        hybrid.merge_ranked_candidates([], [], policy="guess")
    monkeypatch.setattr(hybrid.config, "HYBRID_RRF_K", 0)
    with pytest.raises(ValueError, match="RRF"):
        hybrid.merge_ranked_candidates([], [], policy="rrf")


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


@pytest.mark.parametrize("policy", ["interleave", "rrf"])
def test_real_pair_filter_refill_and_index_policy_precede_each_embedding(
    tmp_path, monkeypatch, policy
):
    from src import db, index_store, tui
    from src.late_chunking import EmbeddedChunk

    monkeypatch.setattr(hybrid.config, "HYBRID_MERGE_POLICY", policy)

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
    diagnostic = {}
    selected = hybrid.retrieve_hybrid(
        handle, "arithmetic", (5, 5), 2, 2, candidate_limit=5, diagnostics=diagnostic
    )
    assert {c.chunk_id for c in selected} == {"3", "4"}
    assert diagnostic["page_filter"] == (5, 5)
    assert diagnostic["policy"] == policy
    assert {h["chunk_id"] for h in diagnostic["raw_dense"]} == {"3", "4"}
    assert {h["metadata"]["physical_pages"] for h in diagnostic["merged"]} == {
        "[4,5]",
        "[5,6]",
    }
    events.append(("resolve", "book"))
    filled = hybrid.retrieve_hybrid(
        handle, "arithmetic", None, 2, 2, candidate_limit=15
    )
    assert len(filled) == 15


@pytest.mark.parametrize(
    "question",
    [
        "What this book about brocacho",
        "What's this textbook about?",
        "What is this book about?",
    ],
)
def test_informal_book_overviews_receive_broad_context(question):
    assert hybrid.is_broad_question(question)


def test_overview_search_and_answer_keep_original_wording(tmp_path, monkeypatch):
    from src import db, index_store, tui
    from src.late_chunking import EmbeddedChunk

    rows = [
        EmbeddedChunk(
            str(i),
            "The book introduces methods for controlling complexity.",
            {
                "book_stem": "book",
                "physical_page": i + 1,
                "physical_page_start": i + 1,
                "physical_page_end": i + 1,
                "physical_pages": f"[{i+1}]",
            },
            [1.0] + [0.0] * 767,
        )
        for i in range(6)
    ]
    index_store.build_index_version("book", rows, root=tmp_path)
    resolve = index_store.resolve_index
    monkeypatch.setattr(
        index_store, "resolve_index", lambda book: resolve(book, root=tmp_path)
    )
    queries = []

    def embed(query, *, prompt_policy):
        queries.append(query)
        return [1.0] + [0.0] * 767

    monkeypatch.setattr(db, "embed_query", embed)
    question = "What this book about brocacho"
    prompt, metadata, _pages, ids = tui.AxiomLMApp()._prepare_query(
        question, "", "book", 3, 1.5, answer_token_counter=lambda p: len(p) // 4
    )
    assert len(ids) == 5
    assert "preface" in queries[0] and "purpose" in queries[0]
    assert "Original user wording: What this book about brocacho" in prompt
    assert "Current question: What is this book about?" in prompt
    assert all(m["retrieval_query"] == queries[0] for m in metadata)


@pytest.mark.parametrize(
    "question",
    ["What is this book about quantum mechanics?", "Summarize physical PDF page 1."],
)
def test_overview_intent_does_not_erase_qualifiers_or_page_requests(question):
    assert not hybrid.is_book_overview(question)


def test_gate_drops_below_threshold_and_ranks_by_score():
    candidates = hybrid.merge_ranked_candidates(
        [hit(c, "dense", i) for i, c in enumerate("abc")], []
    )
    rows, dropped = hybrid.gate_and_rank(
        "claim", candidates, lambda pairs: [0.2, 0.9, 0.6], 0.5
    )
    assert [c.chunk_id for c in rows] == ["b", "c"]
    assert dropped == 1 and rows[0].entailment == 0.9


def test_gate_breaks_ties_by_chunk_id():
    candidates = hybrid.merge_ranked_candidates(
        [hit("z", "dense", 0), hit("a", "dense", 1)], []
    )
    rows, _ = hybrid.gate_and_rank("claim", candidates, lambda pairs: [0.9, 0.9], 0.5)
    assert [c.chunk_id for c in rows] == ["a", "z"]


def test_gate_scores_in_one_call():
    candidates = hybrid.merge_ranked_candidates(
        [hit(str(i), "dense", i) for i in range(15)], []
    )
    calls = []

    def score(pairs):
        calls.append(pairs)
        return [0.0] * len(pairs)

    assert hybrid.gate_and_rank("claim", candidates, score, 0.5) == ([], 15)
    assert calls == [[(c.text, "claim") for c in candidates]]


def test_all_dropped_returns_empty_not_original():
    candidates = hybrid.merge_ranked_candidates([hit("a", "dense")], [])
    assert hybrid.gate_and_rank("claim", candidates, lambda pairs: [0.1], 0.5) == (
        [],
        1,
    )


def test_shipping_merge_default_is_interleave():
    import subprocess
    import sys

    # Read the installed default in a fresh process, outside patched fixtures.
    result = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from src.config import HYBRID_MERGE_POLICY; print(HYBRID_MERGE_POLICY)",
        ],
        text=True,
    )
    assert result.strip() == "interleave"
