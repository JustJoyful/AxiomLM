from scripts import validate_foundation
from src import tui


def test_page_fallback_is_safe_without_a_running_app(monkeypatch):
    app = tui.AxiomLMApp()

    class Collection:
        def query(self, **kwargs):
            if "where" in kwargs:
                return {"documents": [[]]}
            return {
                "documents": [["fallback evidence"]],
                "metadatas": [[{"physical_page": 7}]],
                "distances": [[0.1]],
                "ids": [["chunk"]],
            }

    monkeypatch.setattr(tui, "get_collection", lambda _: Collection())
    monkeypatch.setattr(tui, "ensure_collection_ready", lambda _: None)
    monkeypatch.setattr(tui, "embed_query", lambda _: [0.1])
    prompt, _, pages, _ = app._prepare_legacy_query(
        "Explain page 999", "", "book", 5, 1.5
    )
    assert "fallback evidence" in prompt
    assert pages == [7]


def test_followup_uses_rewritten_query_results_in_validator(monkeypatch):
    embedded_queries = []
    retrieval_calls = []

    class Collection:
        def query(self, **kwargs):
            retrieval_calls.append(kwargs)
            index = len(retrieval_calls)
            return {
                "documents": [[f"Fresh passage {index}"]],
                "metadatas": [
                    [
                        {
                            "Header 1": "Chapter 1",
                            "Header 2": "Examples",
                            "physical_page": 35,
                        }
                    ]
                ],
                "distances": [[0.2]],
                "ids": [[f"chunk_{index}"]],
            }

    class Client:
        def generate(self, *_args, **_kwargs):
            return {"response": "Evidence-based answer."}

    monkeypatch.setattr(
        validate_foundation.db, "get_existing_collection", lambda _book: Collection()
    )
    monkeypatch.setattr(
        validate_foundation.db, "ensure_collection_ready", lambda _collection: None
    )

    def embed_query(query):
        embedded_queries.append(query)
        return [0.1]

    monkeypatch.setattr(validate_foundation.db, "embed_query", embed_query)
    monkeypatch.setattr(validate_foundation, "OllamaClient", Client)

    rows = validate_foundation._run_after_baseline("sicp")

    assert len(embedded_queries) == 5
    assert (
        embedded_queries[1]
        == "What is an interpreter? illustrative examples exercises questions problems"
    )
    assert rows[1]["evidence_ids"] == ["chunk_2"]
    assert rows[1]["evidence_metadata"]
    assert len(retrieval_calls) == 5


def test_followup_uses_rewritten_query_results_instead_of_cached_chunks(monkeypatch):
    app = tui.AxiomLMApp()
    app._conversation = [
        {"role": "user", "content": "What is an interpreter?"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    app._last_chunks = ["STALE CACHED CONTEXT"]
    app._last_retrieved_pages = [355]
    app._sync_tuning_panel = lambda: None
    app.distance_threshold = 1.5
    app.top_k_results = 8
    app._is_query_cancelled = lambda _request_id: False
    calls = {}

    class Collection:
        def query(self, **kwargs):
            calls["retrieval"] = kwargs
            return {
                "documents": [["FRESH EXAMPLE EVIDENCE"]],
                "metadatas": [
                    [
                        {
                            "Header 1": "Chapter 1",
                            "Header 2": "Examples",
                            "physical_page": 35,
                        }
                    ]
                ],
                "distances": [[0.2]],
                "ids": [["sicp_chunk_00000036"]],
            }

    def embed_query(query):
        calls["query"] = query
        return [0.1]

    monkeypatch.setattr(tui, "embed_query", embed_query)
    monkeypatch.setattr(tui, "get_collection", lambda _book: Collection())
    monkeypatch.setattr(tui, "ensure_collection_ready", lambda _collection: None)

    class Stream:
        def __iter__(self):
            yield {"response": "Grounded example."}
            yield {"done": True, "eval_count": 3, "eval_duration": 1_000_000_000}

        def close(self):
            pass

    class Client:
        def generate(self, prompt, **_kwargs):
            calls["prompt"] = prompt
            return Stream()

    monkeypatch.setattr(tui, "OllamaClient", Client)
    app.call_from_thread = lambda *args, **_kwargs: calls.setdefault(
        "callbacks", []
    ).append(args)

    app.execute_rag_query(
        "Give another example of that.",
        "Conversation so far:\nUser: What is an interpreter?\n",
        "sicp",
        "qwen3.5:4b-q4_K_M",
        "request-1",
    )

    assert (
        calls["query"]
        == "What is an interpreter? illustrative examples exercises questions problems"
    )
    assert calls["retrieval"]["n_results"] == 8
    assert "prompt" in calls, calls.get("callbacks")
    assert "FRESH EXAMPLE EVIDENCE" in calls["prompt"]
    assert "STALE CACHED CONTEXT" not in calls["prompt"]
