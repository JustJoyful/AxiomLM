from src import tui


def test_configured_count_and_distance_reach_actual_retrieval(monkeypatch):
    app = tui.AxiomLMApp()
    received = []

    class Collection:
        def query(self, **kwargs):
            received.append(kwargs)
            return {
                "documents": [["keep", "drop"]],
                "metadatas": [[{"physical_page": 7}, {"physical_page": 9}]],
                "distances": [[1.7, 2.5]],
                "ids": [["keep-id", "drop-id"]],
            }

    monkeypatch.setattr(tui, "embed_query", lambda _: [1.0])
    monkeypatch.setattr(tui, "get_collection", lambda _: Collection())
    monkeypatch.setattr(tui, "ensure_collection_ready", lambda _: None)
    prompt, metadata, pages, ids = app._prepare_query(
        "Describe interpreter evaluation strategies using concrete evidence",
        "",
        "book",
        5,
        2.0,
    )
    assert received[0]["n_results"] == 5
    assert pages == [7] and ids == ["keep-id"]
    assert metadata == [{"physical_page": 7}]
    assert "keep" in prompt and "drop" not in prompt
