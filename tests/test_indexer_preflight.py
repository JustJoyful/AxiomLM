import pytest

from src import indexer


def test_cli_index_unloads_ollama_before_starting_ingest(monkeypatch):
    events = []

    class Client:
        def unload_and_wait(self):
            events.append("unloaded")

    monkeypatch.setattr(indexer, "OllamaClient", Client)
    monkeypatch.setattr(
        indexer,
        "index_book",
        lambda book, reindex=False, allow_cpu_embeddings=False: events.append("indexed")
        or 3,
    )
    assert indexer.main(["book"]) == 0
    assert events == ["unloaded", "indexed"]


def test_cli_index_stops_when_ollama_unload_cannot_be_verified(monkeypatch):
    events = []

    class Client:
        def unload_and_wait(self):
            raise RuntimeError("model remained loaded")

    monkeypatch.setattr(indexer, "OllamaClient", Client)
    monkeypatch.setattr(
        indexer, "index_book", lambda *args, **kwargs: events.append("indexed")
    )
    with pytest.raises(RuntimeError, match="remained loaded"):
        indexer.main(["book"])
    assert events == []
