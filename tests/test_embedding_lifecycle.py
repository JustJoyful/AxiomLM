import numpy as np
import pytest
from src import db


@pytest.fixture(autouse=True)
def cached_snapshot(monkeypatch):
    monkeypatch.setattr(db, "_MODEL_SNAPSHOT", None)
    monkeypatch.setattr(db, "snapshot_download", lambda **_: "/cached/jina-snapshot")


class Encoder:
    def __init__(self, name, **kwargs):
        self.name = name
        self.device = kwargs["device"]
        self.options = kwargs
        self.inputs = []

    def encode(self, texts, **kwargs):
        self.inputs.append((texts, kwargs))
        return (
            np.array([1.0, 0.0])
            if isinstance(texts, str)
            else np.array([[1.0, 0.0] for _ in texts])
        )


def test_query_model_loads_the_exact_local_snapshot(monkeypatch):
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/cached/jina-snapshot"

    monkeypatch.setattr(db, "_QUERY_MODEL", None)
    monkeypatch.setattr(db, "snapshot_download", snapshot_download)
    monkeypatch.setattr(db, "SentenceTransformer", Encoder)

    model = db.get_query_model()

    assert calls == [{"repo_id": db.EMBED_MODEL, "revision": db.EMBED_REVISION}]
    assert model.name == "/cached/jina-snapshot"
    assert "revision" not in model.options


def require_split():
    assert callable(
        getattr(db, "get_query_model", None)
    ), "Separate model ownership is missing"


def test_query_stays_cpu_and_preserves_index_prompt_semantics(monkeypatch):
    require_split()
    monkeypatch.setattr(db, "_QUERY_MODEL", None)
    monkeypatch.setattr(db, "SentenceTransformer", Encoder)
    monkeypatch.setenv("AXIOM_EMBED_DEVICE", "cuda")
    assert db.embed_query("example") == [1.0, 0.0]
    model = db.get_query_model()
    assert model.device == "cpu"
    assert model.name == "/cached/jina-snapshot"
    assert model.inputs == [
        (
            "Query: example",
            {"task": "retrieval", "convert_to_numpy": True},
        )
    ]
    assert "revision" not in model.options


def test_ingest_requires_cuda_or_explicit_degradation(monkeypatch):
    require_split()
    import torch

    monkeypatch.setattr(db, "_INGEST_MODEL", None)
    monkeypatch.setattr(db, "SentenceTransformer", Encoder)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("AXIOM_INGEST_ALLOW_CPU", raising=False)
    with pytest.raises(RuntimeError, match="CUDA"):
        db.get_ingest_model()
    model = db.get_ingest_model(allow_cpu=True)
    assert model.device == "cpu"
    db.release_ingest_model()


def test_ingest_rejects_insufficient_free_vram(monkeypatch):
    require_split()
    import torch

    monkeypatch.setattr(db, "_INGEST_MODEL", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1024**3, 6 * 1024**3))
    with pytest.raises(RuntimeError, match="free VRAM"):
        db.get_ingest_model()


def test_gpu_documents_cleanup_order_and_query_survives(monkeypatch):
    require_split()
    import torch

    monkeypatch.setattr(db, "_QUERY_MODEL", None)
    monkeypatch.setattr(db, "_MODEL_SNAPSHOT", None)
    monkeypatch.setattr(db, "_INGEST_MODEL", None)
    monkeypatch.setattr(db, "SentenceTransformer", Encoder)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (4 * 1024**3, 6 * 1024**3))
    query = db.get_query_model()
    np.testing.assert_allclose(
        db.JinaV5DocumentEmbeddingFunction()(["hello"])[0], [1.0, 0.0]
    )
    model = db.get_ingest_model()
    assert model.device == "cuda"
    assert model.inputs[0][0] == ["Document: hello"]
    assert "prompt" not in model.inputs[0][1]
    events = []

    def collect():
        assert db._INGEST_MODEL is None
        events.append("gc")

    monkeypatch.setattr(db.gc, "collect", collect)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("cache"))
    del model
    db.release_ingest_model()
    assert events == ["gc", "cache"]
    assert db.get_query_model() is query
    db.release_ingest_model()
    assert db._INGEST_MODEL is None


def test_index_failure_releases_ingest_model(monkeypatch):
    require_split()
    from src import indexer

    events = []

    class Collection:
        metadata = {}

        def modify(self, **kwargs):
            events.append(kwargs["metadata"]["axiom_index_status"])

        def add(self, **kwargs):
            raise RuntimeError("encode failed")

    monkeypatch.setattr(indexer, "get_collection", lambda *a, **kw: Collection())
    monkeypatch.setattr(
        indexer, "release_ingest_model", lambda: events.append("released")
    )
    with pytest.raises(RuntimeError, match="encode failed"):
        indexer.index_chunks_with_pages("test", [("hello", 1, {})])
    assert events == ["building", "released"]


def test_existing_collection_lookup_does_not_create(monkeypatch):
    calls = []

    class Client:
        def get_collection(self, **kwargs):
            calls.append(("get", kwargs))
            return "existing"

        def get_or_create_collection(self, **kwargs):
            calls.append(("get_or_create", kwargs))
            raise AssertionError("read-only lookup must not create a collection")

    monkeypatch.setattr(db, "_get_client", lambda: Client())
    assert db.get_existing_collection("book") == "existing"
    assert calls == [("get", {"name": "book"})]


def test_partial_index_is_rejected_for_chat():
    class Collection:
        metadata = {"axiom_index_status": "building"}

    with pytest.raises(RuntimeError, match="incomplete"):
        db.ensure_collection_ready(Collection())
