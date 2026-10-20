import sqlite3

import pytest

from src import index_store as store
from src.late_chunking import EmbeddedChunk


def chunks(text="one"):
    return [
        EmbeddedChunk(
            "id-one",
            text,
            {"book_stem": "book", "physical_page": 1, "physical_pages": "[1]"},
            [1.0] + [0.0] * 767,
        )
    ]


def test_rebuild_identity_pair_ids_and_legacy_stay_available(tmp_path):
    client = store.client_for(tmp_path)
    legacy = client.create_collection("book", embedding_function=None)
    legacy.add(ids=["old"], documents=["legacy"], embeddings=[[1.0] + [0.0] * 767])
    old = store.resolve_index("book", root=tmp_path)
    assert old.legacy and old.query_prompt_policy == "legacy_document_query"
    first = store.build_index_version("book", chunks(), root=tmp_path)
    again = store.build_index_version("book", chunks(), root=tmp_path)
    assert first.version_id == again.version_id
    assert first.dense_collection.get()["ids"] == ["id-one"]
    with sqlite3.connect(first.bm25_path) as connection:
        assert connection.execute("select chunk_id from chunks").fetchall() == [
            ("id-one",)
        ]
    assert legacy.get()["documents"] == ["legacy"]
    assert store.resolve_index("book", root=tmp_path).version_id == first.version_id
    assert (
        store.build_index_version("book", chunks("changed"), root=tmp_path).version_id
        != first.version_id
    )


@pytest.mark.parametrize("failure", ["dense", "sqlite", "activation", "interrupt"])
def test_every_failure_preserves_old_pair(tmp_path, monkeypatch, failure):
    old = store.build_index_version("book", chunks(), root=tmp_path)
    before = (tmp_path / "active_indexes.json").read_bytes()

    def fail(*args, **kwargs):
        raise (
            KeyboardInterrupt
            if failure == "interrupt"
            else OSError("injected " + failure)
        )

    monkeypatch.setattr(
        store,
        {
            "dense": "write_dense",
            "sqlite": "write_bm25",
            "activation": "activate",
            "interrupt": "write_bm25",
        }[failure],
        fail,
    )
    with pytest.raises((OSError, KeyboardInterrupt)):
        store.build_index_version("book", chunks("new"), root=tmp_path)
    assert (tmp_path / "active_indexes.json").read_bytes() == before
    assert store.resolve_index("book", root=tmp_path).version_id == old.version_id
    assert old.dense_collection.get()["documents"] == ["one"]


def test_corrupt_or_missing_pair_is_visible_and_staging_hidden(tmp_path):
    handle = store.build_index_version("book", chunks(), root=tmp_path)
    store.client_for(tmp_path).create_collection(
        "axiom_staging_orphan", embedding_function=None
    )
    assert store.list_books(root=tmp_path) == ["book"]
    handle.bm25_path.unlink()
    with pytest.raises(RuntimeError, match="BM25"):
        store.resolve_index("book", root=tmp_path)


def test_old_index_visible_until_real_manifest_replace_and_replace_failure_safe(
    tmp_path, monkeypatch
):
    old = store.build_index_version("book", chunks(), root=tmp_path)
    original = store.write_bm25
    seen = []

    def observe(path, rows):
        seen.append(store.resolve_index("book", root=tmp_path).version_id)
        return original(path, rows)

    monkeypatch.setattr(store, "write_bm25", observe)
    original_replace = store.os.replace

    def fail_replace(source, destination):
        if Path(destination).name == "active_indexes.json":
            raise OSError("manifest replace denied")
        return original_replace(source, destination)

    from pathlib import Path

    monkeypatch.setattr(store.os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.build_index_version("book", chunks("changed"), root=tmp_path)
    assert seen == [old.version_id]
    assert store.resolve_index("book", root=tmp_path).version_id == old.version_id


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_identical_rebuild_repairs_sparse_index_without_replacing_dense(
    tmp_path, damage
):
    first = store.build_index_version("book", chunks(), root=tmp_path)
    before = (tmp_path / "active_indexes.json").read_bytes()
    dense_before = first.dense_collection.get(include=["documents"])
    if damage == "missing":
        first.bm25_path.unlink()
    else:
        first.bm25_path.write_bytes(b"not a sqlite database")
    with pytest.raises((RuntimeError, sqlite3.DatabaseError)):
        store.resolve_index("book", root=tmp_path)
    repaired = store.build_index_version("book", chunks(), root=tmp_path)
    assert repaired.version_id == first.version_id
    assert repaired.dense_collection.id == first.dense_collection.id
    assert repaired.dense_collection.get(include=["documents"]) == dense_before
    assert (tmp_path / "active_indexes.json").read_bytes() == before
    store.validate_pair(repaired.dense_collection, repaired.bm25_path)
