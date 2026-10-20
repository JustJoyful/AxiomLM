"""Versioned dense/FTS5 pairs; a single durable manifest is the activation point."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src import config
from src.index_metadata import validate_book
from src.late_chunking import EmbeddedChunk


@dataclass(frozen=True)
class IndexHandle:
    book_stem: str
    version_id: str
    dense_collection: object
    bm25_path: Path | None
    legacy: bool
    query_prompt_policy: str
    embedding_signature: str


def embedding_config():
    return {
        "schema": config.PHASE2_INDEX_SCHEMA,
        "model": config.EMBED_MODEL,
        "revision": config.EMBED_REVISION,
        "task": "retrieval",
        "query_prompt": "Query: ",
        "document_prompt": "Document: ",
        "serialization": "Document prefix once;EOS per chunk;two newlines between chunks;v1",
        "pooling": config.PHASE2_POOLING_METHOD,
        "chunk_tokens": config.RETRIEVAL_CHUNK_TOKENS,
        "block_tokens": config.CONTEXT_BLOCK_MAX_TOKENS,
        "geometry": "paragraph_offsets_numbered_sicp_titles_v1",
        "query_policy": config.PHASE2_QUERY_POLICY,
        "dimension": config.EMBED_DIMENSION,
        "distance": "cosine",
    }


def identity(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def client_for(root):
    return chromadb.PersistentClient(
        path=str(root), settings=Settings(anonymized_telemetry=False)
    )


def manifest_path(root):
    return Path(root) / config.PHASE2_INDEX_MANIFEST_PATH.name


def read_manifest(root):
    path = manifest_path(root)
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if value.get("schema") != config.PHASE2_INDEX_SCHEMA or not isinstance(
        value.get("books"), dict
    ):
        raise ValueError("Unsupported active index manifest")
    return value["books"]


@contextmanager
def store_lock(root):
    Path(root).mkdir(parents=True, exist_ok=True)
    with (Path(root) / ".index-write.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def activate(root, books):
    path = manifest_path(root)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=root, prefix=".active-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                {"schema": config.PHASE2_INDEX_SCHEMA, "books": books},
                handle,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def pair_path(root, book, version):
    return (
        Path(root) / config.PHASE2_BM25_DIR.name / identity(book) / f"{version}.sqlite3"
    )


def write_dense(client, name, chunks, metadata):
    names = {c.name for c in client.list_collections()}
    if name in names:
        client.delete_collection(name)
    collection = client.create_collection(
        name,
        embedding_function=None,
        metadata={**metadata, "hnsw:space": "cosine", "axiom_index_status": "building"},
    )
    for start in range(0, len(chunks), config.INDEX_WRITE_BATCH_SIZE):
        batch = chunks[start : start + config.INDEX_WRITE_BATCH_SIZE]
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
            embeddings=[c.embedding for c in batch],
        )
    collection.modify(metadata={**metadata, "axiom_index_status": "ready"})
    return collection


def write_bm25(path, chunks):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".sqlite3", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with sqlite3.connect(temporary) as db:
            db.execute(
                "CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, document TEXT NOT NULL, metadata_json TEXT NOT NULL)"
            )
            db.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, document, tokenize='unicode61')"
            )
            db.executemany(
                "INSERT INTO chunks VALUES(?,?,?)",
                [
                    (c.chunk_id, c.text, json.dumps(c.metadata, sort_keys=True))
                    for c in chunks
                ],
            )
            db.executemany(
                "INSERT INTO chunks_fts VALUES(?,?)",
                [(c.chunk_id, c.text) for c in chunks],
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_pair(collection, path, expected_ids=None):
    if not path.is_file():
        raise RuntimeError("BM25 index missing; rebuild the book")
    if (collection.metadata or {}).get("axiom_index_status") != "ready":
        raise RuntimeError("Dense index incomplete")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        dense = set(collection.get(include=[])["ids"])
        sparse = {row[0] for row in db.execute("SELECT chunk_id FROM chunks")}
        fts = {row[0] for row in db.execute("SELECT chunk_id FROM chunks_fts")}
        db.execute("SELECT bm25(chunks_fts) FROM chunks_fts LIMIT 1").fetchall()
    if (
        not dense
        or dense != sparse
        or sparse != fts
        or (expected_ids is not None and dense != set(expected_ids))
    ):
        raise RuntimeError("Dense/BM25 chunk IDs do not match")


def resolve_index(book_stem: str, *, root: Path = config.VECTOR_STORE) -> IndexHandle:
    validate_book(book_stem)
    entry = read_manifest(root).get(book_stem)
    client = client_for(root)
    if entry is None:
        collection = client.get_collection(book_stem, embedding_function=None)
        if (collection.metadata or {}).get("axiom_index_status") == "building":
            raise RuntimeError("Legacy index incomplete; rebuild")
        return IndexHandle(
            book_stem,
            "legacy",
            collection,
            None,
            True,
            "legacy_document_query",
            "legacy",
        )
    signature = identity(embedding_config())
    if (
        entry.get("embedding_signature") != signature
        or entry.get("query_policy") != config.PHASE2_QUERY_POLICY
    ):
        raise RuntimeError(
            "Unsupported embedding signature/query policy; explicit rebuild required"
        )
    version = entry["version_id"]
    if (
        not isinstance(version, str)
        or len(version) != 64
        or any(c not in "0123456789abcdef" for c in version)
    ):
        raise ValueError("Invalid index version")
    expected = f"axiom_v2_{identity(book_stem)[:12]}_{version[:20]}"
    if entry["collection"] != expected:
        raise ValueError("Invalid active collection identity")
    collection = client.get_collection(expected, embedding_function=None)
    path = pair_path(root, book_stem, version)
    validate_pair(collection, path)
    return IndexHandle(
        book_stem,
        version,
        collection,
        path,
        False,
        config.PHASE2_QUERY_POLICY,
        signature,
    )


def build_index_version(
    book_stem: str, chunks: Sequence[EmbeddedChunk], *, root: Path = config.VECTOR_STORE
) -> IndexHandle:
    validate_book(book_stem)
    chunks = list(chunks)
    if not chunks or len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("Require nonempty unique chunks")
    import math

    if any(
        c.metadata.get("book_stem") != book_stem
        or len(c.embedding) != config.EMBED_DIMENSION
        or not all(math.isfinite(v) for v in c.embedding)
        for c in chunks
    ):
        raise ValueError("Invalid chunk book/vector")
    signature = identity(embedding_config())
    version = identity(
        {
            "config": embedding_config(),
            "sources": [(c.chunk_id, c.text, c.metadata) for c in chunks],
        }
    )
    name = f"axiom_v2_{identity(book_stem)[:12]}_{version[:20]}"
    with store_lock(root):
        books = read_manifest(root)
        client = client_for(root)
        existing = {c.name: c for c in client.list_collections()}
        if (
            name in existing
            and (existing[name].metadata or {}).get("axiom_index_status") == "ready"
        ):
            collection = client.get_collection(name, embedding_function=None)
        else:
            collection = write_dense(
                client,
                name,
                chunks,
                {
                    "book_stem": book_stem,
                    "version_id": version,
                    "embedding_signature": signature,
                },
            )
        path = pair_path(root, book_stem, version)
        try:
            validate_pair(collection, path, [c.chunk_id for c in chunks])
        except (RuntimeError, sqlite3.DatabaseError):
            write_bm25(path, chunks)
            validate_pair(collection, path, [c.chunk_id for c in chunks])
        books[book_stem] = {
            "version_id": version,
            "collection": name,
            "query_policy": config.PHASE2_QUERY_POLICY,
            "embedding_signature": signature,
            "config": embedding_config(),
        }
        activate(root, books)
    return resolve_index(book_stem, root=root)


def list_books(*, root: Path = config.VECTOR_STORE):
    books = set(read_manifest(root))
    for collection in client_for(root).list_collections():
        if not collection.name.startswith(
            "axiom_v2_"
        ) and not collection.name.startswith("axiom_staging_"):
            books.add(collection.name)
    return sorted(books)


def delete_book_indexes(book_stem: str, *, root: Path = config.VECTOR_STORE) -> bool:
    validate_book(book_stem)
    with store_lock(root):
        books = read_manifest(root)
        existed = book_stem in books
        books.pop(book_stem, None)
        activate(root, books)
        client = client_for(root)
        for collection in client.list_collections():
            if (
                collection.name == book_stem
                or (collection.metadata or {}).get("book_stem") == book_stem
            ):
                client.delete_collection(collection.name)
                existed = True
        directory = Path(root) / config.PHASE2_BM25_DIR.name / identity(book_stem)
        if directory.exists():
            for path in directory.glob("*.sqlite3"):
                path.unlink()
            directory.rmdir()
    return existed
