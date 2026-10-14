"""
db.py — Local ChromaDB and Embeddings Integration.

Manages connection to our persistent Chroma vector database, explicitly binding
efficient local embeddings directly to the CPU to preserve VRAM for LLM usage.

NOTE on jina-embeddings-v5-text-nano:
  This is a task-aware model — every encode() call MUST specify a task.
  ChromaDB's generic SentenceTransformerEmbeddingFunction has no hook for this,
  so we use a custom EmbeddingFunction subclass instead.

  Asymmetric retrieval prefixes (per Jina docs):
    - Indexing   → task="retrieval", prefix="Document: "
    - Querying   → task="retrieval", prefix="Query: "
"""

from __future__ import annotations
import gc
from typing import List

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBED_MODEL,
    EMBED_REVISION,
    INGEST_EMBED_DEVICE,
    INGEST_MIN_FREE_VRAM_MB,
    QUERY_EMBED_DEVICE,
    VECTOR_STORE,
)

# ── Singletons ────────────────────────────────────────────────────────────────
_CLIENT: chromadb.PersistentClient | None = None
_QUERY_MODEL: SentenceTransformer | None = None
_INGEST_MODEL: SentenceTransformer | None = None
_MODEL_SNAPSHOT: str | None = None


def _model_snapshot() -> str:
    global _MODEL_SNAPSHOT
    if _MODEL_SNAPSHOT is None:
        _MODEL_SNAPSHOT = snapshot_download(
            repo_id=EMBED_MODEL,
            revision=EMBED_REVISION,
        )
    return _MODEL_SNAPSHOT


def _get_client() -> chromadb.PersistentClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=str(VECTOR_STORE))
    return _CLIENT


def get_query_model() -> SentenceTransformer:
    """Return the live-query encoder, always resident on CPU."""
    global _QUERY_MODEL
    if _QUERY_MODEL is None:
        _QUERY_MODEL = SentenceTransformer(
            _model_snapshot(),
            trust_remote_code=True,
            device=QUERY_EMBED_DEVICE,
        )
    return _QUERY_MODEL


def get_ingest_model(*, allow_cpu: bool = False) -> SentenceTransformer:
    """Return the short-lived ingestion encoder on the configured device."""
    global _INGEST_MODEL
    if _INGEST_MODEL is not None:
        return _INGEST_MODEL

    requested_device = "cpu" if allow_cpu else INGEST_EMBED_DEVICE
    if requested_device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable for embedding ingestion. Select the explicit "
                "CPU/degraded ingest option to continue without GPU acceleration."
            )
        else:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            except Exception as exc:
                raise RuntimeError(f"Unable to inspect free CUDA VRAM before ingest: {exc}") from exc
            free_mb = free_bytes // (1024 * 1024)
            total_mb = total_bytes // (1024 * 1024)
            print(f"Embedding ingest CUDA memory: {free_mb} MiB free / {total_mb} MiB total")
            if free_mb < INGEST_MIN_FREE_VRAM_MB:
                raise RuntimeError(
                    f"Embedding ingest needs at least {INGEST_MIN_FREE_VRAM_MB} MiB free VRAM; "
                    f"only {free_mb} MiB is available. Unload GPU workloads or select explicit CPU ingest."
                )

    _INGEST_MODEL = SentenceTransformer(
        _model_snapshot(),
        trust_remote_code=True,
        device=requested_device,
    )
    print(f"Embedding ingest device: {requested_device}; revision: {EMBED_REVISION}")
    return _INGEST_MODEL


def release_ingest_model() -> None:
    """Drop every module-owned ingest-model reference before chat resumes."""
    global _INGEST_MODEL
    model = _INGEST_MODEL
    _INGEST_MODEL = None
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ── Custom ChromaDB-compatible embedding function ─────────────────────────────

class JinaV5DocumentEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB EmbeddingFunction that indexes chunks using jina-v5's
    'retrieval' task with the required 'Document: ' prefix.

    Used by the indexer when calling col.add(documents=...).
    """

    def __init__(self, *, allow_cpu: bool = False) -> None:
        self.allow_cpu = allow_cpu

    @staticmethod
    def name() -> str:
        # ChromaDB persists this name in collection metadata and validates
        # it on reconnect — must be stable across sessions.
        return "jina_v5_retrieval"

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        model = get_ingest_model(allow_cpu=self.allow_cpu)
        prefixed = [f"Document: {text}" for text in input]
        # Preserve the pinned model's configured prompt semantics while Phase 0
        # leaves existing collection vectors untouched. Prompt normalization
        # requires the explicit reindex migration planned with Phase 2.
        vecs = model.encode(prefixed, task="retrieval", convert_to_numpy=True)
        return vecs.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single user query using the 'Query: ' asymmetric prefix.
    Called by the TUI/RAG layer — NOT by the indexer.
    """
    model = get_query_model()
    # Keep query-side prompt behavior aligned with collections built before
    # Phase 0; changing it without rebuilding those vectors would skew search.
    vec = model.encode(f"Query: {query}", task="retrieval", convert_to_numpy=True)
    return vec.tolist()


# ── Collection helpers ────────────────────────────────────────────────────────

def get_collection(book_stem: str, *, allow_cpu_ingest: bool = False) -> chromadb.Collection:
    """Get or create the ChromaDB collection for a specific book."""
    client = _get_client()
    return client.get_or_create_collection(
        name=book_stem,
        embedding_function=JinaV5DocumentEmbeddingFunction(allow_cpu=allow_cpu_ingest),
    )


def get_existing_collection(book_stem: str) -> chromadb.Collection:
    """Read an existing book collection without creating a new index."""
    return _get_client().get_collection(name=book_stem)


def ensure_collection_ready(collection: chromadb.Collection) -> None:
    """Prevent chat from reading a collection left partial by a failed ingest."""
    metadata = collection.metadata or {}
    status = metadata.get("axiom_index_status")
    if status == "building":
        raise RuntimeError(
            "This book's index is incomplete after an interrupted or failed ingest. "
            "Re-run Parse + Index before asking questions."
        )


def list_collections() -> list[str]:
    """List all book stems (collections) currently in the database."""
    client = _get_client()
    return [col.name for col in client.list_collections()]


def delete_collection(book_stem: str) -> bool:
    """
    Delete a ChromaDB collection by its book stem.

    Returns True when deletion succeeds, False when the collection doesn't exist.
    Raises RuntimeError for other failures.
    """
    client = _get_client()
    try:
        client.delete_collection(name=book_stem)
        return True
    except chromadb.errors.NotFoundError:
        return False
    except Exception as exc:  # pragma: no cover - defensive pass-through for Chroma internals
        raise RuntimeError(f"Failed to delete collection '{book_stem}': {exc}") from exc
