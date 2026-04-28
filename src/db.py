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
import os
from typing import List

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

from src.config import VECTOR_STORE, EMBED_DEVICE, EMBED_MODEL

# ── Singletons ────────────────────────────────────────────────────────────────
_CLIENT: chromadb.PersistentClient | None = None
_MODEL:  SentenceTransformer | None = None


def _get_client() -> chromadb.PersistentClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=str(VECTOR_STORE))
    return _CLIENT


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model once and reuse across calls."""
    global _MODEL
    if _MODEL is None:
        requested_device = os.getenv("AXIOM_EMBED_DEVICE", EMBED_DEVICE).strip().lower()
        resolved_device = "cpu"
        if requested_device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    resolved_device = "cuda"
            except Exception:
                resolved_device = "cpu"
        elif requested_device == "auto":
            try:
                import torch
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                resolved_device = "cpu"
        elif requested_device in {"cpu", "mps"}:
            resolved_device = requested_device

        _MODEL = SentenceTransformer(
            EMBED_MODEL,
            trust_remote_code=True,
            device=resolved_device,
        )
    return _MODEL


# ── Custom ChromaDB-compatible embedding function ─────────────────────────────

class JinaV5DocumentEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB EmbeddingFunction that indexes chunks using jina-v5's
    'retrieval' task with the required 'Document: ' prefix.

    Used by the indexer when calling col.add(documents=...).
    """

    @staticmethod
    def name() -> str:
        # ChromaDB persists this name in collection metadata and validates
        # it on reconnect — must be stable across sessions.
        return "jina_v5_retrieval"

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        model = _get_model()
        prefixed = [f"Document: {text}" for text in input]
        vecs = model.encode(prefixed, task="retrieval", convert_to_numpy=True)
        return vecs.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single user query using the 'Query: ' asymmetric prefix.
    Called by the TUI/RAG layer — NOT by the indexer.
    """
    model = _get_model()
    vec = model.encode(f"Query: {query}", task="retrieval", convert_to_numpy=True)
    return vec.tolist()


# ── Collection helpers ────────────────────────────────────────────────────────

def get_collection(book_stem: str) -> chromadb.Collection:
    """Get or create the ChromaDB collection for a specific book."""
    client = _get_client()
    return client.get_or_create_collection(
        name=book_stem,
        embedding_function=JinaV5DocumentEmbeddingFunction(),
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
