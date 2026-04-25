"""
indexer.py — Semantic Document Chunking & Indexing execution logic.

Retrieves aggregated markdown from the parsing stage, dynamically chunks it respecting
both character size bounds and markdown structural headers (h1, h2, h3), and
inserts it iteratively into the ChromaDB Local Vector store.
"""

import argparse
import sys

import chromadb

from src.config import PARSED_MD, VECTOR_STORE
from src.db import get_collection


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(VECTOR_STORE))


def _collection_doc_count(book_stem: str) -> int:
    client = _get_client()
    for collection in client.list_collections():
        if collection.name == book_stem:
            return collection.count()
    return 0


def _delete_collection_if_exists(book_stem: str) -> bool:
    client = _get_client()
    names = {collection.name for collection in client.list_collections()}
    if book_stem not in names:
        return False
    client.delete_collection(name=book_stem)
    return True


def index_book(book_stem: str, reindex: bool = False) -> int:
    """
    Parse a parsed full.md file, chunk it semantically using LangChain,
    and index it via embeddings into ChromaDB.
    
    Args:
        book_stem: The root stem of the book used as folder and collection identifier.
        reindex:   If True, delete existing collection before indexing.
        
    Returns:
        The total number of chunks embedded and stored in the database.
    """
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "Langchain text splitters not installed. Please run: pip install langchain langchain-community"
        ) from exc

    md_path = PARSED_MD / book_stem / "full.md"
    if not md_path.exists():
        print(f"Error: Could not find aggregated markdown at {md_path}")
        print(f"       Did you run the OCR stage for '{book_stem}'?")
        sys.exit(1)

    if reindex:
        if _delete_collection_if_exists(book_stem):
            print(f"Deleted existing collection '{book_stem}'. Re-indexing from scratch...")
        else:
            print(f"No existing collection named '{book_stem}' found. Creating a new one...")

    existing_count = _collection_doc_count(book_stem)
    if existing_count > 0:
        print(f"Error: Collection '{book_stem}' already contains {existing_count} chunks.")
        print("       Re-run with --reindex to replace the existing index.")
        sys.exit(1)

    print(f"Reading {md_path.name} ...")
    raw_text = md_path.read_text(encoding="utf-8")

    # 1. Structural Split (Preserve Chapters/Sections contextually)
    print("Chunking by structural headers...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_splits = markdown_splitter.split_text(raw_text)

    # 2. Constraints Split (Enforce token boundaries to avoid embedding explosion limits)
    print("Enforcing physical chunk size limit configurations...")
    # Max chunk chunk size limits embeddings footprint and context window space
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    final_splits = text_splitter.split_documents(header_splits)

    total_chunks = len(final_splits)
    print(f"Produced {total_chunks} chunk segments. Getting DB Connection...")

    # 3. Connect to DB and Push
    col = get_collection(book_stem)
    
    docs = []
    metadatas = []
    ids = []
    
    batches_inserted = 0
    batch_size = 100  # Avoid memory spikes during insertion map
    
    for idx, chunk in enumerate(final_splits):
        # We strip out empty metadata dicts, otherwise ChromaDB complains
        meta = chunk.metadata if chunk.metadata else {"source": book_stem}
        
        docs.append(chunk.page_content)
        metadatas.append(meta)
        ids.append(f"{book_stem}_chunk_{idx}")

        if len(docs) >= batch_size:
            print(f"    -> Adding batch {batches_inserted+1}...")
            col.add(documents=docs, metadatas=metadatas, ids=ids)
            batches_inserted += 1
            docs, metadatas, ids = [], [], []

    # Final flushed batch
    if docs:
        print(f"    -> Adding batch {batches_inserted+1}...")
        col.add(documents=docs, metadatas=metadatas, ids=ids)

    return total_chunks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AxiomLM Indexer — Chunk & Vectorize Markdown")
    parser.add_argument("book_stem", type=str, help="The book stem identifier (e.g. 'book_title_cleaned')")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete existing collection and re-index from scratch.",
    )
    return parser

if __name__ == "__main__":
    args = _build_parser().parse_args()
    print("=========================================")
    print(f" AxiomLM Semantic Indexer")
    print("=========================================")
    chunks = index_book(args.book_stem, reindex=args.reindex)
    print("=========================================")
    print(f" Success! Embedded {chunks} chunks for '{args.book_stem}' into vector database.")
    print("=========================================")
