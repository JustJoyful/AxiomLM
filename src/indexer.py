"""
indexer.py — Semantic Document Chunking & Indexing execution logic.

Retrieves aggregated markdown from the parsing stage, dynamically chunks it respecting
both character size bounds and markdown structural headers (h1, h2, h3), and
inserts it iteratively into the ChromaDB Local Vector store.
"""

import argparse
import json
import re
import sys

import chromadb

from src import checkpoint
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, PARSED_MD, VECTOR_STORE
from src.db import get_collection

TOKEN_TO_WORD_RATIO = 0.75


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


def _estimate_tokens(text: str) -> int:
    """Approximate token count using whitespace word count."""
    return len(text.split())


def _split_long_unit(unit: str, target_size: int) -> list[str]:
    """
    Split oversized unit by words only when semantic split is impossible.
    Keeps implementation safe for very long paragraphs/code blocks.
    """
    words = unit.split()
    if not words:
        return []

    parts: list[str] = []
    for start in range(0, len(words), target_size):
        part = " ".join(words[start:start + target_size]).strip()
        if part:
            parts.append(part)
    return parts


def _tokens_to_overlap_words(token_count: int) -> int:
    """Convert configured token overlap into approximate word overlap."""
    if token_count <= 0:
        return 0
    return max(1, int(round(token_count * TOKEN_TO_WORD_RATIO)))


def _next_content_index(text: str, start: int) -> int:
    """Advance to next content character, skipping whitespace and closers."""
    skip_chars = set(" \t\r\n\"'”’)]}")
    idx = start
    while idx < len(text) and text[idx] in skip_chars:
        idx += 1
    return idx


def _sentence_split(paragraph: str) -> list[str]:
    """Split paragraph into sentence-like units with basic abbreviation guards."""
    text = paragraph.strip()
    if not text:
        return []

    abbreviations = {"e.g.", "i.e.", "etc.", "mr.", "mrs.", "dr.", "vs.", "fig.", "eq.", "no."}
    boundaries: list[int] = []

    for idx, ch in enumerate(text):
        if ch not in ".!?":
            continue

        next_idx = _next_content_index(text, idx + 1)
        if next_idx >= len(text):
            boundaries.append(idx + 1)
            continue

        prev_char = text[idx - 1] if idx > 0 else ""
        next_char = text[next_idx]
        if ch == ".":
            if prev_char.isdigit() and next_char.isdigit():
                continue

            token_start = idx
            while token_start > 0 and (text[token_start - 1].isalpha() or text[token_start - 1] == "."):
                token_start -= 1
            token = text[token_start:idx + 1].lower()
            if token in abbreviations:
                continue

        if next_char.islower():
            continue

        boundaries.append(idx + 1)

    if not boundaries:
        return [text]

    sentences: list[str] = []
    start = 0
    for end in boundaries:
        segment = text[start:end].strip()
        if segment:
            sentences.append(segment)
        start = end

    trailing = text[start:].strip()
    if trailing:
        sentences.append(trailing)

    return sentences if sentences else [text]


def _flush_prose_units(lines: list[str], units: list[str], target_size: int) -> None:
    """Convert buffered prose lines into paragraph/sentence units."""
    prose = "\n".join(lines).strip()
    if not prose:
        return

    for paragraph in re.split(r"\n\s*\n", prose):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if _estimate_tokens(paragraph) <= target_size:
            units.append(paragraph)
        else:
            units.extend(_sentence_split(paragraph))


def chunk_with_boundaries(
    text: str,
    target_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Chunk text while prioritizing semantic boundaries.

    Strategy:
    1. Preserve fenced code blocks as atomic units.
    2. Split prose by paragraphs, then sentence boundaries.
    3. Build chunks near target_size without splitting sentences when possible.
    4. Apply rolling overlap using token-config converted to approximate words.
    """
    if not text or not text.strip():
        return []

    target_size = max(1, target_size)
    overlap = max(0, overlap)
    overlap_words = min(_tokens_to_overlap_words(overlap), max(0, target_size - 1))

    units: list[str] = []
    prose_buffer: list[str] = []
    in_code_block = False
    code_block_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        fence_line = stripped.startswith("```")

        if in_code_block:
            code_block_lines.append(line)
            if fence_line:
                units.append("\n".join(code_block_lines).strip())
                code_block_lines = []
                in_code_block = False
            continue

        if fence_line:
            _flush_prose_units(prose_buffer, units, target_size)
            prose_buffer = []
            in_code_block = True
            code_block_lines = [line]
            continue

        prose_buffer.append(line)

    if in_code_block:
        units.append("\n".join(code_block_lines).strip())

    _flush_prose_units(prose_buffer, units, target_size)

    chunks: list[str] = []
    current_units: list[str] = []
    current_tokens = 0

    for unit in units:
        if not unit:
            continue

        is_fenced_code = unit.lstrip().startswith("```") and "```" in unit.strip()[3:]
        candidate_parts = (
            _split_long_unit(unit, target_size)
            if (_estimate_tokens(unit) > target_size and not is_fenced_code)
            else [unit]
        )

        for part in candidate_parts:
            part_tokens = _estimate_tokens(part)
            if current_units and current_tokens + part_tokens > target_size:
                chunk_text = "\n\n".join(current_units).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                current_units = [part]
                current_tokens = part_tokens

                if overlap_words and chunks:
                    overlap_terms = chunks[-1].split()[-overlap_words:]
                    if overlap_terms:
                        overlap_text = " ".join(overlap_terms)
                        current_units.insert(0, overlap_text)
                        current_tokens = _estimate_tokens(overlap_text) + part_tokens
            else:
                current_units.append(part)
                current_tokens += part_tokens

    if current_units:
        chunk_text = "\n\n".join(current_units).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def validate_chunk_quality(chunks: list[str]) -> dict[str, int]:
    """
    Return structural integrity report for chunk quality validation.
    """
    total_chunks = len(chunks)
    max_tokens = max(1, int(CHUNK_SIZE * 1.25))
    min_tokens = max(1, int(CHUNK_SIZE * 0.20))
    oversized_chunks = sum(1 for chunk in chunks if _estimate_tokens(chunk) > max_tokens)
    tiny_chunks = sum(1 for chunk in chunks if _estimate_tokens(chunk) < min_tokens)

    return {
        "total_chunks": total_chunks,
        "code_fence_unbalanced": sum(chunk.count("```") % 2 for chunk in chunks),
        "block_math_unbalanced": sum(chunk.count("$$") % 2 for chunk in chunks),
        "empty_chunks": sum(1 for chunk in chunks if not chunk.strip()),
        "oversized_chunks": oversized_chunks,
        "tiny_chunks": tiny_chunks,
        "oversized_pct": int((oversized_chunks / total_chunks) * 100) if total_chunks else 0,
        "tiny_pct": int((tiny_chunks / total_chunks) * 100) if total_chunks else 0,
    }


def _enforce_chunk_quality(quality: dict[str, int]) -> None:
    """Abort indexing when chunk quality thresholds are violated."""
    failures: list[str] = []
    if quality["code_fence_unbalanced"] > 0:
        failures.append(f"unbalanced code fences={quality['code_fence_unbalanced']}")
    if quality["block_math_unbalanced"] > 0:
        failures.append(f"unbalanced block math={quality['block_math_unbalanced']}")
    if quality["empty_chunks"] > 0:
        failures.append(f"empty chunks={quality['empty_chunks']}")
    if quality["oversized_pct"] > 5:
        failures.append(f"oversized_pct={quality['oversized_pct']}%")
    if quality["tiny_pct"] > 5:
        failures.append(f"tiny_pct={quality['tiny_pct']}%")

    if failures:
        raise ValueError("Chunk quality check failed: " + "; ".join(failures))


def _load_physical_page_map(book_stem: str) -> dict[int, int]:
    """
    Load page index -> physical page mapping from OCR manifest.
    Falls back to checkpoint-derived 1-index mapping if manifest is missing.
    """
    manifest_path = PARSED_MD / book_stem / "page_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = payload.get("pages", [])
        page_map: dict[int, int] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            page_idx = row.get("page_idx")
            physical_page = row.get("physical_page")
            if isinstance(page_idx, int) and isinstance(physical_page, int):
                page_map[page_idx] = physical_page
        if page_map:
            return page_map

    fallback_map: dict[int, int] = {}
    for path in checkpoint.all_pages(book_stem):
        page_idx = checkpoint.page_idx_from_path(path)
        fallback_map[page_idx] = page_idx + 1
    return fallback_map


def _build_chunk_data_with_pages(
    book_stem: str,
    markdown_splitter,
) -> list[tuple[str, int, dict[str, str | int]]]:
    """
    Build chunk rows as (chunk_text, physical_page, metadata) from checkpoints.
    """
    checkpoint_paths = checkpoint.all_pages(book_stem)
    if not checkpoint_paths:
        return []

    page_map = _load_physical_page_map(book_stem)
    chunk_data: list[tuple[str, int, dict[str, str | int]]] = []
    last_headers: dict[str, str] = {}

    for page_path in checkpoint_paths:
        page_idx = checkpoint.page_idx_from_path(page_path)
        physical_page = page_map.get(page_idx, page_idx + 1)
        page_text = page_path.read_text(encoding="utf-8")
        if not page_text.strip():
            continue

        page_sections = markdown_splitter.split_text(page_text)
        if page_sections:
            for split_doc in page_sections:
                base_meta: dict[str, str | int] = dict(split_doc.metadata) if split_doc.metadata else {}
                for header_key in ("Header 1", "Header 2", "Header 3"):
                    if header_key in base_meta and isinstance(base_meta[header_key], str) and base_meta[header_key]:
                        last_headers[header_key] = base_meta[header_key]
                    elif header_key in last_headers:
                        base_meta[header_key] = last_headers[header_key]
                chunk_texts = chunk_with_boundaries(
                    split_doc.page_content,
                    target_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                )
                for chunk_text in chunk_texts:
                    if not chunk_text.strip():
                        continue
                    meta: dict[str, str | int] = {
                        **base_meta,
                        "source": page_path.name,
                        "page": physical_page,
                        "physical_page": physical_page,
                    }
                    chunk_data.append((chunk_text, physical_page, meta))
            continue

        # If structural splitting yields no sections, fall back to plain page chunking.
        chunk_texts = chunk_with_boundaries(
            page_text,
            target_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
        for chunk_text in chunk_texts:
            if not chunk_text.strip():
                continue
            chunk_data.append(
                (
                    chunk_text,
                    physical_page,
                    {
                        **last_headers,
                        "source": page_path.name,
                        "page": physical_page,
                        "physical_page": physical_page,
                    },
                )
            )

    return chunk_data


def index_chunks_with_pages(
    book_stem: str,
    chunk_data: list[tuple[str, int, dict[str, str | int]]],
) -> int:
    """
    Index chunks into ChromaDB with page-aware metadata.
    """
    col = get_collection(book_stem)
    docs: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    ids: list[str] = []
    batches_inserted = 0
    batch_size = 100

    for idx, (chunk_text, physical_page, metadata) in enumerate(chunk_data):
        meta: dict[str, str | int] = dict(metadata) if metadata else {}
        meta["physical_page"] = int(physical_page)
        meta.setdefault("page", int(physical_page))
        meta.setdefault("source", book_stem)

        docs.append(chunk_text)
        metadatas.append(meta)
        ids.append(f"{book_stem}_chunk_{idx:08d}")

        if len(docs) >= batch_size:
            print(f"    -> Adding batch {batches_inserted + 1}...")
            col.add(documents=docs, metadatas=metadatas, ids=ids)
            batches_inserted += 1
            docs, metadatas, ids = [], [], []

    if docs:
        print(f"    -> Adding batch {batches_inserted + 1}...")
        col.add(documents=docs, metadatas=metadatas, ids=ids)

    return len(chunk_data)


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
        from langchain_text_splitters import MarkdownHeaderTextSplitter
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

    # 1. Structural splitter definition (used per-page)
    print("Chunking by structural headers...")
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # 2. Semantic chunking with physical page metadata
    print("Enforcing semantic boundary chunking with page metadata...")
    chunk_data = _build_chunk_data_with_pages(book_stem, markdown_splitter)
    if not chunk_data:
        print(
            f"Error: No checkpoint pages found for '{book_stem}'. "
            "Run OCR first to generate per-page checkpoints."
        )
        sys.exit(1)

    total_chunks = len(chunk_data)
    quality = validate_chunk_quality([chunk for chunk, _, _ in chunk_data])
    print(f"Chunk quality: {quality}")
    _enforce_chunk_quality(quality)
    print(f"Produced {total_chunks} chunk segments. Getting DB Connection...")
    return index_chunks_with_pages(book_stem, chunk_data)


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
