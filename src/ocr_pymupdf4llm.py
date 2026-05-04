"""
ocr_pymupdf4llm.py — Fast clean-PDF runner for AxiomLM.

Uses pymupdf4llm (PyMuPDF's built-in markdown extractor) to convert
digital PDFs to per-page markdown checkpoints.

Advantages over Marker:
  - Zero GPU required, runs entirely on CPU via PyMuPDF C library
  - 10-50x faster than Marker on clean digital PDFs
  - No model downloads, no heavy ML pipeline

Use Marker (manual mode) for PDFs with complex math/tables needing
ML-quality rendering; use MinerU for scanned/warped PDFs.

Satisfies REQ-02 (clean PDF path).
"""

import json
from pathlib import Path

from src import checkpoint
from src.config import PARSED_MD


def run(pdf_path: Path, force: bool = False) -> list[Path]:
    """
    Convert a clean digital PDF to per-page markdown checkpoints.

    Args:
        pdf_path: Path to a clean digital PDF.
        force:    If True, re-process all pages even if checkpoints exist.

    Returns:
        Sorted list of per-page checkpoint paths.
    """
    try:
        import pymupdf4llm  # noqa: F401 — verify available before we start
        import fitz
    except ImportError as exc:
        raise ImportError(
            "pymupdf4llm is not installed. Run: pip install pymupdf4llm"
        ) from exc

    pdf_path = Path(pdf_path)
    book_stem = pdf_path.stem
    resume_from = checkpoint.last_completed(book_stem)

    if resume_from >= 0 and not force:
        print(f"  Resuming from page {resume_from + 1} (pages 0–{resume_from} already checkpointed)")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()

    print(f"  Running pymupdf4llm on: {pdf_path.name} ({total_pages} pages) ...")

    # pymupdf4llm.to_markdown() returns a single markdown string for the whole PDF.
    # page_chunks=True gives per-page dicts so we can slice them individually.
    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    pages: list[str] = [chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                        for chunk in page_chunks]

    # Fall back to splitting on form-feed if page_chunks didn't work as expected
    if len(pages) == 1 and "\f" in pages[0]:
        pages = pages[0].split("\f")

    for page_idx, page_text in enumerate(pages):
        if page_idx <= resume_from and not force:
            continue
        checkpoint.save(book_stem, page_idx, page_text)
        if page_idx % 50 == 0:
            print(f"  ✓ page {page_idx:04d}")

    # Write concatenated full.md
    out_dir = PARSED_MD / book_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    full_md = "\f".join(pages)
    (out_dir / "full.md").write_text(full_md, encoding="utf-8")

    # Persist page tracking manifest for downstream metadata-aware indexing.
    # page_idx is checkpoint-native (0-based), physical_page is user-facing (1-based).
    manifest = checkpoint.page_manifest(book_stem)
    (out_dir / "page_manifest.json").write_text(
        json.dumps(
            {
                "book_stem": book_stem,
                "total_pages": len(manifest),
                "pages": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = checkpoint.all_pages(book_stem)
    print(f"  ✓ Done: {len(result)} pages extracted")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="pymupdf4llm OCR — clean PDF → Markdown")
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument("--force", action="store_true", help="Re-process all pages")
    args = parser.parse_args()

    pages = run(args.pdf, force=args.force)
    print(f"\nOCR complete: {len(pages)} pages → {PARSED_MD / args.pdf.stem}/")
