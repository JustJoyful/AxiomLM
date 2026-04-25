"""
ocr_marker.py — Marker OCR runner for clean digital PDFs.

Uses the marker-pdf Python API to convert an entire PDF to markdown,
then splits output into per-page checkpoints so any re-run resumes
from the last successfully processed page.

Satisfies REQ-02.
"""

import sys
from pathlib import Path

from src import checkpoint
from src.config import PARSED_MD


def run(pdf_path: Path, force: bool = False) -> list[Path]:
    """
    Run Marker OCR on a clean digital PDF.

    Steps:
      1. Determine resume point from existing checkpoints.
      2. Call marker-pdf Python API to convert the full PDF to markdown.
      3. Split the resulting markdown on form-feed characters ('\\f') into pages.
      4. Save each page as a checkpoint (skip already-done pages unless force=True).
      5. Write concatenated markdown to PARSED_MD / book_stem / "full.md".
      6. Return the list of checkpoint paths.

    Args:
        pdf_path: Path to a clean digital PDF.
        force:    If True, re-process all pages even if checkpoints exist.

    Returns:
        Sorted list of per-page checkpoint paths.
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as exc:
        raise ImportError(
            "marker-pdf is not installed. Run: pip install marker-pdf"
        ) from exc

    pdf_path = Path(pdf_path)
    book_stem = pdf_path.stem
    resume_from = checkpoint.last_completed(book_stem)

    if resume_from >= 0 and not force:
        print(f"  Resuming from page {resume_from + 1} (pages 0–{resume_from} already checkpointed)")

    # Run Marker on the full PDF
    print(f"  Running Marker on: {pdf_path.name} ...")
    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf_path))
    full_md: str = rendered.markdown

    # Split into pages on form-feed; fall back to single page if no \f present
    pages = full_md.split("\f") if "\f" in full_md else [full_md]

    for page_idx, page_text in enumerate(pages):
        if page_idx <= resume_from and not force:
            continue
        checkpoint.save(book_stem, page_idx, page_text)
        print(f"  ✓ page {page_idx:04d}")

    # Write concatenated full.md
    out_dir = PARSED_MD / book_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "full.md").write_text(full_md, encoding="utf-8")

    return checkpoint.all_pages(book_stem)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Marker OCR — clean PDF → Markdown")
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument("--force", action="store_true", help="Re-process all pages")
    args = parser.parse_args()

    pages = run(args.pdf, force=args.force)
    print(f"\nOCR complete: {len(pages)} pages → {PARSED_MD / args.pdf.stem}/")
