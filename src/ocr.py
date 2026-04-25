"""
ocr.py — Unified OCR entry point for AxiomLM.

Selects Marker (clean PDFs) or MinerU (warped/complex scans) based on
the --mode flag and delegates to the appropriate runner module.

Usage:
    python -m src.ocr --pdf path/to/book.pdf --mode auto
    python -m src.ocr --pdf path/to/scan.pdf --mode mineru --force
"""

import argparse
from pathlib import Path

from src.config import (
    AUTO_OCR_IMAGE_COVERAGE_THRESHOLD,
    AUTO_OCR_SAMPLE_PAGES,
    AUTO_OCR_SCANNED_PAGE_RATIO_THRESHOLD,
    AUTO_OCR_TEXT_CHAR_THRESHOLD,
)


def _sample_indices(page_count: int, sample_size: int) -> list[int]:
    """Return evenly spaced page indices for heuristic analysis."""
    if page_count <= 0:
        return []
    if sample_size >= page_count:
        return list(range(page_count))
    if sample_size <= 1:
        return [0]

    indices = {
        int(round(i * (page_count - 1) / (sample_size - 1)))
        for i in range(sample_size)
    }
    return sorted(indices)


def _page_metrics(page) -> tuple[int, float, bool]:
    """Return (text_chars, image_coverage_ratio, has_image_block) for one page."""
    text_chars = len(page.get_text("text").strip())
    page_area = max(page.rect.width * page.rect.height, 1.0)
    image_area = 0.0
    has_image_block = False

    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        image_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
        has_image_block = True

    image_coverage = min(1.0, image_area / page_area)
    return text_chars, image_coverage, has_image_block


def _choose_auto_engine(pdf_path: Path) -> tuple[str, str]:
    """
    Choose Marker vs MinerU by sampling pages and scoring scan-like traits.

    Scan-like page signal:
      - low extractable text
      - high image coverage
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "marker", "PyMuPDF unavailable for inspection; defaulting to Marker."

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return (
            "marker",
            (
                f"could not inspect PDF ({exc.__class__.__name__}: {detail}); "
                "defaulting to Marker"
            ),
        )
    try:
        page_count = len(doc)
        if page_count == 0:
            return "marker", "PDF has zero pages; defaulting to Marker."

        sample_indices = _sample_indices(page_count, AUTO_OCR_SAMPLE_PAGES)
        if not sample_indices:
            return "marker", "No sample pages available; defaulting to Marker."

        try:
            scanned_votes = 0
            total_chars = 0
            total_image_coverage = 0.0

            for idx in sample_indices:
                text_chars, image_coverage, has_image_block = _page_metrics(doc[idx])
                total_chars += text_chars
                total_image_coverage += image_coverage

                scan_like = (
                    text_chars <= AUTO_OCR_TEXT_CHAR_THRESHOLD
                    and image_coverage >= AUTO_OCR_IMAGE_COVERAGE_THRESHOLD
                ) or (
                    text_chars <= max(25, AUTO_OCR_TEXT_CHAR_THRESHOLD // 3)
                    and has_image_block
                )
                if scan_like:
                    scanned_votes += 1

            sampled = len(sample_indices)
            scanned_ratio = scanned_votes / sampled
            avg_chars = total_chars / sampled
            avg_coverage = total_image_coverage / sampled

            chosen = (
                "mineru"
                if scanned_ratio >= AUTO_OCR_SCANNED_PAGE_RATIO_THRESHOLD
                else "marker"
            )
            reason = (
                f"sampled {sampled}/{page_count} pages | "
                f"avg text chars/page={avg_chars:.0f} | "
                f"avg image coverage={avg_coverage:.0%} | "
                f"scan-like pages={scanned_votes}/{sampled}"
            )
            return chosen, reason
        except Exception as exc:
            detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            return (
                "marker",
                (
                    f"auto inspection failed ({exc.__class__.__name__}: {detail}); "
                    "defaulting to Marker"
                ),
            )
    finally:
        doc.close()


def route(pdf_path: Path, mode: str = "auto", force: bool = False) -> list[Path]:
    """
    Route OCR execution to the appropriate engine.

    Args:
        pdf_path: Path to the input PDF.
        mode:     "marker"  — use Marker (clean digital PDFs).
                  "mineru"  — use MinerU (warped/scanned PDFs).
                  "auto"    — inspect sampled pages, then choose Marker or MinerU.
        force:    If True, re-process all pages ignoring existing checkpoints.

    Returns:
        Sorted list of per-page checkpoint Paths.

    Raises:
        ValueError: If mode is not one of the accepted values.
    """
    pdf_path = Path(pdf_path)
    mode = mode.lower().strip()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Expected a file path, got: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file type for OCR: {pdf_path.suffix or '<none>'}. Expected .pdf")

    if mode == "auto":
        chosen_mode, reason = _choose_auto_engine(pdf_path)
        print(f"Auto mode selected '{chosen_mode}' ({reason})")
        mode = chosen_mode

    if mode == "marker":
        from src.ocr_marker import run
        return run(pdf_path, force=force)

    if mode == "mineru":
        from src.ocr_mineru import run
        return run(pdf_path, force=force)

    raise ValueError(
        f"Unknown OCR mode: '{mode}'. Expected one of: marker, mineru, auto"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AxiomLM OCR — convert PDFs to chunked Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.ocr --pdf book.pdf\n"
            "  python -m src.ocr --pdf scan.pdf --mode mineru\n"
            "  python -m src.ocr --pdf scan.pdf --mode mineru --force\n"
        ),
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Path to input PDF")
    parser.add_argument(
        "--mode",
        choices=["auto", "marker", "mineru"],
        default="auto",
        help=(
            "OCR engine: 'marker' for clean PDFs, 'mineru' for scanned, "
            "'auto' for heuristic routing (default: auto)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process all pages, ignoring existing checkpoints",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Render DPI for page extraction (default: 200)",
    )
    return parser


if __name__ == "__main__":
    from src.config import PARSED_MD

    args = _build_parser().parse_args()

    print(f"Mode: {args.mode} | PDF: {args.pdf.name}")
    pages = route(args.pdf, mode=args.mode, force=args.force)
    print(f"\nMode: {args.mode} | Pages: {len(pages)} | Output: {PARSED_MD / args.pdf.stem}/")
