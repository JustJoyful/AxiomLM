"""
ocr.py — Unified OCR entry point for AxiomLM.

Selects Marker (clean PDFs) or MinerU (warped/complex scans) based on
the --mode flag and delegates to the appropriate runner module.

Usage:
    python -m src.ocr --pdf path/to/book.pdf --mode auto
    python -m src.ocr --pdf path/to/scan.pdf --mode mineru --force
"""

import argparse
import importlib.util
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
        return "pymupdf4llm", "PyMuPDF unavailable for inspection; defaulting to pymupdf4llm."

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return (
            "pymupdf4llm",
            (
                f"could not inspect PDF ({exc.__class__.__name__}: {detail}); "
                "defaulting to pymupdf4llm"
            ),
        )
    try:
        page_count = len(doc)
        if page_count == 0:
            return "pymupdf4llm", "PDF has zero pages; defaulting to pymupdf4llm."

        sample_indices = _sample_indices(page_count, AUTO_OCR_SAMPLE_PAGES)
        if not sample_indices:
            return "pymupdf4llm", "No sample pages available; defaulting to pymupdf4llm."

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
                ) or (
                    # Many scanned PDFs include an OCR text layer, so text_chars can be high.
                    # If a page is effectively a full-page image, still treat it as scan-like.
                    image_coverage >= 0.90 and has_image_block
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
                else "pymupdf4llm"
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
                "pymupdf4llm",
                (
                    f"auto inspection failed ({exc.__class__.__name__}: {detail}); "
                    "defaulting to pymupdf4llm"
                ),
            )
    finally:
        doc.close()


def resolve_mode(pdf_path: Path, mode: str) -> tuple[str, str]:
    """
    Resolve OCR mode to a concrete engine and return (engine, reason).

    For manual modes, returns the selected mode with a manual-selection reason.
    For auto mode, returns the heuristic decision and rationale.

    Engines:
      pymupdf4llm  — fast, zero-GPU, default for clean digital PDFs
      mineru       — ML pipeline for warped/scanned PDFs
      marker       — high-quality ML (8 GB VRAM), manual only
    """
    normalized = mode.lower().strip()
    if normalized == "auto":
        return _choose_auto_engine(pdf_path)
    if normalized in {"pymupdf4llm", "marker", "mineru"}:
        return normalized, "manual mode selected"
    raise ValueError(
        f"Unknown OCR mode: '{mode}'. Expected one of: pymupdf4llm, marker, mineru, auto"
    )


def _mineru_available() -> bool:
    """
    Check whether MinerU runtime is available (legacy SDK or modern CLI).

    Legacy MinerU exposed `magic_pdf`; newer releases expose `mineru` CLI/package.
    """
    if importlib.util.find_spec("magic_pdf") is not None:
        return True
    if importlib.util.find_spec("mineru") is not None:
        return True
    return False


def route(pdf_path: Path, mode: str = "auto", force: bool = False) -> list[Path]:
    """
    Route OCR execution to the appropriate engine.

    Args:
        pdf_path: Path to the input PDF.
        mode:     "pymupdf4llm" — fast zero-GPU engine for clean digital PDFs (default auto).
                  "mineru"      — use MinerU (warped/scanned PDFs).
                  "marker"      — use Marker (high-quality ML, needs ~8 GB VRAM).
                  "auto"        — inspect sampled pages, choose pymupdf4llm or MinerU.
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
        chosen_mode, reason = resolve_mode(pdf_path, mode)
        if chosen_mode == "mineru" and not _mineru_available():
            chosen_mode = "pymupdf4llm"
            reason = (
                f"{reason}; MinerU is not installed, falling back to pymupdf4llm "
                "(install with: uv pip install mineru)"
            )
        print(f"Auto mode selected '{chosen_mode}' ({reason})")
        mode = chosen_mode

    if mode == "pymupdf4llm":
        from src.ocr_pymupdf4llm import run
        return run(pdf_path, force=force)

    if mode == "marker":
        from src.ocr_marker import run
        return run(pdf_path, force=force)

    if mode == "mineru":
        if not _mineru_available():
            raise ImportError(
                "MinerU (magic-pdf) is not installed. Install with: uv pip install mineru"
            )
        from src.ocr_mineru import run
        return run(pdf_path, force=force)

    raise ValueError(
        f"Unknown OCR mode: '{mode}'. Expected one of: pymupdf4llm, marker, mineru, auto"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AxiomLM OCR — convert PDFs to chunked Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.ocr --pdf book.pdf\n"
            "  python -m src.ocr --pdf book.pdf --mode pymupdf4llm\n"
            "  python -m src.ocr --pdf scan.pdf --mode mineru\n"
            "  python -m src.ocr --pdf scan.pdf --mode mineru --force\n"
            "  python -m src.ocr --pdf book.pdf --mode marker  # needs ~8GB VRAM\n"
        ),
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Path to input PDF")
    parser.add_argument(
        "--mode",
        choices=["auto", "pymupdf4llm", "mineru", "marker"],
        default="auto",
        help=(
            "OCR engine: 'pymupdf4llm' (fast, no GPU, default for clean PDFs), "
            "'mineru' for scanned PDFs, "
            "'marker' for high-quality ML output (needs ~8GB VRAM), "
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
