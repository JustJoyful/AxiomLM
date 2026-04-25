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


def route(pdf_path: Path, mode: str = "auto", force: bool = False) -> list[Path]:
    """
    Route OCR execution to the appropriate engine.

    Args:
        pdf_path: Path to the input PDF.
        mode:     "marker"  — use Marker (clean digital PDFs).
                  "mineru"  — use MinerU (warped/scanned PDFs).
                  "auto"    — defaults to Marker; pass --mode mineru for scans.
        force:    If True, re-process all pages ignoring existing checkpoints.

    Returns:
        Sorted list of per-page checkpoint Paths.

    Raises:
        ValueError: If mode is not one of the accepted values.
    """
    pdf_path = Path(pdf_path)

    if mode == "marker" or mode == "auto":
        from src.ocr_marker import run
        return run(pdf_path, force=force)

    elif mode == "mineru":
        from src.ocr_mineru import run
        return run(pdf_path, force=force)

    else:
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
        help="OCR engine: 'marker' for clean PDFs, 'mineru' for scanned (default: auto)",
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
