"""
ocr_mineru.py — MinerU OCR runner for warped/complex scanned PDFs.

Processes PDFs page-by-page, clearing VRAM after each page to stay within
the 6 GB RTX 3050 constraint. Checkpoints each page so any crash is
fully resumable without re-processing completed pages.

Satisfies REQ-03 (MinerU with VRAM guard) and REQ-04 (crash recovery).
"""

import gc
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF — for single-page image extraction

from src import checkpoint
from src.config import CLEAN_IMGS, MINERU_MIN_VRAM_GB, PARSED_MD


# ── VRAM utilities ────────────────────────────────────────────────────────────

def _free_vram_gb() -> float:
    """
    Return estimated free VRAM in GB.
    Returns 999.0 if CUDA is not available (CPU-only fallback).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 999.0
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(0)
        free = total - reserved
        return free / (1024 ** 3)
    except Exception:
        return 999.0


def _clear_vram() -> None:
    """
    Flush CUDA cache and run Python garbage collection.
    No-op if CUDA is not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ── MinerU page runner ────────────────────────────────────────────────────────

def _run_mineru_on_image(image_path: Path, tmp_dir: Path) -> str:
    """
    Run MinerU on a single page image and return the markdown text.
    Raises ImportError with a clear message if MinerU is not installed.
    """
    try:
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.pipe.UNIPipe import UNIPipe
    except ImportError as exc:
        raise ImportError(
            "MinerU (magic-pdf) is not installed. Run: pip install mineru"
        ) from exc

    image_bytes = image_path.read_bytes()
    writer = FileBasedDataWriter(str(tmp_dir))

    pipe = UNIPipe(image_bytes, {"_pdf_type": "", "model_list": []}, writer)
    pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()
    md_text = pipe.pipe_mk_markdown(writer, drop_mode="none")

    return md_text if isinstance(md_text, str) else ""


# ── Main runner ───────────────────────────────────────────────────────────────

def run(pdf_path: Path, force: bool = False) -> list[Path]:
    """
    Run MinerU OCR on a warped/complex scanned PDF page-by-page.

    Steps per page:
      1. Check checkpoint; skip if already done (unless force=True).
      2. Guard: warn if free VRAM < MINERU_MIN_VRAM_GB, prompt user to continue.
      3. Extract page as image using fitz.
      4. Run MinerU on the single-page image.
      5. Save result to checkpoint.
      6. Clear VRAM: torch.cuda.empty_cache() + gc.collect().

    Args:
        pdf_path: Path to the scanned PDF.
        force:    If True, re-process even checkpointed pages.

    Returns:
        Sorted list of per-page checkpoint paths.
    """
    pdf_path = Path(pdf_path)
    book_stem = pdf_path.stem
    resume_from = checkpoint.last_completed(book_stem)

    if resume_from >= 0 and not force:
        print(f"  Resuming from page {resume_from + 1} (pages 0–{resume_from} already done)")

    doc = fitz.open(str(pdf_path))
    zoom = 200 / 72  # 200 DPI render
    mat = fitz.Matrix(zoom, zoom)

    with tempfile.TemporaryDirectory(prefix="axiom_mineru_") as tmp_str:
        tmp_dir = Path(tmp_str)

        for page_idx in range(len(doc)):
            if page_idx <= resume_from and not force:
                print(f"  Skipping page {page_idx:04d} (checkpointed)")
                continue

            # VRAM guard before each page
            free = _free_vram_gb()
            if free < MINERU_MIN_VRAM_GB:
                print(f"\n  ⚠  Low VRAM: {free:.1f} GB free, need ≥{MINERU_MIN_VRAM_GB} GB")
                answer = input("  Continue anyway? [y/N] ").strip().lower()
                if answer != "y":
                    doc.close()
                    raise RuntimeError(
                        f"Aborted: insufficient VRAM ({free:.1f} GB < {MINERU_MIN_VRAM_GB} GB)"
                    )

            # Extract page as PNG
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_path = tmp_dir / f"page_{page_idx:04d}.png"
            pix.save(str(img_path))

            # Run MinerU
            print(f"  Processing page {page_idx:04d} ...", end=" ", flush=True)
            md_text = _run_mineru_on_image(img_path, tmp_dir)
            checkpoint.save(book_stem, page_idx, md_text)
            print("✓")

            # Flush VRAM after every page
            _clear_vram()

    doc.close()

    # Write concatenated full.md
    all_pages = checkpoint.all_pages(book_stem)
    full_md = "\f".join(p.read_text(encoding="utf-8") for p in all_pages)
    out_dir = PARSED_MD / book_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "full.md").write_text(full_md, encoding="utf-8")

    return all_pages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MinerU OCR — warped PDF → Markdown")
    parser.add_argument("pdf", type=Path, help="Path to scanned PDF")
    parser.add_argument("--force", action="store_true", help="Re-process all pages")
    args = parser.parse_args()

    pages = run(args.pdf, force=args.force)
    print(f"\nOCR complete: {len(pages)} pages → {PARSED_MD / args.pdf.stem}/")
