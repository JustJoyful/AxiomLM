"""
ocr_marker.py — Marker OCR runner for clean digital PDFs.

Uses the marker-pdf Python API to convert an entire PDF to markdown,
then splits output into per-page checkpoints so any re-run resumes
from the last successfully processed page.

Satisfies REQ-02.
"""

import gc
import json
import sys
from pathlib import Path

from src import checkpoint
from src.config import PARSED_MD


def _clear_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


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
    try:
        rendered = converter(str(pdf_path))
    except Exception as exc:
        message = str(exc)
        _OOM_SIGNALS = (
            "outofmemoryerror",
            "cuda out of memory",
            "out of memory",
            "cudaerroroutofmemory",
            "cudamalloc failed",
        )
        is_oom = any(sig in message.lower() for sig in _OOM_SIGNALS)
        if is_oom:
            print("  ⚠ Marker GPU OOM. Retrying on CPU...")
            # Aggressively free the GPU before re-loading model onto CPU
            try:
                del converter
            except Exception:
                pass
            _clear_vram()
            try:
                import os
                # Hide the GPU entirely for this in-process retry so Marker
                # cannot accidentally try GPU again during model loading.
                _orig_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                try:
                    converter = PdfConverter(artifact_dict=create_model_dict(device="cpu"))
                    rendered = converter(str(pdf_path))
                finally:
                    # Restore CUDA visibility regardless of success/failure
                    if _orig_visible is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = _orig_visible
            except Exception as retry_exc:
                detail = str(retry_exc).splitlines()[0] if str(retry_exc) else retry_exc.__class__.__name__
                raise RuntimeError(
                    f"Marker GPU OOM and CPU retry failed for '{pdf_path.name}': "
                    f"{retry_exc.__class__.__name__}: {detail}"
                ) from retry_exc
        else:
            detail = message.splitlines()[0] if message else exc.__class__.__name__
            raise RuntimeError(
                f"Marker failed to open or parse '{pdf_path.name}': {exc.__class__.__name__}: {detail}"
            ) from exc
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

    return checkpoint.all_pages(book_stem)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Marker OCR — clean PDF → Markdown")
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument("--force", action="store_true", help="Re-process all pages")
    args = parser.parse_args()

    pages = run(args.pdf, force=args.force)
    print(f"\nOCR complete: {len(pages)} pages → {PARSED_MD / args.pdf.stem}/")
