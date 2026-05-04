"""
ocr_mineru.py — MinerU OCR runner for warped/complex scanned PDFs.

Uses the modern MinerU CLI to parse a full PDF, then materializes
per-page checkpoints for resumable downstream indexing.
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src import checkpoint
from src.config import (
    MINERU_BACKEND_FALLBACK,
    MINERU_BACKEND_PRIMARY,
    MINERU_BATCH_PAGES,
    MINERU_LANG,
    MINERU_MIN_VRAM_GB,
    PARSED_MD,
)


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
        return free / (1024**3)
    except Exception:
        return 999.0


def _clear_vram() -> None:
    """Flush CUDA cache and run Python garbage collection."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _pdf_page_count(pdf_path: Path) -> int:
    """Return PDF page count using PyMuPDF; returns 0 when unavailable."""
    try:
        import fitz
    except Exception:
        return 0

    doc = fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def _run_mineru_cli(
    pdf_path: Path,
    tmp_dir: Path,
    start_page: int | None = None,
    end_page: int | None = None,
    backend: str | None = None,
) -> str:
    """Run MinerU CLI on a full PDF and return markdown text.

    Args:
        backend: MinerU backend string (e.g. 'pipeline', 'hybrid-auto-engine').
                 Defaults to MINERU_BACKEND_PRIMARY from config.
    """
    mineru_bin = shutil.which("mineru")
    if mineru_bin is None:
        candidate = Path(sys.executable).parent / "mineru"
        if candidate.exists():
            mineru_bin = str(candidate)
    if mineru_bin is None:
        raise ImportError("MinerU CLI not found. Install with: uv pip install mineru")

    chosen_backend = backend or MINERU_BACKEND_PRIMARY
    output_dir = tmp_dir / "mineru_out"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        mineru_bin,
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
        "-m",
        "ocr",
        "-b",
        chosen_backend,
    ]
    if MINERU_LANG:
        cmd.extend(["-l", MINERU_LANG])
    if start_page is not None:
        cmd.extend(["-s", str(start_page)])
    if end_page is not None:
        cmd.extend(["-e", str(end_page)])

    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONUNBUFFERED"] = "1"

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        tail = "\n".join(
            line for line in (completed.stdout + "\n" + completed.stderr).splitlines()[-20:]
        )
        raise RuntimeError(f"MinerU CLI failed (exit {completed.returncode}, backend={chosen_backend}). {tail}")

    stem = pdf_path.stem
    candidates = list(output_dir.rglob(f"{stem}.md"))
    if not candidates:
        md_files = list(output_dir.rglob("*.md"))
        if not md_files:
            raise RuntimeError("MinerU produced no markdown output.")
        candidates = [max(md_files, key=lambda p: p.stat().st_size)]

    json_candidates = list(output_dir.rglob(f"{stem}_content_list.json"))
    json_path = json_candidates[0] if json_candidates else None
    
    return candidates[0].read_text(encoding="utf-8"), json_path


def run(pdf_path: Path, force: bool = False) -> list[Path]:
    """
    Run MinerU OCR on scanned PDFs, then checkpoint by page.

    Args:
        pdf_path: Path to the scanned PDF.
        force: If True, re-process even checkpointed pages.
    """
    pdf_path = Path(pdf_path)
    book_stem = pdf_path.stem
    total_pages = _pdf_page_count(pdf_path)
    resume_from = checkpoint.last_completed(book_stem)
    first_page = 0 if force else (resume_from + 1)

    if resume_from >= 0 and not force:
        print(f"  Resuming from page {resume_from + 1} (pages 0–{resume_from} already done)")
    if total_pages > 0:
        print(f"  Detected {total_pages} pages. MinerU batch size: {MINERU_BATCH_PAGES}.")
        if first_page >= total_pages:
            return checkpoint.all_pages(book_stem)

    free = _free_vram_gb()
    if free < MINERU_MIN_VRAM_GB:
        print(f"\n  ⚠  Low VRAM: {free:.1f} GB free, need ≥{MINERU_MIN_VRAM_GB} GB")
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Aborted: insufficient VRAM in non-interactive mode "
                f"({free:.1f} GB < {MINERU_MIN_VRAM_GB} GB)"
            )
        answer = input("  Continue anyway? [y/N] ").strip().lower()
        if answer != "y":
            raise RuntimeError(f"Aborted: insufficient VRAM ({free:.1f} GB < {MINERU_MIN_VRAM_GB} GB)")

    # Determine which backends to try in order (primary → fallback).
    backends_to_try: list[str] = [MINERU_BACKEND_PRIMARY]
    if (
        MINERU_BACKEND_FALLBACK
        and MINERU_BACKEND_FALLBACK != MINERU_BACKEND_PRIMARY
    ):
        backends_to_try.append(MINERU_BACKEND_FALLBACK)

    def _run_with_fallback(
        start_page: int | None = None,
        end_page: int | None = None,
        tmp_dir: Path | None = None,
    ) -> tuple[str, Path | None]:
        """Try each backend in order; raise the last error if all fail."""
        last_exc: Exception | None = None
        _tmp_dir = tmp_dir or Path(tempfile.mkdtemp(prefix="axiom_mineru_"))
        for backend in backends_to_try:
            try:
                label = f" (backend={backend})" if len(backends_to_try) > 1 else ""
                if start_page is not None:
                    print(
                        f"  Running MinerU pages {start_page + 1}–{(end_page or start_page) + 1}"
                        f"/{total_pages}{label} ..."
                    )
                else:
                    print(f"  Running MinerU on: {pdf_path.name}{label} ...")
                return _run_mineru_cli(
                    pdf_path,
                    _tmp_dir,
                    start_page=start_page,
                    end_page=end_page,
                    backend=backend,
                )
            except RuntimeError as exc:
                last_exc = exc
                if len(backends_to_try) > 1:
                    print(f"  ⚠  MinerU backend '{backend}' failed: {str(exc).splitlines()[0]}")
                    _clear_vram()
        raise RuntimeError(
            f"All MinerU backends failed. Last error: {last_exc}"
        ) from last_exc

    if total_pages <= 0:
        with tempfile.TemporaryDirectory(prefix="axiom_mineru_") as tmp_str:
            tmp_dir = Path(tmp_str)
            full_md = _run_with_fallback(tmp_dir=tmp_dir)

        pages = full_md.split("\f") if "\f" in full_md else [full_md]
        for page_idx, page_text in enumerate(pages):
            if page_idx <= resume_from and not force:
                continue
            checkpoint.save(book_stem, page_idx, page_text)
            if page_idx % 25 == 0:
                print(f"  ✓ page {page_idx:04d}")
    else:
        with tempfile.TemporaryDirectory(prefix="axiom_mineru_") as tmp_str:
            tmp_dir = Path(tmp_str)
            for batch_start in range(first_page, total_pages, MINERU_BATCH_PAGES):
                batch_end = min(total_pages - 1, batch_start + MINERU_BATCH_PAGES - 1)
                batch_md, batch_json_path = _run_with_fallback(
                    start_page=batch_start,
                    end_page=batch_end,
                    tmp_dir=tmp_dir,
                )
                
                # Heuristic page splitting using MinerU's content_list.json
                batch_pages = []
                if batch_json_path and batch_json_path.exists():
                    try:
                        import json
                        with open(batch_json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        # Find the first unique text snippet of each page in the JSON
                        page_markers = {}
                        for item in data:
                            p_idx = item.get("page_idx")
                            text = item.get("text", "").strip()
                            if p_idx is not None and text and len(text) > 20: # skip short noise
                                if p_idx not in page_markers:
                                    page_markers[p_idx] = text
                        
                        # Split full markdown based on these markers
                        sorted_pages = sorted(page_markers.keys())
                        current_md = batch_md
                        for i in range(len(sorted_pages)):
                            curr_page = sorted_pages[i]
                            if i + 1 < len(sorted_pages):
                                next_marker = page_markers[sorted_pages[i+1]]
                                # Try to find where the next page starts
                                split_pos = current_md.find(next_marker)
                                if split_pos != -1:
                                    batch_pages.append(current_md[:split_pos].replace("\f", ""))
                                    current_md = current_md[split_pos:]
                                else:
                                    # Marker not found (maybe formula conversion), try a shorter snippet
                                    short_marker = next_marker[:40]
                                    split_pos = current_md.find(short_marker)
                                    if split_pos != -1:
                                        batch_pages.append(current_md[:split_pos].replace("\f", ""))
                                        current_md = current_md[split_pos:]
                                    else:
                                        # Give up on this split, we'll merge this page into the next
                                        pass
                        batch_pages.append(current_md.replace("\f", ""))
                    except Exception as e:
                        print(f"  ⚠ Failed to parse JSON map: {e}. Falling back to standard split.")
                        batch_pages = batch_md.split("\f") if "\f" in batch_md else [batch_md]
                else:
                    batch_pages = batch_md.split("\f") if "\f" in batch_md else [batch_md]

                expected = batch_end - batch_start + 1
                while len(batch_pages) > expected and not batch_pages[-1].strip():
                    batch_pages.pop()
                    
                if len(batch_pages) != expected:
                    print(
                        f"  ⚠ MinerU returned {len(batch_pages)} page(s) for range "
                        f"{batch_start + 1}–{batch_end + 1} (expected {expected}); "
                        "Aura may be slightly shifted."
                    )

                for offset, page_text in enumerate(batch_pages):
                    page_idx = batch_start + offset
                    checkpoint.save(book_stem, page_idx, page_text)
                    print(f"  ✓ page {page_idx:04d}")

    _clear_vram()

    out_dir = PARSED_MD / book_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = checkpoint.all_pages(book_stem)
    if not checkpoint_paths:
        raise RuntimeError("MinerU produced no checkpoint pages.")
    if total_pages > 0 and len(checkpoint_paths) < total_pages:
        print(
            f"  ⚠  Checkpoint set may be incomplete: found {len(checkpoint_paths)} page(s), "
            f"PDF reported {total_pages}. Proceeding with what was produced."
        )
    full_md = "\f".join(path.read_text(encoding="utf-8") for path in checkpoint_paths)
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

    return checkpoint_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MinerU OCR — scanned PDF → Markdown")
    parser.add_argument("pdf", type=Path, help="Path to scanned PDF")
    parser.add_argument("--force", action="store_true", help="Re-process all pages")
    args = parser.parse_args()

    pages = run(args.pdf, force=args.force)
    print(f"\nOCR complete: {len(pages)} pages → {PARSED_MD / args.pdf.stem}/")
