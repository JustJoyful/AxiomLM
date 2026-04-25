"""
preprocessor.py — Phase 1: PDF ingestion and OpenCV cleaning pipeline.

Entry points:
    extract_pages(pdf_path, dpi)  → list of raw PNG paths
    preprocess_page(raw_png, out_dir) → cleaned PNG path
    run(pdf_path, dpi)            → list of clean PNG paths

CLI:
    python -m src.preprocessor path/to/book.pdf
"""

import math
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

from src.config import CLEAN_IMGS


# ── Task 1: PDF → raw PNGs ────────────────────────────────────────────────────

def extract_pages(pdf_path: Path, dpi: int = 200) -> list[Path]:
    """
    Render each page of a PDF to a raw PNG at the given DPI.

    Raw PNGs are written to:
        CLEAN_IMGS / pdf_path.stem / "raw" / page_NNNN.png

    Returns the list of saved PNG paths in page order.
    """
    pdf_path = Path(pdf_path)
    raw_dir = CLEAN_IMGS / pdf_path.stem / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72  # fitz default is 72 DPI
    mat = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = raw_dir / f"page_{i:04d}.png"
        pix.save(str(out_path))
        saved.append(out_path)

    doc.close()
    return saved


# ── Task 2: OpenCV preprocessing pipeline ────────────────────────────────────

def _detect_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate page skew angle in degrees using Hough lines on Canny edges.
    Returns 0.0 if no reliable angle can be detected.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

    if not angles:
        return 0.0

    # Median is more robust than mean against outlier lines
    return float(np.median(angles))


def preprocess_page(raw_png: Path, out_dir: Path) -> Path:
    """
    Apply OpenCV cleaning pipeline to a single raw PNG:
      1. Grayscale
      2. Denoise (fastNlMeansDenoising)
      3. Deskew  (Hough lines, ±15° clamp, skip if < 0.5°)
      4. Adaptive threshold (Gaussian)

    Saves result to out_dir / raw_png.name and returns that path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(raw_png), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {raw_png}")

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 3. Deskew
    angle = _detect_skew_angle(denoised)
    angle = max(-15.0, min(15.0, angle))  # clamp to ±15°

    if abs(angle) >= 0.5:
        h, w = denoised.shape
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            denoised,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        deskewed = denoised

    # 4. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        deskewed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )

    out_path = out_dir / raw_png.name
    cv2.imwrite(str(out_path), thresh)
    return out_path


# ── Top-level runner ──────────────────────────────────────────────────────────

def run(pdf_path: Path, dpi: int = 200) -> list[Path]:
    """
    Full pipeline: PDF → extract raw pages → preprocess each → return clean PNGs.

    Clean PNGs are saved to: CLEAN_IMGS / pdf_path.stem / page_NNNN.png
    """
    pdf_path = Path(pdf_path)
    out_dir = CLEAN_IMGS / pdf_path.stem

    raw_pages = extract_pages(pdf_path, dpi=dpi)
    clean_pages: list[Path] = []

    for raw_png in raw_pages:
        clean_path = preprocess_page(raw_png, out_dir)
        clean_pages.append(clean_path)
        print(f"  ✓ {clean_path.name}")

    return clean_pages


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.preprocessor <path/to/book.pdf> [dpi]")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    print(f"Processing: {pdf.name} at {dpi} DPI")
    pages = run(pdf, dpi=dpi)
    print(f"\nPreprocessed {len(pages)} pages → {CLEAN_IMGS / pdf.stem}/")
