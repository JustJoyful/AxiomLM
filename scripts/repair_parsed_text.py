"""Write repaired copies of a book's checkpoints; the originals stay untouched.

Usage: python scripts/repair_parsed_text.py <book_stem> --pdf /abs/book.pdf

Pages go to data/parsed_md/repaired/<stem>/page_NNNN.md with repair_report.json
beside them. An existing output folder is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pymupdf

from src import config
from src.text_repair import PUA_KINDS, RepairStats, code_lines, repair_page


def repair_book(book_stem: str, pdf: Path, output: Path) -> dict:
    source = config.CHECKPOINTS / book_stem
    pages = sorted(source.glob("page_[0-9][0-9][0-9][0-9].md"))
    if not pages:
        raise SystemExit(f"No checkpoints in {source}")
    if output.exists():
        raise SystemExit(f"{output} exists; choose a new output folder")
    doc = pymupdf.open(pdf)
    if len(doc) != len(pages):
        raise SystemExit(f"PDF has {len(doc)} pages, checkpoints {len(pages)}")
    stats = RepairStats()
    repaired = {}
    for path in pages:
        page_idx = int(path.stem.split("_")[1])
        repaired[path.name] = repair_page(
            path.read_text(encoding="utf-8"),
            code_lines(doc[page_idx]),
            page_idx + 1,  # physical pages are one-based
            stats,
        )
    output.mkdir(parents=True)
    for name, text in repaired.items():
        (output / name).write_text(text, encoding="utf-8")
    report = {
        "book_stem": book_stem,
        "pdf": str(pdf),
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "source": str(source),
        "pages": len(pages),
        "pages_changed": sum(
            text != (source / name).read_text(encoding="utf-8")
            for name, text in repaired.items()
        ),
        "code_lines_repaired": stats.code_lines_repaired,
        "pua_mapped": {
            kind: {
                f"U+{cp:04X} {table[cp]}": stats.pua_mapped[cp]
                for cp in sorted(table)
                if stats.pua_mapped[cp]
            }
            for kind, table in PUA_KINDS.items()
        },
        "pua_unmapped": {f"U+{cp:04X}": n for cp, n in stats.pua_unmapped.items()},
        "skipped_code_lines": stats.skipped,
    }
    (output / "repair_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("book")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: REPAIRED_MD/<book>")
    args = parser.parse_args()
    out = args.output or config.REPAIRED_MD / args.book
    result = repair_book(args.book, args.pdf, out)
    summary = {k: v for k, v in result.items() if k != "skipped_code_lines"}
    summary["skipped_code_lines"] = len(result["skipped_code_lines"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
