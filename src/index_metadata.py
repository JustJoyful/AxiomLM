"""Read-only source provenance for versioned contextual indexes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from src import config

# The five unnumbered titles in the source SICP edition are chapter boundaries.
SICP_CHAPTERS = {
    "Building Abstractions with Procedures": "1",
    "Building Abstractions with Data": "2",
    "Modularity, Objects, and State": "3",
    "Metalinguistic Abstraction": "4",
    "Computing with Register Machines": "5",
}


def validate_book(book_stem: str) -> None:
    if (
        not book_stem
        or book_stem in {".", ".."}
        or Path(book_stem).name != book_stem
        or "\\" in book_stem
    ):
        raise ValueError("Book identity must be a single directory name")


@dataclass(frozen=True)
class SourceSegment:
    text: str
    book_stem: str
    source_pages: tuple[int, ...]
    chapter: str
    section: str
    subsection: str | None
    source_path: str
    source_start: int = 0


def normalize_metadata(segment: SourceSegment) -> dict[str, str | int]:
    validate_book(segment.book_stem)
    pages = sorted(set(segment.source_pages))
    if not pages or any(type(p) is not int or p < 1 for p in pages):
        raise ValueError("Source requires positive physical pages")
    meta = {
        "book_stem": segment.book_stem,
        "chapter": segment.chapter or "Unknown Chapter",
        "section": segment.section or "Unknown Section",
        "physical_page_start": pages[0],
        "physical_page_end": pages[-1],
        "physical_pages": json.dumps(pages),
        "physical_page": pages[0],
        "page": pages[0],
        "source": segment.source_path,
        "Header 1": segment.chapter or "Unknown Chapter",
        "Header 2": segment.section or "Unknown Section",
    }
    if segment.subsection:
        meta["Header 3"] = segment.subsection
    return meta


def stable_chunk_id(
    book_stem: str,
    source_paths: Sequence[str],
    source_pages: Sequence[int],
    chapter: str,
    section: str,
    ordinal: int,
    text: str,
) -> str:
    validate_book(book_stem)
    record = [
        book_stem,
        list(source_paths),
        sorted(set(source_pages)),
        chapter,
        section,
        ordinal,
        text,
    ]
    return hashlib.sha256(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def source_segments(book_stem: str) -> list[SourceSegment]:
    validate_book(book_stem)
    manifest_path = config.PARSED_MD / book_stem / "page_manifest.json"
    page_map = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("book_stem", book_stem) != book_stem:
            raise ValueError("Manifest book identity mismatch")
        for row in manifest.get("pages", []):
            idx = row["page_idx"]
            physical = row["physical_page"]
            if (
                type(idx) is not int
                or idx < 0
                or type(physical) is not int
                or physical != idx + 1
                or row["checkpoint_file"] != f"page_{idx:04d}.md"
                or idx in page_map
            ):
                raise ValueError("Invalid physical-page manifest")
            page_map[idx] = physical
    chapter, section, subsection = "Unknown Chapter", "Unknown Section", None
    output = []
    fenced = False
    for path in sorted((config.CHECKPOINTS / book_stem).glob("page_*.md")):
        if not re.fullmatch(r"page_\d{4}\.md", path.name):
            raise ValueError("Invalid checkpoint filename")
        idx = int(path.stem.split("_")[1])
        text = path.read_text()
        cuts = {0, len(text)}
        for match in re.finditer(r".+?(?:\n\s*\n|\Z)", text, re.DOTALL):
            cuts.add(match.start())
            cuts.add(match.end())
        offset = 0
        scan_fenced = fenced
        for line in text.splitlines(keepends=True):
            if not scan_fenced and re.match(r"\s*#{1,6}\s+", line):
                cuts.add(offset)
            if re.match(r"^\s*(```|~~~)", line):
                scan_fenced = not scan_fenced
            offset += len(line)
        for start, stop in pairwise(sorted(cuts)):
            paragraph = text[start:stop]
            heading = None if fenced else re.match(r"\s*(#{1,6})\s+(.+)", paragraph)
            if heading:
                title = heading[2].strip().strip("*").strip()
                numbered = re.match(r"(\d+(?:\.\d+)+)\b", title)
                if book_stem == "sicp" and title in SICP_CHAPTERS:
                    chapter, section, subsection = (
                        SICP_CHAPTERS[title],
                        "Unknown Section",
                        None,
                    )
                elif numbered:
                    number = numbered[1]
                    parent = number.split(".")[0]
                    if chapter != parent:
                        chapter, section, subsection = parent, "Unknown Section", None
                    section = number
                    subsection = number if number.count(".") > 1 else None
                elif book_stem == "sicp":
                    subsection = title
                elif len(heading[1]) == 1:
                    chapter, section, subsection = title, "Unknown Section", None
                elif len(heading[1]) == 2:
                    section, subsection = title, None
                elif len(heading[1]) == 3:
                    subsection = title
            for line in paragraph.splitlines():
                if re.match(r"^\s*(```|~~~)", line):
                    fenced = not fenced
            output.append(
                SourceSegment(
                    paragraph,
                    book_stem,
                    (page_map.get(idx, idx + 1),),
                    chapter,
                    section,
                    subsection,
                    path.name,
                    start,
                )
            )
    return output
