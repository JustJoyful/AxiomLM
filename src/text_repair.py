"""Deterministic repair of parsed pages against the PDF text layer.

pymupdf4llm 1.28.2 skips whitespace-only spans, so monospaced code lines whose
spaces are separate spans come out squashed ("(+2135127)"). The text also keeps
the fonts' Private Use Area code points for ligatures, small capitals and
old-style figures, which render as nothing ("e" for "The", "prey" for
"pretty"). OCR engines are frozen, so this pass runs downstream: it reads the
checkpoints, writes repaired copies elsewhere and never edits the originals.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Linux Libertine PUA glyphs in the sicp PDF, identified by rendering each one in
# context (the CID-keyed fonts carry no glyph names).
LIGATURES = {
    0xE039: "ft",
    0xE03A: "ck",
    0xE03B: "ch",
    0xE03C: "tt",
    0xE048: "Qu",
    0xE049: "Th",
    0xE0E0: "f",  # an f kerned before ")"
}
SYMBOLS = {0xE050: "&", 0xE06D: "-"}
SMALL_CAPS = {0xE051 + i: chr(ord("A") + i) for i in range(26)}
OLDSTYLE_FIGURES = {0xE118 + i: str(i) for i in range(10)}
PUA_KINDS = {
    "ligature": LIGATURES,
    "symbol": SYMBOLS,
    "small capital": SMALL_CAPS,
    "old-style figure": OLDSTYLE_FIGURES,
}
PUA_MAP = {cp: text for table in PUA_KINDS.values() for cp, text in table.items()}

_FENCE = re.compile(r"^\s*```")
_MONOSPACED = 8  # PyMuPDF span flag


@dataclass
class RepairStats:
    code_lines_repaired: int = 0
    pua_mapped: Counter = field(default_factory=Counter)
    pua_unmapped: Counter = field(default_factory=Counter)
    skipped: list[dict] = field(default_factory=list)


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def map_pua(text: str, stats: RepairStats) -> str:
    """Replace known PUA glyphs; unknown ones are kept and counted."""
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in PUA_MAP:
            out.append(PUA_MAP[cp])
            stats.pua_mapped[cp] += 1
        else:
            if 0xE000 <= cp <= 0xF8FF:
                stats.pua_unmapped[cp] += 1
            out.append(ch)
    return "".join(out)


def code_lines(page) -> list[str]:
    """Monospaced lines of a PyMuPDF page, spaces kept.

    A line counts only when every non-blank span is monospaced, so inline code
    inside prose is left alone.
    """
    found = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line["spans"]
            inked = [s for s in spans if s["text"].strip()]
            if inked and all(s["flags"] & _MONOSPACED for s in inked):
                found.append("".join(s["text"] for s in spans).strip())
    return found


def repair_code(markdown: str, pdf_lines: list[str], page: int, stats) -> str:
    """Restore spacing in fenced code lines that equal a PDF line minus spaces.

    A line whose squashed form matches more than one distinct PDF line, or
    none, is kept as is and logged.
    """
    index: dict[str, set[str]] = {}
    for line in pdf_lines:
        index.setdefault(squash(line), set()).add(line)
    out, fenced = [], False
    for number, line in enumerate(markdown.split("\n"), 1):
        if _FENCE.match(line):
            fenced = not fenced
            out.append(line)
            continue
        key = squash(line)
        if not fenced or not key:
            out.append(line)
            continue
        matches = index.get(key, set())
        if len(matches) != 1:
            reason = "no PDF line" if not matches else "several PDF lines"
            stats.skipped.append(
                {"page": page, "line": number, "text": line, "reason": reason}
            )
            out.append(line)
            continue
        repaired = next(iter(matches))
        if repaired != line:
            stats.code_lines_repaired += 1
        out.append(repaired)
    return "\n".join(out)


def repair_page(markdown: str, pdf_lines: list[str], page: int, stats) -> str:
    return map_pua(repair_code(markdown, pdf_lines, page, stats), stats)
