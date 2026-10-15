"""Display math normalization and byte-preserving fenced code extraction."""

import asyncio
import re
import time
from dataclasses import dataclass
from typing import ClassVar

from src.config import UI_FRAME_INTERVAL


def _latex_to_unicode(text: str) -> str:
    """
    Post-process response to replace LaTeX math with readable Unicode symbols.
    Strips $$ and $ delimiters and handles common symbols/notations.
    """
    # 1. Strip structural delimiters and sizing commands
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", r"\1", text)
    text = re.sub(
        r"\\left\s*|\\right\s*|\\big\s*|\\Big\s*|\\bigg\s*|\\Bigg\s*", "", text
    )
    text = re.sub(r"\\text\s*\{\s*(.+?)\s*\}", r"\1", text)

    # 2. Symbol replacements (Comprehensive Math/Science library)
    # Using \s* to handle cases where the LLM adds spaces inside LaTeX commands
    replacements = {
        # Fractions and basic
        r"\\frac\s*\{\s*(.+?)\s*\}\s*\{\s*(.+?)\s*\}": lambda m: f"({m.group(1)})/({m.group(2)})",
        r"\\sqrt\s*\{\s*(.+?)\s*\}": lambda m: f"√({m.group(1)})",
        r"\\cdot\b": "·",
        r"\\times\b": "×",
        r"\\pm\b": "±",
        r"\\mp\b": "∓",
        r"\\div\b": "÷",
        # Comparison
        r"\\leq\b": "≤",
        r"\\geq\b": "≥",
        r"\\neq\b": "≠",
        r"\\approx\b": "≈",
        r"\\cong\b": "≅",
        r"\\equiv\b": "≡",
        r"\\propto\b": "∝",
        r"\\ll\b": "≪",
        r"\\gg\b": "≫",
        # Greek Lowercase
        r"\\alpha\b": "α",
        r"\\beta\b": "β",
        r"\\gamma\b": "γ",
        r"\\delta\b": "δ",
        r"\\epsilon\b": "ε",
        r"\\zeta\b": "ζ",
        r"\\eta\b": "η",
        r"\\theta\b": "θ",
        r"\\iota\b": "ι",
        r"\\kappa\b": "κ",
        r"\\lambda\b": "λ",
        r"\\mu\b": "μ",
        r"\\nu\b": "ν",
        r"\\xi\b": "ξ",
        r"\\pi\b": "π",
        r"\\rho\b": "ρ",
        r"\\sigma\b": "σ",
        r"\\tau\b": "τ",
        r"\\upsilon\b": "υ",
        r"\\phi\b": "φ",
        r"\\chi\b": "χ",
        r"\\psi\b": "ψ",
        r"\\omega\b": "ω",
        # Greek Uppercase
        r"\\Gamma\b": "Γ",
        r"\\Delta\b": "Δ",
        r"\\Theta\b": "Θ",
        r"\\Lambda\b": "Λ",
        r"\\Xi\b": "Ξ",
        r"\\Pi\b": "Π",
        r"\\Sigma\b": "Σ",
        r"\\Upsilon\b": "Υ",
        r"\\Phi\b": "Φ",
        r"\\Psi\b": "Ψ",
        r"\\Omega\b": "Ω",
        # Math Symbols
        r"\\sum\b": "∑",
        r"\\int\b": "∫",
        r"\\prod\b": "∏",
        r"\\infty\b": "∞",
        r"\\partial\b": "∂",
        r"\\nabla\b": "∇",
        r"\\angle\b": "∠",
        r"\\triangle\b": "△",
        r"\\hbar\b": "ħ",
        # Set theory & Logic
        r"\\in\b": "∈",
        r"\\notin\b": "∉",
        r"\\forall\b": "∀",
        r"\\exists\b": "∃",
        r"\\cap\b": "∩",
        r"\\cup\b": "∪",
        r"\\subset\b": "⊂",
        r"\\subseteq\b": "⊆",
        r"\\supset\b": "⊃",
        r"\\supseteq\b": "⊇",
        r"\\emptyset\b": "∅",
        r"\\land\b": "∧",
        r"\\lor\b": "∨",
        r"\\neg\b": "¬",
        # Arrows
        r"\\rightarrow\b": "→",
        r"\\Rightarrow\b": "⇒",
        r"\\leftarrow\b": "←",
        r"\\Leftarrow\b": "⇐",
        r"\\leftrightarrow\b": "↔",
        r"\\Leftrightarrow\b": "⇔",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
        text = re.sub(pattern, replacement, text)

    # 3. Simple Subscripts and Superscripts (Common in Science)
    sub_super = {
        r"\^2\b": "²",
        r"\^3\b": "³",
        r"\^n\b": "ⁿ",
        r"_2\b": "₂",
        r"_3\b": "₃",
        r"_4\b": "₄",
        r"_n\b": "ₙ",
        r"_i\b": "ᵢ",
        r"_j\b": "ⱼ",
        r"_x\b": "ₓ",
        r"_y\b": "ᵧ",
    }
    for pattern, replacement in sub_super.items():
        text = re.sub(pattern, replacement, text)
        text = re.sub(pattern, replacement, text)

    return text


@dataclass(frozen=True)
class CodeBlock:
    language: str
    source: str


def _fenced_spans(text):
    lines = text.splitlines(keepends=True)
    offset = 0
    start = body = None
    marker = language = ""
    for line in lines:
        if start is None:
            opening = re.match(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)\r?\n?$", line)
            if opening:
                marker = opening[1]
                language = opening[2].strip().split(" ", 1)[0]
                start, body = offset, offset + len(line)
        elif re.fullmatch(
            r" {0,3}"
            + re.escape(marker[0])
            + "{"
            + str(len(marker))
            + r",}[ \t]*\r?\n?",
            line,
        ):
            yield start, offset + len(line), CodeBlock(language, text[body:offset])
            start = body = None
        offset += len(line)
    if start is not None:
        yield start, len(text), CodeBlock(language, text[body:])


def extract_fenced_blocks(raw_text: str) -> list[CodeBlock]:
    return [block for _start, _end, block in _fenced_spans(raw_text)]


def _map_prose(text, transform):
    def inline(value):
        pieces, cursor, search = [], 0, 0
        while opening := re.search(r"`+", value[search:]):
            start, end = search + opening.start(), search + opening.end()
            prefix = value[:start]
            if (len(prefix) - len(prefix.rstrip("\\"))) % 2:
                search = end
                continue
            size = end - start
            closing = next(
                (
                    match
                    for match in re.finditer(r"`+", value[end:])
                    if len(match[0]) == size
                ),
                None,
            )
            stop = end + closing.end() if closing else len(value)
            pieces.extend((transform(value[cursor:start]), value[start:stop]))
            cursor = search = stop
        pieces.append(transform(value[cursor:]))
        return "".join(pieces)

    pieces, cursor = [], 0
    for start, end, _block in _fenced_spans(text):
        pieces.extend((inline(text[cursor:start]), text[start:end]))
        cursor = end
    pieces.append(inline(text[cursor:]))
    return "".join(pieces)


def normalize_math(text: str) -> str:
    def prose(value):
        value = re.sub(
            r"\$\$(.*?)\$\$|\\\[(.*?)\\\]",
            lambda m: "\n    " + _latex_to_unicode(m[1] or m[2]).strip() + "\n",
            value,
            flags=re.DOTALL,
        )
        value = _latex_to_unicode(value)
        value = re.sub(r"\\le\b", "≤", value)
        return re.sub(r"\\ge\b", "≥", value)

    return _map_prose(text, prose)


# Extend native Markdown blocks so streaming and selection remain Textual-owned.
from copy import deepcopy

from rich import box
from rich.table import Table
from textual.content import Content, Span
from textual.widgets import Markdown, Static
from textual.widgets._markdown import MarkdownFence, MarkdownTable


class StudyFence(MarkdownFence):
    @classmethod
    def highlight(cls, code, language, ansi=False, dark=False):
        lines = code.splitlines(keepends=True)
        text = (language + "\n" if language else "") + "".join(
            "│ " + line for line in lines
        )
        spans = []
        cursor = 0
        if language:
            spans.append(Span(0, len(language), "dim"))
            cursor = len(language) + 1
        for line in lines:
            spans.append(Span(cursor, cursor + 1, "dim"))
            cursor += len(line) + 2
        return Content(text, spans)


class StudyEquation(MarkdownFence):
    @classmethod
    def highlight(cls, code, language, ansi=False, dark=False):
        return Content("    " + code)


class StudyTable(MarkdownTable):
    def compose(self):
        headers, rows = self._get_headers_and_rows()
        table = Table(
            box=box.SIMPLE_HEAD,
            show_edge=False,
            pad_edge=False,
            header_style="bold",
            border_style="dim",
            collapse_padding=True,
        )
        for header in headers:
            table.add_column(header.plain)
        for row in rows:
            table.add_row(*(cell.plain for cell in row))
        yield Static(table)


class StudyMarkdown(Markdown):
    BULLETS: ClassVar = ["- "]
    BLOCKS: ClassVar = dict(
        Markdown.BLOCKS,
        fence=StudyFence,
        code_block=StudyEquation,
        table_open=StudyTable,
    )

    _last_append = float("-inf")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._append_budget_lock = asyncio.Lock()

    async def append(self, fragment):
        # Native get_stream may queue writes while an earlier parse is slow.
        async with self._append_budget_lock:
            delay = UI_FRAME_INTERVAL - (time.monotonic() - self._last_append)
            if delay > 0:
                await asyncio.sleep(delay)
            await super().append(fragment)
            self._last_append = time.monotonic()

    def _parse_markdown(self, tokens):
        tokens = deepcopy(list(tokens))
        for token in tokens:
            if token.type == "list_item_open" and token.markup in {"*", "+", "-"}:
                token.markup = "-"
            for child in token.children or []:
                if child.type == "image":
                    child.type = "text"
                    child.content = child.attrs.get("alt", "")
                if child.type == "code_inline":
                    child.content = "`" + child.content + "`"

        def mute_ticks(block):
            content = block._content
            for span in list(content.spans):
                if span.style == ".code_inline":
                    content = content.stylize("dim", span.start, span.start + 1)
                    content = content.stylize("dim", span.end - 1, span.end)
            block.set_content(content)
            for child in block._blocks:
                mute_ticks(child)

        for block in super()._parse_markdown(tokens):
            mute_ticks(block)
            yield block


ABSTENTION = (
    "The indexed material does not provide enough evidence to answer this question."
)


def sanitize_answer(raw: str) -> str:
    if raw.lstrip().startswith(ABSTENTION):
        return ABSTENTION

    def prose(value):
        value = re.sub(r"(?m)^\[\^?[\d⁰¹²³⁴⁵⁶⁷⁸⁹]+\]:[^\n]*\n?", "", value)
        value = re.sub(r"[ \t]*\[\^\d+\]", "", value)
        value = re.sub(r"(?<!\w)\[(?:\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)\](?!\w)", "", value)
        value = re.sub(r"[ \t]+([.,;:!?])", r"\1", value)
        return re.sub(r"\n{3,}", "\n\n", value)

    return _map_prose(raw, prose).strip()


class SafeAnswerBuffer:
    """Retain original bytes while holding ambiguous refusal/citation prefixes."""

    def __init__(self):
        self.raw = ""
        self.visible = ""

    def feed(self, fragment: str) -> str:
        self.raw += fragment
        candidate = self.raw.lstrip()
        if ABSTENTION.startswith(candidate) and candidate != ABSTENTION:
            return ""
        if candidate.startswith(ABSTENTION):
            candidate = ABSTENTION
        else:
            # Hold a potential definition line or inline footnote until it resolves.
            last_line = candidate.rsplit("\n", 1)[-1]
            if re.fullmatch(r"\[(?:\^?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]*\]?:?)?", last_line):
                candidate = candidate[: -len(last_line)] if last_line else candidate
            candidate = re.sub(r"\[(?:\^?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]*)?$", "", candidate)
            candidate = sanitize_answer(candidate)
        if not candidate.startswith(self.visible):
            raise RuntimeError("Answer sanitizer attempted to rewrite visible output")
        delta = candidate[len(self.visible) :]
        self.visible = candidate
        return delta

    def finish(self) -> str:
        return sanitize_answer(self.raw)
