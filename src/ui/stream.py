"""Turn state and a single selectable scrolling study surface."""

import time
from dataclasses import dataclass, field

from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static

from src.config import UI_SCROLLBACK_BLOCKS
from src.page_querying import collapse_page_ranges
from src.ui.render import StudyMarkdown, normalize_math, sanitize_answer


@dataclass(frozen=True)
class SourceRef:
    book: str
    physical_page: int | None = None
    chapter: str | None = None
    section: str | None = None
    physical_pages: tuple[int, ...] = ()
    kind: str = "book"
    url: str | None = None
    title: str | None = None

    def label(self):
        if self.kind == "web":
            return f"web: [{self.title or self.url}]({self.url})"
        return " · ".join(
            str(value)
            for value in (
                self.book,
                self.chapter,
                self.section,
                (
                    f"p. {collapse_page_ranges(list(self.physical_pages))}"
                    if self.physical_pages
                    else f"p. {self.physical_page}" if self.physical_page else None
                ),
            )
            if value
        )


@dataclass
class TurnRecord:
    request_id: str
    question: str
    book: str
    model: str
    raw_answer: str = ""
    copy_text: str = ""
    copy_is_sanitized: bool = False
    display_answer: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    phase: str = "SUBMITTED"
    steps: list[str] = field(default_factory=list)
    retrieval_query: str = ""

    def set_answer(self, raw):
        self.raw_answer = raw
        self.copy_text = sanitize_answer(raw)
        self.copy_is_sanitized = self.copy_text != raw
        self.display_answer = normalize_math(self.copy_text)

    def copy_payload(self):
        return (
            "[AxiomLM sanitized answer]\n" if self.copy_is_sanitized else ""
        ) + self.copy_text


class TurnView(Vertical):
    def __init__(self, record: TurnRecord):
        super().__init__(classes="turn")
        self.record = record
        self.question = Static("› " + record.question, classes="question", markup=False)
        self.step_line = Static("", classes="tool-line", markup=False)
        self.step_line.display = record.phase != "DONE"
        self.answer = StudyMarkdown(
            record.display_answer, classes="answer", open_links=False
        )
        self.source_line = Static("", classes="citation", markup=False)
        self.source_line.display = bool(record.sources)
        self.started_at = time.monotonic()
        self.tail = Static("▌", classes="stream-tail", markup=False)
        self.tail.display = False
        self._format_pending = False
        self.detail_line = Static("", classes="tool-line", markup=False)
        self.detail_line.display = False

    def compose(self):
        yield self.question
        yield self.step_line
        yield self.answer
        yield self.source_line
        yield self.detail_line
        yield self.tail

    def show_sources(self):
        self.source_line.display = True
        self.source_line.update(
            "retrieved evidence · not verified claim citations\n"
            + "\n".join(
                f"[{i}] {source.label()}"
                for i, source in enumerate(self.record.sources, 1)
            )
        )

    def collapse_sources(self):
        count = len(self.record.sources)
        self.source_line.display = bool(count)
        self.source_line.update(
            f"{count} retrieved sources · /sources" if count else ""
        )

    async def finalize(self, phase="DONE"):
        self.record.phase = phase
        self.tail.display = False
        self._format_pending = self.answer.source != self.record.display_answer
        await self.apply_pending_format()
        self.collapse_sources()
        self.step_line.display = bool(self.record.steps) or phase != "DONE"
        self.step_line.update(
            f"{len(self.record.steps)} steps · /details" if self.record.steps else ""
        )

    async def apply_pending_format(self):
        # Native full Markdown updates remove selected blocks; wait for selection to clear.
        if self._format_pending and not any(
            widget is self.answer or self.answer in widget.ancestors
            for widget in self.screen.selections
        ):
            await self.answer.update(self.record.display_answer)
            self._format_pending = False


class StudyStream(VerticalScroll):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dropped = 0
        self.summary = Static("", classes="tool-line", markup=False)
        self.summary.display = False
        self.spacer = Static("", id="stream_spacer")

    def compose(self):
        yield self.summary
        yield self.spacer

    def on_mount(self):
        self.anchor()

    async def add_turn(self, view):
        await self.mount(view, before=self.spacer)
        await self.trim_history()

    class Trimmed(Message):
        pass

    async def trim_history(self):
        removed = False
        blocks = list(self.query_children(".turn, .notice"))
        for oldest in blocks[: max(0, len(blocks) - UI_SCROLLBACK_BLOCKS)]:
            if isinstance(oldest, TurnView) and oldest.record.phase not in {
                "DONE",
                "ERROR",
                "CANCELLED",
            }:
                break
            await oldest.remove()
            self.dropped += 1
            removed = True
        self.summary.display = bool(self.dropped)
        self.summary.update(
            f"○ {self.dropped} earlier blocks omitted" if self.dropped else ""
        )
        if removed:
            self.post_message(self.Trimmed())
