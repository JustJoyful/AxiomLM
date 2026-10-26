"""Command-driven Textual study stream and local workload orchestration."""

import asyncio
import importlib.util
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import ClassVar
from uuid import uuid4

from httpx import Client, ConnectError, HTTPError, HTTPStatusError, ReadTimeout
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Static
from textual.worker import WorkerCancelled, WorkerFailed

from . import config
from .clipboard import copy_text
from .config import (
    DISTANCE_THRESHOLD,
    INDEX_TIMEOUT_SECONDS,
    LOGS_DIR,
    MARKER_OCR_TIMEOUT_SECONDS,
    MINERU_OCR_TIMEOUT_SECONDS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    PROCESS_TERMINATE_GRACE_SECONDS,
    PYMUPDF4LLM_OCR_TIMEOUT_SECONDS,
    TOP_K_RESULTS,
)
from .llm import OllamaClient
from .page_querying import build_citation as build_page_citation
from .page_querying import (
    collapse_page_ranges,
    physical_pages_for,
    strip_model_footnotes,
)
from .page_querying import extract_page_intent as parse_page_intent
from .query_rewriting import ConversationExchange, rewrite_followup
from .ui.commands import (
    COMMANDS,
    AxiomCommandProvider,
    CommandOverlay,
    dispatch_command,
)
from .ui.motion import MotionClock, StudyScreen, render_progress
from .ui.prompt import QueryComposer, complete_pdf_path
from .ui.render import SafeAnswerBuffer, extract_fenced_blocks
from .ui.render import normalize_math as latex_to_unicode
from .ui.settings import load_settings, save_theme_selection
from .ui.stream import SourceRef, StudyStream, TurnRecord, TurnView
from .ui.terminal import PaletteDriver
from .ui.themes import (
    ANSI_COLORS,
    THEMES,
    ThemePreview,
    load_custom_themes,
    resolve_theme,
)
from .workloads import WorkloadCoordinator, WorkloadKind


# Delay embedding/database imports until background discovery or retrieval.
def list_collections():
    from .index_store import list_books as operation

    return operation()


def get_collection(book):
    from .index_store import resolve_index

    return resolve_index(book).dense_collection


def embed_query(query):
    from .db import embed_query as operation

    return operation(query)


def ensure_collection_ready(collection):
    from .db import ensure_collection_ready as operation

    return operation(collection)


def delete_collection(book):
    from .index_store import delete_book_indexes as operation

    return operation(book)


LOGO = """ █████╗ ██╗  ██╗██╗ ██████╗ ███╗   ███╗
██╔══██╗╚██╗██╔╝██║██╔═══██╗████╗ ████║
███████║ ╚███╔╝ ██║██║   ██║██╔████╔██║
██╔══██║ ██╔██╗ ██║██║   ██║██║╚██╔╝██║
██║  ██║██╔╝ ██╗██║╚██████╔╝██║ ╚═╝ ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝     ╚═╝"""

PROMPT_TEMPLATE = """You are Axie, a study assistant built into a terminal UI. Responses render in a Rich-enabled terminal supporting markdown tables and markup.

Important: You have access to conversation history below. Use it only to understand references like "this topic", "that exercise", or "the previous example". Use retrieved context to support factual claims.

Formatting rules:
- Use **bold** for key terms and important values
- Use `code` for inline code/symbols, triple backticks for code blocks with language tag
- Avoid LaTeX commands like \frac, \\left, \right, or \text. Use plain notation like P(A∩B)/P(A).
- Use a simple pipe | or forward slash / for math. NEVER use backslashes followed by spaces or slashes (like \\ /).
- If the user asks for examples/questions, prioritize any retrieved sections titled 'Illustrative Examples', 'Exercises', or 'Solved Problems'.
- Use the pipe symbol | for conditional probability, e.g., P(B|A). Never use fractions for this.
- For Chemistry, use subscripts like H₂O or CO₂. For Physics, use common superscripts like m/s² when possible.
- Use numbered lists for steps/derivations, bullet lists for properties
- Use markdown tables for comparisons, definitions, structured data
- Keep answers dense and scannable — study notes style, not essay style

Conversation history (use this to understand what we're discussing):
{history}

Retrieved context from {book}:
{context}

Current question: {question}

Evidence rules override formatting and verbosity: Every factual sentence must be supported by a retrieved excerpt. Conversation history is never evidence; use it only to resolve references. If retrieved context does not support the answer, say only: 'The indexed material does not provide enough evidence to answer this question.' If it supports only part of the answer, state only those supported facts and identify the gap. Never infer book content from its title, previous answers, or general knowledge. Do not invent or alter examples or code; reuse only examples present in retrieved context. Do not write citations; the application adds citations from retrieved page metadata."""


class AxiomLMApp(App):
    """One study surface; workers own local inference and ingestion."""

    TITLE = "AxiomLM"
    CSS_PATH = config.UI_STYLESHEET_PATH
    COMMANDS: ClassVar = {AxiomCommandProvider}
    ENABLE_COMMAND_PALETTE = False
    top_k_results = reactive(TOP_K_RESULTS)
    distance_threshold = reactive(DISTANCE_THRESHOLD)
    BINDINGS: ClassVar = [
        Binding("ctrl+p", "commands", show=False, priority=True),
        Binding("tab", "complete_prompt", show=False, priority=True),
        Binding("escape", "escape", show=False, priority=True),
        Binding("ctrl+c", "interrupt", show=False, priority=True),
        Binding("ctrl+shift+c,super+c", "copy_selection", show=False, priority=True),
        Binding("ctrl+l", "clear_chat", show=False, priority=True),
        Binding("pageup", "page_up", show=False, priority=True),
        Binding("pagedown", "page_down", show=False, priority=True),
        Binding("end", "follow", show=False, priority=True),
    ]

    def __init__(self):
        super().__init__(
            driver_class=PaletteDriver if sys.platform == "linux" else None
        )
        self.active_collection_name = None
        self.selected_model = OLLAMA_MODEL
        self._book_counts = {}
        self._models = []
        self._conversation = []
        self._last_retrieved_pages = []
        self._request_books = {}
        self._request_context = {}
        self._request_exchanges = {}
        self._transport_closed = {}
        self._answer_streams = {}
        self._answer_flush_lock = asyncio.Lock()
        self._last_answer_flush = float("-inf")
        self._answer_flush_worker = None
        self._pending_assistants = {}
        self._active_query_worker = None
        self._active_query_request_id = None
        self._cancelled_query_ids = set()
        self._query_workload_token = None
        self._parse_workload_token = None
        self._workload_coordinator = WorkloadCoordinator()
        self._is_parsing = False
        self._use_gpu_for_embeddings = True
        self._active_processes = set()
        self._process_lock = Lock()
        self._process_shutdown_started = False
        self._parse_started_at = None
        self._parse_current_stage = "Idle"
        self._parse_current_engine = ""
        self._parse_detail = ""
        self._ingest_cancelled = False
        self._ingest_view = None
        self._ingest_path = ""
        self.queued_query = None
        self.web_enabled = config.WEB_ENABLED_DEFAULT
        self._agent_routing = config.AGENT_ROUTING_ENABLED
        self._request_web = {}
        self._sandbox = None
        self.turns = []
        self.notices = []
        self._home_visible = True
        self._status = "○ checking…"
        self._terminal_focused = True
        self._focus_reports_seen = False
        self.motion = MotionClock(self)
        self._copy_confirmation = None
        self._copy_expiry = 0
        self._completion_flash_until = 0
        self._completion_view = None
        self._completion_text = ""
        self._ingest_stage = "preparing"
        self._ingest_counts = {}
        self._ingest_completed = set()
        self._progress_shown = 0
        self._progress_updated = 0
        self._ingest_count_started = time.monotonic()
        self._pending_remove = None
        self._unavailable_books = {}
        self._deleting = False
        self._last_interrupt = 0.0
        self.overlay = None
        self._preview = None
        self._warned_truecolor = False
        self._warned_ansi_fallback = False
        self._overlay_suppressed = False
        self._last_question = ""
        self.settings_path = config.UI_CONFIG_PATH
        self.settings, self._settings_errors = load_settings(
            self.settings_path, os.environ
        )
        self.custom_themes, errors = load_custom_themes(config.UI_THEMES_DIR)
        self._settings_errors.extend(errors)
        if any(error.startswith("✗") for error in errors):
            self.settings.theme = "terminal"

    def get_default_screen(self):
        return StudyScreen(id="_default")

    async def on_load(self):
        self.set_class(self._home_visible, "home")
        await resolve_theme(self, self.settings, os.environ)

    async def _on_compose(self, event):
        event.prevent_default()
        await resolve_theme(self, self.settings, os.environ)
        await super()._on_compose()

    def compose(self) -> ComposeResult:
        yield Static("resize terminal", id="resize_warning", markup=False)
        with Vertical(id="surface"):
            yield Static("", id="header", markup=False)
            with StudyStream(id="chat_scroll"), Vertical(id="home"):
                yield Static(LOGO, id="logo", markup=False)
                yield Static(
                    "Your textbooks. Locally. Answered.", id="tagline", markup=False
                )
            yield Static("", id="new_output", markup=False)
            yield Static("", id="rule", markup=False)
            with Horizontal(id="prompt_row"):
                yield Static("›", id="prompt_glyph", markup=False)
                yield QueryComposer(
                    id="query_input",
                    compact=True,
                    highlight_cursor_line=False,
                    placeholder="Ask about a book, or type / for commands",
                )
                yield Static("", id="ghost", markup=False)
            yield Static(
                "○ no book loaded · /parse <path>", id="condition", markup=False
            )
            yield Static("", id="hint", markup=False)

    def on_mount(self):
        self.query_one(QueryComposer).focus()
        self._layout_changed()
        self.run_worker(
            self._run_startup_checks(), group="startup", exit_on_error=False
        )

    async def _run_startup_checks(self):
        async def books():
            try:
                discovered = await asyncio.to_thread(self._discover_books)
                self._book_counts = discovered
                if self.active_collection_name is None and discovered:
                    self.active_collection_name = next(iter(discovered))
                for book, error in self._unavailable_books.items():
                    await self.write_line(
                        f"✗ {book}: incomplete or unavailable index ({error}) · /remove {shlex.quote(book)} or /parse <path>"
                    )
                self._refresh_condition()
                self._set_header()
            except Exception as exc:  # noqa: BLE001
                await self.write_line(f"✗ book discovery failed: {exc} · /books")

        async def models():
            try:
                self._models = await asyncio.to_thread(self._fetch_ollama_models)
                if not self.selected_model:
                    await self.write_line("○ no model · /model <name>")
                elif self.selected_model not in self._models:
                    await self.write_line(
                        f"✗ model {self.selected_model} not found · /models"
                    )
                self._status = "● ready"
            except Exception:  # noqa: BLE001
                self._status = "○ offline"
                await self.write_line(
                    "✗ can't reach Ollama at localhost:11434 · start it, then /retry"
                )
            self._set_header()
            self._refresh_condition()

        await self._theme_notice(self.settings.theme)
        for error in self._settings_errors:
            await self.write_line(error)
        await asyncio.gather(books(), models())

    def _discover_books(self):
        result, unavailable = {}, {}
        for book in sorted(list_collections()):
            try:
                collection = get_collection(book)
                ensure_collection_ready(collection)
                result[book] = collection.count()
            except Exception as exc:  # noqa: BLE001
                unavailable[book] = str(exc).splitlines()[0] or type(exc).__name__
        self._unavailable_books = unavailable
        return result

    def _refresh_condition(self):
        condition = self.query_one("#condition", Static)
        condition.display = self._home_visible
        if self.active_collection_name:
            condition.update(
                self._status
                + " · "
                + self._format_book_name(self.active_collection_name)
            )
        else:
            condition.update("○ no book loaded · /parse <path>")

    def _set_header(self):
        if not self.is_mounted:
            return
        width = self.size.width
        if width < 60:
            left, right = "AXIOM", self._status.split()[0]
        else:
            book = self.active_collection_name
            left = "AxiomLM" + (
                f" · {self._format_book_name(book)} ({self._book_counts.get(book, 0)})"
                if book
                else ""
            )
            if width >= 100 and self.selected_model:
                left += " · " + self.selected_model
            right = self._status
        text = Text(left, style="bold" if width < 60 else "")
        right_text = Text(right)
        available = max(0, width - 4 - right_text.cell_len - 1)
        text.truncate(available, overflow="ellipsis")
        text.append(" " * max(1, width - 4 - text.cell_len - right_text.cell_len))
        text.append_text(right_text)
        self.query_one("#header", Static).update(text)

    def _format_book_name(self, stem):
        return stem.replace("_", " ").title()

    def on_resize(self, event):
        if self.is_mounted:
            self._layout_changed(event.size)

    def _layout_changed(self, size=None):
        size = size or self.size
        tiny = size.width < 40 or size.height < 12
        self.query_one("#resize_warning").display = tiny
        self.query_one("#surface").display = not tiny
        self.query_one("#logo", Static).update("AXIOM" if size.width < 60 else LOGO)
        self.query_one("#tagline").display = size.height >= 24
        self.query_one("#home").display = self._home_visible
        self.query_one("#header").display = not self._home_visible
        self.query_one("#rule", Static).update("─" * max(0, size.width - 4))
        self._set_header()
        self._update_hint()

    def _hide_home(self):
        self._home_visible = False
        self.remove_class("home")
        self.query_one("#home").display = False
        self.query_one("#header").display = True
        self.query_one("#query_input", QueryComposer).placeholder = ""
        self._refresh_condition()

    def _busy(self):
        return (
            self._workload_coordinator.active_kind is not None
            or self._deleting
            or self.queued_query is not None
        )

    async def write_line(self, text):
        self.notices.append(text)
        self.notices = self.notices[-config.UI_SCROLLBACK_BLOCKS :]
        stream = self.query_one(StudyStream)
        await stream.mount(
            Static(
                text,
                classes="notice failed" if text.startswith("✗") else "notice tool-line",
                markup=False,
            ),
            before=stream.spacer,
        )
        await stream.trim_history()

    def on_study_stream_trimmed(self, _event):
        self.turns[:] = [view for view in self.turns if view.is_attached]

    async def on_study_screen_selection_changed(self, _event):
        for view in self.turns:
            if view.is_attached:
                await view.apply_pending_format()

    def _update_hint(self):
        if not self.is_mounted:
            return
        if self.overlay:
            hint = (
                "↑↓ preview   enter keep   esc revert"
                if self.overlay.kind == "themes"
                else "↑↓ move   tab fill   enter run   esc close"
            )
        elif self._busy():
            hint = "esc cancel"
        else:
            hint = "enter send   shift+enter newline   / commands"
        self.query_one("#hint", Static).update(hint)

    def on_text_area_changed(self, event):
        if event.control.id != "query_input":
            return
        text = event.control.text
        name = text.split(" ", 1)[0]
        spec = COMMANDS.get(name)
        self.query_one("#ghost", Static).update(
            spec.usage.split(" ", 1)[0] if spec and " " not in text else ""
        )
        if self._overlay_suppressed:
            self._overlay_suppressed = False
            return
        if text.startswith("/") and " " not in text and "\n" not in text:
            self.run_worker(
                self._show_overlay([n for n in COMMANDS if n.startswith(text)]),
                group="completion",
                exclusive=True,
                exit_on_error=False,
            )
        elif self.overlay and self.overlay.kind == "commands":
            self.run_worker(
                self.close_overlay(),
                group="completion",
                exclusive=True,
                exit_on_error=False,
            )

    async def _show_overlay(self, names, kind="commands"):
        if self.overlay:
            if self.overlay.kind == "themes" and self._preview:
                await self._preview.revert()
                self._preview = None
            await self.overlay.remove()
        labels = {}
        if kind == "themes" and os.environ.get("COLORTERM") not in {
            "truecolor",
            "24bit",
        }:
            for name in names:
                spec = (THEMES | self.custom_themes).get(name)
                if spec and any(
                    value.startswith("#")
                    for value in vars(spec).values()
                    if isinstance(value, str)
                ):
                    labels[name] = name + " · needs truecolor"
        self.overlay = CommandOverlay(names, kind=kind, labels=labels)
        await self.query_one("#surface").mount(self.overlay)
        self._update_hint()

    async def action_commands(self):
        await self._show_overlay(list(COMMANDS))
        self.overlay.focus()

    async def close_overlay(self, revert=True):
        if self.overlay:
            await self.overlay.remove()
            self.overlay = None
        if self._preview and revert:
            await self._preview.revert()
        self._preview = None
        self.query_one(QueryComposer).focus()
        self._update_hint()

    def fill_command(self, command):
        self._overlay_suppressed = True
        prompt = self.query_one(QueryComposer)
        prompt.load_text(
            command + (" " if COMMANDS.get(command) and COMMANDS[command].usage else "")
        )
        prompt.move_cursor(prompt.document.end)
        prompt.focus()

    def move_overlay(self, direction):
        if self.overlay:
            if direction < 0:
                self.overlay.action_cursor_up()
            else:
                self.overlay.action_cursor_down()

    async def action_complete_prompt(self):
        await self.on_query_composer_completion_requested(None)

    async def on_query_composer_completion_requested(self, _event):
        if self.overlay and self.overlay.highlighted is not None:
            name = self.overlay.get_option_at_index(self.overlay.highlighted).id
            self.fill_command(name)
            await self.close_overlay()
            return
        prompt = self.query_one(QueryComposer)
        if prompt.text.startswith("/parse "):
            prefix = prompt.text[len("/parse ") :].strip().strip("\"'")
            matches = await asyncio.to_thread(complete_pdf_path, prefix)
            if matches:
                self._overlay_suppressed = True
                prompt.load_text("/parse " + shlex.quote(matches[0]))
                prompt.move_cursor(prompt.document.end)

    async def _theme_notice(self, name):
        if self.theme == "textual-ansi" and not self._warned_ansi_fallback:
            self._warned_ansi_fallback = True
            await self.write_line(
                "○ terminal light/dark detection unavailable · using textual-ansi"
            )
        spec = (THEMES | self.custom_themes).get(name)
        if (
            spec
            and any(
                value.startswith("#")
                for value in vars(spec).values()
                if isinstance(value, str)
            )
            and os.environ.get("COLORTERM") not in {"truecolor", "24bit"}
            and not self._warned_truecolor
        ):
            self._warned_truecolor = True
            await self.write_line("○ truecolor unavailable · try /theme terminal")

    async def on_option_list_option_highlighted(self, event):
        if (
            self.overlay is event.option_list
            and self.overlay.kind == "themes"
            and self._preview
        ):
            await self._preview.preview(event.option.id)
            await self._theme_notice(event.option.id)

    async def on_option_list_option_selected(self, event):
        if self.overlay is not event.option_list:
            return
        name, kind = event.option.id, self.overlay.kind
        if kind == "themes":
            try:
                await self._preview.keep(name)
            except (OSError, ValueError) as exc:
                await self.write_line(
                    f"✗ theme save failed: {exc} · previous theme restored"
                )
            await self.close_overlay(revert=False)
        elif kind == "settings":
            self.fill_command(name)
            await self.close_overlay()
        elif kind == "books":
            await self.close_overlay()
            await self.handle_command("/book", [name])
        else:
            await self.close_overlay()
            await dispatch_command(self, name)

    async def action_escape(self):
        if self.overlay:
            await self.close_overlay()
        elif self._busy():
            self.action_cancel_query()
        else:
            self.query_one(QueryComposer).clear()

    async def action_copy_selection(self):
        selected = self.screen.get_selected_text()
        if not selected and isinstance(self.focused, QueryComposer):
            selected = self.focused.selected_text
        if selected:
            await self._copy(selected)
            self._last_interrupt = 0.0
            return True
        return False

    async def action_interrupt(self):
        if await self.action_copy_selection():
            return
        if self.overlay or self._busy():
            await self.action_escape()
        elif time.monotonic() - self._last_interrupt <= config.UI_QUIT_CONFIRM_SECONDS:
            await self.action_quit()
        else:
            self._last_interrupt = time.monotonic()
            await self.write_line("○ press Ctrl+C again to quit")

    def action_page_up(self):
        self.query_one(StudyStream).scroll_page_up(animate=False)

    def action_page_down(self):
        self.query_one(StudyStream).scroll_page_down(animate=False)

    def action_follow(self):
        self.query_one(StudyStream).anchor()
        self.query_one("#new_output", Static).update("")

    async def action_clear_chat(self):
        if self._busy():
            await self.write_line("○ work is active · /cancel before clearing")
            return
        stream = self.query_one(StudyStream)
        await stream.query(".turn, .notice").remove()
        self.turns.clear()
        self.notices.clear()
        stream.dropped = 0
        stream.summary.update("")
        stream.summary.display = False

    async def on_query_composer_submitted(self, event):
        query = event.value.strip()
        if not query:
            return
        if self.overlay:
            if self.overlay.kind != "commands":
                self.overlay.action_select()
                return
            if (
                " " not in query
                and (option := self.overlay.highlighted_option) is not None
                and option.id.startswith(query)
            ):
                query = option.id
            await self.close_overlay()
        prompt = self.query_one(QueryComposer)
        prompt.clear()
        if self._pending_remove:
            book = self._pending_remove
            self._pending_remove = None
            if query.lower() == "y":
                await self._remove_book(book)
            else:
                await self.write_line("○ removal cancelled")
            return
        if query.startswith("/"):
            await dispatch_command(self, query)
            return
        if self._busy():
            active = self._pending_assistants.get(self._active_query_request_id)
            if (
                active
                and active.record.phase not in {"CANCELLED", "ERROR"}
                and not self._is_parsing
            ):
                if self.queued_query is None:
                    self.queued_query = (
                        query,
                        self.active_collection_name,
                        self.selected_model,
                    )
                    await self.write_line("○ queued")
                else:
                    await self.write_line(
                        "○ queue full · one question is already queued"
                    )
                    prompt.load_text(query)
            else:
                await self.write_line("○ work is active · /cancel or wait")
                prompt.load_text(query)
            return
        self._hide_home()
        if not self.active_collection_name:
            await self.write_line("○ no book loaded · /parse <path>")
            return
        if not self.selected_model:
            await self.write_line("○ no model · /model <name>")
            return
        await self._submit_question(
            query, self.active_collection_name, self.selected_model
        )

    async def _flush_answer(self, request_id):
        async with self._answer_flush_lock:
            entry = self._answer_streams.get(request_id)
            if not entry or request_id != self._active_query_request_id:
                return
            updater, pending = entry
            if not pending:
                return
            delay = config.UI_FRAME_INTERVAL - (
                time.monotonic() - self._last_answer_flush
            )
            if delay > 0:
                await asyncio.sleep(delay)
            fragment = "".join(pending)
            pending.clear()
            self._last_answer_flush = time.monotonic()
            await updater.write(fragment)

    def _motion_active(self):
        return bool(
            self._active_query_request_id
            or self._is_parsing
            or self._copy_confirmation is not None
            or time.monotonic() < self._completion_flash_until
        )

    def on_app_focus(self):
        self._terminal_focused = True
        self._focus_reports_seen = True
        if self._motion_active():
            self.motion.start()

    def on_app_blur(self):
        self._terminal_focused = False
        self._focus_reports_seen = True
        self.motion.stop()

    def _motion_frame(self, now, clock):
        if self._copy_confirmation is not None and now >= self._copy_expiry:
            self._copy_confirmation.remove()
            self._copy_confirmation = None
        view = self._pending_assistants.get(self._active_query_request_id)
        if view:
            entry = self._answer_streams.get(view.record.request_id)
            if (
                entry
                and entry[1]
                and (
                    self._answer_flush_worker is None
                    or not self._answer_flush_worker.is_running
                )
            ):
                self._answer_flush_worker = self.run_worker(
                    self._flush_answer(view.record.request_id),
                    group="answer-render",
                    exit_on_error=False,
                )
            record = view.record
            elapsed = clock.elapsed(view.started_at, now)
            if record.phase in {"SUBMITTED", "RETRIEVING", "THINKING"}:
                label = (
                    "thinking" if record.phase == "THINKING" else "search_local_index"
                )
                text = Text(
                    "\n".join(record.steps) + ("\n" if record.steps else ""),
                    style="dim",
                )
                text.append("↳ ")
                text.append_text(clock.label(label, now))
                text.append(f" {clock.spinner} {elapsed}")
                view.step_line.update(text)
                self._status = f"{clock.spinner} working"
                view.tail.display = record.phase == "THINKING" and clock.blink(now)
            elif record.phase == "STREAMING":
                self._status = "▌ writing"
                view.tail.display = True
            else:
                view.tail.display = False
            if view.tail.display:
                self._position_tail(view)
            stream = self.query_one(StudyStream)
            if record.phase == "STREAMING" and not stream.is_vertical_scroll_end:
                self.query_one("#new_output", Static).update("↓ new output · End")
        if self._is_parsing and self._ingest_view:
            lines = Text()
            for stage in ("parse", "chunk", "embed", "index"):
                if lines.plain:
                    lines.append("\n")
                lines.append("↳ ")
                lines.append_text(
                    clock.label(stage, now)
                    if stage == self._ingest_stage
                    else Text(stage, style="dim")
                )
                if stage in self._ingest_completed:
                    lines.append(" ✓", style="dim")
                elif stage == self._ingest_stage:
                    counts = self._ingest_counts.get(stage)
                    if counts:
                        done, total = counts
                        actual = done / total
                        if now - self._progress_updated >= config.UI_COUNTER_INTERVAL:
                            self._progress_shown += (
                                actual - self._progress_shown
                            ) * config.UI_PROGRESS_EASING
                            self._progress_updated = now
                        progress = render_progress(
                            round(self._progress_shown * total * 8),
                            total * 8,
                            config.UI_PROGRESS_WIDTH,
                        )
                        rate_elapsed = max(0, now - self._ingest_count_started)
                        eta = (
                            f"~{(total-done)*rate_elapsed/done:.0f}s left"
                            if done and rate_elapsed
                            else "ETA unknown"
                        )
                        lines.append(
                            f" {progress} {done}/{total} {int(actual*100)}% {clock.spinner} {eta}"
                        )
                    else:
                        lines.append(f" {clock.spinner} · ETA unknown")
                else:
                    lines.append(" ○", style="dim")
            if self._parse_detail:
                lines.append("\n" + self._parse_detail, style="dim")
            self._ingest_view.step_line.update(lines)
            self._status = f"{clock.spinner} working"
        if self._completion_view is not None:
            self._completion_view.step_line.update(
                Text(
                    self._completion_text,
                    style="bold" if now < self._completion_flash_until else "dim",
                )
            )
            if now >= self._completion_flash_until:
                self._completion_view = None
        self._set_header()
        if not self._motion_active():
            clock.stop()

    def _position_tail(self, view):
        leaves = [
            widget for widget in view.answer.query(Static) if widget.content_size.height
        ]
        leaf = leaves[-1] if leaves else view.answer
        row = max(0, leaf.content_size.height - 1)
        line = leaf.render_line(row).text.rstrip() if leaves else ""
        x = leaf.content_region.x - view.region.x + Text(line).cell_len
        y = leaf.content_region.y - view.region.y + row
        view.tail.styles.offset = (
            min(max(0, x), max(0, view.size.width - 1)),
            max(0, y),
        )

    def _ingest_output(self, fallback_stage, line):
        text = line.strip()
        if not text:
            return
        stage = (
            self._ingest_stage
            if self._ingest_stage in {"parse", "chunk", "embed", "index"}
            else fallback_stage
        )
        if fallback_stage == "parse":
            stage = "parse"
        elif re.search(
            r"\b(?:embedding|embeddings|embedded|encoding|adding batch)\b",
            text,
            re.IGNORECASE,
        ):
            stage = "embed"
        elif re.search(r"\bchunk(?:ing|s)?\b", text, re.IGNORECASE):
            stage = "chunk"
        elif re.search(r"\bindex(?:ing)?\b", text, re.IGNORECASE):
            stage = "index"
        elif fallback_stage == "parse":
            stage = "parse"
        if stage != self._ingest_stage:
            if self._ingest_stage in {"parse", "chunk", "embed", "index"}:
                self._ingest_completed.add(self._ingest_stage)
            self._ingest_stage = stage
            self._progress_shown = 0
            self._ingest_count_started = time.monotonic()
        ratio = re.search(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?![\d.])", text)
        if ratio and 0 <= int(ratio[1]) <= int(ratio[2]) and int(ratio[2]) > 0:
            self._ingest_counts[stage] = (int(ratio[1]), int(ratio[2]))
        self._parse_detail = text[:180]
        self.motion.start()

    async def _submit_question(self, query, book, model):
        token = self._try_acquire_workload("generation")
        if token is None:
            await self.write_line("○ GPU work is busy · /cancel or wait")
            return
        request_id = uuid4().hex
        record = TurnRecord(request_id, query, book, model)
        view = TurnView(record)
        self.turns.append(view)
        self._pending_assistants[request_id] = view
        self._request_books[request_id] = book
        self._request_context[request_id] = (
            self._format_history(),
            self.top_k_results,
            self.distance_threshold,
        )
        self._request_web[request_id] = self.web_enabled
        completed_history = self._conversation[-4:]
        self._request_exchanges[request_id] = tuple(
            ConversationExchange(question["content"], answer["content"], book)
            for question, answer in zip(
                completed_history[::2], completed_history[1::2], strict=True
            )
            if question["role"] == "user" and answer["role"] == "assistant"
        )
        self._transport_closed[request_id] = asyncio.Event()
        self._query_workload_token = token
        self._active_query_request_id = request_id
        self._last_question = query
        await self.query_one(StudyStream).add_turn(view)
        view.step_line.update("↳ search_local_index …")
        self._status = self.motion.spinner + " working"
        self.motion.start()
        self.motion.tick(time.monotonic())
        self._set_header()
        painted = asyncio.Event()
        self.call_after_refresh(painted.set)
        # Refresh callbacks need the app message pump; only the worker may wait.
        self._active_query_worker = self.run_worker(
            self._run_query(request_id, query, book, model, token, painted),
            group="queries",
            exit_on_error=False,
        )
        self._update_hint()

    def _restore_queued_prompt(self):
        if self.queued_query:
            queued, _book, _model = self.queued_query
            self.queued_query = None
            prompt = self.query_one(QueryComposer)
            draft = prompt.text
            prompt.load_text(
                queued + ("\n\n" + draft if draft and draft != queued else "")
            )
            prompt.move_cursor(prompt.document.end)

    async def _run_query(
        self, request_id, query, book, model, workload_token, painted=None
    ):
        view = self._pending_assistants[request_id]
        history, k, distance = self._request_context[request_id]
        record = view.record
        buffer = SafeAnswerBuffer()
        client = OllamaClient()
        retrieval = None
        response = None
        updater = None
        pending = []
        terminal_phase = "ERROR"
        failure = ""
        cleanup_ok = False
        saw_done = False
        started = time.monotonic()
        try:
            if painted is not None:
                await painted.wait()
            rewrite_started = time.monotonic()
            rewritten = await rewrite_followup(
                query,
                self._request_exchanges.get(request_id, ()),
                book,
                model=model,
                client=client,
            )
            record.retrieval_query = rewritten.retrieval_query
            if (
                rewritten.applied
                or rewritten.fallback_reason != "not an eligible follow-up"
            ):
                record.steps.append(
                    f"↳ rewrite {'applied' if rewritten.applied else 'fallback'} · {time.monotonic()-rewrite_started:.1f}s · {rewritten.fallback_reason or rewritten.retrieval_query}"
                )
            counter = await self._agent_counter(model, record)
            if counter is not None:
                updater = view.answer.get_stream(view.answer)
                self._answer_streams[request_id] = (updater, pending)
                record.phase = "RETRIEVING"
                response = self._agent_events(
                    request_id,
                    query,
                    book,
                    model,
                    rewritten,
                    k,
                    distance,
                    counter,
                    client,
                )
                outcome = None
                try:
                    async for event in response:
                        if (
                            self._active_query_request_id != request_id
                            or self._is_query_cancelled(request_id)
                        ):
                            raise asyncio.CancelledError
                        outcome = (
                            self._apply_agent_event(
                                request_id,
                                event,
                                record,
                                view,
                                buffer,
                                pending,
                                book,
                                started,
                            )
                            or outcome
                        )
                        await asyncio.sleep(0)
                finally:
                    await response.aclose()
                    self._transport_closed[request_id].set()
                if outcome is None:
                    raise RuntimeError("agent ended without a result")
                if outcome != "DONE":
                    failure = outcome
                else:
                    terminal_phase = "DONE"
            else:
                record.phase = "RETRIEVING"
                retrieval = asyncio.create_task(
                    asyncio.to_thread(
                        self._prepare_query,
                        query,
                        history,
                        book,
                        k,
                        distance,
                        rewritten.retrieval_query,
                        model,
                    )
                )
                prompt, metadata, pages, _ids = await asyncio.shield(retrieval)
                if metadata:
                    record.retrieval_query = metadata[0].get(
                        "retrieval_query", rewritten.retrieval_query
                    )
                if self._is_query_cancelled(request_id):
                    raise asyncio.CancelledError
                self._last_retrieved_pages = pages
                self._set_turn_evidence(request_id, "", book, metadata)
                if any(meta.get("page_filter_fallback") == "True" for meta in metadata):
                    record.steps.append(
                        "↳ no chunks found for requested pages · used semantic fallback"
                    )
                for meta in metadata:
                    record.steps.append(
                        f"↳ {meta.get('index_mode', 'dense')} · {meta.get('chunk_id', '?')} · dense {meta.get('dense_rank', '—')} / BM25 {meta.get('bm25_rank', '—')} · pages {meta.get('physical_pages', '[]')}"
                    )
                record.steps.append(
                    f"↳ search_local_index ✓ {len(metadata)} chunks · {time.monotonic()-started:.1f}s"
                )
                view.step_line.update(record.steps[-1] + "\n↳ thinking …")
                record.phase = "THINKING"
                updater = view.answer.get_stream(view.answer)

                self._answer_streams[request_id] = (updater, pending)
                response = client.astream_generate(prompt, model=model, think=False)
                try:
                    async for data in response:
                        if (
                            self._active_query_request_id != request_id
                            or self._is_query_cancelled(request_id)
                        ):
                            raise asyncio.CancelledError
                        token = data.get("response", "")
                        if token:
                            pending.append(buffer.feed(token))
                            record.raw_answer = buffer.raw
                            record.copy_text = buffer.visible
                            record.copy_is_sanitized = (
                                record.copy_text != record.raw_answer
                            )
                            if record.phase != "STREAMING":
                                record.steps.append(
                                    f"↳ thinking ✓ {time.monotonic()-started:.1f}s"
                                )
                                view.step_line.update("\n".join(record.steps))
                            record.phase = "STREAMING"
                            self._status = "▌ writing"
                        await asyncio.sleep(0)
                        if data.get("done"):
                            saw_done = True
                            break
                finally:
                    await response.aclose()
                    self._transport_closed[request_id].set()
                if not saw_done:
                    raise RuntimeError("Ollama stream ended before completion")
                terminal_phase = "DONE"
        except asyncio.CancelledError:
            terminal_phase = "CANCELLED"
            self._cancelled_query_ids.add(request_id)
        except ConnectError:
            failure = "✗ can't reach Ollama at localhost:11434 · start it, then /retry"
        except ReadTimeout:
            failure = "✗ local response timed out · /retry"
        except HTTPStatusError as exc:
            failure = (
                f"✗ model {model} not found · /models"
                if exc.response.status_code == 404
                else f"✗ Ollama request failed ({exc.response.status_code}) · /retry"
            )
        except Exception as exc:  # noqa: BLE001
            failure = f"✗ {type(exc).__name__}: {exc} · /retry"
        finally:
            if response is None:
                self._transport_closed[request_id].set()
            await self._flush_answer(request_id)
            if updater:
                await updater.stop()
            record.set_answer(buffer.raw)
            await view.finalize(terminal_phase)
            if terminal_phase == "DONE":
                self._completion_view = view
                self._completion_text = f"✓ {len(record.steps)} steps · {time.monotonic()-started:.1f}s · /details"
                self._completion_flash_until = (
                    time.monotonic() + config.UI_COMPLETION_FLASH_SECONDS
                )
                self.motion.start()
            if terminal_phase == "CANCELLED":
                view.step_line.update("○ cancelled")
            elif failure:
                view.step_line.add_class("failed")
                view.step_line.update(failure)
            else:
                self._conversation.extend(
                    [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": record.copy_text},
                    ]
                )
                self._conversation = self._conversation[-10:]
            if terminal_phase != "DONE":
                self._restore_queued_prompt()
            self._status = "○ cleaning up"
            self._set_header()
            # Cancelling the await cannot stop a CPU embedding thread.
            if retrieval is not None and not retrieval.done():
                try:
                    await asyncio.shield(retrieval)
                except Exception:  # noqa: BLE001, S110
                    pass
            try:
                await asyncio.to_thread(client.unload_and_wait)
            except Exception as exc:  # noqa: BLE001
                await self.write_line(
                    f"✗ could not confirm local model cleanup; GPU work remains blocked. {exc}"
                )
                self._restore_queued_prompt()
                self._status = "○ cleanup blocked"
            else:
                self._release_workload(workload_token)
                self._query_workload_token = None
                cleanup_ok = True
                self._status = "● ready"
            self._pending_assistants.pop(request_id, None)
            self._request_books.pop(request_id, None)
            self._request_context.pop(request_id, None)
            self._request_exchanges.pop(request_id, None)
            self._request_web.pop(request_id, None)
            self._transport_closed.pop(request_id, None)
            self._answer_streams.pop(request_id, None)
            self._cancelled_query_ids.discard(request_id)
            if self._active_query_request_id == request_id:
                self._active_query_request_id = None
                self._active_query_worker = None
            self._set_header()
            self._update_hint()
            if cleanup_ok and terminal_phase == "DONE" and self.queued_query:
                queued = self.queued_query
                self.queued_query = None
                await self._submit_question(*queued)

    async def _agent_counter(self, model, record):
        """Chat accounting gates the agent; unvalidated models keep the Phase 2 path."""
        if not self._agent_routing:
            return None
        from .chat_tokens import load_chat_counter

        try:
            return await asyncio.to_thread(load_chat_counter, model)
        except (ValueError, OSError, HTTPError) as exc:
            record.steps.append(
                f"↳ tools unavailable for {model} · {exc} · Phase 2 path"
            )
            return None

    def _ensure_sandbox(self):
        # One warm supervisor per session; a cancelled call stops only its child.
        if self._sandbox is None:
            from .sandbox.client import SandboxClient

            self._sandbox = SandboxClient(request_prefix="tui")
        return self._sandbox

    async def _close_sandbox(self):
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            try:
                await sandbox.aclose()
            except Exception as exc:  # noqa: BLE001
                await self.write_line(
                    f"✗ sandbox cleanup could not be confirmed: {exc}"
                )

    def _agent_events(
        self, request_id, query, book, model, rewritten, k, distance, counter, client
    ):
        from .agent import AgentRequest, run_agent
        from .tools.web import WebTools

        history = tuple(
            turn
            for exchange in self._request_exchanges.get(request_id, ())
            if exchange.book_stem == book
            for turn in (
                {"role": "user", "content": exchange.question},
                {"role": "assistant", "content": exchange.answer},
            )
        )
        request = AgentRequest(
            request_id,
            query,
            book,
            model,
            history=history,
            web_enabled=self._request_web.get(request_id, False),
            top_k=k,
            distance=distance,
            retrieval_query=rewritten.retrieval_query,
        )
        return run_agent(
            request,
            client=client,
            sandbox=self._ensure_sandbox(),
            counter=counter,
            web=WebTools(),
        )

    def _apply_agent_event(
        self, request_id, event, record, view, buffer, pending, book, started
    ):
        """Render one agent event; returns "DONE", a failure line, or None."""
        data = event.data
        if event.kind == "token":
            pending.append(buffer.feed(data["text"]))
            record.raw_answer = buffer.raw
            record.copy_text = buffer.visible
            record.copy_is_sanitized = record.copy_text != record.raw_answer
            if record.phase != "STREAMING":
                record.steps.append(f"↳ thinking ✓ {time.monotonic()-started:.1f}s")
                view.step_line.update("\n".join(record.steps))
            record.phase = "STREAMING"
            self._status = "▌ writing"
            return None
        if event.kind == "sources":
            sources = []
            for ref in data["sources"]:
                source = (
                    SourceRef(book, None, None, None, (), "web", ref.url, ref.title)
                    if ref.kind == "web"
                    else SourceRef(
                        book,
                        ref.physical_pages[0] if ref.physical_pages else None,
                        ref.chapter,
                        ref.section,
                        ref.physical_pages,
                    )
                )
                if source not in sources:
                    sources.append(source)
            record.sources = sources
            self._last_retrieved_pages = sorted(
                {p for s in sources for p in s.physical_pages}
            )
            return None
        elapsed = f"{time.monotonic()-started:.1f}s"
        if event.kind == "step":
            stage = data["stage"]
            line = {
                "route": f"↳ route {data.get('mode')}",
                "local_search_done": f"↳ search_local_index {data.get('status')} · {data.get('passages')} passages · {elapsed}",
                "assessment": f"↳ local evidence {'sufficient' if data.get('sufficient') else 'insufficient'} · {data.get('reason', '')}",
                "compute_done": f"↳ execute_python {data.get('status')} · {data.get('result')}",
                "verify_done": f"↳ verify {data.get('status')} · {data.get('reason')}",
                "optional_web": "↳ web_search …",
            }.get(stage)
            if line:
                record.steps.append(line)
            phase = {
                "local_search": "RETRIEVING",
                "compute": "COMPUTING",
                "final": "THINKING",
            }
            record.phase = phase.get(stage, record.phase)
            view.step_line.update("\n".join(record.steps[-3:]) + f"\n↳ {stage} …")
            return None
        if event.kind == "result":
            verification = data.get("verification")
            if verification:
                record.steps.append(f"↳ verification {verification['status']}")
            record.steps.append(
                f"↳ {data['model_requests']} model requests · {data['tool_calls']} tool calls · {elapsed}"
            )
            if not data["complete"]:
                return "✗ answer reached the output limit and is incomplete · /retry"
            return "DONE"
        return f"✗ {data.get('message', 'agent error')} · /retry"

    async def _cancel_active_query(self):
        request_id = self._active_query_request_id
        view = self._pending_assistants.get(request_id)
        if view is None or view.record.phase == "CANCELLED":
            return
        if view.record.phase in {"DONE", "ERROR"}:
            self._restore_queued_prompt()
            return
        self._cancelled_query_ids.add(request_id)
        view.record.set_answer(view.record.raw_answer)
        view.record.phase = "CANCELLED"
        view.step_line.update("○ cancelled")
        self._restore_queued_prompt()
        self._status = "○ cleaning up"
        self._set_header()
        closed = self._transport_closed[request_id]
        if self._active_query_worker:
            # Let a newly scheduled worker enter its cleanup-protected body.
            await asyncio.sleep(0)
            self._active_query_worker.cancel()
        try:
            await asyncio.wait_for(closed.wait(), timeout=config.UI_CANCEL_TIMEOUT)
        except TimeoutError:
            await self.write_line(
                "✗ client abort is still pending · GPU work remains blocked"
            )
        self._update_hint()

    async def handle_command(self, name, arguments):
        spec = COMMANDS[name]
        if name == "/delete":
            name = "/remove"

        def usage():
            return self.write_line(f"○ usage: {spec.name} {spec.usage}".rstrip())

        if name in {"/thinking", "/web"}:
            if len(arguments) != 1 or arguments[0] not in {"on", "off", "show"}:
                await usage()
            elif name == "/thinking":
                await self.write_line(
                    "○ thinking is automatic: on for the solve computation step, off for Q&A"
                    + ("" if arguments[0] == "show" else " · not changed")
                )
            elif arguments[0] == "show":
                endpoint = "configured" if config.WEB_SEARCH_URL else "not configured"
                await self.write_line(
                    f"○ web evidence {'on' if self.web_enabled else 'off'}"
                    f" · search endpoint {endpoint} · retrieval only, no cloud inference"
                )
            elif self._busy() or self.queued_query:
                await self.write_line(
                    "○ web permission is locked while work is pending"
                )
            else:
                self.web_enabled = arguments[0] == "on"
                await self.write_line(
                    f"○ web evidence {arguments[0]}"
                    + (
                        " · no search endpoint configured (WEB_SEARCH_URL)"
                        if self.web_enabled and not config.WEB_SEARCH_URL
                        else ""
                    )
                )
            return
        if spec.available_phase > 1:
            await self.write_line(
                f"○ {name} is available in Phase {spec.available_phase}"
            )
            return
        if (
            name in {"/book", "/model", "/remove", "/k", "/d", "/accent"}
            and not arguments
        ):
            await usage()
            return
        if (
            name
            in {
                "/help",
                "/models",
                "/retry",
                "/clear",
                "/new",
                "/settings",
                "/cancel",
                "/quit",
                "/gpu-parsing",
                "/gpu-status",
            }
            and arguments
        ):
            await usage()
            return
        if name == "/help":
            await self.write_line(
                "\n".join(
                    f"{s.name} {s.usage} · {s.description}".replace("  ", " ")
                    for s in COMMANDS.values()
                )
            )
        elif name == "/books":
            try:
                self._book_counts = await asyncio.to_thread(self._discover_books)
                filtered = [
                    f"{book} ({count})"
                    for book, count in self._book_counts.items()
                    if " ".join(arguments).casefold() in book.casefold()
                ]
                filtered.extend(
                    f"{book} (incomplete or unavailable · /remove {shlex.quote(book)} or /parse <path>)"
                    for book in self._unavailable_books
                    if " ".join(arguments).casefold() in book.casefold()
                )
                await self.write_line(
                    "\n".join(filtered) or "○ no book loaded · /parse <path>"
                )
                names = [
                    book
                    for book in self._book_counts
                    if " ".join(arguments).casefold() in book.casefold()
                ]
                if names:
                    await self._show_overlay(names, "books")
                    self.overlay.focus()
            except Exception as exc:  # noqa: BLE001
                await self.write_line(f"✗ book discovery failed: {exc} · /books")
        elif name in {"/book", "/model", "/remove"}:
            if self._busy():
                await self.write_line(
                    "○ work is active · /cancel before switching or removing"
                )
                return
            value = " ".join(arguments)
            if name == "/model":
                if value not in self._models:
                    await self.write_line(f"✗ model {value} not found · /models")
                    return
                self.selected_model = value
            else:
                matches = [
                    book
                    for book in self._book_counts | self._unavailable_books
                    if value.casefold()
                    in {book.casefold(), self._format_book_name(book).casefold()}
                ]
                if len(matches) != 1:
                    await self.write_line("○ book not found · /books")
                    return
                value = matches[0]
                if name == "/remove":
                    self._pending_remove = value
                    await self.write_line(f"○ remove {value}? type y to confirm")
                    return
                if value in self._unavailable_books:
                    await self.write_line(
                        f"✗ {value}: incomplete or unavailable index · /remove {shlex.quote(value)} or /parse <path>"
                    )
                    return
                self.active_collection_name = value
                self._conversation.clear()
                self._last_retrieved_pages.clear()
            self._refresh_condition()
            self._set_header()
        elif name == "/models":
            try:
                self._models = await asyncio.to_thread(self._fetch_ollama_models)
                await self.write_line(
                    "\n".join(self._models) or "○ no model · /model <name>"
                )
            except Exception:  # noqa: BLE001
                await self.write_line(
                    "✗ can't reach Ollama at localhost:11434 · start it, then /retry"
                )
        elif name in {"/k", "/d"}:
            try:
                if len(arguments) != 1:
                    raise ValueError("one value required")
                value = int(arguments[0]) if name == "/k" else float(arguments[0])
                if (
                    name == "/k"
                    and not 3 <= value <= config.HYBRID_CANDIDATE_LIMIT
                    or name == "/d"
                    and (not math.isfinite(value) or value < 0)
                ):
                    raise ValueError("out of range")
            except ValueError:
                await usage()
                return
            if name == "/k":
                self.top_k_results = value
            else:
                self.distance_threshold = value
            await self.write_line(f"○ {name[1:]} = {value}")
        elif name in {"/sources", "/details", "/copy"}:
            code = name == "/copy" and arguments[:1] == ["code"]
            numbers = arguments[1:] if code else arguments
            try:
                if len(numbers) > 1:
                    raise ValueError("one index required")
                index = int(numbers[0]) if numbers else 1
                if index < 1:
                    raise ValueError("positive index required")
            except ValueError:
                await usage()
                return
            completed = [turn for turn in self.turns if turn.record.phase == "DONE"]
            if code:
                blocks = [
                    block
                    for turn in completed
                    for block in extract_fenced_blocks(turn.record.raw_answer)
                ]
                if index > len(blocks):
                    await self.write_line("○ no code block to copy")
                    return
                await self._copy(blocks[-index].source)
                return
            if index > len(completed):
                await self.write_line("○ no completed answer")
                return
            turn = completed[-index]
            if name == "/copy":
                await self._copy(turn.record.copy_payload())
            elif name == "/sources":
                turn.show_sources()
            else:
                turn.detail_line.display = True
                turn.detail_line.update(
                    "\n".join(turn.record.steps) or "○ no recorded tool steps"
                )
        elif name in {"/clear", "/new"}:
            if self._busy():
                await self.write_line("○ work is active · /cancel before clearing")
                return
            await self.action_clear_chat()
            if name == "/new":
                self._conversation.clear()
                self._last_retrieved_pages.clear()
                self._request_books.clear()
                self.queued_query = None
        elif name == "/retry":
            if self._last_question:
                await self.on_query_composer_submitted(
                    QueryComposer.Submitted(
                        self.query_one(QueryComposer), self._last_question
                    )
                )
            else:
                self.run_worker(
                    self._run_startup_checks(),
                    group="startup",
                    exclusive=True,
                    exit_on_error=False,
                )
        elif name == "/settings":
            await self._show_overlay(
                [
                    f"/k {self.top_k_results}",
                    f"/d {self.distance_threshold}",
                    "/theme",
                    "/thinking off",
                    "/web off",
                    "/gpu-status",
                ],
                "settings",
            )
            self.overlay.focus()
        elif name == "/theme":
            names = ["terminal", "mono", *THEMES, *self.custom_themes]
            if len(arguments) > 1:
                await usage()
                return
            if self.overlay:
                await self.close_overlay()
            self._preview = ThemePreview(
                self, self.settings, self.settings_path, os.environ
            )
            if arguments:
                value = arguments[0]
                if value not in names:
                    await self.write_line(
                        f'✗ theme "{value}" invalid: unknown theme · using terminal'
                    )
                    await resolve_theme(self, type(self.settings)(), os.environ)
                    self._preview = None
                    return
                try:
                    await self._preview.preview(value)
                    await self._theme_notice(value)
                    await self._preview.keep(value)
                except (OSError, ValueError) as exc:
                    await self.write_line(
                        f"✗ theme save failed: {exc} · previous theme restored"
                    )
                self._preview = None
            else:
                await self._show_overlay(names, "themes")
                self.overlay.focus()
        elif name == "/accent":
            if len(arguments) != 1 or arguments[0] not in ANSI_COLORS:
                await usage()
                return
            previous = self.settings.accent
            try:
                save_theme_selection(
                    self.settings_path, self.settings.theme, arguments[0]
                )
                self.settings.accent = arguments[0]
                await resolve_theme(self, self.settings, os.environ)
            except (OSError, ValueError) as exc:
                self.settings.accent = previous
                await self.write_line(f"✗ settings save failed: {exc}")
        elif name == "/parse":
            await self._start_parse_command(arguments)
        elif name == "/gpu-parsing":
            if self._busy():
                await self.write_line(
                    "○ work is active · wait before changing ingest device"
                )
                return
            self._use_gpu_for_embeddings = not self._use_gpu_for_embeddings
            await self.write_line(
                "○ ingest embeddings: "
                + (
                    "GPU"
                    if self._use_gpu_for_embeddings
                    else "CPU (explicit degraded mode)"
                )
            )
        elif name == "/gpu-status":
            await self.write_line(
                "○ ingest embeddings: "
                + (
                    "GPU"
                    if self._use_gpu_for_embeddings
                    else "CPU (explicit degraded mode)"
                )
            )
        elif name == "/cancel":
            self.action_cancel_query()
        elif name == "/quit":
            await self.action_quit()

    async def _copy(self, text):
        ok, message = await asyncio.to_thread(copy_text, self, text)
        if ok:
            # Native TextArea paste reads Textual's local clipboard.
            self._clipboard = text
        lines = message.splitlines()
        if ok and lines[-1].startswith("copied "):
            for line in lines[:-1]:
                await self.write_line(line)
            if self._copy_confirmation:
                await self._copy_confirmation.remove()
            self._copy_confirmation = Static(
                lines[-1], classes="tool-line copy-confirmation", markup=False
            )
            self._copy_expiry = time.monotonic() + config.UI_COPY_CONFIRM_SECONDS
            stream = self.query_one(StudyStream)
            await stream.mount(self._copy_confirmation, before=stream.spacer)
            self.motion.start()
        else:
            await self.write_line(message)

    async def _remove_book(self, book):
        if self._busy():
            await self.write_line("○ work is active · /cancel before removing")
            return
        self._deleting = True
        try:
            deleted = await asyncio.to_thread(delete_collection, book)
            if not deleted:
                await self.write_line("○ book not found · /books")
                return
            self._book_counts.pop(book, None)
            self._unavailable_books.pop(book, None)
            for turn in list(self.turns):
                if turn.record.book == book:
                    await turn.remove()
                    self.turns.remove(turn)
            self._conversation.clear()
            self._last_retrieved_pages.clear()
            if self.active_collection_name == book:
                self.active_collection_name = next(iter(self._book_counts), None)
            self._refresh_condition()
            self._set_header()
            await self.write_line(f"✓ removed {book}")
        except Exception as exc:  # noqa: BLE001
            await self.write_line(f"✗ remove failed: {exc} · /books")
        finally:
            self._deleting = False

    def action_cancel_query(self):
        if self._is_parsing:
            self._ingest_cancelled = True
            self.run_worker(
                lambda: self._terminate_job_processes(),
                thread=True,
                exit_on_error=False,
            )
            return
        if self._active_query_request_id:
            self.run_worker(
                self._cancel_active_query(),
                group="cancel-query",
                exclusive=True,
                exit_on_error=False,
            )

    def _set_turn_evidence(self, request_id, raw, book, metadata):
        view = self._pending_assistants.get(request_id)
        if view is None or request_id in self._cancelled_query_ids:
            return
        view.record.set_answer(raw)
        sources = []
        for meta in metadata:
            page = meta.get("physical_page", meta.get("page"))
            try:
                page = int(page)
                if page < 1:
                    page = None
            except (TypeError, ValueError):
                page = None
            source = SourceRef(
                book,
                page,
                meta.get("chapter") or meta.get("Header 1"),
                meta.get("section") or meta.get("Header 2"),
                physical_pages_for(meta),
            )
            if source not in sources:
                sources.append(source)
        view.record.sources = sources

    async def _stream_query_token(self, request_id, token):
        view = self._pending_assistants.get(request_id)
        if view and request_id not in self._cancelled_query_ids:
            await view.answer.append(token)

    async def _finalize_query(
        self, request_id, message, query, is_error, telemetry_badge=None
    ):
        view = self._pending_assistants.get(request_id)
        if view is None or request_id in self._cancelled_query_ids:
            return
        if not view.record.raw_answer or is_error:
            view.record.set_answer(message)
        await view.finalize("ERROR" if is_error else "DONE")
        if not is_error:
            self._conversation.extend(
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": view.record.copy_text},
                ]
            )
            self._conversation = self._conversation[-10:]
        self._pending_assistants.pop(request_id, None)
        self._request_books.pop(request_id, None)
        if self._active_query_request_id == request_id:
            self._active_query_request_id = None
            self._status = "● ready"
            self._set_header()
        self._update_hint()

    async def _finalize_query_cancelled(self, request_id):
        view = self._pending_assistants.pop(request_id, None)
        if view:
            view.record.phase = "CANCELLED"
            view.step_line.update("○ cancelled")
        if self._active_query_request_id == request_id:
            self._active_query_request_id = None
            self._status = "○ cleaning up"
            self._set_header()
        self._update_hint()

    async def _finalize_query_transient_error(self, request_id, message):
        view = self._pending_assistants.get(request_id)
        if view:
            await self._finalize_query(request_id, message, view.record.question, True)

    async def _start_parse_command(self, arguments):
        if not arguments:
            await self.write_line(
                "○ usage: /parse <path> [--mode auto|pymupdf4llm|mineru|marker] [--cpu-embeddings]"
            )
            return
        path = arguments[0]
        mode = "auto"
        cpu = False
        rest = arguments[1:]
        while rest:
            option = rest.pop(0)
            if option == "--cpu-embeddings":
                cpu = True
            elif option == "--mode" and rest:
                mode = rest.pop(0)
            else:
                await self.write_line(
                    "○ usage: /parse <path> [--mode auto|pymupdf4llm|mineru|marker] [--cpu-embeddings]"
                )
                return
        if mode not in {"auto", "pymupdf4llm", "mineru", "marker"}:
            await self.write_line("○ invalid parser mode · /help")
            return
        await self._start_parse_and_index(path, mode, cpu)

    async def _start_parse_and_index(self, raw_path, mode_value="auto", cpu=False):
        if self._busy():
            await self.write_line("○ GPU work is busy · /cancel or wait")
            return
        pdf_path = Path(raw_path).expanduser().resolve()
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            await self.write_line("✗ PDF file not found · /parse <path>")
            return
        token = self._try_acquire_workload("ingest")
        if token is None:
            return
        self._hide_home()
        self._is_parsing = True
        self._ingest_cancelled = False
        self._parse_workload_token = token
        self._ingest_path = str(pdf_path)
        self._parse_started_at = time.monotonic()
        if cpu:
            self._use_gpu_for_embeddings = False
        record = TurnRecord(
            uuid4().hex,
            "/parse " + shlex.quote(str(pdf_path)),
            pdf_path.stem,
            self.selected_model,
            phase="INGEST",
        )
        self._ingest_view = TurnView(record)
        self.turns.append(self._ingest_view)
        await self.query_one(StudyStream).add_turn(self._ingest_view)
        self._ingest_stage = "preparing"
        self._ingest_counts.clear()
        self._ingest_completed.clear()
        self._set_parse_runtime_status(stage="Preparing parse job", engine=mode_value)
        self.run_worker(
            lambda: self._parse_and_index_worker(str(pdf_path), mode_value, token),
            thread=True,
            group="parse",
            exit_on_error=False,
        )
        self._update_hint()

    def _set_parse_runtime_status(
        self, stage=None, engine=None, detail=None, target_progress=None
    ):
        if stage:
            self._parse_current_stage = stage
            fallback = (
                "parse"
                if "Parsing PDF" in stage
                else "index" if "Indexing" in stage else "preparing"
            )
            self._ingest_output(fallback, stage)
        if engine:
            self._parse_current_engine = engine
        if detail:
            self._parse_detail = detail
        self.motion.start()

    async def _finish_parse_and_index(self, book_stem, page_count, chunks, error):
        self._is_parsing = False
        if self._ingest_view:
            self._ingest_view.record.phase = "ERROR" if error else "DONE"
            message = (
                f"✗ ingest failed: {error} · partial index may be incomplete · parsed checkpoints preserved · retry /parse {shlex.quote(self._ingest_path)}"
                if error
                else f"✓ {book_stem} · {page_count} pages · {chunks} chunks"
            )
            if self._ingest_cancelled:
                self._ingest_view.record.phase = "CANCELLED"
                message = "○ cancelled · resume with /parse " + shlex.quote(
                    self._ingest_path
                )
            if not error and not self._ingest_cancelled:
                message = Text.assemble(
                    "✓ ",
                    (book_stem, f"link {Path(self._ingest_path).as_uri()}"),
                    f" · {page_count} pages · {chunks} chunks",
                )
            self._ingest_view.step_line.set_class(
                bool(error) and not self._ingest_cancelled, "failed"
            )
            self._ingest_view.step_line.update(message)
        if not error and not self._ingest_cancelled:
            self._book_counts[book_stem] = chunks
            self.active_collection_name = book_stem
            self._conversation.clear()
            self._last_retrieved_pages.clear()
        self._status = "● ready"
        self._refresh_condition()
        self._set_header()
        self._update_hint()

    def _terminate_job_processes(self):
        with self._process_lock:
            processes = list(self._active_processes)
        for process in processes:
            self._stop_process_group(process)

    async def action_quit(self):
        self._restore_queued_prompt()
        worker = self._active_query_worker
        if self._active_query_request_id:
            await self._cancel_active_query()
        self._ingest_cancelled = True
        await asyncio.to_thread(self._terminate_active_processes)
        if worker:
            try:
                await worker.wait()
            except (WorkerCancelled, WorkerFailed):
                pass
        await asyncio.to_thread(self._release_model_owners)
        await self._close_sandbox()
        self.motion.stop()
        self.exit()

    @staticmethod
    def _release_model_owners():
        import gc

        module = sys.modules.get("src.db")
        if module is not None:
            # Startup discovery may still be importing the database dependencies.
            if hasattr(module, "_QUERY_MODEL"):
                module._QUERY_MODEL = None
            if getattr(module, "_INGEST_MODEL", None) is not None:
                module.release_ingest_model()
            gc.collect()

    def on_unmount(self):
        self.motion.stop()
        if self._active_query_request_id:
            self._cancelled_query_ids.add(self._active_query_request_id)
        self._terminate_active_processes()

    def _try_acquire_workload(self, kind: WorkloadKind) -> str | None:
        return self._workload_coordinator.try_acquire(kind)

    def _release_workload(self, token: str) -> None:
        self._workload_coordinator.release(token)

    def _execute_rag_query_with_admission(
        self,
        query: str,
        history_str: str,
        book_stem: str,
        model_name: str,
        request_id: str,
        workload_token: str,
    ) -> None:
        try:
            self.execute_rag_query(
                query, history_str, book_stem, model_name, request_id
            )
        finally:
            try:
                # Closing a client stream requests server-side cancellation, but
                # admission stays owned until Ollama confirms model cleanup.
                OllamaClient().unload_and_wait()
            except Exception as exc:  # noqa: BLE001
                try:
                    self.call_from_thread(
                        self._report_generation_cleanup_failure,
                        f"{type(exc).__name__}: {exc}",
                    )
                except Exception:  # noqa: BLE001, S110
                    pass
            else:
                self._release_workload(workload_token)
                if self._query_workload_token == workload_token:
                    self._query_workload_token = None
                if self._loop is not None:
                    self.call_from_thread(self._generation_cleaned)

    def _generation_cleaned(self):
        self._status = "● ready"
        self._set_header()
        self._update_hint()

    def _report_generation_cleanup_failure(self, detail: str) -> None:
        message = f"✗ could not confirm local model cleanup; GPU work remains blocked. {detail}"
        if self._loop is not None:
            self.run_worker(self.write_line(message), exit_on_error=False)
        else:
            self.notices.append(message)

    def _format_history(self) -> str:
        if not self._conversation:
            return ""
        lines = ["Conversation so far:"]
        for turn in self._conversation[-6:]:  # last 3 exchanges
            role = "User" if turn["role"] == "user" else "Axie"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines) + "\n"

    def build_citation(self, book_stem: str) -> str:
        """
        Build citation string from last retrieved physical pages.
        """
        return build_page_citation(book_stem, self._last_retrieved_pages)

    def extract_page_intent(self, query: str) -> list[int] | None:
        """
        Extract explicit page intent from query.

        Accepted forms (v1):
          - page N
          - p.N / pN
          - pg.N / pgN
          - pages N-M (inclusive range)
        """
        return parse_page_intent(query)

    def _prepare_query(
        self,
        query,
        history_str,
        book_stem,
        k,
        distance_threshold,
        retrieval_query=None,
        model=None,
        answer_token_counter=None,
    ):
        from .answer_tokens import load_answer_counter
        from .hybrid_retrieval import select_context
        from .retrieval import search_book

        found = search_book(
            query,
            book_stem=book_stem,
            distance=distance_threshold,
            retrieval_query=retrieval_query,
        )
        candidates, fallback, overview = (
            list(found.candidates),
            found.page_fallback,
            found.overview,
        )
        retrieval_query = found.retrieval_query

        def prompt_for(rows):
            context = (
                "\n\n".join(
                    f"[{c.metadata.get('chapter', c.metadata.get('Header 1', 'Unknown Chapter'))} · {c.metadata.get('section', c.metadata.get('Header 2', 'Unknown Section'))} · p.{collapse_page_ranges(list(physical_pages_for(c.metadata)))}]\n{c.text}"
                    for c in rows
                )
                or "No direct context retrieved from indexed files."
            )
            prompt = PROMPT_TEMPLATE.format(
                book=book_stem, history=history_str, context=context, question=query
            )
            if overview:
                prompt = prompt.replace(
                    f"Current question: {query}\n\nEvidence rules",
                    f"Original user wording: {query}\n\nCurrent question: What is this book about?\n\nEvidence rules",
                )
            return prompt

        counter = answer_token_counter or load_answer_counter(
            model or self.selected_model
        )
        selected = select_context(
            query,
            candidates,
            k,
            lambda rows: counter(prompt_for(rows)),
            config.OLLAMA_CONTEXT_TOKENS - config.OLLAMA_OUTPUT_TOKENS,
        )
        metadata = [
            dict(
                c.metadata,
                chunk_id=c.chunk_id,
                dense_rank=c.dense_rank if c.dense_rank is not None else "—",
                bm25_rank=c.bm25_rank if c.bm25_rank is not None else "—",
                dense_distance=(
                    c.dense_distance if c.dense_distance is not None else "—"
                ),
                bm25_score=c.bm25_score if c.bm25_score is not None else "—",
                index_mode=(
                    "dense-only legacy · rebuild for hybrid"
                    if found.legacy
                    else "hybrid"
                ),
                page_filter_fallback=str(fallback),
                index_version=found.index_version,
                retrieval_query=retrieval_query,
            )
            for c in selected
        ]
        return (
            prompt_for(selected),
            metadata,
            sorted({p for c in selected for p in physical_pages_for(c.metadata)}),
            [c.chunk_id for c in selected],
        )

    def _prepare_legacy_query(
        self, query, history_str, book_stem, k, distance_threshold
    ):
        # Augment retrieval query with history if it's vague/short
        retrieval_query = query
        vague_terms = {
            "it",
            "this",
            "that",
            "they",
            "example",
            "question",
            "more",
            "explain",
            "help",
            "detail",
        }
        query_words = set(query.lower().split())
        is_vague = bool(query_words & vague_terms) or len(query_words) < 5

        if is_vague and self._conversation:
            # Add context from the last user message to improve retrieval relevance
            for msg in reversed(self._conversation):
                if msg["role"] == "user":
                    # If the user is asking for examples/questions, broaden search significantly
                    if any(
                        term in query.lower()
                        for term in ["example", "question", "exercise", "problem"]
                    ):
                        retrieval_query = f"{msg['content']} illustrative examples exercises questions problems"
                    else:
                        prev_context = " ".join(msg["content"].split()[:20])
                        retrieval_query = f"{prev_context} {query}"
                    break

        # Always retrieve the rewritten query so follow-ups get fresh evidence.
        target_pages = self.extract_page_intent(query)
        q_vec = embed_query(retrieval_query)
        collection = get_collection(book_stem)
        ensure_collection_ready(collection)
        where_clause: dict[str, object] | None = None
        if target_pages:
            if len(target_pages) == 1:
                where_clause = {"physical_page": target_pages[0]}
            else:
                where_clause = {"physical_page": {"$in": target_pages}}

        query_kwargs: dict[str, object] = {
            "query_embeddings": [q_vec],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause is not None:
            query_kwargs["where"] = where_clause
        results = collection.query(**query_kwargs)

        context_blocks = []
        metadatas = []
        retrieved_pages: list[int] = []
        selected_ids = []
        retrieved_ids = (results.get("ids") or [[]])[0]
        retrieved_docs = results.get("documents") or [[]]
        retrieved_meta = results.get("metadatas") or [[]]
        retrieved_distances = results.get("distances") or [[]]
        has_retrieved_context = bool(retrieved_docs and retrieved_docs[0])

        # If explicit page filter finds nothing, gracefully fall back to global semantic search.
        if target_pages and not has_retrieved_context:
            fallback_results = collection.query(
                query_embeddings=[q_vec],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            retrieved_ids = (fallback_results.get("ids") or [[]])[0]
            retrieved_docs = fallback_results.get("documents") or [[]]
            retrieved_meta = fallback_results.get("metadatas") or [[]]
            retrieved_distances = fallback_results.get("distances") or [[]]
            has_retrieved_context = bool(retrieved_docs and retrieved_docs[0])
            message = "○ no chunks found for requested pages · used semantic fallback"
            if self._loop is not None:
                self.call_from_thread(self.write_line, message)

        if has_retrieved_context:
            chunks = retrieved_docs[0]
            raw_metadatas = (
                retrieved_meta[0]
                if retrieved_meta and retrieved_meta[0]
                else [{} for _ in chunks]
            )
            distances = retrieved_distances[0] if retrieved_distances else []
            for idx, (chunk, meta) in enumerate(zip(chunks, raw_metadatas)):
                if idx < len(distances):
                    distance = distances[idx]
                    # Be more lenient for vague queries
                    limit = distance_threshold * 1.4 if is_vague else distance_threshold
                    if isinstance(distance, (int, float)) and distance >= limit:
                        continue
                metadatas.append(meta)
                if idx < len(retrieved_ids):
                    selected_ids.append(retrieved_ids[idx])
                h1 = meta.get("Header 1", "Unknown Chapter")
                h2 = meta.get("Header 2", "Unknown Section")
                page_info_raw = meta.get("physical_page", meta.get("page", "N/A"))
                normalized_page: int | None = None
                try:
                    parsed_page = int(page_info_raw)
                    if parsed_page >= 1:
                        normalized_page = parsed_page
                except (TypeError, ValueError):
                    normalized_page = None
                if normalized_page is not None:
                    retrieved_pages.append(normalized_page)
                    page_info = normalized_page
                else:
                    page_info = page_info_raw
                context_blocks.append(f"[{h1} · {h2} · p.{page_info}]\n{chunk}")

        prompt = PROMPT_TEMPLATE.format(
            book=book_stem,
            chapter=(
                ",".join(sorted({m.get("Header 1", "Unknown") for m in metadatas}))
                if metadatas
                else "General Knowledge"
            ),
            pages=(
                ",".join(
                    sorted(
                        {
                            str(m.get("physical_page", m.get("page", "?")))
                            for m in metadatas
                        }
                    )
                )
                if metadatas
                else "N/A"
            ),
            history=history_str,
            context=(
                "\n\n".join(context_blocks)
                if context_blocks
                else "No direct context retrieved from indexed files."
            ),
            question=query,
        )

        return prompt, metadatas, retrieved_pages, selected_ids

    def execute_rag_query(
        self,
        query: str,
        history_str: str,
        book_stem: str,
        model_name: str,
        request_id: str,
    ) -> None:
        try:
            if self._is_query_cancelled(request_id):
                self.call_from_thread(self._finalize_query_cancelled, request_id)
                return

            prompt, metadatas, retrieved_pages, _ids = self._prepare_legacy_query(
                query,
                history_str,
                book_stem,
                self.top_k_results,
                self.distance_threshold,
            )
            self._last_retrieved_pages = retrieved_pages

            accumulated_response = ""
            inference_badge: str | None = None
            was_cancelled = False
            stream_started = time.monotonic()
            stream_metrics: dict[str, object] = {}
            stream = OllamaClient().generate(
                prompt, model=model_name, stream=True, think=False
            )
            if isinstance(stream, dict):
                raise TypeError(
                    "Ollama returned a non-stream payload for a stream request."
                )
            try:
                for data in stream:
                    if self._is_query_cancelled(request_id):
                        was_cancelled = True
                        break
                    if data.get("done"):
                        stream_metrics = data
                    token = data.get("response", "")
                    if token:
                        accumulated_response += token
            finally:
                stream.close()

            if was_cancelled:
                self.call_from_thread(self._finalize_query_cancelled, request_id)
                return

            inference_badge = self._build_inference_badge(
                stream_metrics,
                time.monotonic() - stream_started,
            )

            self.call_from_thread(
                self._set_turn_evidence,
                request_id,
                accumulated_response,
                book_stem,
                metadatas,
            )

            # Post-process for math rendering
            accumulated_response = strip_model_footnotes(
                latex_to_unicode(accumulated_response)
            )

            # Stream only the final response to avoid flashing an initial refusal.
            # Stream by completed lines so markdown tables/lists do not flicker mid-row.
            for chunk in self._iter_display_chunks(accumulated_response):
                if self._is_query_cancelled(request_id):
                    self.call_from_thread(self._finalize_query_cancelled, request_id)
                    return
                self.call_from_thread(
                    self._stream_query_token,
                    request_id,
                    chunk,
                )

            self.call_from_thread(
                self._finalize_query,
                request_id,
                accumulated_response,
                query,
                False,
                inference_badge,
            )

        except ConnectError:
            self.call_from_thread(
                self._finalize_query_transient_error,
                request_id,
                "Cannot reach Ollama. Is it running? (`ollama serve`)",
            )
        except ReadTimeout:
            self.call_from_thread(
                self._finalize_query_transient_error,
                request_id,
                "Ollama timed out. The model may still be loading; try again shortly.",
            )
        except HTTPStatusError as exc:
            if exc.response.status_code == 404:
                detail = (
                    f"Model '{model_name}' was not found in Ollama. "
                    "Use /models and /model to select an installed model."
                )
            else:
                detail = f"Ollama request failed ({exc.response.status_code}): {exc.response.reason_phrase}"
            self.call_from_thread(self._finalize_query, request_id, detail, query, True)
        except Exception as exc:  # noqa: BLE001
            if self._is_query_cancelled(request_id):
                self.call_from_thread(self._finalize_query_cancelled, request_id)
                return
            detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            self.call_from_thread(
                self._finalize_query,
                request_id,
                f"{exc.__class__.__name__}: {detail}",
                query,
                True,
            )

    def _is_query_cancelled(self, request_id: str) -> bool:
        return request_id in self._cancelled_query_ids

    def _iter_display_chunks(self, response: str) -> list[str]:
        if not response:
            return []
        lines = response.splitlines(keepends=True)
        if lines:
            return lines
        return [response]

    def _build_inference_badge(
        self, payload: dict[str, object], elapsed_seconds: float | None = None
    ) -> str | None:
        def _to_positive_float(value: object) -> float | None:
            if isinstance(value, bool):
                return None
            if not isinstance(value, (int, float)):
                return None
            number = float(value)
            return number if number > 0 else None

        eval_count_value = _to_positive_float(payload.get("eval_count"))
        eval_duration_ns = _to_positive_float(payload.get("eval_duration"))
        total_duration_ns = _to_positive_float(payload.get("total_duration"))

        tokens_per_second: float | None = None
        if eval_count_value is not None and eval_duration_ns is not None:
            eval_seconds = eval_duration_ns / 1_000_000_000
            if eval_seconds > 0:
                tokens_per_second = eval_count_value / eval_seconds

        total_seconds: float | None = None
        if total_duration_ns is not None:
            total_seconds = total_duration_ns / 1_000_000_000
        elif elapsed_seconds is not None and elapsed_seconds > 0:
            total_seconds = elapsed_seconds

        if tokens_per_second is not None and total_seconds is not None:
            return (
                f"[ {tokens_per_second:.1f} tokens/sec · {total_seconds:.1f}s total ]"
            )
        if tokens_per_second is not None:
            return f"[ {tokens_per_second:.1f} tokens/sec ]"
        if total_seconds is not None:
            return f"[ {total_seconds:.1f}s total ]"
        return None

    def _fetch_ollama_models(self) -> list[str]:
        with Client(timeout=4.0) as client:
            response = client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            payload = response.json()
        models = [
            model.get("name")
            for model in payload.get("models", [])
            if model.get("name")
        ]
        return sorted(set(models))

    def _parse_and_index_worker(
        self, pdf_path: str, mode: str, workload_token: str
    ) -> None:
        try:
            from . import checkpoint
            from .ocr import resolve_mode

            pdf = Path(pdf_path)
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logs_dir = LOGS_DIR
            logs_dir.mkdir(parents=True, exist_ok=True)
            self.call_from_thread(
                self._set_parse_runtime_status,
                stage="Releasing GPU memory",
                detail="Unloading Ollama sessions to free VRAM before OCR.",
            )
            self._release_ollama_models()

            resolved_mode, reason = resolve_mode(pdf, mode)
            if resolved_mode == "mineru" and not self._is_mineru_available():
                if mode == "auto":
                    resolved_mode = "pymupdf4llm"
                    reason = (
                        f"{reason}; MinerU unavailable, falling back to pymupdf4llm "
                        "(install with: uv pip install mineru)"
                    )
                else:
                    raise RuntimeError(
                        "MinerU mode selected but MinerU is not installed. "
                        "Install with: uv pip install mineru"
                    )

            use_gpu_for_embeddings = self._use_gpu_for_embeddings
            gpu_note = self._gpu_runtime_note(use_gpu_for_embeddings)
            parse_env = self._build_parse_env(use_gpu_for_embeddings)
            engine_label = {
                "pymupdf4llm": "pymupdf4llm",
                "marker": "Marker (8GB VRAM)",
                "mineru": "MinerU",
            }.get(resolved_mode, resolved_mode)
            if mode == "auto":
                engine_label = f"{engine_label} (auto)"
            detail_base = f"{reason} | {gpu_note}"
            self.call_from_thread(
                self._set_parse_runtime_status,
                stage="Parsing PDF pages",
                engine=engine_label,
                detail=detail_base,
            )

            ocr_cmd = [
                sys.executable,
                "-u",
                "-m",
                "src.ocr",
                "--pdf",
                str(pdf),
                "--mode",
                resolved_mode,
            ]

            def on_ocr_output(line: str) -> None:
                self.call_from_thread(self._ingest_output, "parse", line)

            ocr_log = logs_dir / f"{pdf.stem}_{run_stamp}_ocr.log"
            self._write_stage_log(ocr_log, ocr_cmd, parse_env, "")
            ocr_timeout = self._ocr_timeout_seconds(resolved_mode)
            ocr_code, ocr_output = self._run_external_job(
                ocr_cmd,
                env=parse_env,
                on_output=on_ocr_output,
                live_log_path=ocr_log,
                timeout_seconds=ocr_timeout,
            )
            if ocr_code != 0:
                raise RuntimeError(
                    f"{self._format_external_failure('OCR', ocr_code, ocr_output)} (log: {ocr_log})"
                )
            page_count = len(checkpoint.all_pages(pdf.stem))

            self.call_from_thread(
                self._set_parse_runtime_status,
                stage="Indexing chunks",
                detail=f"Parsed {page_count} pages. Building embeddings... | {gpu_note}",
            )
            index_cmd = [
                sys.executable,
                "-u",
                "-m",
                "src.indexer",
                pdf.stem,
                "--reindex",
            ]
            if not use_gpu_for_embeddings:
                index_cmd.append("--allow-cpu-embeddings")
            index_log = logs_dir / f"{pdf.stem}_{run_stamp}_index.log"
            self._write_stage_log(index_log, index_cmd, parse_env, "")
            index_code, index_output = self._run_external_job(
                index_cmd,
                env=parse_env,
                on_output=lambda line: self.call_from_thread(
                    self._ingest_output, "index", line
                ),
                live_log_path=index_log,
                timeout_seconds=INDEX_TIMEOUT_SECONDS,
            )
            if index_code != 0:
                raise RuntimeError(
                    f"{self._format_external_failure('Indexing', index_code, index_output)} (log: {index_log})"
                )

            collection = get_collection(pdf.stem)
            ensure_collection_ready(collection)
            chunks = collection.count()
            if chunks <= 0:
                raise RuntimeError(
                    "Index completed without usable chunks; retry /parse"
                )

            self.call_from_thread(
                self._set_parse_runtime_status,
                stage="Finalizing",
                detail=f"Indexed {chunks} chunks. Logs: {index_log.name}",
            )
            self.call_from_thread(
                self._finish_parse_and_index,
                pdf.stem,
                page_count,
                chunks,
                "",
            )
        except Exception as exc:  # noqa: BLE001
            detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            book_stem = Path(pdf_path).stem if pdf_path else ""
            self.call_from_thread(
                self._finish_parse_and_index,
                book_stem,
                0,
                0,
                f"{exc.__class__.__name__}: {detail}",
            )
        finally:
            self._release_workload(workload_token)
            if self._parse_workload_token == workload_token:
                self._parse_workload_token = None

    @staticmethod
    def _is_mineru_available() -> bool:
        if importlib.util.find_spec("magic_pdf") is not None:
            return True
        return importlib.util.find_spec("mineru") is not None

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parent.parent

    def _run_external_job(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        on_output: Callable[[str], None] | None = None,
        live_log_path: Path | None = None,
        timeout_seconds: int | None = None,
    ) -> tuple[int, str]:
        with self._process_lock:
            if self._process_shutdown_started or self._ingest_cancelled:
                raise RuntimeError(
                    "TUI is shutting down; refusing to start an external job."
                )
            proc = subprocess.Popen(
                command,
                cwd=str(self._project_root()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
                env=env,
                start_new_session=True,
            )
            self._active_processes.add(proc)
        lines: list[str] = []
        log_handle = None
        line_queue: Queue[str | None] = Queue()
        timed_out = False
        reader: Thread | None = None
        timeout_at = (
            time.monotonic() + float(timeout_seconds)
            if timeout_seconds is not None and timeout_seconds > 0
            else None
        )

        def _reader() -> None:
            if proc.stdout is None:
                line_queue.put(None)
                return
            try:
                for raw_line in proc.stdout:
                    line_queue.put(raw_line.rstrip())
            finally:
                try:
                    proc.stdout.close()
                except Exception:  # noqa: BLE001, S110
                    pass
                line_queue.put(None)

        try:
            if live_log_path is not None:
                log_handle = live_log_path.open("a", encoding="utf-8")
            reader = Thread(target=_reader, daemon=True)
            reader.start()
            while True:
                if timeout_at is not None and time.monotonic() >= timeout_at:
                    timed_out = True
                    timeout_msg = (
                        f"Timed out after {int(timeout_seconds)}s; "
                        "sending SIGTERM to OCR process group."
                    )
                    lines.append(timeout_msg)
                    if log_handle is not None:
                        log_handle.write(f"{timeout_msg}\n")
                        log_handle.flush()
                    if on_output is not None:
                        on_output(timeout_msg)
                    self._terminate_process_group(proc)
                    break

                wait_seconds = 0.5
                if timeout_at is not None:
                    wait_seconds = max(
                        0.05, min(wait_seconds, timeout_at - time.monotonic())
                    )
                try:
                    line = line_queue.get(timeout=wait_seconds)
                except Empty:
                    if proc.poll() is not None and not reader.is_alive():
                        break
                    continue

                if line is None:
                    break
                if not line:
                    continue
                lines.append(line)
                if log_handle is not None:
                    log_handle.write(f"{line}\n")
                    log_handle.flush()
                if on_output is not None:
                    on_output(line)

            if timed_out:
                try:
                    returncode = proc.wait(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    self._kill_process_group(proc)
                    returncode = proc.wait()
            else:
                returncode = proc.wait()
        except BaseException:
            self._stop_process_group(proc)
            raise
        finally:
            try:
                # A normal leader exit can leave grandchildren alive in its session.
                if self._process_group_exists(proc):
                    self._stop_process_group(proc)
            finally:
                try:
                    if reader is not None:
                        reader.join(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
                    elif proc.stdout is not None:
                        proc.stdout.close()
                    if log_handle is not None:
                        log_handle.close()
                finally:
                    with self._process_lock:
                        self._active_processes.discard(proc)
        output = "\n".join(lines)
        return returncode, output

    def _terminate_active_processes(self) -> None:
        with self._process_lock:
            self._process_shutdown_started = True
            processes = list(self._active_processes)
        for proc in processes:
            self._stop_process_group(proc)

    @classmethod
    def _stop_process_group(cls, proc: subprocess.Popen[str]) -> None:
        """Stop a process session, checking descendants after the leader exits."""
        cls._terminate_process_group(proc)
        deadline = time.monotonic() + PROCESS_TERMINATE_GRACE_SECONDS
        while time.monotonic() < deadline:
            proc.poll()
            if not cls._process_group_exists(proc):
                break
            time.sleep(0.05)
        if cls._process_group_exists(proc):
            cls._kill_process_group(proc)
        try:
            proc.wait(timeout=PROCESS_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            cls._kill_process_group(proc)
            proc.wait()

    @staticmethod
    def _process_group_exists(proc: subprocess.Popen[str]) -> bool:
        try:
            os.killpg(proc.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    @staticmethod
    def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001, S110
                pass

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001, S110
                pass

    @staticmethod
    def _ocr_timeout_seconds(mode: str) -> int:
        normalized = mode.strip().lower()
        if normalized == "mineru":
            return MINERU_OCR_TIMEOUT_SECONDS
        if normalized == "marker":
            return MARKER_OCR_TIMEOUT_SECONDS
        return PYMUPDF4LLM_OCR_TIMEOUT_SECONDS

    @staticmethod
    def _write_stage_log(
        log_path: Path,
        command: list[str],
        env: dict[str, str] | None,
        output: str,
    ) -> None:
        redacted_env = {}
        if env:
            for key in (
                "AXIOM_EMBED_DEVICE",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONUNBUFFERED",
                "PYTORCH_CUDA_ALLOC_CONF",
            ):
                if key in env:
                    redacted_env[key] = env[key]
        content = [
            f"command: {' '.join(command)}",
            f"env: {redacted_env}",
            "",
            output.strip(),
            "",
        ]
        log_path.write_text("\n".join(content), encoding="utf-8")

    @staticmethod
    def _build_parse_env(use_gpu_for_embeddings: bool) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if use_gpu_for_embeddings:
            env["AXIOM_EMBED_DEVICE"] = "cuda"
            if AxiomLMApp._torch_cuda_available():
                env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        else:
            env["AXIOM_EMBED_DEVICE"] = "cpu"
        return env

    @staticmethod
    def _torch_cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _nvidia_gpu_present() -> bool:
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
            return probe.returncode == 0 and bool(probe.stdout.strip())
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _gpu_runtime_note(use_gpu_for_embeddings: bool) -> str:
        if not use_gpu_for_embeddings:
            return "GPU embeddings disabled (toggle with /gpu-parsing)"
        if AxiomLMApp._torch_cuda_available():
            return "GPU embeddings enabled (CUDA)"
        if AxiomLMApp._nvidia_gpu_present():
            return "GPU requested but Torch is CPU-only (install CUDA torch wheel)"
        return "GPU requested but no CUDA GPU runtime detected"

    @staticmethod
    def _format_external_failure(stage: str, returncode: int, output: str) -> str:
        if returncode < 0:
            status = f"terminated by signal {-returncode}"
        else:
            status = f"exit code {returncode}"

        if not output.strip():
            return f"{stage} failed ({status})."

        lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
        priority_patterns = (
            "cuda out of memory",
            "outofmemoryerror",
            "runtimeerror:",
            "error:",
            "timed out after",
            "marker gpu oom",
            "failed",
        )

        for pattern in priority_patterns:
            for line in reversed(lines):
                if pattern in line.lower():
                    return f"{stage} failed ({status}). {line}"

        tail = "\n".join(lines[-6:])
        return f"{stage} failed ({status}). {tail}"

    def _release_ollama_models(self) -> None:
        """Unload currently loaded Ollama models so OCR has free VRAM."""
        OllamaClient().unload_and_wait()


def main():
    AxiomLMApp().run()


if __name__ == "__main__":
    main()
