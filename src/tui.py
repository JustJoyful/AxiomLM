"""
tui.py — AxiomLM Terminal UI for querying documents.

Provides a Textual application with a sidebar to switch between indexed textbooks,
and a chat panel to interact with Gemma via Ollama using RAG.
"""

import json
import time
from pathlib import Path
from uuid import uuid4

from httpx import Client, ConnectError, HTTPStatusError, ReadTimeout
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer
from textual.worker import Worker
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    ProgressBar,
    Select,
    Static,
    TextArea,
)

from src.config import OLLAMA_MODEL, OLLAMA_URL, TOP_K_RESULTS
from src.db import embed_query, get_collection, list_collections


PROMPT_TEMPLATE = """You are a study assistant for engineering students.
Answer ONLY using the provided context from the textbook.
Do not use any prior knowledge. If the context does not contain the answer, say:
"This topic is not covered in the loaded documents."
Always cite your source at the end: [Source: {book} · {chapter} · p.{pages}]
Be concise and precise.
Use LaTeX notation for all equations: $...$ for inline, $$...$$ for display.

Context:
{context}

Question: {question}

Answer:"""


class QueryComposer(TextArea):
    """Multi-line query input: Enter sends, Shift+Enter inserts newline."""

    BINDINGS = [
        Binding("enter", "submit_query", show=False, priority=True),
        Binding("shift+enter", "insert_soft_newline", show=False, priority=True),
    ]

    class Submitted(Message):
        """Posted when the user submits the query composer."""

        def __init__(self, composer: "QueryComposer", value: str) -> None:
            super().__init__()
            self.composer = composer
            self.value = value

    def action_submit_query(self) -> None:
        self.post_message(self.Submitted(self, self.text))

    def action_insert_soft_newline(self) -> None:
        self.insert("\n")


class SelectableResponse(TextArea):
    """Read-only response body that supports text selection and copy."""

    BINDINGS = [
        Binding("ctrl+a", "select_all", show=False, priority=True),
    ]

    def __init__(self, text: str = "", classes: str | None = None) -> None:
        super().__init__(
            text=text,
            read_only=True,
            soft_wrap=True,
            tab_behavior="focus",
            show_line_numbers=False,
            compact=True,
            highlight_cursor_line=False,
            classes=classes,
        )


class UserMessage(Static):
    """Styled chat block for user messages."""

    DEFAULT_CSS = """
    UserMessage {
        margin: 1 6 0 2;
        padding: 0 1;
        border-left: tall ansi_cyan;
        background: transparent;
    }

    UserMessage .message-label {
        color: ansi_cyan;
        text-style: bold;
    }
    UserMessage .message-content {
        color: ansi_white;
    }
    """

    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content

    def compose(self) -> ComposeResult:
        yield Label("You", classes="message-label")
        yield Static(self._content, classes="message-content")


class AssistantMessage(Static):
    """Styled chat block for assistant messages."""

    thinking = reactive(False)
    dots = reactive(0)

    DEFAULT_CSS = """
    AssistantMessage {
        margin: 1 6 0 2;
        padding: 0 1;
        border-left: tall ansi_green;
        background: transparent;
    }

    AssistantMessage.error {
        border-left: tall ansi_red;
    }

    AssistantMessage .message-label {
        color: ansi_green;
        text-style: bold;
    }

    AssistantMessage.error .message-label {
        color: ansi_red;
    }
    AssistantMessage .message-content {
        color: ansi_white;
        border: none;
        background: transparent;
        margin: 0;
        padding: 0;
        height: auto;
        min-height: 1;
    }
    """

    def __init__(self, content: str = "", thinking: bool = False, is_error: bool = False) -> None:
        super().__init__()
        self._content = content
        self._is_error = is_error
        self._thinking_timer: Timer | None = None
        self.thinking = thinking
        if is_error:
            self.add_class("error")

    def compose(self) -> ComposeResult:
        yield Label("", classes="message-label")
        yield SelectableResponse("", classes="message-content")

    def on_mount(self) -> None:
        self._thinking_timer = self.set_interval(0.35, self._advance_thinking, pause=not self.thinking)
        self._render_content()

    def _advance_thinking(self) -> None:
        self.dots = (self.dots + 1) % 4

    def watch_thinking(self, thinking: bool) -> None:
        if self._thinking_timer is not None:
            if thinking:
                self._thinking_timer.resume()
            else:
                self._thinking_timer.pause()
        self._render_content()

    def watch_dots(self, dots: int) -> None:
        if self.thinking:
            self._render_content()

    def _render_content(self) -> None:
        if not self.is_mounted:
            return
        label = self.query_one(".message-label", Label)
        body = self.query_one(".message-content", SelectableResponse)

        if self.thinking:
            suffix = "." * (self.dots % 4)
            label.update(f"AxiomLM{suffix}")
            body.load_text("")
            return

        if self._is_error:
            label.update("AxiomLM Error")
            body.load_text(self._content)
            return

        label.update("AxiomLM")
        body.load_text(self._content)

    def stream_token(self, token: str) -> None:
        if self._is_error:
            self._is_error = False
            self.remove_class("error")
        if self.thinking:
            self.thinking = False
        self._content += token
        self._render_content()

    def finalize(self, content: str, is_error: bool = False) -> None:
        self._is_error = is_error
        if is_error:
            self.add_class("error")
        else:
            self.remove_class("error")
        self._content = content
        self.thinking = False
        self._render_content()


class AxiomLMApp(App):
    """AxiomLM terminal UI with sidebar, chat widgets, and Gemma integration."""

    CSS = """
    Screen {
        layout: vertical;
        /* Force true transparent background to match terminal theme completely */
        background: transparent; 
    }

    #main_container {
        layout: horizontal;
        height: 1fr;
    }

    /* ── Sidebar ───────────────────────────────────────────────────────── */
    #sidebar {
        width: 35%;
        min-width: 64;
        border-right: solid ansi_bright_black;
        background: transparent;
        padding-top: 1;
        overflow-y: auto;
        scrollbar-size: 1 1;
    }

    #brand {
        padding: 0 1;
        layout: vertical;
        height: auto;
        margin-bottom: 0;
    }

    #headline {
        layout: horizontal;
        align: left middle;
        height: auto;
        margin-bottom: 1;
    }

    #mascot_art {
        width: auto;
        height: auto;
        background: transparent;
    }

    #brand_title {
        text-style: bold;
        color: ansi_cyan;
        width: auto;
        margin-left: 1;
        content-align: center middle;
    }

    #brand_subtitle {
        color: ansi_bright_black;
        text-style: italic;
        width: 100%;
        text-align: left;
        margin-bottom: 1;
    }

    .status_label {
        padding: 0 1;
        color: ansi_bright_white;
        text-style: bold;
    }


    #empty_state {
        padding: 1 2;
        color: ansi_bright_black;
    }

    .hidden {
        display: none;
    }

    ListView {
        height: 1fr;
        margin: 0 1 1 1;
        border: none;
        background: transparent;
        scrollbar-size: 1 1;
    }

    ListItem {
        padding: 0 1;
        background: transparent;
    }
    
    ListItem > Label {
        color: ansi_bright_white;
        background: transparent;
    }

    ListItem.-highlight {
        background: ansi_bright_black;
    }
    
    ListItem.-highlight > Label {
        color: ansi_bright_cyan;
        text-style: bold;
        background: transparent;
    }

    /* ── Inputs and Selects (Minimalist styling) ───────────────────────── */
    Select, Input, Button {
        border: solid ansi_bright_black;
        background: transparent;
        color: ansi_white;
        margin: 0 1 1 1;
    }

    Select:focus, Input:focus, Button:focus {
        border: solid ansi_cyan;
    }
    
    Button:hover {
        background: ansi_bright_black;
    }

    /* Removing internal backgrounds for Select widget components */
    Select > SelectCurrent {
        background: transparent;
        border: none;
        color: ansi_white;
    }
    SelectOverlay {
        background: ansi_black;
        border: solid ansi_bright_black;
    }
    SelectOverlay > OptionList {
        background: transparent;
    }
    OptionList > .option-list--option {
        background: transparent;
        color: ansi_white;
    }
    OptionList > .option-list--option-highlighted {
        background: ansi_bright_black;
        color: ansi_cyan;
    }

    /* ── Chat panel ────────────────────────────────────────────────────── */
    #chat_panel {
        width: 70%;
        layout: vertical;
        background: transparent;
    }

    #chat_scroll {
        height: 1fr;
        padding: 1 2;
    }

    #query_input {
        dock: bottom;
        margin: 1 2;
        height: 3;
        max-height: 8;
    }

    /* ── Utilities ─────────────────────────────────────────────────────── */
    #parse_status_card {
        margin: 0 1 1 1;
        padding: 0 1;
        border: round ansi_bright_black;
        background: transparent;
        height: auto;
    }

    #parse_status_title {
        color: ansi_cyan;
        text-style: bold;
    }

    #parse_status_stage {
        color: ansi_bright_white;
    }

    #parse_status_engine {
        color: ansi_bright_white;
    }

    #parse_status_eta {
        color: ansi_bright_black;
    }

    #parse_status_detail {
        color: ansi_bright_black;
        text-style: italic;
    }

    #parse_progress {
        margin: 0 1 1 1;
        background: transparent;
    }
    ProgressBar > .bar--bar {
        color: ansi_bright_black;
        background: transparent;
    }
    ProgressBar > .bar--complete {
        color: ansi_cyan;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
        Binding("tab", "toggle_focus", "Focus", show=True),
        Binding("escape", "cancel_query", "Cancel", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.active_collection_name: str | None = None
        self.selected_model: str = OLLAMA_MODEL
        self._book_labels: dict[str, Label] = {}
        self._book_counts: dict[str, int] = {}
        self._pending_assistants: dict[str, AssistantMessage] = {}
        self._is_parsing = False
        self._parse_progress_timer: Timer | None = None
        self._parse_started_at: float | None = None
        self._parse_target_progress: float = 0.0
        self._parse_current_stage: str = "Idle"
        self._parse_current_engine: str = "-"
        self._parse_detail: str = ""
        self._active_query_worker: Worker | None = None
        self._active_query_request_id: str | None = None
        self._cancelled_query_ids: set[str] = set()
        self._query_input_min_height = 3
        self._query_input_max_height = 8

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="main_container"):
            with Vertical(id="sidebar"):
                with Vertical(id="brand"):
                    with Horizontal(id="headline"):
                        yield Static(id="mascot_art")
                        yield Label("AxiomLM", id="brand_title")
                    yield Label("Your textbooks. Locally. Answered.", id="brand_subtitle")
                
                yield Label("Ollama Model", classes="status_label")
                yield Select([("Loading models...", "__loading__")], id="model_select", value="__loading__")
                yield Button("Refresh Models", id="refresh_models_btn")
                
                yield Label("Parse PDF", classes="status_label")
                yield Input(placeholder="/path/to/book.pdf", id="pdf_path_input")
                yield Select(
                    [
                        ("Auto (smart routing)", "auto"),
                        ("Marker (clean PDFs)", "marker"),
                        ("MinerU (scanned PDFs)", "mineru"),
                    ],
                    id="ocr_mode_select",
                    value="auto",
                )
                yield Button("Parse + Index", id="parse_pdf_btn")
                with Vertical(id="parse_status_card", classes="hidden"):
                    yield Label("Parse / Index Status", id="parse_status_title")
                    yield Label("Stage: Idle", id="parse_status_stage")
                    yield Label("Engine: -", id="parse_status_engine")
                    yield Label("ETA: --:--", id="parse_status_eta")
                    yield Label("", id="parse_status_detail")
                yield ProgressBar(total=100, show_percentage=True, show_eta=False, id="parse_progress", classes="hidden")
                
                yield Label("Loaded Books", classes="status_label")
                yield Label("No books indexed.", id="empty_state", classes="hidden")
                yield ListView(id="book_list")

            with Vertical(id="chat_panel"):
                yield ScrollableContainer(id="chat_scroll")
                yield QueryComposer(
                    id="query_input",
                    compact=True,
                    highlight_cursor_line=False,
                    placeholder="Ask a question (Enter to send, Shift+Enter for newline)...",
                )

        yield Footer()

    async def on_mount(self) -> None:
        await self._populate_sidebar()
        self._set_header()
        self.query_one("#query_input", QueryComposer).focus()
        self._resize_query_input()
        
        self._render_mascot()
        self._refresh_models()

    async def _populate_sidebar(self) -> None:
        book_list = self.query_one("#book_list", ListView)
        await book_list.clear()
        self._book_labels.clear()
        self._book_counts.clear()

        try:
            collections = sorted(list_collections())
        except Exception:
            collections = []

        if not collections:
            self.active_collection_name = None
            self._toggle_empty_state(has_books=False)
            return

        for stem in collections:
            try:
                count = get_collection(stem).count()
            except Exception:
                count = 0
            self._book_counts[stem] = count
            label = Label("", classes="book-label")
            self._book_labels[stem] = label
            book_list.append(ListItem(label, id=stem))

        self.active_collection_name = collections[0]
        book_list.index = 0
        self._toggle_empty_state(has_books=True)
        self._refresh_book_labels()

    def _format_book_name(self, stem: str) -> str:
        return stem.replace("_", " ").title()

    def _refresh_book_labels(self) -> None:
        for stem, label in self._book_labels.items():
            prefix = "📖 " if stem == self.active_collection_name else "   "
            title = self._format_book_name(stem)
            count = self._book_counts.get(stem, 0)
            label.update(f"{prefix}{title} ({count})")

    def _toggle_empty_state(self, has_books: bool) -> None:
        empty_state = self.query_one("#empty_state", Label)
        book_list = self.query_one("#book_list", ListView)
        if has_books:
            empty_state.add_class("hidden")
            book_list.remove_class("hidden")
        else:
            empty_state.remove_class("hidden")
            book_list.add_class("hidden")

    def _set_header(self) -> None:
        self.title = "AxiomLM"
        model_name = self.selected_model or OLLAMA_MODEL
        if self.active_collection_name:
            book = self._format_book_name(self.active_collection_name)
            self.sub_title = f"Active context: {book} · Engine: {model_name}"
        else:
            self.sub_title = f"No book selected · Engine: {model_name}"

    def action_toggle_focus(self) -> None:
        input_widget = self.query_one("#query_input", QueryComposer)
        book_list = self.query_one("#book_list", ListView)
        if self.focused is input_widget:
            book_list.focus()
        else:
            input_widget.focus()

    def _resize_query_input(self) -> None:
        composer = self.query_one("#query_input", QueryComposer)
        line_breaks = composer.text.count("\n")
        target = min(self._query_input_max_height, self._query_input_min_height + line_breaks)
        composer.styles.height = target

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.control.id != "query_input":
            return
        self._resize_query_input()

    def action_cancel_query(self) -> None:
        worker = self._active_query_worker
        request_id = self._active_query_request_id
        if worker is None or not worker.is_running or request_id is None:
            self.notify("No query is currently running.", severity="information")
            return

        self._cancelled_query_ids.add(request_id)
        worker.cancel()
        self.notify("Cancelling query...", severity="warning")

    async def action_clear_chat(self) -> None:
        chat_scroll = self.query_one("#chat_scroll", ScrollableContainer)
        await chat_scroll.remove_children("*")
        self._pending_assistants.clear()

    async def _append_chat_widget(self, widget: Static, animate: bool = True) -> None:
        chat_scroll = self.query_one("#chat_scroll", ScrollableContainer)
        await chat_scroll.mount(widget)
        chat_scroll.scroll_end(animate=animate)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        selected_book = event.item.id if event.item and event.item.id else None
        if selected_book and selected_book != self.active_collection_name:
            self.active_collection_name = selected_book
            self._refresh_book_labels()
            self._set_header()
            self.query_one("#query_input", QueryComposer).focus()

    async def on_query_composer_submitted(self, event: QueryComposer.Submitted) -> None:
        if event.composer.id != "query_input":
            return

        if self._active_query_worker is not None and self._active_query_worker.is_running:
            self.notify("A query is already running. Press Esc to cancel it.", severity="warning")
            return

        query = event.value.strip()
        if not query:
            return

        if not self.active_collection_name:
            self.notify("No active book selected. Please parse a PDF first.", severity="error")
            return

        input_widget = self.query_one("#query_input", QueryComposer)
        input_widget.clear()
        self._resize_query_input()

        await self._append_chat_widget(UserMessage(query))

        request_id = uuid4().hex
        assistant = AssistantMessage(thinking=True)
        self._pending_assistants[request_id] = assistant
        await self._append_chat_widget(assistant, animate=False)

        active_book = self.active_collection_name
        model_name = self.selected_model or OLLAMA_MODEL
        worker = self.run_worker(
            lambda: self.execute_rag_query(query, active_book, model_name, request_id),
            thread=True,
            group="queries",
            exit_on_error=False,
        )
        self._active_query_worker = worker
        self._active_query_request_id = request_id

    def execute_rag_query(self, query: str, book_stem: str, model_name: str, request_id: str) -> None:
        try:
            if self._is_query_cancelled(request_id):
                self.call_from_thread(self._finalize_query_cancelled, request_id)
                return

            q_vec = embed_query(query)
            collection = get_collection(book_stem)
            results = collection.query(
                query_embeddings=[q_vec],
                n_results=TOP_K_RESULTS,
                include=["documents", "metadatas"],
            )

            if self._is_query_cancelled(request_id):
                self.call_from_thread(self._finalize_query_cancelled, request_id)
                return

            if not results["documents"] or not results["documents"][0]:
                self.call_from_thread(
                    self._finalize_query,
                    request_id,
                    "No relevant content found. Try rephrasing your question.",
                    True,
                )
                return

            chunks = results["documents"][0]
            metadatas = results["metadatas"][0]

            context_blocks = []
            for chunk, meta in zip(chunks, metadatas):
                h1 = meta.get("Header 1", "Unknown Chapter")
                h2 = meta.get("Header 2", "Unknown Section")
                page_info = meta.get("page", "N/A")
                context_blocks.append(f"[{h1} · {h2} · p.{page_info}]\n{chunk}")

            prompt = PROMPT_TEMPLATE.format(
                book=book_stem,
                chapter=",".join(sorted({m.get("Header 1", "Unknown") for m in metadatas})),
                pages=",".join(sorted({str(m.get("page", "?")) for m in metadatas})),
                context="\n\n".join(context_blocks),
                question=query,
            )

            accumulated_response = ""
            was_cancelled = False
            with Client(timeout=240.0) as client:
                with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": True},
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if self._is_query_cancelled(request_id):
                            was_cancelled = True
                            break
                        if not line:
                            continue
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            accumulated_response += token
                            self.call_from_thread(self._stream_query_token, request_id, token)

            if was_cancelled:
                self.call_from_thread(self._finalize_query_cancelled, request_id)
                return

            if not accumulated_response.strip():
                accumulated_response = "I couldn't generate a response from the model."

            self.call_from_thread(self._finalize_query, request_id, accumulated_response, False)

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
                    "Select an installed model from the sidebar and retry."
                )
            else:
                detail = f"Ollama request failed ({exc.response.status_code}): {exc.response.reason_phrase}"
            self.call_from_thread(self._finalize_query, request_id, detail, True)
        except Exception as exc:
            if self._is_query_cancelled(request_id):
                self.call_from_thread(self._finalize_query_cancelled, request_id)
                return
            detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            self.call_from_thread(
                self._finalize_query,
                request_id,
                f"{exc.__class__.__name__}: {detail}",
                True,
            )

    def _is_query_cancelled(self, request_id: str) -> bool:
        return request_id in self._cancelled_query_ids

    async def _stream_query_token(self, request_id: str, token: str) -> None:
        assistant = self._pending_assistants.get(request_id)
        if assistant is None or not assistant.is_attached:
            return
        assistant.stream_token(token)
        self.query_one("#chat_scroll", ScrollableContainer).scroll_end(animate=False)

    def _fetch_ollama_models(self) -> list[str]:
        with Client(timeout=4.0) as client:
            response = client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            payload = response.json()
        models = [model.get("name") for model in payload.get("models", []) if model.get("name")]
        return sorted(set(models))

    def _refresh_models(self) -> None:
        try:
            models = self._fetch_ollama_models()
            self._apply_model_options(models)
        except Exception as exc:
            self._apply_model_options([])

    def _apply_model_options(self, models: list[str]) -> None:
        model_select = self.query_one("#model_select", Select)

        if not models:
            model_select.set_options([("Ollama unavailable", "__offline__")])
            model_select.value = "__offline__"
            model_select.disabled = True
            self.selected_model = ""
            self._set_header()
            return

        model_select.disabled = False
        model_select.set_options([(model, model) for model in models])
        preferred = self.selected_model if self.selected_model in models else models[0]
        model_select.value = preferred
        self.selected_model = preferred
        self._set_header()

    def _set_parse_controls_enabled(self, enabled: bool) -> None:
        parse_button = self.query_one("#parse_pdf_btn", Button)
        path_input = self.query_one("#pdf_path_input", Input)
        mode_select = self.query_one("#ocr_mode_select", Select)
        progress = self.query_one("#parse_progress", ProgressBar)
        status_card = self.query_one("#parse_status_card", Vertical)

        parse_button.disabled = not enabled
        path_input.disabled = not enabled
        mode_select.disabled = not enabled
        parse_button.label = "Parse + Index" if enabled else "Parsing..."

        if enabled:
            if self._parse_progress_timer is not None:
                self._parse_progress_timer.stop()
                self._parse_progress_timer = None
            progress.add_class("hidden")
            progress.update(progress=0)
            status_card.add_class("hidden")
            self._parse_started_at = None
            self._parse_target_progress = 0.0
            self._parse_current_stage = "Idle"
            self._parse_current_engine = "-"
            self._parse_detail = ""
            return

        status_card.remove_class("hidden")
        progress.remove_class("hidden")
        progress.update(progress=1)
        self._set_parse_runtime_status(
            stage="Preparing parse job",
            detail="Setting up OCR pipeline...",
            target_progress=8.0,
        )
        self._parse_progress_timer = self.set_interval(0.35, self._tick_parse_progress)

    def _format_duration(self, seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _set_parse_runtime_status(
        self,
        stage: str | None = None,
        engine: str | None = None,
        detail: str | None = None,
        target_progress: float | None = None,
    ) -> None:
        if stage is not None:
            self._parse_current_stage = stage
        if engine is not None:
            self._parse_current_engine = engine
        if detail is not None:
            self._parse_detail = detail
        if target_progress is not None:
            self._parse_target_progress = max(0.0, min(100.0, target_progress))

        stage_label = self.query_one("#parse_status_stage", Label)
        engine_label = self.query_one("#parse_status_engine", Label)
        eta_label = self.query_one("#parse_status_eta", Label)
        detail_label = self.query_one("#parse_status_detail", Label)

        stage_label.update(f"Stage: {self._parse_current_stage}")
        engine_label.update(f"Engine: {self._parse_current_engine}")
        detail_label.update(self._parse_detail)

        if self._parse_started_at is None:
            eta_label.update("ETA: estimating...")

    def _tick_parse_progress(self) -> None:
        if not self._is_parsing:
            return

        progress = self.query_one("#parse_progress", ProgressBar)
        current = progress.progress if progress.progress is not None else 0
        target = min(100.0, max(current, self._parse_target_progress))

        if current < target:
            step = max(0.2, (target - current) * 0.16)
            current = min(target, current + step)
            progress.update(progress=current)
        elif current < 99:
            # Keep a subtle heartbeat so the UI feels alive during long operations.
            progress.update(advance=0.05)
            current = progress.progress if progress.progress is not None else current

        eta_label = self.query_one("#parse_status_eta", Label)
        if self._parse_started_at is None or current < 1:
            eta_label.update("ETA: estimating...")
            return

        elapsed = max(0.1, time.monotonic() - self._parse_started_at)
        remaining = max(0.0, (elapsed / current) * (100 - current))
        eta_label.update(
            f"Elapsed: {self._format_duration(elapsed)} · ETA: ~{self._format_duration(remaining)}"
        )

    async def _start_parse_and_index(self) -> None:
        if self._is_parsing:
            self.notify("A parse job is already running.", severity="warning")
            return

        path_input = self.query_one("#pdf_path_input", Input)
        raw_path = path_input.value.strip()
        if not raw_path:
            self.notify("Enter a PDF path first.", severity="warning")
            return

        pdf_path = Path(raw_path).expanduser()
        if not pdf_path.is_absolute():
            pdf_path = (Path.cwd() / pdf_path).resolve()

        if not pdf_path.exists() or not pdf_path.is_file():
            self.notify("PDF file not found.", severity="error")
            return
        if pdf_path.suffix.lower() != ".pdf":
            self.notify("Only .pdf files are supported.", severity="error")
            return

        mode_select = self.query_one("#ocr_mode_select", Select)
        mode_value = str(mode_select.value) if mode_select.value and mode_select.value != Select.BLANK else "auto"

        self._is_parsing = True
        self._parse_started_at = time.monotonic()
        self._parse_target_progress = 8.0
        self._set_parse_controls_enabled(False)
        requested_engine = {
            "auto": "Auto (choosing best engine...)",
            "marker": "Marker",
            "mineru": "MinerU",
        }.get(mode_value, mode_value)
        self._set_parse_runtime_status(
            stage="Preparing parse job",
            engine=requested_engine,
            detail=f"Book: {pdf_path.name}",
            target_progress=8.0,
        )
        await self._append_chat_widget(
            AssistantMessage(f"Starting parse + index for `{pdf_path.name}` using `{mode_value}` mode.")
        )

        self.run_worker(
            lambda: self._parse_and_index_worker(str(pdf_path), mode_value),
            thread=True,
            group="parse",
            exclusive=True,
            exit_on_error=False,
        )

    def _parse_and_index_worker(self, pdf_path: str, mode: str) -> None:
        try:
            from src.indexer import index_book
            from src.ocr import resolve_mode, route

            pdf = Path(pdf_path)
            self.call_from_thread(
                self._set_parse_runtime_status,
                stage="Releasing GPU memory",
                detail="Unloading Ollama sessions to free VRAM before OCR.",
                target_progress=12.0,
            )
            self._release_ollama_models()

            resolved_mode, reason = resolve_mode(pdf, mode)
            engine_label = "Marker" if resolved_mode == "marker" else "MinerU"
            if mode == "auto":
                engine_label = f"{engine_label} (auto)"
            self.call_from_thread(
                self._set_parse_runtime_status,
                stage="Parsing PDF pages",
                engine=engine_label,
                detail=reason,
                target_progress=72.0,
            )

            pages = route(pdf, mode=resolved_mode, force=False)
            self.call_from_thread(
                self._set_parse_runtime_status,
                stage="Indexing chunks",
                detail=f"Parsed {len(pages)} pages. Building embeddings...",
                target_progress=96.0,
            )
            chunks = index_book(pdf.stem, reindex=True)
            self.call_from_thread(
                self._set_parse_runtime_status,
                stage="Finalizing",
                detail=f"Indexed {chunks} chunks.",
                target_progress=100.0,
            )
            self.call_from_thread(self._finish_parse_and_index, pdf.stem, len(pages), chunks, "")
        except Exception as exc:
            detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            self.call_from_thread(self._finish_parse_and_index, "", 0, 0, f"{exc.__class__.__name__}: {detail}")

    def _release_ollama_models(self) -> None:
        """Unload currently loaded Ollama models so OCR has free VRAM."""
        try:
            with Client(timeout=8.0) as client:
                response = client.get(f"{OLLAMA_URL}/api/ps")
                response.raise_for_status()
                loaded = response.json().get("models", [])
                for model in loaded:
                    name = model.get("name")
                    if not name:
                        continue
                    client.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={"model": name, "prompt": "", "stream": False, "keep_alive": 0},
                    )
        except Exception:
            return

    async def _finish_parse_and_index(
        self, book_stem: str, page_count: int, chunk_count: int, error_message: str
    ) -> None:
        self._is_parsing = False
        self._set_parse_controls_enabled(True)

        if error_message:
            guidance = ""
            if "OutOfMemoryError" in error_message or "CUDA out of memory" in error_message:
                guidance = " Free GPU VRAM and retry (close GPU-heavy apps / Ollama sessions)."
            await self._append_chat_widget(
                AssistantMessage(f"Parse/index failed: {error_message}.{guidance}".strip(), is_error=True)
            )
            return

        await self._append_chat_widget(
            AssistantMessage(
                f"Done: parsed `{book_stem}` ({page_count} pages) and indexed {chunk_count} chunks."
            )
        )

        await self._populate_sidebar()
        if book_stem in self._book_labels:
            self.active_collection_name = book_stem
            book_list = self.query_one("#book_list", ListView)
            for idx, item in enumerate(book_list.children):
                if getattr(item, "id", None) == book_stem:
                    book_list.index = idx
                    break
            self._refresh_book_labels()
            self._set_header()
        self.query_one("#query_input", QueryComposer).focus()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "refresh_models_btn":
            self._refresh_models()
            return

        if event.button.id == "parse_pdf_btn":
            await self._start_parse_and_index()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "model_select":
            return

        if event.value == Select.BLANK:
            return
        value = str(event.value)
        if value in {"__loading__", "__offline__"}:
            return
        self.selected_model = value
        self._set_header()

    def _clear_query_state(self, request_id: str) -> None:
        if request_id in self._cancelled_query_ids:
            self._cancelled_query_ids.remove(request_id)
        if self._active_query_request_id == request_id:
            self._active_query_request_id = None
            self._active_query_worker = None

    async def _finalize_query_cancelled(self, request_id: str) -> None:
        await self._finalize_assistant_message(request_id, "Query cancelled.", is_error=False)
        self._clear_query_state(request_id)

    async def _finalize_query_transient_error(self, request_id: str, message: str) -> None:
        self.notify(message, severity="warning")
        await self._finalize_assistant_message(request_id, message, is_error=True)
        self._clear_query_state(request_id)

    async def _finalize_query(self, request_id: str, message: str, is_error: bool) -> None:
        await self._finalize_assistant_message(request_id, message, is_error=is_error)
        self._clear_query_state(request_id)

    async def _finalize_assistant_message(self, request_id: str, message: str, is_error: bool) -> None:
        assistant = self._pending_assistants.pop(request_id, None)
        if assistant is None or not assistant.is_attached:
            await self._append_chat_widget(AssistantMessage(message, is_error=is_error))
            return
        assistant.finalize(message, is_error=is_error)
        self.query_one("#chat_scroll", ScrollableContainer).scroll_end(animate=False)

    def _render_mascot(self) -> None:
        try:
            mascot_widget = self.query_one("#mascot_art", Static)
        except Exception:
            return

        logo = (
            " █████╗ ██╗  ██╗██╗ ██████╗ ███╗   ███╗  ██╗     ███╗   ███╗\n"
            "██╔══██╗╚██╗██╔╝██║██╔═══██╗████╗ ████║  ██║     ████╗ ████║\n"
            "███████║ ╚███╔╝ ██║██║   ██║██╔████╔██║  ██║     ██╔████╔██║\n"
            "██╔══██║ ██╔██╗ ██║██║   ██║██║╚██╔╝██║  ██║     ██║╚██╔╝██║\n"
            "██║  ██║██╔╝ ██╗██║╚██████╔╝██║ ╚═╝ ██║  ███████╗██║ ╚═╝ ██║\n"
            "╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝     ╚═╝  ╚══════╝╚═╝     ╚═╝"
        )
        mascot_widget.update(Text(logo, style="bold ansi_cyan"))


if __name__ == "__main__":
    AxiomLMApp().run()
