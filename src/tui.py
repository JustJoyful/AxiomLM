"""
tui.py — AxiomLM Terminal UI for querying documents.

Provides a Textual application with a sidebar to switch between indexed textbooks,
and a chat panel to interact with Gemma via Ollama using RAG.
"""

import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Callable
from uuid import uuid4

from httpx import Client, ConnectError, HTTPStatusError, ReadTimeout
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
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
    Markdown,
    ProgressBar,
    Select,
    Static,
    TextArea,
)

from src.config import (
    DISTANCE_THRESHOLD,
    INDEX_TIMEOUT_SECONDS,
    LOGS_DIR,
    MARKER_OCR_TIMEOUT_SECONDS,
    MINERU_OCR_TIMEOUT_SECONDS,
    OLLAMA_MIROSTAT,
    OLLAMA_MIROSTAT_ETA,
    OLLAMA_MIROSTAT_TAU,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    OLLAMA_URL,
    PROCESS_TERMINATE_GRACE_SECONDS,
    PYMUPDF4LLM_OCR_TIMEOUT_SECONDS,
    TOP_K_RESULTS,
)
from src.clipboard import copy_to_clipboard, extract_code_blocks
from src.db import delete_collection, embed_query, get_collection, list_collections


def latex_to_unicode(text: str) -> str:
    """
    Post-process response to replace LaTeX math with readable Unicode symbols.
    Strips $$ and $ delimiters and handles common symbols/notations.
    """
    # 1. Strip structural delimiters and sizing commands
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", r"\1", text)
    text = re.sub(r"\\left\s*|\\right\s*|\\big\s*|\\Big\s*|\\bigg\s*|\\Bigg\s*", "", text)
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
        r"\\alpha\b": "α", r"\\beta\b": "β", r"\\gamma\b": "γ", r"\\delta\b": "δ",
        r"\\epsilon\b": "ε", r"\\zeta\b": "ζ", r"\\eta\b": "η", r"\\theta\b": "θ",
        r"\\iota\b": "ι", r"\\kappa\b": "κ", r"\\lambda\b": "λ", r"\\mu\b": "μ",
        r"\\nu\b": "ν", r"\\xi\b": "ξ", r"\\pi\b": "π", r"\\rho\b": "ρ",
        r"\\sigma\b": "σ", r"\\tau\b": "τ", r"\\upsilon\b": "υ", r"\\phi\b": "φ",
        r"\\chi\b": "χ", r"\\psi\b": "ψ", r"\\omega\b": "ω",

        # Greek Uppercase
        r"\\Gamma\b": "Γ", r"\\Delta\b": "Δ", r"\\Theta\b": "Θ", r"\\Lambda\b": "Λ",
        r"\\Xi\b": "Ξ", r"\\Pi\b": "Π", r"\\Sigma\b": "Σ", r"\\Upsilon\b": "Υ",
        r"\\Phi\b": "Φ", r"\\Psi\b": "Ψ", r"\\Omega\b": "Ω",

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
        r"\^2\b": "²", r"\^3\b": "³", r"\^n\b": "ⁿ",
        r"_2\b": "₂", r"_3\b": "₃", r"_4\b": "₄",
        r"_n\b": "ₙ", r"_i\b": "ᵢ", r"_j\b": "ⱼ",
        r"_x\b": "ₓ", r"_y\b": "ᵧ",
    }
    for pattern, replacement in sub_super.items():
        text = re.sub(pattern, replacement, text)
        text = re.sub(pattern, replacement, text)

    return text


PROMPT_TEMPLATE = """You are Axie, a study assistant built into a terminal UI. Responses render in a Rich-enabled terminal supporting markdown tables and markup.

Important: You have access to conversation history below. Use it to understand references like "this topic", "that exercise", or "the previous example". If a follow-up question is asked and the history makes the intent clear, use the retrieved context to answer even if it seems tangential.

Formatting rules:
- Use **bold** for key terms and important values
- Use `code` for inline code/symbols, triple backticks for code blocks with language tag
- Avoid LaTeX commands like \frac, \left, \right, or \text. Use plain notation like P(A∩B)/P(A).
- Use a simple pipe | or forward slash / for math. NEVER use backslashes followed by spaces or slashes (like \ /).
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

Answer in study notes style. Never refuse due to missing context — always provide the best answer you can with available information.

Citation (last line only):
[{book} · {chapter} · p.{pages}]"""


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
        height: auto;
    }
    """

    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content

    def compose(self) -> ComposeResult:
        yield Label("You", classes="message-label")
        yield Static(RichMarkdown(self._content), classes="message-content")


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
    }

    AssistantMessage .message-telemetry {
        display: none;
        color: ansi_bright_black;
        text-style: dim italic;
        margin-top: 1;
    }

    AssistantMessage .message-telemetry.visible {
        display: block;
    }

    AssistantMessage .message-toolbar {
        display: none;
        margin-top: 1;
        height: auto;
        background: transparent;
    }

    AssistantMessage .message-toolbar.visible {
        display: block;
    }

    AssistantMessage .copy-btn {
        width: auto;
        min-width: 12;
        margin: 0 1 0 0;
        border: round ansi_bright_black;
        background: transparent;
    }

    AssistantMessage .copy-status {
        color: ansi_bright_black;
        text-style: dim;
        width: 1fr;
        content-align: left middle;
    }
    """

    def __init__(
        self,
        content: str = "",
        thinking: bool = False,
        is_error: bool = False,
        telemetry_badge: str | None = None,
    ) -> None:
        super().__init__()
        self._content = content
        self._is_error = is_error
        self._telemetry_badge = telemetry_badge or ""
        self._thinking_timer: Timer | None = None
        self.thinking = thinking
        if is_error:
            self.add_class("error")

    def compose(self) -> ComposeResult:
        yield Label("", classes="message-label")
        yield Static("", classes="message-content")
        yield Label("", classes="message-telemetry")
        with Horizontal(classes="message-toolbar"):
            yield Button("📋 Copy", classes="copy-btn copy-full-btn")
            yield Button("</> Code", classes="copy-btn copy-code-btn")
            yield Label("", classes="copy-status")

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
        body = self.query_one(".message-content", Static)
        telemetry = self.query_one(".message-telemetry", Label)
        toolbar = self.query_one(".message-toolbar", Horizontal)
        copy_status = self.query_one(".copy-status", Label)

        if self.thinking:
            suffix = "." * (self.dots % 4)
            label.update(f"AxiomLM{suffix}")
            body.update("")
            telemetry.update("")
            telemetry.remove_class("visible")
            toolbar.remove_class("visible")
            copy_status.update("")
            return

        if self._is_error:
            label.update("AxiomLM Error")
            body.update(self._content)
            telemetry.update("")
            telemetry.remove_class("visible")
            toolbar.remove_class("visible")
            copy_status.update("")
            return

        label.update("AxiomLM")
        body.update(RichMarkdown(self._content))
        toolbar.add_class("visible")
        if self._telemetry_badge:
            telemetry.update(self._telemetry_badge)
            telemetry.add_class("visible")
        else:
            telemetry.update("")
            telemetry.remove_class("visible")

    def stream_token(self, token: str) -> None:
        if self._is_error:
            self._is_error = False
            self.remove_class("error")
        if self.thinking:
            self.thinking = False
        self._content += token
        self._render_content()

    def finalize(self, content: str, is_error: bool = False, telemetry_badge: str | None = None) -> None:
        self._is_error = is_error
        if is_error:
            self.add_class("error")
        else:
            self.remove_class("error")
        self._telemetry_badge = "" if is_error else (telemetry_badge or "")
        self._content = content
        self.thinking = False
        self._render_content()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.has_class("copy-full-btn"):
            success, message = copy_to_clipboard(self._content)
            self._show_copy_status(success, message)
            event.stop()
            return
        if event.button.has_class("copy-code-btn"):
            code = extract_code_blocks(self._content)
            if not code:
                self._show_copy_status(False, "No fenced code blocks found.")
            else:
                success, message = copy_to_clipboard(code)
                self._show_copy_status(success, message)
            event.stop()

    def copy_full(self) -> tuple[bool, str]:
        return copy_to_clipboard(self._content)

    def copy_code_only(self) -> tuple[bool, str]:
        code = extract_code_blocks(self._content)
        if not code:
            return False, "No fenced code blocks found."
        return copy_to_clipboard(code)

    def _show_copy_status(self, success: bool, message: str) -> None:
        if not self.is_mounted:
            return
        status = self.query_one(".copy-status", Label)
        if success:
            status.styles.color = "ansi_green"
            status.styles.text_style = "bold"
            status.update(f"✓ {message}")
        else:
            status.styles.color = "ansi_red"
            status.styles.text_style = "bold"
            status.update(f"✗ {message}")
        self.set_timer(3.0, lambda: self._clear_copy_status())

    def _clear_copy_status(self) -> None:
        if not self.is_mounted:
            return
        status = self.query_one(".copy-status", Label)
        status.styles.color = "ansi_bright_black"
        status.styles.text_style = "dim"
        status.update("")


class BookListRow(Static):
    """Single sidebar row with title/count and a compact delete button."""

    DEFAULT_CSS = """
    BookListRow {
        layout: horizontal;
        width: 100%;
        height: 1;
        min-height: 1;
        align: left middle;
        background: transparent;
    }

    BookListRow .book-label {
        width: 1fr;
        height: 1;
        margin: 0;
        padding: 0;
        color: ansi_bright_white;
        background: transparent;
        content-align: left middle;
    }

    BookListRow .book-delete-btn {
        width: 5;
        min-width: 5;
        height: 1;
        min-height: 1;
        margin: 0 0 0 1;
        padding: 0 1;
        border: round ansi_red;
        background: transparent;
        color: ansi_red;
        text-style: bold;
        content-align: center middle;
    }

    BookListRow .book-delete-btn:hover {
        background: ansi_red 15%;
        color: ansi_bright_white;
    }
    """

    def __init__(self, book_stem: str) -> None:
        super().__init__()
        self.book_stem = book_stem
        self.book_label = Label("", classes="book-label")

    def compose(self) -> ComposeResult:
        yield self.book_label
        yield Button(
            "🗑️",
            classes="book-delete-btn",
            tooltip=f"Delete {self.book_stem}",
            compact=True,
            variant="error",
        )


class DeleteCollectionConfirm(ModalScreen[bool]):
    """Confirm irreversible collection deletion."""

    DEFAULT_CSS = """
    DeleteCollectionConfirm {
        align: center middle;
    }

    #delete_dialog {
        width: 68;
        max-width: 90%;
        height: auto;
        border: round ansi_red;
        background: ansi_black;
        padding: 1 2;
    }

    .delete-dialog-title {
        color: ansi_red;
        text-style: bold;
        margin-bottom: 1;
    }

    .delete-dialog-body {
        color: ansi_white;
    }

    .delete-dialog-warning {
        color: ansi_yellow;
        text-style: bold;
    }

    #delete_dialog_actions {
        margin-top: 1;
        height: auto;
        align: right middle;
    }

    #delete_dialog_actions Button {
        width: auto;
        min-width: 12;
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, book_name: str, chunk_count: int, estimated_size_mb: float) -> None:
        super().__init__()
        self.book_name = book_name
        self.chunk_count = chunk_count
        self.estimated_size_mb = estimated_size_mb

    def compose(self) -> ComposeResult:
        with Vertical(id="delete_dialog"):
            yield Label("Delete Collection", classes="delete-dialog-title")
            yield Static(
                f"Are you sure you want to delete '{self.book_name}'?",
                classes="delete-dialog-body",
            )
            yield Static("", classes="delete-dialog-body")
            yield Static("This will permanently erase:", classes="delete-dialog-warning")
            yield Static(
                f"• {self.chunk_count} indexed chunks\n"
                "• All vector embeddings and metadata\n"
                f"• ~{self.estimated_size_mb:.1f} MB of local storage",
                classes="delete-dialog-body",
            )
            yield Static("", classes="delete-dialog-body")
            yield Static("This action cannot be undone.", classes="delete-dialog-warning")
            with Horizontal(id="delete_dialog_actions"):
                yield Button("Cancel", id="delete_cancel_btn")
                yield Button("Delete", id="delete_confirm_btn", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "delete_confirm_btn":
            self.dismiss(True)
            return
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class TuningPanel(Static):
    """Compact, live retrieval controls shown under the query input."""

    class TopkChanged(Message):
        def __init__(self, value: int) -> None:
            super().__init__()
            self.value = value

    class DistanceChanged(Message):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

    TOP_K_MIN = 3
    TOP_K_MAX = 20
    DIST_MIN = 0.0
    DIST_MAX = 1.0
    DIST_STEP = 0.05

    def __init__(self, top_k: int, distance: float) -> None:
        super().__init__(id="query_tuning")
        self._top_k = self._clamp_top_k(top_k)
        self._distance = self._clamp_distance(distance)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="tuning-row"):
            yield Label("K", classes="tuning-key")
            yield Button("-", id="topk_dec_btn", classes="tune-btn", compact=True)
            yield Label("", id="topk_value", classes="tuning-value")
            yield Button("+", id="topk_inc_btn", classes="tune-btn", compact=True)
            yield Label("", classes="tuning-gap")
            yield Label("D", classes="tuning-key")
            yield Button("-", id="dist_dec_btn", classes="tune-btn", compact=True)
            yield Label("", id="dist_value", classes="tuning-value")
            yield Button("+", id="dist_inc_btn", classes="tune-btn", compact=True)

    def on_mount(self) -> None:
        self._refresh_labels()

    def set_values(self, top_k: int, distance: float) -> None:
        self._top_k = self._clamp_top_k(top_k)
        self._distance = self._clamp_distance(distance)
        self._refresh_labels()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "topk_dec_btn":
            self._top_k = self._clamp_top_k(self._top_k - 1)
            self._refresh_labels()
            self.post_message(self.TopkChanged(self._top_k))
            event.stop()
            return
        if button_id == "topk_inc_btn":
            self._top_k = self._clamp_top_k(self._top_k + 1)
            self._refresh_labels()
            self.post_message(self.TopkChanged(self._top_k))
            event.stop()
            return
        if button_id == "dist_dec_btn":
            self._distance = self._clamp_distance(self._distance - self.DIST_STEP)
            self._refresh_labels()
            self.post_message(self.DistanceChanged(self._distance))
            event.stop()
            return
        if button_id == "dist_inc_btn":
            self._distance = self._clamp_distance(self._distance + self.DIST_STEP)
            self._refresh_labels()
            self.post_message(self.DistanceChanged(self._distance))
            event.stop()

    def _refresh_labels(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#topk_value", Label).update(f"{self._top_k}")
        self.query_one("#dist_value", Label).update(f"{self._distance:.2f}")

    @classmethod
    def _clamp_top_k(cls, value: int) -> int:
        return max(cls.TOP_K_MIN, min(cls.TOP_K_MAX, int(value)))

    @classmethod
    def _clamp_distance(cls, value: float) -> float:
        rounded = round(float(value), 2)
        return max(cls.DIST_MIN, min(cls.DIST_MAX, rounded))


class AxiomLMApp(App):
    """AxiomLM terminal UI with sidebar, chat widgets, and Gemma integration."""
    top_k_results = reactive(TOP_K_RESULTS)
    distance_threshold = reactive(DISTANCE_THRESHOLD)

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
        width: 25%;
        min-width: 58;
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
        background: transparent;
    }
    
    ListItem.-highlight > Label {
        color: ansi_bright_cyan;
        text-style: bold;
        background: transparent;
    }

    ListItem.-highlight .book-label {
        color: ansi_bright_cyan;
        text-style: bold;
    }

    ListItem.-highlight .book-delete-btn {
        color: ansi_red;
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
        width: 1fr;
        min-width: 0;
        layout: vertical;
        background: transparent;
        padding-right: 1;
    }

    #chat_scroll {
        height: 1fr;
        padding: 1 2 0 2;
        scrollbar-size: 1 1;
        scrollbar-color: ansi_cyan 35%;
        scrollbar-color-hover: ansi_cyan;
        scrollbar-color-active: ansi_bright_cyan;
    }

    #query_input {
        margin: 1 2 0 2;
        height: 3;
        max-height: 8;
        border: round ansi_bright_black;
        background: ansi_black 5%;
        padding: 0 1;
    }

    #query_input:focus {
        border: round ansi_cyan;
    }

    #query_tuning {
        margin: 0 2 1 2;
        border: none;
        background: transparent;
        height: 1;
        min-height: 1;
        padding: 0;
    }

    #query_tuning .tuning-title {
        color: ansi_bright_black;
        text-style: bold;
        width: 6;
    }

    #query_tuning .tuning-row {
        layout: horizontal;
        align: left middle;
        height: 1;
        margin: 0;
    }

    #query_tuning .tuning-key {
        width: 2;
        color: ansi_bright_white;
        content-align: center middle;
    }

    #query_tuning .tuning-gap {
        width: 2;
        min-width: 2;
    }

    #query_tuning .tuning-value {
        width: 6;
        min-width: 6;
        color: ansi_cyan;
        text-style: bold;
        content-align: center middle;
    }

    #query_tuning .tune-btn {
        width: 3;
        min-width: 3;
        height: 1;
        min-height: 1;
        margin: 0;
        padding: 0;
        border: none;
        background: transparent;
        color: ansi_bright_black;
        content-align: center middle;
        text-style: bold;
    }

    #query_tuning .tune-btn:hover {
        color: ansi_cyan;
        text-style: bold;
        background: transparent;
    }

    #query_tuning .tune-btn:focus {
        border: none;
        background: transparent;
        color: ansi_cyan;
        text-style: bold;
    }

    #query_tuning Button {
        margin: 0;
        border: none;
        background: transparent;
        padding: 0;
    }

    #query_tuning Button:hover {
        background: transparent;
    }

    #query_tuning Button:focus {
        border: none;
        background: transparent;
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
        Binding("ctrl+c", "copy_focused_message", "Copy", show=True),
        Binding("ctrl+d", "delete_selected_book", "Delete Book", show=True),
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

        # Conversation memory
        self._conversation: list[dict[str, str]] = []
        self._last_chunks: list[str] = []

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
                        ("Auto (detect best engine)", "auto"),
                        ("pymupdf4llm — Fast, no GPU (clean PDFs)", "pymupdf4llm"),
                        ("MinerU — Scanned / complex PDFs", "mineru"),
                        ("Marker — High quality (8GB VRAM)", "marker"),
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
                yield TuningPanel(TOP_K_RESULTS, DISTANCE_THRESHOLD)

        yield Footer()

    async def on_mount(self) -> None:
        await self._populate_sidebar()
        self._set_header()
        self.query_one("#query_input", QueryComposer).focus()
        self._resize_query_input()
        
        self._render_mascot()
        self._refresh_models()
        self._sync_tuning_panel()

    def watch_top_k_results(self, value: int) -> None:
        self._sync_tuning_panel()

    def watch_distance_threshold(self, value: float) -> None:
        self._sync_tuning_panel()

    def on_tuning_panel_topk_changed(self, message: TuningPanel.TopkChanged) -> None:
        self._apply_top_k_change(message.value)

    def _apply_top_k_change(self, value: int) -> None:
        self.top_k_results = max(TuningPanel.TOP_K_MIN, min(TuningPanel.TOP_K_MAX, int(value)))

    def on_tuning_panel_distance_changed(self, message: TuningPanel.DistanceChanged) -> None:
        self.distance_threshold = max(
            TuningPanel.DIST_MIN,
            min(TuningPanel.DIST_MAX, round(float(message.value), 2)),
        )

    def _sync_tuning_panel(self) -> None:
        if not self.is_mounted:
            return
        panel = self.query_one("#query_tuning", TuningPanel)
        panel.set_values(self.top_k_results, self.distance_threshold)

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
            row = BookListRow(stem)
            self._book_labels[stem] = row.book_label
            book_list.append(ListItem(row, id=stem))

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
        if composer.styles.height != target:
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
        self.conversation_history.clear()
        self.last_chunks.clear()

    def action_copy_focused_message(self) -> None:
        focused = self.focused
        while focused is not None:
            if isinstance(focused, AssistantMessage):
                success, message = focused.copy_full()
                focused._show_copy_status(success, message)
                self.notify(message, severity="information" if success else "warning")
                return
            focused = getattr(focused, "parent", None)
        self.notify("Focus an assistant response to copy it.", severity="information")

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

        input_widget = self.query_one("#query_input", QueryComposer)
        input_widget.clear()
        self._resize_query_input()

        if query.startswith("/"):
            await self._handle_chat_command(query)
            return

        if not self.active_collection_name:
            self.notify("No active book selected. Please parse a PDF first.", severity="error")
            return

        history_str = self._format_history()
        await self._append_chat_widget(UserMessage(query))

        request_id = uuid4().hex
        assistant = AssistantMessage(thinking=True)
        self._pending_assistants[request_id] = assistant
        await self._append_chat_widget(assistant, animate=False)

        active_book = self.active_collection_name
        model_name = self.selected_model or OLLAMA_MODEL

        worker = self.run_worker(
            lambda: self.execute_rag_query(query, history_str, active_book, model_name, request_id),
            thread=True,
            group="queries",
            exit_on_error=False,
        )
        self._active_query_worker = worker
        self._active_query_request_id = request_id

    def _format_history(self) -> str:
        if not self._conversation:
            return ""
        lines = ["Conversation so far:"]
        for turn in self._conversation[-6:]:  # last 3 exchanges
            role = "User" if turn["role"] == "user" else "Axie"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines) + "\n"

    def _is_followup(self, query: str) -> bool:
        FOLLOWUP_SIGNALS = [
            "this topic", "give me", "an example", "another", 
            "that", "this", "same", "more", "elaborate", "explain further",
            "it", "they", "them", "those"
        ]
        q = query.lower()
        return any(s in q for s in FOLLOWUP_SIGNALS)

    def execute_rag_query(self, query: str, history_str: str, book_stem: str, model_name: str, request_id: str) -> None:
        try:
            if self._is_query_cancelled(request_id):
                self.call_from_thread(self._finalize_query_cancelled, request_id)
                return

            # Augment retrieval query with history if it's vague/short
            retrieval_query = query
            vague_terms = {"it", "this", "that", "they", "example", "question", "more", "explain", "help", "detail"}
            query_words = set(query.lower().split())
            is_vague = bool(query_words & vague_terms) or len(query_words) < 5
            
            if is_vague and self._conversation:
                # Add context from the last user message to improve retrieval relevance
                for msg in reversed(self._conversation):
                    if msg["role"] == "user":
                        # If the user is asking for examples/questions, broaden search significantly
                        if any(term in query.lower() for term in ["example", "question", "exercise", "problem"]):
                            retrieval_query = f"{msg['content']} illustrative examples exercises questions problems"
                        else:
                            prev_context = " ".join(msg["content"].split()[:20])
                            retrieval_query = f"{prev_context} {query}"
                        break

            # Follow-up logic: Reuse chunks if signals detected
            is_followup = self._is_followup(query)
            if is_followup and self._last_chunks:
                context_blocks = self._last_chunks
                metadatas = [] # Metadata info might be lost but context is preserved
            else:
                q_vec = embed_query(retrieval_query)
                collection = get_collection(book_stem)
                results = collection.query(
                    query_embeddings=[q_vec],
                    n_results=8,
                    include=["documents", "metadatas", "distances"],
                )
                
                context_blocks = []
                metadatas = []
                retrieved_docs = results.get("documents") or [[]]
                retrieved_meta = results.get("metadatas") or [[]]
                retrieved_distances = results.get("distances") or [[]]
                has_retrieved_context = bool(retrieved_docs and retrieved_docs[0])
                
                if has_retrieved_context:
                    chunks = retrieved_docs[0]
                    raw_metadatas = retrieved_meta[0] if retrieved_meta else []
                    distances = retrieved_distances[0] if retrieved_distances else []
                    for idx, (chunk, meta) in enumerate(zip(chunks, raw_metadatas)):
                        if idx < len(distances):
                            distance = distances[idx]
                            # Be more lenient for vague queries
                            limit = self.distance_threshold * 1.4 if is_vague else self.distance_threshold
                            if isinstance(distance, (int, float)) and distance >= limit:
                                continue
                        metadatas.append(meta)
                        h1 = meta.get("Header 1", "Unknown Chapter")
                        h2 = meta.get("Header 2", "Unknown Section")
                        page_info = meta.get("page", "N/A")
                        context_blocks.append(f"[{h1} · {h2} · p.{page_info}]\n{chunk}")
                
                # Store for future follow-ups
                self._last_chunks = context_blocks

            # DEBUG PRINTS
            print(f"\n[DEBUG RAG] history_str: {history_str[:100]}...")
            print(f"[DEBUG RAG] is_followup: {is_followup}")
            print(f"[DEBUG RAG] context size: {len(context_blocks)} blocks")
            if context_blocks:
                print(f"[DEBUG RAG] context snippet: {context_blocks[0][:200]}...")
            
            prompt = PROMPT_TEMPLATE.format(
                book=book_stem,
                chapter=",".join(sorted({m.get("Header 1", "Unknown") for m in metadatas}))
                if metadatas
                else "General Knowledge",
                pages=",".join(sorted({str(m.get("page", "?")) for m in metadatas}))
                if metadatas
                else "N/A",
                history=history_str,
                context="\n\n".join(context_blocks)
                if context_blocks
                else "No direct context retrieved from indexed files. Use your general knowledge.",
                question=query,
            )

            accumulated_response = ""
            inference_badge: str | None = None
            was_cancelled = False
            stream_started = time.monotonic()
            stream_metrics: dict[str, object] = {}
            with Client(timeout=240.0) as client:
                with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/generate",
                    json=self._build_generate_payload(model_name, prompt, stream=True),
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if self._is_query_cancelled(request_id):
                            was_cancelled = True
                            break
                        if not line:
                            continue
                        data = json.loads(line)
                        if data.get("done"):
                            stream_metrics = data
                        token = data.get("response", "")
                        if token:
                            accumulated_response += token

            if was_cancelled:
                self.call_from_thread(self._finalize_query_cancelled, request_id)
                return

            inference_badge = self._build_inference_badge(
                stream_metrics,
                time.monotonic() - stream_started,
            )

            # Post-process for math rendering
            accumulated_response = latex_to_unicode(accumulated_response)

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
                    "Select an installed model from the sidebar and retry."
                )
            else:
                detail = f"Ollama request failed ({exc.response.status_code}): {exc.response.reason_phrase}"
            self.call_from_thread(self._finalize_query, request_id, detail, query, True)
        except Exception as exc:
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

    def _build_ollama_options(self) -> dict[str, object]:
        options: dict[str, object] = {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
        }
        if OLLAMA_MIROSTAT in (1, 2):
            options["mirostat"] = OLLAMA_MIROSTAT
            options["mirostat_tau"] = OLLAMA_MIROSTAT_TAU
            options["mirostat_eta"] = OLLAMA_MIROSTAT_ETA
        return options

    def _build_generate_payload(self, model_name: str, prompt: str, stream: bool) -> dict[str, object]:
        return {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": self._build_ollama_options(),
        }

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
            return f"[ ⚡ {tokens_per_second:.1f} tokens/sec | ⏱️ {total_seconds:.1f}s total ]"
        if tokens_per_second is not None:
            return f"[ ⚡ {tokens_per_second:.1f} tokens/sec ]"
        if total_seconds is not None:
            return f"[ ⏱️ {total_seconds:.1f}s total ]"
        return None

    def _run_non_stream_completion(self, model_name: str, prompt: str) -> tuple[str, str | None]:
        started = time.monotonic()
        with Client(timeout=120.0) as client:
            response = client.post(
                f"{OLLAMA_URL}/api/generate",
                json=self._build_generate_payload(model_name, prompt, stream=False),
            )
            response.raise_for_status()
            payload = response.json()
        completion = str(payload.get("response", "")).strip()
        telemetry_badge = self._build_inference_badge(payload, time.monotonic() - started)
        return completion, telemetry_badge

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
        except Exception:
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
            "pymupdf4llm": "pymupdf4llm",
            "marker": "Marker (8GB VRAM)",
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
            from src import checkpoint
            from src.ocr import resolve_mode

            pdf = Path(pdf_path)
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logs_dir = LOGS_DIR
            logs_dir.mkdir(parents=True, exist_ok=True)
            self.call_from_thread(
                self._set_parse_runtime_status,
                stage="Releasing GPU memory",
                detail="Unloading Ollama sessions to free VRAM before OCR.",
                target_progress=12.0,
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

            gpu_note = self._gpu_runtime_note()
            total_pages = self._pdf_page_count(pdf)
            parse_env = self._build_parse_env()
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
                target_progress=20.0,
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
            ocr_progress_floor = 12.0
            ocr_progress_ceiling = 90.0
            last_ocr_progress = 20.0
            page_line_pattern = re.compile(r"page\s+(\d{1,6})", flags=re.IGNORECASE)

            def on_ocr_output(line: str) -> None:
                nonlocal last_ocr_progress
                stripped = line.strip()
                if not stripped:
                    return

                trimmed = stripped if len(stripped) <= 180 else f"{stripped[:177]}..."
                progress = last_ocr_progress

                page_match = page_line_pattern.search(stripped)
                if page_match and total_pages > 0:
                    page_idx = int(page_match.group(1))
                    ratio = min(1.0, max(0.0, (page_idx + 1) / total_pages))
                    progress = ocr_progress_floor + (ocr_progress_ceiling - ocr_progress_floor) * ratio
                    progress = max(last_ocr_progress, progress)
                    last_ocr_progress = progress

                self.call_from_thread(
                    self._set_parse_runtime_status,
                    detail=f"{detail_base} | {trimmed}",
                    target_progress=progress,
                )

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
                target_progress=96.0,
            )
            index_cmd = [
                sys.executable,
                "-u",
                "-m",
                "src.indexer",
                pdf.stem,
                "--reindex",
            ]
            index_log = logs_dir / f"{pdf.stem}_{run_stamp}_index.log"
            self._write_stage_log(index_log, index_cmd, parse_env, "")
            index_code, index_output = self._run_external_job(
                index_cmd,
                env=parse_env,
                live_log_path=index_log,
                timeout_seconds=INDEX_TIMEOUT_SECONDS,
            )
            if index_code != 0:
                raise RuntimeError(
                    f"{self._format_external_failure('Indexing', index_code, index_output)} (log: {index_log})"
                )

            chunks = self._extract_indexed_chunk_count(index_output)
            if chunks <= 0:
                try:
                    chunks = get_collection(pdf.stem).count()
                except Exception:
                    chunks = 0

            self.call_from_thread(
                self._set_parse_runtime_status,
                stage="Finalizing",
                detail=f"Indexed {chunks} chunks. Logs: {index_log.name}",
                target_progress=100.0,
            )
            self.call_from_thread(
                self._finish_parse_and_index,
                pdf.stem,
                page_count,
                chunks,
                "",
            )
        except Exception as exc:
            detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            book_stem = Path(pdf_path).stem if pdf_path else ""
            self.call_from_thread(
                self._finish_parse_and_index,
                book_stem,
                0,
                0,
                f"{exc.__class__.__name__}: {detail}",
            )

    @staticmethod
    def _is_mineru_available() -> bool:
        if importlib.util.find_spec("magic_pdf") is not None:
            return True
        if importlib.util.find_spec("mineru") is not None:
            return True
        return False

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
        lines: list[str] = []
        log_handle = live_log_path.open("a", encoding="utf-8") if live_log_path is not None else None
        line_queue: Queue[str | None] = Queue()
        timed_out = False
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
                except Exception:
                    pass
                line_queue.put(None)

        reader = Thread(target=_reader, daemon=True)
        reader.start()

        try:
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
                    wait_seconds = max(0.05, min(wait_seconds, timeout_at - time.monotonic()))
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
        finally:
            if log_handle is not None:
                log_handle.close()
        output = "\n".join(lines)
        return returncode, output

    @staticmethod
    def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
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
    def _build_parse_env() -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if AxiomLMApp._torch_cuda_available():
            env["AXIOM_EMBED_DEVICE"] = "cuda"
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        else:
            env["AXIOM_EMBED_DEVICE"] = "cpu"
        return env

    @staticmethod
    def _torch_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _pdf_page_count(pdf_path: Path) -> int:
        try:
            import fitz
        except Exception:
            return 0

        doc = fitz.open(str(pdf_path))
        try:
            return len(doc)
        finally:
            doc.close()

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
        except Exception:
            return False

    @staticmethod
    def _gpu_runtime_note() -> str:
        if AxiomLMApp._torch_cuda_available():
            return "GPU acceleration active"
        if AxiomLMApp._nvidia_gpu_present():
            return "NVIDIA GPU detected but Torch is CPU-only (install CUDA torch wheel)"
        return "No CUDA GPU runtime detected"

    @staticmethod
    def _extract_indexed_chunk_count(output: str) -> int:
        match = re.search(r"embedded\s+(\d+)\s+chunks", output, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 0

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
        if event.button.has_class("tune-btn"):
            return

        if event.button.has_class("book-delete-btn"):
            event.stop()
            current = event.button.parent
            while current is not None and not isinstance(current, BookListRow):
                current = current.parent
            if isinstance(current, BookListRow):
                self._prompt_delete_collection(current.book_stem)
            return

        if event.button.id == "refresh_models_btn":
            self._refresh_models()
            return

        if event.button.id == "parse_pdf_btn":
            await self._start_parse_and_index()

    def action_delete_selected_book(self) -> None:
        if not self.active_collection_name:
            self.notify("No active book selected.", severity="warning")
            return
        self._prompt_delete_collection(self.active_collection_name)

    async def _handle_chat_command(self, command_text: str) -> None:
        command, _, arg = command_text.partition(" ")
        command = command.strip().lower()
        argument = arg.strip()

        if command != "/delete":
            self.notify("Unknown command. Supported: /delete [book]", severity="warning")
            return

        target = argument or self.active_collection_name or ""
        if not target:
            self.notify("Usage: /delete [book]", severity="warning")
            return

        resolved, error = self._resolve_book_stem(target)
        if not resolved:
            self.notify(error, severity="warning")
            return
        self._prompt_delete_collection(resolved)

    def _resolve_book_stem(self, raw: str) -> tuple[str | None, str]:
        needle = raw.strip().lower()
        if not needle:
            return None, "Usage: /delete [book]"

        stems = sorted(self._book_labels.keys())
        if not stems:
            return None, "No books indexed."

        exact_matches = [
            stem
            for stem in stems
            if stem.lower() == needle or self._format_book_name(stem).lower() == needle
        ]
        if exact_matches:
            return exact_matches[0], ""

        partial_matches = [
            stem
            for stem in stems
            if needle in stem.lower() or needle in self._format_book_name(stem).lower()
        ]
        if len(partial_matches) == 1:
            return partial_matches[0], ""
        if len(partial_matches) > 1:
            choices = ", ".join(self._format_book_name(stem) for stem in partial_matches[:4])
            return None, f"Ambiguous book name. Matches: {choices}"
        return None, f"Book not found: {raw}"

    def _prompt_delete_collection(self, book_stem: str) -> None:
        if self._active_query_worker is not None and self._active_query_worker.is_running:
            self.notify("Cancel the running query before deleting a book.", severity="warning")
            return
        if self._is_parsing:
            self.notify("Wait for parse/index to finish before deleting a book.", severity="warning")
            return
        if book_stem not in self._book_labels:
            self.notify("Book is no longer available.", severity="warning")
            return

        book_name = self._format_book_name(book_stem)
        chunk_count = self._book_counts.get(book_stem, 0)
        estimated_size_mb = self._get_collection_size_mb(book_stem)
        dialog = DeleteCollectionConfirm(book_name, chunk_count, estimated_size_mb)
        self.push_screen(
            dialog,
            callback=lambda confirmed, stem=book_stem: self._handle_delete_confirm_result(stem, bool(confirmed)),
        )

    def _get_collection_size_mb(self, book_stem: str) -> float:
        chunk_count = self._book_counts.get(book_stem)
        if chunk_count is None:
            try:
                chunk_count = get_collection(book_stem).count()
            except Exception:
                chunk_count = 0
        # Rough estimate: ~2KB per chunk including vector + metadata overhead.
        return (chunk_count * 2.0) / 1024.0

    def _handle_delete_confirm_result(self, book_stem: str, confirmed: bool) -> None:
        if not confirmed:
            return
        self.run_worker(
            lambda: self._delete_collection_worker(book_stem),
            thread=True,
            group="delete-collection",
            exclusive=True,
            exit_on_error=False,
        )

    def _delete_collection_worker(self, book_stem: str) -> None:
        try:
            deleted = delete_collection(book_stem)
            self.call_from_thread(self._finalize_collection_delete, book_stem, deleted, "")
        except Exception as exc:
            detail = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            self.call_from_thread(self._finalize_collection_delete, book_stem, False, detail)

    async def _finalize_collection_delete(self, book_stem: str, deleted: bool, error: str) -> None:
        if error:
            self.notify(f"Delete failed: {error}", severity="error")
            return
        if not deleted:
            self.notify(f"Collection '{self._format_book_name(book_stem)}' was not found.", severity="warning")
            await self._populate_sidebar()
            self._set_header()
            return

        deleted_name = self._format_book_name(book_stem)
        await self._populate_sidebar()
        self._set_header()
        self.query_one("#query_input", QueryComposer).focus()
        self.notify(f"Deleted '{deleted_name}'.", severity="information")

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

    async def _finalize_query(
        self,
        request_id: str,
        message: str,
        query: str,
        is_error: bool,
        telemetry_badge: str | None = None,
    ) -> None:
        if not is_error:
            self._conversation.append({"role": "user", "content": query})
            self._conversation.append({"role": "assistant", "content": message})
            
            # Keep only the last 10 messages (approx 5 turns) for context
            if len(self._conversation) > 10:
                self._conversation = self._conversation[-10:]

        await self._finalize_assistant_message(
            request_id,
            message,
            is_error=is_error,
            telemetry_badge=telemetry_badge,
        )
        self._clear_query_state(request_id)

    async def _finalize_assistant_message(
        self,
        request_id: str,
        message: str,
        is_error: bool,
        telemetry_badge: str | None = None,
    ) -> None:
        assistant = self._pending_assistants.pop(request_id, None)
        if assistant is None or not assistant.is_attached:
            await self._append_chat_widget(
                AssistantMessage(message, is_error=is_error, telemetry_badge=telemetry_badge)
            )
            return
        assistant.finalize(message, is_error=is_error, telemetry_badge=telemetry_badge)
        self.query_one("#chat_scroll", ScrollableContainer).scroll_end(animate=False)

    def _render_mascot(self) -> None:
        try:
            mascot_widget = self.query_one("#mascot_art", Static)
        except Exception:
            return

        logo = (
            " _______         __                    _____   _______ \n"
            "|   _   |.--.--.|__|.-----.--------.  |     |_|   |   |\n"
            "|       ||_   _||  ||  _  |        |  |       |       |\n" 
            "|___|___||__.__||__||_____|__|__|__|  |_______|__|_|__|\n"
        )
        mascot_widget.update(Text(logo, style="bold ansi_cyan"))
                                                         
def main():
    app = AxiomLMApp()
    app.run()

if __name__ == "__main__":
    main()
