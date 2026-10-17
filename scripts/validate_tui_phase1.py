"""Offline Phase 1 UI measurements; physical terminal evidence stays explicit."""

import asyncio
import gc
import json
import os
import time
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any
from unittest.mock import patch

from textual.widgets import Static

from src import config, tui
from src.ui.stream import StudyStream


async def validate() -> dict[str, Any]:
    # Prior headless Apps leave large cycles; measure a fresh App, as the CLI does.
    gc.collect()
    discovery = Event()
    entered = asyncio.Event()
    allow_cleanup = Event()
    metrics: dict[str, Any] = {}
    paints: list[float] = []
    submitted = 0.0
    backend_entered = 0.0
    mode = ""
    app = None

    class Client:
        async def astream_generate(self, prompt, **_kwargs):
            nonlocal mode, backend_entered
            mode = "cancel" if "cancel probe" in prompt else "performance"
            backend_entered = time.monotonic()
            assert any(
                "performance probe" in line.text or "cancel probe" in line.text
                for line in app.screen._compositor.render_strips()
            )
            entered.set()
            try:
                if mode == "performance":
                    for _ in range(2000):
                        yield {"response": "token "}
                        await asyncio.sleep(0.0005)
                    yield {"done": True}
                else:
                    yield {"response": "Partial \\sum x"}
                    await asyncio.Event().wait()
            finally:
                if mode == "cancel":
                    metrics["client_closed_at"] = time.monotonic()

        def unload_and_wait(self):
            if mode == "cancel" and not allow_cleanup.wait(3):
                raise TimeoutError("Diagnostic cleanup gate was not released")

    class Probe(tui.AxiomLMApp):
        def _discover_books(self):
            discovery.wait(3)
            return {"probe_book": 1}

        def _fetch_ollama_models(self):
            discovery.wait(3)
            return [config.OLLAMA_MODEL]

        def _prepare_query(self, query, *_args):
            return query, [], [], []

        def _display(self, screen, renderable):
            if renderable is not None and self._active_query_request_id:
                now = time.monotonic()
                paints.append(now)
                if "echo_ms" not in metrics and any(
                    "performance probe" in line.text
                    for line in screen._compositor.render_strips()
                ):
                    metrics["echo_ms"] = (now - submitted) * 1000
                    metrics["working_ms"] = (
                        metrics["echo_ms"] if "working" in self._status else None
                    )
            return super()._display(screen, renderable)

    async def wait_idle():
        deadline = time.monotonic() + 5
        while app._active_query_request_id is not None:
            if time.monotonic() > deadline:
                raise TimeoutError("Diagnostic query did not complete")
            await asyncio.sleep(0.01)

    environment = dict(os.environ)
    environment.pop("NO_COLOR", None)
    environment.update(AXIOM_THEME="terminal", COLORFGBG="0;0", COLORTERM="truecolor")
    with (
        TemporaryDirectory(prefix="axiom-phase1-") as temporary,
        patch.dict(os.environ, environment, clear=True),
        patch.object(config, "UI_CONFIG_PATH", Path(temporary) / "config.toml"),
        patch.object(config, "UI_THEMES_DIR", Path(temporary) / "themes"),
        patch.object(tui, "OllamaClient", Client),
    ):
        started = time.monotonic()
        app = Probe()
        try:
            async with app.run_test(size=(100, 30)) as pilot:
                metrics["first_paint_ms"] = (time.monotonic() - started) * 1000
                metrics["initial_theme"] = app.theme
                metrics["discovery_held_at_first_paint"] = not discovery.is_set()
                discovery.set()
                await app.workers.wait_for_complete()
                prompt = app.query_one(tui.QueryComposer)
                prompt.load_text("performance probe")
                submitted = time.monotonic()
                await pilot.press("enter")
                await entered.wait()
                assert metrics["echo_ms"] / 1000 + submitted <= backend_entered
                await pilot.press("d", "r", "a", "f", "t")
                metrics["prompt_keystrokes_preserved"] = prompt.text == "draft"
                await wait_idle()
                assert app.turns[-1].record.raw_answer == "token " * 2000
                metrics["streamed_tokens"] = 2000
                intervals = [b - a for a, b in pairwise(paints)]
                metrics["maximum_paint_hz"] = 1 / min(intervals) if intervals else None
                metrics["paint_count"] = len(paints)
                metrics["answer_widget_count"] = len(app.query("StudyMarkdown"))
                entered.clear()
                prompt.load_text("cancel probe")
                await pilot.press("enter")
                await entered.wait()
                await asyncio.sleep(0.1)
                cancelled = time.monotonic()
                await app._cancel_active_query()
                metrics["cancel_ui_ms"] = (time.monotonic() - cancelled) * 1000
                metrics["client_close_ms"] = (
                    metrics.pop("client_closed_at") - cancelled
                ) * 1000
                metrics["admission_held_until_cleanup"] = (
                    app._workload_coordinator.active_kind == "generation"
                )
                assert metrics["admission_held_until_cleanup"]
                record = app.turns[-1].record
                metrics["partial_raw_preserved"] = (
                    record.raw_answer == "Partial \\sum x"
                )
                allow_cleanup.set()
                await wait_idle()
                await pilot.pause(0.08)
                metrics["shared_clock_idle"] = not app.motion.running
                pumps = [app, app.screen, *app.screen.walk_children()]
                metrics["idle_active_timers"] = sum(
                    timer._active.is_set()
                    and timer._task is not None
                    and not timer._task.done()
                    for pump in pumps
                    for timer in pump._timers
                )
                stream = app.query_one(StudyStream)
                await stream.mount(
                    *(
                        Static(str(n), classes="notice", markup=False)
                        for n in range(2001)
                    ),
                    before=stream.spacer,
                )
                await stream.trim_history()
                metrics["retained_blocks"] = len(
                    stream.query_children(".turn, .notice")
                )
                metrics["retained_widgets"] = len(list(app.screen.walk_children()))
                assert metrics["retained_blocks"] == config.UI_SCROLLBACK_BLOCKS
        finally:
            discovery.set()
            allow_cleanup.set()
    metrics["backend"] = "deterministic fake; no network, embeddings, CUDA or indexes"
    metrics["measurement_scope"] = (
        "App construction onward; Python imports and collection of prior closed test Apps excluded"
    )
    metrics["physical_terminal_checks"] = "not measured by headless diagnostic"
    metrics["missing_manual_evidence"] = [
        "human physical keypress confirmation in Kitty and Alacritty",
        "native mouse selection with drag auto-scroll in both terminals",
        "human NO_COLOR/reduced-motion readability confirmation in Alacritty",
    ]
    return metrics


if __name__ == "__main__":
    report = asyncio.run(validate())
    config.UI_ACCEPTANCE_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
