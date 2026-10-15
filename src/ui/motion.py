"""One bounded clock for glyph feedback, counters and transient confirmations."""

import math
import os
import time

from rich.text import Text
from textual.message import Message
from textual.screen import Screen

from src import config

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def render_progress(done: int, total: int, width: int) -> str:
    width = max(0, width)
    eighths = int(min(1, max(0, done / total)) * width * 8) if total > 0 else 0
    full, edge = divmod(eighths, 8)
    return (
        "━" * full
        + ("▏▎▍▌▋▊▉"[edge - 1] if edge else "")
        + "─" * max(0, width - full - bool(edge))
    )


class MotionClock:
    def __init__(self, app):
        self.app = app
        self.origin = time.monotonic()
        self.reduced = (
            os.environ.get("AXIOM_REDUCE_MOTION") == "1" or "NO_MOTION" in os.environ
        )
        self.spinner = "…" if self.reduced else SPINNER[0]
        self.last_frame = float("-inf")
        self.timer = None
        self.running = False

    def start(self):
        if self.timer is None:
            self.timer = self.app.set_interval(
                config.UI_FRAME_INTERVAL,
                lambda: self.tick(time.monotonic()),
                pause=True,
            )
        self.running = True
        self.timer.resume()

    def stop(self):
        self.running = False
        if self.timer:
            self.timer.pause()

    def tick(self, now):
        if not self.app._motion_active() or not self.app._terminal_focused:
            self.stop()
            return
        if now - self.last_frame + 1e-9 < config.UI_FRAME_INTERVAL:
            return
        self.last_frame = now
        self.spinner = (
            "…"
            if self.reduced
            else SPINNER[
                math.floor((now - self.origin + 1e-9) / config.UI_SPINNER_INTERVAL)
                % len(SPINNER)
            ]
        )
        self.app._motion_frame(now, self)

    @staticmethod
    def elapsed(started, now):
        elapsed = max(0, now - started)
        if elapsed > 10:
            return f"{int(elapsed)//60}:{int(elapsed)%60:02}"
        return f"{math.floor((elapsed + 1e-9)/config.UI_COUNTER_INTERVAL)*config.UI_COUNTER_INTERVAL:.1f}s"

    def label(self, label, now):
        text = Text(label, style="dim")
        if self.reduced:
            return text
        cycle = (len(label) + 3) * config.UI_SHIMMER_INTERVAL + config.UI_SHIMMER_PAUSE
        position = int(((now - self.origin) % cycle) / config.UI_SHIMMER_INTERVAL)
        for index in range(max(0, position - 2), min(len(label), position + 1)):
            text.stylize(
                "bold" if index == position - 1 else "not dim", index, index + 1
            )
        return text

    def blink(self, now):
        return (
            self.reduced or int((now - self.origin) / config.UI_BLINK_INTERVAL) % 2 == 0
        )


class StudyScreen(Screen):
    """Keep native layout/selection, with the application's paint budget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._paint_timer = None
        self._last_paint = float("-inf")

    class SelectionChanged(Message):
        pass

    async def _watch_selections(self, old_selections, selections):
        await super()._watch_selections(old_selections, selections)
        self.post_message(self.SelectionChanged())

    @property
    def _update_timer(self):
        if self._paint_timer is None:
            self._paint_timer = self.set_interval(
                config.UI_FRAME_INTERVAL,
                self._on_timer_update,
                name="screen_update",
                pause=True,
            )
        return self._paint_timer

    def _compositor_refresh(self):
        now = time.monotonic()
        if now - self._last_paint < config.UI_FRAME_INTERVAL:
            self.refresh()
            return
        super()._compositor_refresh()
        self._last_paint = time.monotonic()
