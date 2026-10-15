"""OSC 11 replies through Textual's existing Linux input thread."""

import asyncio
import os
import re
import selectors
import time
from codecs import getincrementaldecoder

from textual._xterm_parser import XTermParser
from textual.drivers.linux_driver import LinuxDriver

from src.config import UI_BACKGROUND_TIMEOUT


class PaletteParser(XTermParser):
    def __init__(self, debug=False):
        super().__init__(debug)
        self.pending = ""
        self.background_dark = None

    def feed(self, data):
        from textual import constants

        prefixes = ("\x1b]11;", "\x1b[200~", "\x1b[201~")
        for character in data:
            if getattr(self, "pasting", False):
                self.paste_buffer += character
                if self.paste_buffer.endswith("\x1b[201~"):
                    from textual.events import Paste

                    yield Paste(self.paste_buffer[:-6].replace("\x00", ""))
                    self.paste_buffer = ""
                    self.pasting = False
                continue
            self.pending += character
            self.deadline = time.monotonic() + constants.ESCAPE_DELAY
            if self.pending in prefixes[1:]:
                self.pasting = self.pending == prefixes[1]
                self.paste_buffer = ""
                self.pending = ""
            elif not getattr(self, "pasting", False) and self.pending.startswith(
                prefixes[0]
            ):
                if self.pending.endswith(("\x07", "\x1b\\")):
                    match = re.fullmatch(
                        r"\x1b\]11;rgb:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})(?:\x07|\x1b\\)",
                        self.pending,
                    )
                    if match:
                        r, g, b = (
                            int(v, 16) / (16 ** len(v) - 1) for v in match.groups()
                        )
                        self.background_dark = (
                            0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5
                        )
                    self.pending = ""
                elif len(self.pending) > 128:
                    self.pending = ""
            elif not any(prefix.startswith(self.pending) for prefix in prefixes):
                yield from super().feed(self.pending)
                self.pending = ""

    def tick(self):
        from textual.events import Key

        if self.pending and time.monotonic() >= self.deadline:
            if self.pending == "\x1b":
                yield Key("escape", "\x1b")
            elif not self.pending.startswith("\x1b]11;"):
                yield from super().feed(self.pending)
            self.pending = ""
        yield from super().tick()


async def detect_background(driver):
    driver.write("\x1b]11;?\x1b\\")
    deadline = asyncio.get_running_loop().time() + UI_BACKGROUND_TIMEOUT
    while driver.background_dark is None:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        await asyncio.sleep(min(0.005, remaining))
    return driver.background_dark


class PaletteDriver(LinuxDriver):
    background_dark = None

    async def detect_background(self):
        return await detect_background(self)

    def run_input_thread(self):
        # Keep one reader; extend only the pinned driver's parser boundary.
        parser = PaletteParser(self._debug)
        decode = getincrementaldecoder("utf-8")().decode
        with selectors.SelectSelector() as selector:
            selector.register(self.fileno, selectors.EVENT_READ)
            while not self.exit_event.is_set():
                for _, _mask in selector.select(0.05):
                    data = os.read(self.fileno, 4096)
                    if not data:
                        return
                    for event in parser.feed(decode(data)):
                        self.process_message(event)
                    self.background_dark = parser.background_dark
                for event in parser.tick():
                    self.process_message(event)
