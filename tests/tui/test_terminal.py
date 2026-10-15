import asyncio

from textual.events import Key

from src.ui.terminal import PaletteParser, detect_background


def test_split_osc11_never_becomes_typed_prompt_text():
    parser = PaletteParser(False)
    emitted = []
    for part in ["a", "\x1b]11;rgb:ffff/", "ffff/ffff\x1b", "\\b"]:
        emitted.extend(parser.feed(part))
    assert [event.character for event in emitted if isinstance(event, Key)] == [
        "a",
        "b",
    ]
    assert parser.background_dark is False


async def test_detection_timeout_and_response_are_bounded():
    class Driver:
        background_dark = None

        def write(self, text):
            assert text == "\x1b]11;?\x1b\\"

    start = asyncio.get_running_loop().time()
    assert await detect_background(Driver()) is None
    assert asyncio.get_running_loop().time() - start < 0.15
    driver = Driver()
    asyncio.get_running_loop().call_later(
        0.01, setattr, driver, "background_dark", True
    )
    assert await detect_background(driver) is True


def test_escape_and_pasted_osc_text_are_preserved(monkeypatch):
    from src.ui import terminal

    now = [1.0]
    monkeypatch.setattr(terminal.time, "monotonic", lambda: now[0])
    parser = PaletteParser(False)
    assert list(parser.feed("\x1b")) == []
    now[0] += 0.2
    assert [event.key for event in parser.tick()] == ["escape"]
    parser = PaletteParser(False)
    events = list(parser.feed("\x1b[200~\x1b]11;rgb:ffff/ffff/ffff\x07\x1b[201~x"))
    from textual.events import Paste

    assert [event.text for event in events if isinstance(event, Paste)] == [
        "\x1b]11;rgb:ffff/ffff/ffff\x07"
    ]
    assert parser.background_dark is None
