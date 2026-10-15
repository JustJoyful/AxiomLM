from dataclasses import asdict

import pytest
from textual.app import App
from textual.widgets import Markdown

from src.ui.settings import UISettings
from src.ui.themes import THEMES, contrast_ratio, load_custom_themes, resolve_theme


@pytest.mark.parametrize("spec", THEMES.values())
def test_builtin_contrast(spec):
    for slot, minimum in [("text", 7), ("accent", 4.5), ("muted", 3)]:
        assert contrast_ratio(getattr(spec, slot), spec.background) >= minimum


def test_custom_theme_validation_and_contrast_warning(tmp_path):
    (tmp_path / "bad.toml").write_text('name = "paper"\n')
    themes, errors = load_custom_themes(tmp_path)
    assert not themes
    assert errors == ['✗ theme "paper" invalid: duplicate name · using terminal']
    (tmp_path / "bad.toml").unlink()
    spec = asdict(THEMES["paper"])
    spec["muted"] = "#fafafa"
    (tmp_path / "custom.toml").write_text(
        'name = "custom"\n'
        + "\n".join(
            f"{k} = {str(v).lower() if isinstance(v, bool) else repr(v)}"
            for k, v in spec.items()
        )
    )
    themes, errors = load_custom_themes(tmp_path)
    assert "custom" in themes
    assert len(errors) == 1 and "contrast" in errors[0]


async def test_palette_switch_keeps_native_markdown():
    class Probe(App):
        def compose(self):
            yield Markdown("```python\nx = 'λ'", id="answer")

    app = Probe()
    async with app.run_test() as pilot:
        answer = app.query_one(Markdown)
        await resolve_theme(app, UISettings("paper"), {})
        await pilot.pause()
        assert app.theme == "paper"
        await resolve_theme(app, UISettings("terminal"), {"COLORFGBG": "0;15"})
        await pilot.pause()
        assert app.theme == "ansi-light"
        assert answer.source == "```python\nx = 'λ'"
        await resolve_theme(app, UISettings("nord"), {"NO_COLOR": "1"})
        assert app.no_color
        await resolve_theme(app, UISettings("missing"), {})
        assert app.theme == "ansi-dark"


async def test_preview_revert_and_failed_save_preserve_source_and_preferences(
    tmp_path, monkeypatch
):
    from src.ui import themes
    from src.ui.themes import ThemePreview

    class Probe(App):
        def compose(self):
            yield Markdown("```python\nx = 'λ'", id="answer")

    settings = UISettings("terminal")
    path = tmp_path / "config.toml"
    path.write_text('theme = "terminal"\n# untouched\n')
    app = Probe()
    async with app.run_test() as pilot:
        await resolve_theme(app, settings, {})
        transaction = ThemePreview(app, settings, path, {})
        await transaction.preview("paper")
        await pilot.pause()
        assert app.theme == "paper"
        await transaction.revert()
        assert app.theme == "ansi-dark"
        await transaction.preview("nord")
        monkeypatch.setattr(
            themes,
            "save_theme_selection",
            lambda *_: (_ for _ in ()).throw(OSError("read only")),
        )
        with pytest.raises(OSError):
            await transaction.keep("nord")
        assert settings.theme == "terminal"
        assert app.theme == "ansi-dark"
        assert path.read_text() == 'theme = "terminal"\n# untouched\n'
        assert app.query_one(Markdown).source == "```python\nx = 'λ'"


async def test_mono_strips_all_hue_and_terminal_keeps_ansi_indexes():
    from rich.segment import Segment
    from rich.style import Style
    from textual.color import Color

    app = App()
    await resolve_theme(app, UISettings("mono"), {})
    segments = [Segment("alert", Style(color="red", bgcolor="blue", bold=True))]
    for line_filter in app.get_line_filters():
        segments = line_filter.apply(segments, Color.parse("ansi_default"))
    assert segments[0].style.bold
    assert segments[0].style.color is None or segments[0].style.color.is_default
    await resolve_theme(app, UISettings("terminal", "bright-magenta"), {})
    assert app.native_ansi_color
    assert app.current_theme.accent == "ansi_bright_magenta"
