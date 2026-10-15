"""The five RGB palettes and terminal palette resolution."""

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from textual.app import App
from textual.filter import Monochrome, NoColor
from textual.theme import Theme

from src.ui.settings import save_theme_selection


@dataclass(frozen=True)
class ThemeSpec:
    background: str
    text: str
    muted: str
    accent: str
    alert: str
    dark: bool


THEMES = {
    "catppuccin-mocha": ThemeSpec(
        "#1e1e2e", "#cdd6f4", "#6c7086", "#cba6f7", "#f38ba8", True
    ),
    "gruvbox-dark": ThemeSpec(
        "#282828", "#ebdbb2", "#928374", "#fabd2f", "#fb4934", True
    ),
    "nord": ThemeSpec("#2e3440", "#d8dee9", "#7b88a1", "#88c0d0", "#bf616a", True),
    "tokyo-night": ThemeSpec(
        "#1a1b26", "#c0caf5", "#737aa2", "#7aa2f7", "#f7768e", True
    ),
    "paper": ThemeSpec("#fafafa", "#1f2328", "#6e7781", "#0969da", "#cf222e", False),
}
ANSI_COLORS = {
    f"{prefix}{name}"
    for prefix in ("", "bright-")
    for name in ("red", "green", "yellow", "blue", "magenta", "cyan")
}


def contrast_ratio(foreground: str, background: str) -> float:
    def luminance(color):
        channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in channels
        ]
        return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    low, high = sorted((luminance(foreground), luminance(background)))
    return (high + 0.05) / (low + 0.05)


def load_custom_themes(directory: Path) -> tuple[dict[str, ThemeSpec], list[str]]:
    themes, errors = {}, []
    for path in sorted(directory.glob("*.toml")):
        name = path.stem
        try:
            data = tomllib.loads(path.read_text())
            name = data.pop("name", name)
            if (
                not isinstance(name, str)
                or not name
                or name in THEMES
                or name in {"terminal", "mono"}
                or name in themes
            ):
                raise ValueError("duplicate name")
            if set(data) != set(ThemeSpec.__dataclass_fields__):
                raise ValueError("expected five colors and dark flag")
            if not isinstance(data["dark"], bool):
                raise TypeError("dark must be boolean")
            for key in ("background", "text", "muted", "accent", "alert"):
                value = data[key]
                if not isinstance(value, str) or not (
                    re.fullmatch(r"#[0-9a-fA-F]{6}", value)
                    or value.startswith("ansi:")
                    and value[5:]
                    in ANSI_COLORS | {"black", "white", "bright-black", "bright-white"}
                    or key == "background"
                    and value == "terminal"
                ):
                    raise ValueError(f"bad {key}")
            spec = ThemeSpec(**data)
            themes[name] = spec
            for slot, minimum in [("text", 7), ("accent", 4.5), ("muted", 3)]:
                if (
                    getattr(spec, slot).startswith("#")
                    and spec.background.startswith("#")
                    and contrast_ratio(getattr(spec, slot), spec.background) < minimum
                ):
                    errors.append(
                        f'○ theme "{name}" contrast: {slot} below {minimum}:1'
                    )
                    break
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f'✗ theme "{name}" invalid: {exc} · using terminal')
    return themes, errors


async def resolve_theme(app: App, settings, environ: Mapping[str, str]) -> str:
    palettes = THEMES | getattr(app, "custom_themes", {})
    for name, spec in palettes.items():

        def color(value):
            return (
                "ansi_default"
                if value == "terminal"
                else value.replace("ansi:", "ansi_").replace("bright-", "bright_")
            )

        app.register_theme(
            Theme(
                name=name,
                dark=spec.dark,
                primary=color(spec.accent),
                accent=color(spec.accent),
                foreground=color(spec.text),
                background=color(spec.background),
                surface=color(spec.background),
                panel=color(spec.background),
                warning=color(spec.alert),
                error=color(spec.alert),
                variables={"text-muted": color(spec.muted)},
            )
        )
    selected = settings.theme
    mono = selected == "mono" or "NO_COLOR" in environ
    app.no_color = mono
    for line_filter in app._filters:
        if isinstance(line_filter, (NoColor, Monochrome)):
            line_filter.enabled = mono
    if mono and not any(isinstance(f, NoColor) for f in app._filters):
        app._filters.append(NoColor())
    if selected not in palettes or mono:
        dark = True
        background = environ.get("COLORFGBG", "").split(";")[-1]
        if background.isdigit():
            dark = int(background) < 7
        driver = getattr(app, "_driver", None)
        if driver and hasattr(driver, "detect_background"):
            detected = await driver.detect_background()
            if detected is not None:
                dark = detected
        selected = "ansi-dark" if dark else "ansi-light"
        if selected not in app.available_themes:
            selected = "textual-ansi"
        base = app.available_themes[selected]
        accent = settings.accent if settings.accent in ANSI_COLORS else "cyan"
        value = "ansi_default" if mono else "ansi_" + accent.replace("-", "_")
        from dataclasses import replace

        app.register_theme(
            replace(
                base,
                variables=dict(
                    base.variables or {},
                    **{"text-muted": "ansi_default" if mono else "ansi_bright_black"},
                ),
                primary=value,
                accent=value,
                warning="ansi_default" if mono else "ansi_yellow",
                error="ansi_default" if mono else "ansi_yellow",
            )
        )
    app.theme = selected
    # Custom themes containing ANSI slots also need palette indexes preserved.
    app.ansi_color = (
        selected in {"ansi-dark", "ansi-light", "textual-ansi"}
        or selected in palettes
        and any(
            v.startswith("ansi:") or v == "terminal"
            for v in vars(palettes[selected]).values()
            if isinstance(v, str)
        )
    )
    return selected


class ThemePreview:
    """Preview without writes; failed saves restore the previous palette."""

    def __init__(self, app, settings, path, environ):
        from dataclasses import replace

        self.app, self.settings, self.path, self.environ = app, settings, path, environ
        self.previous = replace(settings)

    async def preview(self, name):
        from dataclasses import replace

        await resolve_theme(self.app, replace(self.settings, theme=name), self.environ)

    async def revert(self):
        await resolve_theme(self.app, self.previous, self.environ)

    async def keep(self, name):
        try:
            save_theme_selection(self.path, name, self.settings.accent)
        except (OSError, ValueError):
            await self.revert()
            raise
        self.settings.theme = name
