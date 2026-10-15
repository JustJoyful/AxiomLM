"""Small, atomic updates to user-owned TOML preferences."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import tomlkit


@dataclass
class UISettings:
    theme: str = "terminal"
    accent: str = "cyan"


def load_settings(
    path: Path, environ: Mapping[str, str]
) -> tuple[UISettings, list[str]]:
    settings = UISettings()
    errors = []
    try:
        data = tomlkit.parse(path.read_text()) if path.exists() else {}
        for name in ("theme", "accent"):
            value = data.get(name, getattr(settings, name))
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            setattr(settings, name, value)
    except (OSError, ValueError, TypeError, tomlkit.exceptions.ParseError) as exc:
        settings = UISettings()
        errors.append(f"✗ settings invalid: {exc} · using terminal")
    settings.theme = environ.get("AXIOM_THEME", settings.theme)
    if "NO_COLOR" in environ:
        settings.theme = "mono"
    return settings, errors


def save_theme_selection(path: Path, theme: str, accent: str) -> None:
    data = tomlkit.parse(path.read_text()) if path.exists() else tomlkit.document()
    data["theme"] = theme
    data["accent"] = accent
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(tomlkit.dumps(data))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
