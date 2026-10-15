import os

import pytest

from src.ui.settings import UISettings, load_settings, save_theme_selection


def test_environment_theme_does_not_persist(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('theme = "terminal"\n')
    settings, errors = load_settings(path, {"AXIOM_THEME": "paper"})
    assert settings.theme == "paper"
    assert not errors
    assert path.read_text() == 'theme = "terminal"\n'


def test_first_run_and_no_color(tmp_path):
    assert load_settings(tmp_path / "missing", {}) == (UISettings(), [])
    settings, _ = load_settings(
        tmp_path / "missing", {"NO_COLOR": "", "AXIOM_THEME": "paper"}
    )
    assert settings.theme == "mono"


def test_atomic_update_preserves_comments_and_other_tables(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('# my settings\ntheme = "terminal" # keep\n[other]\nvalue = 9\n')
    save_theme_selection(path, "paper", "cyan")
    assert "# my settings" in path.read_text()
    assert "# keep" in path.read_text()
    assert "[other]\nvalue = 9" in path.read_text()
    previous = path.read_bytes()
    monkeypatch.setattr(
        os, "replace", lambda *_: (_ for _ in ()).throw(OSError("read only"))
    )
    with pytest.raises(OSError):
        save_theme_selection(path, "nord", "red")
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_toml_returns_readable_error(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("theme = [")
    settings, errors = load_settings(path, {})
    assert settings == UISettings()
    assert errors and errors[0].startswith("✗")
