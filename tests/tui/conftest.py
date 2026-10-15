"""Offline app construction: never discover user books, HTTP services or GPUs."""

import pytest


@pytest.fixture
def app_factory(monkeypatch, tmp_path):
    from src import config, tui

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("AXIOM_THEME", raising=False)
    monkeypatch.setenv("COLORTERM", "truecolor")

    monkeypatch.setattr(config, "UI_CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(config, "UI_THEMES_DIR", tmp_path / "themes")
    monkeypatch.setattr(tui, "list_collections", list)
    monkeypatch.setattr(
        tui.AxiomLMApp, "_fetch_ollama_models", lambda self: [tui.OLLAMA_MODEL]
    )
    return tui.AxiomLMApp
