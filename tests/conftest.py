"""Suite-wide defaults: TUI turns use the Phase 2 path unless a test opts in."""

import pytest


@pytest.fixture(autouse=True)
def _pin_phase2_tui_path(monkeypatch):
    # Agent-routing tests set AGENT_ROUTING_ENABLED explicitly; nothing else may
    # depend on a live Ollama chat counter or the host sandbox by accident.
    from src import config

    monkeypatch.setattr(config, "AGENT_ROUTING_ENABLED", False)
