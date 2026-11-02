import asyncio

import pytest

from src import tui


@pytest.mark.asyncio
async def test_cancel_feature_releases_only_after_confirmed_unload(
    app_factory, monkeypatch
):
    app = app_factory()
    started = asyncio.Event()

    class Client:
        def unload_and_wait(self):
            assert app._workload_coordinator.active_kind == "generation"

    monkeypatch.setattr(tui, "OllamaClient", Client)

    async def operation(client):
        started.set()
        await asyncio.Event().wait()

    async with app.run_test():
        token = app._try_acquire_workload("generation")
        task = asyncio.create_task(app._run_feature(operation, token))
        await started.wait()
        assert app._busy()
        task.cancel()
        await task
        assert not app._busy()


@pytest.mark.asyncio
async def test_cleanup_failure_preserves_admission(app_factory, monkeypatch):
    app = app_factory()

    class Client:
        def unload_and_wait(self):
            raise RuntimeError("still loaded")

    monkeypatch.setattr(tui, "OllamaClient", Client)

    async def operation(client):
        pass

    async with app.run_test():
        token = app._try_acquire_workload("generation")
        await app._run_feature(operation, token)
        assert app._busy()
        assert any("cleanup unconfirmed" in line for line in app.notices)


@pytest.mark.asyncio
async def test_cancel_before_worker_starts_does_not_leak_admission(app_factory):
    app = app_factory()
    ran = []

    async def operation(client):
        ran.append(True)

    async with app.run_test() as pilot:
        app._start_feature(operation)
        app.action_cancel_query()
        await pilot.pause()
        assert ran == []
        assert not app._busy()


@pytest.mark.asyncio
async def test_repeated_cancel_during_unload_keeps_cleanup_owner(
    app_factory, monkeypatch
):
    import threading

    app = app_factory()
    entered, release = threading.Event(), threading.Event()

    class Client:
        def unload_and_wait(self):
            entered.set()
            assert release.wait(3)

    monkeypatch.setattr(tui, "OllamaClient", Client)

    async def operation(client):
        await asyncio.Event().wait()

    async with app.run_test() as pilot:
        app._start_feature(operation)
        await pilot.pause()
        app.action_cancel_query()
        await asyncio.to_thread(entered.wait, 2)
        assert entered.is_set() and app._busy()
        app.action_cancel_query()
        release.set()
        await pilot.pause()
        assert not app._busy()
        assert app._feature_worker is None


@pytest.mark.asyncio
async def test_unavailable_export_needs_no_ollama_cleanup(
    app_factory, monkeypatch, tmp_path
):
    from src import documents

    app = app_factory()
    monkeypatch.setattr(documents, "compiler_capability", lambda: (False, "missing"))
    monkeypatch.setattr(documents.config, "EXPORT_DIR", tmp_path)

    class Client:
        def unload_and_wait(self):
            raise RuntimeError("Ollama offline; never called for unavailable export")

    monkeypatch.setattr(tui, "OllamaClient", Client)
    source = documents.ExportInput("r", "book", "m", "answer", "", ())
    async with app.run_test():
        token = app._try_acquire_workload("generation")
        await app._run_feature(app._export_operation(source), token)
        assert not app._busy()
        assert any("export unavailable" in line for line in app.notices)
