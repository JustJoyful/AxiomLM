"""Exercise Textual query dispatch, cancellation, cleanup, and GPU admission."""

import asyncio
import json
from threading import Event

from src import tui


async def validate() -> dict[str, bool | str]:
    entered = asyncio.Event()
    finish_cleanup = Event()
    events: list[str] = []

    class Client:
        async def astream_generate(self, *_args, **_kwargs):
            events.append("started")
            entered.set()
            try:
                yield {"response": "partial answer"}
                await asyncio.Event().wait()
            finally:
                events.append("closed")

        def unload_and_wait(self):
            events.append("cleanup_started")
            if not finish_cleanup.wait(timeout=5):
                raise TimeoutError("Probe cleanup did not receive release")
            events.append("unloaded")

    ollama_client = tui.OllamaClient
    tui.OllamaClient = Client
    app = tui.AxiomLMApp()
    app._agent_routing = False  # validates the Phase 2 path
    app.active_collection_name = "phase0_probe"
    app._discover_books = lambda: {"phase0_probe": 1}
    app._fetch_ollama_models = lambda: [tui.OLLAMA_MODEL]
    app._prepare_query = lambda *_args: ("prompt", [], [], [])
    try:
        async with app.run_test():
            composer = app.query_one("#query_input", tui.QueryComposer)
            await app.on_query_composer_submitted(
                tui.QueryComposer.Submitted(composer, "phase 0 admission probe")
            )
            await asyncio.wait_for(entered.wait(), timeout=2)
            await app._cancel_active_query()
            if app._workload_coordinator.active_kind != "generation":
                raise AssertionError("Cancellation released generation admission early")
            if app._try_acquire_workload("ingest") is not None:
                raise AssertionError("Ingest admitted before confirmed cleanup")
            finish_cleanup.set()
            deadline = asyncio.get_running_loop().time() + 2
            while app._workload_coordinator.active_kind is not None:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError(
                        "Generation admission stayed closed after cleanup"
                    )
                await asyncio.sleep(0.01)
            if events != ["started", "closed", "cleanup_started", "unloaded"]:
                raise AssertionError(
                    f"Unexpected transport/cleanup ordering: {events!r}"
                )
            token = app._try_acquire_workload("ingest")
            if token is None:
                raise AssertionError("Ingest admission did not reopen")
            app._release_workload(token)
    finally:
        finish_cleanup.set()
        tui.OllamaClient = ollama_client
    return {
        "tui_dispatch_verified": True,
        "cancellation_cleanup_verified": True,
        "admission_held_through_unload": True,
        "note": "Actual async Textual submit/cancel lifecycle; controlled local transport and unload adapters.",
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(validate()), indent=2))
