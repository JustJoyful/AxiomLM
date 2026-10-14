"""Exercise Textual query dispatch, cancellation, cleanup, and GPU admission."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from src import tui


async def validate() -> dict[str, bool | str]:
    entered = Event()
    finish_query = Event()
    cleaned_up = Event()
    events: list[str] = []

    class Client:
        def unload_and_wait(self) -> None:
            events.append("unloaded")
            cleaned_up.set()

    ollama_client = tui.OllamaClient
    executor = ThreadPoolExecutor(max_workers=2)
    asyncio.get_running_loop().set_default_executor(executor)
    tui.OllamaClient = Client
    app = tui.AxiomLMApp()
    app.active_collection_name = "phase0_probe"

    def execute_query(*_args: object) -> None:
        events.append("started")
        entered.set()
        if not finish_query.wait(timeout=5):
            raise TimeoutError("Probe query did not receive release.")
        events.append("returned")

    app.execute_rag_query = execute_query

    try:
        async with app.run_test():
            composer = app.query_one("#query_input", tui.QueryComposer)
            await app.on_query_composer_submitted(
                tui.QueryComposer.Submitted(composer, "phase 0 admission probe")
            )
            if not await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=6):
                raise AssertionError("Textual did not dispatch the query worker.")

            app.action_cancel_query()
            if app._workload_coordinator.active_kind != "generation":
                raise AssertionError("Cancellation released generation admission early.")
            if app._try_acquire_workload("ingest") is not None:
                raise AssertionError("Ingest was admitted during generation cleanup.")

            finish_query.set()
            if not await asyncio.wait_for(asyncio.to_thread(cleaned_up.wait), timeout=6):
                raise AssertionError("Ollama cleanup was not reached after the worker returned.")

            deadline = asyncio.get_running_loop().time() + 2
            while app._workload_coordinator.active_kind is not None:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("Generation admission stayed closed after cleanup.")
                await asyncio.sleep(0.01)

            if events != ["started", "returned", "unloaded"]:
                raise AssertionError(f"Unexpected query/cleanup ordering: {events!r}")

            ingest_token = app._try_acquire_workload("ingest")
            if ingest_token is None:
                raise AssertionError("Ingest admission did not reopen after cleanup.")
            app._release_workload(ingest_token)
    finally:
        finish_query.set()
        tui.OllamaClient = ollama_client
        executor.shutdown(wait=True)

    return {
        "tui_dispatch_verified": True,
        "cancellation_cleanup_verified": True,
        "admission_held_through_unload": True,
        "note": "Actual Textual submit/cancel handler; Ollama transport replaced with a controlled unload adapter.",
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(validate()), indent=2))
