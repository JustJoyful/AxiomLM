"""Measure local client abort and server unload separately; no ingest or downloads."""

import asyncio
import json
import time

import httpx

from src import config
from src.llm import OllamaClient


async def validate():
    async with httpx.AsyncClient(
        trust_env=False, timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS
    ) as http:
        response = await http.get(config.OLLAMA_URL + "/api/ps")
        response.raise_for_status()
        if response.json().get("models"):
            raise RuntimeError(
                "Ollama is already using a model; wait for other work to finish"
            )
    first = asyncio.Event()
    partial = []
    client = OllamaClient()

    async def consume():
        async for chunk in client.astream_generate(
            "Explain how an interpreter evaluates Scheme expressions, in numbered steps.",
            model=config.OLLAMA_MODEL,
            think=False,
        ):
            if chunk.get("response"):
                partial.append(chunk["response"])
                first.set()

    started = time.monotonic()
    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(
            first.wait(), timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS
        )
        first_seconds = time.monotonic() - started
        await asyncio.sleep(config.UI_FRAME_INTERVAL)
        abort_started = time.monotonic()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        abort_seconds = time.monotonic() - abort_started
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        unload_started = time.monotonic()
        await asyncio.to_thread(client.unload_and_wait)
        unload_seconds = time.monotonic() - unload_started
    report = {
        "model": config.OLLAMA_MODEL,
        "first_token_seconds": first_seconds,
        "client_abort_seconds": abort_seconds,
        "client_abort_under_200ms": abort_seconds <= 0.2,
        "confirmed_server_unload_seconds": unload_seconds,
        "partial_characters": len("".join(partial)),
        "note": "Real local HTTP transport; UI latency is measured separately with the deterministic fake backend.",
    }
    config.UI_LIVE_REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(asyncio.run(validate()), indent=2))
