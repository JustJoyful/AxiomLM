"""Small synchronous client for the configured local Ollama service."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx

from src.config import (
    OLLAMA_CONTEXT_TOKENS,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MIROSTAT,
    OLLAMA_MIROSTAT_ETA,
    OLLAMA_MIROSTAT_TAU,
    OLLAMA_MODEL,
    OLLAMA_OUTPUT_TOKENS,
    OLLAMA_POLL_INTERVAL_SECONDS,
    OLLAMA_READ_TIMEOUT_SECONDS,
    OLLAMA_REQUEST_TIMEOUT_SECONDS,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    OLLAMA_UNLOAD_TIMEOUT_SECONDS,
    OLLAMA_URL,
)

JsonObject = dict[str, Any]


class OllamaClient:
    """Call only the local Ollama endpoint, with bounded configured budgets."""

    def __init__(
        self,
        *,
        base_url: str = OLLAMA_URL,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self._async_transport = async_transport
        self._sleep = sleep
        self._monotonic = monotonic

    def _http_client(self, *, streaming: bool = False) -> httpx.Client:
        timeout = httpx.Timeout(
            OLLAMA_REQUEST_TIMEOUT_SECONDS,
            read=(
                OLLAMA_READ_TIMEOUT_SECONDS
                if streaming
                else OLLAMA_REQUEST_TIMEOUT_SECONDS
            ),
        )
        return httpx.Client(timeout=timeout, transport=self._transport, trust_env=False)

    @staticmethod
    def _options() -> JsonObject:
        options: JsonObject = {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "num_ctx": OLLAMA_CONTEXT_TOKENS,
            "num_predict": OLLAMA_OUTPUT_TOKENS,
        }
        if OLLAMA_MIROSTAT in (1, 2):
            options.update(
                mirostat=OLLAMA_MIROSTAT,
                mirostat_tau=OLLAMA_MIROSTAT_TAU,
                mirostat_eta=OLLAMA_MIROSTAT_ETA,
            )
        return options

    def generate(
        self,
        prompt: str,
        *,
        model: str = OLLAMA_MODEL,
        stream: bool,
        think: bool | None = None,
    ) -> JsonObject | Iterator[JsonObject]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": self._options(),
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }
        if think is not None:
            payload["think"] = think
        if not stream:
            with self._http_client() as client:
                response = client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                return response.json()
        return self._generate_stream(payload)

    def _generate_stream(self, payload: JsonObject) -> Iterator[JsonObject]:
        with (
            self._http_client(streaming=True) as client,
            client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as response,
        ):
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)

    async def agenerate(
        self, prompt: str, *, model: str, think: bool, num_predict: int, timeout: float
    ) -> JsonObject:
        payload = {
            "model": model,
            "prompt": prompt,
            "think": think,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {**self._options(), "num_predict": num_predict},
        }
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(
                timeout=timeout, transport=self._async_transport, trust_env=False
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate", json=payload
                )
                response.raise_for_status()
                data = response.json()
                if (
                    not isinstance(data, dict)
                    or data.get("error")
                    or data.get("done") is not True
                    or not isinstance(data.get("response"), str)
                ):
                    raise ValueError("Invalid or incomplete Ollama completion")
                return data

    async def astream_generate(
        self, prompt: str, *, model: str = OLLAMA_MODEL, think: bool = False
    ) -> AsyncIterator[JsonObject]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "think": think,
            "options": self._options(),
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }
        # Cold model startup may be silent; task cancellation closes the read directly.
        timeout = httpx.Timeout(OLLAMA_REQUEST_TIMEOUT_SECONDS)
        async with (
            httpx.AsyncClient(
                timeout=timeout, transport=self._async_transport, trust_env=False
            ) as client,
            client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if data.get("error"):
                        raise RuntimeError(str(data["error"]))
                    yield data

    async def astream_chat(
        self,
        messages: list[JsonObject],
        *,
        model: str,
        tools: list[JsonObject] | None,
        think: bool,
        num_predict: int,
        timeout: float,
    ) -> AsyncIterator[JsonObject]:
        """Stream native chat chunks under one total deadline, silent gaps included."""
        payload: JsonObject = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {**self._options(), "num_predict": num_predict},
        }
        if tools:
            payload["tools"] = tools
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        def remaining() -> float:
            left = deadline - loop.time()
            if left <= 0:
                raise TimeoutError("Ollama chat exceeded its total deadline")
            return left

        async with (
            httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                transport=self._async_transport,
                trust_env=False,
            ) as client,
            client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response,
        ):
            response.raise_for_status()
            lines = response.aiter_lines()
            while True:
                try:
                    line = await asyncio.wait_for(anext(lines), remaining())
                except StopAsyncIteration:
                    return
                if not line:
                    continue
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("Invalid Ollama chat chunk")  # noqa: TRY004
                if data.get("error"):
                    raise RuntimeError(str(data["error"]))
                yield data

    async def achat(
        self,
        messages: list[JsonObject],
        *,
        model: str,
        tools: list[JsonObject] | None,
        think: bool,
        num_predict: int,
        timeout: float,
        on_chunk: Callable[[JsonObject], None] | None = None,
    ) -> JsonObject:
        """Aggregate one streamed turn; raw thinking/tool_calls are kept for replay."""
        content, thinking, calls, final = [], [], [], None
        async for chunk in self.astream_chat(
            messages,
            model=model,
            tools=tools,
            think=think,
            num_predict=num_predict,
            timeout=timeout,
        ):
            message = chunk.get("message") or {}
            content.append(message.get("content") or "")
            thinking.append(message.get("thinking") or "")
            calls.extend(message.get("tool_calls") or [])
            if on_chunk is not None:
                on_chunk(chunk)
            if chunk.get("done"):
                final = chunk
        if final is None:
            raise ValueError("Ollama chat stream ended without a final chunk")
        return {
            "message": {
                "role": "assistant",
                "content": "".join(content),
                "thinking": "".join(thinking),
                "tool_calls": calls,
            },
            "done_reason": final.get("done_reason"),
            # A length stop is never a complete answer or a complete tool call.
            "complete": final.get("done_reason") == "stop",
            "prompt_eval_count": final.get("prompt_eval_count"),
            "prompt_eval_cached_count": final.get("prompt_eval_cached_count"),
            "eval_count": final.get("eval_count"),
            "total_duration": final.get("total_duration"),
            "load_duration": final.get("load_duration"),
        }

    def chat(
        self,
        messages: list[JsonObject],
        *,
        model: str = OLLAMA_MODEL,
        tools: list[JsonObject] | None = None,
        think: bool = False,
    ) -> JsonObject:
        payload: JsonObject = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": self._options(),
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }
        if tools is not None:
            payload["tools"] = tools
        with self._http_client() as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()

    def unload_and_wait(self) -> None:
        """Unload Ollama's current models and fail unless they leave `/api/ps`."""
        deadline = self._monotonic() + OLLAMA_UNLOAD_TIMEOUT_SECONDS

        def remaining_timeout() -> float:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise TimeoutError("Ollama unload exceeded its configured deadline.")
            return remaining

        with self._http_client() as client:
            response = client.get(
                f"{self.base_url}/api/ps", timeout=remaining_timeout()
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            names = [model["name"] for model in models if model.get("name")]

            for name in names:
                unload = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": name,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": OLLAMA_KEEP_ALIVE,
                    },
                    timeout=remaining_timeout(),
                )
                unload.raise_for_status()

            while True:
                response = client.get(
                    f"{self.base_url}/api/ps", timeout=remaining_timeout()
                )
                response.raise_for_status()
                still_loaded = response.json().get("models", [])
                remaining = [
                    model.get("name") for model in still_loaded if model.get("name")
                ]
                if not remaining:
                    return
                if self._monotonic() >= deadline:
                    raise TimeoutError(
                        "Ollama models remained loaded after the configured unload deadline: "
                        + ", ".join(remaining)
                    )
                self._sleep(OLLAMA_POLL_INTERVAL_SECONDS)
