import httpx
import pytest

from src import config
from src.llm import OllamaClient


def test_generate_uses_local_budget_and_keep_alive_zero():
    seen = []

    def handler(request):
        seen.append((request.url.path, request.content))
        return httpx.Response(200, json={"response": "answer"})

    client = OllamaClient(transport=httpx.MockTransport(handler))
    result = client.generate("question", model="qwen3.5:4b-q4_K_M", stream=False)
    assert result["response"] == "answer"
    path, raw = seen[0]
    import json

    payload = json.loads(raw)
    assert path == "/api/generate"
    assert "think" not in payload
    assert payload["keep_alive"] == 0
    assert payload["options"]["num_ctx"] == 4096
    assert payload["options"]["num_predict"] == 768


def test_generate_can_disable_qwen_thinking():
    request_body = {}

    def handler(request):
        import json

        request_body.update(json.loads(request.content))
        return httpx.Response(200, json={"response": "answer", "thinking": None})

    client = OllamaClient(transport=httpx.MockTransport(handler))
    client.generate("question", stream=False, think=False)
    assert request_body["think"] is False


def test_chat_preserves_tool_messages_and_thinking_separately():
    request_body = {}

    def handler(request):
        import json

        request_body.update(json.loads(request.content))
        return httpx.Response(
            200, json={"message": {"thinking": "reason", "content": "done"}}
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "add", "arguments": {"a": 1, "b": 2}}}
            ],
        },
        {"role": "tool", "content": "3"},
    ]
    response = client.chat(
        messages,
        model="qwen3.5:4b-q4_K_M",
        tools=[{"type": "function", "function": {"name": "add"}}],
        think=False,
    )
    assert request_body["messages"] == messages
    assert request_body["tools"][0]["function"]["name"] == "add"
    assert request_body["keep_alive"] == 0
    assert request_body["think"] is False
    assert response["message"]["thinking"] == "reason"
    assert response["message"]["content"] == "done"


def test_unload_verifies_empty_process_list_and_sets_keep_alive_zero():
    calls = []
    timeouts = []
    process_lists = iter([[{"name": "model:tag"}], []])

    def handler(request):
        import json

        calls.append((request.url.path, json.loads(request.content or b"{}")))
        timeouts.append(request.extensions.get("timeout", {}).get("read"))
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": next(process_lists)})
        return httpx.Response(200, json={"done": True})

    client = OllamaClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    client.unload_and_wait()
    assert calls[1][0] == "/api/generate"
    assert calls[1][1]["keep_alive"] == 0
    assert timeouts and all(
        timeout <= config.OLLAMA_UNLOAD_TIMEOUT_SECONDS for timeout in timeouts
    )


def test_unload_raises_when_model_remains_loaded():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "model:tag"}]})

    ticks = iter([0.0, 0.0, 0.0, 20.1])
    client = OllamaClient(
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
        monotonic=lambda: next(ticks),
    )
    with pytest.raises(TimeoutError, match="deadline"):
        client.unload_and_wait()


def test_http_errors_propagate():
    client = OllamaClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.generate("q", model="local", stream=False)


def test_stream_idle_timeout_is_distinct_from_nonstream_response_timeout():
    client = OllamaClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    with client._http_client(streaming=True) as stream_client:
        assert stream_client.timeout.read == config.OLLAMA_READ_TIMEOUT_SECONDS
    with client._http_client() as request_client:
        assert request_client.timeout.read == config.OLLAMA_REQUEST_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_async_stream_allows_cold_model_response_budget():
    def handler(request):
        assert (
            request.extensions["timeout"]["read"]
            == config.OLLAMA_REQUEST_TIMEOUT_SECONDS
        )
        return httpx.Response(200, text='{"response":"answer","done":true}\n')

    client = OllamaClient(async_transport=httpx.MockTransport(handler))
    chunks = [chunk async for chunk in client.astream_generate("question")]
    assert chunks == [{"response": "answer", "done": True}]


@pytest.mark.asyncio
async def test_async_nonstream_override_budget_and_total_deadline():
    import asyncio
    import json

    seen = []

    async def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"done": True, "response": "Standalone question?"}
        )

    client = OllamaClient(async_transport=httpx.MockTransport(handler))
    assert (
        await client.agenerate(
            "rewrite", model="local", think=False, num_predict=96, timeout=15
        )
    )["done"]
    assert (
        seen[0]["options"]["num_predict"] == 96
        and seen[0]["keep_alive"] == 0
        and not seen[0]["stream"]
    )

    async def stalled(request):
        await asyncio.Event().wait()

    client = OllamaClient(async_transport=httpx.MockTransport(stalled))
    with pytest.raises(TimeoutError):
        await client.agenerate(
            "q", model="local", think=False, num_predict=96, timeout=0.01
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [{"error": "runner failed"}, {"done": False, "response": "partial"}, ["bad shape"]],
)
async def test_async_completion_rejects_error_or_incomplete_json(body):
    client = OllamaClient(
        async_transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    )
    with pytest.raises(ValueError, match="completion"):
        await client.agenerate(
            "q", model="local", think=False, num_predict=96, timeout=15
        )


def _chat_stream(chunks):
    import json

    body = "".join(json.dumps(c) + "\n" for c in chunks).encode()
    return httpx.Response(200, content=body)


async def test_native_chat_keeps_budget_tools_and_raw_turn_fields():
    import json

    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _chat_stream(
            [
                {"message": {"thinking": "plan ", "content": ""}, "done": False},
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "f", "arguments": {"q": "x"}}}
                        ],
                    },
                    "done": False,
                },
                {
                    "message": {"content": "ok"},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "eval_count": 3,
                },
            ]
        )

    client = OllamaClient(async_transport=httpx.MockTransport(handler))
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    turn = await client.achat(
        [{"role": "user", "content": "q"}],
        model=config.OLLAMA_MODEL,
        tools=tools,
        think=True,
        num_predict=768,
        timeout=120.0,
    )
    assert seen["keep_alive"] == 0 and seen["stream"] is True
    assert seen["options"]["num_predict"] == 768 and seen["think"] is True
    assert seen["tools"] == tools
    assert turn["message"]["thinking"] == "plan "
    assert turn["message"]["tool_calls"][0]["function"]["name"] == "f"
    assert turn["complete"] and turn["prompt_eval_count"] == 10


async def test_length_limited_chat_is_not_complete():
    def handler(request):
        return _chat_stream(
            [{"message": {"content": "partial"}, "done": True, "done_reason": "length"}]
        )

    client = OllamaClient(async_transport=httpx.MockTransport(handler))
    turn = await client.achat(
        [{"role": "user", "content": "q"}],
        model=config.OLLAMA_MODEL,
        tools=None,
        think=False,
        num_predict=8,
        timeout=5.0,
    )
    assert turn["complete"] is False and turn["done_reason"] == "length"


@pytest.mark.parametrize(
    "chunks",
    [
        [{"error": "model failed"}],
        [{"message": {"content": "no final"}, "done": False}],
    ],
)
async def test_chat_errors_and_missing_final_chunk_fail(chunks):
    client = OllamaClient(
        async_transport=httpx.MockTransport(lambda request: _chat_stream(chunks))
    )
    with pytest.raises((RuntimeError, ValueError)):
        await client.achat(
            [{"role": "user", "content": "q"}],
            model=config.OLLAMA_MODEL,
            tools=None,
            think=False,
            num_predict=8,
            timeout=5.0,
        )


async def test_chat_total_deadline_covers_silent_streams():
    import asyncio

    class Silent(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b""

    client = OllamaClient(
        async_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=Silent())
        )
    )
    with pytest.raises(TimeoutError):
        await client.achat(
            [{"role": "user", "content": "q"}],
            model=config.OLLAMA_MODEL,
            tools=None,
            think=False,
            num_predict=8,
            timeout=0.05,
        )
