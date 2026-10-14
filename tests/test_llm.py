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
    assert payload["options"]["num_predict"] == 512


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
