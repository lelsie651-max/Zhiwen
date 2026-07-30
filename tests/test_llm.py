from __future__ import annotations

import json

import httpx
import pytest

from app.core.config import Settings
from app.services.llm import (
    LLMConfigurationError,
    LLMMessage,
    LLMResponseError,
    LLMTransportError,
    MockLLMClient,
    OpenAICompatibleClient,
    build_llm_client,
    extract_json,
)


def test_extract_json_from_bare_object():
    assert extract_json('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_extract_json_from_fenced_block():
    text = "Here is the result:\n```json\n{\"name\": \"角色\"}\n```\nThanks!"
    assert extract_json(text) == {"name": "角色"}


def test_extract_json_from_prose_wrapped_span():
    text = 'The answer is [{"x": 1}] which follows the schema.'
    assert extract_json(text) == [{"x": 1}]


def test_extract_json_handles_braces_inside_strings():
    text = '{"note": "a } inside a string", "ok": true}'
    assert extract_json(text) == {"note": "a } inside a string", "ok": True}


def test_extract_json_raises_on_garbage():
    with pytest.raises(LLMResponseError):
        extract_json("no json here at all")


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_mock_client_returns_responses_in_order_and_records_calls():
    client = MockLLMClient(["first", "second"])
    out1 = _run(client.complete([LLMMessage("user", "hi")]))
    out2 = _run(client.complete([LLMMessage("user", "again")]))
    out3 = _run(client.complete([LLMMessage("user", "third")]))
    assert (out1, out2) == ("first", "second")
    # Last response repeats once the list is exhausted.
    assert out3 == "second"
    assert len(client.calls) == 3
    assert client.calls[0][0].content == "hi"


def test_mock_client_supports_handler():
    def handler(messages):
        return json.dumps({"echo": messages[-1].content})

    client = MockLLMClient(handler=handler)
    out = _run(client.complete([LLMMessage("user", "payload")]))
    assert extract_json(out) == {"echo": "payload"}


def test_mock_client_requires_responses_or_handler():
    with pytest.raises(ValueError):
        MockLLMClient()


def test_build_llm_client_requires_api_key():
    settings = Settings(llm_api_key="")
    with pytest.raises(LLMConfigurationError):
        build_llm_client(settings)


def test_build_llm_client_constructs_openai_compatible():
    settings = Settings(llm_api_key="sk-test", llm_model="deepseek-chat")
    client = build_llm_client(settings)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "deepseek-chat"


def test_openai_client_rejects_blank_key():
    with pytest.raises(LLMConfigurationError):
        OpenAICompatibleClient(api_key="  ", base_url="https://x/v1", model="m")


def test_openai_client_parses_content_via_mock_transport():
    def responder(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "deepseek-chat"
        assert body["messages"][0]["role"] == "system"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    transport = httpx.MockTransport(responder)
    client = OpenAICompatibleClient(
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        transport=transport,
    )
    content = _run(
        client.complete(
            [LLMMessage("system", "you are a helper"), LLMMessage("user", "go")],
            response_format_json=True,
        )
    )
    assert extract_json(content) == {"ok": True}


def test_openai_client_retries_then_fails_on_server_error():
    attempts = {"count": 0}

    def responder(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(503, text="unavailable")

    transport = httpx.MockTransport(responder)
    client = OpenAICompatibleClient(
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        max_retries=2,
        transport=transport,
    )
    with pytest.raises(LLMTransportError):
        _run(client.complete([LLMMessage("user", "go")]))
    # 1 initial + 2 retries.
    assert attempts["count"] == 3


def test_openai_client_raises_on_4xx_without_retry():
    attempts = {"count": 0}

    def responder(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(401, text="unauthorized")

    transport = httpx.MockTransport(responder)
    client = OpenAICompatibleClient(
        api_key="sk-bad",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        max_retries=2,
        transport=transport,
    )
    with pytest.raises(LLMTransportError):
        _run(client.complete([LLMMessage("user", "go")]))
    assert attempts["count"] == 1
