import asyncio
import json

import httpx

from app.llm import OpenAICompatibleLLM, extract_json


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"b": [1, 2]} hope that helps') == {"b": [1, 2]}


def test_client_falls_back_when_json_mode_is_unsupported():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        assert request.headers["authorization"] == "Bearer k"
        if "response_format" in body:
            return httpx.Response(400, text='{"error": "response_format is not supported"}')
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    llm = OpenAICompatibleLLM("https://example.test/v1", "some-model", "k", transport=httpx.MockTransport(handler))
    out = asyncio.run(llm.generate("extractor", "sys", "user", {}))
    assert json.loads(out) == {"ok": True}
    assert len(calls) == 2 and calls[0]["model"] == "some-model"


def test_client_waits_and_retries_on_rate_limit():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, text="rate limit")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})

    llm = OpenAICompatibleLLM("https://example.test/v1", "openai/gpt-oss-120b", "k", transport=httpx.MockTransport(handler))
    assert json.loads(asyncio.run(llm.generate("critic", "s", "u", {}))) == {"ok": 1}
    assert len(calls) == 2
    assert calls[0]["reasoning_effort"] == "low" and calls[0]["max_tokens"] > 0
