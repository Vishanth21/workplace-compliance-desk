"""Tests for the OpenAI-compatible adapter (agent/model.py).

The adapter is tested at two levels.

**Offline, always run.** Construction, configuration errors, request-body shape,
and every failure path -- all against a fake transport. These run in CI with no
endpoint and no key.

**Live, opt-in.** One end-to-end call against a real endpoint, skipped unless
``LLM_LIVE_URL`` is set. It exists because a fake transport proves the adapter
matches my belief about the wire format, not that it matches the server's.

Run the live test with:

    LLM_LIVE_URL=https://api.example.com/v1 \\
    LLM_MODEL=your-model-name \\
    OPENAI_API_KEY=... \\
    uv run pytest tests/test_openai_adapter.py -k live
"""

from __future__ import annotations

import json
import os

import pytest
from pydantic import BaseModel

from agent.model import Model, ModelError, OpenAIModel, get_model


class Verdict(BaseModel):
    supported: bool
    reason: str


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | str) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self) -> dict:
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


def chat_reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


@pytest.fixture
def endpoint(monkeypatch):
    """A configured adapter with httpx.post replaced by a recording fake."""
    monkeypatch.setenv("LLM_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    calls: list[dict] = []
    replies: list[FakeResponse] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        if not replies:
            raise AssertionError("no reply queued for this call")
        return replies.pop(0)

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    return OpenAIModel(), calls, replies


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_satisfies_protocol(endpoint):
    model, _, _ = endpoint
    assert isinstance(model, Model)


def test_reads_all_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://model.test/v1/")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    m = OpenAIModel()
    # Trailing slash stripped so url building never produces a double slash.
    assert m.base_url == "http://model.test/v1"
    assert m.model == "test-model"
    assert m.api_key == "k"


def test_llm_api_key_takes_precedence_over_openai_api_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("OPENAI_API_KEY", "from-openai")
    monkeypatch.setenv("LLM_API_KEY", "from-llm")
    assert OpenAIModel().api_key == "from-llm"


def test_missing_base_url_fails_loudly(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(ModelError, match="LLM_BASE_URL"):
        OpenAIModel()


def test_missing_model_fails_loudly(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ModelError, match="LLM_MODEL"):
        OpenAIModel()


def test_get_model_selects_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    assert get_model().name == "openai"


def test_openai_is_a_known_provider():
    """The error for an unknown provider must not list openai as unknown."""
    from agent.model import _KNOWN_PROVIDERS

    assert "openai" in _KNOWN_PROVIDERS


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_sends_bearer_token_and_url(endpoint):
    model, calls, replies = endpoint
    replies.append(FakeResponse(200, chat_reply("hi")))
    model.complete([{"role": "user", "content": "hello"}])

    assert calls[0]["url"] == "http://model.test/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[0]["json"]["model"] == "test-model"


def test_no_auth_header_when_no_key(monkeypatch, endpoint):
    model, calls, replies = endpoint
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    model.api_key = None
    replies.append(FakeResponse(200, chat_reply("hi")))
    model.complete([{"role": "user", "content": "hello"}])

    assert "Authorization" not in calls[0]["headers"]


def test_structured_call_requests_json_object_and_appends_schema(endpoint):
    model, calls, replies = endpoint
    replies.append(FakeResponse(200, chat_reply('{"supported": true, "reason": "ok"}')))
    model.complete(
        [{"role": "user", "content": "verify"}], response_schema=Verdict
    )

    body = calls[0]["json"]
    assert body["response_format"] == {"type": "json_object"}
    # The schema must reach the model, or JSON mode only guarantees *some* JSON.
    last = body["messages"][-1]
    assert "JSON Schema" in last["content"]
    assert "supported" in last["content"]
    # The caller's messages are preserved, not replaced.
    assert body["messages"][0]["content"] == "verify"


def test_unstructured_call_does_not_request_json(endpoint):
    model, calls, replies = endpoint
    replies.append(FakeResponse(200, chat_reply("plain")))
    model.complete([{"role": "user", "content": "hi"}])

    assert "response_format" not in calls[0]["json"]
    # No schema message appended in the unstructured case.
    assert len(calls[0]["json"]["messages"]) == 1


def test_max_tokens_and_temperature_forwarded_only_when_given(endpoint):
    model, calls, replies = endpoint
    replies.append(FakeResponse(200, chat_reply("a")))
    model.complete([{"role": "user", "content": "x"}])
    assert "max_tokens" not in calls[0]["json"]
    assert "temperature" not in calls[0]["json"]

    replies.append(FakeResponse(200, chat_reply("b")))
    model.complete([{"role": "user", "content": "x"}], max_tokens=64, temperature=0.2)
    assert calls[1]["json"]["max_tokens"] == 64
    assert calls[1]["json"]["temperature"] == 0.2


# ---------------------------------------------------------------------------
# Parsing and failure paths
# ---------------------------------------------------------------------------


def test_structured_reply_is_validated_into_schema(endpoint):
    model, _, replies = endpoint
    replies.append(FakeResponse(200, chat_reply('{"supported": false, "reason": "no"}')))
    out = model.complete([{"role": "user", "content": "v"}], response_schema=Verdict)

    assert isinstance(out, Verdict)
    assert out.supported is False
    assert out.reason == "no"


def test_malformed_json_for_schema_raises_with_the_reply(endpoint):
    model, _, replies = endpoint
    replies.append(FakeResponse(200, chat_reply("this is prose, not JSON")))
    with pytest.raises(ModelError, match="not valid JSON for Verdict"):
        model.complete([{"role": "user", "content": "v"}], response_schema=Verdict)


def test_json_that_does_not_match_schema_raises(endpoint):
    """Valid JSON of the wrong shape is still a failure, not a silent default."""
    model, _, replies = endpoint
    replies.append(FakeResponse(200, chat_reply('{"unexpected": 1}')))
    with pytest.raises(ModelError, match="not valid JSON for Verdict"):
        model.complete([{"role": "user", "content": "v"}], response_schema=Verdict)


def test_http_error_is_a_model_error_naming_the_status(endpoint):
    model, _, replies = endpoint
    replies.append(FakeResponse(401, {"error": "Invalid API key"}))
    with pytest.raises(ModelError, match="HTTP 401"):
        model.complete([{"role": "user", "content": "x"}])


def test_unexpected_body_shape_is_a_model_error(endpoint):
    model, _, replies = endpoint
    replies.append(FakeResponse(200, {"no_choices": True}))
    with pytest.raises(ModelError, match="unexpected body"):
        model.complete([{"role": "user", "content": "x"}])


def test_connection_failure_is_a_model_error(endpoint, monkeypatch):
    model, _, _ = endpoint

    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(ModelError, match="unreachable"):
        model.complete([{"role": "user", "content": "x"}])


def test_calls_are_recorded_for_assertions(endpoint):
    """Mirrors ScriptedModel.calls so tests can read what was actually asked."""
    model, _, replies = endpoint
    msgs = [{"role": "user", "content": "record me"}]
    replies.append(FakeResponse(200, chat_reply("ok")))
    model.complete(msgs)
    assert model.calls == [msgs]


# ---------------------------------------------------------------------------
# Live endpoint (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("LLM_LIVE_URL"),
    reason="set LLM_LIVE_URL to run against a real endpoint",
)
def test_live_unstructured_call():
    m = OpenAIModel(
        base_url=os.environ["LLM_LIVE_URL"],
        model=os.environ.get("LLM_MODEL", "your-model-name"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    out = m.complete(
        [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=10
    )
    assert isinstance(out, str)
    assert out.strip()


@pytest.mark.skipif(
    not os.environ.get("LLM_LIVE_URL"),
    reason="set LLM_LIVE_URL to run against a real endpoint",
)
def test_live_structured_call():
    """The one that matters: does this server honour JSON mode + schema?"""
    m = OpenAIModel(
        base_url=os.environ["LLM_LIVE_URL"],
        model=os.environ.get("LLM_MODEL", "your-model-name"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    verdict = m.complete(
        [
            {
                "role": "user",
                "content": (
                    "A passage says 'Gratuity is payable after five years'. "
                    "The claim is 'Gratuity is payable after five years'. "
                    "Is the claim supported?"
                ),
            }
        ],
        response_schema=Verdict,
    )
    assert isinstance(verdict, Verdict)
    assert verdict.supported is True