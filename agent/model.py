"""The one model interface every agent goes through.

The brief: "place the model behind an interface you control so the provider can
be changed by configuration." This module is that interface. No agent imports a
provider SDK directly; they take a ``Model`` and call ``complete``.

Three things are deliberately narrow:

* **One method.** ``complete`` is the only call an agent makes. Anything an
  agent needs beyond text (a tool call, a structured verdict) is expressed as
  a message or requested through ``response_schema`` -- not as a new method.
* **Provider chosen by config.** ``get_model()`` reads ``LLM_PROVIDER`` from the
  environment and returns the matching class. Swapping providers edits ``.env``,
  not code.
* **A scripted implementation ships with it.** ``ScriptedModel`` returns
  recorded responses with no network and no API key. It is what the test suite
  uses, what the Day-6 load test uses, and what unblocks the loop on a day when
  no key exists.

Messages are plain dicts in the shape every provider uses::

    {"role": "system" | "user" | "assistant" | "tool", "content": str, ...}

Keeping messages as dicts rather than a Pydantic class is intentional: it is the
lowest common denominator across OpenAI, Anthropic, and local models, so a new
provider adapter does not need a translation layer for its own message type.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, TypeVar, runtime_checkable

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

#: A message in provider-neutral shape.
Message = dict[str, Any]


class ModelError(RuntimeError):
    """A model call failed in a way the caller cannot fix by retrying.

    Missing credentials, unknown provider, an unparseable structured response.
    Retryable failures (rate limits, timeouts) are the caller's to handle: it
    retries, and raises this type only once it gives up.
    """


@runtime_checkable
class Model(Protocol):
    """What every provider adapter must implement.

    ``complete`` takes the conversation and returns the assistant's reply.

    When ``response_schema`` is given, the reply must be valid JSON conforming
    to that Pydantic model, and is returned as an instance of it. Agents that
    need a decision (supported / not supported) ask for a schema rather than
    parsing prose -- a verifier cannot be built on regex over an essay.

    ``max_tokens`` and ``temperature`` are honoured where the provider supports
    them and ignored where it does not. Neither is required.
    """

    name: str

    def complete(
        self,
        messages: list[Message],
        *,
        response_schema: type[T] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | T:
        ...


# ---------------------------------------------------------------------------
# Scripted: the implementation that always works
# ---------------------------------------------------------------------------


class ScriptedModel:
    """Replays recorded responses, in order, with no network.

    Construct it with a list of replies. Each ``complete`` call returns the next
    one. When the replies run out, it raises -- a silent fallback to an empty
    string would let a test pass while the agent produced nothing.

    For structured calls, pass ``BaseModel`` instances (or dicts) in the same
    list; they are returned as-is when ``response_schema`` matches.

    >>> m = ScriptedModel(["first", "second"])
    >>> m.complete([{"role": "user", "content": "hi"}])
    'first'
    """

    name = "scripted"

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self._index = 0
        #: Every call made, for tests that assert on what the agent asked.
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        response_schema: type[T] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | T:
        self.calls.append(messages)

        if self._index >= len(self._responses):
            raise ModelError(
                f"ScriptedModel ran out of responses after {self._index} call(s). "
                f"The agent made one more call than the script expected."
            )

        reply = self._responses[self._index]
        self._index += 1

        if response_schema is None:
            if isinstance(reply, BaseModel):
                raise ModelError(
                    f"ScriptedModel got a {type(reply).__name__} but the caller "
                    f"asked for unstructured text."
                )
            return reply

        # Structured call: normalise to the requested schema.
        if isinstance(reply, response_schema):
            return reply
        if isinstance(reply, BaseModel):
            return response_schema.model_validate(reply.model_dump())
        if isinstance(reply, dict):
            return response_schema.model_validate(reply)
        if isinstance(reply, str):
            # A recorded JSON string, as a real provider would return.
            try:
                return response_schema.model_validate_json(reply)
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                raise ModelError(
                    f"ScriptedModel string reply is not valid JSON for "
                    f"{response_schema.__name__}: {exc}"
                ) from exc

        raise ModelError(
            f"ScriptedModel cannot produce {response_schema.__name__} from "
            f"{type(reply).__name__}."
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP endpoint
# ---------------------------------------------------------------------------


class OpenAIModel:
    """Talks to any OpenAI-compatible ``/chat/completions`` endpoint.

    Covers OpenAI itself, Azure-compatible gateways, and local inference servers
    (vLLM, llama.cpp, LM Studio). Nothing in the class is specific to one
    vendor: the base URL is configuration.

    Configuration, read at call time so a process can be reconfigured without
    a rebuild:

    * ``LLM_BASE_URL``   -- e.g. ``https://api.example.com/v1``
    * ``LLM_API_KEY``    -- falls back to ``OPENAI_API_KEY``
    * ``LLM_MODEL``      -- the model name the endpoint expects
    * ``LLM_TIMEOUT_S``  -- per-request timeout, default 60

    Retries are deliberately *not* handled here. ``ModelError`` documents that
    retryable failures belong to the caller, and the circuit breaker (A5) is
    that caller. An adapter that retried internally would hide the failures the
    breaker exists to detect.
    """

    name = "openai"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.base_url = (base_url or _env("LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or _env("LLM_API_KEY") or _env("OPENAI_API_KEY")
        self.model = model or _env("LLM_MODEL")
        raw_timeout = timeout_s if timeout_s is not None else _env("LLM_TIMEOUT_S")
        self.timeout_s = float(raw_timeout) if raw_timeout else 60.0

        if not self.base_url:
            raise ModelError(
                "LLM_PROVIDER=openai needs LLM_BASE_URL, e.g. "
                "https://api.example.com/v1"
            )
        if not self.model:
            raise ModelError(
                "LLM_PROVIDER=openai needs LLM_MODEL, e.g. your-model-name"
            )
        # An endpoint with no auth is unusual but legal; do not fail on it. The
        # server answers 401 and that surfaces as a ModelError with context.

        #: Every call made, matching ScriptedModel, for tests that assert on it.
        self.calls: list[list[Message]] = []

    def _request_body(
        self,
        messages: list[Message],
        response_schema: type[T] | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature

        if response_schema is not None:
            # JSON mode plus the schema in the prompt. Function calling would
            # also work, but both call sites (Step, Verdict) are flat objects
            # and several OpenAI-compatible servers implement JSON mode and
            # not tools. The schema instruction is what makes the difference
            # in practice; validation below is what makes it safe.
            body["response_format"] = {"type": "json_object"}
            schema_json = json.dumps(
                response_schema.model_json_schema(), separators=(",", ":")
            )
            body["messages"] = [
                *messages,
                {
                    "role": "system",
                    "content": (
                        "Reply with a single JSON object and nothing else. "
                        "It must conform to this JSON Schema:\n" + schema_json
                    ),
                },
            ]
        return body

    def complete(
        self,
        messages: list[Message],
        *,
        response_schema: type[T] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | T:
        self.calls.append(messages)
        body = self._request_body(
            messages, response_schema, max_tokens, temperature
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        try:
            response = httpx.post(
                url, json=body, headers=headers, timeout=self.timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - normalised to ModelError
            raise ModelError(f"model endpoint {url} unreachable: {exc}") from exc

        if response.status_code != 200:
            # Truncate: an HTML error page from a misrouted proxy is not
            # useful in an exception message.
            raise ModelError(
                f"model endpoint {url} returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ModelError(
                f"model endpoint {url} returned an unexpected body: "
                f"{response.text[:300]}"
            ) from exc

        if response_schema is None:
            return content

        try:
            return response_schema.model_validate_json(content)
        except Exception as exc:  # noqa: BLE001 - re-raised with the payload
            raise ModelError(
                f"model reply is not valid JSON for {response_schema.__name__}: "
                f"{exc}. Reply was: {content[:300]}"
            ) from exc


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

#: Providers with an adapter in this module.
_KNOWN_PROVIDERS = ("scripted", "openai")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is not None:
        value = value.strip()
    return value or None


def get_model(provider: str | None = None, *, wrap_resilient: bool = True) -> Model:
    """Return the configured model.

    Reads ``LLM_PROVIDER`` (default ``scripted``) unless ``provider`` is given.
    Reads ``LLM_MODEL`` and ``LLM_API_KEY`` for providers that need them.

    Defaulting to ``scripted`` is deliberate: a fresh clone with a placeholder
    API key still imports and still runs its tests. Asking for a real provider
    is an explicit act.

    When ``wrap_resilient`` is True and ``LLM_RESILIENCE`` is not "off", wraps
    the model with ``ResilientModel`` (A5 circuit breaker and backoff retry).
    """
    chosen = (provider or _env("LLM_PROVIDER") or "scripted").lower()

    model: Model
    if chosen == "scripted":
        # An empty script is a valid starting point for wiring; any real call
        # raises with a clear message.
        raw = _env("LLM_SCRIPTED_RESPONSES")
        responses = json.loads(raw) if raw else []
        model = ScriptedModel(responses)
    elif chosen == "openai":
        model = OpenAIModel()
    else:
        raise ModelError(
            f"unknown LLM_PROVIDER {chosen!r}. Known providers: "
            f"{', '.join(_KNOWN_PROVIDERS)}. Add an adapter in agent/model.py and "
            f"register it in _KNOWN_PROVIDERS to use another."
        )

    if wrap_resilient and (_env("LLM_RESILIENCE") or "on").lower() != "off":
        from .resilience import ResilientModel

        return ResilientModel(model)

    return model
