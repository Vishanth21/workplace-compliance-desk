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
# Provider selection
# ---------------------------------------------------------------------------

#: Providers with an adapter in this module.
_KNOWN_PROVIDERS = ("scripted",)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is not None:
        value = value.strip()
    return value or None


def get_model(provider: str | None = None) -> Model:
    """Return the configured model.

    Reads ``LLM_PROVIDER`` (default ``scripted``) unless ``provider`` is given.
    Reads ``LLM_MODEL`` and ``LLM_API_KEY`` for providers that need them.

    Defaulting to ``scripted`` is deliberate: a fresh clone with a placeholder
    API key still imports and still runs its tests. Asking for a real provider
    is an explicit act.
    """
    chosen = (provider or _env("LLM_PROVIDER") or "scripted").lower()

    if chosen == "scripted":
        # An empty script is a valid starting point for wiring; any real call
        # raises with a clear message.
        raw = _env("LLM_SCRIPTED_RESPONSES")
        responses = json.loads(raw) if raw else []
        return ScriptedModel(responses)

    raise ModelError(
        f"unknown LLM_PROVIDER {chosen!r}. Known providers: "
        f"{', '.join(_KNOWN_PROVIDERS)}. Add an adapter in agent/model.py and "
        f"register it in _KNOWN_PROVIDERS to use another."
    )