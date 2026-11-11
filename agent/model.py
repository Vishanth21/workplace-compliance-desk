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
import logging
import os
from typing import Any, Protocol, TypeVar, runtime_checkable

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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


def _schema_instruction(response_schema: type[BaseModel]) -> str:
    """The output-format block that makes JSON mode honour the schema.

    JSON mode alone only guarantees *some* JSON object. Three things are
    needed to get the shape right, each verified against a real endpoint:

    * The block must be part of the *first* system message, not a separate
      one. A second system message is ignored whether it comes before or
      after the persona prompt -- and when it comes first it also displaces
      that prompt, so the agent stops calling its tools and answers from
      memory. Merging keeps one system message and both instructions live.
    * The required keys must be named in prose, not only in the schema. A
      model that reads a bare schema substitutes its own field names.
    * For an agent whose schema has an ``action`` field, the block has to say
      that tool use precedes answering. Otherwise ``action: "answer"`` is the
      easy way to satisfy the format, and retrieval is skipped entirely.

    Validation downstream is still the safety net; this only raises the hit
    rate.
    """
    schema = response_schema.model_json_schema()
    keys = list(schema.get("properties") or {})
    lines = [
        "OUTPUT FORMAT -- applies to this and every later turn.",
        "Reply with exactly one JSON object: no prose, no markdown, no code fence.",
        "Include every one of these keys: " + ", ".join(keys) + ".",
    ]
    if "action" in keys:
        lines.append(
            "If the question needs a tool before you can answer, set action to "
            "'tool' and fill tool and arguments; do not answer from memory."
        )
    lines.append(
        "It must conform to this JSON Schema:\n"
        + json.dumps(schema, separators=(",", ":"))
    )
    return "\n".join(lines)


def _with_schema_instruction(
    messages: list[Message], response_schema: type[BaseModel]
) -> list[Message]:
    """Fold the schema instruction into the existing system message.

    A separate system message does not work: it is either ignored (when the
    persona prompt precedes it) or overrides the persona (when it leads), and
    the latter silently turns a tool-using agent into one that answers from
    memory. Merging is the only arrangement verified to satisfy both.
    """
    instruction = _schema_instruction(response_schema)
    if messages and messages[0].get("role") == "system":
        first = dict(messages[0])
        first["content"] = (first.get("content") or "") + "\n\n" + instruction
        return [first, *messages[1:]]
    return [{"role": "system", "content": instruction}, *messages]


def _strip_markdown_fences(content: str) -> str:
    """Remove ``` fences some models wrap JSON in.

    Gemini 3 returns valid JSON wrapped in markdown code fences. The trailing
    fence parses as "trailing characters" in ``model_validate_json``, and the
    verifier reads the parse error as an unsupported claim -- a false negative
    on a correct positive verdict.
    """
    stripped = content.strip()
    if not stripped.startswith("```"):
        return content
    first_newline = stripped.find("\n")
    if first_newline == -1:
        return content
    stripped = stripped[first_newline + 1:]
    if stripped.rstrip().endswith("```"):
        stripped = stripped.rstrip()[:-3].rstrip()
    return stripped


#: Google's sentinel for callers that cannot carry a real thought_signature
#: through the OpenAI wire format.
#: https://ai.google.dev/gemini-api/docs/thought-signatures
_GEMINI_THOUGHT_SIG_BYPASS = "skip_thought_signature_validator"


def _patch_gemini_thought_signatures(body: dict[str, Any], base_url: str) -> None:
    """Stamp the bypass signature onto replayed Gemini tool calls.

    Gemini rejects a replayed functionCall without a thought_signature
    (HTTP 400). The OpenAI chat-completions format has no field for it, so an
    adapter round-tripping ``tool_calls`` drops it and turn 2 dies.
    """
    if "generativelanguage.googleapis.com" not in base_url:
        return
    for message in body.get("messages", []):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call.setdefault(
                "extra_content",
                {"google": {"thought_signature": _GEMINI_THOUGHT_SIG_BYPASS}},
            )


#: finish_reason values that mean "the provider refused to produce content for
#: safety reasons". Treated as an empty escalation rather than a provider error,
#: so a user who asks a triggering question gets routed to counsel rather than
#: seeing a raw ``model_error``. `finish_reason: "error"` is deliberately *not*
#: in this set -- it is a provider failure, not a content refusal, and should
#: raise so the breaker and the escalation reason reflect that.
_SAFETY_FINISH_REASONS = frozenset(
    {"SAFETY", "CONTENT_FILTER", "OTHER", "content_filter", "safeguard"}
)

#: Step-shaped reply returned when a model response is blocked by safety
#: filters. Validates as ``Step`` with action=answer and no citations, which
#: means the analyst loop terminates normally and the verifier's stage-1
#: ``no citations`` gate escalates to legal counsel.
_SAFETY_BLOCK_REPLY = json.dumps(
    {
        "action": "answer",
        "answer": "I cannot answer this question due to safety guidelines.",
        "citations": [],
        "claims": [],
    },
    separators=(",", ":"),
)


def _extract_content(payload: dict[str, Any], url: str, response: Any) -> str:
    """Pull the assistant text out of an OpenAI-compatible reply.

    Five paths lead to ``content`` being absent in a valid HTTP 200:

    * The model refuses JSON mode (``finish_reason: "error"``): provider bug,
      raise.
    * Reasoning models (deepseek, some Claude variants) put the entire analysis
      in ``reasoning_content`` and leave ``content`` empty: fall back.
    * Some models return a refusal string in ``message.refusal``: fall back.
    * Content filters fire (``finish_reason: "SAFETY"``, ``"content_filter"``,
      ``"safeguard"``, ``"OTHER"``): return a Step-shaped escalation so the
      analyst loop terminates cleanly.
    * Anything else: raise with ``finish_reason`` and the first 300 bytes of
      the body, so the operator sees what came back.

    The safety path was added after observing ``cohere/north-mini-code:free``
    and ``google/gemma-4-26b-a4b-it:free`` return 200 with ``content: null``
    and ``finish_reason: "error"`` in production. That specific combination
    is treated as a provider error; the safety set is narrower.
    """
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError(
            f"model endpoint {url} returned an unexpected body: "
            f"{response.text[:300]}"
        ) from exc

    message = choice.get("message") or {}
    content = message.get("content")
    if content is None:
        # Some reasoning models put the entire answer here.
        content = message.get("reasoning_content")
    if content is None:
        # Some models return a refusal field instead of content.
        content = message.get("refusal")
    if content is not None:
        return content

    finish = choice.get("finish_reason")

    # Gracefully handle safety blocks and content filters as empty escalations.
    # Returning a Step-shaped payload lets the analyst loop terminate normally
    # (route=ANSWER, no citations) and the verifier's stage-1 gate escalates.
    # A Verdict-shaped caller will fail to parse this and fail closed in the
    # verifier's exception branch -- also correct.
    if finish in _SAFETY_FINISH_REASONS:
        logger.warning(
            "Model response blocked by safety filter (finish_reason=%s); "
            "returning empty escalation",
            finish,
        )
        return _SAFETY_BLOCK_REPLY

    raise ModelError(
        f"model endpoint {url} returned HTTP 200 with message.content=null. "
        f"finish_reason={finish!r}. "
        f"The model likely does not support JSON mode "
        f"(response_format={{'type': 'json_object'}}) "
        f"or refused to produce a structured reply. "
        f"Body: {response.text[:300]}"
    )


class OpenAIModel:
    """Talks to any OpenAI-compatible ``/chat/completions`` endpoint.

    Covers OpenAI itself, Azure-compatible gateways, Gemini's OpenAI-compat
    endpoint, OpenRouter, and local inference servers (vLLM, llama.cpp, LM
    Studio). Nothing in the class is specific to one vendor: the base URL is
    configuration.

    Configuration, read at call time so a process can be reconfigured without
    a rebuild:

    * ``LLM_BASE_URL``   -- e.g. ``https://openrouter.ai/api/v1``
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
                "https://openrouter.ai/api/v1"
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
            # JSON mode plus the schema folded into the system message.
            # Function calling would also work, but both call sites (Step,
            # Verdict) are flat objects and several OpenAI-compatible servers
            # implement JSON mode and not tools. Validation below is what
            # makes it safe; the instruction only raises the hit rate.
            body["response_format"] = {"type": "json_object"}
            body["messages"] = _with_schema_instruction(messages, response_schema)

        _patch_gemini_thought_signatures(body, self.base_url)
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
        except ValueError as exc:
            raise ModelError(
                f"model endpoint {url} returned an unexpected body: "
                f"{response.text[:300]}"
            ) from exc

        content = _extract_content(payload, url, response)

        if response_schema is None:
            return content

        content = _strip_markdown_fences(content)
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
