# ADR 0001: Provider-agnostic model seam, scripted now

- **Status:** Accepted
- **Date:** 2026-10-29
- **Deciders:** project owner

## Context

The Analyst agent needs an LLM. Three constraints shaped the decision:

1. **The demo must be reproducible.** A graded demo that depends on a live
   third-party API can fail for reasons unrelated to the project: an expired
   key, a rate limit, a network drop, a model deprecation mid-week.
2. **No provider is fixed yet.** OpenAI-compatible endpoints, Gemini, and
   self-hosted inference are all plausible, and the evaluation criteria do not
   name one.
3. **Tests must not touch the network.** A test suite that calls an external
   API is slow, flaky, and non-hermetic.

`agent/model.py` already defines the right abstraction: a `Model` protocol with
one method, `complete(messages, *, response_schema, max_tokens, temperature)`.
Structured replies are requested by passing a Pydantic schema, so agents get a
validated decision rather than prose to parse. That is the seam.

## Decision

**Ship scripted-only. Keep the seam provider-agnostic and make the swap a
single-file change.**

- `LLM_PROVIDER` selects the adapter. `scripted` is the default and the only
  registered adapter (`_KNOWN_PROVIDERS`).
- An unknown provider raises `ModelError` naming the known ones. It does not
  silently fall back — a deployment that believes it is calling a real model
  and is not would be worse than one that refuses to start.
- `ScriptedModel` replays recorded responses in order and raises when the list
  is exhausted. No silent empty-string fallback: a test must not pass on a
  blank answer.
- Any future provider implements `Model`, is added to `_KNOWN_PROVIDERS`, and
  is constructed inside `get_model()`. Nothing above `get_model()` changes.

### Contract for a future adapter

An adapter is accepted when it:

1. Satisfies the `Model` protocol (`isinstance(m, Model)` holds — the protocol
   is `runtime_checkable`).
2. Returns an instance of `response_schema` when one is given, or raises
   `ModelError` if the provider's reply cannot be parsed into it.
3. Raises `ModelError` for unrecoverable failures (bad credentials, unknown
   model). Retryable failures (rate limits, timeouts) are retried by the caller,
   which raises `ModelError` only after giving up.
4. Does not leak provider exceptions. `agent/analyst.py:think` catches
   `Exception` and converts any escape into a `legal-counsel-review` route, so a
   provider outage degrades the answer rather than the service.

For OpenAI-compatible endpoints, items 1–3 mean one module-level class and one
dictionary entry. Gemini differs in request shape but not in contract.

## Consequences

**Good**

- The full test suite runs offline in under three seconds.
- The demo cannot fail on someone else's infrastructure.
- The hard part — the bounded loop, tool selection, verification — is tested
  against a deterministic model, so failures point at our code, not the LLM's
  sampling.
- Swapping providers later is additive: no call site changes.

**Bad**

- The shipped service cannot answer questions it was not scripted for. With
  `ANALYST_MODE=real` and no scripted responses, every question escalates to
  legal counsel. This is honest but limited.
- Nothing here exercises a real model's adherence to the `Step` schema. Token
  drift, refusal text, and malformed tool-call JSON are untested until an
  adapter exists.

**Neutral**

- `.env.example` previously advertised `LLM_PROVIDER=openai`, which raised
  `ModelError`. That file is corrected to match this decision.

## Alternatives considered

- **OpenAI adapter now.** Best answer quality for the demo, but makes the
  graded run depend on a key and a network. Rejected as a default; still the
  first adapter to add.
- **Adapters for all three providers up front.** Three untested code paths
  instead of one tested one. Deferred until at least one has a consumer.
- **No abstraction, call the SDK directly in `think`.** Cheapest today,
  untestable offline, and makes the seam a rewrite later.