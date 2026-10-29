# ADR 0002: OpenAI-compatible model adapter and multi-provider registry

- **Status:** Accepted
- **Date:** 2026-10-29
- **Deciders:** project owner
- **Supersedes:** ADR 0001 (clause: "Ship scripted-only. Keep the seam provider-agnostic...")

## Context

ADR 0001 established the single-method `Model` protocol (`complete`) and shipped
with `ScriptedModel` as the sole registered adapter to guarantee reproducible,
offline-first testing and bounded execution without network dependencies.

To support live demonstrations, live model evaluations, and self-hosted inference
servers (such as vLLM, llama.cpp, or OpenAI endpoints), a production adapter was
required that satisfies the provider contract without changing call sites above
`get_model()`.

## Decision

Supersede the scripted-only restriction from ADR 0001. Formally introduce and
register `OpenAIModel` alongside `ScriptedModel` in `agent/model._KNOWN_PROVIDERS`:

1. `_KNOWN_PROVIDERS = ("scripted", "openai")`.
2. `OpenAIModel` talks to any standard `/chat/completions` HTTP endpoint
   configured through environment variables:
   - `LLM_BASE_URL` (e.g. `https://api.openai.com/v1`)
   - `LLM_API_KEY` (or `OPENAI_API_KEY`)
   - `LLM_MODEL` (model identifier)
   - `LLM_TIMEOUT_S` (per-request timeout, default 60s)
3. Structured output is obtained via `response_format={"type": "json_object"}`
   and prompt schema injection, validated back into the target Pydantic model.
4. `LLM_PROVIDER` continues to default to `scripted`, ensuring clean clones
   and local test runs remain completely offline and hermetic unless explicitly
   opted into live execution.

## Consequences

**Good**
- Resolves the architectural mismatch between ADR 0001 and the codebase.
- Enables live execution against any compliant inference service.
- Preserves the fast, hermetic test suite (180+ tests offline in ~3s).
- Keeps retry, backoff, and circuit breaker responsibilities in the caller (A5).

**Neutral**
- Live tests are guarded by environment checks (`LLM_LIVE_URL`) and skipped during
  standard CI runs.

