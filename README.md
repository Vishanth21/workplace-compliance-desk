# Workplace Compliance Advisory Desk

Answers Indian labour-law questions from a statute corpus. Every drafted answer is verified claim-by-claim against the passages retrieved for that run; answers that fail verification are routed to legal counsel instead of being returned.

---

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Open <http://localhost:8000/ui/>.

Without Docker:

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The web client is served by the API process at `/ui/`; `/` redirects there. There is no separate frontend build.

---

## Flagship case (§4.1)

Request:

> We employ 50 people in an IT company in Karnataka. An employee engaged on a
> written fixed-term contract of 18 months completed the contract. Is gratuity
> payable?

Request fields: headcount `50`, industry `Information Technology`, state `Karnataka`, has fixed-term employees `true`.

Expected result:

| Field | Value |
|---|---|
| `route` | `answer-with-citation` |
| `verification.passed` | `true` |
| citations | `css2020-s53-proviso_6` and `css2020-s53-proviso_2` |

The main clause of section 53 sets a five-year continuous-service threshold. The sixth proviso waives that threshold for fixed-term employment; the second proviso requires pro-rata payment. An answer citing only the main clause is incorrect for this question.

With the default `ANALYST_MODE=stub`, the same advisory is returned for every question. This exercises the routing, citation and verification path. Reasoning about the specific question requires `ANALYST_MODE=real` with a configured model.

From the shell:

```bash
curl -s -X POST http://localhost:8000/runs \
  -H 'Content-Type: application/json' \
  -d @sample.json
# -> 202 {"run_id": "...", "correlation_id": "...", "status_url": "/runs/<id>"}

curl -s http://localhost:8000/runs/<id>
```

`verify_api.py` runs five end-to-end checks (health, submit, poll, malformed rejection, metrics) against a server on port 8000:

```bash
uv run python verify_api.py
```

---

## Configuration

Two environment variables select behaviour.

### `ANALYST_MODE`

| Value | Behaviour |
|---|---|
| `stub` (default) | Returns a fixed advisory for every question. Keeps the test suite deterministic. |
| `real` | Runs the bounded LangGraph loop over the corpus tools. |

An unrecognised value raises at startup.

### `LLM_PROVIDER`

| Value | Behaviour |
|---|---|
| `scripted` (default) | Replays a JSON list from `LLM_SCRIPTED_RESPONSES`. Used by tests. |
| `openai` | Any OpenAI-compatible chat-completions endpoint. Requires `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`. |

Any other provider raises `ModelError`. See [ADR 0001](docs/adr/0001-model-provider-seam.md) and [ADR 0002](docs/adr/0002-openai-compatible-model-adapter.md).

With `ANALYST_MODE=real` and the default scripted model, every live question escalates to legal counsel: the scripted model replays a fixed list and produces no answer for an arbitrary question. `tests/test_analyst_selection.py` pins this behaviour.

Running against a live endpoint:

```bash
LLM_PROVIDER=openai \
LLM_BASE_URL=https://api.example.com/v1 \
LLM_MODEL=your-model-name \
LLM_API_KEY=sk-... \
ANALYST_MODE=real \
uv run uvicorn app.main:app --port 8000
```

Request shape, schema validation and retry behaviour are covered by `tests/test_openai_adapter.py`. Its two live-endpoint tests run only when `LLM_LIVE_URL` is set:

```bash
LLM_LIVE_URL=https://api.example.com/v1 \
OPENAI_API_KEY=sk-... \
LLM_MODEL=your-model-name \
uv run pytest tests/test_openai_adapter.py -k live
```

---

## Implementation status

### Implemented

- **Contracts** (`contracts/`) — REST schema, `AnalystResult`, route enum, status enum.
- **API** (`app/main.py`) — async submit/poll (`POST /runs` → 202, `GET /runs/{id}`), ingestion validation with field-level errors, quarantine of rejected requests, `/health`, `/metrics`.
- **Queue** (`app/queue.py`) — bounded in-process queue with a fixed worker pool. Carries run ids.
- **Verifier** (`agent/verifier.py`) — citation resolution, claim-vs-passage support check, hard triggers that force escalation.
- **Analyst** (`agent/analyst.py`) — LangGraph loop: classify → search → derive a second query → draft with citations, bounded iteration count.
- **Model seam** (`agent/model.py`) — provider protocol, scripted implementation, OpenAI-compatible adapter, error on unknown providers.
- **Model resilience** (`agent/resilience.py`) — exponential backoff retry on transient errors, three-state circuit breaker (`CLOSED`, `OPEN`, `HALF_OPEN`), fallback to `legal-counsel-review` on upstream failure.
- **Injection guard** (`app/guard.py`) — instruction-override attempts are quarantined at ingestion with a `prompt_injection` field error, before a run is created.
- **Eval set** (`tests/eval_scenarios.py`, `tests/test_eval.py`) — 20 scenarios through the loop and the deterministic verifier.
- **Web client** (`app/static/index.html`) — submit, poll, render route/answer/citations/escalation.
- **Corpus pipeline and retrieval** (`index/build.py`, `index/chunker.py`, `index/embedder.py`, `index/retrieve.py`) — 621 statutory chunks embedded with `all-MiniLM-L6-v2` into a persistent Chroma collection. `corpus_search` and `lookup_clause` query Chroma at runtime.
- **Compose stack** — four services: API, background worker, queue broker, model stub.

### Not implemented

- Trace visualisation in Jaeger or Grafana. Run reconstruction uses structured logs with a correlation ID.
- Locust load testing and measured p95. The declared SLO is p95 end-to-end latency ≤ 2,500 ms under nominal load.
- `--scale` measurement.
- Response caching.
- Feedback endpoint.
- Toxicity detection.
- Gemini support.
- Automated live-model verification; the live tests are opt-in.

---

## Architecture

```
POST /runs ──► validation ──► quarantine (400, no run)
                  │
                  ▼
             run id ──► queue ──► worker pool
                                     │
                                     ▼
                              Analyst (LangGraph)
                              classify → search → derive query → draft
                                     │
                                     ▼
                              AnalystResult
                            (draft, claims, retrieved chunks)
                                     │
                                     ▼
                                 Verifier
                        citation resolution · claim-vs-passage
                              · hard triggers
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
                answer-with-citation        legal-counsel-review
                        │                         │
                        └──────────► GET /runs/{id} ◄──────────┘
```

The analyst produces a draft. The verifier decides whether it is returned or escalated, and can overrule the analyst in either direction.

---

## Tests

```bash
uv run pytest
```

192 tests passed, 7 skipped. The skips are 5 eval scenarios whose verification outcome is intentionally unspecified, and 2 endpoint tests in `test_openai_adapter.py` that require `LLM_LIVE_URL`.

| File | Coverage |
|---|---|
| `tests/test_api.py` | Endpoints, validation, injection quarantine, queue, verification, metrics, UI mount |
| `tests/test_circuit_breaker.py` | Backoff retries, three-state circuit breaker, fast-fail on OPEN, manual trip |
| `tests/test_verifier.py` | Citation resolution, claim support, hard triggers |
| `tests/test_analyst.py` | Bounded loop, derived second query, model failure → escalation |
| `tests/test_analyst_selection.py` | `ANALYST_MODE` selection and the stub/real difference |
| `tests/test_model.py` | Provider protocol and unknown-provider error |
| `tests/test_openai_adapter.py` | Request shape, config errors, malformed JSON, schema retry, live endpoint |
| `tests/test_guard.py` | Injection patterns and the benign phrasings they must not catch |
| `tests/test_eval.py` | The 20-scenario eval set |

### Eval set

`tests/eval_scenarios.py` defines 20 scenarios: six answerable from section 53, four flagship-shaped, four outside the corpus that must escalate, three repealed-law traps, and three prompt-injection attempts quarantined at ingestion. Each carries a scripted model transcript, so the LangGraph loop, tool dispatch, citation resolution and deterministic verifier execute for real.

The set benchmarks routing and grounding. It does not measure legal accuracy. Most variation is in question phrasing and expected route; scenarios in `ESCALATE` are expected to fail verification.

The flagship loop can be exercised without the API:

```bash
uv run python -m agent.analyst --scripted
```

---

## Repository layout

```
agent/        analyst (LangGraph), verifier, model seam, resilience, tools
app/          FastAPI service, queue, store, pipeline wiring, web client
contracts/    shared schemas
index/        chunker, embedder, Chroma build, retrieval
corpus/       source PDFs (Code on Social Security 2020, POSH 2013, Gratuity Act 1972)
docs/         brief, glossary, runbook, incident log, ADRs
tests/        the suite
eval/         prompt-variant comparison
load_tests/   Locust load test
```

The Payment of Gratuity Act 1972 is included in the corpus marked `repealed`. The flagship case depends on the contrast between it and the 2020 Code.

---

## Known limitations

1. **Verification checks soundness, not completeness.** The verifier asks whether cited passages support the claim. It does not ask whether the agent retrieved enough — it sees only the chunks the agent cited, not the corpus. If the agent retrieves `main` and `proviso_1` but not `proviso_6`, an answer stating that five years' service is required passes verification, because those passages do support it. `verification.passed: true` means the reasoning is grounded in the retrieved text, not that the answer is correct. A structural fix — requiring a section's deciding provisos to be retrieved before an answer may cite that section — is not implemented.

2. **State is in-process.** Runs, queue and metrics are held in memory. A restart loses them, and the service does not scale past one process.

3. **Chunk `status` is hand-assigned.** In-force/repealed status is declared in the build configuration, not derived from the documents.

4. **The live model path is not covered by CI.** The live tests require `LLM_LIVE_URL` and skip otherwise, so a regression in the adapter or in the endpoint's JSON-mode behaviour would not be caught automatically.