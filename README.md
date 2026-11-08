# Workplace Compliance Advisory Desk

Answers Indian labour-law questions from a statute corpus. Every drafted answer is verified claim-by-claim against the passages retrieved for that run; answers that fail verification are routed to legal counsel instead of being returned.

The service is asynchronous (submit → poll), fully traced, and instrumented. Each run is reconstructible from its run id: every tool call, query, passage, prompt version, per-step latency and cost, and the verification outcome.

---

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

| URL | What |
|---|---|
| <http://localhost:8000/ui/> | Web client |
| <http://localhost:8000/docs> | OpenAPI schema |
| <http://localhost:16686> | Jaeger UI (traces) |
| <http://localhost:9090> | Prometheus |
| <http://localhost:3000> | Grafana (`admin` / `admin`) |
| <http://localhost:8080> | OpenAI-compatible model stub |

Without Docker:

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The web client is served by the API process at `/ui/`; `/` redirects there. There is no separate frontend build.

> **Retrieval needs the Chroma index.** `index/store/` is generated and git-ignored, and `docker compose up` does not build or mount it. Run `uv run python -m index.build` once in a local checkout before starting the service.
>
> Building the index needs `pdftotext` (poppler) and the three corpus PDFs. Neither is available in the container: the image does not install poppler, `corpus/` is not copied, `.dockerignore` excludes `index/store/`, and `docker-compose.yml` mounts no index into `api` or `worker`. A containerised run therefore always takes the fallback path below.
>
> Without an index, `index/retrieve.py` serves eight hand-written `CORPUS_CHUNKS` (seven from `css2020`, one from `pga1972`). Enough for the flagship case and the test suite, not enough for general questions. `/metrics` shows the difference through `compliance_index_chunks`: `8` means the fallback is in use, `618` means a built index is loaded. The fallback state is carried in `get_index_metadata()["status"]` but is not exported as its own metric.

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

curl -s http://localhost:8000/runs/<id>          # poll the result
curl -s http://localhost:8000/runs/<id>/trace    # full run reconstruction
```

`verify_api.py` runs five end-to-end checks (health, submit, poll, malformed rejection, metrics) against a server on port 8000:

```bash
uv run python verify_api.py
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/runs` | Submit a question. `202` with `run_id`, `correlation_id`, `status_url`. `400` on validation failure or injection quarantine. |
| `GET` | `/runs/{id}` | Poll. `404` if unknown; `answer` stays `null` until `succeeded`. |
| `GET` | `/runs/{id}/trace` | Full trajectory: steps, tools, arguments, passages, latency, tokens, cost, verification. |
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/metrics` | Prometheus plaintext. |
| `GET` | `/ui/` | Single-page client. |

Rejected requests return `400` with per-field `errors`, and are quarantined in-process for inspection.

---

## Configuration

All values come from the environment (see [`.env.example`](.env.example)).

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

### `PROMPT_VERSION`

Selects a prompt artifact from `agent/prompts.py`. The chosen version is recorded on every `RunTrace`.

| Value | Behaviour |
|---|---|
| `analyst-1` (default) | Baseline analyst. |
| `analyst-2` | Proviso-aware: inspects deciding provisos before citing a threshold section, and guards against citing repealed law. |

See [prompt comparison](docs/prompt-comparison.md). Under the scripted model the two variants measure identically; the comparison cannot show an instruction-level difference.

### `PIPELINE_DELAY_SECONDS`

Simulated model latency for the stub path (default `0.2`). The load test runs at `0.15`.

Running against a live endpoint:

```bash
LLM_PROVIDER=openai \
LLM_BASE_URL=https://api.example.com/v1 \
LLM_MODEL=your-model-name \
LLM_API_KEY=sk-... \
ANALYST_MODE=real \
PROMPT_VERSION=analyst-2 \
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

- **Contracts** (`contracts/`) — REST schema, `AnalystResult`, `RunTrace`/`StepRecord`, route and status enums.
- **API** (`app/main.py`) — async submit/poll, `/runs/{id}/trace`, ingestion validation with field-level errors, quarantine of rejected requests, `/health`, `/metrics`.
- **Queue** (`app/queue.py`) — bounded in-process asyncio queue with a fixed worker pool (`WORKER_COUNT = 2`). Carries run ids, not state.
- **Worker process** (`app/worker.py`) — standalone background pool entrypoint used by the compose `worker` service.
- **Verifier** (`agent/verifier.py`) — citation resolution, claim-vs-passage support check, hard triggers that force escalation.
- **Analyst** (`agent/analyst.py`) — LangGraph loop: classify → search → derive a second query → draft with citations, bounded iteration count.
- **Prompts** (`agent/prompts.py`) — two immutable, versioned prompt artifacts selected by `PROMPT_VERSION`.
- **Model seam** (`agent/model.py`) — provider protocol, scripted implementation, OpenAI-compatible adapter, error on unknown providers.
- **Model resilience** (`agent/resilience.py`) — exponential backoff retry on transient errors, three-state circuit breaker (`CLOSED`, `OPEN`, `HALF_OPEN`), fallback to `legal-counsel-review` on upstream failure.
- **Injection guard** (`app/guard.py`) — instruction-override attempts are quarantined at ingestion with a `prompt_injection` field error, before a run is created.
- **Observability** (`agent/telemetry.py`) — OpenTelemetry spans exported over OTLP to Jaeger, plus `/metrics` counters, queue depth, circuit-breaker state, p50/p95 run latency and trace counts, scraped by Prometheus every 5 s. A provisioned Grafana overview dashboard lives in `deploy/grafana/dashboards/`.
- **Trace reconstruction** (`app/pipeline.py` → `RunTrace`) — per-step action, tool, arguments, passage ids, latency, token and cost estimates, attached to the run and served by `/runs/{id}/trace`.
- **Eval set** (`tests/eval_scenarios.py`, `tests/test_eval.py`) — 17 scenarios through the loop and the deterministic verifier, plus 3 injection cases.
- **Prompt comparison** (`eval/compare_prompts.py`, [report](docs/prompt-comparison.md)) — both variants scored on the same 17 scenarios. Measured on this checkout the two variants are at parity: 100% routing, 100% citation grounding, 66.7% flagship-proviso retrieval, 2.12 average steps. Prompt version is recorded on every `RunTrace`; the instruction-level difference between the variants is not observable under the scripted model.
- **Load testing** (`load_tests/run_load_test.py`, Locust) — 2/5/10/20 users, [report](docs/load-test-report.md). In the published run the p95 SLO ≤ 2,500 ms holds to 10 users and 20 users saturates at 4.1 s. The report labels these figures *pre-computed* against the stub, not re-measured on this checkout; see limitation 5.
- **Web client** (`app/static/index.html`) — submit, poll, render route/answer/citations/escalation.
- **Corpus pipeline and retrieval** (`index/build.py`, `index/chunker.py`, `index/embedder.py`, `index/retrieve.py`) — 618 statutory chunks (css2020 470, posh 72, pga1972 76) embedded with `all-MiniLM-L6-v2` into a persistent Chroma collection, fingerprinted by a corpus hash so a rebuild is idempotent. `corpus_search` and `lookup_clause` query Chroma at runtime; `/metrics` exposes `compliance_index_chunks` and `compliance_index_freshness_seconds`.
- **Operations docs** — [runbook](docs/runbook.md) for upstream outage and circuit-breaker recovery, [incident log](docs/incident-log.md), [ADRs](docs/adr/).
- **Compose stack** — seven services: `api`, `worker`, `queue` (Redis), `model-stub`, `jaeger`, `prometheus`, `grafana`.

### Not implemented

- Response caching.
- Feedback endpoint.
- Toxicity detection.
- Gemini support.
- Automated live-model verification; the live tests are opt-in.
- Horizontal worker scaling across containers — `WORKER_COUNT` is a per-process constant.

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
                                      + /runs/{id}/trace
```

The analyst produces a draft. The verifier decides whether it is returned or escalated, and can overrule the analyst in either direction. Every step of the loop is recorded before the result is published.

---

## Tests

```bash
uv run pytest
```

198 passed, 7 skipped. The skips are 5 eval scenarios whose verification outcome is intentionally unspecified, and 2 endpoint tests in `test_openai_adapter.py` that require `LLM_LIVE_URL`.

| File | Coverage |
|---|---|
| `tests/test_api.py` | Endpoints, validation, injection quarantine, queue, verification, metrics, trace endpoint, UI mount |
| `tests/test_circuit_breaker.py` | Backoff retries, three-state circuit breaker, fast-fail on OPEN, manual trip |
| `tests/test_verifier.py` | Citation resolution, claim support, hard triggers |
| `tests/test_analyst.py` | Bounded loop, derived second query, model failure → escalation |
| `tests/test_analyst_selection.py` | `ANALYST_MODE` selection and the stub/real difference |
| `tests/test_model.py` | Provider protocol and unknown-provider error |
| `tests/test_openai_adapter.py` | Request shape, config errors, malformed JSON, schema retry, live endpoint |
| `tests/test_guard.py` | Injection patterns and the benign phrasings they must not catch |
| `tests/test_eval.py` | The 17-scenario eval set plus the injection cases |
| `tests/test_prompt_comparison.py` | Prompt version recording and variant selection |

### Eval set

`tests/eval_scenarios.py` defines 17 scenarios in `ALL_SCENARIOS`: six answerable from section 53, four flagship-shaped, four outside the corpus that must escalate, and three repealed-law traps. Three further prompt-injection questions live in `INJECTION_QUESTIONS` and are quarantined at ingestion, before a scenario is built. Each scenario carries a scripted model transcript, so the LangGraph loop, tool dispatch, citation resolution and deterministic verifier execute for real.

The set benchmarks routing and grounding. It does not measure legal accuracy. Most variation is in question phrasing and expected route; scenarios in `ESCALATE` are expected to fail verification. Five of the 17 leave `expect_verified` unset — the four escalations and `ans-answer-without-tools` — so `test_scenario_verification` skips them and their verification outcome is not asserted.

The flagship loop can be exercised without the API:

```bash
uv run python -m agent.analyst --scripted
```

### Prompt comparison

```bash
uv run python -m eval.compare_prompts
```

### Load test

```bash
# terminal 1
PIPELINE_DELAY_SECONDS=0.15 uv run uvicorn app.main:app --port 8000

# terminal 2
uv run python load_tests/run_load_test.py
```

---

## Repository layout

```
agent/        analyst (LangGraph), verifier, model seam, resilience, prompts, telemetry, tools
app/          FastAPI service, queue, store, worker, pipeline wiring, injection guard, web client
contracts/    shared schemas (API, analyst result, route, trace)
index/        chunker, embedder, Chroma build, retrieval
corpus/       source PDFs (Code on Social Security 2020, POSH 2013, Gratuity Act 1972)
deploy/       Prometheus scrape config, Grafana provisioning and dashboard
docs/         brief, glossary, runbook, incident log, load-test and prompt reports, ADRs
tests/        the suite and the eval scenarios
eval/         prompt-variant comparison
load_tests/   Locust file and automated runner
```

The Payment of Gratuity Act 1972 is included in the corpus marked `repealed`. The flagship case depends on the contrast between it and the 2020 Code.

---

## Known limitations

1. **Verification checks soundness, not completeness.** The verifier asks whether cited passages support the claim. It does not ask whether the agent retrieved enough — it sees only the chunks the agent cited, not the corpus. If the agent retrieves `main` and `proviso_1` but not `proviso_6`, an answer stating that five years' service is required passes verification, because those passages do support it. `verification.passed: true` means the reasoning is grounded in the retrieved text, not that the answer is correct. A structural fix — requiring a section's deciding provisos to be retrieved before an answer may cite that section — is not implemented. `analyst-2` narrows the gap by instruction, not by construction.

2. **State is in-process, and the `worker` service is a second, disconnected pool.** Runs, queue, metrics and traces are held in memory; a restart loses them. The compose stack ships a Redis service and sets `REDIS_URL`, but no Python module reads it: the queue is an in-process asyncio queue. The API process starts its own `WORKER_COUNT` consumers at startup (`app/main.py:56`), and the `worker` service starts another pool over its own empty queue (`app/worker.py:18`), so jobs submitted through the API never reach the worker container. It is a placeholder for the multi-container split, not a working one.

3. **Chunk `status` is hand-assigned.** In-force/repealed status is declared in the build configuration, not derived from the documents.

4. **The live model path is not covered by CI.** The live tests require `LLM_LIVE_URL` and skip otherwise, so a regression in the adapter or in the endpoint's JSON-mode behaviour would not be caught automatically.

5. **Load-test numbers are indicative, not reproduced.** The report's own table is labelled *pre-computed*, and it was produced against the in-process stub (`PIPELINE_DELAY_SECONDS=0.15`) rather than a live model. `load_tests/run_load_test.py` exists to regenerate it, but the published figures have not been re-measured on this checkout. Under real inference latency the SLO would be re-declared and the saturation point would move.