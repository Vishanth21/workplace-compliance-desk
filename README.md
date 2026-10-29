# Workplace Compliance Advisory Desk

Answers Indian labour-law questions from a statute corpus — and refuses to return
any answer it cannot verify against a provision it actually retrieved.

The core move is that **the system would rather escalate than guess.** Every
drafted answer is checked claim-by-claim against the retrieved passages; if a
claim cites a provision that was not retrieved, or a passage does not support
it, the run is routed to legal counsel instead of returning text.

---

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Then open **<http://localhost:8000/ui/>**.

From a checkout without Docker:

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The web client is served by the API process itself at `/ui/`; `/` redirects
there. There is no separate frontend build and no second server.

---

## The flagship case (§4.1)

Paste this into the web client:

> We employ 50 people in an IT company in Karnataka. An employee engaged on a
> written fixed-term contract of 18 months completed the contract. Is gratuity
> payable?

with headcount `50`, industry `Information Technology`, state `Karnataka`, and
**has fixed-term employees** checked.

A correct run returns:

| Field | Expected |
|---|---|
| `route` | `answer-with-citation` |
| `verification.passed` | `true` |
| citations | `css2020-s53-proviso_6` **and** `css2020-s53-proviso_2` |

The two provisos are the point of the exercise. The main clause of section 53
sets a five-year continuous-service threshold; the sixth proviso waives it for
fixed-term employment, and the second proviso requires pro-rata payment. An
answer citing only the main clause is wrong for this question.

> **Which mode produces this?** The default `ANALYST_MODE=stub`. The stub
> returns this answer for *any* question, so it demonstrates the routing,
> citation and verification path — not that the question was understood. For
> the latter, `ANALYST_MODE=real` is required, and that needs a model that is
> not yet implemented (see [Configuration](#configuration)); today it escalates.

You can also drive it from the shell:

```bash
curl -s -X POST http://localhost:8000/runs \
  -H 'Content-Type: application/json' \
  -d @sample.json
# -> 202 {"run_id": "...", "correlation_id": "...", "status_url": "/runs/<id>"}

curl -s http://localhost:8000/runs/<id>
```

`verify_api.py` runs five end-to-end checks (health, submit, poll, malformed
rejection, metrics) against a server on port 8000:

```bash
uv run python verify_api.py
```

---

## Configuration

Two independent environment variables select behaviour.

### `ANALYST_MODE` — which analyst runs

| Value | Behaviour |
|---|---|
| `stub` (default) | Replays a fixed flagship advisory for **every** question. Keeps the test suite deterministic. |
| `real` | Runs the bounded LangGraph loop over the corpus tools. |

An unrecognised value raises at startup rather than falling back to the stub —
a deployment that believes it runs the real agent should not silently run a
replay.

### `LLM_PROVIDER` — which model backs the real analyst

| Value | Behaviour |
|---|---|
| `scripted` (default) | Replays a JSON list from `LLM_SCRIPTED_RESPONSES`. Used for tests and the flagship gate. |
| `openai` | Any OpenAI-compatible chat-completions endpoint. Set `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`. |

**Gemini is not implemented.** `get_model()` raises `ModelError` for any other
provider (see [ADR 0001](docs/adr/0001-model-provider-seam.md)).

Consequence worth stating plainly: **with `ANALYST_MODE=real` and the default
scripted model, every live question escalates to legal counsel.** The scripted
model replays a fixed list and has no answer for an arbitrary question, so the
loop ends without one and routes to counsel. That is the intended safe failure,
and `tests/test_analyst_selection.py` pins it.

Running against a real endpoint:

```bash
LLM_PROVIDER=openai \
LLM_BASE_URL=https://api.example.com/v1 \
LLM_MODEL=your-model-name \
LLM_API_KEY=sk-... \
ANALYST_MODE=real \
uv run uvicorn app.main:app --port 8000
```

The adapter's request shape, schema-validation and retry behaviour are covered
by `tests/test_openai_adapter.py`. Its two live-endpoint tests skip unless
`LLM_LIVE_URL` is set.

**This path has been exercised against a live OpenAI-compatible endpoint**,
not just mocked. What that run established:

- The endpoint honours `response_format: json_object` plus the `Verdict` schema,
  which is what makes the claim-vs-passage check possible at all.
- The flagship question is answered from genuine reasoning, not replay: the model
  searched, found the main provision, then reached *both* deciding provisos
  (`proviso_6` removing the five-year bar, `proviso_2` requiring pro-rata) and the
  Verifier confirmed the passages support the claim.
- Out-of-corpus questions are refused. The Analyst over-claims
  (`answer-with-citation`) and the **Verifier overrides it** to
  `legal-counsel-review` — the safety net, not the model, is what refuses.

That last point is worth dwelling on: the Analyst being overconfident is expected.
The Verifier is the component that must not be, and this is the evidence it isn't.

To run them against a live endpoint:

```bash
LLM_LIVE_URL=https://api.example.com/v1 \
OPENAI_API_KEY=sk-... \
LLM_MODEL=your-model-name \
uv run pytest tests/test_openai_adapter.py -k live
```

---

## What is actually implemented

Be precise about this — the honest version is more useful than an aspirational one.

**Real and working:**

- **Contracts** (`contracts/`) — REST schema, `AnalystResult`, route enum, status enum. Shared by both agents.
- **API** (`app/main.py`) — async submit/poll (`POST /runs` → 202, `GET /runs/{id}`), ingestion validation with field-level errors, quarantine of rejected requests, `/health`, `/metrics`.
- **Queue** (`app/queue.py`) — bounded in-process queue with a fixed worker pool. The queue carries run ids, not state.
- **Verifier** (`agent/verifier.py`) — citation resolution (every cited id must be a retrieved chunk), claim-vs-passage support check, and hard triggers that force escalation.
- **Analyst** (`agent/analyst.py`) — LangGraph loop: classify → search → derive a second query → draft with citations, with a bounded iteration count.
- **Model seam** (`agent/model.py`) — provider protocol, scripted implementation, an OpenAI-compatible adapter, clear error for unknown providers.
- **Injection guard** (`app/guard.py`) — instruction-override attempts are quarantined at ingestion with a `prompt_injection` field error, before any run is created.
- **Eval set** (`tests/eval_scenarios.py`, `tests/test_eval.py`) — 20 scenarios through the real loop and the deterministic verifier.
- **Web client** (`app/static/index.html`) — A8. Submit, poll, render route/answer/citations/escalation.
- **Corpus pipeline code** (`index/build.py`, `index/chunker.py`, `index/embedder.py`) — chunking, local `all-MiniLM-L6-v2` embeddings, idempotent Chroma build with a build fingerprint.

**Present but not wired into the running service:**

- **Retrieval is in-memory, not semantic.** `index/retrieve.py` answers from a hardcoded `CORPUS_CHUNKS` list. `corpus_search` is a fixed ranking over that list, not a vector search. The Chroma index that `index/build.py` builds is never queried at runtime, and nothing in `agent/` or `app/` imports `chromadb`. Treat `index/build.py` as the intended pipeline, not the active one.

**Cut deliberately** (see PLAN.md for the reasoning):

- Trace visualisation in Jaeger/Grafana. Run reconstruction is via structured logs with a correlation ID.
- Locust load testing and measured p95. The SLO is declared, not measured.
- `--scale` measurement, response caching, the feedback endpoint, toxicity detection.
- Gemini support, and any live-model verification (see above).

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

The analyst never decides its own answer is safe. The verifier does, and it can
overrule the analyst in either direction — pass a supported answer through, or
force an escalation.

---

## Tests

```bash
uv run pytest
```

184 tests, 7 skipped. The skips are the cases that need a live model
(`OPENAI_API_KEY`, and `LLM_LIVE_URL` for the two endpoint tests). Coverage by module:

| File | What it pins |
|---|---|
| `tests/test_api.py` | Endpoints, validation, injection quarantine, queue, verification, metrics, the UI mount |
| `tests/test_verifier.py` | Citation resolution, claim support, hard triggers |
| `tests/test_analyst.py` | The bounded loop, derived second query, model failure → escalation |
| `tests/test_analyst_selection.py` | `ANALYST_MODE` selection and the stub/real difference |
| `tests/test_model.py` | Provider protocol and the unknown-provider error |
| `tests/test_openai_adapter.py` | Request shape, config errors, malformed JSON, schema retry, live endpoint (skipped) |
| `tests/test_guard.py` | Injection patterns, and the benign phrasings they must not catch |
| `tests/test_eval.py` | The 20-scenario eval set — see below |

### The eval set

`tests/eval_scenarios.py` holds twenty scenarios: six answerable from section 53,
four flagship-shaped, four outside the corpus that must escalate, three
repealed-law traps, and three prompt-injection attempts quarantined at
ingestion. Each carries its own scripted model transcript, so the LangGraph
loop, tool dispatch, citation resolution and deterministic verifier all run
for real — only the model's judgement is scripted.

It is a **routing and grounding benchmark, not a legal-accuracy one.** The
corpus is eight chunks, so most of the variation is in question phrasing and
expected route. Scenarios in `ESCALATE` are expected to be hard; if they start
passing trivially, the verifier has stopped working.

The flagship loop can be exercised without the API:

```bash
uv run python -m agent.analyst --scripted
```

---

## Repository layout

```
agent/        analyst (LangGraph), verifier, model seam, tools
app/          FastAPI service, queue, store, pipeline wiring, web client
contracts/    shared schemas — the interface both agents agree on
index/        chunker, embedder, Chroma build, retrieval
corpus/       source PDFs (Code on Social Security 2020, POSH 2013, Gratuity Act 1972)
docs/         brief, glossary, ADRs
tests/        the suite
```

The Payment of Gratuity Act 1972 sits in the corpus **deliberately**, marked
`repealed`. It is what most material online still describes, and the flagship
case depends on the contrast between it and the 2020 Code.

---

## Known limitations

1. **The live model path is verified by hand, not by CI.** The `openai` adapter
   has been run against a real OpenAI-compatible endpoint and the flagship,
   escalation and repeal-trap cases all behaved correctly (see the
   `LLM_PROVIDER` section). But the live tests skip unless `LLM_LIVE_URL` is
   set, so nothing in `uv run pytest` exercises them — a regression in the
   adapter or in the endpoint's JSON-mode behaviour would not be caught
   automatically. This is also the only configuration where the model is
   actually reasoning rather than replaying.
2. **Verification checks soundness, not completeness.** The Verifier asks
   whether the cited passages support the claim. It cannot ask whether the
   agent retrieved *enough* — it never sees the corpus, only the chunks the
   agent cited. Consequence, measured against a live model on the flagship
   question: if the agent fetches `main` and `proviso_1` but never
   `proviso_6`, it can answer "no gratuity is payable, five years' service is
   required" and **pass verification**, because those two passages really do
   say that. The answer is supported and still wrong. On the same question
   across repeated live runs, one run produced both deciding provisos and
   passed, one produced the incomplete pair and passed, and one was refused by
   the claim-vs-passage check. So `passed: true` means "the reasoning is
   grounded in what it read", **not** "the answer is correct".
   The structural fix is to make completeness checkable — e.g. requiring that
   a section's deciding provisos be retrieved before an answer may cite that
   section — and it is not implemented.
2. **Retrieval is not semantic.** See above. The corpus tool's ranking is
   hand-tuned so that a generic search misses the deciding provisos — which is
   what makes `lookup_clause` load-bearing, but it is not a retrieval system.
3. **State is in-process.** Runs, the queue, and metrics live in memory. Restart
   loses them, and the service does not scale past one process.
4. **`status` on chunks is hand-assigned.** Nothing derives in-force/repealed
   from the documents themselves.