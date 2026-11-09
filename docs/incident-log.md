# Incident Log — Workplace Compliance Advisory Desk

Record of operational incidents, root cause analyses, and remediations (Requirement D8).

| ID | Date | Severity | Status |
|---|---|---|---|
| [INC-2026-10-28-01](#inc-2026-10-28-01-upstream-inference-503-outage--circuit-breaker-degradation) | 2026-10-28 | Major | Resolved |
| [INC-2026-11-09-01](#inc-2026-11-09-01-gemini-rejects-replayed-tool-calls-without-thought_signature) | 2026-11-09 | Minor | Resolved |
| [INC-2026-11-09-02](#inc-2026-11-09-02-gemini-json-fences-cause-verifier-false-negative) | 2026-11-09 | Major (verifier) | Resolved |
| [INC-2026-11-09-03](#inc-2026-11-09-03-flagship-41-cites-only-proviso_6-not-proviso_2) | 2026-11-09 | Minor (demo caveat) | Accepted limitation |
| [INC-2026-11-09-04](#inc-2026-11-09-04-runtrace-synthesizes-tool-records-from-retrieved-chunks) | 2026-11-09 | Minor (D5 defect) | Open |
| [INC-2026-11-09-05](#inc-2026-11-09-05-stage-2-verifier-lenient-on-partially-supported-claims) | 2026-11-09 | Minor | Open |

---

## INC-2026-10-28-01: Upstream Inference 503 Outage & Circuit Breaker Degradation

| Attribute | Details |
|---|---|
| **Incident ID** | INC-2026-10-28-01 |
| **Date & Time** | 2026-10-28 14:15:00 UTC to 14:32:00 UTC (Duration: 17 min) |
| **Severity** | Major (Advisory service degraded to legal counsel fallback) |
| **Impacted Services** | `compliance-api`, `compliance-worker`, LLM inference seam |
| **Author** | Operations Team |

### 1. Summary

At 14:15 UTC, the external OpenAI-compatible model provider experienced an unannounced gateway degradation, returning HTTP 503 Service Unavailable to `/chat/completions` requests. The compliance desk resilience layer (A5) executed exponential backoff retries across two attempts, registered consecutive failures, and tripped the circuit breaker OPEN at 14:16 UTC. During the outage, incoming client requests were accepted with HTTP 202 and cleanly escalated to `route: "legal-counsel-review"` with no silent failures, no data corruption, and no hung worker threads. Full automatic recovery occurred at 14:32 UTC.

### 2. Timeline

- **14:15:10 UTC:** First worker task encounters HTTP 503 from `LLM_BASE_URL`. Backoff retry triggers (0.5s, 1.0s).
- **14:15:14 UTC:** Retries exhausted on 3 concurrent runs; `CircuitBreaker` failure count hits threshold (3). Breaker trips OPEN.
- **14:15:15 UTC:** Prometheus metric `compliance_circuit_breaker_open` transitions from 0 to 1. Alert fires.
- **14:16:00 UTC:** On-call engineer responds; confirms via `/metrics` and run logs that circuit breaker is open.
- **14:18:00 UTC:** Direct curl to upstream inference endpoint confirms HTTP 503 from provider proxy layer.
- **14:28:00 UTC:** Upstream provider reports resolution of proxy infrastructure.
- **14:30:00 UTC:** Breaker enters `HALF_OPEN` state on expiry of recovery window. A trial probe request is dispatched by the worker.
- **14:30:05 UTC:** Probe succeeds; circuit breaker automatically transitions to `CLOSED`. Metric `compliance_circuit_breaker_open` returns to 0.
- **14:32:00 UTC:** Follow-up verification run completed successfully with `route: "answer-with-citation"`. Incident closed.

### 3. Root Cause

Third-party model inference provider experienced a transient gateway outage, returning HTTP 503 for all `/chat/completions` payloads.

### 4. What Went Well

- **Zero data loss:** Incoming requests were safely captured, queued, and stored in the status store.
- **Safe failure mode:** No ungrounded, hallucinated, or unverified answers were served. The defined fallback path (graceful degradation to `legal-counsel-review`) protected user compliance safety.
- **Observability:** Correlation IDs accurately grouped the retry sequence and breaker state change. Prometheus metrics clearly flagged `compliance_circuit_breaker_open = 1`.

### 5. Corrective & Preventative Actions

1. **Multi-provider failover:** Implement automatic secondary provider fallback in `agent/model.py` before opening the breaker.
2. **Dashboarding:** Add an alert rule for `compliance_circuit_breaker_open == 1` with a 2-minute persistence window.

---

## INC-2026-11-09-01: Gemini rejects replayed tool calls without thought_signature

| Attribute | Details |
|---|---|
| **Incident ID** | INC-2026-11-09-01 |
| **Date** | 2026-11-09 |
| **Severity** | Minor (temporary; new provider) |
| **Impacted Services** | `agent/model.py` `OpenAIModel` against `generativelanguage.googleapis.com` |
| **Author** | Engineering |

### 1. Summary

Every live run against Gemini 3's OpenAI-compatible endpoint failed at step 2 with `HTTP 400 INVALID_ARGUMENT: Function call is missing a thought_signature in functionCall parts`. The first tool call succeeded (step 1); the replay failed (step 2), so no run could complete a multi-step loop.

### 2. Root Cause

Gemini 2.5/3 "thinking" models attach an opaque `thought_signature` to each `functionCall` in their response. A client that replays the assistant's `tool_calls` on the next turn must return that signature verbatim. The OpenAI chat-completions format has no field for it, so `OpenAIModel` (which round-trips `tool_calls` unmodified) silently dropped it and Gemini rejected the replay.

### 3. Fix

`agent/model.py` now injects Google's documented bypass sentinel (`skip_thought_signature_validator`) via `extra_content.google.thought_signature` on every replayed Gemini tool call, gated to `generativelanguage.googleapis.com`. Non-Gemini endpoints are unaffected.

### 4. Trade-off

Per Google's documentation, the bypass is a last resort: the model loses its reasoning context between turns, which can degrade response quality. Acceptable for the current 3-4 step compliance loop. The correct fix — capture and replay the real signature — also requires `agent/analyst.py` to preserve `extra_content` in the transcript (the assistant turn is currently built from `step.model_dump_json()`, which does not carry it).

### 5. Follow-up

1. Proper signature capture: add `extra_content` to the assistant turn in `agent/analyst.py` `think`, and replay it verbatim on the next call.
2. Add a live test in `tests/test_openai_adapter.py` that exercises a two-turn tool loop against Gemini, not just the single-call schema test.

---

## INC-2026-11-09-02: Gemini JSON fences cause verifier false negative

| Attribute | Details |
|---|---|
| **Incident ID** | INC-2026-11-09-02 |
| **Date** | 2026-11-09 |
| **Severity** | Major (verifier correctness) |
| **Impacted Services** | `agent/model.py` structured parsing, `agent/verifier.py` stage 2 |
| **Author** | Engineering |

### 1. Summary

A live flagship run that should have passed verification was escalated to `legal-counsel-review` with `verification.passed: false`. The server log showed the model's stage-2 verdict was in fact `{"supported": true, ...}`, yet the verifier had parsed it as a failure:

```text
Model claim-vs-passage verification failed with error: model reply is not valid JSON for Verdict: 1 validation error for Verdict
Invalid JSON: trailing characters at line 5 column 1
[input_value='{\n "supported": true,...pro rata basis."\n}\n``']
```

### 2. Root Cause

Gemini 3 wraps JSON-mode replies in Markdown code fences. The fenced content contains valid JSON, but the trailing fence is "trailing characters" to `model_validate_json`. The verifier's exception branch fails closed (`return False, "Model verification error: ..."`), so a positive verdict became `passed: false` and a valid run escalated. This is a **false negative** in the verifier — worse than a true negative because it silently degrades a correct answer.

### 3. Fix

`agent/model.py` gains `_strip_markdown_fences`, called in `OpenAIModel.complete` before `model_validate_json` on every structured call. Regression test `test_markdown_fenced_reply_is_parsed` in `tests/test_openai_adapter.py` pins the behaviour with the exact fenced shape Gemini returns.

### 4. What Went Well

- The false negative was caught in live testing, not shipped. Without a live run, the offline suite (which uses `ScriptedModel` with pre-parsed responses) cannot expose this class of provider-specific failure.
- The verifier's fail-closed policy is correct: on an unparseable verdict, treating the claim as unsupported is the safe choice. The bug was in the parse layer, not in the verifier's disposition.

### 5. Follow-up

1. Add an offline test that pins the parser against other provider-specific wrapping patterns (e.g. fenced with a `json` language tag, leading/trailing whitespace).
2. Consider moving fence-stripping into a provider-specific hook rather than the generic `complete` path, so a strict-parsing provider is not affected.

---

## INC-2026-11-09-03: Flagship §4.1 cites only proviso_6, not proviso_2

| Attribute | Details |
|---|---|
| **Incident ID** | INC-2026-11-09-03 |
| **Date** | 2026-11-09 |
| **Severity** | Minor (demo item 10 caveat; soundness unaffected) |
| **Impacted Services** | Analyst live path against Gemini 3 |
| **Author** | Engineering |

### 1. Summary

13 successful live runs of the §4.1 flagship question across three prompt configurations returned `answer-with-citation` with `verification.passed: true` in 13/13 cases. None cited `css2020-s53-proviso_2` (threshold waiver). All cited `css2020-s53-proviso_6` (pro-rata for fixed-term); `analyst-2` additionally cited `css2020-s53-main`.

The conclusion of every answer was legally correct (gratuity is payable). The verifier correctly confirmed the cited chunks support the claims. The gap is **completeness**, not soundness.

### 2. Evidence

| Prompt variant | Runs | Both provisos cited | `passed: true` | Grounded correctly |
|---|---|---|---|---|
| analyst-1 (baseline, before Rule 6) | 5 | 0/5 | 5/5 | 5/5 |
| analyst-1 + Rule 6 | 5 | 0/5 | 5/5 | 5/5 |
| analyst-2 + Rule 6 | 3 | 0/3 | 3/3 | 2/3 (see INC-2026-11-09-05) |

Trace inspection (with the caveat in INC-2026-11-09-04) shows the model makes one `corpus_search` call and does not call `lookup_clause` in any run. Top-k semantic results include `proviso_6`, which alone supports an affirmative answer.

### 3. Root Cause

Semantic search for the flagship question returns `proviso_6` directly. That proviso states fixed-term employees receive pro-rata gratuity — enough to answer "is gratuity payable?" affirmatively. The model has no reason to engage the five-year threshold stated in the main clause unless it reads the section's full provisos via `lookup_clause`.

This is the **soundness-vs-completeness gap** documented in README §1:

> Verification checks soundness, not completeness. The verifier asks whether cited
> passages support the claim. It does not ask whether the agent retrieved enough.

Stage-2 verification confirms the claim is grounded. It cannot confirm the claim set is complete relative to the question's framing ("under five years").

### 4. Attempts

Three prompt configurations across 13 successful runs:

1. **Baseline `analyst-1`** — 5/5 proviso_6 only. Answer contained an uncited overstatement ("regardless of the contract's duration").
2. **`analyst-1` + Rule 6** — 5/5 proviso_6 only. Answer text cleanly grounded. Tool calls reduced from 2 to 1 (model becomes more concise, not more thorough).
3. **`analyst-2` + Rule 6** — 3/3 proviso_6 (+ s53-main). Introduced an unsupported threshold-waiver claim that stage-2 accepted (see INC-2026-11-09-05).

Rule 6 improved the quality of the answer text but did not move the citation pattern. Neither variant triggered a `lookup_clause` call.

### 5. Disposition

**Accepted as a documented limitation.** The system's guarantee — no unverified answer reaches the caller — held 13/13. All answers were legally correct.

Phase 3.2 of the plan (`docs/plan-2026-11-09.md`) required "3/3 with both provisos". That criterion is **not met**. Recorded here so the demo is presented honestly. Demo item 10 becomes: "answered correctly and verified; citation reasoning trace incomplete — see INC-2026-11-09-03."

### 6. Follow-ups (not addressed in this submission)

1. **Capture real `StepRecord`s** (see INC-2026-11-09-04) — required before the trace can be trusted to inspect the model's actual tool selection.
2. **Structural completeness check** — before an answer may cite a section, require the section's deciding provisos to be in `retrieved_chunks`. This is the "future work" the README §1 names but does not implement.
3. **Prompt variants** — further tuning not recommended. 13 runs across three variants show the model's behaviour is stable under instruction. Structural fixes are the appropriate lever.

---

## INC-2026-11-09-04: RunTrace synthesizes tool records from retrieved chunks

| Attribute | Details |
|---|---|
| **Incident ID** | INC-2026-11-09-04 |
| **Date** | 2026-11-09 |
| **Severity** | Minor (D5 defect; demo caveat) |
| **Impacted Services** | `app/pipeline.py` `RunTrace` construction, `GET /runs/{id}/trace` |
| **Author** | Engineering |

### 1. Summary

`GET /runs/{id}/trace` returns `total_tool_calls: 1` alongside five `steps[]` entries of `action: "tool"`. The arguments on the `lookup_clause` entry — `{"section": "53", "doc": "css2020"}` — are inferred from chunk metadata, not from what was actually sent. The model never called `lookup_clause` in these runs; the trace says it did.

D5 requires "given a run identifier, the entire run must be reconstructible — every tool call, query, passage, prompt version, per-step latency and cost, and the verification outcome." The current `/trace` cannot reconstruct the actual queries.

### 2. Root Cause

`app/pipeline.py` constructs `StepRecord`s by iterating `analyst_result.retrieved_chunks`, one record per chunk:

```python
for idx, chunk in enumerate(analyst_result.retrieved_chunks):
    tool_name = "corpus_search" if "main" in chunk.part else "lookup_clause"
    step_records.append(
        StepRecord(
            tool=tool_name,
            arguments={"section": chunk.section, "doc": chunk.doc},
```

- `tool` is guessed from the `part` suffix (`"main" in part` → `corpus_search`,
  else → `lookup_clause`), which is wrong for chunks retrieved by
  `lookup_clause` whose parts are not `"main"`.
- `arguments` is derived from `chunk.section`/`chunk.doc` — metadata of the returned
  chunk, not the arguments passed to the tool.
- The actual query strings are not recorded anywhere.

### 3. Impact

- **Demo item 3** (traced end-to-end) cannot be truthfully claimed from `/trace`.
  Use `GET /runs/{id}` (answer + citations + verification), which is fully recorded
  from the actual `RunView` and reflects what the caller received.
- **Investigation** — during INC-2026-11-09-03 triage, the trace falsely showed
  `lookup_clause` calls that never occurred, which initially obscured the diagnosis
  (the model was not reaching for the section's provisos; it was answering from a
  semantic search).

### 4. Fix (out of scope for this submission)

Add `steps: list[StepRecord]` to `AnalystResult`. Populate in `agent/analyst.py`
`think`/`act` from actual tool dispatches — the argument dicts and the retrieval
results are both available at that point. `app/pipeline.py` uses the list directly
and stops synthesizing.

Estimated effort: half a day. Requires schema addition in `contracts/analyst_result.py`
and test updates in `tests/test_analyst.py`.

Interim mitigation if time is short: cap synthesized `StepRecord`s at
`analyst_result.loop.tool_calls` and set `tool=None`, `arguments={}`, `action="retrieved"`
on the synthesized entries — no fake call claims, but the trace is honest about being
partial.

---

## INC-2026-11-09-05: Stage-2 verifier lenient on partially-supported claims

| Attribute | Details |
|---|---|
| **Incident ID** | INC-2026-11-09-05 |
| **Date** | 2026-11-09 |
| **Severity** | Minor (soundness-adjacent; observed once) |
| **Impacted Services** | `agent/verifier.py` `Verifier.verify` stage 2 |
| **Author** | Engineering |

### 1. Summary

One `analyst-2` live run produced the answer:

> "Under the Code on Social Security, 2020, the requirement of five years of continuous
> service does not apply to fixed-term employment; instead, the employer is required to
> pay gratuity on a pro-rata basis."

Citations: `css2020-s53-main`, `css2020-s53-proviso_6`.

The first clause is the content of `proviso_2`. Neither cited chunk states it:
`s53-main` says the opposite ("after he has rendered continuous service for not less
than five years"), and `proviso_6` is silent on the threshold. The verifier accepted
the claim as supported.

### 2. Root Cause

Stage-2 asks the model to verify each claim against the union of its cited passages. The
union of `s53-main` + `proviso_6` contains enough adjacent content (a mention of
fixed-term in s53-main clause (d), pro-rata payment in proviso_6) that a lenient check
can read partial support as sufficient for a compound claim about threshold + quantum.

The verifier is not partitioned per-chunk: it does not ask "does chunk A support the
claim?" and "does chunk B support the claim?" separately. A claim that requires two
chunks to support it partially is accepted if the union supports it approximately.

### 3. Impact

- Observed once in 13 successful flagship runs. Not systematic but not ruled out.
- `analyst-1` (default) does not trigger this because its answer is entirely
  `proviso_6`-supported. Reverting default to `analyst-1` mitigates for the demo.
- The underlying weakness is real: **the verifier's stage-2 is not a completeness check**
  and should not be relied on as one.

### 4. Status

**Open, not fixed in this submission.** Reverting the default prompt to `analyst-1`
reduces exposure for the demo but does not close the underlying leniency.

### 5. Follow-up

1. **Partition the claim check per chunk.** For each claim, verify `(chunk, claim)` pairs
   independently and require at least one supporting pair. This refuses claims that no
   single cited chunk supports on its own.
2. **Per-claim citations.** Tighten the prompt block to require each claim's `citations`
   list to contain only chunks that individually support the claim. (Currently the
   prompt says "citing only the chunks whose text fully supports that assertion on its
   own" — but there is no runtime enforcement of this.)
3. **Regression test:** add a `ScriptedVerifier` case with a compound claim whose
   support is split across two chunks, asserting `passed: false`.