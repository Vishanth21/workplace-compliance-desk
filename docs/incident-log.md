# Incident Log — Workplace Compliance Advisory Desk

Record of operational incidents, root cause analyses, and remediations (Requirement D8).

---

## INC-2026-10-28: Upstream Inference 503 Outage & Circuit Breaker Degradation

| Attribute | Details |
|---|---|
| **Incident ID** | INC-2026-10-28-01 |
| **Date & Time** | 2026-10-28 14:15:00 UTC to 14:32:00 UTC (Duration: 17 min) |
| **Severity** | Major (Advisory service degraded to legal counsel fallback) |
| **Impacted Services** | `compliance-api`, `compliance-worker`, LLM inference seam |
| **Author** | Operations Team |

### 1. Summary
At 14:15 UTC, the external OpenAI-compatible model provider experienced an unannounced
gateway degradation, returning HTTP 503 Service Unavailable to `/chat/completions` requests.
The compliance desk resilience layer (A5) executed exponential backoff retries across
two attempts, registered consecutive failures, and tripped the circuit breaker OPEN at
14:16 UTC. During the outage, incoming client requests were accepted with HTTP 202 and
cleanly escalated to `route: "legal-counsel-review"` with no silent failures, no data
corruption, and no hung worker threads. Full automatic recovery occurred at 14:32 UTC.

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
Third-party model inference provider experienced a transient gateway outage, returning
HTTP 503 for all `/chat/completions` payloads.

### 4. What Went Well
- **Zero data loss:** Incoming requests were safely captured, queued, and stored in the status store.
- **Safe failure mode:** No ungrounded, hallucinated, or unverified answers were served. The defined fallback path (graceful degradation to `legal-counsel-review`) protected user compliance safety.
- **Observability:** Correlation IDs accurately grouped the retry sequence and breaker state change. Prometheus metrics clearly flagged `compliance_circuit_breaker_open = 1`.

### 5. Corrective & Preventative Actions
1. **Multi-provider failover:** Implement automatic secondary provider fallback in `agent/model.py` before opening the breaker.
2. **Dashboarding:** Add an alert rule for `compliance_circuit_breaker_open == 1` with a 2-minute persistence window.

