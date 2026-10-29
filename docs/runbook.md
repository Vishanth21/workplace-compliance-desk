# Operational Runbook — Workplace Compliance Advisory Desk

This runbook documents operational diagnosis and recovery procedures for the two
primary production failure modes (Requirement D8).

---

## Failure Mode 1: Upstream LLM Outage & Circuit Breaker Tripping

### Symptom
- Client queries accept with HTTP 202, but subsequent polling against `/runs/{id}`
  shows `status: "succeeded"` with `route: "legal-counsel-review"`.
- Response error field contains: `Circuit breaker 'llm-breaker' is OPEN`.
- Prometheus metrics show:
  ```prometheus
  compliance_circuit_breaker_open 1
  ```
  and `compliance_circuit_breaker_trips_total` increments.
- Application logs show consecutive upstream failure warnings followed by:
  `Circuit breaker 'llm-breaker' tripped OPEN (3 consecutive failures)`.

### Confirmation
1. **Check metrics endpoint:**
   ```bash
   curl -s http://localhost:8000/metrics | grep circuit_breaker
   ```
   Confirm `compliance_circuit_breaker_open` is `1`.

2. **Probe the upstream LLM endpoint directly:**
   ```bash
   curl -s -i "$LLM_BASE_URL/chat/completions" \
     -H "Authorization: Bearer $LLM_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model": "'"$LLM_MODEL"'", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}'
   ```
   Inspect HTTP status code. Look for HTTP 429 (quota/rate limit exceeded), 502/503 (upstream outage), or 401 (expired credential).

3. **Trace correlation IDs in structured logs:**
   Filter logs for the affected `correlation_id` to confirm exponential backoff attempts:
   ```text
   Model call failed (attempt 1/3): ... Retrying in 0.50s...
   Model call failed (attempt 2/3): ... Retrying in 1.00s...
   Circuit breaker 'llm-breaker' tripped OPEN after retries exhausted
   ```

### Recovery
1. **Remediate upstream condition:**
   - If HTTP 429 / rate limited: adjust upstream quota or switch `LLM_BASE_URL` to secondary inference replica.
   - If HTTP 401 / bad credentials: update `LLM_API_KEY` in deployment environment.
   - If local mock/stub is in use: ensure `model-stub` container is running and healthy.
2. **Breaker recovery:**
   - The circuit breaker automatically enters `HALF_OPEN` state after the recovery window (`LLM_CB_RECOVERY_TIMEOUT_S`, default 30s).
   - The next incoming request serves as a probe. If it succeeds, the breaker transitions automatically back to `CLOSED`.
   - To force immediate recovery without waiting for timeout:
     ```bash
     docker compose restart api worker
     ```
3. **Verify resolution:**
   Submit test query:
   ```bash
   curl -X POST http://localhost:8000/runs \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the statutory threshold for gratuity?", "establishment": {"headcount": 50}}'
   ```
   Confirm run completes with `route: "answer-with-citation"` and `compliance_circuit_breaker_open 0`.

---

## Failure Mode 2: Corpus Vector Store Inconsistency or Hash Mismatch

### Symptom
- Containers fail startup healthcheck or crash on boot with:
  `RuntimeError: index built with ... querying with ... Rebuild with --force`
  or `RuntimeError: count mismatch`.
- Runtime `corpus_search` or `lookup_clause` yields empty results (`retrieved_chunks: []`)
  for standard queries, forcing all inquiries to fallback to legal counsel review.
- Metric check reveals `compliance_index_chunks 0` or missing collection metadata.

### Confirmation
1. **Inspect `/metrics` for index status:**
   ```bash
   curl -s http://localhost:8000/metrics | grep compliance_index
   ```
   Confirm whether `compliance_index_chunks` is below expected (expected: 621 chunks across CSS 2020, POSH 2013, and PGA 1972).

2. **Check physical index directory:**
   ```bash
   ls -la index/store/
   ```
   Confirm `chroma.sqlite3` and segment files exist and are readable by the container user.

3. **Verify build hash:**
   Run the idempotent index verification:
   ```bash
   python -m index.build
   ```
   If corpus files or chunking rules changed, it prints:
   `rebuilding (force=False, hash ...)`

### Recovery
1. **Execute full forced re-index:**
   Rebuild the Chroma collection idempotently from the authoritative corpus PDFs:
   ```bash
   python -m index.build --force
   ```
   Output must conclude with:
   `done: 621 chunks in .../index/store`

2. **Restart services:**
   Restart API and background worker so open Chroma handles re-bind to the refreshed persistent store:
   ```bash
   docker compose restart api worker
   ```

3. **Validate index freshness:**
   Verify Prometheus metric:
   ```bash
   curl -s http://localhost:8000/metrics | grep compliance_index
   ```
   Output must show:
   ```prometheus
   compliance_index_chunks 621
   compliance_index_freshness_seconds <small_number>
   ```

4. **Verify flagship query resolution:**
   Run verification script:
   ```bash
   python verify_api.py
   ```
   Confirm that §53(1) provisos resolve and citations point to indexed statutes.

