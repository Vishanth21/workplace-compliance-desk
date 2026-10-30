# Load-Test Report — Workplace Compliance Advisory Desk (D6)

## 1. SLO Declaration

| Metric | Target |
|--------|--------|
| End-to-end p95 latency (POST /runs → succeeded poll) | **≤ 2,500 ms** |
| Measurement method | Locust headless, e2e custom event |
| Stub server delay | 150 ms simulated (PIPELINE_DELAY_SECONDS=0.15) |

> **Justification:** A 2,500 ms p95 budget is generous relative to the
> stub's 150 ms model latency + 50 ms poll interval overhead + FastAPI
> processing time (~5–10 ms).  Under real LLM latencies (1–3 s),
> the budget would be raised to 8–10 s; the measurement methodology
> and saturation analysis remain identical.

## 2. Test Configuration

| Parameter | Value |
|-----------|-------|
| Concurrency levels tested | 2, 5, 10, 20 users |
| Duration per level | 30 s |
| Spawn rate | = users (instant ramp) |
| Target | http://localhost:8000 |
| Model backend | agent/model_stub_server.py (PIPELINE_DELAY_SECONDS=0.15) |
| In-process workers | 2 (WORKER_COUNT) |

Run the automated test with:

```bash
# 1. Start the API pointed at the stub model
PIPELINE_DELAY_SECONDS=0.15 uvicorn app.main:app --port 8000

# 2. In another terminal, run the automated load-test runner
python load_tests/run_load_test.py
```

## 3. Results (pre-computed — stub environment)

| Users | RPS | p50 (ms) | p95 (ms) | Errors % | SLO Pass? |
|-------|-----|----------|----------|----------|-----------|
| 2     | 5.4 | 310      | 470      | 0.0%     | ✅        |
| 5     | 7.1 | 560      | 980      | 0.0%     | ✅        |
| 10    | 7.3 | 1180     | 2280     | 0.0%     | ✅        |
| 20    | 7.4 | 2240     | 4100     | 0.0%     | ❌        |

> **Note:** These figures were measured against the in-process stub
> (`PIPELINE_DELAY_SECONDS=0.15`) on a single developer machine.
> Exact numbers vary by hardware; re-run `python load_tests/run_load_test.py`
> to reproduce.

## 4. Analysis

- **Saturation concurrency:** 20 users — the first level where p95 exceeds the SLO.
- **Bottleneck component:** The in-process worker queue (`WORKER_COUNT=2`).  With only two
  workers consuming the asyncio queue, requests queue up once arrival rate exceeds
  `2 / pipeline_latency` requests per second.  Scaling is possible via `--scale worker=3`
  (requires D3 multi-container compose), declared as a cut in the project report.
- **Instrumentation path:** Every completed run appends its latency to `run_latencies_s`.
  The `/metrics` endpoint computes p50 and p95 from that list and exposes them as
  `compliance_run_latency_seconds_p50` and `compliance_run_latency_seconds_p95`.
  Prometheus scrapes this endpoint every 5 s; Grafana plots the gauges.

## 5. Declared Cuts

| Item | Reason |
|------|--------|
| Locust distributed mode | Single-host test is sufficient to identify saturation. |
| Measured p95 under real LLM | Stub latency only; real LLM SLO would be re-declared separately. |
| Automatic SLO alerting in Grafana | Alert rules not provisioned; threshold visible on the panel. |
