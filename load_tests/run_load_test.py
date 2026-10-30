"""Automated load-test runner for the Workplace Compliance Advisory Desk (D6).

Runs Locust headlessly at several concurrency levels against the local API
(which must already be running with the model-stub).  Prints a summary table
and writes results to ``docs/load-test-report.md``.

Usage:
    # Start the API first:
    PIPELINE_DELAY_SECONDS=0.15 uvicorn app.main:app --port 8000 &

    # Then run:
    python load_tests/run_load_test.py

The runner spawns a subprocess for each concurrency level so each run starts
with a fresh Locust environment.  It parses the CSV stats output to extract
p50, p95, RPS, and error rate for the custom "e2e" request type.
"""

import csv
import io
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).parent.parent

# Concurrency levels to test
CONCURRENCIES = [2, 5, 10, 20]

# Test duration per concurrency level (seconds)
RUN_TIME_S = 30

# API host
HOST = "http://localhost:8000"

# Declared SLO
SLO_P95_MS = 2500.0


class Result(NamedTuple):
    users: int
    rps: float
    p50_ms: float
    p95_ms: float
    error_pct: float
    slo_pass: bool


def run_locust(users: int) -> Result:
    """Run Locust headlessly for one concurrency level and parse results."""
    csv_prefix = ROOT / f"load_tests/.run_{users}u"
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(ROOT / "load_tests" / "locustfile.py"),
        "--headless",
        f"--users={users}",
        f"--spawn-rate={users}",
        f"--run-time={RUN_TIME_S}s",
        f"--host={HOST}",
        f"--csv={csv_prefix}",
        "--csv-full-history",
        "--only-summary",
        "--loglevel=WARNING",
    ]

    print(f"  Running u={users} for {RUN_TIME_S}s ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIME_S + 30)

    stats_file = Path(f"{csv_prefix}_stats.csv")
    if not stats_file.exists():
        print(f"  [WARN] CSV not found for u={users}; using zeros")
        return Result(users=users, rps=0, p50_ms=0, p95_ms=0, error_pct=0, slo_pass=False)

    with stats_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Name") in ("GET /runs (e2e async run)", "Aggregated"):
                rps = float(row.get("Requests/s", 0) or 0)
                p50 = float(row.get("50%", 0) or 0)
                p95 = float(row.get("95%", 0) or 0)
                failures = int(row.get("Failure Count", 0) or 0)
                total = int(row.get("Request Count", 1) or 1)
                err_pct = (failures / total) * 100 if total else 0
                return Result(
                    users=users,
                    rps=round(rps, 2),
                    p50_ms=p50,
                    p95_ms=p95,
                    error_pct=round(err_pct, 1),
                    slo_pass=(p95 <= SLO_P95_MS),
                )

    return Result(users=users, rps=0, p50_ms=0, p95_ms=0, error_pct=0, slo_pass=False)


def write_report(results: list[Result]) -> None:
    """Write the load-test report markdown."""
    report_path = ROOT / "docs" / "load-test-report.md"
    lines = [
        "# Load-Test Report — Workplace Compliance Advisory Desk (D6)",
        "",
        "## 1. SLO Declaration",
        "",
        "| Metric | Target |",
        "|--------|--------|",
        "| End-to-end p95 latency (POST /runs → succeeded poll) | **≤ 2,500 ms** |",
        "| Measurement method | Locust headless, e2e custom event |",
        "| Stub server delay | 150 ms simulated (PIPELINE_DELAY_SECONDS=0.15) |",
        "",
        "> **Justification:** A 2,500 ms p95 budget is generous relative to the",
        "> stub's 150 ms model latency + 50 ms poll interval overhead + FastAPI",
        "> processing time (~5–10 ms).  Under real LLM latencies (1–3 s),",
        "> the budget would be raised to 8–10 s; the measurement methodology",
        "> and saturation analysis remain identical.",
        "",
        "## 2. Test Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Concurrency levels tested | {', '.join(str(u) for u in CONCURRENCIES)} users |",
        f"| Duration per level | {RUN_TIME_S} s |",
        "| Spawn rate | = users (instant ramp) |",
        "| Target | http://localhost:8000 |",
        "| Model backend | agent/model_stub_server.py (PIPELINE_DELAY_SECONDS=0.15) |",
        "| In-process workers | 2 (WORKER_COUNT) |",
        "",
        "## 3. Results",
        "",
        "| Users | RPS | p50 (ms) | p95 (ms) | Errors % | SLO Pass? |",
        "|-------|-----|----------|----------|----------|-----------|",
    ]
    for r in results:
        slo = "✅" if r.slo_pass else "❌"
        lines.append(
            f"| {r.users} | {r.rps} | {r.p50_ms:.0f} | {r.p95_ms:.0f} | {r.error_pct:.1f}% | {slo} |"
        )

    passing = [r for r in results if r.slo_pass]
    failing = [r for r in results if not r.slo_pass]
    sat_users = failing[0].users if failing else ">20"

    lines += [
        "",
        "## 4. Analysis",
        "",
        f"- **Saturation concurrency:** {sat_users} users — the first level where p95 exceeds the SLO.",
        "- **Bottleneck component:** The in-process worker queue (`WORKER_COUNT=2`).  With only two",
        "  workers consuming the asyncio queue, requests queue up once arrival rate exceeds",
        "  `2 / pipeline_latency` requests per second.  Scaling is possible via `--scale worker=3`",
        "  (requires D3 multi-container compose), declared as a cut in the project report.",
        "- **Instrumentation path:** Every completed run appends its latency to `run_latencies_s`.",
        "  The `/metrics` endpoint computes p50 and p95 from that list and exposes them as",
        "  `compliance_run_latency_seconds_p50` and `compliance_run_latency_seconds_p95`.",
        "  Prometheus scrapes this endpoint every 5 s; Grafana plots the gauges.",
        "",
        "## 5. Declared Cuts",
        "",
        "| Item | Reason |",
        "|------|--------|",
        "| Locust distributed mode | Single-host test is sufficient to identify saturation. |",
        "| Measured p95 under real LLM | Stub latency only; real LLM SLO would be re-declared separately. |",
        "| Automatic SLO alerting in Grafana | Alert rules not provisioned; threshold visible on the panel. |",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {report_path}")


def main() -> None:
    print(f"Load-test runner — SLO: p95 <= {SLO_P95_MS:.0f} ms")
    print(f"Host: {HOST}")
    print(f"Levels: {CONCURRENCIES}\n")

    results: list[Result] = []
    for users in CONCURRENCIES:
        try:
            r = run_locust(users)
            results.append(r)
            status = "PASS" if r.slo_pass else "FAIL"
            print(
                f"  u={r.users:2d}  RPS={r.rps:5.1f}  "
                f"p50={r.p50_ms:6.0f}ms  p95={r.p95_ms:6.0f}ms  "
                f"err={r.error_pct:.1f}%  [{status}]"
            )
        except subprocess.TimeoutExpired:
            print(f"  u={users} TIMED OUT — skipping")
            results.append(Result(users=users, rps=0, p50_ms=0, p95_ms=0, error_pct=100, slo_pass=False))
        except Exception as exc:  # noqa: BLE001
            print(f"  u={users} ERROR: {exc}")

    write_report(results)


if __name__ == "__main__":
    main()

