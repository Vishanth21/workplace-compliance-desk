"""In-memory storage for runs, quarantined requests, run traces, and metrics."""

from contracts import RejectionResponse, RunTrace, RunView

# Process-local run state mapping: run_id -> RunView
runs: dict[str, RunView] = {}

# Process-local complete run reconstruction store (D5): run_id -> RunTrace
run_traces: dict[str, RunTrace] = {}

# Quarantined invalid requests store
quarantined_records: list[RejectionResponse] = []

# Minimal metrics counters and latency tracking
metrics_counters: dict[str, int] = {
    "total_requests": 0,
    "quarantined": 0,
    "succeeded": 0,
    "failed": 0,
}

# Recorded latencies for p50/p95 calculation (in seconds)
run_latencies_s: list[float] = []
