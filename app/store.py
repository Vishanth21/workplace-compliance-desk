"""In-memory storage for runs, quarantined requests, and metrics."""

from contracts import RejectionResponse, RunView

# Process-local run state mapping: run_id -> RunView
runs: dict[str, RunView] = {}

# Quarantined invalid requests store
quarantined_records: list[RejectionResponse] = []

# Minimal metrics counters
metrics_counters: dict[str, int] = {
    "total_requests": 0,
    "quarantined": 0,
    "succeeded": 0,
    "failed": 0,
}
