"""Locust load-test suite for the Workplace Compliance Advisory Desk (D6).

Measures the complete run lifecycle from POST /runs submission through polling
until status == 'succeeded', emitting a custom event for end-to-end latency.

Declared SLO: p95 <= 2,500 ms at sustained load.

Usage (headless, against local stub server):
    locust -f load_tests/locustfile.py --headless \\
        -u 5 -r 1 --run-time 30s --host http://localhost:8000

Or via the automated runner:
    python load_tests/run_load_test.py
"""

import time
from locust import HttpUser, between, events, task

# Payload for a typical compliance advisory request
_PAYLOAD = {
    "question": "Is gratuity payable on completion of a 12-month fixed-term contract?",
    "establishment": {
        "headcount": 25,
        "industry": "Information Technology",
        "state": "Karnataka",
        "has_fixed_term_employees": True,
    },
}

# Maximum time (seconds) to poll before declaring the run timed out.
POLL_TIMEOUT_S = 10.0
POLL_INTERVAL_S = 0.05


class ComplianceUser(HttpUser):
    """Simulates a user submitting a compliance query and polling for results."""

    wait_time = between(0.1, 0.3)

    @task
    def submit_and_poll(self) -> None:
        """Submit a compliance run and poll until succeeded. Emits custom e2e event."""
        t_start = time.perf_counter()

        # 1. Submit
        with self.client.post(
            "/runs",
            json=_PAYLOAD,
            catch_response=True,
            name="POST /runs",
        ) as submit_resp:
            if submit_resp.status_code != 202:
                submit_resp.failure(f"Submit returned {submit_resp.status_code}")
                return
            run_id = submit_resp.json().get("run_id")
            if not run_id:
                submit_resp.failure("No run_id in response")
                return

        # 2. Poll until succeeded or timeout
        deadline = t_start + POLL_TIMEOUT_S
        final_status = None
        while time.perf_counter() < deadline:
            poll_resp = self.client.get(f"/runs/{run_id}", name="GET /runs/{id}")
            if poll_resp.status_code == 200:
                data = poll_resp.json()
                final_status = data.get("status")
                if final_status in ("succeeded", "failed"):
                    break
            time.sleep(POLL_INTERVAL_S)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # 3. Emit custom event for e2e latency (captured by the runner)
        events.request.fire(
            request_type="e2e",
            name="GET /runs (e2e async run)",
            response_time=elapsed_ms,
            response_length=0,
            exception=None if final_status == "succeeded" else Exception(
                f"Run did not succeed: status={final_status}"
            ),
            context={},
        )

