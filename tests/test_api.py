"""Tests for API endpoints, validation, and pipeline execution.

Validates:
- Health check endpoint
- Ingestion validation, field-level rejection & quarantine
- Async submission and polling pattern
- Stub execution and citation resolution
- Metrics endpoint
"""

import asyncio
import time
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.store import metrics_counters, quarantined_records, runs
from contracts import Route, RunStatus


@pytest.fixture(autouse=True)
def reset_store():
    """Reset the in-memory state before each test."""
    runs.clear()
    quarantined_records.clear()
    metrics_counters["total_requests"] = 0
    metrics_counters["quarantined"] = 0
    metrics_counters["succeeded"] = 0
    metrics_counters["failed"] = 0


def test_health_endpoint():
    """Verify GET /health returns 200 with status ok."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint():
    """Verify GET /metrics returns Prometheus format plaintext."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "compliance_runs_total" in response.text
    assert "compliance_runs_quarantined_total" in response.text


def test_submit_valid_run_returns_202():
    """Verify POST /runs returns 202 with run_id, correlation_id and status_url."""
    client = TestClient(app)
    payload = {
        "question": "Is gratuity payable on expiry of a 18-month fixed-term contract?",
        "establishment": {
            "headcount": 50,
            "industry": "Information Technology",
            "state": "Karnataka",
            "has_fixed_term_employees": True,
        },
        "requested_by": "HR Operations",
    }
    response = client.post("/runs", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "run_id" in data
    assert "correlation_id" in data
    assert data["status_url"] == f"/runs/{data['run_id']}"

    # Verify initial run state is queued or running in store
    run_id = data["run_id"]
    assert run_id in runs
    assert runs[run_id].status in (RunStatus.QUEUED, RunStatus.RUNNING)
    assert runs[run_id].answer is None


def test_malformed_request_rejected_and_quarantined():
    """Verify malformed request returns 400 with field errors and quarantine status.

    No agent run must be started.
    """
    client = TestClient(app)
    # Test fixture: empty question (min_length=1) and headcount=0 (ge=1)
    # Valid industry and state provided so exactly 2 field errors occur.
    payload = {
        "question": "",
        "establishment": {
            "headcount": 0,
            "industry": "Software",
            "state": "Karnataka",
        },
    }
    response = client.post("/runs", json=payload)
    assert response.status_code == 400

    data = response.json()
    assert data["status"] == "quarantined"
    assert "correlation_id" in data
    assert len(data["errors"]) == 2

    fields = {err["field"] for err in data["errors"]}
    assert "question" in fields
    assert "establishment.headcount" in fields

    # Critical check: verify that NO agent run was created
    assert len(runs) == 0

    # Verify record was stored in quarantine
    assert len(quarantined_records) == 1
    assert metrics_counters["quarantined"] == 1


@pytest.mark.asyncio
async def test_async_pipeline_execution_and_poll():
    """Verify the async lifecycle: queued -> running -> succeeded with citation."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "question": "Is gratuity payable on fixed-term contract completion?",
            "establishment": {
                "headcount": 25,
                "industry": "Consulting",
                "state": "Maharashtra",
                "has_fixed_term_employees": True,
            },
        }

        # 1. Submit
        submit_res = await ac.post("/runs", json=payload)
        assert submit_res.status_code == 202
        run_id = submit_res.json()["run_id"]

        # 2. Poll immediately: answer must be null while queued/running
        poll_res_1 = await ac.get(f"/runs/{run_id}")
        assert poll_res_1.status_code == 200
        poll_data_1 = poll_res_1.json()
        assert poll_data_1["status"] in (RunStatus.QUEUED, RunStatus.RUNNING)
        assert poll_data_1["answer"] is None

        # 3. Wait for stub completion (stub sleeps for 2 seconds)
        await asyncio.sleep(2.5)

        # 4. Poll again: must be succeeded with citations and answer
        poll_res_2 = await ac.get(f"/runs/{run_id}")
        assert poll_res_2.status_code == 200
        poll_data_2 = poll_res_2.json()
        assert poll_data_2["status"] == RunStatus.SUCCEEDED
        assert poll_data_2["route"] == Route.ANSWER
        assert poll_data_2["answer"] is not None
        assert "Section 53(1)" in poll_data_2["answer"]
        assert len(poll_data_2["citations"]) == 1
        assert poll_data_2["citations"][0]["chunk_id"] == "css2020-s53-1-prov-2"
        assert poll_data_2["verification"]["passed"] is True


def test_poll_nonexistent_run_returns_404():
    """Verify polling for unknown run_id returns 404."""
    client = TestClient(app)
    response = client.get("/runs/nonexistent-uuid-1234")
    assert response.status_code == 404
