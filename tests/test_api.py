"""Tests for API endpoints, validation, queue execution, and verification.

Validates:
- Health check endpoint
- Ingestion validation, field-level rejection & quarantine
- Async submission and polling pattern backed by queue workers
- Citation resolution and verification in end-to-end pipeline
- Prevention of unverified or fabricated answers reaching the caller
- Prometheus metrics endpoint including queue depth
"""

import asyncio
import time
import pytest
from fastapi.testclient import TestClient

from app.analyst import StubAnalyst
from app.main import app
from app.pipeline_stub import set_analyst, set_verifier
from app.queue import reset_queue
from app.store import metrics_counters, quarantined_records, runs
from contracts import (
    AnalystResult,
    Chunk,
    Claim,
    LoopStats,
    ProvisionStatus,
    Route,
    RunRequest,
    RunStatus,
)


@pytest.fixture(autouse=True)
def reset_store():
    """Reset the in-memory state and queue before each test."""
    runs.clear()
    quarantined_records.clear()
    reset_queue()
    set_analyst(StubAnalyst())
    from agent.verifier import Verifier
    set_verifier(Verifier())
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
    """Verify GET /metrics returns Prometheus format plaintext with queue depth."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "compliance_runs_total" in response.text
    assert "compliance_runs_quarantined_total" in response.text
    assert "compliance_queue_depth" in response.text
    assert "compliance_workers_total" in response.text


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

        # 3. Wait for pipeline completion
        await asyncio.sleep(0.6)

        # 4. Poll again: must be succeeded with citations and verified answer
        poll_res_2 = await ac.get(f"/runs/{run_id}")
        assert poll_res_2.status_code == 200
        poll_data_2 = poll_res_2.json()
        assert poll_data_2["status"] == RunStatus.SUCCEEDED
        assert poll_data_2["route"] == Route.ANSWER
        assert poll_data_2["answer"] is not None
        assert "section 53" in poll_data_2["answer"]
        cited = {c["chunk_id"] for c in poll_data_2["citations"]}
        assert cited == {"css2020-s53-proviso_2", "css2020-s53-proviso_6"}
        assert poll_data_2["verification"]["passed"] is True


def test_poll_nonexistent_run_returns_404():
    """Verify polling for unknown run_id returns 404."""
    client = TestClient(app)
    response = client.get("/runs/nonexistent-uuid-1234")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_api_never_leaks_unverified_answer_on_fabricated_citation():
    """Gate 3: Inject AnalystResult with fabricated citation.

    API must never return answer-with-citation; route escalates to legal-counsel-review, answer is null.
    """
    from httpx import ASGITransport, AsyncClient

    class FabricatingAnalyst:
        async def analyze(self, request: RunRequest, correlation_id: str) -> AnalystResult:
            chunk = Chunk(
                chunk_id="css2020-s53-proviso_2",
                doc="css2020",
                section="53(1)",
                part="proviso_2",
                status=ProvisionStatus.IN_FORCE,
                text="Real chunk text.",
            )
            # Citing a fabricated chunk not in retrieved_chunks
            claim = Claim(
                text="Fabricated rule assertion.",
                citations=["css2020-s999-fabricated-id"],
            )
            return AnalystResult(
                route=Route.ANSWER,
                draft_answer="This answer must not leak to the caller.",
                claims=[claim],
                retrieved_chunks=[chunk],
                loop=LoopStats(),
                prompt_version="v1",
                correlation_id=correlation_id,
            )

    set_analyst(FabricatingAnalyst())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "question": "Is gratuity payable?",
            "establishment": {
                "headcount": 10,
                "industry": "Retail",
                "state": "Delhi",
            },
        }

        submit_res = await ac.post("/runs", json=payload)
        assert submit_res.status_code == 202
        run_id = submit_res.json()["run_id"]

        await asyncio.sleep(0.6)

        poll_res = await ac.get(f"/runs/{run_id}")
        assert poll_res.status_code == 200
        poll_data = poll_res.json()

        # Gate assertion: route is legal-counsel-review, answer null, verification failed
        assert poll_data["route"] == Route.LEGAL_COUNSEL
        assert poll_data["answer"] is None
        assert poll_data["citations"] == []
        assert poll_data["verification"]["passed"] is False
        assert 0 in poll_data["verification"]["failed_claims"]


@pytest.mark.asyncio
async def test_api_repealed_only_grounding_escalates():
    """Gate 4: Inject AnalystResult grounded only in repealed provisions.

    API must escalate to legal-counsel-review, answer null.
    """
    from httpx import ASGITransport, AsyncClient

    class RepealedOnlyAnalyst:
        async def analyze(self, request: RunRequest, correlation_id: str) -> AnalystResult:
            chunk = Chunk(
                chunk_id="pga1972-s4-1",
                doc="pga1972",
                section="4(1)",
                part="main",
                status=ProvisionStatus.REPEALED,
                text="Repealed 1972 Act provision requiring 5 years.",
            )
            claim = Claim(
                text="Continuous service of five years required.",
                citations=["pga1972-s4-1"],
            )
            return AnalystResult(
                route=Route.ANSWER,
                draft_answer="Under 1972 Act, 5 years required.",
                claims=[claim],
                retrieved_chunks=[chunk],
                loop=LoopStats(),
                prompt_version="v1",
                correlation_id=correlation_id,
            )

    set_analyst(RepealedOnlyAnalyst())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "question": "Can employee claim gratuity before 5 years?",
            "establishment": {
                "headcount": 30,
                "industry": "Manufacturing",
                "state": "Gujarat",
            },
        }

        submit_res = await ac.post("/runs", json=payload)
        assert submit_res.status_code == 202
        run_id = submit_res.json()["run_id"]

        await asyncio.sleep(0.6)

        poll_res = await ac.get(f"/runs/{run_id}")
        assert poll_res.status_code == 200
        poll_data = poll_res.json()

        assert poll_data["route"] == Route.LEGAL_COUNSEL
        assert poll_data["answer"] is None
        assert poll_data["citations"] == []
        assert poll_data["verification"]["passed"] is False
        assert 0 in poll_data["verification"]["failed_claims"]
