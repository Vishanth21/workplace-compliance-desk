"""FastAPI Application for Workplace Compliance Advisory Desk.

Features:
- REST interface against published data contract
- Ingestion validation, field-level rejection, quarantine
- Asynchronous submit & poll pattern
- Bounded process-local queue with fixed worker pool (A4)
- Two-agent execution: Analyst -> Verifier (A2, A3)
"""

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from contracts import (
    FieldError,
    RejectionResponse,
    RunAccepted,
    RunRequest,
    RunStatus,
    RunView,
)

from .queue import (
    WORKER_COUNT,
    get_queue_depth,
    run_correlations,
    run_queue,
    run_requests,
    start_workers,
    stop_workers,
)
from .store import metrics_counters, quarantined_records, runs


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle and worker pool."""
    start_workers(WORKER_COUNT)
    yield
    await stop_workers()


app = FastAPI(
    title="Workplace Compliance Advisory Desk",
    description="Multi-agent compliance advisory API for labour statutes and codes",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Enforce schema, range, and type validation at the ingestion boundary.

    Quarantines rejected records with field-level errors. No agent run is started.
    """
    correlation_id = str(uuid.uuid4())
    field_errors: list[FieldError] = []

    for err in exc.errors():
        # Clean field location without the leading "body"
        parts = [str(part) for part in err["loc"] if part != "body"]
        field_name = ".".join(parts) if parts else "body"

        field_errors.append(
            FieldError(
                field=field_name,
                code=err.get("type", "validation_error"),
                message=err.get("msg", "Invalid value"),
            )
        )

    rejection = RejectionResponse(
        status=RunStatus.QUARANTINED,
        correlation_id=correlation_id,
        errors=field_errors,
    )

    # Quarantine the rejection for inspection
    quarantined_records.append(rejection)
    metrics_counters["quarantined"] += 1

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=rejection.model_dump(mode="json"),
    )


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    """Liveness probe endpoint."""
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Plaintext metrics endpoint with run counters, queue depth, and worker count."""
    return (
        "# HELP compliance_runs_total Total compliance runs initiated.\n"
        "# TYPE compliance_runs_total counter\n"
        f"compliance_runs_total {metrics_counters['total_requests']}\n"
        "# HELP compliance_runs_quarantined_total Total requests quarantined at boundary.\n"
        "# TYPE compliance_runs_quarantined_total counter\n"
        f"compliance_runs_quarantined_total {metrics_counters['quarantined']}\n"
        "# HELP compliance_runs_succeeded_total Total successful runs.\n"
        "# TYPE compliance_runs_succeeded_total counter\n"
        f"compliance_runs_succeeded_total {metrics_counters['succeeded']}\n"
        "# HELP compliance_runs_failed_total Total failed runs.\n"
        "# TYPE compliance_runs_failed_total counter\n"
        f"compliance_runs_failed_total {metrics_counters['failed']}\n"
        "# HELP compliance_queue_depth Current number of runs waiting in queue.\n"
        "# TYPE compliance_queue_depth gauge\n"
        f"compliance_queue_depth {get_queue_depth()}\n"
        "# HELP compliance_workers_total Number of active background queue workers.\n"
        "# TYPE compliance_workers_total gauge\n"
        f"compliance_workers_total {WORKER_COUNT}\n"
    )


@app.post(
    "/runs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunAccepted,
)
async def submit_run(run_request: RunRequest) -> RunAccepted:
    """Async submit endpoint.

    Validates request, assigns run identifier and correlation identifier,
    initializes status to QUEUED in status store, and places run id onto queue.
    Returns 202 immediately.
    """
    run_id = str(uuid.uuid4())
    correlation_id = str(uuid.uuid4())

    # Initialize queued run view in status store
    run_view = RunView(
        run_id=run_id,
        status=RunStatus.QUEUED,
        route=None,
        answer=None,
        citations=[],
        verification=None,
        error=None,
    )
    runs[run_id] = run_view
    run_requests[run_id] = run_request
    run_correlations[run_id] = correlation_id
    metrics_counters["total_requests"] += 1

    # Ensure queue worker pool is active
    start_workers(WORKER_COUNT)

    # Queue carries ids, not state
    await run_queue.put(run_id)

    return RunAccepted(
        run_id=run_id,
        correlation_id=correlation_id,
        status_url=f"/runs/{run_id}",
    )


@app.get(
    "/runs/{run_id}",
    response_model=RunView,
    status_code=status.HTTP_200_OK,
)
async def get_run(run_id: str) -> RunView:
    """Poll endpoint.

    Returns current run state. Answer remains null unless status is SUCCEEDED.
    """
    run_view = runs.get(run_id)
    if run_view is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )
    return run_view
