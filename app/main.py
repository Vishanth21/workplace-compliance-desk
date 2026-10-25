"""FastAPI Application for Workplace Compliance Advisory Desk.

Features:
- REST interface against published data contract
- Ingestion validation, field-level rejection, quarantine
- Asynchronous submit & poll pattern
"""

import asyncio
import uuid
from typing import Any

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

from .pipeline_stub import run_pipeline
from .store import metrics_counters, quarantined_records, runs

app = FastAPI(
    title="Workplace Compliance Advisory Desk",
    description="Multi-agent compliance advisory API for labour statutes and codes",
    version="0.1.0",
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
    """Minimal plaintext metrics stub."""
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
    )


@app.post(
    "/runs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunAccepted,
)
async def submit_run(run_request: RunRequest) -> RunAccepted:
    """Async submit endpoint.

    Validates request, assigns run identifier and correlation identifier,
    initializes status to QUEUED, and returns 202 immediately.
    """
    run_id = str(uuid.uuid4())
    correlation_id = str(uuid.uuid4())

    # Initialize queued run view
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
    metrics_counters["total_requests"] += 1

    # Launch async task runner
    asyncio.create_task(run_pipeline(run_id, run_request))

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
