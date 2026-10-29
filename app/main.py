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
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from contracts import (
    FieldError,
    RejectionResponse,
    RunAccepted,
    RunRequest,
    RunStatus,
    RunView,
)

from .guard import InjectionMatch, detect_injection
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
from agent.resilience import get_circuit_breaker
from index.retrieve import get_index_metadata


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

# A8: the demo web client. Served from the API process so `compose up` yields a
# usable UI with no second server or build step. Anchored to this file's
# directory so it resolves the same way from a checkout and from /app.
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
async def index() -> RedirectResponse:
    """Send the bare host to the demo client."""
    return RedirectResponse(url="/ui/")


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

    return _make_quarantine_response(correlation_id, field_errors)


def _make_quarantine_response(
    correlation_id: str, errors: list[FieldError]
) -> JSONResponse:
    """Build and record a quarantine response for validation or guard failure.

    Appends to ``quarantined_records``, increments the ``quarantined`` metric
    counter, and returns an HTTP 400 response with ``RejectionResponse`` payload.
    """
    rejection = RejectionResponse(
        status=RunStatus.QUARANTINED,
        correlation_id=correlation_id,
        errors=errors,
    )
    quarantined_records.append(rejection)
    metrics_counters["quarantined"] += 1
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=rejection.model_dump(mode="json"),
    )


def _quarantine_injection(match: InjectionMatch) -> JSONResponse:
    """Reject a prompt-injection attempt before any run is created.

    Deliberately shares the schema-validation path: same 400 status, same
    ``RejectionResponse`` body, same ``quarantined`` counter and same
    ``quarantined_records`` entry. One quarantine path means the metrics and
    the demo UI do not need to know *why* a request was refused.
    """
    errors = [
        FieldError(
            field="question",
            code="prompt_injection",
            message=(
                f"Question rejected by the ingestion guard "
                f"({match.category}). Rephrase as a compliance question."
            ),
        )
    ]
    return _make_quarantine_response(str(uuid.uuid4()), errors)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    """Liveness probe endpoint."""
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Plaintext metrics endpoint with run counters, queue depth, worker count, index metadata and circuit breaker state."""
    index_meta = get_index_metadata()
    cb = get_circuit_breaker()
    cb_open_flag = 1 if cb.state.value == "open" else 0

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
        "# HELP compliance_index_chunks Total statutory chunks indexed in Chroma (D2).\n"
        "# TYPE compliance_index_chunks gauge\n"
        f"compliance_index_chunks {index_meta['chunk_count']}\n"
        "# HELP compliance_index_freshness_seconds Age of the Chroma index in seconds (D2).\n"
        "# TYPE compliance_index_freshness_seconds gauge\n"
        f"compliance_index_freshness_seconds {index_meta['freshness_seconds']}\n"
        "# HELP compliance_circuit_breaker_open Whether LLM circuit breaker is OPEN (1) or CLOSED/HALF_OPEN (0) (A5).\n"
        "# TYPE compliance_circuit_breaker_open gauge\n"
        f"compliance_circuit_breaker_open {cb_open_flag}\n"
        "# HELP compliance_circuit_breaker_trips_total Total times the circuit breaker has tripped OPEN (A5).\n"
        "# TYPE compliance_circuit_breaker_trips_total counter\n"
        f"compliance_circuit_breaker_trips_total {cb.trip_count}\n"
    )


@app.post(
    "/runs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunAccepted,
    responses={400: {"model": RejectionResponse}},
)
async def submit_run(run_request: RunRequest) -> RunAccepted:
    """Async submit endpoint.

    Validates request, assigns run identifier and correlation identifier,
    initializes status to QUEUED in status store, and places run id onto queue.
    Returns 202 immediately.
    """
    # Injection guard runs before any run state exists, so a rejected request
    # leaves nothing queued. It reuses the quarantine path taken by schema
    # validation: same 400 body, same counter, same stored record.
    injection = detect_injection(run_request.question)
    if injection is not None:
        return _quarantine_injection(injection)

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
