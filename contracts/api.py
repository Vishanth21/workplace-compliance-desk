"""REST data contract for the external interface.

POST a question, receive a run id, poll for the result.
"""

from pydantic import BaseModel, Field

from .analyst_result import VerificationOutcome
from .route import Route, RunStatus


class Establishment(BaseModel):
    """The employer context."""

    headcount: int = Field(ge=1, description="Total employees at the establishment.")
    industry: str = Field(min_length=1, max_length=120)
    state: str = Field(min_length=1, max_length=120)
    has_fixed_term_employees: bool = False


class RunRequest(BaseModel):
    """Inbound compliance question."""

    question: str = Field(min_length=1, max_length=2000)
    establishment: Establishment
    requested_by: str | None = Field(default=None, max_length=120)


class RunAccepted(BaseModel):
    """202 response."""

    run_id: str
    correlation_id: str
    status_url: str


class Citation(BaseModel):
    """A resolved citation."""

    chunk_id: str
    doc: str
    section: str
    text: str


class RunView(BaseModel):
    """Poll response."""

    run_id: str
    status: RunStatus
    route: Route | None = None
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    verification: VerificationOutcome | None = None
    error: str | None = None


class FieldError(BaseModel):
    """Field-level rejection detail."""

    field: str
    code: str
    message: str


class RejectionResponse(BaseModel):
    """400 body for a request that never started a run."""

    status: RunStatus = RunStatus.QUARANTINED
    correlation_id: str
    errors: list[FieldError] = Field(default_factory=list)