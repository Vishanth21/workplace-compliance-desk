"""The Analyst -> Verifier hand-off.

Serialisable in full: logged, persisted, passed between agents. The
Verifier reads ``retrieved_chunks`` from here.
"""

import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from .route import Route


class ProvisionStatus(StrEnum):
    """Whether a passage is currently law."""

    IN_FORCE = "in_force"
    REPEALED = "repealed"


_PART_RE = re.compile(
    r"main|proviso_[1-9]\d*|explanation(_[1-9]\d*)?|definition"
)


class Chunk(BaseModel):
    """One retrievable unit of statute."""

    chunk_id: str = Field(description="Stable id, e.g. 'css2020-s53-proviso_2'.")
    doc: str = Field(description="Source document short name, e.g. 'css2020'.")
    section: str = Field(description="Section identifier as printed, e.g. '53(1)'.")
    part: str = Field(
        description=(
            "Where in the section this chunk sits: 'main', 'proviso_<n>' "
            "(n>=1, 1-based within the section), 'explanation' or 'definition'. "
            "Provisos are split from the main clause at parse time."
        )
    )
    status: ProvisionStatus
    text: str

    @field_validator("part")
    @classmethod
    def _validate_part(cls, v: str) -> str:
        if not _PART_RE.fullmatch(v):
            raise ValueError(
                f"part must be 'main', 'proviso_<n>' (n>=1), 'explanation' "
                f"or 'definition'; got {v!r}"
            )
        return v


class Claim(BaseModel):
    """One assertion in a draft answer, with the chunk ids supporting it."""

    text: str
    citations: list[str] = Field(default_factory=list)


class LoopStats(BaseModel):
    """Loop bounds."""

    steps: int = 0
    tool_calls: int = 0
    wall_clock_ms: int = 0


class AnalystResult(BaseModel):
    """Agent 1's complete output."""

    route: Route
    draft_answer: str
    claims: list[Claim] = Field(default_factory=list)
    retrieved_chunks: list[Chunk] = Field(default_factory=list)
    loop: LoopStats = Field(default_factory=LoopStats)
    prompt_version: str = Field(description="Which prompt version produced this.")
    correlation_id: str = Field(description="Ties every step of the run together.")


class VerificationOutcome(BaseModel):
    """Verification verdict."""

    passed: bool
    final_route: Route
    reason: str
    failed_claims: list[int] = Field(
        default_factory=list,
        description="Indices into AnalystResult.claims that could not be grounded.",
    )