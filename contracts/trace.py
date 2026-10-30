"""Audit trail and run reconstruction contracts for full observability (Requirement D5).

Requirement D5:
"Given a run identifier, the entire run must be reconstructible -- every tool call,
query, passage, prompt version, per-step latency and cost, and the verification outcome."
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from contracts.analyst_result import Route, VerificationOutcome


class StepRecord(BaseModel):
    """Reconstructed telemetry for one execution step in the analyst loop."""

    step_index: int
    action: str  # "tool" or "answer"
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_output_summary: str | None = None
    passages_retrieved: list[str] = Field(
        default_factory=list,
        description="Chunk IDs or statutory passage identifiers retrieved in this step.",
    )
    latency_ms: float = 0.0
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


class RunTrace(BaseModel):
    """Complete reconstructed trajectory for a compliance advisory run (D5)."""

    run_id: str
    correlation_id: str
    question: str
    prompt_version: str
    status: str
    route: Route | None = None
    answer: str | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    total_steps: int = 0
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    citations: list[str] = Field(default_factory=list)
    verification: VerificationOutcome | None = None
    created_at: str
    completed_at: str | None = None

