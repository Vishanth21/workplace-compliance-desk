"""Analyst agent contract and implementations.

Provides:
- AnalystProtocol
- StubAnalyst: returns the flagship compliance result for fixed-term gratuity
- RealAnalyst: wraps agent.analyst.analyse using LangGraph state machine
"""

import asyncio
from typing import Protocol, runtime_checkable

from agent.analyst import Bounds, analyse
from agent.model import Model
from contracts import (
    AnalystResult,
    Chunk,
    Claim,
    LoopStats,
    ProvisionStatus,
    Route,
    RunRequest,
)


@runtime_checkable
class AnalystProtocol(Protocol):
    """Protocol for Analyst Agent."""

    async def analyze(self, request: RunRequest, correlation_id: str) -> AnalystResult:
        """Run analysis and produce an AnalystResult."""
        ...


class StubAnalyst:
    """Stub Analyst producing the flagship compliance advisory."""

    def __init__(self, prompt_version: str = "v1") -> None:
        self.prompt_version = prompt_version

    async def analyze(self, request: RunRequest, correlation_id: str) -> AnalystResult:
        # Flagship provision: Section 53(1) second proviso of Code on Social Security, 2020
        chunk = Chunk(
            chunk_id="css2020-s53-1-prov-2",
            doc="css2020",
            section="53(1)",
            part="proviso_2",
            status=ProvisionStatus.IN_FORCE,
            text=(
                "Provided further that in the case of an employee employed on "
                "fixed term employment or on a contract basis, the employer shall pay "
                "gratuity on pro rata basis."
            ),
        )

        claims = [
            Claim(
                text=(
                    "Under Section 53(1), second proviso of the Code on Social Security, 2020, "
                    "the five-year continuous service threshold does not apply to employees engaged on "
                    "fixed-term contracts."
                ),
                citations=["css2020-s53-1-prov-2"],
            ),
            Claim(
                text="Gratuity is payable on a pro-rata basis upon the completion of the 18-month fixed-term contract.",
                citations=["css2020-s53-1-prov-2"],
            ),
        ]

        return AnalystResult(
            route=Route.ANSWER,
            draft_answer=(
                "Under Section 53(1), second proviso of the Code on Social Security, 2020, "
                "the five-year continuous service threshold does not apply to employees engaged on "
                "fixed-term contracts. Gratuity is payable on a pro-rata basis upon the completion "
                "of the 18-month fixed-term contract."
            ),
            claims=claims,
            retrieved_chunks=[chunk],
            loop=LoopStats(steps=1, tool_calls=1, wall_clock_ms=200),
            prompt_version=self.prompt_version,
            correlation_id=correlation_id,
        )


class RealAnalyst:
    """Real Analyst wrapping the LangGraph agent in agent.analyst."""

    def __init__(self, model: Model | None = None, bounds: Bounds | None = None) -> None:
        self.model = model
        self.bounds = bounds

    async def analyze(self, request: RunRequest, correlation_id: str) -> AnalystResult:
        return await asyncio.to_thread(
            analyse,
            question=request.question,
            request=request.model_dump(mode="json"),
            model=self.model,
            bounds=self.bounds,
            correlation_id=correlation_id,
        )
