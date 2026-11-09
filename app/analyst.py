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

    def __init__(self, prompt_version: str = "analyst-1") -> None:
        self.prompt_version = prompt_version

    async def analyze(self, request: RunRequest, correlation_id: str) -> AnalystResult:
        # Flagship provisions: section 53 of the Code on Social Security, 2020.
        # Two provisos are load-bearing: the second removes the five-year
        # continuous-service threshold when the term expires, and the sixth
        # makes fixed-term employment gratuity-eligible on a pro-rata basis.
        threshold = Chunk(
            chunk_id="css2020-s53-proviso_2",
            doc="css2020",
            section="53",
            part="proviso_2",
            status=ProvisionStatus.IN_FORCE,
            text=(
                "Provided further that the completion of continuous service of five years shall not "
                "be necessary where the termination of the employment of any employee is due to "
                "death or disablement or expiration of fixed term employment or happening of any "
                "such event as may be notified by the Central Government."
            ),
        )
        pro_rata = Chunk(
            chunk_id="css2020-s53-proviso_6",
            doc="css2020",
            section="53",
            part="proviso_6",
            status=ProvisionStatus.IN_FORCE,
            text=(
                "Provided also that in the case of an employee employed on fixed term "
                "employment or a deceased employee, the employer shall pay gratuity on pro rata basis."
            ),
        )

        claims = [
            Claim(
                text=(
                    "Under the second proviso to section 53 of the Code on Social "
                    "Security, 2020, five years' continuous service is not necessary "
                    "where the employment ends on expiry of a fixed term."
                ),
                citations=["css2020-s53-proviso_2"],
            ),
            Claim(
                text=(
                    "Under the sixth proviso to section 53, an employee on fixed-term "
                    "employment is paid gratuity on a pro-rata basis."
                ),
                citations=["css2020-s53-proviso_6"],
            ),
        ]

        return AnalystResult(
            route=Route.ANSWER,
            draft_answer=(
                "Yes. Under the second proviso to section 53 of the Code on Social "
                "Security, 2020, five years' continuous service is not necessary where "
                "the employment ends on expiry of a fixed term; under the sixth proviso "
                "an employee on fixed-term employment is paid gratuity on a pro-rata "
                "basis."
            ),
            claims=claims,
            retrieved_chunks=[threshold, pro_rata],
            loop=LoopStats(steps=1, tool_calls=1, wall_clock_ms=200),
            prompt_version=self.prompt_version,
            correlation_id=correlation_id,
        )


class RealAnalyst:
    """Real Analyst wrapping the LangGraph agent in agent.analyst."""

    def __init__(
        self,
        model: Model | None = None,
        bounds: Bounds | None = None,
        prompt_version: str | None = None,
    ) -> None:
        self.model = model
        self.bounds = bounds
        self.prompt_version = prompt_version

    async def analyze(self, request: RunRequest, correlation_id: str) -> AnalystResult:
        return await asyncio.to_thread(
            analyse,
            question=request.question,
            request=request.model_dump(mode="json"),
            model=self.model,
            bounds=self.bounds,
            correlation_id=correlation_id,
            prompt_version=self.prompt_version,
        )
