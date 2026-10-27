"""Pipeline executing Analyst -> Verifier -> RunView.

Replaces the hardcoded Day 2 pipeline stub with a two-agent workflow:
1. Analyst (Agent 1) produces draft answer, claims, and retrieved chunks.
2. Verifier (Agent 2) resolves citations and verifies claims against statutory text.
3. Maps VerificationOutcome to RunView, ensuring unverified answers are never returned.
"""

import asyncio
import os
from typing import Any

from agent.verifier import Verifier, VerifierProtocol
from contracts import (
    Citation,
    Route,
    RunRequest,
    RunStatus,
    VerificationOutcome,
)

from .analyst import AnalystProtocol, StubAnalyst
from .queue import run_correlations
from .store import metrics_counters, runs

# Default agent instances
# When Person A delivers the real Analyst, swap this single constructor call:
analyst: AnalystProtocol = StubAnalyst()
verifier: VerifierProtocol = Verifier()


def set_analyst(new_analyst: AnalystProtocol) -> None:
    """Set or override the active analyst agent."""
    global analyst
    analyst = new_analyst


def set_verifier(new_verifier: VerifierProtocol) -> None:
    """Set or override the active verifier agent."""
    global verifier
    verifier = new_verifier


async def run_pipeline(
    run_id: str,
    request: RunRequest,
    analyst_override: Any = None,
    verifier_override: Any = None,
) -> None:
    """Execute the multi-agent compliance advisory pipeline."""
    if run_id not in runs:
        return

    run_view = runs[run_id]
    run_view.status = RunStatus.RUNNING

    # Non-blocking async sleep to simulate model and retrieval latency
    simulated_delay = float(os.getenv("PIPELINE_DELAY_SECONDS", "0.2"))
    if simulated_delay > 0:
        await asyncio.sleep(simulated_delay)

    # 1. Agent 1: Analyst
    active_analyst = analyst_override or analyst
    correlation_id = run_correlations.get(run_id, run_id)
    analyst_result = await active_analyst.analyze(request, correlation_id=correlation_id)

    # 2. Agent 2: Verifier
    active_verifier = verifier_override or verifier
    outcome: VerificationOutcome = active_verifier.verify(analyst_result)

    # 3. Write RunView
    run_view.verification = outcome
    run_view.route = outcome.final_route

    if outcome.passed:
        # Verified answer: Ground citations ONLY for surviving claims
        surviving_chunk_ids: set[str] = set()
        for idx, claim in enumerate(analyst_result.claims):
            if idx not in outcome.failed_claims:
                surviving_chunk_ids.update(claim.citations)

        chunk_map = {c.chunk_id: c for c in analyst_result.retrieved_chunks}
        citations: list[Citation] = []
        for cid in surviving_chunk_ids:
            if cid in chunk_map:
                c = chunk_map[cid]
                citations.append(
                    Citation(
                        chunk_id=c.chunk_id,
                        doc=c.doc,
                        section=c.section,
                        text=c.text,
                    )
                )

        run_view.status = RunStatus.SUCCEEDED
        run_view.answer = analyst_result.draft_answer
        run_view.citations = citations
        run_view.error = None
        metrics_counters["succeeded"] += 1
    else:
        # Verification failed: Escalate to legal-counsel-review before Day 7
        # "passed=False must never produce answer-with-citation."
        # "An unverified answer must not be returned."
        run_view.status = RunStatus.SUCCEEDED
        run_view.route = Route.LEGAL_COUNSEL
        run_view.answer = None
        run_view.citations = []
        run_view.error = outcome.reason
        metrics_counters["succeeded"] += 1
