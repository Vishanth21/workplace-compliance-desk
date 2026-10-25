"""Pipeline stub for async compliance execution.

Simulates analyst and verifier agents asynchronously.
"""

import asyncio

from contracts import (
    Citation,
    Route,
    RunRequest,
    RunStatus,
    VerificationOutcome,
)

from .store import metrics_counters, runs


async def run_pipeline(run_id: str, request: RunRequest) -> None:
    """Simulates async pipeline processing."""
    if run_id not in runs:
        return

    # Transition to RUNNING
    runs[run_id].status = RunStatus.RUNNING

    # Non-blocking async sleep to simulate model and retrieval latency
    await asyncio.sleep(2)

    # Cited provision from the Code on Social Security 2020 (s.53(1) second proviso)
    citation = Citation(
        chunk_id="css2020-s53-1-prov-2",
        doc="css2020",
        section="53(1)",
        text=(
            "Provided further that in the case of an employee employed on "
            "fixed term employment or on a contract basis, the employer shall pay "
            "gratuity on pro rata basis."
        ),
    )

    verification = VerificationOutcome(
        passed=True,
        final_route=Route.ANSWER,
        reason=(
            "Grounded in Section 53(1), second proviso of Code on Social Security, 2020. "
            "Provision verified in force."
        ),
        failed_claims=[],
    )

    # Transition to SUCCEEDED and populate answer + citations
    runs[run_id].status = RunStatus.SUCCEEDED
    runs[run_id].route = Route.ANSWER
    runs[run_id].answer = (
        "Under Section 53(1), second proviso of the Code on Social Security, 2020, "
        "the five-year continuous service threshold does not apply to employees engaged on "
        "fixed-term contracts. Gratuity is payable on a pro-rata basis upon the completion "
        "of the 18-month fixed-term contract."
    )
    runs[run_id].citations = [citation]
    runs[run_id].verification = verification

    metrics_counters["succeeded"] += 1
