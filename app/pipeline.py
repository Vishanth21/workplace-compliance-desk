"""Pipeline executing Analyst -> Verifier -> RunView.

Runs the two-agent workflow:
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

from .analyst import AnalystProtocol, RealAnalyst, StubAnalyst
from .queue import run_correlations
from .store import metrics_counters, runs


def _build_analyst() -> AnalystProtocol:
    """Select the analyst from ``ANALYST_MODE``.

    ``stub`` (the default) replays the flagship advisory for every question and
    keeps the test suite deterministic. ``real`` runs the bounded LangGraph loop
    over the corpus tools with whatever ``LLM_PROVIDER`` resolves to.

    Defaulting to ``stub`` is deliberate: a fresh clone with no model
    configured still starts, still passes its tests, and still answers. Asking
    for the real loop is an explicit act.
    """
    # Strip first: a blank or whitespace-only value must mean "unset", not
    # "an unknown mode". ``os.getenv(...) or "stub"`` would not catch "   ".
    mode = (os.getenv("ANALYST_MODE") or "").strip().lower() or "stub"
    if mode == "real":
        return RealAnalyst()
    if mode != "stub":
        raise ValueError(
            f"unknown ANALYST_MODE {mode!r}; expected 'stub' or 'real'"
        )
    return StubAnalyst()


def _build_verifier() -> VerifierProtocol:
    """Build the verifier, attaching a model when one is configured.

    The deterministic checks (citation resolution, cite-nothing, cite-only-
    repealed) run with or without a model. The model is what adds the
    claim-vs-passage check -- does the cited text actually support the claim?

    Without a model that check is silently skipped, which was invisible while
    the stub analyst produced pre-verified output by construction. As soon as a
    real LLM drafts answers, skipping it means an answer can cite real chunk ids
    and still assert something those chunks do not say.

    ``VERIFIER_MODEL=off`` keeps the deterministic-only path, for tests and for
    running without a model at all. Anything else attaches ``get_model()``.
    ``skip`` is checked first so a test suite without credentials never builds
    a provider.
    """
    mode = (os.getenv("VERIFIER_MODEL") or "on").strip().lower()
    if mode == "off":
        return Verifier()

    provider = (os.getenv("LLM_PROVIDER") or "scripted").strip().lower()
    if provider == "scripted":
        # A scripted provider with no script raises on first call, and the
        # verifier would turn every claim into a failure. The default path must
        # stay useful, so only attach a model when the provider can answer.
        if not os.getenv("LLM_SCRIPTED_RESPONSES"):
            return Verifier()

    from agent.model import get_model

    return Verifier(model=get_model())


# Default agent instances.
# Every caller reads these module-level bindings, so this is the seam.
analyst: AnalystProtocol = _build_analyst()
verifier: VerifierProtocol = _build_verifier()


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
        # Verification failed: escalate to legal-counsel-review.
        # "passed=False must never produce answer-with-citation."
        # "An unverified answer must not be returned."
        run_view.status = RunStatus.SUCCEEDED
        run_view.route = Route.LEGAL_COUNSEL
        run_view.answer = None
        run_view.citations = []
        run_view.error = outcome.reason
        metrics_counters["succeeded"] += 1
