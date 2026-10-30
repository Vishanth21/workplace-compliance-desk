"""Pipeline executing Analyst -> Verifier -> RunView.

Runs the two-agent workflow:
1. Analyst (Agent 1) produces draft answer, claims, and retrieved chunks.
2. Verifier (Agent 2) resolves citations and verifies claims against statutory text.
3. Maps VerificationOutcome to RunView, ensuring unverified answers are never returned.
"""

import asyncio
import os
import time
from datetime import UTC, datetime
from typing import Any

from agent.telemetry import trace_span
from agent.verifier import Verifier, VerifierProtocol
from contracts import (
    Citation,
    Route,
    RunRequest,
    RunStatus,
    RunTrace,
    StepRecord,
    VerificationOutcome,
)

from .analyst import AnalystProtocol, RealAnalyst, StubAnalyst
from .queue import run_correlations
from .store import metrics_counters, run_latencies_s, run_traces, runs


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
    """Execute the multi-agent compliance advisory pipeline with full tracing (D5)."""
    if run_id not in runs:
        return

    start_perf = time.perf_counter()
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    correlation_id = run_correlations.get(run_id, run_id)

    with trace_span("pipeline.execute", {"run_id": run_id, "correlation_id": correlation_id}):
        run_view = runs[run_id]
        run_view.status = RunStatus.RUNNING

        # Non-blocking async sleep to simulate model and retrieval latency
        simulated_delay = float(os.getenv("PIPELINE_DELAY_SECONDS", "0.2"))
        if simulated_delay > 0:
            await asyncio.sleep(simulated_delay)

        # 1. Agent 1: Analyst
        active_analyst = analyst_override or analyst
        with trace_span("agent.analyst", {"correlation_id": correlation_id}):
            analyst_result = await active_analyst.analyze(request, correlation_id=correlation_id)

        # 2. Agent 2: Verifier
        active_verifier = verifier_override or verifier
        with trace_span("agent.verifier", {"correlation_id": correlation_id}):
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
            run_view.status = RunStatus.SUCCEEDED
            run_view.route = Route.LEGAL_COUNSEL
            run_view.answer = None
            run_view.citations = []
            run_view.error = outcome.reason
            metrics_counters["succeeded"] += 1

        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        run_latencies_s.append(elapsed_ms / 1000.0)

        # 4. Reconstruct Run Trajectory (Requirement D5)
        step_records: list[StepRecord] = []
        for idx, chunk in enumerate(analyst_result.retrieved_chunks):
            tool_name = "corpus_search" if "main" in chunk.part else "lookup_clause"
            step_records.append(
                StepRecord(
                    step_index=idx + 1,
                    action="tool",
                    tool=tool_name,
                    arguments={"section": chunk.section, "doc": chunk.doc},
                    tool_output_summary=chunk.text[:120],
                    passages_retrieved=[chunk.chunk_id],
                    latency_ms=round(analyst_result.loop.wall_clock_ms / max(1, analyst_result.loop.steps), 1),
                    estimated_prompt_tokens=150,
                    estimated_completion_tokens=40,
                    estimated_cost_usd=0.00015,
                )
            )

        step_records.append(
            StepRecord(
                step_index=len(step_records) + 1,
                action="answer",
                tool=None,
                arguments={},
                tool_output_summary=(analyst_result.draft_answer[:120] if analyst_result.draft_answer else None),
                passages_retrieved=[c for cl in analyst_result.claims for c in cl.citations],
                latency_ms=round(analyst_result.loop.wall_clock_ms / max(1, analyst_result.loop.steps), 1),
                estimated_prompt_tokens=350,
                estimated_completion_tokens=120,
                estimated_cost_usd=0.00035,
            )
        )

        total_cost = sum(s.estimated_cost_usd for s in step_records)

        run_traces[run_id] = RunTrace(
            run_id=run_id,
            correlation_id=correlation_id,
            question=request.question,
            prompt_version=analyst_result.prompt_version,
            status=run_view.status.value,
            route=outcome.final_route,
            answer=run_view.answer,
            steps=step_records,
            total_steps=analyst_result.loop.steps or len(step_records),
            total_tool_calls=analyst_result.loop.tool_calls or max(0, len(step_records) - 1),
            total_latency_ms=round(elapsed_ms, 1),
            total_cost_usd=round(total_cost, 5),
            citations=[c.chunk_id for c in run_view.citations],
            verification=outcome,
            created_at=created_at,
            completed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
