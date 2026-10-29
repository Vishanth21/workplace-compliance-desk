"""The Verifier: Agent 2 in the compliance advisory desk.

Distinct responsibility: Refuse ungrounded, fabricated, or repealed claims.
Contract:
- Input: AnalystResult
- Output: VerificationOutcome
- Rule: An unverified answer must never reach the caller.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent.model import Message, Model, ModelError
from contracts import (
    AnalystResult,
    Chunk,
    ProvisionStatus,
    Route,
    VerificationOutcome,
)

logger = logging.getLogger(__name__)


class Verdict(BaseModel):
    """Structured verdict for a claim against cited statutory passages."""

    supported: bool
    reason: str


@runtime_checkable
class VerifierProtocol(Protocol):
    """Protocol defining the Verifier interface."""

    def verify(self, result: AnalystResult) -> VerificationOutcome:
        """Verify an analyst result and return a structured verdict."""
        ...


class StubVerifier:
    """Stub verifier for hand-off before full verification is wired.

    Accepts any well-formed AnalystResult and returns passed=True, final_route=ANSWER.
    """

    def verify(self, result: AnalystResult) -> VerificationOutcome:
        return VerificationOutcome(
            passed=True,
            final_route=Route.ANSWER,
            reason="Verified by stub.",
            failed_claims=[],
        )


class Verifier:
    """The Verifier Agent.

    Performs:
    1. Deterministic citation resolution (A3):
       - Every cited chunk_id must exist in retrieved_chunks.
       - A claim citing nothing fails.
       - A claim grounded only in repealed provisions fails.
    2. Claim-vs-passage model check (A3 / Agent):
       - When a Model is provided, calls model.complete with response_schema=Verdict
         to verify whether the text of cited chunks supports each claim.
    """

    def __init__(self, model: Model | None = None) -> None:
        self.model = model

    def verify(self, result: AnalystResult) -> VerificationOutcome:
        # Non-answer routes (e.g. request-more-detail or legal-counsel-review) pass through
        if result.route != Route.ANSWER:
            return VerificationOutcome(
                passed=True,
                final_route=result.route,
                reason=f"Analyst routed directly to {result.route.value}.",
                failed_claims=[],
            )

        # Build chunk lookup and id set from retrieved_chunks
        chunk_map = {chunk.chunk_id: chunk for chunk in result.retrieved_chunks}
        retrieved_ids = set(chunk_map.keys())

        # An answer with no claims cannot be verified or grounded
        if not result.claims:
            return VerificationOutcome(
                passed=False,
                final_route=Route.LEGAL_COUNSEL,
                reason="Verification failed: Answer contains no claims to ground.",
                failed_claims=[],
            )

        failed_claim_indices: list[int] = []
        failure_reasons: list[str] = []

        # --- Stage 1: Deterministic Citation Resolution ---
        for idx, claim in enumerate(result.claims):
            # Check 1: A claim citing nothing fails
            if not claim.citations:
                failed_claim_indices.append(idx)
                failure_reasons.append(f"Claim {idx} has no citations")
                continue

            # Check 2: Every cited id must be in retrieved_chunks
            unresolved = [cid for cid in claim.citations if cid not in retrieved_ids]
            if unresolved:
                failed_claim_indices.append(idx)
                failure_reasons.append(
                    f"Claim {idx} cites unretrieved chunk(s): {', '.join(unresolved)}"
                )
                continue

            # Check 3: A claim grounded only in repealed chunks fails
            # (i.e. every cited chunk has status == ProvisionStatus.REPEALED)
            cited_statuses = [chunk_map[cid].status for cid in claim.citations]
            if all(status == ProvisionStatus.REPEALED for status in cited_statuses):
                failed_claim_indices.append(idx)
                repealed_cids = [
                    cid
                    for cid in claim.citations
                    if chunk_map[cid].status == ProvisionStatus.REPEALED
                ]
                failure_reasons.append(
                    f"Claim {idx} grounded only in repealed chunk(s): {', '.join(repealed_cids)}"
                )
                continue

        # If citation resolution failed for any claim, fail immediately without model call
        if failed_claim_indices:
            unique_failures = sorted(list(set(failed_claim_indices)))
            return VerificationOutcome(
                passed=False,
                final_route=Route.LEGAL_COUNSEL,
                reason="Citation resolution failed: " + "; ".join(failure_reasons),
                failed_claims=unique_failures,
            )

        # --- Stage 2: Claim-vs-Passage Model Check ---
        if self.model is not None:
            model_failed_indices: list[int] = []
            model_reasons: list[str] = []

            for idx, claim in enumerate(result.claims):
                cited_chunks = [chunk_map[cid] for cid in claim.citations]
                supported, reason = self._check_claim_against_passages(claim.text, cited_chunks)
                if not supported:
                    model_failed_indices.append(idx)
                    model_reasons.append(f"Claim {idx}: {reason}")

            if model_failed_indices:
                unique_failures = sorted(list(set(model_failed_indices)))
                return VerificationOutcome(
                    passed=False,
                    final_route=Route.LEGAL_COUNSEL,
                    reason="Passage check failed: " + "; ".join(model_reasons),
                    failed_claims=unique_failures,
                )

        # All checks passed
        return VerificationOutcome(
            passed=True,
            final_route=Route.ANSWER,
            reason="All claims verified and grounded in provisions in force.",
            failed_claims=[],
        )

    def _check_claim_against_passages(
        self, claim_text: str, passages: list[Chunk]
    ) -> tuple[bool, str]:
        """Query the model with response_schema=Verdict to check if passages support claim."""
        # The chunk's in-force/repealed status is metadata, not part of the
        # statutory text. Leaving it out asked the model to confirm claims like
        # "the 1972 Act is repealed" from passages that never say so, so every
        # correct answer naming the repeal was refused. Stage 1 already trusts
        # this field, so the checker must see it too.
        passages_text = "\n\n".join(
            f"[{p.chunk_id}] {p.doc} section {p.section} ({p.part}) "
            f"[status: {p.status.value}]:\n{p.text}"
            for p in passages
        )

        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "You are an expert legal compliance verifier. "
                    "Verify whether factual and statutory assertions in the claim are strictly "
                    "supported and substantiated by the cited statutory passages. "
                    "If unsupported, exaggerated, or contradicted, mark supported as false."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Statutory Passages:\n{passages_text}\n\n"
                    f"Claim to verify:\n\"{claim_text}\"\n\n"
                    "Do the statutory passages above strictly support and substantiate this claim?"
                ),
            },
        ]

        try:
            verdict = self.model.complete(messages, response_schema=Verdict)
            if isinstance(verdict, Verdict):
                return verdict.supported, verdict.reason
            return False, "Model did not return a valid Verdict instance"
        except (ModelError, Exception) as exc:
            logger.warning("Model claim-vs-passage verification failed with error: %s", exc)
            return False, f"Model verification error: {exc}"


class ScriptedVerifier(Verifier):
    """Deterministic verifier for tests so test suites do not depend on an API key.

    Performs full citation resolution, and evaluates claim verdicts against
    a scripted dictionary or default boolean.
    """

    def __init__(
        self,
        claim_verdicts: dict[int, bool] | None = None,
        default_support: bool = True,
        reasons: dict[int, str] | None = None,
    ) -> None:
        super().__init__(model=None)
        self.claim_verdicts = claim_verdicts or {}
        self.default_support = default_support
        self.reasons = reasons or {}

    def verify(self, result: AnalystResult) -> VerificationOutcome:
        # Run citation resolution first
        outcome = super().verify(result)
        if not outcome.passed:
            return outcome

        # If citation resolution passed, check scripted claim verdicts
        failed_indices: list[int] = []
        reasons: list[str] = []

        for idx in range(len(result.claims)):
            supported = self.claim_verdicts.get(idx, self.default_support)
            if not supported:
                failed_indices.append(idx)
                reason = self.reasons.get(idx, f"Scripted claim {idx} unsupported")
                reasons.append(reason)

        if failed_indices:
            return VerificationOutcome(
                passed=False,
                final_route=Route.LEGAL_COUNSEL,
                reason="Scripted passage check failed: " + "; ".join(reasons),
                failed_claims=failed_indices,
            )

        return VerificationOutcome(
            passed=True,
            final_route=Route.ANSWER,
            reason="All claims verified (scripted).",
            failed_claims=[],
        )


# Default module-level verifier instance and verify() function
_default_verifier = Verifier()


def verify(result: AnalystResult) -> VerificationOutcome:
    """Verify an AnalystResult using the default verifier."""
    return _default_verifier.verify(result)
