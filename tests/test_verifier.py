"""Unit tests for the Verifier agent.

Covers:
- Citation resolution (unretrieved chunk id, empty citations, repealed-only citations)
- Model claim-vs-passage verification with agent.model.ScriptedModel
- StubVerifier and ScriptedVerifier contracts
- Verification outcome structure and failed_claims indices
"""

import pytest

from agent.model import ScriptedModel
from agent.verifier import (
    ScriptedVerifier,
    StubVerifier,
    Verdict,
    Verifier,
    verify,
)
from contracts import (
    AnalystResult,
    Chunk,
    Claim,
    LoopStats,
    ProvisionStatus,
    Route,
)


def make_sample_chunk(
    chunk_id: str = "css2020-s53-proviso_2",
    status: ProvisionStatus = ProvisionStatus.IN_FORCE,
    part: str = "proviso_2",
    section: str = "53(1)",
    text: str = "Gratuity payable on pro rata basis for fixed term employees.",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc="css2020",
        section=section,
        part=part,
        status=status,
        text=text,
    )


def make_analyst_result(
    claims: list[Claim],
    retrieved_chunks: list[Chunk],
    route: Route = Route.ANSWER,
    draft_answer: str = "Sample draft answer.",
    stop_reason: str | None = None,
) -> AnalystResult:
    return AnalystResult(
        route=route,
        draft_answer=draft_answer,
        claims=claims,
        retrieved_chunks=retrieved_chunks,
        loop=LoopStats(),
        prompt_version="v1",
        correlation_id="test-corr-id-123",
        stop_reason=stop_reason,
    )


# --- 1. StubVerifier tests ---

def test_stub_verifier_accepts_analyst_result():
    verifier = StubVerifier()
    chunk = make_sample_chunk()
    claim = Claim(text="Fixed term employees get gratuity.", citations=[chunk.chunk_id])
    result = make_analyst_result([claim], [chunk])

    outcome = verifier.verify(result)
    assert outcome.passed is True
    assert outcome.final_route == Route.ANSWER
    assert outcome.failed_claims == []


# --- 2. Deterministic Citation Resolution tests (A3) ---

def test_citation_resolution_valid():
    verifier = Verifier()
    chunk = make_sample_chunk()
    claim = Claim(text="Pro rata gratuity is payable.", citations=[chunk.chunk_id])
    result = make_analyst_result([claim], [chunk])

    outcome = verifier.verify(result)
    assert outcome.passed is True
    assert outcome.final_route == Route.ANSWER
    assert outcome.failed_claims == []


def test_citation_resolution_fabricated_chunk_fails():
    """Gate 3: Chunk cited by Analyst is not in retrieved_chunks -> passed=False."""
    verifier = Verifier()
    chunk = make_sample_chunk(chunk_id="css2020-s53-proviso_2")
    # Claim cites a fabricated chunk id not present in retrieved_chunks
    claim = Claim(
        text="Fabricated rule assertion.",
        citations=["css2020-s99-nonexistent-prov-1"],
    )
    result = make_analyst_result([claim], [chunk])

    outcome = verifier.verify(result)
    assert outcome.passed is False
    assert outcome.final_route == Route.LEGAL_COUNSEL
    assert outcome.failed_claims == [0]
    assert "unretrieved chunk" in outcome.reason


def test_citation_resolution_empty_citations_fails():
    """A claim citing nothing fails."""
    verifier = Verifier()
    chunk = make_sample_chunk()
    claim = Claim(text="Uncited claim assertion.", citations=[])
    result = make_analyst_result([claim], [chunk])

    outcome = verifier.verify(result)
    assert outcome.passed is False
    assert outcome.final_route == Route.LEGAL_COUNSEL
    assert outcome.failed_claims == [0]
    assert "no citations" in outcome.reason


def test_citation_resolution_repealed_only_grounding_fails():
    """Gate 4: A claim grounded only in repealed provisions fails."""
    verifier = Verifier()
    repealed_chunk = make_sample_chunk(
        chunk_id="pga1972-s4-1",
        status=ProvisionStatus.REPEALED,
        part="main",
        section="4(1)",
        text="Requires 5 years continuous service under 1972 Act.",
    )
    claim = Claim(
        text="Five years continuous service is mandatory.",
        citations=["pga1972-s4-1"],
    )
    result = make_analyst_result([claim], [repealed_chunk])

    outcome = verifier.verify(result)
    assert outcome.passed is False
    assert outcome.final_route == Route.LEGAL_COUNSEL
    assert outcome.failed_claims == [0]
    assert "repealed chunk" in outcome.reason


def test_citation_resolution_multiple_claims_tracks_failed_indices():
    """Verifier records failing claim indices into failed_claims."""
    verifier = Verifier()
    in_force_chunk = make_sample_chunk(chunk_id="c-in-force", status=ProvisionStatus.IN_FORCE)
    repealed_chunk = make_sample_chunk(chunk_id="c-repealed", status=ProvisionStatus.REPEALED)

    claims = [
        Claim(text="Claim 0 is valid", citations=["c-in-force"]),
        Claim(text="Claim 1 cites unretrieved chunk", citations=["c-missing"]),
        Claim(text="Claim 2 cites repealed chunk only", citations=["c-repealed"]),
    ]
    result = make_analyst_result(claims, [in_force_chunk, repealed_chunk])

    outcome = verifier.verify(result)
    assert outcome.passed is False
    assert outcome.final_route == Route.LEGAL_COUNSEL
    assert outcome.failed_claims == [1, 2]


# --- 3. Claim-vs-Passage Model check tests (A3 / Agent) ---

def test_verifier_with_model_supported():
    mock_model = ScriptedModel(
        [Verdict(supported=True, reason="Text explicitly covers fixed term gratuity")]
    )
    verifier = Verifier(model=mock_model)
    chunk = make_sample_chunk()
    claim = Claim(text="Pro rata gratuity is payable.", citations=[chunk.chunk_id])
    result = make_analyst_result([claim], [chunk])

    outcome = verifier.verify(result)
    assert outcome.passed is True
    assert outcome.final_route == Route.ANSWER
    assert outcome.failed_claims == []
    assert len(mock_model.calls) == 1


def test_verifier_with_model_unsupported():
    mock_model = ScriptedModel(
        [Verdict(supported=False, reason="Passage does not mention 100 days notice")]
    )
    verifier = Verifier(model=mock_model)
    chunk = make_sample_chunk()
    claim = Claim(text="Notice period of 100 days applies.", citations=[chunk.chunk_id])
    result = make_analyst_result([claim], [chunk])

    outcome = verifier.verify(result)
    assert outcome.passed is False
    assert outcome.final_route == Route.LEGAL_COUNSEL
    assert outcome.failed_claims == [0]
    assert "Passage check failed" in outcome.reason
    assert "Claim 0: Passage does not mention 100 days notice" in outcome.reason


def test_verifier_shows_chunk_status_to_model():
    """The checker must see in-force/repealed, or repeal claims fail wrongly.

    Chunk status is metadata, not statutory text. A correct answer that says
    "the 1972 Act is repealed" cites passages that never contain the word, so
    if the prompt omits status the model rightly calls the claim unsupported
    and the flagship contrast -- the point of the corpus -- is always refused.
    """
    mock_model = ScriptedModel(
        [Verdict(supported=True, reason="Status confirms the provision is repealed")]
    )
    verifier = Verifier(model=mock_model)
    chunk = make_sample_chunk()
    claim = Claim(
        text="Section 4 of the 1972 Act is repealed.", citations=[chunk.chunk_id]
    )
    result = make_analyst_result([claim], [chunk])

    verifier.verify(result)

    assert len(mock_model.calls) == 1
    # calls is list[list[Message]]: one entry per complete(), each a message list.
    prompt = mock_model.calls[0][1]["content"]
    assert f"status: {chunk.status.value}" in prompt, (
        "the claim checker was not told the chunk's provision status"
    )


# --- 4. ScriptedVerifier tests ---

def test_scripted_verifier_deterministic():
    verifier = ScriptedVerifier(
        claim_verdicts={0: True, 1: False},
        reasons={1: "Passage contradicts maternity period length"},
    )
    chunk = make_sample_chunk()
    claims = [
        Claim(text="First claim is good.", citations=[chunk.chunk_id]),
        Claim(text="Second claim is unsupported.", citations=[chunk.chunk_id]),
    ]
    result = make_analyst_result(claims, [chunk])

    outcome = verifier.verify(result)
    assert outcome.passed is False
    assert outcome.final_route == Route.LEGAL_COUNSEL
    assert outcome.failed_claims == [1]


def test_top_level_verify_function():
    chunk = make_sample_chunk()
    claim = Claim(text="Valid claim.", citations=[chunk.chunk_id])
    result = make_analyst_result([claim], [chunk])

    outcome = verify(result)
    assert outcome.passed is True
    assert outcome.final_route == Route.ANSWER


# --- 4. Non-answer routes: deliberate escalation vs provider failure ---


@pytest.mark.parametrize("route", [Route.LEGAL_COUNSEL, Route.MORE_DETAIL])
def test_deliberate_escalation_passes(route: Route):
    """Declining to answer is a legitimate outcome, not a verification failure."""
    verifier = Verifier()
    result = make_analyst_result([], [], route=route, draft_answer="")

    outcome = verifier.verify(result)
    assert outcome.passed is True, "a deliberate escalation must not read as a failure"
    assert outcome.final_route == route
    assert "routed directly" in outcome.reason


def test_model_error_escalation_does_not_pass():
    """The regression this guard exists for.

    The analyst catches a ModelError and routes to counsel so the run degrades
    instead of crashing. Before this guard, ``route != ANSWER`` short-circuited
    to ``passed=True``, so a 100%-broken adapter produced a clean pass. That is
    the difference between a demo that degrades loudly and one that lies.
    """
    verifier = Verifier()
    result = make_analyst_result(
        [],
        [],
        route=Route.LEGAL_COUNSEL,
        draft_answer="",
        stop_reason="model_error: CircuitBreakerOpenError: breaker open",
    )

    outcome = verifier.verify(result)
    assert outcome.passed is False, (
        "an escalation caused by a provider failure must not verify as a pass"
    )
    # The route is unchanged -- it still goes to a human. Only the verdict differs.
    assert outcome.final_route == Route.LEGAL_COUNSEL
    assert "provider failure" in outcome.reason
    assert "breaker open" in outcome.reason
    assert outcome.failed_claims == []


def test_non_model_error_stop_reason_still_passes():
    """``stop_reason`` is also used for deliberate stops; only model_error flips."""
    verifier = Verifier()
    result = make_analyst_result(
        [], [], route=Route.MORE_DETAIL, draft_answer="", stop_reason="max_steps"
    )

    outcome = verifier.verify(result)
    assert outcome.passed is True
    assert outcome.final_route == Route.MORE_DETAIL


def test_scripted_verifier_inherits_the_model_error_guard():
    """ScriptedVerifier delegates to super().verify() first, so it inherits."""
    verifier = ScriptedVerifier()
    result = make_analyst_result(
        [],
        [],
        route=Route.LEGAL_COUNSEL,
        draft_answer="",
        stop_reason="model_error: ModelError: unreachable",
    )

    outcome = verifier.verify(result)
    assert outcome.passed is False
    assert outcome.final_route == Route.LEGAL_COUNSEL
