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
) -> AnalystResult:
    return AnalystResult(
        route=route,
        draft_answer=draft_answer,
        claims=claims,
        retrieved_chunks=retrieved_chunks,
        loop=LoopStats(),
        prompt_version="v1",
        correlation_id="test-corr-id-123",
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
