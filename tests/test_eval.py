"""The eval suite: 20 scenarios through the real loop and the real Verifier.

Each scenario supplies its own scripted model transcript, so the LangGraph
loop, the tool dispatch, the citation resolution and the Verifier all run for
real. Only the model's judgement is scripted -- that is the part a live
provider supplies, and the part the live-marked tests in
``test_openai_adapter.py`` cover separately.

The counts are deliberately not asserted as "20/20". A benchmark that only
asks questions the system was built for measures nothing. The scenarios in
``ESCALATE`` and the ``rep-cites-1972-only`` trap are expected to be *hard*;
if they start passing trivially, the Verifier has stopped working.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent.analyst import analyse
from agent.model import ScriptedModel
from agent.verifier import Verifier
from app.main import app
from app.store import quarantined_records, runs
from contracts import Route, RunStatus
from eval_scenarios import (
    ALL_SCENARIOS,
    ESCALATE,
    ESTABLISHMENT,
    INJECTION_QUESTIONS,
    Scenario,
)


def _request(question: str) -> dict[str, object]:
    return {"question": question, "establishment": dict(ESTABLISHMENT)}


def _run_scenario(scenario: Scenario):
    """Drive one scenario through the real analyst loop, then verify it.

    The verifier deliberately gets **no** model. ``Verifier()`` with
    ``model=None`` runs the three deterministic checks -- citation
    resolution, cite-nothing, repealed-only -- and skips claim-vs-passage.
    That is the right split for an eval: the deterministic half must agree
    with the scenario table, while the model half depends on a live
    provider and is covered by the live-marked adapter tests. Passing a
    scripted verifier here would make the scenario's own script the
    oracle, which measures nothing.
    """
    model = ScriptedModel(list(scenario.script))
    result = analyse(
        scenario.question,
        _request(scenario.question),
        model=model,
        correlation_id=f"eval-{scenario.id}",
    )
    outcome = Verifier().verify(result)
    return result, outcome


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.id)
def test_scenario_route(scenario: Scenario):
    """The analyst must reach the expected route for every scenario."""
    result, _ = _run_scenario(scenario)

    assert result.route == scenario.expected_route, (
        f"{scenario.id}: expected route {scenario.expected_route.value}, "
        f"got {result.route.value}.\n"
        f"  rationale: {scenario.rationale}\n"
        f"  answer: {result.draft_answer!r}"
    )


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.id)
def test_scenario_citations(scenario: Scenario):
    """Required citations survive; forbidden ones never appear."""
    result, _ = _run_scenario(scenario)

    cited: set[str] = set()
    for claim in result.claims:
        cited.update(claim.citations)

    missing = scenario.required_citations - cited
    assert not missing, (
        f"{scenario.id}: required citations missing: {sorted(missing)}.\n"
        f"  cited: {sorted(cited)}\n"
        f"  rationale: {scenario.rationale}"
    )

    forbidden = scenario.forbidden_citations & cited
    assert not forbidden, (
        f"{scenario.id}: forbidden citations present: {sorted(forbidden)}.\n"
        f"  rationale: {scenario.rationale}"
    )


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.id)
def test_scenario_verification(scenario: Scenario):
    """Where a verification outcome is specified, it must match."""
    if scenario.expect_verified is None:
        pytest.skip("verification outcome not specified for this scenario")

    _, outcome = _run_scenario(scenario)
    assert outcome.passed is scenario.expect_verified, (
        f"{scenario.id}: expected verified={scenario.expect_verified}, "
        f"got {outcome.passed} (reason: {outcome.reason}).\n"
        f"  rationale: {scenario.rationale}"
    )


@pytest.mark.parametrize("scenario", ESCALATE, ids=lambda s: s.id)
def test_escalation_carries_no_citations(scenario: Scenario):
    """An escalation must not smuggle an ungrounded answer out with it."""
    result, outcome = _run_scenario(scenario)

    assert result.route == Route.LEGAL_COUNSEL
    assert not result.claims, "an escalation must not carry claims"
    assert not result.draft_answer.strip(), "an escalation must not carry an answer"
    assert outcome.final_route == Route.LEGAL_COUNSEL


def test_repealed_only_answer_fails_verification():
    """The trap that justifies the Verifier's existence.

    A well-formed answer citing only the repealed 1972 Act must be refused.
    If this ever passes, the repealed-provision check has regressed and the
    system will confidently advise on law that no longer applies.
    """
    scenario = next(s for s in ALL_SCENARIOS if s.id == "rep-cites-1972-only")
    result, outcome = _run_scenario(scenario)

    assert result.route == Route.ANSWER, "the analyst does produce an answer"
    assert outcome.passed is False, (
        "an answer grounded solely in a repealed provision must not verify"
    )
    assert outcome.final_route == Route.LEGAL_COUNSEL


@pytest.mark.parametrize("case_id,question", INJECTION_QUESTIONS, ids=lambda v: v if isinstance(v, str) else "")
def test_injection_attempt_never_reaches_the_analyst(case_id: str, question: str):
    """Group 5: the guard rejects before a run exists."""
    runs.clear()
    quarantined_records.clear()

    client = TestClient(app)
    response = client.post("/runs", json=_request(question))

    assert response.status_code == 400, f"{case_id} was accepted"
    body = response.json()
    assert body["status"] == RunStatus.QUARANTINED
    assert body["errors"][0]["field"] == "question"
    assert body["errors"][0]["code"] == "prompt_injection"
    assert not runs, "an injection attempt must not create a run"


def test_eval_set_is_twenty_scenarios():
    """The count is a deliverable, so it is asserted rather than assumed.

    Twenty covers the whole lifecycle: seventeen reach the agent (six
    answerable, four flagship-shaped, four that must escalate, three
    repealed-law traps) and three are injection attempts stopped at ingestion
    before a run exists. Counting only the agent half would report 17 and
    quietly drop the security cases.
    """
    agent_total = len(ALL_SCENARIOS)
    injection_total = len(INJECTION_QUESTIONS)

    assert agent_total == 17, (
        f"expected 17 agent scenarios, found {agent_total}: "
        f"{[s.id for s in ALL_SCENARIOS]}"
    )
    assert injection_total == 3, (
        f"expected 3 injection scenarios, found {injection_total}"
    )
    assert agent_total + injection_total == 20, (
        "the eval set is 20 scenarios across the full request lifecycle"
    )


def test_every_scenario_has_a_rationale():
    """A scenario whose rationale cannot be stated is testing nothing."""
    for scenario in ALL_SCENARIOS:
        assert len(scenario.rationale) > 40, (
            f"{scenario.id} has no meaningful rationale"
        )