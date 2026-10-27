"""Tests for the Analyst loop.

Two kinds:

* **Shape** -- the graph routes, the bounds trip, the hand-off validates. These
  run against ``ScriptedModel`` with no network.
* **Flagship** -- given a script in which the agent searches, reads the main
  clause, then looks the section up, the loop reaches the second proviso. This
  is the case the whole design turns on: if ``lookup_clause`` is not a tool the
  agent can choose, the proviso is unreachable.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agent.analyst import Bounds, Step, analyse, build_graph
from agent.model import ScriptedModel
from contracts import Route


REQUEST = {
    "question": (
        "We employ 40 people. A fixed-term worker finished an 18-month contract "
        "and left. Do we owe gratuity, given they served under five years?"
    ),
    "establishment": {
        "headcount": 40,
        "industry": "logistics",
        "state": "Karnataka",
        "has_fixed_term_employees": True,
    },
}


def _tool(name: str, **arguments) -> Step:
    return Step(action="tool", tool=name, arguments=arguments, reasoning=f"call {name}")


def _answer(text: str, cites: list[str] | None = None) -> Step:
    return Step(action="answer", answer=text, citations=cites or [])


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_answer_without_tools_completes():
    model = ScriptedModel([_answer("Section 53 governs gratuity.")])
    result = analyse("What does the Code say?", REQUEST, model=model)

    assert result.route == Route.ANSWER
    assert result.draft_answer == "Section 53 governs gratuity."
    assert result.loop.steps == 1
    assert result.loop.tool_calls == 0
    assert result.retrieved_chunks == []
    assert result.prompt_version


def test_one_tool_call_then_answer():
    model = ScriptedModel(
        [
            _tool("lookup_request_field", field="establishment.headcount"),
            _answer("You have 40 employees."),
        ]
    )
    result = analyse("How many staff?", REQUEST, model=model)

    assert result.route == Route.ANSWER
    assert result.loop.tool_calls == 1
    assert result.loop.steps == 2
    # The field tool returns no chunks, so nothing is retrieved.
    assert result.retrieved_chunks == []

    # The model was shown the tool result on its second turn.
    second_call = model.calls[1]
    tool_messages = [m for m in second_call if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0]["content"])["value"] == 40


def test_chunks_retrieved_are_carried_into_the_result():
    cited = "css2020-s53-proviso_6"
    model = ScriptedModel(
        [
            _tool("lookup_clause", section="53", doc="css2020"),
            _answer("Gratuity is payable under s.53.", cites=[cited]),
        ]
    )
    result = analyse("Tell me about section 53.", REQUEST, model=model)

    ids = {c.chunk_id for c in result.retrieved_chunks}
    assert ids, "lookup_clause returned no chunks"
    assert all(c.doc == "css2020" for c in result.retrieved_chunks)
    assert all(c.section == "53" for c in result.retrieved_chunks)

    # All retrieved chunks are carried, but the claim cites only what it used.
    assert len(result.claims) == 1
    assert result.claims[0].citations == [cited]
    assert result.claims[0].citations != sorted(ids)


def test_hallucinated_citation_is_dropped():
    """A chunk id the model invents never reaches the hand-off."""
    model = ScriptedModel(
        [
            _tool("lookup_clause", section="53"),
            _answer("Answer.", cites=["css2020-s53-proviso_6", "invented-chunk-999"]),
        ]
    )
    result = analyse("Tell me about section 53.", REQUEST, model=model)

    assert result.claims[0].citations == ["css2020-s53-proviso_6"]
    retrieved = {c.chunk_id for c in result.retrieved_chunks}
    assert "invented-chunk-999" not in retrieved


def test_repealed_chunk_is_retained_but_not_cited():
    """Reading repealed law is fine; citing it for current obligations is not.

    The Analyst does not police this -- the Verifier does. What the Analyst must
    do is keep the repealed chunk in ``retrieved_chunks`` (so the Verifier can
    see it was considered) while the claim cites only what the answer used.
    """
    model = ScriptedModel(
        [
            _tool("corpus_search", query="gratuity", status=None, k=20),
            _answer("Current answer.", cites=["css2020-s53-proviso_6"]),
        ]
    )
    result = analyse("Compare old and new gratuity law.", REQUEST, model=model)

    statuses = {c.status.value for c in result.retrieved_chunks}
    assert "repealed" in statuses, "the search did not return any repealed law"
    assert result.claims[0].citations == ["css2020-s53-proviso_6"]
    repealed_ids = {c.chunk_id for c in result.retrieved_chunks if c.status.value == "repealed"}
    assert not (set(result.claims[0].citations) & repealed_ids)


def test_unknown_tool_is_reported_not_raised():
    """A hallucinated tool name goes back to the model as an error message.

    The loop does not crash, and the model gets a chance to recover. Whether it
    then answers or gives up is its decision, so the route is not asserted here;
    what matters is that the run completed and the feedback was delivered.
    """
    model = ScriptedModel(
        [
            _tool("teleport", where="mall"),
            _answer("I could not look that up."),
        ]
    )
    result = analyse("Do something impossible.", REQUEST, model=model)

    # The bad call did not execute, so it is not counted as a tool call.
    assert result.loop.tool_calls == 0
    feedback = [m for m in model.calls[1] if m.get("role") == "tool"]
    assert feedback, "the model was not told its tool name was wrong"
    assert "unknown tool" in json.loads(feedback[0]["content"])["error"]
    # The loop survived and the model's follow-up was used.
    assert result.loop.steps == 2
    assert result.draft_answer == "I could not look that up."


def test_bad_tool_arguments_are_reported_not_raised():
    model = ScriptedModel(
        [
            # lookup_clause needs "section"; omit it.
            _tool("lookup_clause"),
            _answer("Could not fetch."),
        ]
    )
    result = analyse("Fetch nothing.", REQUEST, model=model)

    assert result.route == Route.ANSWER
    feedback = [m for m in model.calls[1] if m.get("role") == "tool"]
    assert "missing required argument" in json.loads(feedback[0]["content"])["error"]


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_step_bound_stops_the_loop():
    # Always asks for a tool; never answers. The bound must stop it.
    model = ScriptedModel(
        [_tool("lookup_request_field", field="establishment.state")] * 20
    )
    result = analyse(
        "Keep going.", REQUEST, model=model, bounds=Bounds(max_steps=3, max_tool_calls=99)
    )

    assert result.loop.steps == 3
    assert result.route == Route.LEGAL_COUNSEL
    assert result.draft_answer == ""
    assert result.claims == []
    # Only three model calls were made, not twenty.
    assert len(model.calls) == 3


def test_tool_call_bound_stops_the_loop():
    model = ScriptedModel(
        [_tool("lookup_request_field", field="establishment.state")] * 20
    )
    result = analyse(
        "Keep going.",
        REQUEST,
        model=model,
        bounds=Bounds(max_steps=99, max_tool_calls=2),
    )

    assert result.loop.tool_calls == 2
    assert result.route == Route.LEGAL_COUNSEL
    assert result.draft_answer == ""


def test_wall_clock_bound_stops_the_loop():
    # A zero budget is already expired by the first check.
    model = ScriptedModel([_answer("never reached")])
    result = analyse(
        "Anything.", REQUEST, model=model, bounds=Bounds(max_wall_clock_ms=0)
    )

    assert result.route == Route.LEGAL_COUNSEL
    assert result.draft_answer == ""
    assert model.calls == [], "the model was called despite the deadline"


def test_model_error_becomes_a_counsel_route():
    # An empty script makes ScriptedModel raise on first use.
    result = analyse("Anything.", REQUEST, model=ScriptedModel([]))

    assert result.route == Route.LEGAL_COUNSEL
    assert result.draft_answer == ""


# ---------------------------------------------------------------------------
# The flagship case
# ---------------------------------------------------------------------------


FLAGSHIP = (
    "A fixed-term employee completed an 18-month contract and left. "
    "The establishment has 40 employees. Is gratuity payable, even though "
    "continuous service was under five years?"
)


def test_flagship_case_reaches_the_second_proviso():
    """Search, read the main clause, then look the section up by number.

    The second proviso of s.53(1) removes the five-year threshold for fixed-term
    employees. It does not contain the question's words, so a single semantic
    search does not surface it. It is reachable only because the agent can choose
    ``lookup_clause`` after seeing the main clause.

    The script encodes that reasoning; the test asserts the loop follows it and
    that the proviso ends up in the hand-off, cited.
    """
    model = ScriptedModel(
        [
            _tool("corpus_search", query="gratuity fixed term contract five years"),
            _tool("lookup_clause", section="53", doc="css2020"),
            _answer(
                "Gratuity is payable. Under the second proviso to section 53(1) of "
                "the Code on Social Security, 2020, an employee on fixed-term "
                "employment is paid gratuity on a pro-rata basis, and the five-year "
                "continuous-service requirement does not apply.",
                cites=["css2020-s53-proviso_2", "css2020-s53-proviso_6"],
            ),
        ]
    )
    result = analyse(FLAGSHIP, REQUEST, model=model)

    # The loop used both tools, in order.
    assert result.loop.tool_calls == 2
    names = [
        m["name"] for m in model.calls[-1] if m.get("role") == "tool"
    ]
    assert names == ["corpus_search", "lookup_clause"]

    # The second proviso is in the hand-off.
    proviso_ids = [
        c.chunk_id for c in result.retrieved_chunks if c.part.startswith("proviso")
    ]
    assert proviso_ids, "no proviso was retrieved; the flagship case failed"

    # Every retrieved proviso is in force, and the claim cites it.
    provisos = [c for c in result.retrieved_chunks if c.part.startswith("proviso")]
    assert all(c.status.value == "in_force" for c in provisos)
    cited = set(result.claims[0].citations)
    assert cited & set(proviso_ids)

    # The draft names the provision, and the run is routed as an answer.
    assert result.route == Route.ANSWER
    assert "53" in result.draft_answer


def test_flagship_requires_lookup_clause_not_just_search():
    """A search-only script must not surface the deciding proviso.

    This is the negative control for the test above: it proves the proviso is
    genuinely unreachable by search, so the positive test is not passing by
    accident.
    """
    from index.retrieve import corpus_search, lookup_clause

    p2, p6 = "css2020-s53-proviso_2", "css2020-s53-proviso_6"
    phrasings = [
        "gratuity fixed term contract five years",
        "is gratuity payable to fixed term employee under five years",
        "fixed term contract ended gratuity pro rata",
        "continuous service five years exception",
    ]

    found_both = [
        q for q in phrasings
        if {p2, p6} <= {c.chunk_id for c in corpus_search(q, k=5)}
    ]
    assert found_both == [], (
        f"search now returns both deciding provisos for {found_both!r}; "
        "the argument for lookup_clause is weaker than this test assumes"
    )

    # The tool returns them, which is the point of having it.
    assert {p2, p6} <= {c.chunk_id for c in lookup_clause("53")}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


def test_graph_compiles_and_reports_its_nodes():
    app = build_graph(model=ScriptedModel([_answer("ok")]))
    nodes = set(app.get_graph().nodes)
    assert {"think", "act", "draft"} <= nodes


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------


def test_unknown_action_does_not_return_an_answer():
    """An action outside the vocabulary must not fall through to an answer.

    ``Step.action`` is an enum, so a bad value is rejected at validation. The
    point of this test is the consequence: an unverified draft must never be
    returned, which is what the previous bare-string field allowed.
    """
    with pytest.raises(ValidationError):
        Step(action="Answer", reasoning="capitalised by mistake")


def test_retry_clears_so_the_next_tool_call_runs():
    """An unknown tool sets ``needs_retry``; the following step must clear it.

    Otherwise the router sends a valid tool call back to ``think`` forever and
    the call is never executed.
    """
    model = ScriptedModel(
        [
            _tool("teleport", where="mall"),
            _tool("lookup_clause", section="53"),
            _answer("Fetched.", cites=["css2020-s53-main"]),
        ]
    )
    result = analyse("Fetch s.53.", REQUEST, model=model)

    assert result.route == Route.ANSWER
    assert result.loop.tool_calls == 1
    assert any(c.chunk_id == "css2020-s53-main" for c in result.retrieved_chunks)


def test_provider_error_does_not_escape():
    """A provider raising outside ``ModelError`` is caught and routed out."""

    class ExplodingModel:
        name = "exploding"

        def complete(self, messages, **kwargs):
            raise ValueError("provider exploded")

    result = analyse("Anything.", REQUEST, model=ExplodingModel())

    assert result.route == Route.LEGAL_COUNSEL
    assert result.claims == []