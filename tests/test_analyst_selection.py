"""Tests for analyst selection at the pipeline seam.

The service used to bind ``StubAnalyst`` unconditionally, so every question --
whatever it asked -- came back with the flagship fixed-term-gratuity advisory.
These tests pin the fix: ``ANALYST_MODE`` chooses the analyst, ``stub`` stays
the default, and ``real`` reaches the actual LangGraph loop.

The distinguishing property is that a stub answers *one* canned question. A
real analyst, given a model that cannot answer, escalates instead of inventing
the flagship answer. That difference is what these tests assert on.
"""

import pytest

from app.analyst import RealAnalyst, StubAnalyst
from app.pipeline_stub import _build_analyst


def test_default_mode_is_stub(monkeypatch):
    """An unset ANALYST_MODE keeps the deterministic stub."""
    monkeypatch.delenv("ANALYST_MODE", raising=False)
    assert isinstance(_build_analyst(), StubAnalyst)


def test_empty_mode_is_stub(monkeypatch):
    """A blank ANALYST_MODE is treated as unset, not as an error."""
    monkeypatch.setenv("ANALYST_MODE", "   ")
    assert isinstance(_build_analyst(), StubAnalyst)


def test_real_mode_selects_real_analyst(monkeypatch):
    """ANALYST_MODE=real reaches the bounded LangGraph loop."""
    monkeypatch.setenv("ANALYST_MODE", "real")
    assert isinstance(_build_analyst(), RealAnalyst)


def test_mode_is_case_and_whitespace_insensitive(monkeypatch):
    """Environment values arrive from compose files and shells; normalise them."""
    monkeypatch.setenv("ANALYST_MODE", "  REAL  ")
    assert isinstance(_build_analyst(), RealAnalyst)


def test_unknown_mode_fails_loudly(monkeypatch):
    """A typo must not silently fall back to the stub.

    Falling back would mean a deployment that believes it runs the real agent
    and does not. Better to refuse to start.
    """
    monkeypatch.setenv("ANALYST_MODE", "reel")
    with pytest.raises(ValueError, match="unknown ANALYST_MODE 'reel'"):
        _build_analyst()


@pytest.mark.asyncio
async def test_stub_answers_every_question_identically():
    """Document the stub's limitation the real analyst exists to fix.

    This is not a bug to preserve; it is the baseline. If this test ever fails,
    the stub grew behaviour and the reason for ANALYST_MODE should be revisited.
    """
    from contracts import Route, RunRequest

    request = RunRequest(
        question="What is the notice period for resignation?",
        establishment={"headcount": 10, "industry": "Retail", "state": "Delhi"},
    )
    result = await StubAnalyst().analyze(request, correlation_id="c-1")

    # The stub answers the gratuity question no matter what was asked.
    assert "fixed term" in result.draft_answer.lower()
    assert result.route == Route.ANSWER


@pytest.mark.asyncio
async def test_real_analyst_escalates_when_it_cannot_answer(monkeypatch):
    """With no usable model the real loop escalates rather than inventing.

    An empty scripted model raises on first call. ``think`` converts that into
    a counsel route, so the caller sees an escalation -- never the flagship
    advisory the stub would have returned.
    """
    from contracts import Route, RunRequest

    monkeypatch.setenv("LLM_PROVIDER", "scripted")
    monkeypatch.delenv("LLM_SCRIPTED_RESPONSES", raising=False)

    request = RunRequest(
        question="What is the notice period for resignation?",
        establishment={"headcount": 10, "industry": "Retail", "state": "Delhi"},
    )
    result = await RealAnalyst().analyze(request, correlation_id="c-2")

    assert result.route == Route.LEGAL_COUNSEL
    assert result.draft_answer == ""
    # The decisive assertion: the stub's canned answer did not leak through.
    assert "fixed term" not in result.draft_answer.lower()