"""Tests for Requirement L2: Prompt versioning and comparative evaluation."""

from __future__ import annotations

import pytest

from agent.analyst import analyse
from agent.model import ScriptedModel
from agent.prompts import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_VARIANTS,
    PROMPT_V1,
    PROMPT_V2,
    get_prompt_text,
)
from contracts import Route
from eval_scenarios import ALL_SCENARIOS, Scenario, tool, answer


def test_prompt_variants_exist_and_distinct():
    """At least two distinct prompt variants are registered (L2)."""
    assert "analyst-1" in PROMPT_VARIANTS
    assert "analyst-2" in PROMPT_VARIANTS
    assert PROMPT_VARIANTS["analyst-1"] != PROMPT_VARIANTS["analyst-2"]
    assert "Provisos Control Over General Rules" in PROMPT_VARIANTS["analyst-2"]


def test_prompt_version_recorded_on_response():
    """AnalystResult records the exact prompt version used (L2)."""
    model1 = ScriptedModel([answer("Advisory 1", cites=["css2020-s53-main"])])
    res1 = analyse(
        "Is gratuity payable?",
        {"question": "Is gratuity payable?"},
        model=model1,
        prompt_version="analyst-1",
    )
    assert res1.prompt_version == "analyst-1"

    model2 = ScriptedModel([answer("Advisory 2", cites=["css2020-s53-main"])])
    res2 = analyse(
        "Is gratuity payable?",
        {"question": "Is gratuity payable?"},
        model=model2,
        prompt_version="analyst-2",
    )
    assert res2.prompt_version == "analyst-2"


def test_system_prompt_passed_to_model():
    """The model receives the exact prompt text corresponding to the requested version."""
    model_v1 = ScriptedModel([answer("ok")])
    analyse("Query", {}, model=model_v1, prompt_version="analyst-1")
    assert model_v1.calls[0][0]["content"] == PROMPT_V1

    model_v2 = ScriptedModel([answer("ok")])
    analyse("Query", {}, model=model_v2, prompt_version="analyst-2")
    assert model_v2.calls[0][0]["content"] == PROMPT_V2


def test_compare_prompt_variants_on_eval_suite():
    """Compare analyst-1 and analyst-2 across the evaluation scenarios (L2)."""
    results_v1: list[dict] = []
    results_v2: list[dict] = []

    for sc in ALL_SCENARIOS:
        # Run under analyst-1
        m1 = ScriptedModel(list(sc.script))
        r1 = analyse(
            sc.question,
            {"question": sc.question, "establishment": {"headcount": 50}},
            model=m1,
            prompt_version="analyst-1",
            correlation_id=f"cmp-v1-{sc.id}",
        )
        results_v1.append({
            "id": sc.id,
            "route_match": r1.route == sc.expected_route,
            "steps": r1.loop.steps,
            "tool_calls": r1.loop.tool_calls,
        })

        # Run under analyst-2
        m2 = ScriptedModel(list(sc.script))
        r2 = analyse(
            sc.question,
            {"question": sc.question, "establishment": {"headcount": 50}},
            model=m2,
            prompt_version="analyst-2",
            correlation_id=f"cmp-v2-{sc.id}",
        )
        results_v2.append({
            "id": sc.id,
            "route_match": r2.route == sc.expected_route,
            "steps": r2.loop.steps,
            "tool_calls": r2.loop.tool_calls,
        })

    v1_acc = sum(1 for r in results_v1 if r["route_match"]) / len(results_v1)
    v2_acc = sum(1 for r in results_v2 if r["route_match"]) / len(results_v2)

    assert v1_acc == 1.0, "analyst-1 must route all scenarios correctly"
    assert v2_acc == 1.0, "analyst-2 must route all scenarios correctly"

