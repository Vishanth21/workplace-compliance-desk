"""Tests for the shared model interface (agent/model.py).

These are the contract B builds the verifier against. If a change breaks a
test here, it breaks the other agent too.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent.model import Model, ModelError, ScriptedModel, get_model


class Verdict(BaseModel):
    """Stand-in for the verifier's structured reply."""

    supported: bool
    reason: str


def test_scripted_returns_text_in_order():
    m = ScriptedModel(["first", "second"])
    assert m.complete([{"role": "user", "content": "a"}]) == "first"
    assert m.complete([{"role": "user", "content": "b"}]) == "second"


def test_scripted_records_calls():
    m = ScriptedModel(["ok"])
    msgs = [{"role": "system", "content": "be terse"}]
    m.complete(msgs)
    assert m.calls == [msgs]


def test_scripted_exhaustion_is_loud():
    m = ScriptedModel([])
    with pytest.raises(ModelError, match="ran out of responses"):
        m.complete([{"role": "user", "content": "hi"}])


def test_scripted_structured_from_model_instance():
    m = ScriptedModel([Verdict(supported=True, reason="grounded")])
    out = m.complete([{"role": "user", "content": "x"}], response_schema=Verdict)
    assert isinstance(out, Verdict)
    assert out.supported is True


def test_scripted_structured_from_dict():
    m = ScriptedModel([{"supported": False, "reason": "chunk not retrieved"}])
    out = m.complete([{"role": "user", "content": "x"}], response_schema=Verdict)
    assert out == Verdict(supported=False, reason="chunk not retrieved")


def test_scripted_structured_from_json_string():
    m = ScriptedModel(['{"supported": true, "reason": "quoted"}'])
    out = m.complete([{"role": "user", "content": "x"}], response_schema=Verdict)
    assert out.supported is True


def test_scripted_bad_json_is_a_model_error():
    m = ScriptedModel(["not json at all"])
    with pytest.raises(ModelError, match="not valid JSON"):
        m.complete([{"role": "user", "content": "x"}], response_schema=Verdict)


def test_scripted_model_object_where_text_expected_fails():
    m = ScriptedModel([Verdict(supported=True, reason="x")])
    with pytest.raises(ModelError, match="unstructured text"):
        m.complete([{"role": "user", "content": "x"}])


def test_scripted_satisfies_protocol():
    assert isinstance(ScriptedModel([]), Model)


def test_get_model_defaults_to_scripted(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert get_model().name == "scripted"


def test_get_model_unknown_provider_names_the_known_ones(monkeypatch):
    """An unknown name must list the providers that do exist.

    This used to use 'openai' as the unknown example. openai is now an
    implemented provider (see docs/adr/0002), so the example moved to a name
    that is still genuinely unknown -- the property under test is the error
    message, not which provider happens to be missing.
    """
    monkeypatch.setenv("LLM_PROVIDER", "cohere")
    with pytest.raises(ModelError, match="unknown LLM_PROVIDER 'cohere'"):
        get_model()


def test_get_model_honours_explicit_arg(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert get_model("scripted").name == "scripted"


def test_get_model_builds_scripted_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "scripted")
    monkeypatch.setenv("LLM_SCRIPTED_RESPONSES", '["hello", "world"]')
    m = get_model()
    assert m.complete([{"role": "user", "content": "x"}]) == "hello"