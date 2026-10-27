"""Analyst: reads the corpus and drafts a cited answer.

A LangGraph state machine where the model decides, at each step, whether to
call a tool or answer.

    START -> think --(tool call)--> act --> think --> ... --> draft -> END
                  \\--(answer / limit)--> draft

Three bounds, enforced in ``think``: steps, tool calls, wall-clock time. When a
bound trips the loop stops and ``draft`` reports the reason in ``loop``, so a
truncated run is never mistaken for a complete one.

The model is injected, so the graph runs against ``ScriptedModel`` with no
network and no API key.
"""

from __future__ import annotations

import json
import time
import uuid
from enum import StrEnum
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from contracts import AnalystResult, Chunk, Claim, LoopStats, Route

from .model import Message, Model, get_model
from .tools import TOOL_NAMES, dispatch, schemas

#: Bump when a prompt change alters output, so regressions trace to a revision.
PROMPT_VERSION = "analyst-1"

DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_TOOL_CALLS = 6
DEFAULT_MAX_WALL_CLOCK_MS = 60_000


class Action(StrEnum):
    """What the model chose to do this turn."""

    TOOL = "tool"
    ANSWER = "answer"


class Step(BaseModel):
    """One decision by the model: call a tool, or answer."""

    action: Action = Field(description="'tool' to call a tool, 'answer' to finish.")
    tool: str | None = Field(
        default=None, description="Tool name when action is 'tool'."
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool."
    )
    reasoning: str = Field(default="", description="Why this step, in one sentence.")
    answer: str = Field(
        default="",
        description="When action is 'answer': the answer text shown to the user.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description=(
            "When action is 'answer': chunk ids the answer relies on. Cite only "
            "chunks you retrieved and used, not every chunk you saw."
        ),
    )


class Bounds(BaseModel):
    """Loop limits. Each is checked before the model is asked again."""

    max_steps: int = DEFAULT_MAX_STEPS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_wall_clock_ms: int = DEFAULT_MAX_WALL_CLOCK_MS


class AnalystState(TypedDict, total=False):
    """Graph state. LangGraph merges each node's return into this."""

    messages: list[Message]
    request: dict[str, Any]
    correlation_id: str
    started_ms: float
    steps: int
    tool_calls: int
    stop_reason: str | None
    #: Chunks seen so far, keyed by id, so a repeat lookup does not duplicate.
    chunks: dict[str, Chunk]
    #: The step's rationale, once it stops calling tools.
    answer: str
    #: The answer text shown to the user.
    answer_text: str
    #: Chunk ids the model says its answer relies on.
    cited_ids: list[str]
    #: Set when the model made a mistake it can correct, so ``think`` runs again.
    needs_retry: bool
    claims: list[Claim]
    route: Route
    #: The assembled hand-off. Declared in the state so LangGraph keeps it.
    result: AnalystResult


SYSTEM_PROMPT = """You are a compliance analyst answering questions about Indian labour law.

You have three tools. You must call them; you do not know the law without them.

- corpus_search: semantic search when you do not know the section.
- lookup_clause: fetch a whole section by number. Use this whenever a search
  shows you a provision that is qualified by a proviso or explanation -- the
  words that qualify it are often absent from the question.
- lookup_request_field: read a field from the employer's request.

Rules:
1. Cite only chunk ids you have actually retrieved. Never invent one.
2. Note whether each provision is in_force or repealed. A repealed provision may
   not be the basis of an answer about current obligations.
3. If the request lacks a field you need, say which field is missing instead of
   guessing.
4. If the corpus cannot answer the question, say so rather than inventing law.
5. Answer the question directly and briefly, in plain language.

When you are ready to answer, set action to 'answer', put the reply in
``answer``, and list the chunk ids you relied on in ``citations``.
"""


def _build_messages(state: AnalystState) -> list[Message]:
    """System prompt, the question, then the tool transcript so far."""
    base: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
    base.extend(state["messages"])
    return base


def _deadline_passed(state: AnalystState, bounds: Bounds) -> bool:
    elapsed_ms = (time.perf_counter() - state["started_ms"]) * 1000
    return elapsed_ms >= bounds.max_wall_clock_ms


def think(state: AnalystState, *, model: Model, bounds: Bounds) -> dict[str, Any]:
    """Ask the model for the next step, or stop if a bound has tripped."""
    steps = state.get("steps", 0)
    tool_calls = state.get("tool_calls", 0)

    if steps >= bounds.max_steps:
        return {"stop_reason": "max_steps", "route": Route.LEGAL_COUNSEL}
    if tool_calls >= bounds.max_tool_calls:
        return {"stop_reason": "max_tool_calls", "route": Route.LEGAL_COUNSEL}
    if _deadline_passed(state, bounds):
        return {"stop_reason": "max_wall_clock_ms", "route": Route.LEGAL_COUNSEL}

    try:
        step = model.complete(
            _build_messages(state),
            response_schema=Step,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - a provider failure must not escape
        return {
            "stop_reason": f"model_error: {type(exc).__name__}: {exc}",
            "route": Route.LEGAL_COUNSEL,
        }

    assert isinstance(step, Step)

    if step.action == Action.TOOL:
        if step.tool not in TOOL_NAMES:
            # Not counted as a tool call: nothing executed. The step count still
            # advances, so a model that keeps guessing trips max_steps.
            return {
                "messages": state["messages"]
                + [
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {"error": f"unknown tool {step.tool!r}; "
                                      f"available: {list(TOOL_NAMES)}"}
                        ),
                        "name": step.tool or "",
                    }
                ],
                "steps": steps + 1,
                "needs_retry": True,
            }
        # Hand the pending call to ``act``.
        return {
            "messages": state["messages"]
            + [
                {
                    "role": "assistant",
                    "content": step.reasoning,
                    "tool_calls": [
                        {
                            "function": {
                                "name": step.tool,
                                "arguments": json.dumps(step.arguments),
                            }
                        }
                    ],
                }
            ],
            "steps": steps + 1,
            "needs_retry": False,
        }

    if step.action != Action.ANSWER:
        # An unrecognised action is a mistake the model can correct. Without this
        # the fall-through below would return an uncited answer, which A3 forbids.
        return {
            "messages": state["messages"]
            + [
                {
                    "role": "tool",
                    "content": json.dumps(
                        {"error": f"unknown action {step.action!r}; "
                                  f"expected {[a.value for a in Action]}"}
                    ),
                    "name": "",
                }
            ],
            "steps": steps + 1,
            "needs_retry": True,
        }

    return {
        "steps": steps + 1,
        "answer": step.reasoning,
        "answer_text": step.answer,
        # Filtered against what was retrieved in ``draft``.
        "cited_ids": list(step.citations),
        "route": Route.ANSWER,
        "needs_retry": False,
    }


def act(state: AnalystState, *, bounds: Bounds) -> dict[str, Any]:
    """Execute the tool call the last assistant message asked for."""
    last = state["messages"][-1]
    calls = last.get("tool_calls") or []
    if not calls:
        return {}

    call = calls[0]["function"]
    name, raw_args = call["name"], call["arguments"]
    result = dispatch(name, raw_args, state["request"])

    # The Verifier checks citations against this set and nothing else.
    chunks = dict(state.get("chunks", {}))
    if name != "lookup_request_field":
        for item in json.loads(result):
            if isinstance(item, dict) and "chunk_id" in item:
                chunks[item["chunk_id"]] = Chunk.model_validate(item)

    return {
        "messages": state["messages"]
        + [{"role": "tool", "content": result, "name": name}],
        "tool_calls": state.get("tool_calls", 0) + 1,
        "chunks": chunks,
    }


def _pending_tool_call(state: AnalystState) -> bool:
    """True when the last message is an assistant turn asking for a tool."""
    messages = state.get("messages") or []
    if not messages:
        return False
    last = messages[-1]
    return last.get("role") == "assistant" and bool(last.get("tool_calls"))


def _route_after_think(state: AnalystState) -> str:
    if state.get("stop_reason"):
        return "draft"
    if state.get("answer"):
        return "draft"
    if state.get("needs_retry"):
        # A rejected call: the error is already in the transcript, ask again.
        return "think"
    if _pending_tool_call(state):
        return "act"
    return "draft"


def draft(state: AnalystState, *, correlation_id: str) -> dict[str, Any]:
    """Assemble the hand-off from whatever the loop produced."""
    elapsed_ms = int((time.perf_counter() - state["started_ms"]) * 1000)

    answer = state.get("answer_text", "").strip()
    stop_reason = state.get("stop_reason")

    route = state.get("route") or Route.LEGAL_COUNSEL
    if not answer:
        # No answer and no explicit route: the loop ended without producing one.
        route = Route.LEGAL_COUNSEL
        if stop_reason is None:
            stop_reason = "no_answer"

    # Invented ids are dropped, so every citation names a retrieved chunk.
    chunks = state.get("chunks", {})
    if answer:
        cited = [cid for cid in state.get("cited_ids", []) if cid in chunks]
        claims = [Claim(text=answer, citations=sorted(set(cited)))]
    else:
        claims = []

    result = AnalystResult(
        route=route,
        draft_answer=answer,
        claims=claims,
        retrieved_chunks=list(state.get("chunks", {}).values()),
        loop=LoopStats(
            steps=state.get("steps", 0),
            tool_calls=state.get("tool_calls", 0),
            wall_clock_ms=elapsed_ms,
        ),
        prompt_version=PROMPT_VERSION,
        correlation_id=correlation_id,
    )
    return {"claims": claims, "route": route, "answer": answer, "result": result}


def build_graph(*, model: Model | None = None, bounds: Bounds | None = None):
    """Compile the Analyst graph. Model and bounds are captured by the nodes."""
    model = model or get_model()
    bounds = bounds or Bounds()

    graph = StateGraph(AnalystState)
    graph.add_node("think", lambda s: think(s, model=model, bounds=bounds))
    graph.add_node("act", lambda s: act(s, bounds=bounds))
    graph.add_node(
        "draft",
        lambda s: draft(s, correlation_id=s["correlation_id"]),
    )

    graph.add_edge(START, "think")
    graph.add_conditional_edges(
        "think",
        _route_after_think,
        {"act": "act", "draft": "draft", "think": "think"},
    )
    graph.add_edge("act", "think")
    graph.add_edge("draft", END)
    return graph.compile()


def analyse(
    question: str,
    request: dict[str, Any],
    *,
    model: Model | None = None,
    bounds: Bounds | None = None,
    correlation_id: str | None = None,
) -> AnalystResult:
    """Run the Analyst over one question and return the hand-off.

    ``request`` is the full request body, so ``lookup_request_field`` can read
    any part of it.
    """
    correlation_id = correlation_id or str(uuid.uuid4())
    app = build_graph(model=model, bounds=bounds)

    initial: AnalystState = {
        "messages": [{"role": "user", "content": question}],
        "request": request,
        "correlation_id": correlation_id,
        "started_ms": time.perf_counter(),
        "steps": 0,
        "tool_calls": 0,
        "chunks": {},
    }
    final = app.invoke(initial)
    return final["result"]