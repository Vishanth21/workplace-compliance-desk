"""Tools the Analyst may call, with their JSON schemas.

Each tool is a callable plus a schema, kept apart from the graph so the
dispatch table can be tested without a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from contracts import ProvisionStatus
from index.retrieve import corpus_search, lookup_clause, lookup_request_field

#: Names the model may emit. Order is stable so prompts are reproducible.
TOOL_NAMES = ("corpus_search", "lookup_clause", "lookup_request_field")


@dataclass(frozen=True)
class Tool:
    """One callable with the schema the model is shown."""

    name: str
    description: str
    parameters: dict[str, Any]
    #: ``(arguments, request) -> JSON-serialisable result``
    fn: Callable[[dict[str, Any], dict[str, Any]], Any]

    def schema(self) -> dict[str, Any]:
        """The function-calling schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _corpus_search(args: dict[str, Any], request: dict[str, Any]) -> Any:
    status = args.get("status")
    parsed_status = None if status is None else ProvisionStatus(status)
    chunks = corpus_search(
        args["query"],
        status=parsed_status,
        doc=args.get("doc"),
        section=args.get("section"),
        k=int(args.get("k", 5)),
    )
    return [c.model_dump(mode="json") for c in chunks]


def _lookup_clause(args: dict[str, Any], request: dict[str, Any]) -> Any:
    chunks = lookup_clause(
        args["section"],
        doc=args.get("doc", "css2020"),
        part_prefix=args.get("part_prefix"),
    )
    return [c.model_dump(mode="json") for c in chunks]


def _lookup_request_field(args: dict[str, Any], request: dict[str, Any]) -> Any:
    return {"field": args["field"], "value": lookup_request_field(args["field"], request)}


TOOLS: dict[str, Tool] = {
    "corpus_search": Tool(
        name="corpus_search",
        description=(
            "Semantic search over the statute corpus. Use when you do not know "
            "which section applies. Only in-force provisions are returned unless "
            "status='repealed' or status=null."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "status": {
                    "type": ["string", "null"],
                    "enum": ["in_force", "repealed", None],
                    "description": (
                        "'in_force' for current law only, 'repealed' for repealed "
                        "law, or null for both."
                    ),
                },
                "doc": {
                    "type": ["string", "null"],
                    "description": "Restrict to one document, e.g. 'css2020'.",
                },
                "section": {
                    "type": ["string", "null"],
                    "description": "Restrict to one section, e.g. '53'.",
                },
                "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
            "required": ["query"],
        },
        fn=_corpus_search,
    ),
    "lookup_clause": Tool(
        name="lookup_clause",
        description=(
            "Fetch every part of one section by its number, in document order: "
            "main clause, then provisos, then explanations. Use after a search "
            "identifies the relevant section, because provisos that qualify the "
            "main clause often share no words with the question."
        ),
        parameters={
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Section number as printed, e.g. '53'. Not '53(1)'.",
                },
                "doc": {"type": "string", "default": "css2020"},
                "part_prefix": {
                    "type": ["string", "null"],
                    "description": (
                        "'proviso' to return only provisos, 'explanation' for "
                        "explanations, or null for all parts."
                    ),
                },
            },
            "required": ["section"],
        },
        fn=_lookup_clause,
    ),
    "lookup_request_field": Tool(
        name="lookup_request_field",
        description=(
            "Read one field off the employer's request, e.g. "
            "'establishment.headcount' or 'establishment.has_fixed_term_employees'. "
            "Returns null when the field is absent, which is a reason to ask for "
            "more detail rather than to guess."
        ),
        parameters={
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "description": "Dotted path into the request, e.g. 'establishment.state'.",
                }
            },
            "required": ["field"],
        },
        fn=_lookup_request_field,
    ),
}


def schemas() -> list[dict[str, Any]]:
    """Every tool schema, in stable order."""
    return [TOOLS[name].schema() for name in TOOL_NAMES]


def dispatch(name: str, arguments: str | dict[str, Any], request: dict[str, Any]) -> str:
    """Run one tool, returning its result as a JSON string.

    Errors come back as text so the agent can act on them.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"unknown tool {name!r}. Available: {list(TOOL_NAMES)}"})

    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"arguments are not valid JSON: {exc}"})

    if not isinstance(args, dict):
        return json.dumps({"error": "arguments must be a JSON object"})

    try:
        return json.dumps(tool.fn(args, request), default=str)
    except KeyError as exc:
        return json.dumps({"error": f"missing required argument: {exc}"})
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})