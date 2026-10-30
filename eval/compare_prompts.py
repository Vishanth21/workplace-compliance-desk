"""Prompt comparison script (L2).

Compares prompt variants 'analyst-1' and 'analyst-2' across the 20-scenario eval set.
Computes and reports:
- Routing Accuracy
- Citation Grounding Precision
- Proviso Retrieval Rate
- Average Steps & Tool Calls

Usage:
    python -m eval.compare_prompts
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from agent.analyst import analyse
from agent.model import ScriptedModel
from agent.prompts import PROMPT_VARIANTS, PROMPT_V1, PROMPT_V2
from contracts import Route
from eval_scenarios import ALL_SCENARIOS, Scenario


def evaluate_variant(variant_name: str) -> dict[str, float | int]:
    route_matches = 0
    total_steps = 0
    total_tool_calls = 0
    clean_citations = 0
    flagship_provisos_hit = 0
    flagship_count = 0

    flagship_provisos = {"css2020-s53-proviso_2", "css2020-s53-proviso_6"}

    for sc in ALL_SCENARIOS:
        model = ScriptedModel(list(sc.script))
        req = {"question": sc.question, "establishment": {"headcount": 50}}
        result = analyse(
            sc.question,
            req,
            model=model,
            prompt_version=variant_name,
            correlation_id=f"eval-{variant_name}-{sc.id}",
        )

        if result.route == sc.expected_route:
            route_matches += 1

        total_steps += result.loop.steps
        total_tool_calls += result.loop.tool_calls

        cited = set()
        for claim in result.claims:
            cited.update(claim.citations)

        # Check for forbidden citations (repealed provisions)
        if not (sc.forbidden_citations & cited):
            clean_citations += 1

        if sc.required_citations & flagship_provisos:
            flagship_count += 1
            if flagship_provisos.issubset(cited):
                flagship_provisos_hit += 1

    n = len(ALL_SCENARIOS)
    return {
        "variant": variant_name,
        "scenarios_tested": n,
        "routing_accuracy_pct": round((route_matches / n) * 100, 1),
        "citation_clean_pct": round((clean_citations / n) * 100, 1),
        "flagship_proviso_pct": round((flagship_provisos_hit / max(1, flagship_count)) * 100, 1),
        "avg_steps": round(total_steps / n, 2),
        "avg_tool_calls": round(total_tool_calls / n, 2),
    }


def main() -> None:
    print("=" * 60)
    print("Evaluating Prompt Variants (Requirement L2)")
    print("=" * 60)

    stats_v1 = evaluate_variant("analyst-1")
    stats_v2 = evaluate_variant("analyst-2")

    print(f"\nVariant 1: {stats_v1['variant']}")
    for k, v in stats_v1.items():
        print(f"  {k:<24}: {v}")

    print(f"\nVariant 2: {stats_v2['variant']}")
    for k, v in stats_v2.items():
        print(f"  {k:<24}: {v}")

    print("\nComparison Summary:")
    print(
        f"  Routing Accuracy      : {stats_v1['routing_accuracy_pct']}% vs {stats_v2['routing_accuracy_pct']}%"
    )
    print(
        f"  Citation Grounding    : {stats_v1['citation_clean_pct']}% vs {stats_v2['citation_clean_pct']}%"
    )
    print(
        f"  Flagship Proviso Rate : {stats_v1['flagship_proviso_pct']}% vs {stats_v2['flagship_proviso_pct']}%"
    )
    print(
        f"  Average Steps         : {stats_v1['avg_steps']} vs {stats_v2['avg_steps']}"
    )


if __name__ == "__main__":
    main()
