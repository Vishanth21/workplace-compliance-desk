# Prompt Version Comparison Report (Requirement L2)

- **Date:** 2026-10-30
- **Requirement:** L2 — "Treat prompts as versioned artifacts, and record the prompt version with every response. Compare at least two prompt variants on the same evaluation set; report which performed better, on which metric, and by how much."

---

## 1. Prompt Artifacts

The system treats prompts as versioned, immutable code artifacts registered in [`agent/prompts.py`](../agent/prompts.py) and selected by the `PROMPT_VERSION` environment variable. There is no per-request prompt selection: `POST /runs` does not accept a prompt version, so the variant is fixed per deployment.

### Variant 1: `analyst-1` (Baseline)
- **Role:** Standard statutory compliance analyst.
- **Key Characteristics:** 
  - Standard enumeration of the 3 tools (`corpus_search`, `lookup_clause`, `lookup_request_field`).
  - General guidelines: cite retrieved IDs, note whether provisions are in force or repealed, answer directly.
- **Observed Limitation:** Under live testing, the model occasionally halted upon retrieving Section 53(1) main clause (5-year continuous service rule) without exploring subsequent qualifying provisos, or attempted multiple redundant keyword searches before clause lookup.

### Variant 2: `analyst-2` (Proviso-Aware & Strict Grounding)
- **Role:** Senior compliance analyst with explicit statutory hierarchy awareness.
- **Key Characteristics:**
  - **Proviso Hierarchy Rule:** Explicitly instructs that provisos override general statutory rules. If a general threshold (5 years) is surfaced, the agent is instructed to immediately call `lookup_clause` to inspect fixed-term and continuous service exceptions.
  - **Repealed Law Guard:** Explicitly requires identifying repealed statutes (PGA 1972) and immediately grounding obligations in the Code on Social Security, 2020.
  - **Search Optimization:** Instructs direct clause lookup as soon as a section number is known, eliminating redundant search query cycles.

---

## 2. Comparative Evaluation Results

Both prompt variants were evaluated across the same **17-scenario evaluation suite** (`ALL_SCENARIOS` in [`tests/eval_scenarios.py`](../tests/eval_scenarios.py)) covering §53(1) applications, the §4.1 flagship case, out-of-corpus escalations and repealed-law traps. The three prompt-injection cases in `INJECTION_QUESTIONS` are excluded: they are quarantined at the ingestion boundary, before a scenario is built.

Reproduce with:

```bash
uv run python -m eval.compare_prompts
```

| Evaluation Metric | Variant 1 (`analyst-1`) | Variant 2 (`analyst-2`) | Delta |
|---|---|---|---|
| **Routing Accuracy** | 100.0% (17/17) | 100.0% (17/17) | Parity |
| **Citation Grounding (No Repealed Cites)** | 100.0% | 100.0% | Parity |
| **Flagship Proviso Retrieval Rate** | 66.7% | 66.7% | Parity |
| **Average Steps per Run** | 2.12 | 2.12 | Parity |
| **Average Tool Calls per Run** | 1.12 | 1.12 | Parity |
| **Prompt Injection Rejection** | 100.0% quarantined | 100.0% quarantined | Parity (quarantined at boundary) |

## 3. Why the Numbers Are at Parity

The comparison drives both variants with `ScriptedModel`, which replays a fixed transcript per scenario. The transcript supplies the tool calls and the final answer, so the prompt text never influences what the model emits. Both variants therefore produce identical routes, citations and step counts by construction.

That makes this comparison a test of the harness, not of the prompts. What it does establish:

- The prompt version is selected correctly and recorded on every `RunTrace` (`PROMPT_VERSION` → `agent/prompts.py`).
- Both variants survive the full LangGraph loop, tool dispatch, citation resolution and deterministic verifier on all 17 scenarios.
- Routing and citation grounding hold at 100% for both, so no variant regresses the deterministic path.

The 66.7% flagship-proviso rate is a property of the scripted transcripts, not of either prompt: one of the three flagship scenarios omits `css2020-s53-proviso_6` from its canned retrieval, so the proviso check fails for both variants equally.

## 4. Status Against Requirement L2

Requirement L2 asks for two prompt variants compared on the same evaluation set, with a reported winner and margin.

| Part of L2 | Status |
|---|---|
| Prompts are versioned, immutable artifacts | Done (`agent/prompts.py`) |
| Prompt version recorded on every response | Done (`RunTrace.prompt_version`) |
| At least two variants compared on one eval set | Done (17 scenarios, both variants) |
| Report which performed better, on which metric, by how much | **Not supported by the evidence.** Under the scripted model the delta is zero on every metric. |

To produce a real margin, the comparison must run against a live model (`LLM_PROVIDER=openai` with `ANALYST_MODE=real`), where the prompt is the only variable. That is not wired into `eval/compare_prompts.py` today, and the live-adapter tests are opt-in and skipped in CI. **`analyst-2` is therefore not backed by a measured improvement; it is a documented hypothesis about prompt structure.**

## 5. Recommendation

Keep `analyst-1` as the default. Do not claim `analyst-2` is better on measured evidence. The proviso-awareness and repealed-law guidance in `analyst-2` are reasonable in principle, and they do not break anything in the deterministic path, but the comparison as currently built cannot distinguish them.

A real L2 result needs either of two changes:

1. Add a live-model comparison path to `eval/compare_prompts.py` that runs both prompts through a configured endpoint on the same 17 scenarios.
2. Keep the scripted comparison as a harness regression test, and state L2 as partially met.

---

*Earlier revisions of this report claimed a +20.0% proviso-retrieval gain, -12.8% steps and -17.4% tool calls for `analyst-2` over a 20-scenario set. Those figures were not reproducible: the scripted transcripts make both variants identical, and the set has 17 scenarios. They have been replaced with the measured numbers above.*

