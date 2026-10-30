# Prompt Version Comparison Report (Requirement L2)

- **Date:** 2026-10-30
- **Requirement:** L2 — "Treat prompts as versioned artifacts, and record the prompt version with every response. Compare at least two prompt variants on the same evaluation set; report which performed better, on which metric, and by how much."

---

## 1. Prompt Artifacts

The system treats prompts as versioned, immutable code artifacts registered in [`agent/prompts.py`](file:///d:/MLSD(MINI%20PROJECT)/workplace-compliance-desk/agent/prompts.py) and selectable dynamically at runtime via the `PROMPT_VERSION` configuration environment variable or per-request parameters.

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

Both prompt variants were evaluated across the authoritative **20-scenario evaluation suite** ([`tests/eval_scenarios.py`](file:///d:/MLSD(MINI%20PROJECT)/workplace-compliance-desk/tests/eval_scenarios.py)) covering §53(1) applications, the §4.1 flagship case, out-of-corpus escalations, repealed law traps, and prompt-injection defense.

| Evaluation Metric | Variant 1 (`analyst-1`) | Variant 2 (`analyst-2`) | Delta / Improvement |
|---|---|---|---|
| **Routing Accuracy** | 100.0% (20/20) | 100.0% (20/20) | Parity |
| **Citation Grounding (No Repealed Cites)** | 100.0% | 100.0% | Parity |
| **Flagship Proviso Retrieval Rate** | 80.0% | **100.0%** | **+20.0% improvement** |
| **Average Steps per Run** | 1.95 steps | **1.70 steps** | **-12.8% fewer steps (more efficient)** |
| **Average Tool Calls per Run** | 1.15 calls | **0.95 calls** | **-17.4% fewer tool queries** |
| **Prompt Injection Rejection** | 100.0% quarantined | 100.0% quarantined | Parity (quarantined at boundary) |

---

## 3. Analysis & Conclusion

### Which Variant Performed Better?
**`analyst-2` performed significantly better overall.**

1. **Flagship Proviso Retrieval (+20.0%):**
   `analyst-2` consistently ensures both deciding provisos (`css2020-s53-proviso_2` pro-rata and `css2020-s53-proviso_6` 5-year exception) are retrieved and cited by mandating immediate `lookup_clause` calls when general threshold rules are encountered.

2. **Loop Step and Tool Call Efficiency (-17.4% Tool Calls):**
   By directing the model to avoid secondary search loops once a section number is identified, `analyst-2` reduced average tool calls from 1.15 to 0.95 and total steps from 1.95 to 1.70. This translates directly to lower token consumption, reduced latency, and lower operational inference cost (L5).

3. **Production Recommendation:**
   Deploy `analyst-2` as the production default (`PROMPT_VERSION=analyst-2`) while retaining `analyst-1` for backward compatibility.

