# Results — Phase 2 live router evaluation (MESH-708)

> **STATUS: pipeline built, NOT yet run live.** Every quality/cost number below is a
> `<placeholder>` until the operator runs the live pass with a real key (the prompts are
> real user PII — see the run command at the bottom). The dry-run reproduces the whole
> flow with mock providers; its numbers are wiring checks, not results.

## What Phase 2 measures (that Phase 1 can't)

Phase 1 replays a fixed public dataset with precomputed, contaminated scores — it tests the
routing **decision logic**, not answer quality on our traffic. Phase 2 closes that gap: it
runs each strategy over the **live Mesh catalog** on **91 real Mesh prompts**, calls the
picked models through the Mesh API, **LLM-judges** the answers (0–1 on correctness +
helpfulness + instruction-following), and plots judged-quality vs cost (incl. the classifier
tax) against the model **actually served** and its **user feedback** — on the same plane
Phase 1 uses.

## Live-call budget (deduped) — predictable before spend

Printed by the pipeline before any inference/judge call. Counts below are from the DRY RUN
over the real 91 prompts (mock classifier), so the classifier total is exact and the
inference/judge totals are close estimates — the live run reprints the exact counts (real
classifier picks shift a few) before it spends:

| calls | count (est., 91 prompts × 7 strategies) |
| --- | --- |
| unique (prompt, picked_model) pairs across strategies | 455 |
| …already served (pre-seeded from `response_raw`, no call) | 49 |
| **inference** calls to make (deduped, minus served) | **406** |
| **judge** calls (one per unique answer incl. served) | **497** |
| **classifier** calls (deduped by content: 91 gpt-4o-mini + 91 gemini-3-flash) | **182** |
| **TOTAL live calls** | **~1085** |

Dedupe + a disk cache mean a (prompt, model) pair picked by several strategies is paid for
once, and the 91 already-served answers are reused from the traffic (no re-call). The
classifier tax uses the same priced token model as Phase 1 (`metrics.py`).

## Per-strategy — judged quality vs cost (PLACEHOLDER)

| strategy | n | judged_quality (0–1) | infer_cost ($/req) | classifier_tax ($/req) | cost+tax ($/req) | distinct models |
| --- | --- | --- | --- | --- | --- | --- |
| random | 91 | `<q>` | `<$>` | 0.0 | `<$>` | `<k>` |
| always_cheapest | 91 | `<q>` | `<$>` | 0.0 | `<$>` | 1 |
| always_premium | 91 | `<q>` | `<$>` | 0.0 | `<$>` | 1 |
| benchmark | 91 | `<q>` | `<$>` | `<$>` | `<$>` | `<k>` |
| heuristic | 91 | `<q>` | `<$>` | `<$>` (fast-lane hits pay 0) | `<$>` | `<k>` |
| weighted | 91 | `<q>` | `<$>` | `<$>` | `<$>` | `<k>` |
| registry | 91 | `<q>` | `<$>` | `<$>` (gemini-3-flash) | `<$>` | `<k>` |
| **[served]** (reference) | 91 | `<q>` | `<$>` | 0.0 | `<$>` | — |

Served feedback (from the traffic, real): **NULL 85, rejected 5, dislike 1** — 6 negative
signals, no explicit positives. The judge score is what lets us compare strategies to the
served model despite the sparse thumbs.

## Reading notes / honest boundaries

- **`registry` is measured here, not in Phase 1** — it needs the live classifier free-selecting
  a model id over the live catalog, which has no offline analogue.
- **`heuristic` is finally exercised.** On the 91 real prompts the fast-lane gate fires on
  **4** of them (short conversational asks) — vs **0** on all of RouterBench — so here it
  genuinely diverges from `benchmark` (and pays no classifier on those 4).
- **No oracle.** A hindsight oracle would need every catalog model's answer judged per prompt
  (cost-prohibitive). The **actually-served model + its feedback** is the ground-truth
  reference instead. This is a deliberate deviation from Phase 1's four baselines (documented).
- **Served answer reuse.** The served baseline judges the answer the user actually received
  (`response_raw`), not a fresh call — faithful to "what shipped", but generated at a
  different time/temperature than the strategy answers (a small apples-to-oranges caveat).
- **Judge caveats.** Single-judge, single-pass, rubric-based; the judge model itself is a
  variable. Default `anthropic/claude-opus-4.8` (configurable via `--judge-model`); a judge
  should not grade its own family's answers unchallenged — vary the judge to check.
- **Cost axis.** Inference cost uses live catalog prices × served/returned token counts; the
  classifier tax is the Phase-1 model. Same plane as Phase 1's `cost+tax`.

## Run it (operator, with a real key — prompts are real PII)

```bash
# Pre-flight: print the exact deduped call budget and stop (no spend):
MESH_API_KEY=sk-... python -m router_eval.phase2 --live --estimate-only

# Full live run (inference + judge, cached/deduped):
MESH_API_KEY=sk-... python -m router_eval.phase2 --live \
    --judge-model anthropic/claude-opus-4.8

# Dry run (default; safe, offline, mock providers):
python -m router_eval.phase2
```

Outputs land in `router_eval/phase2/out/` and the cache in `router_eval/phase2/.cache/` —
both **gitignored** (they contain real prompts/answers). Fill the placeholder tables above
from `out/strategy_aggregate.csv` after the live run.
