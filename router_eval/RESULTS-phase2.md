# Results — Phase 2 live router evaluation (MESH-708)

> **STATUS: run 2026-08-27.** 91 real Mesh prompts, 7 strategies, each pick answered live
> through the Mesh API and scored 0–1 by an LLM judge (`anthropic/claude-opus-4.8`). ~1,270
> live calls, deduped + cached. Raw prompts/answers are PII and are **not** committed
> (`out/`, `.cache/`, `mesh_traffic.jsonl` are gitignored); this file is the committed record.

## What Phase 2 measures (that Phase 1 can't)

Phase 1 replays a fixed public dataset (RouterBench) with precomputed, contaminated scores —
it tests routing **decision logic**, not answer quality on our traffic. Phase 2 runs each
strategy over the **live Mesh catalog** on **91 real Mesh prompts** (multilingual, real users),
calls each strategy's picked model for a real answer, and has a strong judge score it. The
prompts carry the model actually served + user `feedback_rating` as a reference point.

## Headline

| strategy | judge quality (0–1) | cost/req (incl. classifier tax) | distinct models | note |
|---|---|---|---|---|
| **heuristic** | **0.675** | $0.00694 | 4 | routes to premium (sonnet/haiku) |
| **benchmark** | **0.673** | $0.00687 | 4 | " |
| registry | 0.660 | $0.00386 | **1** | ⚠️ picks only `ai21/jamba-1-5-large-v1` |
| **[actually served]** | 0.657 | $0.01436 | — | the production reality |
| weighted | 0.636 | $0.00322 | 4 | routes to cheaper (qwen-flash/grok-fast) |
| random\* | 0.475 | $0.00384 | 78 | *corrupted — see caveat 1* |
| always_cheapest\* | 0.332 | ~$0 | 1 | picks a free model (`minimax/m2-her`) |
| always_premium\* | **0.000** | $0.00695 | 1 | *corrupted — the top-cost model can't serve* |

**Every real routing strategy delivered quality comparable to or better than what customers
were actually served (0.657), at a fraction of the cost.** benchmark/heuristic edge out the
served quality at ~half the cost; weighted trades ~3% quality for a ~4.5× cost cut; registry
matches served quality at ~4× cheaper (but see caveat 3). This is the empirical evidence
MESH-708 was chartered to produce: **routing adds value on our own traffic.**

The strategies also differ in *character*: benchmark/heuristic route to premium models
(claude-sonnet-4.6, claude-haiku-4.5, deepseek-v3.2, gemini-3-flash); weighted deliberately
routes cheaper (deepseek-v3.2, kimi-k2.5, qwen-flash, grok-4.1-fast) — visibly its cost term
working.

## Caveats — read before trusting any number

1. **The baselines are corrupted, not the strategies.** `always_premium` scored 0 because the
   single most-expensive catalog model **cannot serve a basic chat request** — and it is not
   alone: **104 of 588 (18%) of "chat-capable" catalog models (`supports_completions_api=true`)
   failed a live call** (unservable, or needing params this simple eval client doesn't send).
   That failure mass drags `random` and breaks `always_premium`. The four real strategies route
   to narrow, known-good sets, so their numbers are clean. **The 18% is itself a catalog-health
   finding.** Fix for clean baselines: restrict the candidate pool to models that actually serve.
2. **Cost is directionally right, magnitude soft.** Strategy costs use the eval's answers
   (capped at 1024 output tokens); `[served]` uses real production token counts (uncapped, some
   long). So "strategies are cheaper" holds, but the exact multiple is not apples-to-apples.
3. **Registry is degenerate.** It picked ONE model (`ai21/jamba-1-5-large-v1`) for all 91 diverse
   prompts. It didn't error here (scored 0.66), but in prod its first real firing logged
   `invalid_response` (the gemini-3-flash classifier output didn't parse). Combined, registry is
   **not doing meaningful routing** — the not_diamond→registry swap (MESH-783) needs a real fix
   (classifier-model compatibility), tracked separately.
4. **Judge is a single model** (opus-4.8) — a strong but single perspective; no multi-judge
   agreement check. Contamination doesn't apply (real user prompts, not public benchmarks), but
   the judge's own biases are uncorrected.

## What is NOT covered (per the mesh-benchmarks house convention)

- Clean baselines / oracle headroom (blocked on caveat 1 — the servable-pool fix).
- All 128 prompts — 91/128 parsed from the sheet (multi-line answer cells broke ~37 rows); a
  clean CSV export would recover the full set.
- Auto-routed vs user-picked separation — the export doesn't mark which prompts were `model=auto`,
  so the `[served]` comparison mixes both.
- Latency — quality + cost only; no per-strategy latency measured here.
- Multi-judge agreement, per-category breakdown, and statistical significance (n=91).

## Reproduce (operator, live)

```bash
MESH_API_KEY=... python -m router_eval.phase2 --live --estimate-only            # counts, no spend
MESH_API_KEY=... python -m router_eval.phase2 --live --judge-model anthropic/claude-opus-4.8
```
Default (no `--live`) is a mock dry run — wiring only, not results.
