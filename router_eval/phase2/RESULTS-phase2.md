# Results — Phase 2 live router evaluation

> **STATUS: re-run 2026-09-03** for MESH-232, judged by `anthropic/claude-sonnet-4-6` on
> **692 real Mesh prompts**. Supersedes the 2026-08-27 n=91 run: two harness defects found
> since then (below) changed which models the `benchmark` baseline routed to, so the older
> numbers are not comparable. Raw prompts/answers are PII and are **not** committed
> (`out/`, `.cache/`, `mesh_traffic.jsonl` are gitignored); this file is the committed record.

## The question

MESH-232: `model=auto` ranks **brands, not models**, and the benchmark path only ever picks
from tier-1 — so production (routing-data version **v4**) can land on just **11** model ids
out of a 1,000+ catalog. Candidate version **v7** expands the pool. *Does the expanded
version pick **better** models, or merely **different** ones?*

## Two harness defects had to be fixed first

Both changed what the baseline arm measured, so they are results in their own right:

1. **The ranking table had drifted from production.** `router_eval/benchmark_table.py`
   claimed to be "byte-for-byte identical" to routersvc's `SUPERMODE_BENCHMARKS` but
   differed in four `General reasoning / Q&A` categories — three demoted `claude` out of
   tier-1. Since only tier-1 is ever picked, three of those changed the baseline's route.
   Every `benchmark` number produced before this fix was scored against a table production
   does not run.
2. **Tier-1 ties were broken deterministically** (`sorted(...)[0]`) where the gateway does
   `random.choice`. That under-sampled the reachable set — 6 distinct models measured for a
   version whose tier-1 set is 11 — and, fatally for this comparison, made a version that
   *adds* brands to tie-groups look inert: under the old rule v7 picked **fewer** distinct
   models than v4 (5 vs 6); under the gateway's rule it picked **14 vs 8**.

The tie-break is now seeded **per prompt**, so the two arms are *paired*: an unchanged
tie-group yields the same pick in both, and any difference comes only from the routing data.

## Round 1 — v7 with tier-1 promotions: WORSE

Six new brands were promoted into tier-1 on the rule *"a newer member of a family already
proven at tier-1, or a task specialist"*.

| strategy | n | judge quality (0–1) | cost/req (incl. classifier tax) | distinct models |
|---|---|---|---|---|
| **benchmark (v4 — production)** | 692 | **0.533** | $0.00863 | 8 |
| **benchmark_v7 (tier-1 promotions)** | 692 | **0.497** | $0.00889 | 13 |
| heuristic | 692 | 0.535 | $0.00987 | 8 |
| weighted | 692 | 0.506 | $0.00306 | 6 |
| registry | 692 | 0.296 | $0.00284 | 1 |
| *[served]* (production reference) | 692 | 0.565 | $0.03870 | — |

Paired, on the **282** prompts where the arms differed:

```
delta (v7 - v4)   -0.0889      95% CI [-0.1307, -0.0472]   (bootstrap agrees)
v7 better  75     v7 worse 147     tie 60      sign test p < 0.0001
```

**More models, worse answers, slightly higher cost.** The dominant drivers:

| swap | n | mean delta |
|---|---|---|
| `deepseek-v3.2` → `deepseek-v4-flash` | 42 | **−0.282** |
| `deepseek-v3.2` → `claude-haiku-4.5` | 92 | −0.061 |
| `gemini-3-flash` → `deepseek-v4-flash` | 20 | −0.107 |
| `gpt-5.4-mini` → `kimi-k2.6` | 10 | −0.342 |
| `deepseek-r1` → `deepseek-v4-pro` | 9 | −0.183 |
| `gemini-3-flash` → `claude-haiku-4.5` | 60 | **+0.054** |
| `claude-sonnet-4.6` → `claude-sonnet-5` | 11 | +0.009 |

`deepseek-v4-flash`/`-pro` are newer than the models they replaced and carry 41k/72k
successful production completions at 97–99% success — and are materially worse on our own
traffic. **Production success proves a model SERVES; it says nothing about whether it is
GOOD.**

## Round 2 — v7 as shipped (pool-only): NEUTRAL by construction

The promotions were reverted. v7 keeps all 17 new brands but places them at **tier-2 or
lower**, so its tier-1 is byte-identical to v4's.

| strategy | n | judge quality | cost/req | distinct models |
|---|---|---|---|---|
| benchmark (v4) | 692 | 0.531 | $0.01083 | 8 |
| **benchmark_v7 (shipped)** | 692 | **0.531** | **$0.01083** | 8 |

**692/692 identical picks.** The pool behind the pick grows from 17 to 44 model ids —
available to MESH-497 capability fallover and to the weighted strategy — while the pick
itself cannot change.

## The servability gate, validated

v7's candidates were gated on production evidence (≥100 successful completions with
`completion_tokens>0`, ≥3 API keys, ≥5 days, ≥90% success, over 90 days of `usage_events`).

| population | live answer attempts | failed |
|---|---|---|
| **v7's new models** | 100 | **0 (0.0%)** |
| `xai/grok-4.1-fast-non-reasoning` *(already reachable in v4)* | 237 | **138 (58.2%)** |
| `ai21/jamba-1-5-large-v1` *(registry's single pick)* | 692 | 297 (42.9%) |

The gate works. The worst model in the whole run is one **already in production routing**
that would not pass it.

## Reading notes / honest boundaries

- **Round 2 proves neutrality, not benefit.** An identical pick set is the intended result,
  but it means this run cannot demonstrate the fallover/weighted upside. That needs a run
  with `weighted` enabled over the v7 pool, or a fallover-injection test. Not done.
- **Single judge, single pass** (`claude-sonnet-4-6`); no multi-judge agreement check.
  Contamination does not apply (real user prompts), but the judge's biases are uncorrected.
- **One traffic sample** (692 prompts, one export). The round-1 direction is well outside
  noise; individual per-swap numbers at n≈10 are not.
- **Cost magnitudes are soft.** Strategy answers are capped at 1024 output tokens;
  `[served]` uses uncapped production token counts. Strategy-vs-strategy is like-for-like;
  the multiple against `[served]` is not.
- **`registry` remains degenerate** — one model (`ai21/jamba-1-5-large-v1`) for all 692
  diverse prompts, 42.9% of those calls failing. Tracked separately (MESH-783).
- **No oracle.** The actually-served model + its feedback is the reference instead.
  Served feedback this run: NULL 661, rejected 28, dislike 1, like 2.

## Reproduce (operator, live — prompts are real PII)

```bash
MESH_API_KEY=... PHASE2_REAL_ONLY=1 python -m router_eval.phase2 --live --estimate-only
MESH_API_KEY=... PHASE2_REAL_ONLY=1 python -m router_eval.phase2 --live \
    --judge-model anthropic/claude-sonnet-4-6
```
`PHASE2_REAL_ONLY=1` drops the random/always_* baselines, which are corrupted by the
catalog's unservable models and account for most of the judge spend. Default (no `--live`)
is a mock dry run — wiring only, not results.
