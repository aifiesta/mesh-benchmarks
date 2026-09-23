# MESH-941 — paired comparison of the v11 routing pool, with confidence intervals

The 692-prompt judged run committed in `out_v11/` reported **means only**. MESH-941 AC3 asks
for the comparison to be *paired, with a stated confidence interval*. This is that interval.

Nothing was re-scored. The per-prompt judge scores were **recovered** from the run's own
content-addressed answer/judgment cache and verified against the committed aggregate — see
*Provenance*. No new model or judge call was made.

- Run: `out_v11/` (2026-09-10), 692 real Mesh prompts, 13 strategy arms.
- Judge: `anthropic/claude-sonnet-4-6`, one pass, score in [0,1].
- Method: paired bootstrap, 20,000 resamples of the **prompt index** (so both arms are
  re-weighted by the same prompt on every draw), percentile 95% CI, seed 20260923.
- Script: `router_eval/phase2/paired_bootstrap.py` (tests in `tests/test_paired_bootstrap.py`).

## Two denominators, both reported

Every comparison is given twice, because they answer different questions:

- **Unconditional** — over all 692 prompts. This is the fleet-wide effect. Prompts where both
  arms pick the same model contribute exactly 0 and correctly dilute it.
- **Conditional** — over only the prompts where the two arms picked **different models**. This
  is the effect size where the change actually bites, and it is the scale `RESULTS-phase2.md`
  quotes for the v7 round (*"paired, on the 282 prompts where the arms differed"*).

Quoting one as the other is the easy mistake here; the v7 figure this ticket cites as its
risk #1 (−0.0889) is a **conditional** number.

## Results

Unconditional, all 692 prompts:

| comparison | mean A | mean B | paired diff (A−B) | 95% CI | verdict |
|---|---|---|---|---|---|
| `weighted_v11` vs `weighted_v10` | 0.448194 | 0.485462 | **−0.037269** | [−0.051546, −0.023439] | **A significantly WORSE** |
| `weighted_v10` vs `weighted` (v4 pool) | 0.485462 | 0.555361 | **−0.069899** | [−0.088092, −0.052052] | **A significantly WORSE** |
| `weighted_v11` vs `weighted` (v4 pool) | 0.448194 | 0.555361 | **−0.107168** | [−0.127009, −0.087355] | **A significantly WORSE** |
| `weighted_v11_aa` vs `weighted_v10` | 0.507298 | 0.485462 | +0.021835 | [−0.004118, +0.047847] | not significant |
| `weighted_v10` vs `registry` | 0.485462 | 0.498309 | −0.012847 | [−0.036098, +0.010665] | not significant |

Conditional on the prompts where the arms picked different models:

| comparison | n differing | conditional diff | 95% CI | A wins / B wins / tie |
|---|---|---|---|---|
| `weighted_v11` vs `weighted_v10` | 141 | **−0.182908** | [−0.245532, −0.120355] | 28 / 89 / 24 |
| `weighted_v10` vs `weighted` | 351 | **−0.137806** | [−0.172165, −0.103989] | 70 / 200 / 81 |
| `weighted_v11` vs `weighted` | 439 | **−0.168929** | [−0.198428, −0.139294] | 74 / 272 / 93 |
| `weighted_v11_aa` vs `weighted_v10` | 631 | +0.023946 | [−0.004992, +0.052837] | 278 / 206 / 147 |
| `weighted_v10` vs `registry` | 692 | −0.012847 | [−0.036474, +0.010592] | 294 / 262 / 136 |

Sign test (exact two-sided binomial over discordant prompts, magnitude-free — a handful of
large deltas cannot carry it): 1.4e-08, 1.1e-15, 9.9e-28, 1.2e-03, 1.9e-01 respectively.

The last two rows are the control: the method returns *not significant* when the evidence is
thin, so the three significant results are not an artifact of the procedure.

## What this means

1. **v11 does not activate.** −0.0373 [−0.0515, −0.0234] fleet-wide; −0.1829 [−0.2455,
   −0.1204] on the 141 prompts where the pools actually diverge. On the ticket's own scale
   that is **roughly double the v7 tier-1 regression (−0.0889)** which is exactly why v7
   shipped pool-only at tier-2+. AC3 permits a negative result; this is one.
2. **The deficit is not a dead-model artifact.** v11 returned *fewer* empty/failed answers
   than v10 (7 vs 17). v11 picks models that answer and score lower — a servability fix would
   not recover the gap.
3. **v10 itself is the larger open problem.** The pool production serves is −0.0699 [−0.0881,
   −0.0521] below the `weighted` arm's v4 pool — ~1.9x the v11 regression. Prod is on v10
   (`auto_router_data_version = v10` since 2026-09-04, `auto_router_weighted_enabled = true`
   since 2026-09-02, both read from prod `system_settings` on 2026-09-23). Out of MESH-941's
   scope; needs its own ticket.
4. **Correction to `521fec9`.** It called `weighted_v11_aa` "the only arm that beat v10".
   Paired, it does **not**: +0.0218 [−0.0041, +0.0478], CI spans zero (p = 0.097). Its mean is
   also propped up by a different problem — 113 of its 692 picks (16%) returned empty answer
   text, 112 from `openai/gpt-5.6-luna` (112 of the 396 prompts it routed there). It is not
   evidence in either direction, and it remains a design mesh-gateway PR #1728 rejected.

## Validation — the method reproduces a number computed months ago

Run against the **v7 round-2** output (`out_v7paired/`, a different run, a different cache),
the script reproduces the figure `RESULTS-phase2.md` line 59 has carried since:

| | published (RESULTS-phase2.md) | this script |
|---|---|---|
| delta (v7 − v4), conditional | −0.0889 | **−0.088901** |
| 95% CI | [−0.1307, −0.0472] | **[−0.131099, −0.046950]** |
| n differing | 282 | **282** |
| v7 better / worse / tie | 75 / 147 / 60 | **75 / 147 / 60** |

That is an end-to-end check of recovery *and* bootstrap against an independently produced
result, on data this script was not written against.

## Provenance — how the per-prompt scores were recovered

The pipeline writes `picks.csv` (strategy, prompt_index, picked_model) and per-strategy means.
It never writes per-prompt scores. They are still recoverable because every cache key is a
pure function of inputs that were retained:

```
answer : sha256("answer" \0 <model_id> \0 <prompt>)                              -> {"answer": ...}
judge  : sha256("judge" \0 <judge_model> \0 <model_id> \0 <prompt> \0 sha256(answer)[:16]) -> {"raw": ...}
```

For each `(prompt_index, picked_model)` the script re-derives both keys, reads the cached judge
reply, and parses it with `LiveJudge._parse` — the same parser the run used, fail-soft 0.0 and
all. **The recovery is self-verifying**: `--aggregate` recomputes each of the 13 arms' means
from the recovered vectors and compares them to the committed `strategy_aggregate.csv`. All 13
match to 1e-6, and all 3,430 distinct (prompt, model) pairs resolved with zero misses.

**Trap for whoever reruns this:** the v11 run used `--judge-model anthropic/claude-sonnet-4-6`,
but the harness default is `anthropic/claude-opus-4.8`. The judge model is part of the cache
key. With the default, nothing resolves — the script errors rather than guessing — and a live
rerun with the default would re-pay for all 3,874 judge calls. This is not hypothetical: the
older `.cache/` holds *both* judges' entries for some pairs, so a wrong `--judge-model` there
silently resolves a partial, wrong set (162 of 2,452 pairs, means off by +0.13 to +0.36). The
`--aggregate` gate is what turns that into a loud failure instead of a plausible-looking table.

## Boundaries

- **One judge, one pass** (`claude-sonnet-4-6`). The interval quantifies prompt-sampling noise,
  not judge noise; a second judge could move the point estimate. It would have to move it by
  0.023 to erase the unconditional v11 result.
- **One traffic sample** (692 prompts, a single export). The CI describes resampling *these*
  prompts; it says nothing about drift in what customers ask.
- **Percentile bootstrap**, not BCa. At n=692 with a near-symmetric bootstrap distribution the
  difference is immaterial here.
- Answers are capped at 1024 output tokens, which plausibly explains the `gpt-5.6-luna` empty
  replies (reasoning tokens consuming the cap). Not confirmed — **inferred** from the pattern.

## Reproduce (operator, local — the cache and traffic file are real user PII)

```bash
python -m router_eval.phase2.paired_bootstrap \
    --picks router_eval/phase2/out_v11/picks.csv \
    --cache-dir router_eval/phase2/.cache_v11 \
    --aggregate router_eval/phase2/out_v11/strategy_aggregate.csv \
    --judge-model anthropic/claude-sonnet-4-6 \
    --compare weighted_v11:weighted_v10 --compare weighted_v10:weighted
```

Offline, no network, no key. `.cache_v11/` and `mesh_traffic.jsonl` are gitignored and stay
local; the script prints aggregate numbers only — never a prompt, an answer, or a rationale.
