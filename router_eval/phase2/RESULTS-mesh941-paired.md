# MESH-941 — paired comparison of the v11 routing pool, with confidence intervals

The 692-prompt judged run committed in `out_v11/` reported **means only**. MESH-941 AC3 asks
for the comparison to be *paired, with a stated confidence interval*. This is that interval.

Nothing was re-scored. The per-prompt judge scores were **recovered** from the run's own
content-addressed answer/judgment cache and verified against the committed aggregate — see
*Provenance* below. No new model or judge call was made.

- Run: `out_v11/` (2026-09-10), 692 real Mesh prompts, 13 strategy arms.
- Judge: `anthropic/claude-sonnet-4-6`, one pass, score in [0,1].
- Method: paired bootstrap, 20,000 resamples of the **prompt index** (so both arms are
  re-weighted by the same prompt on every draw), percentile 95% CI, seed 20260923.
- Script: `router_eval/phase2/paired_bootstrap.py` (tests in `tests/test_paired_bootstrap.py`).

## Results

| comparison | mean A | mean B | paired diff (A−B) | 95% CI | verdict |
|---|---|---|---|---|---|
| `weighted_v11` vs `weighted_v10` | 0.448194 | 0.485462 | **−0.037269** | [−0.051546, −0.023439] | **A significantly WORSE** |
| `weighted_v10` vs `weighted` (v4 pool) | 0.485462 | 0.555361 | **−0.069899** | [−0.088092, −0.052052] | **A significantly WORSE** |
| `weighted_v11` vs `weighted` (v4 pool) | 0.448194 | 0.555361 | **−0.107168** | [−0.127009, −0.087355] | **A significantly WORSE** |
| `weighted_v11_aa` vs `weighted_v10` | 0.507298 | 0.485462 | +0.021835 | [−0.004118, +0.047847] | not significant |
| `weighted_v10` vs `registry` | 0.485462 | 0.498309 | −0.012847 | [−0.036098, +0.010665] | not significant |

Supporting per-prompt detail (the bootstrap is over all 692; ties included):

| comparison | A wins | B wins | tie | identical pick | sign-test p | empty/failed answers A / B |
|---|---|---|---|---|---|---|
| `weighted_v11` vs `weighted_v10` | 28 | 89 | 575 | 551 | 1.4e-08 | 7 / 17 |
| `weighted_v10` vs `weighted` | 70 | 200 | 422 | 341 | 1.1e-15 | 17 / 5 |
| `weighted_v11` vs `weighted` | 74 | 272 | 346 | 253 | 9.9e-28 | 7 / 5 |
| `weighted_v11_aa` vs `weighted_v10` | 278 | 206 | 208 | 61 | 1.2e-03 | **113** / 17 |
| `weighted_v10` vs `registry` | 294 | 262 | 136 | 0 | 1.9e-01 | 17 / 0 |

The sign test is an exact two-sided binomial over the discordant prompts only — a magnitude-free
robustness check, so a handful of large deltas cannot carry the result on their own.

## What this means

1. **v11 is significantly worse than v10, and the interval does not come near zero.**
   −0.0373 [−0.0515, −0.0234]. The two pools pick the same model on 551 of 692 prompts; on the
   141 where they differ, v10 wins 89 to 28. Per AC3 this is a negative-but-valid result and it
   **blocks activating v11**. Prod stays on v10.

2. **The deficit is not a dead-model artifact.** v11 actually returned *fewer* empty/failed
   answers than v10 (7 vs 17). v11 is picking models that answer and score lower — not models
   that fail to answer. A servability fix would not recover the gap.

3. **The larger open problem is v10 itself.** The pool production serves scores −0.0699
   [−0.0881, −0.0521] below the `weighted` arm's v4 pool. That gap is ~1.9x the v11 regression
   and it is live right now. It deserves its own ticket; it is out of MESH-941's scope.

4. **Correction to the previous commit's reading.** `521fec9` called `weighted_v11_aa` "the only
   arm that beat v10". Paired, it does **not**: +0.0218 [−0.0041, +0.0478], CI spans zero
   (bootstrap p = 0.097). Its mean is also inflated by a different problem — 113 of its 692 picks
   (16%) came back with empty answer text, 112 of them from `openai/gpt-5.6-luna` (112 of the 396
   prompts it routed there). So `weighted_v11_aa` is not evidence for anything, in either
   direction. It remains a design mesh-gateway PR #1728 rejected.

## Provenance — how the per-prompt scores were recovered

The pipeline writes `picks.csv` (strategy, prompt_index, picked_model) and a per-strategy mean.
It never writes per-prompt scores. They are still recoverable because every cache key is a pure
function of inputs that were retained:

```
answer : sha256("answer" \0 <model_id> \0 <prompt>)                              -> {"answer": ...}
judge  : sha256("judge" \0 <judge_model> \0 <model_id> \0 <prompt> \0 sha256(answer)[:16]) -> {"raw": ...}
```

For each `(prompt_index, picked_model)` the script re-derives both keys, reads the cached judge
reply, and parses it with `LiveJudge._parse` — the same parser the run used. **The recovery is
self-verifying**: `--aggregate` re-computes each of the 13 arms' means from the recovered vectors
and compares them to the committed `strategy_aggregate.csv`. All 13 match to 1e-6, and all 3,430
distinct (prompt, model) pairs resolved with zero misses. That is what makes these the run's own
scores rather than a second scoring pass.

**Trap for whoever reruns this:** the v11 run used `--judge-model anthropic/claude-sonnet-4-6`,
but the harness default is `anthropic/claude-opus-4.8`. The judge model is part of the cache key,
so running with the default resolves **nothing** (the script reports `missing judgment 3430`
rather than guessing) — and a live rerun with the default would re-pay for all 3,874 judge calls.

## Boundaries

- **One judge, one pass** (`claude-sonnet-4-6`). The interval quantifies prompt-sampling noise,
  not judge noise; a second judge could move the point estimate. It would have to move it by
  0.023 to erase the v11 result.
- **One traffic sample** (692 prompts, a single export). The CI describes resampling *these*
  prompts; it does not cover drift in what customers ask.
- **Percentile bootstrap**, not BCa. With n=692 and a near-symmetric bootstrap distribution the
  difference is immaterial here.
- **Ties are real, not padding.** 551 of the v11-vs-v10 prompts have an identical pick and
  contribute exactly 0. They correctly shrink the *mean* difference; the sign test reports the
  discordant subset separately.
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
