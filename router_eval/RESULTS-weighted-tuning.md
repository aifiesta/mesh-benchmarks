# MESH-644 weighted-strategy profile tuning — offline frontier + recommended vectors

**Status:** offline analysis + recommendation. **No live inference / judge / classifier calls,
no prod writes, no config activation.** The tuned vectors and the go-live flip are Venky/Suvajit's
call — this doc produces the data and a recommendation, not the activation.

The `weighted` auto-router strategy scores each candidate model as

    score(m) = w_q·Quality(m) + w_c·Cost(m) + w_l·Latency(m)      (weights sum to 1.0, higher wins)

and ships DARK behind four named profiles whose `(w_q, w_c, w_l)` vectors in
`app/auto_router/weighted_strategy.py::WEIGHT_PROFILES` are **first-draft placeholders**. This
analysis sweeps those weights offline over RouterBench (36,497 items, zero spend) to trace the
quality/cost frontier, cross-checks against the cached Phase-2 judged results on 91 real prompts,
and recommends a tuned vector per profile.

---

## TL;DR — recommended vectors

| profile | current `(w_q,w_c,w_l)` | **recommended** | current α | **rec. α** | why |
|---|---|---|---|---|---|
| `quality_first` | (0.70, 0.15, 0.15) | **(0.75, 0.15, 0.10)** | 0.824 | 0.833 | confirm quality-max; demote latency to the smallest tie-breaker |
| `balanced` | (0.40, 0.30, 0.30) | **(0.40, 0.35, 0.25)** | 0.571 | 0.533 | move toward the true knee; raise cost weight closer to quality |
| `cost_first` | (0.20, 0.65, 0.15) | **(0.15, 0.70, 0.15)** | 0.235 | 0.176 | firmer cost-min; keep a 0.15 quality floor |
| `latency_first` | (0.25, 0.15, 0.60) | **(0.15, 0.25, 0.60)** | 0.625 | 0.375 | **fix the mis-composition** (see below); residual now cost-leaning |

where **α = w_q / (w_q + w_c)** is the quality share of the quality+cost weight — the *only* thing
that moves the offline pick (the latency term has no signal offline; see Method).

**The one must-fix:** `latency_first`'s current α = 0.625 is **higher** than `balanced`'s
α = 0.571 — i.e. after its dominant latency weight, its residual leans *quality*, so offline it is
*more* quality-leaning than "balanced". That is incoherent for a "fastest acceptable" profile. The
recommended (0.15, 0.25, 0.60) keeps latency dominant (0.60) but makes the residual lean **cost**
(fast models are typically the cheap ones), so the profile is coherent and distinct.

The recommendation also spreads the four α values apart (0.18 / 0.38 / 0.53 / 0.83) and gives a
coherent latency ladder (w_l: 0.10 < 0.15 < 0.25 < 0.60 for quality/cost/balanced/latency). The
current draft has three of the four profiles at α > 0.5, so they collapse onto the *same* operating
point offline (see the frontier table).

---

## Method

Sweep driver: **`router_eval/weight_sweep.py`** (`python -m router_eval.weight_sweep --source routerbench`).
It reuses the Phase-1 replay: each policy names a model per item and we look up that model's
**precomputed** RouterBench score + cost. Nothing is inferred, judged, or called.

Two structural facts make the sweep both exact and instant:

1. **Only α drives the offline pick.** RouterBench has no latency field, so the ported scorer
   (`policies.score_pool_quality_cost`, the faithful Quality+Cost subset of the production scorer)
   drops the latency term and renormalizes the remaining weights — *exactly* how the production
   scorer degrades a missing signal. The pick then depends only on
   `α = w_q/(w_q+w_c)`. Two vectors with the same α route identically offline; **the latency axis is
   invisible here** (a real limitation — see "Not covered").
2. **The pick depends on an item only through its task category** (the replay always offers the full
   11-model candidate set). So we precompute, per category, each pool model's summed score and cost
   once, then score any weight vector in O(#categories) by calling the *same*
   `score_pool_quality_cost` the production `WeightedPolicy` uses.

`weight_sweep.py --self-check` re-runs the full per-item `WeightedPolicy` on a sample and asserts the
fast path is byte-identical (passes on both the fixture and the 36k set). The sweep needed a
one-line, production-safe extension: `WeightedPolicy.__init__` now accepts an optional explicit
`weights` vector (default `None` → the frozen named profile, so **production behaviour is
unchanged**; only the sweep passes an arbitrary vector). Covered by `tests/test_weight_sweep.py`.

Outputs (written under `results/routerbench_0shot/`): `weight_sweep.csv` (the full 66-point
`(w_q,w_c,w_l)` step-0.1 simplex grid) and `weight_frontier_alpha.csv` (the fine α curve).

---

## The RouterBench frontier (36,497 items, premium tier, offline)

Reference points (mean judged-proxy quality, mean inference $ / request):

| policy | mean quality | infer $ |
|---|---|---|
| oracle (per-item hindsight ceiling) | 0.9121 | 0.000242 |
| always_premium | 0.7814 | 0.003293 |
| benchmark (pool rank-0) | 0.7411 | 0.003116 |
| random | 0.5192 | 0.000824 |
| always_cheapest | 0.3061 | 0.000046 |

*(A constant classifier tax of $0.000151/req applies to every `weighted` point — the benchmark
`gpt-4o-mini` classify call it always pays — so it never changes the frontier shape.)*

**Weighted frontier as α sweeps 0 → 1** — three distinct operating points:

| α range | mean quality | infer $ | interpretation |
|---|---|---|---|
| 0.000 – 0.474 | 0.5471 | 0.000135 | **cost corner** — picks the cheapest pool model everywhere |
| 0.475 – 0.499 | 0.5456 | 0.000782 | dominated middle (see note) |
| 0.500 – 1.000 | 0.6999 | 0.003066 | **quality corner** — picks pool rank-0 (≈ benchmark) |

The frontier is **bimodal with a cliff at α = 0.5**, not a smooth curve. Cause: the premium-tier pool
is only the **3 brand-mapped RouterBench models** — `claude-v2`, `gpt-4-1106-preview`,
`mixtral-8x7b-chat` — so as cost pressure rises the argmax simply flips from the top-ranked model to
`mixtral` (the cheap one) at a category-dependent threshold (≈0.474 for claude-v2-topped categories,
exactly 0.5 for gpt-4-topped ones). There is no rich intermediate to land on.

> **Note on the dominated middle:** in [0.475, 0.499] the factuality categories switch to `claude-v2`
> while the rest stay on `mixtral`, and mean quality actually *drops* (0.5471 → 0.5456) at higher
> cost — because on these 2023-era models the frozen SUPERMODE rank disagrees with RouterBench's
> per-task scores (`mixtral` out-scores `claude-v2` on factuality here). This is a RouterBench
> artifact, not a scorer bug; it's one reason RouterBench is good for the frontier *shape* and the α
> parameterization but not for the exact quality numbers.

**Current draft profiles projected onto this frontier:**

| profile | α | lands at | mean q | infer $ |
|---|---|---|---|---|
| `quality_first` | 0.824 | quality corner | 0.6999 | 0.003066 |
| `balanced` | 0.571 | **quality corner** | 0.6999 | 0.003066 |
| `latency_first` | 0.625 | **quality corner** | 0.6999 | 0.003066 |
| `cost_first` | 0.235 | cost corner | 0.5471 | 0.000135 |

Three of the four profiles collapse onto the identical quality-corner point — the current draft is
**under-differentiated on the quality/cost axis**. Only `cost_first` actually binds.

---

## Cross-check on 91 real prompts (Phase-2, cached, judged — no live calls)

RouterBench's 3-model pool is coarse and its models are from 2023. The Phase-2 harness routes over
the **live Mesh catalog** (current models, real prices) and scores answers with an Opus judge. Its
cached results (`phase2/out/strategy_aggregate.csv`, from the earlier live run — reused here, **not
re-run**) give real anchors:

| strategy | mean judge quality | infer $ | note |
|---|---|---|---|
| benchmark (pool rank-0 ≈ weighted quality corner, α→1) | 0.6734 | 0.006727 | |
| **weighted @ `balanced` (α = 0.571)** | **0.6364** | **0.003086** | the live run's actual profile |
| registry | 0.6598 | 0.003594 | |
| served (production today) | 0.6570 | 0.014362 | ground-truth reference |

**Key result:** on the *real* catalog, `balanced` (α = 0.571) **binds hard** — moving from the
quality corner (benchmark) to `balanced` costs **−5.5% quality for −54% cost** (0.6734→0.6364;
$0.00673→$0.00309), and beats the model production actually served on cost by **4.6×** at −3%
quality. This is the opposite of the RouterBench picture (where α = 0.571 sat inertly at the quality
corner): the real catalog's **richer pool smooths the α = 0.5 cliff**, so a moderate α *does* buy a
big, cheap cost reduction. This is the evidence that anchors `balanced` around α ≈ 0.53–0.57 rather
than pushing it to the corners.

### Why the *full* 91-prompt profile sweep is a live follow-up, not done here

Re-scoring every profile offline over the 91 prompts would need the exact live catalog the run used
— its **model-id set and its real per-model prices** — to reproduce weighted's picks, then a cached
judge score for each pick. Neither is snapshotted: the pipeline fetches the catalog live and never
persists it, and only *sample* prices are on disk. Reconstructing from the sample catalog diverges
(34/91 picks differ from the recorded run, because real prices reorder the cheap end) and ~15–30% of
the swept picks land on `(prompt, model)` pairs that were never answered/judged → they'd need live
calls. So a faithful swept validation is **deferred to a live run** (see below). The four cached
anchors above are real and sufficient to confirm the *direction* of the recommendation.

---

## Per-profile recommendation & justification

Weights sum to 1.0. "Offline" = RouterBench position; "real" = the Phase-2 real-catalog expectation.

### `quality_first` → (0.75, 0.15, 0.10), α = 0.833  *(confirm)*
Intent (best model, cost/latency only as tie-breakers) is already met — the current vector sits at
the quality corner. The only change is demoting latency (0.15→0.10) below cost so a quality-max user,
who cares least about speed, has latency as the *smallest* tie-breaker. Offline unchanged
(0.6999 / $0.003066); real ≈ benchmark (0.6734 / $0.00673).

### `balanced` → (0.40, 0.35, 0.25), α = 0.533  *(tune)*
The knee. Keep the quality weight at 0.40 but raise cost 0.30→0.35 (nearer parity) and trim latency
0.30→0.25, dropping α from 0.571 to 0.533 — staying inside the **validated knee band** (the Phase-2
anchor at α = 0.571 already halves cost for −5.5% quality) while nudging picks a touch cheaper.
*Offline this is indistinguishable from `quality_first`* (both α ≥ 0.5 → quality corner): RouterBench's
3-model pool cannot render a balanced middle, so this profile's benefit shows up only on the real
catalog. Expected real: ≈ 0.63–0.64 quality at ≈ $0.0030 or slightly below — half the cost of the
quality corner. (Do **not** push α < 0.474: on any coarse pool that collapses `balanced` onto the
`cost_first` corner.)

### `cost_first` → (0.15, 0.70, 0.15), α = 0.176  *(confirm, slightly firmer)*
Already at the cost corner. Lower α 0.235→0.176 for a firmer cost-min, keeping a 0.15 quality floor so
it never rides pure price into a pathologically weak model. Offline unchanged
(0.5471 / $0.000135 — a **22× cheaper** than the quality corner for −0.153 quality); real: the cheapest
of the four.

### `latency_first` → (0.15, 0.25, 0.60), α = 0.375  *(fix)*
Keep latency dominant at 0.60, but **flip the residual 0.40 from quality-leaning (0.25 q / 0.15 c,
α = 0.625) to cost-leaning (0.15 q / 0.25 c, α = 0.375)**. Rationale: (a) among similarly fast models,
prefer the cheaper one — fast models are usually the cheap ones, so this is coherent; (b) it makes
`latency_first` distinct from `quality_first` instead of accidentally *more* quality-leaning than
`balanced`. Offline it now drops to the cost corner (0.5471 / $0.000135) because the latency term is
inert there; **in production the 0.60 latency weight dominates** and steers to the lowest-p95 models —
the real behaviour change, which needs the live/perf-signal validation.

### Expected deltas vs the current first-draft

| profile | Δ offline (RouterBench) | where the real change lands |
|---|---|---|
| `quality_first` | none (both quality corner) | negligible; latency only a smaller tie-breaker |
| `balanced` | none offline (pool too coarse) | real catalog: picks ~equal/cheaper than today's balanced |
| `cost_first` | none (both cost corner) | real catalog: firmly the cheapest profile |
| `latency_first` | quality corner → cost corner (−0.153 q, −$0.00293) | prod: latency term (0.60) drives low-p95 picks |

---

## Not covered / caveats (read before activating)

- **Latency is a proxy offline.** RouterBench has no latency field, so every offline number reflects
  only the quality+cost projection (α). The recommended `w_l` values (0.10 / 0.15 / 0.25 / 0.60) are
  set by intent and the coherence ladder; their *actual* effect uses the production
  `ProviderPerfProfile` p95 signal (MESH-407) and is **not measured here**.
- **The balanced knee's exact quality is extrapolated.** Only α = 0.571 has a measured real-catalog
  point (Phase-2). α = 0.533 is expected to be equal-or-cheaper at slightly lower quality, but is not
  directly measured.
- **Coarse offline pool.** The premium RouterBench pool is 3 brand-mapped models with 2023 prices, and
  the frozen SUPERMODE rank disagrees with RouterBench's per-task scores in places (the dominated
  middle). Use RouterBench for the frontier *shape* and the α insight, the real catalog for magnitudes.
- **Production defaults untouched.** `WEIGHT_PROFILES` is read, never written, by this analysis; the
  sweep's only code change is the optional `weights` override (default `None`).

### Validation follow-up (needs a live run — operator's call)
Once the tuned vectors are set in `WEIGHT_PROFILES`, validate each on the 91 real prompts with the
existing flag — no new code needed:

    MESH_API_KEY=sk-... python -m router_eval.phase2 --live --weight-profile quality_first
    #  ... repeat for balanced / cost_first / latency_first

This will answer + judge the new picks (the ~15–30% not already cached) and produce a faithful
real-catalog quality/cost point per tuned profile, including the latency term's real effect. That
live run — not this offline analysis — is what should gate flipping `auto_router_weighted_enabled`.

## Reproduce

    # offline frontier sweep (uses the cached RouterBench pkl; zero spend)
    HF_HUB_OFFLINE=1 python -m router_eval.weight_sweep --source routerbench --self-check
    # unit tests (pure stdlib fixture, no network)
    python -m pytest router_eval/tests/test_weight_sweep.py -q
