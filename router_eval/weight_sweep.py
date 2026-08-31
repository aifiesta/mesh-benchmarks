"""
Offline weight-vector frontier sweep for the MESH-644 `weighted` strategy (tuning aid).

    python -m router_eval.weight_sweep                       # fixture (offline, tiny)
    python -m router_eval.weight_sweep --source routerbench  # real 36k set (cached pkl)

Sweeps the weighted policy's ``(w_q, w_c, w_l)`` vector over RouterBench OFFLINE — ZERO
spend, ZERO live/judge/classifier calls — and traces the quality/cost frontier so the
four named ``WEIGHT_PROFILES`` in ``app/auto_router/weighted_strategy.py`` can be tuned
with data before the strategy goes live. Companion write-up: ``RESULTS-weighted-tuning.md``.

Why this is both EXACT and fast
-------------------------------
Two structural facts collapse the search:

  * On RouterBench the LATENCY term has no signal (no latency field), so the ported
    scorer (``policies.score_pool_quality_cost``) drops it and renormalizes the remaining
    weights — exactly as the production scorer degrades a missing signal. So the pick
    depends ONLY on ``alpha = w_q / (w_q + w_c)`` (the quality share of the *present*
    weight). Two vectors with the same alpha route identically offline; the latency axis
    is INVISIBLE here (that is a real limitation, called out in the write-up).

  * The replay always offers the full 11-model candidate set, so the category pool — and
    therefore the pick — depends on an item ONLY through its task category.

So we precompute, per category, each pool model's summed score and cost over the items in
that category, then score any weight vector in O(#categories) by reusing the SAME
``score_pool_quality_cost`` the production ``WeightedPolicy`` calls. ``--self-check``
re-runs the full per-item ``WeightedPolicy`` on a sample and asserts the fast path matches.

Nothing here changes production defaults: ``WEIGHT_PROFILES`` is read, never written.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from router_eval.data import Item, load_fixture, load_routerbench
from router_eval.metrics import (
    BENCHMARK_CLASSIFIER,
    classifier_call_cost_usd,
    estimate_prompt_tokens,
    evaluate_policy,
)
from router_eval.policies import (
    WEIGHT_PROFILES,
    AlwaysCheapestPolicy,
    AlwaysPremiumPolicy,
    BenchmarkPolicy,
    OraclePolicy,
    RandomPolicy,
    Weights,
    WeightedPolicy,
    score_pool_quality_cost,
)
from router_eval.routerbench_bridge import eval_to_category, ranked_routerbench_models

DEFAULT_SEED = 20260821
RESULTS_ROOT = Path(__file__).parent / "results"


# ── Per-category precompute ──────────────────────────────────────────────────────
@dataclass
class CategoryAgg:
    """Everything the fast path needs for one task category."""

    category: str
    pool: list[str]  # ranked pool (rank 0 = best), the models weighted chooses among
    n: int = 0
    score_sum: dict[str, float] = field(default_factory=dict)  # model -> Σ score over cat items
    cost_sum: dict[str, float] = field(default_factory=dict)  # model -> Σ cost over cat items


@dataclass
class SweepModel:
    aggs: dict[str, CategoryAgg]
    mean_cost: dict[str, float]  # global per-model mean cost (weighted's cost signal)
    n_items: int
    mean_classifier_tax_usd: float  # constant across the weighted sweep


def build_sweep_model(items: list[Item], tier: str = "premium") -> SweepModel:
    """One O(N) pass: per-category pools + score/cost sums, the global mean-cost signal,
    and the (constant) benchmark-classifier tax the weighted strategy always pays."""
    candidates = sorted(items[0].models)

    # Global mean cost per model — identical to WeightedPolicy.fit.
    tot: dict[str, float] = {}
    cnt: dict[str, int] = {}
    for it in items:
        for m, c in it.costs.items():
            tot[m] = tot.get(m, 0.0) + c
            cnt[m] = cnt.get(m, 0) + 1
    mean_cost = {m: tot[m] / cnt[m] for m in tot if cnt[m]}

    pool_by_cat: dict[str, list[str]] = {}
    aggs: dict[str, CategoryAgg] = {}
    tax_sum = 0.0
    for it in items:
        cat = eval_to_category(it.task)
        pool = pool_by_cat.get(cat)
        if pool is None:
            pool = ranked_routerbench_models(it.task, candidates, tier=tier)
            pool_by_cat[cat] = pool
            aggs[cat] = CategoryAgg(category=cat, pool=pool,
                                    score_sum=dict.fromkeys(pool, 0.0),
                                    cost_sum=dict.fromkeys(pool, 0.0))
        agg = aggs[cat]
        agg.n += 1
        for m in pool:
            agg.score_sum[m] += it.scores.get(m, 0.0)
            agg.cost_sum[m] += it.costs.get(m, 0.0)
        tax_sum += classifier_call_cost_usd(BENCHMARK_CLASSIFIER, estimate_prompt_tokens(it.prompt))

    return SweepModel(aggs=aggs, mean_cost=mean_cost, n_items=len(items),
                      mean_classifier_tax_usd=tax_sum / len(items) if items else 0.0)


# ── Scoring a weight vector on the fast path ─────────────────────────────────────
@dataclass
class FrontierPoint:
    w_q: float
    w_c: float
    w_l: float
    alpha: float  # w_q / (w_q + w_c); the only thing that moves the pick offline
    mean_score: float
    mean_cost: float
    mean_cost_with_tax: float
    # Deterministic signature of the per-category pick set — distinct signatures are the
    # distinct operating points on the frontier staircase.
    pick_signature: tuple[tuple[str, str], ...]


def _alpha(w: Weights) -> float:
    denom = w.q + w.c
    return (w.q / denom) if denom > 0 else 1.0  # w_q=w_c=0 → scorer falls back to quality


def score_weights(model: SweepModel, w: Weights) -> FrontierPoint:
    """Mean quality/cost for one weight vector, reusing the production scorer per category."""
    score_sum = 0.0
    cost_sum = 0.0
    sig: list[tuple[str, str]] = []
    for cat, agg in model.aggs.items():
        if not agg.pool:
            continue
        pick = score_pool_quality_cost(agg.pool, model.mean_cost, w)[0]
        score_sum += agg.score_sum[pick]
        cost_sum += agg.cost_sum[pick]
        sig.append((cat, pick))
    n = model.n_items or 1
    ms, mc = score_sum / n, cost_sum / n
    return FrontierPoint(
        w_q=w.q, w_c=w.c, w_l=w.l, alpha=_alpha(w),
        mean_score=ms, mean_cost=mc, mean_cost_with_tax=mc + model.mean_classifier_tax_usd,
        pick_signature=tuple(sorted(sig)),
    )


# ── Grids ────────────────────────────────────────────────────────────────────────
def simplex_grid(step: int = 1, total: int = 10) -> list[Weights]:
    """All (w_q, w_c, w_l) on the simplex summing to 1.0 at 1/total granularity."""
    out: list[Weights] = []
    for i in range(0, total + 1, step):
        for j in range(0, total - i + 1, step):
            k = total - i - j
            out.append(Weights(i / total, j / total, k / total))
    return out


def alpha_grid(n: int = 1001) -> list[Weights]:
    """Fine sweep of the true offline free parameter alpha = w_q/(w_q+w_c) in [0,1],
    encoded as (alpha, 1-alpha, 0.0). Used to locate the frontier's breakpoints."""
    return [Weights(i / (n - 1), 1 - i / (n - 1), 0.0) for i in range(n)]


# Candidate TUNED vectors this analysis recommends (filled from the frontier; see the
# write-up for the derivation). Kept alongside the current defaults for a side-by-side.
CURRENT_PROFILES = dict(WEIGHT_PROFILES)


# ── Reference policies (context for the frontier) ────────────────────────────────
def reference_points(items: list[Item], tier: str, seed: int) -> dict[str, tuple[float, float]]:
    refs = {
        "random": RandomPolicy(),
        "always_cheapest": AlwaysCheapestPolicy(),
        "always_premium": AlwaysPremiumPolicy(),
        "benchmark": BenchmarkPolicy(tier=tier),
        "oracle": OraclePolicy(),
    }
    out: dict[str, tuple[float, float]] = {}
    for name, pol in refs.items():
        r = evaluate_policy(pol, items, seed)
        out[name] = (r.mean_score, r.mean_cost)
    return out


# ── Self-check: fast path vs the real per-item policy ────────────────────────────
def self_check(items: list[Item], model: SweepModel, tier: str, seed: int) -> None:
    sample = items if len(items) <= 4000 else random.Random(seed).sample(items, 4000)
    for w in (Weights(0.7, 0.15, 0.15), Weights(0.4, 0.3, 0.3), Weights(0.2, 0.65, 0.15),
              Weights(0.0, 0.5, 0.5), Weights(1.0, 0.0, 0.0)):
        pol = WeightedPolicy(tier=tier, weights=w)
        pol.fit(items)
        rng = random.Random(seed)
        s_full = c_full = 0.0
        for it in sample:
            m = pol.pick(it, it.models, rng)
            s_full += it.scores.get(m, 0.0)
            c_full += it.costs.get(m, 0.0)
        s_full /= len(sample)
        c_full /= len(sample)
        # Fast path restricted to the same sample.
        sm = build_sweep_model(sample, tier=tier)
        fp = score_weights(sm, w)
        assert abs(fp.mean_score - s_full) < 1e-9 and abs(fp.mean_cost - c_full) < 1e-9, (
            f"fast-path mismatch at {w}: fast=({fp.mean_score},{fp.mean_cost}) "
            f"full=({s_full},{c_full})"
        )
    print("self-check OK: fast path == full WeightedPolicy on the sample")


# ── Report ───────────────────────────────────────────────────────────────────────
def distinct_operating_points(model: SweepModel) -> list[tuple[float, float, FrontierPoint]]:
    """Sweep alpha finely; return the distinct frontier vertices as
    (alpha_lo, alpha_hi, point), i.e. the staircase of Pareto operating points."""
    points: list[tuple[float, FrontierPoint]] = [(_alpha(w), score_weights(model, w)) for w in alpha_grid()]
    vertices: list[tuple[float, float, FrontierPoint]] = []
    cur_sig = None
    a_lo = 0.0
    prev_a = 0.0
    for a, fp in points:
        if fp.pick_signature != cur_sig:
            if cur_sig is not None:
                vertices.append((a_lo, prev_a, last_fp))
            cur_sig = fp.pick_signature
            a_lo = a
        prev_a = a
        last_fp = fp
    vertices.append((a_lo, prev_a, last_fp))
    return vertices


def _fmt(w: Weights) -> str:
    return f"({w.q:.2f},{w.c:.2f},{w.l:.2f})"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="router_eval.weight_sweep", description=__doc__)
    p.add_argument("--source", choices=["fixture", "routerbench"], default="fixture")
    p.add_argument("--shots", type=int, choices=[0, 5], default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tier", choices=["premium", "standard"], default="premium")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--self-check", action="store_true", help="assert fast path == full policy")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    items = load_fixture() if args.source == "fixture" else load_routerbench(shots=args.shots, limit=args.limit)
    if not items:
        print("No items loaded — aborting.", file=sys.stderr)
        return 1
    tag = "fixture" if args.source == "fixture" else f"routerbench_{args.shots}shot"

    model = build_sweep_model(items, tier=args.tier)
    if args.self_check:
        self_check(items, model, args.tier, args.seed)

    refs = reference_points(items, args.tier, args.seed)

    print(f"\nWeight-vector frontier sweep — source={tag}, items={len(items)}, tier={args.tier}\n")
    print("Reference points (mean_score, infer_cost$):")
    for name in ("oracle", "always_premium", "benchmark", "random", "always_cheapest"):
        s, c = refs[name]
        print(f"  {name:<16} q={s:.4f}  ${c:.6f}")
    print(f"  classifier tax (const, all weighted pts): ${model.mean_classifier_tax_usd:.6f}\n")

    # Distinct operating points (the frontier staircase).
    vertices = distinct_operating_points(model)
    print("Frontier — distinct operating points as alpha=w_q/(w_q+w_c) sweeps 0→1:")
    print(f"  {'alpha_range':<18}{'mean_q':>9}{'infer$':>11}{'q_vs_bench':>12}{'cost_vs_bench':>14}")
    bq, bc = refs["benchmark"]
    for a_lo, a_hi, fp in vertices:
        print(f"  {a_lo:.3f}–{a_hi:.3f}      {fp.mean_score:>9.4f}{fp.mean_cost:>11.6f}"
              f"{fp.mean_score - bq:>+12.4f}{(fp.mean_cost - bc):>+14.6f}")

    # Named profiles: current defaults.
    print("\nCurrent WEIGHT_PROFILES (first-draft) projected onto the offline frontier:")
    print(f"  {'profile':<16}{'vector':<20}{'alpha':>7}{'mean_q':>9}{'infer$':>11}")
    for name, w in CURRENT_PROFILES.items():
        fp = score_weights(model, w)
        print(f"  {name:<16}{_fmt(w):<20}{fp.alpha:>7.3f}{fp.mean_score:>9.4f}{fp.mean_cost:>11.6f}")

    # Full simplex grid CSV.
    out_dir = args.out or (RESULTS_ROOT / tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = simplex_grid()
    with (out_dir / "weight_sweep.csv").open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["w_q", "w_c", "w_l", "alpha", "mean_score", "mean_infer_cost_usd",
                      "mean_cost_with_classifier_usd", "n_distinct_category_picks"])
        for w in grid:
            fp = score_weights(model, w)
            wtr.writerow([f"{w.q:.2f}", f"{w.c:.2f}", f"{w.l:.2f}", f"{fp.alpha:.4f}",
                          f"{fp.mean_score:.6f}", f"{fp.mean_cost:.8f}",
                          f"{fp.mean_cost_with_tax:.8f}", len(fp.pick_signature)])
    # Fine alpha CSV (the frontier curve).
    with (out_dir / "weight_frontier_alpha.csv").open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["alpha", "mean_score", "mean_infer_cost_usd", "mean_cost_with_classifier_usd"])
        for w in alpha_grid():
            fp = score_weights(model, w)
            wtr.writerow([f"{fp.alpha:.4f}", f"{fp.mean_score:.6f}", f"{fp.mean_cost:.8f}",
                          f"{fp.mean_cost_with_tax:.8f}"])
    print(f"\nWrote {out_dir/'weight_sweep.csv'} (simplex grid) and "
          f"{out_dir/'weight_frontier_alpha.csv'} (fine curve)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
