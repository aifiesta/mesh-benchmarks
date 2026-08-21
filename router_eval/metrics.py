"""
Replay metrics: turn each policy's per-item picks into a quality/cost point,
a gap-to-oracle, and the AC3 "does benchmark beat random?" answer.

Each policy is summarised as a POINT in (mean_cost, mean_score) space:
  * mean_score — mean RouterBench performance score of the picked models ([0,1]).
  * mean_cost  — mean USD cost of the picked responses.
  * gap_to_oracle — oracle mean_score minus this policy's mean_score (quality
    headroom left on the table). 0.0 for the oracle itself.

Classifier-cost TODO hook
-------------------------
In production the `benchmark` and `registry` strategies pay an EXTRA per-request
LLM classifier call (cost + latency) before they can route. `not_diamond` makes an
external router call but NO internal classifier LLM call; `heuristic` is rule-based
with no LLM call. This replay does not yet charge any of that: `classifier_cost_usd`
is a hook wired to `CLASSIFIER_COST_USD_PER_REQUEST` (currently 0.0). Set it from
real classifier telemetry (tokens x price of the classifier model) to make the
cost axis fair across strategies — until then, treat benchmark/registry cost as a
LOWER BOUND and mind the asymmetry with not_diamond/heuristic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from router_eval.data import Item
from router_eval.policies import Policy

# TODO(MESH-708): source this from classifier telemetry (classifier model tokens x
# unit price) instead of 0.0. Until then the benchmark/registry cost points omit the
# per-request classifier surcharge they really pay. See module docstring.
CLASSIFIER_COST_USD_PER_REQUEST = 0.0


@dataclass
class Pick:
    """One policy's decision on one item, with the realised (known) outcome."""

    sample_id: str
    task: str
    model: str
    score: float
    cost: float


@dataclass
class PolicyResult:
    name: str
    n: int
    mean_score: float
    mean_cost: float
    pays_classifier_call: bool
    classifier_cost_usd: float
    mean_cost_with_classifier: float
    gap_to_oracle: float | None = None
    picks: list[Pick] = field(default_factory=list)


def evaluate_policy(policy: Policy, items: list[Item], seed: int) -> PolicyResult:
    """Run one policy over all items with a fresh seeded RNG (order-independent)."""
    rng = random.Random(seed)
    policy.fit(items)

    picks: list[Pick] = []
    score_sum = 0.0
    cost_sum = 0.0
    for it in items:
        model = policy.pick(it, it.models, rng)
        if model is None:
            continue
        score = it.scores.get(model, 0.0)
        cost = it.costs.get(model, 0.0)
        score_sum += score
        cost_sum += cost
        picks.append(Pick(it.sample_id, it.task, model, score, cost))

    n = len(picks)
    mean_score = score_sum / n if n else 0.0
    mean_cost = cost_sum / n if n else 0.0
    classifier_cost = CLASSIFIER_COST_USD_PER_REQUEST if policy.pays_classifier_call else 0.0
    return PolicyResult(
        name=policy.name,
        n=n,
        mean_score=mean_score,
        mean_cost=mean_cost,
        pays_classifier_call=policy.pays_classifier_call,
        classifier_cost_usd=classifier_cost,
        mean_cost_with_classifier=mean_cost + classifier_cost,
        picks=picks,
    )


def evaluate_all(policies: list[Policy], items: list[Item], seed: int) -> list[PolicyResult]:
    """Evaluate every policy and fill in gap_to_oracle (if an oracle is present)."""
    results = [evaluate_policy(p, items, seed) for p in policies]
    oracle = next((r for r in results if r.name == "oracle"), None)
    if oracle is not None:
        for r in results:
            r.gap_to_oracle = oracle.mean_score - r.mean_score
    return results


@dataclass
class AC3Answer:
    benchmark_mean_score: float
    random_mean_score: float
    delta: float  # benchmark - random
    relative_uplift: float | None  # delta / random, None if random == 0
    benchmark_beats_random: bool


def ac3_benchmark_vs_random(results: list[PolicyResult]) -> AC3Answer | None:
    """AC3: does the frozen SUPERMODE_BENCHMARKS beat the random baseline?"""
    by_name = {r.name: r for r in results}
    bench = by_name.get("benchmark")
    rand = by_name.get("random")
    if bench is None or rand is None:
        return None
    delta = bench.mean_score - rand.mean_score
    rel = (delta / rand.mean_score) if rand.mean_score else None
    return AC3Answer(
        benchmark_mean_score=bench.mean_score,
        random_mean_score=rand.mean_score,
        delta=delta,
        relative_uplift=rel,
        benchmark_beats_random=delta > 0,
    )
