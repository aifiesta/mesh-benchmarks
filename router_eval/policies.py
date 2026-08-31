"""
Routing policies for the offline replay.

The adapter interface every policy implements:

    policy.fit(items)                     # optional: precompute global state
    policy.pick(item, candidates, rng)    # -> model_id chosen for this item
    policy.classifier_calls(item, picked) # -> classifier model_ids invoked (tax driver)

`candidates` is the model-id set available for the item (RouterBench's fixed 11).
Realistic policies read only `item.task` / `item.prompt` + `candidates`; only
OraclePolicy reads `item.scores` / `item.costs` (hindsight — that is the whole
point of the oracle).

`pays_classifier_call` marks policies that, IN PRODUCTION, spend an extra per-request
LLM classifier call (cost + latency). `classifier_calls()` returns the classifier
model id(s) actually invoked FOR A GIVEN ITEM, which is what metrics.py prices into
the classifier tax (AC2). The two differ for `heuristic`: it pays NO classifier on a
fast-lane hit but DOES pay benchmark's classifier when its gate declines, so its
`classifier_calls` is per-item, not a flat bool.

Implemented: RandomPolicy, AlwaysPremiumPolicy, AlwaysCheapestPolicy, OraclePolicy,
BenchmarkPolicy, HeuristicPolicy (Phase-2 A: ported), WeightedPolicy (Phase-2 A:
portable Q+C subset). Stubbed: RegistryStub (needs a live classifier over the live
registry — Phase 2), NotDiamondStub (external API — Phase 2).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import NamedTuple

from router_eval.data import Item
from router_eval.heuristic_gate import gate as heuristic_gate
from router_eval.routerbench_bridge import (
    BRAND_TO_ROUTERBENCH_PREMIUM,
    ranked_routerbench_models,
    resolve_benchmark_model,
    resolve_heuristic_conversation_model,
)

# Classifier model each classifying strategy drives in production (routersvc
# settings: primary/benchmark = gpt-4o-mini, registry = gemini-3-flash-preview).
# metrics.py maps these ids to a priced ClassifierSpec.
BENCHMARK_CLASSIFIER_MODEL = "openai/gpt-4o-mini"
REGISTRY_CLASSIFIER_MODEL = "google/gemini-3-flash-preview"


class Policy:
    """Base adapter. Subclasses set `name` and implement `pick`."""

    name: str = "policy"
    # Does this policy pay a per-request classifier LLM call on EVERY request?
    pays_classifier_call: bool = False
    # The classifier model it drives (None for rule-based / no-classifier policies).
    classifier_model_id: str | None = None

    def fit(self, items: list[Item]) -> None:
        """Optional hook to precompute global state before the replay loop."""

    def pick(self, item: Item, candidates: list[str], rng: random.Random) -> str | None:
        raise NotImplementedError

    def classifier_calls(self, item: Item, picked_model: str | None) -> list[str]:
        """Classifier model id(s) this policy invokes FOR THIS ITEM — the real tax
        driver metrics.py prices. Default: one call to `classifier_model_id` iff the
        policy classifies on every request. Overridden by `heuristic` (conditional)."""
        if self.pays_classifier_call and self.classifier_model_id:
            return [self.classifier_model_id]
        return []


# ── Baselines ──────────────────────────────────────────────────────────────────
class RandomPolicy(Policy):
    """Uniform random pick from the candidate set. The floor a router must beat."""

    name = "random"

    def pick(self, item: Item, candidates: list[str], rng: random.Random) -> str | None:
        return rng.choice(sorted(candidates))


class _FixedModelPolicy(Policy):
    """Always route to one fixed model, chosen in fit() by mean cost across items."""

    #: True -> pick the most expensive model, False -> the cheapest.
    most_expensive: bool = True

    def __init__(self) -> None:
        self.model: str | None = None

    def fit(self, items: list[Item]) -> None:
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for it in items:
            for m, c in it.costs.items():
                totals[m] += c
                counts[m] += 1
        means = {m: totals[m] / counts[m] for m in totals if counts[m]}
        if not means:
            return
        self.model = max(means, key=means.get) if self.most_expensive else min(means, key=means.get)

    def pick(self, item: Item, candidates: list[str], rng: random.Random) -> str | None:
        if self.model in candidates:
            return self.model
        # Fixed model absent from this item's candidates: fall back sensibly.
        key = item.costs.get
        return (max if self.most_expensive else min)(candidates, key=lambda m: key(m, 0.0))


class AlwaysPremiumPolicy(_FixedModelPolicy):
    """Always route to the single most-expensive model (a stand-in for 'best')."""

    name = "always_premium"
    most_expensive = True


class AlwaysCheapestPolicy(_FixedModelPolicy):
    """Always route to the single cheapest model."""

    name = "always_cheapest"
    most_expensive = False


class OraclePolicy(Policy):
    """Best model per item with hindsight: max score, ties broken by lowest cost.

    This is the quality-headroom CEILING — no online policy can beat it. (It is a
    quality-max oracle; RouterBench also ships its own cheapest-correct
    `oracle_model_to_route_to`, a different, cost-first oracle. See RESULTS.md.)
    """

    name = "oracle"

    def pick(self, item: Item, candidates: list[str], rng: random.Random) -> str | None:
        if not candidates:
            return None
        return max(
            sorted(candidates),
            key=lambda m: (item.scores.get(m, 0.0), -item.costs.get(m, 0.0)),
        )


# ── The strategy under test ─────────────────────────────────────────────────────
class BenchmarkPolicy(Policy):
    """
    Port of the routersvc `benchmark` strategy's PORTABLE core.

    Maps the item's task to a SUPERMODE_BENCHMARKS category, walks the frozen brand
    ranking best-first, and picks the first ranked brand that maps to a RouterBench
    model in the candidate set (random among ties). Falls back to `chatgpt`'s model,
    then to any candidate, when no ranked+mapped brand is available.

    Replayed under PERFECT task classification (uses ground-truth eval_name instead
    of an LLM reading the prompt) and CHARGING the per-request classifier call it
    makes in production via metrics.py's tax (AC2). See RESULTS.md caveats.
    """

    name = "benchmark"
    pays_classifier_call = True  # prod: per-request LLM classifier call (cost+latency)
    classifier_model_id = BENCHMARK_CLASSIFIER_MODEL

    def __init__(self, tier: str = "premium") -> None:
        self.tier = tier

    def pick(self, item: Item, candidates: list[str], rng: random.Random) -> str | None:
        chosen = resolve_benchmark_model(item.task, candidates, tier=self.tier, rng=rng)
        if chosen is not None:
            return chosen
        # Fallback chain mirrors the live default-model behaviour.
        default = BRAND_TO_ROUTERBENCH_PREMIUM.get("chatgpt")
        if default in candidates:
            return default
        return sorted(candidates)[0] if candidates else None


class HeuristicPolicy(Policy):
    """
    Port of the routersvc `heuristic` fast-lane strategy (origin/main).

    A rule-based prefilter in front of the classifier: `heuristic_gate.gate` string-
    checks the prompt, and when it is unambiguously trivial-conversational the request
    is routed to the conversation category's STANDARD-tier model with ZERO LLM calls.
    Any hint of task work / recency / code / links / digits / length declines, and — as
    in the live waterfall — the request falls through to the benchmark path. This port
    composes `BenchmarkPolicy` as that fallthrough.

    Portable in full: the gate reads only the prompt (threaded through `Item.prompt`),
    and the accepted-branch model resolution reuses the SUPERMODE table + the standard-
    tier brand map. The only non-ported bits are runtime plumbing (async version
    lookup, structured logging) with no bearing on the decision.

    Classifier tax: a fast-lane HIT pays nothing; a MISS pays benchmark's classifier
    (it fell through). So `classifier_calls` is per-item — see metrics.py AC2.

    NOTE on RouterBench: its prompts are benchmark tasks (MMLU/GSM-8K/code/…), none of
    which read as trivial small-talk, so the gate essentially never fires here and
    `heuristic` collapses onto `benchmark`. The fast lane is meaningfully exercised only
    on real conversational traffic (Phase 2) and on the unit fixtures. Documented in
    RESULTS.md.
    """

    name = "heuristic"
    pays_classifier_call = False  # fast lane makes NO classifier call; misses fall through
    classifier_model_id = BENCHMARK_CLASSIFIER_MODEL  # the classifier a MISS falls through to

    def __init__(self, tier: str = "premium") -> None:
        self.tier = tier
        self._benchmark = BenchmarkPolicy(tier=tier)

    def fit(self, items: list[Item]) -> None:
        self._benchmark.fit(items)

    def _fast_lane_model(self, item: Item, candidates: list[str]) -> str | None:
        """The fast-lane pick, or None if the gate declines OR the conversation
        standard model is not routable for this item (both → fall through)."""
        matched, _reason = heuristic_gate((item.prompt or "").strip())
        if not matched:
            return None
        return resolve_heuristic_conversation_model(candidates)

    def pick(self, item: Item, candidates: list[str], rng: random.Random) -> str | None:
        fast = self._fast_lane_model(item, candidates)
        if fast is not None:
            return fast
        return self._benchmark.pick(item, candidates, rng)  # waterfall → benchmark

    def classifier_calls(self, item: Item, picked_model: str | None) -> list[str]:
        # Fast-lane hit → no classifier. Miss → benchmark's classifier was paid.
        if self._fast_lane_model(item, item.models) is not None:
            return []
        return [BENCHMARK_CLASSIFIER_MODEL]


# ── Weighted scoring primitives (ported from weighted_strategy.py) ──────────────
class Weights(NamedTuple):
    q: float  # quality
    c: float  # cost
    l: float  # latency  # noqa: E741


# FROZEN first-draft profiles, verbatim from routersvc weighted_strategy.WEIGHT_PROFILES.
WEIGHT_PROFILES: dict[str, Weights] = {
    "quality_first": Weights(0.70, 0.15, 0.15),
    "balanced": Weights(0.40, 0.30, 0.30),  # default
    "cost_first": Weights(0.20, 0.65, 0.15),
    "latency_first": Weights(0.25, 0.15, 0.60),
}
DEFAULT_PROFILE = "balanced"
_LOG_EPSILON = 1e-6  # guards log10(0) for a free model


def _quality(rank: int, n: int) -> float:
    """Q(m) = 1 - rank/(N-1); rank 0 → 1.0, last → 0.0. Single-candidate pool → 1.0."""
    if n <= 1:
        return 1.0
    return 1.0 - rank / (n - 1)


def _minmax_lower_better(values: dict[str, float]) -> dict[str, float]:
    """Min-max a 'lower is better' quantity to a 0–1 higher-better scale. Empty → {};
    all-equal → all 1.0 (no spread to discriminate, so nobody is penalised)."""
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi == lo:
        return dict.fromkeys(values, 1.0)
    span = hi - lo
    return {k: 1.0 - (v - lo) / span for k, v in values.items()}


def score_pool_quality_cost(
    pool: list[str], mean_cost: dict[str, float], weights: Weights
) -> list[str]:
    """Rank a category pool by the PORTABLE weighted objective — quality + cost only.

    Faithful subset of routersvc `weighted_strategy.score_pool`: Q is the candidate's
    rank in the category ranking (1 - rank/(N-1)); C is the model's cost, log10-scaled
    then min-max inverted so cheaper → higher. The production LATENCY term is DROPPED —
    RouterBench has no latency field — which is exactly the source's "missing signal →
    drop the term and renormalize the remaining weights" degradation, applied uniformly.
    Each candidate's score renormalizes over its present terms. Returns model ids
    best-first; deterministic tie-break: score desc, pool rank, lexical id.
    """
    n = len(pool)
    cost_raw = {m: math.log10(max(mean_cost[m], _LOG_EPSILON)) for m in pool if m in mean_cost}
    cost_norm = _minmax_lower_better(cost_raw)

    scored: list[tuple[float, int, str]] = []
    for rank, m in enumerate(pool):
        q = _quality(rank, n)
        c = cost_norm.get(m)
        wsum = weights.q
        acc = weights.q * q
        if c is not None:
            wsum += weights.c
            acc += weights.c * c
        score = acc / wsum if wsum > 0 else q
        scored.append((score, rank, m))

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [m for _, _, m in scored]


class WeightedPolicy(Policy):
    """
    Port of routersvc `weighted` — the PORTABLE Quality+Cost subset (MESH-644).

    Classifies exactly as `benchmark` does (so it pays the same classifier tax), maps
    the category to its SUPERMODE ranking, intersects with the candidate pool, then —
    instead of taking rank-0 — scores each candidate on quality (rank) + cost and
    returns the argmax.

    HONEST portability boundary (see RESULTS.md):
      * Quality term Q — FULLY portable (the SUPERMODE rank).
      * Cost term C — a PROXY: production blends the model's $/1M prompt+completion price
        rows; RouterBench exposes only per-response $ cost, so we use each model's MEAN
        per-request cost across the replay set (log10 + min-max inverted). Directionally
        the same signal, but it folds in each model's verbosity, so it is not identical
        to the production price blend.
      * Latency term L — NOT portable: RouterBench has no latency field. Dropped and
        renormalized away, exactly as the source degrades a missing signal — so offline
        `weighted` scores on quality+cost with the 'balanced' profile's L weight removed.

    Ships DARK in prod (default-off kill switch → abstain → benchmark); the replay
    measures its ENABLED behaviour. On an empty category pool it abstains → benchmark.
    """

    name = "weighted"
    pays_classifier_call = True  # reuses benchmark's classifier (same model + cost)
    classifier_model_id = BENCHMARK_CLASSIFIER_MODEL

    def __init__(
        self,
        tier: str = "premium",
        profile: str = DEFAULT_PROFILE,
        weights: Weights | None = None,
    ) -> None:
        self.tier = tier
        self.profile = profile if profile in WEIGHT_PROFILES else DEFAULT_PROFILE
        # Optional EXPLICIT weight override, used only by the offline frontier sweep
        # (MESH-644 tuning). When None (every production/replay caller), the frozen
        # named-profile vector is used, so production behaviour is unchanged. The
        # sweep passes an arbitrary Weights to trace the quality/cost frontier.
        self.weights: Weights = weights if weights is not None else WEIGHT_PROFILES[self.profile]
        self.mean_cost: dict[str, float] = {}
        self._benchmark = BenchmarkPolicy(tier=tier)

    def fit(self, items: list[Item]) -> None:
        self._benchmark.fit(items)
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for it in items:
            for m, c in it.costs.items():
                totals[m] += c
                counts[m] += 1
        self.mean_cost = {m: totals[m] / counts[m] for m in totals if counts[m]}

    def pick(self, item: Item, candidates: list[str], rng: random.Random) -> str | None:
        pool = ranked_routerbench_models(item.task, candidates, tier=self.tier)
        if not pool:
            return self._benchmark.pick(item, candidates, rng)  # abstain → benchmark
        ranked = score_pool_quality_cost(pool, self.mean_cost, self.weights)
        return ranked[0]


# ── Stubs — same interface, not implemented in the offline scope ─────────────────
class _StubPolicy(Policy):
    """A registered-but-unimplemented strategy. Constructing + listing it is fine;
    calling pick() raises so it can never silently produce a bogus number."""

    reason: str = "not implemented"

    def pick(self, item: Item, candidates: list[str], rng: random.Random) -> str | None:
        raise NotImplementedError(f"{self.name} policy is stubbed offline: {self.reason}")


class RegistryStub(_StubPolicy):
    name = "registry"
    pays_classifier_call = True  # prod: LLM classifier over the live model registry
    classifier_model_id = REGISTRY_CLASSIFIER_MODEL
    # Registry presents the FULL live catalog to an LLM classifier that free-selects a
    # model id — there is no portable table, and RouterBench's fixed 11 models are not
    # the catalog the classifier reasons over. Its decision cannot be reproduced offline
    # without a live classifier call, so it stays stubbed here and is measured in Phase 2
    # (router_eval/phase2). The classifier tax IS defined for it (gemini-3-flash-preview)
    # so Phase 2 prices it correctly.
    reason = "needs a live classifier over the live registry (Phase 2)"


class NotDiamondStub(_StubPolicy):
    name = "not_diamond"
    pays_classifier_call = False  # external NotDiamond router call, no internal LLM classifier
    # Needs an external NotDiamond API call per item. Asymmetry vs benchmark/registry:
    # no internal classifier LLM call, so its cost/latency profile differs and must be
    # modelled separately. Out of scope for both phases here.
    reason = "needs external NotDiamond API call (out of scope)"


# ── Registry ────────────────────────────────────────────────────────────────────
def build_policies(benchmark_tier: str = "premium", weight_profile: str = DEFAULT_PROFILE) -> list[Policy]:
    """The policies the replay runs (implemented ones only)."""
    return [
        RandomPolicy(),
        AlwaysCheapestPolicy(),
        AlwaysPremiumPolicy(),
        BenchmarkPolicy(tier=benchmark_tier),
        HeuristicPolicy(tier=benchmark_tier),
        WeightedPolicy(tier=benchmark_tier, profile=weight_profile),
        OraclePolicy(),
    ]


def stub_policies() -> list[Policy]:
    """Registered-but-stubbed strategies, surfaced so the report is honest about
    what is NOT yet measured offline."""
    return [RegistryStub(), NotDiamondStub()]
