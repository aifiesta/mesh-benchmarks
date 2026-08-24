"""
Routing policies for the offline replay.

The adapter interface every policy implements:

    policy.fit(items)                     # optional: precompute global state
    policy.pick(item, candidates, rng)    # -> model_id chosen for this item

`candidates` is the model-id set available for the item (RouterBench's fixed 11).
Realistic policies read only `item.task` + `candidates`; only OraclePolicy reads
`item.scores` / `item.costs` (hindsight — that is the whole point of the oracle).

`pays_classifier_call` marks policies that, IN PRODUCTION, spend an extra per-request
LLM classifier call (cost + latency) that this offline replay does NOT yet charge.
See metrics.py `classifier_cost_usd` for the TODO hook and the asymmetry note.

Implemented: RandomPolicy, AlwaysPremiumPolicy, AlwaysCheapestPolicy, OraclePolicy,
BenchmarkPolicy. Stubbed (Phase-1 out of scope): RegistryStub, NotDiamondStub,
HeuristicStub — each needs routersvc runtime and/or an external call.
"""

from __future__ import annotations

import random
from collections import defaultdict

from router_eval.data import Item
from router_eval.routerbench_bridge import resolve_benchmark_model


class Policy:
    """Base adapter. Subclasses set `name` and implement `pick`."""

    name: str = "policy"
    # Does this policy pay a per-request classifier LLM call in production?
    pays_classifier_call: bool = False

    def fit(self, items: list[Item]) -> None:
        """Optional hook to precompute global state before the replay loop."""

    def pick(self, item: Item, candidates: list[str], rng: random.Random) -> str | None:
        raise NotImplementedError


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
    of an LLM reading the prompt) and WITHOUT charging the classifier call it makes
    in production — both optimistic. See RESULTS.md caveats.
    """

    name = "benchmark"
    pays_classifier_call = True  # prod: per-request LLM classifier call (cost+latency)

    def __init__(self, tier: str = "premium") -> None:
        self.tier = tier

    def pick(self, item: Item, candidates: list[str], rng: random.Random) -> str | None:
        chosen = resolve_benchmark_model(item.task, candidates, tier=self.tier, rng=rng)
        if chosen is not None:
            return chosen
        # Fallback chain mirrors the live default-model behaviour.
        from router_eval.routerbench_bridge import BRAND_TO_ROUTERBENCH_PREMIUM  # noqa: PLC0415

        default = BRAND_TO_ROUTERBENCH_PREMIUM.get("chatgpt")
        if default in candidates:
            return default
        return sorted(candidates)[0] if candidates else None


# ── Stubs — same interface, not implemented in Phase-1 initial scope ────────────
class _StubPolicy(Policy):
    """A registered-but-unimplemented strategy. Constructing + listing it is fine;
    calling pick() raises so it can never silently produce a bogus number."""

    reason: str = "not implemented"

    def pick(self, item: Item, candidates: list[str], rng: random.Random) -> str | None:
        raise NotImplementedError(f"{self.name} policy is stubbed for Phase 1: {self.reason}")


class RegistryStub(_StubPolicy):
    name = "registry"
    pays_classifier_call = True  # prod: LLM classifier over the live model registry
    # TODO(MESH-708 Phase 2): needs routersvc registry (get_enabled_models) + the
    # classifier prompt/parse. No portable table — the decision depends on the live
    # catalog, so it cannot be replayed against RouterBench's fixed 11 models as-is.
    reason = "needs routersvc registry + classifier runtime (Phase 2)"


class NotDiamondStub(_StubPolicy):
    name = "not_diamond"
    pays_classifier_call = False  # external NotDiamond router call, no internal LLM classifier
    # TODO(MESH-708 Phase 2): needs an external NotDiamond API call per item. Note the
    # asymmetry vs benchmark/registry — not_diamond makes NO internal classifier LLM
    # call, so its cost/latency profile differs and must be modelled separately.
    reason = "needs external NotDiamond API call (Phase 2)"


class HeuristicStub(_StubPolicy):
    name = "heuristic"
    pays_classifier_call = False  # rule-based (length/keyword), no LLM classifier
    # TODO(MESH-708 Phase 2): port the routersvc heuristic rules. Rule-based, so no
    # classifier LLM call — but the rules read prompt features the replay would need
    # to reconstruct from RouterBench prompts first.
    reason = "needs routersvc heuristic rules ported (Phase 2)"


# ── Registry ────────────────────────────────────────────────────────────────────
def build_policies(benchmark_tier: str = "premium") -> list[Policy]:
    """The policies the replay runs (implemented ones only)."""
    return [
        RandomPolicy(),
        AlwaysCheapestPolicy(),
        AlwaysPremiumPolicy(),
        BenchmarkPolicy(tier=benchmark_tier),
        OraclePolicy(),
    ]


def stub_policies() -> list[Policy]:
    """Registered-but-stubbed strategies, surfaced so the report is honest about
    what is NOT yet measured."""
    return [RegistryStub(), NotDiamondStub(), HeuristicStub()]
