"""
Phase-2 strategy adapters — pick a model from the LIVE Mesh catalog for a prompt.

Same strategies as Phase 1, re-homed onto the live catalog + a real (or mock) classifier
instead of RouterBench's fixed 11 + ground-truth eval_name:

  random / always_cheapest / always_premium  — runnable baselines (price-based).
  benchmark   — classify → v4 SUPERMODE rank → top brand's tier model in the catalog.
  heuristic   — string-gate → conversation standard model (no classifier), else benchmark.
  weighted    — classify → category pool ∩ catalog, argmax on quality(rank)+cost (reusing
                the Phase-1 scorer). Latency term dropped (no live perf signal here).
  registry    — classifier free-selects a model id from the whole catalog.

ORACLE is intentionally absent: a hindsight oracle would need every catalog model's answer
JUDGED per prompt (cost-prohibitive), so Phase 2 uses the ACTUALLY-SERVED model + its real
feedback as the ground-truth reference instead (handled in the pipeline). Documented in
RESULTS-phase2.md.

Each strategy exposes `classifier_calls(prompt, ctx)` — the classifier model id(s) it pays
for this prompt — which drives both the classifier tax (metrics.py prices it) and the
live classifier-call estimate (deduped by content, as prod caches classify).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from router_eval.heuristic_gate import gate as heuristic_gate
from router_eval.policies import WEIGHT_PROFILES, score_pool_quality_cost
from router_eval.phase2.catalog import Catalog
from router_eval.phase2.classifier import (
    CATEGORY_CLASSIFIER_MODEL,
    MODEL_CLASSIFIER_MODEL,
    ClassifierBackend,
)
from router_eval.phase2.routing_data import (
    V4,
    V7,
    V8,
    V9,
    BRAND_PREMIUM,
    RoutingData,
    conversation_standard_model,
    ranked_models_for_category,
    resolve_benchmark_model,
)


@dataclass
class RouteContext:
    catalog: Catalog
    classifier: ClassifierBackend
    rng: random.Random


class Phase2Strategy:
    name = "strategy"

    def pick(self, prompt: str, ctx: RouteContext) -> str | None:
        raise NotImplementedError

    def classifier_calls(self, prompt: str, ctx: RouteContext) -> list[str]:
        return []


# ── Baselines ────────────────────────────────────────────────────────────────────
class RandomStrategy(Phase2Strategy):
    name = "random"

    def pick(self, prompt: str, ctx: RouteContext) -> str | None:
        ids = sorted(ctx.catalog.ids())
        return ctx.rng.choice(ids) if ids else None


class AlwaysCheapestStrategy(Phase2Strategy):
    name = "always_cheapest"

    def pick(self, prompt: str, ctx: RouteContext) -> str | None:
        priced = ctx.catalog.priced()
        if not priced:
            return None
        return min(priced, key=lambda m: (m.blended_usd_per_1m, m.model_id)).model_id


class AlwaysPremiumStrategy(Phase2Strategy):
    name = "always_premium"

    def pick(self, prompt: str, ctx: RouteContext) -> str | None:
        priced = ctx.catalog.priced()
        if not priced:
            return None
        return max(priced, key=lambda m: (m.blended_usd_per_1m, m.model_id)).model_id


# ── Classifying strategies ─────────────────────────────────────────────────────
class BenchmarkStrategy(Phase2Strategy):
    """classify → rank → top brand's tier model. Parameterised by routing-data version so
    the SAME implementation can be run for v4 (what production serves) and v7 (the
    MESH-232 coverage expansion) — the two differ only in their data, which is exactly
    what the A/B is meant to isolate."""

    name = "benchmark"

    def __init__(self, data: RoutingData = V4, name: str | None = None) -> None:
        self.data = data
        if name is not None:
            self.name = name

    @staticmethod
    def _tie_rng(prompt: str) -> random.Random:
        """A tie-break RNG seeded from the PROMPT, not from the shared run RNG.

        This is what makes v4-vs-v7 a paired comparison. With one shared RNG the two
        arms consume draws in interleaved sequence, so they pick different models on the
        same prompt even in categories where their tier-1 groups are IDENTICAL — the
        measured difference then mixes the data change with pure tie-break noise. Seeding
        per prompt means an unchanged tie-group yields the SAME pick in both arms, and a
        difference can only come from the routing data."""
        return random.Random(hashlib.sha256(prompt.encode("utf-8", "replace")).hexdigest())

    def pick(self, prompt: str, ctx: RouteContext) -> str | None:
        ids = set(ctx.catalog.ids())
        category, mode = ctx.classifier.category(prompt)
        # Mirrors the gateway's random.choice among equally-ranked tier-1 brands, but
        # seeded per prompt so the two data versions stay paired (see _tie_rng).
        chosen = resolve_benchmark_model(category, mode, ids, self.data, self._tie_rng(prompt))
        if chosen is not None:
            return chosen
        default = BRAND_PREMIUM.get("chatgpt")
        if default in ids:
            return default
        return sorted(ids)[0] if ids else None

    def classifier_calls(self, prompt: str, ctx: RouteContext) -> list[str]:
        return [CATEGORY_CLASSIFIER_MODEL]


class HeuristicStrategy(Phase2Strategy):
    name = "heuristic"

    def __init__(self) -> None:
        self._benchmark = BenchmarkStrategy()

    def _fast_lane(self, prompt: str, ctx: RouteContext) -> str | None:
        matched, _reason = heuristic_gate((prompt or "").strip())
        if not matched:
            return None
        return conversation_standard_model(set(ctx.catalog.ids()))

    def pick(self, prompt: str, ctx: RouteContext) -> str | None:
        fast = self._fast_lane(prompt, ctx)
        if fast is not None:
            return fast
        return self._benchmark.pick(prompt, ctx)

    def classifier_calls(self, prompt: str, ctx: RouteContext) -> list[str]:
        # No classifier on a fast-lane hit; benchmark's classifier on a miss.
        return [] if self._fast_lane(prompt, ctx) is not None else [CATEGORY_CLASSIFIER_MODEL]


class WeightedStrategy(Phase2Strategy):
    name = "weighted"

    def __init__(self, profile: str = "balanced") -> None:
        self.profile = profile if profile in WEIGHT_PROFILES else "balanced"
        self._benchmark = BenchmarkStrategy()

    def pick(self, prompt: str, ctx: RouteContext) -> str | None:
        ids = set(ctx.catalog.ids())
        category, mode = ctx.classifier.category(prompt)
        pool = ranked_models_for_category(category, mode, ids)
        if not pool:
            return self._benchmark.pick(prompt, ctx)
        blended = {
            m: ctx.catalog.get(m).blended_usd_per_1m
            for m in pool
            if ctx.catalog.get(m) and ctx.catalog.get(m).blended_usd_per_1m is not None
        }
        ranked = score_pool_quality_cost(pool, blended, WEIGHT_PROFILES[self.profile])
        return ranked[0]

    def classifier_calls(self, prompt: str, ctx: RouteContext) -> list[str]:
        return [CATEGORY_CLASSIFIER_MODEL]


class RegistryStrategy(Phase2Strategy):
    name = "registry"

    def pick(self, prompt: str, ctx: RouteContext) -> str | None:
        return ctx.classifier.select_model(prompt, ctx.catalog.ids())

    def classifier_calls(self, prompt: str, ctx: RouteContext) -> list[str]:
        return [MODEL_CLASSIFIER_MODEL]


def build_strategies(
    weight_profile: str = "balanced", *, real_only: bool = False
) -> list[Phase2Strategy]:
    """`real_only` drops the random/always_* baselines. They are corrupted by the
    catalog's unservable models (documented in RESULTS), and at n=692 they account for
    most unique (prompt, model) pairs — i.e. most of the judge spend — for numbers we
    do not use. The four real strategies + the served reference answer the question."""
    _base = {'random','always_cheapest','always_premium'}
    _all = [
        RandomStrategy(),
        AlwaysCheapestStrategy(),
        AlwaysPremiumStrategy(),
        BenchmarkStrategy(),
        # The MESH-232 candidate: identical strategy, v7 data. Runs alongside `benchmark`
        # so both see the same prompts, the same catalog and the same judge.
        BenchmarkStrategy(data=V7, name="benchmark_v7"),
        BenchmarkStrategy(data=V8, name="benchmark_v8"),
        BenchmarkStrategy(data=V9, name="benchmark_v9"),
        HeuristicStrategy(),
        WeightedStrategy(profile=weight_profile),
        RegistryStrategy(),
    ]
    return [s for s in _all if not (real_only and s.name in _base)]
