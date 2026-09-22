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
import json
import random
from pathlib import Path
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
    V10,
    V11,
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
    """score the CATEGORY POOL on quality/cost — parameterised by routing-data version.

    The `data` argument is the whole point of the v10-vs-v11 comparison: v11 is a POOL
    expansion, and this is the only strategy whose behaviour a pool change can move
    (benchmark reads tier-1, which v11 leaves byte-identical). It also fixes a real
    mismatch — this class used to call `ranked_models_for_category` WITHOUT a version,
    silently taking its `data=V4` default, so every weighted number this harness has
    produced scored v4's 17-model pool while production's weighted scored v10's 43. Any
    weighted figure from a run before this change is on v4's pool; do not compare one
    across it."""

    name = "weighted"

    def __init__(
        self, profile: str = "balanced", data: RoutingData = V4, name: str | None = None
    ) -> None:
        self.profile = profile if profile in WEIGHT_PROFILES else "balanced"
        self.data = data
        self._benchmark = BenchmarkStrategy(data=data)
        if name is not None:
            self.name = name

    def pick(self, prompt: str, ctx: RouteContext) -> str | None:
        ids = set(ctx.catalog.ids())
        category, mode = ctx.classifier.category(prompt)
        pool = ranked_models_for_category(category, mode, ids, self.data)
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


class WeightedAAStrategy(WeightedStrategy):
    """weighted, but QUALITY comes from the measured Artificial Analysis score instead of
    a model's RANK POSITION in the hand-written table (MESH-941 part B).

    WHY THIS ARM EXISTS. v11's pool expansion is very nearly INERT on its own, and this
    harness proved it before the judged run was paid for: across the 96 category x mode
    slots, v11 changes weighted's pick in only 8 — and in none of those 8 does a NEWLY
    ADDED model win. The four additions appear in 302 pool slots at a median rank of 15
    of 17, which under `Q = 1 - rank/(N-1)` is a median quality of **0.077**. With the
    balanced profile's 0.40 quality weight, a model scoring 0.077 cannot beat a tier-1
    model scoring 1.0 unless it is overwhelmingly cheaper. So adding gpt-5.6-luna at AA
    37.5 changes nothing while Q remains rank-derived: the pool grew and the scorer still
    cannot see that the new entry is good. The 8 slots that did move are incidental
    re-normalisation, won by incumbents.

    That is the whole argument for part B, and it is why running v10-vs-v11 without this
    arm would have measured a near-null result at full judge cost.

    Q here is the model's `intelligence_index` min-max normalised ACROSS THE POOL, so it
    stays in [0,1] and comparable with the cost term. A model with no AA score keeps the
    rank-derived Q — falling back to 0 would silently delete every unscored model from
    contention, and `qwen/qwen-flash` (39.5% of production's weighted picks) is unscored.

    ⚠ THIS IS NOT THE DESIGN THAT SHIPPED. This arm min-maxes the measured index across
    the pool and mixes it with rank-derived Q for unscored models. mesh-gateway PR #1728
    REJECTED that design as a P1 defect: two scales in one pool force the worst MEASURED
    candidate to exactly 0.0, so an unscored model outranks it even when the indices agree
    perfectly with the curated order. What shipped instead keeps rank-derived Q and
    re-orders only the measured models among their own curated slots. So the 0.507 this
    arm scored in out_v11 is evidence about a REJECTED design; the shipped one has not
    been benchmarked by this harness."""

    name = "weighted_aa"

    _AA: dict[str, dict] | None = None

    @classmethod
    def _aa_scores(cls) -> dict[str, dict]:
        if cls._AA is None:
            path = Path(__file__).parent / "fixtures" / "aa_scores.json"
            cls._AA = json.loads(path.read_text()) if path.exists() else {}
        return cls._AA

    def pick(self, prompt: str, ctx: RouteContext) -> str | None:
        ids = set(ctx.catalog.ids())
        category, mode = ctx.classifier.category(prompt)
        pool = ranked_models_for_category(category, mode, ids, self.data)
        if not pool:
            return self._benchmark.pick(prompt, ctx)
        blended = {
            m: ctx.catalog.get(m).blended_usd_per_1m
            for m in pool
            if ctx.catalog.get(m) and ctx.catalog.get(m).blended_usd_per_1m is not None
        }
        aa = self._aa_scores()
        # Category-aware: coding categories score on coding_index, math on math_index,
        # everything else on the general intelligence index — mirroring what part B would
        # do in the gateway.
        field = (
            "coding" if category.startswith("Coding")
            else "math" if category.startswith("Math")
            else "intelligence"
        )
        scored = {m: aa[m][field] for m in pool if m in aa and field in aa[m]}
        quality: dict[str, float] = {}
        if scored:
            lo, hi = min(scored.values()), max(scored.values())
            span = (hi - lo) or 1.0
            for m, v in scored.items():
                quality[m] = (v - lo) / span
        # Unscored models keep the rank-derived Q so they stay reachable.
        n = len(pool)
        for rank, m in enumerate(pool):
            quality.setdefault(m, 1.0 - rank / (n - 1) if n > 1 else 1.0)
        ranked = score_pool_quality_cost(
            pool, blended, WEIGHT_PROFILES[self.profile], quality=quality
        )
        return ranked[0]


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
        # MESH-941. v11's tier-1 is v10's, so `benchmark_v11` should be identical to
        # `benchmark_v10` — that arm is the CONTROL that proves the expansion is
        # pick-neutral for the benchmark path, exactly as benchmark_v9 checked v9.
        BenchmarkStrategy(data=V10, name="benchmark_v10"),
        BenchmarkStrategy(data=V11, name="benchmark_v11"),
        HeuristicStrategy(),
        WeightedStrategy(profile=weight_profile),
        # THE COMPARISON v11 exists for. weighted is the only strategy a pool change can
        # move, and `weighted_v10` is also the first weighted arm here that scores the
        # pool production actually serves (the bare `weighted` above is v4's, kept for
        # continuity with prior runs).
        WeightedStrategy(profile=weight_profile, data=V10, name="weighted_v10"),
        WeightedStrategy(profile=weight_profile, data=V11, name="weighted_v11"),
        # MESH-941 part B: the same v11 pool, but quality read from the MEASURED AA score
        # instead of rank position. Without this arm the v11 comparison is near-null —
        # see WeightedAAStrategy's docstring for the 8-of-96 measurement that showed it.
        WeightedAAStrategy(profile=weight_profile, data=V11, name="weighted_v11_aa"),
        WeightedAAStrategy(profile=weight_profile, data=V10, name="weighted_v10_aa"),
        RegistryStrategy(),
    ]
    return [s for s in _all if not (real_only and s.name in _base)]
