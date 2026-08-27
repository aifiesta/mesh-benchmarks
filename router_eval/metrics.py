"""
Replay metrics: turn each policy's per-item picks into a quality/cost point,
a gap-to-oracle, and the AC3 "does benchmark beat random?" answer.

Each policy is summarised as a POINT in (mean_cost, mean_score) space:
  * mean_score — mean RouterBench performance score of the picked models ([0,1]).
  * mean_cost  — mean USD cost of the picked responses.
  * gap_to_oracle — oracle mean_score minus this policy's mean_score (quality
    headroom left on the table). 0.0 for the oracle itself.

Classifier tax (AC2)
--------------------
In production the `benchmark`, `registry` and `weighted` strategies pay an EXTRA
per-request LLM classifier call (cost + latency) before they can route; `heuristic`
pays it only when its fast-lane gate DECLINES (it then falls through to benchmark's
classifier). This module now CHARGES that tax instead of zeroing it.

Per classifier call the cost is modelled as

    input_tokens  = template_tokens + min(prompt_tokens, MAX_CLASSIFIER_PROMPT_TOKENS)
    cost_usd      = input_tokens/1e6 * prompt_usd_per_1m
                    + output_tokens/1e6 * completion_usd_per_1m

where the per-model prices come from the Mesh catalog (app/usage/pricing.py +
scripts/model_prices.json), and the token counts are grounded in the real routing
prompts (benchmark: system + the 48-category block ≈ 571 tok + wrapper; registry:
system + the candidate list + wrapper). `prompt_tokens` is estimated from the item's
prompt text (the classifier truncates user content to 2000 chars ≈ 500 tokens, the
cap here). Which classifier a policy pays — and whether it pays at all this request —
comes from `policy.classifier_calls(item, picked_model)`.

These are documented, reviewable constants — swap them for measured classifier
telemetry when available. `mean_cost_with_classifier` is the fair, cross-strategy
cost axis; `mean_cost` alone remains the inference-only cost.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from router_eval.data import Item
from router_eval.policies import Policy

# ── Classifier price/token model ────────────────────────────────────────────────
CHARS_PER_TOKEN = 4  # coarse token estimate for the prompt portion (no tokenizer dep)
# The classifier truncates user_content to 2000 chars before sending it (routersvc
# classifier._build_user_message / benchmark_classifier._build_benchmark_user_message).
MAX_CLASSIFIER_PROMPT_TOKENS = 2000 // CHARS_PER_TOKEN  # = 500


@dataclass(frozen=True)
class ClassifierSpec:
    """A priced classifier model + its fixed prompt overhead.

    prices: USD per 1e6 tokens (prompt/completion), from the Mesh catalog.
    template_tokens: fixed input overhead — system prompt + the category/candidate
      list + wrapper lines (the request prompt is added on top, per item).
    output_tokens: the classifier's short reply (a category/mode/effort JSON or a
      single model id); capped in prod by auto_router_classifier_max_tokens=100.
    latency_ms: representative classifier round-trip (routing overhead, not inference).
    """

    model_id: str
    prompt_usd_per_1m: float
    completion_usd_per_1m: float
    template_tokens: int
    output_tokens: int
    latency_ms: int


# benchmark/weighted classifier — openai/gpt-4o-mini @ $0.15/$0.60 per 1M
# (app/usage/pricing.py). template ≈ system(~150) + 48-category block(~571) + wrapper.
# latency 1300ms: benchmark_classifier.py docstring "p50 ~1.3s measured on prod".
BENCHMARK_CLASSIFIER = ClassifierSpec(
    model_id="openai/gpt-4o-mini",
    prompt_usd_per_1m=0.15,
    completion_usd_per_1m=0.60,
    template_tokens=750,
    output_tokens=16,
    latency_ms=1300,
)
# registry classifier — google/gemini-3-flash-preview @ $0.50/$3.00 per 1M
# (scripts/model_prices.json: prompt_usd_per_1k=0.0005, completion_usd_per_1k=0.003).
# template ≈ system(~59) + 11-candidate list(~239) + wrapper; latency: flash-class est.
REGISTRY_CLASSIFIER = ClassifierSpec(
    model_id="google/gemini-3-flash-preview",
    prompt_usd_per_1m=0.50,
    completion_usd_per_1m=3.00,
    template_tokens=330,
    output_tokens=16,
    latency_ms=900,
)
CLASSIFIER_SPECS: dict[str, ClassifierSpec] = {
    BENCHMARK_CLASSIFIER.model_id: BENCHMARK_CLASSIFIER,
    REGISTRY_CLASSIFIER.model_id: REGISTRY_CLASSIFIER,
}


def estimate_prompt_tokens(text: str) -> int:
    """Coarse prompt-token estimate (chars/CHARS_PER_TOKEN), capped at the classifier
    truncation. No tokenizer dependency — deliberately simple and offline."""
    if not text:
        return 0
    tokens = math.ceil(len(text) / CHARS_PER_TOKEN)
    return min(tokens, MAX_CLASSIFIER_PROMPT_TOKENS)


def classifier_call_cost_usd(spec: ClassifierSpec, prompt_tokens: int) -> float:
    """USD cost of one classifier call: (template + prompt) input priced at the prompt
    rate + a short output priced at the completion rate."""
    input_tokens = spec.template_tokens + min(prompt_tokens, MAX_CLASSIFIER_PROMPT_TOKENS)
    return (
        input_tokens / 1_000_000 * spec.prompt_usd_per_1m
        + spec.output_tokens / 1_000_000 * spec.completion_usd_per_1m
    )


def item_classifier_cost(policy: Policy, item: Item, picked_model: str | None) -> tuple[float, float]:
    """(classifier_cost_usd, classifier_latency_ms) this policy pays FOR THIS ITEM.

    Sums over `policy.classifier_calls(item, picked_model)`. An unknown classifier id
    contributes 0 (surfaced, never crashes) — offline only the two known specs occur."""
    cost = 0.0
    latency = 0.0
    prompt_tokens = estimate_prompt_tokens(item.prompt)
    for model_id in policy.classifier_calls(item, picked_model):
        spec = CLASSIFIER_SPECS.get(model_id)
        if spec is None:
            continue
        cost += classifier_call_cost_usd(spec, prompt_tokens)
        latency += spec.latency_ms
    return cost, latency


@dataclass
class Pick:
    """One policy's decision on one item, with the realised (known) outcome."""

    sample_id: str
    task: str
    model: str
    score: float
    cost: float
    classifier_cost: float = 0.0  # per-item classifier tax this pick incurred


@dataclass
class PolicyResult:
    name: str
    n: int
    mean_score: float
    mean_cost: float
    pays_classifier_call: bool
    classifier_cost_usd: float  # mean per-request classifier tax
    mean_cost_with_classifier: float
    classifier_latency_ms: float = 0.0  # mean per-request classifier round-trip (overhead)
    gap_to_oracle: float | None = None
    picks: list[Pick] = field(default_factory=list)


def evaluate_policy(policy: Policy, items: list[Item], seed: int) -> PolicyResult:
    """Run one policy over all items with a fresh seeded RNG (order-independent)."""
    rng = random.Random(seed)
    policy.fit(items)

    picks: list[Pick] = []
    score_sum = 0.0
    cost_sum = 0.0
    clf_cost_sum = 0.0
    clf_latency_sum = 0.0
    for it in items:
        model = policy.pick(it, it.models, rng)
        if model is None:
            continue
        score = it.scores.get(model, 0.0)
        cost = it.costs.get(model, 0.0)
        clf_cost, clf_latency = item_classifier_cost(policy, it, model)
        score_sum += score
        cost_sum += cost
        clf_cost_sum += clf_cost
        clf_latency_sum += clf_latency
        picks.append(Pick(it.sample_id, it.task, model, score, cost, clf_cost))

    n = len(picks)
    mean_score = score_sum / n if n else 0.0
    mean_cost = cost_sum / n if n else 0.0
    mean_clf_cost = clf_cost_sum / n if n else 0.0
    mean_clf_latency = clf_latency_sum / n if n else 0.0
    return PolicyResult(
        name=policy.name,
        n=n,
        mean_score=mean_score,
        mean_cost=mean_cost,
        pays_classifier_call=policy.pays_classifier_call,
        classifier_cost_usd=mean_clf_cost,
        mean_cost_with_classifier=mean_cost + mean_clf_cost,
        classifier_latency_ms=mean_clf_latency,
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
