"""
Offline tests for the Phase-2 Part-A additions: the ported `heuristic` fast lane,
the classifier tax (AC2), and the portable `weighted` Q+C policy. Pure stdlib +
pytest — no network, no key.
"""

from __future__ import annotations

import random

import pytest

from router_eval.data import Item
from router_eval.heuristic_gate import gate
from router_eval.metrics import (
    BENCHMARK_CLASSIFIER,
    REGISTRY_CLASSIFIER,
    MAX_CLASSIFIER_PROMPT_TOKENS,
    classifier_call_cost_usd,
    estimate_prompt_tokens,
    evaluate_all,
    evaluate_policy,
)
from router_eval.policies import (
    BenchmarkPolicy,
    HeuristicPolicy,
    OraclePolicy,
    RandomPolicy,
    WeightedPolicy,
    WEIGHT_PROFILES,
    score_pool_quality_cost,
    build_policies,
)
from router_eval.routerbench_bridge import (
    BRAND_TO_ROUTERBENCH_STANDARD,
    ROUTERBENCH_MODELS,
    ranked_routerbench_models,
)

SEED = 20260821

# Conversation category's rank-0 brand → its standard-tier RouterBench model: the
# model the heuristic fast lane routes an accepted prompt to.
CONV_STANDARD_MODEL = BRAND_TO_ROUTERBENCH_STANDARD["claude"]  # "claude-instant-v1"


def _item(prompt: str, task: str = "grade-school-math", *, models=None) -> Item:
    """A synthetic Item over the given models (default all 11) with trivial score/cost."""
    models = models or ROUTERBENCH_MODELS
    scores = {m: 0.5 for m in models}
    costs = {m: 0.001 for m in models}
    return Item(sample_id="s", task=task, scores=scores, costs=costs, prompt=prompt)


# ── Heuristic gate (verbatim port) ──────────────────────────────────────────────
@pytest.mark.parametrize("text", ["hi there", "how are you?", "what is a comet?", "thanks!"])
def test_gate_accepts_trivial_conversational(text):
    matched, reason = gate(text)
    assert matched, (text, reason)
    assert reason == "trivial_conversational"


@pytest.mark.parametrize(
    "text,reason",
    [
        ("", "empty"),
        ("x" * 141, "too_long"),
        ("hi\nthere?", "multiline"),
        ("what does ```code``` do?", "code_or_link"),
        ("see http://x.com ?", "code_or_link"),
        ("is 1234567 prime?", "digits_dense"),
        ("what is the latest news?", "recency_term"),
        ("what happened in 2026?", "recency_term"),
        ("write me a poem?", "task_verb"),
        ("the sky is blue", "not_conversational"),
    ],
)
def test_gate_declines_with_reason(text, reason):
    matched, got = gate(text)
    assert not matched
    assert got == reason


def test_gate_now_word_boundary_not_substring():
    # "know" must NOT trip the \bnow\b recency guard (that traffic is the target).
    assert gate("do you know it?") == (True, "trivial_conversational")
    assert gate("what should I do now?")[0] is False


# ── Heuristic policy: fast-lane hit vs miss ─────────────────────────────────────
def test_heuristic_fast_lane_hit_routes_conversation_standard_and_pays_no_tax():
    pol = HeuristicPolicy()
    pol.fit([_item("hello?")])
    it = _item("hello, how are you?")
    rng = random.Random(SEED)
    assert pol.pick(it, it.models, rng) == CONV_STANDARD_MODEL
    # Fast-lane hit makes NO classifier call → zero tax.
    assert pol.classifier_calls(it, CONV_STANDARD_MODEL) == []


def test_heuristic_miss_falls_through_to_benchmark_and_pays_that_classifier():
    it = _item("Write a full essay comparing two algorithms in detail.")  # task_verb → decline
    bench = BenchmarkPolicy()
    heur = HeuristicPolicy()
    bench.fit([it])
    heur.fit([it])
    r1, r2 = random.Random(SEED), random.Random(SEED)
    assert heur.pick(it, it.models, r1) == bench.pick(it, it.models, r2)
    assert heur.classifier_calls(it, None) == [BENCHMARK_CLASSIFIER.model_id]


def test_heuristic_gate_hit_but_model_absent_falls_through():
    # Accepted prompt, but the conversation-standard model is NOT a candidate →
    # heuristic must decline (fall through), not crash.
    models = [m for m in ROUTERBENCH_MODELS if m != CONV_STANDARD_MODEL]
    it = _item("hi there?", models=models)
    pol = HeuristicPolicy()
    pol.fit([it])
    pick = pol.pick(it, it.models, random.Random(SEED))
    assert pick in it.models and pick != CONV_STANDARD_MODEL
    # It fell through → benchmark's classifier is paid.
    assert pol.classifier_calls(it, pick) == [BENCHMARK_CLASSIFIER.model_id]


# ── Classifier tax (AC2) ────────────────────────────────────────────────────────
def test_estimate_prompt_tokens_caps_at_truncation():
    assert estimate_prompt_tokens("") == 0
    assert estimate_prompt_tokens("abcd") == 1  # 4 chars / 4
    assert estimate_prompt_tokens("x" * 100_000) == MAX_CLASSIFIER_PROMPT_TOKENS == 500


def test_classifier_call_cost_matches_price_times_tokens():
    # benchmark: template 750 + 10 prompt tokens = 760 input @ $0.15/1M; 16 out @ $0.60/1M.
    prompt_tokens = 10
    expected = 760 / 1e6 * 0.15 + 16 / 1e6 * 0.60
    assert classifier_call_cost_usd(BENCHMARK_CLASSIFIER, prompt_tokens) == pytest.approx(expected)
    # registry uses the gemini price (0.50/3.00), template 330.
    expected_reg = (330 + 10) / 1e6 * 0.50 + 16 / 1e6 * 3.00
    assert classifier_call_cost_usd(REGISTRY_CLASSIFIER, prompt_tokens) == pytest.approx(expected_reg)


def test_registry_classifier_pricier_than_benchmark_for_same_prompt():
    # gemini-3-flash is more $/token than gpt-4o-mini, so the registry tax is higher
    # even though its template is smaller — a real cross-classifier asymmetry.
    assert classifier_call_cost_usd(REGISTRY_CLASSIFIER, 200) > classifier_call_cost_usd(
        BENCHMARK_CLASSIFIER, 200
    )


def test_benchmark_pays_positive_tax_baselines_pay_zero():
    items = [_item("Solve this arithmetic word problem about 12 apples and 3 baskets.")]
    bench = evaluate_policy(BenchmarkPolicy(), items, seed=SEED)
    rand = evaluate_policy(RandomPolicy(), items, seed=SEED)
    orac = evaluate_policy(OraclePolicy(), items, seed=SEED)
    assert bench.classifier_cost_usd > 0
    assert bench.mean_cost_with_classifier == pytest.approx(bench.mean_cost + bench.classifier_cost_usd)
    assert bench.classifier_latency_ms == pytest.approx(BENCHMARK_CLASSIFIER.latency_ms)
    assert rand.classifier_cost_usd == 0.0
    assert orac.classifier_cost_usd == 0.0
    assert rand.classifier_latency_ms == 0.0


def test_heuristic_tax_is_conditional_on_gate():
    # A conversational item pays no tax; a task item pays benchmark's tax. The mean
    # over a mixed set lands strictly between 0 and benchmark's flat tax.
    conv = _item("hi, how are you?")
    task = _item("Write and debug a sorting function for me.")
    heur = evaluate_policy(HeuristicPolicy(), [conv, task], seed=SEED)
    bench = evaluate_policy(BenchmarkPolicy(), [conv, task], seed=SEED)
    assert 0.0 < heur.classifier_cost_usd < bench.classifier_cost_usd


# ── Weighted (portable Q + C, latency dropped) ──────────────────────────────────
def test_score_pool_quality_only_returns_rank_order():
    # With no cost signal, the score is pure quality rank → pool order is preserved.
    pool = ["a", "b", "c"]
    assert score_pool_quality_cost(pool, {}, WEIGHT_PROFILES["balanced"]) == ["a", "b", "c"]


def test_score_pool_cost_term_promotes_cheaper_model():
    # cost_first weights cost heavily: a much cheaper rank-1 model overtakes rank-0.
    pool = ["expensive", "cheap"]
    mean_cost = {"expensive": 1.0, "cheap": 0.0001}
    assert score_pool_quality_cost(pool, mean_cost, WEIGHT_PROFILES["cost_first"])[0] == "cheap"
    # quality_first keeps the rank-0 model on top despite the price gap.
    assert score_pool_quality_cost(pool, mean_cost, WEIGHT_PROFILES["quality_first"])[0] == "expensive"


def test_weighted_picks_within_category_pool():
    it = _item("bug in my code", task="mbpp")  # coding category
    pool = set(ranked_routerbench_models(it.task, it.models, tier="premium"))
    assert pool, "coding category must map to at least one RouterBench model"
    pol = WeightedPolicy()
    pol.fit([it])
    assert pol.pick(it, it.models, random.Random(SEED)) in pool


def test_weighted_pays_benchmark_classifier_tax():
    pol = WeightedPolicy()
    it = _item("solve x^2 = 9", task="grade-school-math")
    assert pol.pays_classifier_call is True
    assert pol.classifier_calls(it, None) == [BENCHMARK_CLASSIFIER.model_id]


def test_weighted_never_beats_oracle_and_below_or_equal_benchmark_quality():
    from router_eval.data import load_fixture

    items = load_fixture()
    results = {r.name: r for r in evaluate_all(build_policies(), items, seed=SEED)}
    assert results["weighted"].mean_score <= results["oracle"].mean_score + 1e-9
    # weighted trades quality for cost, so on the fixture it should not exceed benchmark.
    assert results["weighted"].mean_score <= results["benchmark"].mean_score + 1e-9
    assert results["weighted"].mean_cost <= results["benchmark"].mean_cost + 1e-9


def test_ranked_pool_is_rank_ordered_and_deduped():
    models = ranked_routerbench_models("mbpp", ROUTERBENCH_MODELS, tier="premium")
    assert len(models) == len(set(models))  # deduped
    assert all(m in ROUTERBENCH_MODELS for m in models)
