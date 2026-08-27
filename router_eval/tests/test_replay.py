"""
Smoke + invariant tests for the offline replay. Pure stdlib + pytest — they run on
the synthetic fixture with no network and no pandas/datasets, so CI stays green on a
clean checkout.
"""

from __future__ import annotations

import random

import pytest

from router_eval.data import _infer_model_ids, load_fixture
from router_eval.metrics import ac3_benchmark_vs_random, evaluate_all, evaluate_policy
from router_eval.policies import (
    AlwaysCheapestPolicy,
    AlwaysPremiumPolicy,
    BenchmarkPolicy,
    OraclePolicy,
    build_policies,
    stub_policies,
)
from router_eval.routerbench_bridge import (
    BRAND_TO_ROUTERBENCH_PREMIUM,
    ROUTERBENCH_MODELS,
    UNMAPPED_BRANDS,
    eval_to_category,
    resolve_benchmark_model,
)

SEED = 20260821


@pytest.fixture(scope="module")
def items():
    data = load_fixture()
    assert data, "fixture must load at least one item"
    return data


def test_fixture_schema(items):
    # All 11 RouterBench models present, and every item carries score+cost per model.
    for it in items:
        assert set(it.models) == set(ROUTERBENCH_MODELS)
        for m in it.models:
            assert 0.0 <= it.scores[m] <= 1.0
            assert it.costs[m] > 0.0


def test_infer_model_ids_ignores_metadata_and_response_cols():
    cols = ["sample_id", "prompt", "eval_name", "oracle_model_to_route_to",
            "gpt-4-1106-preview", "gpt-4-1106-preview|total_cost", "gpt-4-1106-preview|model_response"]
    assert _infer_model_ids(cols) == ["gpt-4-1106-preview"]


def test_replay_runs_end_to_end(items):
    results = evaluate_all(build_policies(), items, seed=SEED)
    names = {r.name for r in results}
    assert names == {
        "random", "always_cheapest", "always_premium",
        "benchmark", "heuristic", "weighted", "oracle",
    }
    for r in results:
        assert r.n == len(items)


def test_oracle_is_the_ceiling(items):
    results = evaluate_all(build_policies(), items, seed=SEED)
    by = {r.name: r for r in results}
    oracle_score = by["oracle"].mean_score
    for name, r in by.items():
        assert r.mean_score <= oracle_score + 1e-9, f"{name} exceeded oracle"
        if name == "oracle":
            assert r.gap_to_oracle == pytest.approx(0.0)
        else:
            assert r.gap_to_oracle == pytest.approx(oracle_score - r.mean_score)


def test_ac3_benchmark_beats_random(items):
    results = evaluate_all(build_policies(), items, seed=SEED)
    ac3 = ac3_benchmark_vs_random(results)
    assert ac3 is not None
    assert ac3.benchmark_beats_random
    assert ac3.delta > 0


def test_deterministic_given_seed(items):
    a = {r.name: r.mean_score for r in evaluate_all(build_policies(), items, seed=SEED)}
    b = {r.name: r.mean_score for r in evaluate_all(build_policies(), items, seed=SEED)}
    assert a == b


def test_premium_is_most_expensive_cheapest_is_least(items):
    prem = AlwaysPremiumPolicy()
    cheap = AlwaysCheapestPolicy()
    prem.fit(items)
    cheap.fit(items)
    assert prem.model == "gpt-4-1106-preview"
    assert cheap.model == "mistralai/mistral-7b-chat"


def test_benchmark_only_picks_mapped_routerbench_models(items):
    mapped = set(BRAND_TO_ROUTERBENCH_PREMIUM.values())
    pol = BenchmarkPolicy()
    rng = random.Random(SEED)
    for it in items:
        pick = pol.pick(it, it.models, rng)
        # Benchmark can only ever land on a brand-mapped model (or the chatgpt fallback).
        assert pick in mapped


def test_oracle_picks_max_score_per_item(items):
    pol = OraclePolicy()
    rng = random.Random(SEED)
    for it in items:
        pick = pol.pick(it, it.models, rng)
        assert it.scores[pick] == max(it.scores.values())


def test_bridge_mapping_and_gaps():
    # Exactly the three brands with a RouterBench representative are mapped.
    assert set(BRAND_TO_ROUTERBENCH_PREMIUM) == {"claude", "chatgpt", "mistral"}
    # The frozen table ranks brands with no RouterBench model — must be surfaced.
    for gap in ("gemini", "grok", "deepseek", "qwen", "moonshot", "perplexity", "bytedance"):
        assert gap in UNMAPPED_BRANDS
    # eval_name families map to real SUPERMODE categories.
    assert eval_to_category("grade-school-math").startswith("Math")
    assert eval_to_category("mmlu-professional-law").startswith("General reasoning")
    assert eval_to_category("mbpp").startswith("Coding")


def test_resolve_benchmark_math_prefers_gpt4_over_unmapped_deepseek():
    # Math ranking is [deepseek, chatgpt, ...]; deepseek is unmapped, so the walk
    # must fall through to chatgpt -> gpt-4 (proves the skip-unmapped logic).
    rng = random.Random(SEED)
    pick = resolve_benchmark_model("grade-school-math", ROUTERBENCH_MODELS, rng=rng)
    assert pick == "gpt-4-1106-preview"


def test_stubs_raise(items):
    rng = random.Random(SEED)
    stubs = stub_policies()
    assert {s.name for s in stubs} == {"registry", "not_diamond"}
    for stub in stubs:
        with pytest.raises(NotImplementedError):
            stub.pick(items[0], items[0].models, rng)
