"""
Offline tests for the Phase-2 pipeline — NO network, NO key.

They use the committed SYNTHETIC traffic fixture (fixtures/traffic_sample.jsonl), the
sample catalog, and injected fake/mock providers. The real mesh_traffic.jsonl is gitignored
and never touched here. A dedicated test asserts the network gate refuses to make a real
call outside live mode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from router_eval.phase2.answers import LiveAnswerer, MockAnswerer
from router_eval.phase2.cache import DiskCache, make_key
from router_eval.phase2.catalog import load_sample_catalog
from router_eval.phase2.classifier import MockClassifier
from router_eval.phase2.judge import LiveJudge, MockJudge
from router_eval.phase2.mesh_client import LiveCallBlocked, MeshClient
from router_eval.phase2.pipeline import (
    PipelineConfig,
    Providers,
    build_estimate,
    compute_picks,
    normalize_served,
    plan,
    run_pipeline,
)
from router_eval.phase2.strategies import RouteContext, HeuristicStrategy, build_strategies
from router_eval.phase2.traffic import load_traffic

SAMPLE = Path(__file__).parent.parent / "fixtures" / "traffic_sample.jsonl"


@pytest.fixture(scope="module")
def traffic():
    rows = load_traffic(SAMPLE)
    assert len(rows) == 5
    return rows


@pytest.fixture()
def catalog():
    return load_sample_catalog()


# ── Fakes that record calls (no network) ────────────────────────────────────────
@dataclass
class FakeClient:
    """Stands in for MeshClient: records chat calls, returns canned text. Never networks."""

    chat_calls: list = field(default_factory=list)

    def chat(self, model, prompt, *, system=None, max_tokens=1024, temperature=0.7):
        self.chat_calls.append((model, prompt))
        return f"answer[{model}]", {"prompt_tokens": 5, "completion_tokens": 7}


# ── Traffic loading ──────────────────────────────────────────────────────────────
def test_traffic_fields_parsed(traffic):
    t3 = next(r for r in traffic if r.response_id == "t3")
    assert t3.served_model == "deepseek-v4-flash"
    assert t3.feedback_rating == "rejected" and t3.feedback_is_negative
    assert t3.input_tokens == 18 and t3.output_tokens == 25


def test_normalize_served_maps_short_ids(catalog):
    assert normalize_served("gpt-5.4-mini", catalog) == "openai/gpt-5.4-mini"
    assert normalize_served("gemini-3-flash", catalog) == "google/gemini-3-flash-preview"
    # An unknown short id is left as-is (still reported, just unpriced).
    assert normalize_served("nano-banana-pro", catalog) == "nano-banana-pro"


# ── Network gate ────────────────────────────────────────────────────────────────
def test_mesh_client_refuses_without_live_or_key():
    dry = MeshClient(api_key=None, live=False)
    with pytest.raises(LiveCallBlocked):
        dry.chat("openai/gpt-5.4", "hi")
    with pytest.raises(LiveCallBlocked):
        dry.list_model_ids()
    live_no_key = MeshClient(api_key=None, live=True)
    with pytest.raises(LiveCallBlocked):
        live_no_key.chat("openai/gpt-5.4", "hi")


def test_from_env_requires_key_in_live(monkeypatch):
    monkeypatch.delenv("MESH_API_KEY", raising=False)
    with pytest.raises(LiveCallBlocked):
        MeshClient.from_env(live=True)
    # dry-run build never needs a key
    assert MeshClient.from_env(live=False).live is False


def test_build_providers_live_without_key_blocks(monkeypatch):
    monkeypatch.delenv("MESH_API_KEY", raising=False)
    from router_eval.phase2.pipeline import build_providers

    with pytest.raises(LiveCallBlocked):
        build_providers(PipelineConfig(live=True))


# ── Cache + dedupe ──────────────────────────────────────────────────────────────
def test_disk_cache_roundtrip_and_counts(tmp_path):
    c = DiskCache(root=tmp_path)
    assert c.get("ns", "k") is None and c.misses == 1
    c.put("ns", "k", {"v": 1})
    assert c.get("ns", "k") == {"v": 1} and c.hits == 1
    # get_or_compute only computes on a miss.
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"v": 2}

    assert c.get_or_compute("ns", "k2", compute) == {"v": 2}
    assert c.get_or_compute("ns", "k2", compute) == {"v": 2}
    assert calls["n"] == 1


def test_live_answerer_dedupes_by_prompt_model(tmp_path):
    client = FakeClient()
    ans = LiveAnswerer(client=client, cache=DiskCache(root=tmp_path))
    a1 = ans.answer("hello", "openai/gpt-5.4")
    a2 = ans.answer("hello", "openai/gpt-5.4")  # cached → no new chat call
    ans.answer("hello", "anthropic/claude-sonnet-4.6")  # different model → new call
    assert a1 == a2
    assert len(client.chat_calls) == 2  # one per unique (prompt, model)


def test_seed_prevents_live_call(tmp_path):
    client = FakeClient()
    ans = LiveAnswerer(client=client, cache=DiskCache(root=tmp_path))
    ans.seed("hi", "openai/gpt-5.4", "served answer", 3, 4)
    got = ans.answer("hi", "openai/gpt-5.4")  # served → cached, no client call
    assert got["answer"] == "served answer" and got["seeded"] is True
    assert client.chat_calls == []


# ── Classifier + strategies ─────────────────────────────────────────────────────
def test_mock_classifier_is_deterministic_and_records(catalog):
    c = MockClassifier()
    cat1, mode1 = c.category("fix this python bug")
    cat2, mode2 = c.category("fix this python bug")
    assert (cat1, mode1) == (cat2, mode2)
    assert cat1.startswith("Coding")
    assert len(c.calls) == 2 and all(k.kind == "category" for k in c.calls)


def test_heuristic_strategy_fast_lane_and_miss(traffic, catalog):
    ctx = RouteContext(catalog, MockClassifier(), random.Random(1))
    h = HeuristicStrategy()
    conv = next(r for r in traffic if r.response_id == "t1")  # "hi, how are you today?"
    essay = next(r for r in traffic if r.response_id == "t4")  # long → miss
    # fast-lane hit → conversation standard model, NO classifier call recorded for it
    assert h.classifier_calls(conv.prompt, ctx) == []
    assert h.pick(conv.prompt, ctx) is not None
    # miss → benchmark classifier
    assert h.classifier_calls(essay.prompt, ctx) == ["openai/gpt-4o-mini"]


def test_all_strategies_pick_within_catalog(traffic, catalog):
    ids = set(catalog.ids())
    for strat in build_strategies():
        ctx = RouteContext(catalog, MockClassifier(), random.Random(7))
        for row in traffic:
            pick = strat.pick(row.prompt, ctx)
            assert pick in ids, (strat.name, pick)


# ── Estimate ────────────────────────────────────────────────────────────────────
def test_estimate_dedupes_and_counts(traffic, catalog):
    providers = Providers(catalog, MockClassifier(), MockAnswerer(), MockJudge())
    p = plan(PipelineConfig(), providers, traffic)
    est = p.estimate
    # Derived, not hardcoded: adding a strategy (e.g. benchmark_v7) must not fail this
    # test for a reason that has nothing to do with dedupe/estimate arithmetic.
    n_strategies = len(build_strategies())
    assert est.n_prompts == 5 and est.n_strategies == n_strategies
    # benchmark + weighted (+ heuristic misses) all classify with gpt-4o-mini, deduped by
    # content → at most one per prompt; registry adds gemini per prompt.
    assert est.classifier_calls_by_model["google/gemini-3-flash-preview"] == 5
    assert est.classifier_calls_by_model["openai/gpt-4o-mini"] == 5
    assert est.classifier_calls == 10
    # inference calls are deduped and never exceed strategies × prompts.
    assert 0 < est.live_inference_calls <= n_strategies * 5
    assert est.judge_calls >= est.live_inference_calls
    assert est.total_live_calls == est.live_inference_calls + est.judge_calls + est.classifier_calls


def test_estimate_only_makes_no_answer_or_judge_calls(traffic, catalog):
    answerer = MockAnswerer()
    judge = MockJudge()
    providers = Providers(catalog, MockClassifier(), answerer, judge)
    run_pipeline(PipelineConfig(estimate_only=True), providers, traffic)
    assert answerer.calls == 0 and judge.calls == 0


# ── Full dry-run pipeline ───────────────────────────────────────────────────────
def test_dry_run_pipeline_end_to_end(traffic, catalog, tmp_path):
    providers = Providers(catalog, MockClassifier(), MockAnswerer(), MockJudge())
    result = run_pipeline(PipelineConfig(out_dir=tmp_path), providers, traffic)
    names = {s.name for s in result.strategies}
    assert names == {"random", "always_cheapest", "always_premium",
                     "benchmark", "benchmark_v7", "heuristic", "weighted", "registry"}
    for s in result.strategies:
        assert s.n == 5
        assert 0.0 <= s.mean_judge_score <= 1.0
        assert s.mean_cost_with_tax_usd == pytest.approx(
            s.mean_infer_cost_usd + s.mean_classifier_tax_usd
        )
    # served reference carries the real feedback distribution from the fixture.
    assert result.served.n == 5
    assert result.served.feedback_counts.get("rejected") == 1
    assert result.served.feedback_counts.get("dislike") == 1


def test_pipeline_deterministic(traffic, catalog):
    def run():
        providers = Providers(catalog, MockClassifier(), MockAnswerer(), MockJudge())
        r = run_pipeline(PipelineConfig(), providers, traffic)
        return {s.name: round(s.mean_judge_score, 6) for s in r.strategies}

    assert run() == run()


# ── Judge parsing ───────────────────────────────────────────────────────────────
def test_live_judge_parses_json_and_clamps():
    j = LiveJudge(client=None)
    assert j._parse('{"score": 0.8, "rationale": "good"}') == (0.8, "good")
    assert j._parse('noise {"score": 1.7} tail')[0] == 1.0  # clamped
    assert j._parse("not json")[0] == 0.0
    assert j._parse("")[0] == 0.0


def test_mock_judge_scores_empty_low_and_bounded():
    j = MockJudge()
    assert j.score("q", "", "m")["score"] == 0.0
    s = j.score("q", "a real answer of some length", "m")["score"]
    assert 0.0 <= s <= 1.0


# ── Routing data: v4 vs v7, and the tie-break rule ──────────────────────────────
def test_resolve_picks_randomly_among_tier1_ties_like_the_gateway():
    """The gateway's resolve_from_benchmark_category does `random.choice` over the
    tier-1 tie-group. Taking sorted(...)[0] instead collapses every tie to one member,
    which under-samples the reachable pool AND makes a version that ADDS brands to
    tie-groups look like it changed almost nothing. Any A/B over tie-group membership
    depends on this."""
    from router_eval.phase2.routing_data import V4, resolve_benchmark_model

    cat = "General reasoning / Q&A - General Conversation, Chatting"
    tier1 = V4.benchmarks[cat][0]
    expected = {V4.brand_premium[b] for b in tier1}
    catalog = set(V4.brand_premium.values())

    rng = random.Random(7)
    got = {resolve_benchmark_model(cat, "premium", catalog, V4, rng) for _ in range(300)}
    assert got == expected, "random tie-break must reach every tier-1 brand"

    # rng=None keeps the deterministic first-of-tie behaviour.
    fixed = {resolve_benchmark_model(cat, "premium", catalog, V4) for _ in range(10)}
    assert len(fixed) == 1 and fixed <= expected


def test_v7_picks_the_same_winners_as_v4_but_offers_a_deeper_pool():
    """v7 is a POOL expansion, not a re-ranking: its tier-1 is identical to v4's, so the
    benchmark strategy's WINNER is unchanged, while ranked_models_for_category (the
    fallover alternates and the weighted pool) gets strictly deeper.

    Tier-1 promotion was tried and measured WORSE than v4 on 692 real prompts
    (-0.0889 judged, p<0.0001), which is why the winner must not move."""
    from router_eval.phase2.routing_data import V4, V7, resolve_benchmark_model, ranked_models_for_category

    catalog = (
        set(V4.brand_premium.values()) | set(V4.brand_standard.values())
        | set(V7.brand_premium.values()) | set(V7.brand_standard.values())
    )
    deeper = 0
    for cat in V4.benchmarks:
        for mode in ("premium", "standard"):
            rng4, rng7 = random.Random(11), random.Random(11)
            for _ in range(30):
                assert resolve_benchmark_model(cat, mode, catalog, V4, rng4) == \
                       resolve_benchmark_model(cat, mode, catalog, V7, rng7), f"{cat}/{mode}"
            p4 = ranked_models_for_category(cat, mode, catalog, V4)
            p7 = ranked_models_for_category(cat, mode, catalog, V7)
            assert set(p4) <= set(p7), f"{cat}/{mode} lost a v4 model"
            if len(p7) > len(p4):
                deeper += 1
    assert deeper >= 60, f"only {deeper} (category, mode) pools got deeper"


def test_v7_is_derived_over_v4_and_never_drops_a_v4_brand():
    from router_eval.phase2.routing_data import V4, V7

    for tier in ("brand_premium", "brand_standard"):
        v4_map, v7_map = getattr(V4, tier), getattr(V7, tier)
        for brand, model in v4_map.items():
            assert v7_map[brand] == model, f"{tier}[{brand}] changed"
    # Web-research categories are untouched (the web-capability guarantee).
    for cat in (c for c in V7.benchmarks if c.startswith("Web research / citations")):
        assert V7.benchmarks[cat] == V4.benchmarks[cat], cat
