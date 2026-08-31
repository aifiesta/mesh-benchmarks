"""
Tests for the MESH-644 weight-vector frontier sweep. Pure stdlib + pytest on the
synthetic fixture — no network, no pandas — so CI stays green on a clean checkout.

Covers the two things the tuning analysis relies on:
  * the production-safe `weights` override on WeightedPolicy (default None keeps the
    frozen named-profile behaviour — production is untouched), and
  * the sweep fast path (`build_sweep_model` + `score_weights`) being byte-identical to
    running the full per-item WeightedPolicy, plus the offline invariant that only
    alpha = w_q/(w_q+w_c) moves the pick (the latency weight is inert with no perf signal).
"""

from __future__ import annotations

import random

import pytest

from router_eval.data import load_fixture
from router_eval.policies import WEIGHT_PROFILES, Weights, WeightedPolicy
from router_eval.weight_sweep import (
    alpha_grid,
    build_sweep_model,
    score_weights,
    simplex_grid,
)

SEED = 20260821


@pytest.fixture(scope="module")
def items():
    data = load_fixture()
    assert data
    return data


def _full_policy_point(items, weights, tier="premium"):
    pol = WeightedPolicy(tier=tier, weights=weights)
    pol.fit(items)
    rng = random.Random(SEED)
    s = c = 0.0
    for it in items:
        m = pol.pick(it, it.models, rng)
        s += it.scores.get(m, 0.0)
        c += it.costs.get(m, 0.0)
    return s / len(items), c / len(items)


def test_weights_override_takes_precedence(items):
    # An explicit vector overrides the named profile; default None uses the profile.
    explicit = WeightedPolicy(profile="quality_first", weights=Weights(0.0, 1.0, 0.0))
    assert explicit.weights == Weights(0.0, 1.0, 0.0)
    default = WeightedPolicy(profile="quality_first")
    assert default.weights == WEIGHT_PROFILES["quality_first"]  # production untouched


def test_fast_path_matches_full_policy(items):
    model = build_sweep_model(items)
    for w in (Weights(0.7, 0.15, 0.15), Weights(0.4, 0.3, 0.3),
              Weights(0.2, 0.65, 0.15), Weights(0.0, 0.5, 0.5), Weights(1.0, 0.0, 0.0)):
        fp = score_weights(model, w)
        fs, fc = _full_policy_point(items, w)
        assert fp.mean_score == pytest.approx(fs, abs=1e-9)
        assert fp.mean_cost == pytest.approx(fc, abs=1e-9)


def test_only_alpha_moves_the_pick_offline(items):
    # Two vectors with the same alpha = w_q/(w_q+w_c) route identically offline; a pure
    # change to the (inert) latency weight does not move quality or cost.
    model = build_sweep_model(items)
    a = score_weights(model, Weights(0.4, 0.4, 0.2))   # alpha 0.5
    b = score_weights(model, Weights(0.3, 0.3, 0.4))   # alpha 0.5, more latency
    assert a.alpha == pytest.approx(b.alpha)
    assert a.pick_signature == b.pick_signature
    assert (a.mean_score, a.mean_cost) == (b.mean_score, b.mean_cost)


def test_simplex_grid_is_on_the_unit_simplex():
    grid = simplex_grid()
    assert len(grid) == 66  # (10+1)(10+2)/2
    for w in grid:
        assert w.q + w.c + w.l == pytest.approx(1.0)


def test_alpha_grid_spans_zero_to_one():
    g = alpha_grid(n=11)
    alphas = [w.q for w in g]  # encoded as (alpha, 1-alpha, 0)
    assert alphas[0] == pytest.approx(0.0)
    assert alphas[-1] == pytest.approx(1.0)
