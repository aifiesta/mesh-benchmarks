"""
Offline tests for the paired bootstrap — NO network, NO key, NO real prompts.

They build a throwaway cache from the committed SYNTHETIC traffic fixture, write it the
same way the live run would, and assert the recovery reads back exactly what was written.
That is the property the whole analysis rests on: the per-prompt scores are recovered from
the content-addressed cache, not re-derived by a second scoring pass.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from router_eval.phase2.cache import DiskCache, make_key
from router_eval.phase2.judge import _answer_hash
from router_eval.phase2.paired_bootstrap import (
    Recovered,
    _sign_test_p,
    load_picks,
    paired_bootstrap,
    recover_scores,
    verify_against_aggregate,
)
from router_eval.phase2.traffic import load_traffic

SAMPLE = Path(__file__).parent.parent / "fixtures" / "traffic_sample.jsonl"
JUDGE = "anthropic/claude-sonnet-4-6"


@pytest.fixture()
def prompts():
    return [row.prompt for row in load_traffic(SAMPLE)]


def _seed(cache: DiskCache, prompt: str, model: str, answer: str, score: float) -> None:
    """Write an (answer, judgment) pair exactly as LiveAnswerer/LiveJudge would."""
    cache.put("answer", make_key("answer", model, prompt), {
        "answer": answer, "prompt_tokens": 10, "completion_tokens": 20, "model": model,
    })
    cache.put("judge", make_key("judge", JUDGE, model, prompt, _answer_hash(answer)), {
        "raw": '{"score": %s, "rationale": "synthetic"}' % score,
    })


@pytest.fixture()
def recovered(tmp_path, prompts):
    """Two arms over the 5 fixture prompts: arm_a scores 0.8, arm_b 0.4 — except prompt 0
    where both pick the same model (so the pair is a tie by construction)."""
    cache = DiskCache(root=tmp_path / "cache")
    picks_rows = []
    for i, prompt in enumerate(prompts):
        a_model = "vendor/shared" if i == 0 else "vendor/model-a"
        b_model = "vendor/shared" if i == 0 else "vendor/model-b"
        _seed(cache, prompt, a_model, f"answer a {i}", 0.8 if i else 0.6)
        if b_model != a_model:
            _seed(cache, prompt, b_model, f"answer b {i}", 0.4)
        picks_rows.append(("arm_a", i, a_model))
        picks_rows.append(("arm_b", i, b_model))

    picks_csv = tmp_path / "picks.csv"
    with picks_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "prompt_index", "picked_model"])
        w.writerows(picks_rows)

    picks = load_picks(picks_csv, len(prompts))
    return picks, recover_scores(picks, prompts, cache, JUDGE), tmp_path


def test_recovery_reads_back_exactly_what_the_cache_holds(recovered):
    _picks, rec, _tmp = recovered
    assert rec.missing_answer == 0 and rec.missing_judge == 0
    assert rec.scores["arm_a"] == [0.6, 0.8, 0.8, 0.8, 0.8]
    assert rec.scores["arm_b"] == [0.6, 0.4, 0.4, 0.4, 0.4]
    # prompt 0 is the shared pick, so both arms read the SAME cached judgment.
    assert rec.scores["arm_a"][0] == rec.scores["arm_b"][0]


def test_a_wrong_judge_model_resolves_nothing_rather_than_guessing(recovered, prompts, tmp_path):
    """The judge model is part of the cache key. Asking for the wrong one must miss loudly,
    not silently fall back to some other model's scores."""
    picks, _rec, cache_parent = recovered
    cache = DiskCache(root=cache_parent / "cache")
    rec = recover_scores(picks, prompts, cache, "anthropic/claude-opus-4.8")
    assert rec.missing_judge > 0
    assert rec.resolved == 0


def test_verify_against_aggregate_catches_a_mismatch(recovered, tmp_path):
    picks, rec, _ = recovered
    agg = tmp_path / "agg.csv"
    header = "strategy,n,mean_judge_score\n"
    agg.write_text(header + "arm_a,5,0.760000\narm_b,5,0.440000\n")
    assert verify_against_aggregate(rec, agg) == []

    agg.write_text(header + "arm_a,5,0.999999\narm_b,5,0.440000\n")
    failures = verify_against_aggregate(rec, agg)
    assert len(failures) == 1 and "arm_a" in failures[0]


def test_paired_bootstrap_separates_two_clearly_different_arms(recovered):
    picks, rec, _ = recovered
    r = paired_bootstrap(rec, "arm_a", "arm_b", picks, resamples=2000, seed=7)
    assert r.n_paired == 5
    assert r.diff == pytest.approx(0.32)  # 4 prompts at +0.4, one tie, over 5
    assert r.ci_low > 0  # the whole interval is above zero -> significant
    assert (r.a_wins, r.b_wins, r.ties) == (4, 0, 1)
    assert r.identical_picks == 1


def test_paired_bootstrap_reports_no_difference_when_there_is_none(recovered):
    picks, rec, _ = recovered
    same = Recovered(
        scores={"x": list(rec.scores["arm_a"]), "y": list(rec.scores["arm_a"])},
        failed={"x": [False] * 5, "y": [False] * 5},
        n_prompts=5, missing_answer=0, missing_judge=0, resolved=5,
    )
    r = paired_bootstrap(same, "x", "y", {"x": ["m"] * 5, "y": ["m"] * 5}, resamples=2000, seed=7)
    assert r.diff == 0.0
    assert r.ci_low <= 0 <= r.ci_high  # CI must span zero
    assert r.p_sign_test == 1.0


def test_paired_bootstrap_rejects_an_unknown_arm(recovered):
    picks, rec, _ = recovered
    with pytest.raises(KeyError):
        paired_bootstrap(rec, "arm_a", "does_not_exist", picks, resamples=10)


@pytest.mark.parametrize(
    "wins_a,wins_b,expected",
    [(0, 0, 1.0), (5, 5, 1.0), (10, 0, 2 * (1 / 1024)), (1, 0, 1.0)],
)
def test_sign_test_p_matches_the_exact_binomial(wins_a, wins_b, expected):
    assert _sign_test_p(wins_a, wins_b) == pytest.approx(expected)
