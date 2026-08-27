"""
Printing + CSV output for the Phase-2 pipeline.

CSVs go under the (gitignored) out dir — in a live run they contain judged scores keyed
by prompt index, so they stay local like the rest of the PII. The console prints the
call-count estimate (always) and, after a run, the per-strategy quality/cost table.
"""

from __future__ import annotations

import csv
from pathlib import Path

from router_eval.phase2.pipeline import Estimate, PipelineResult


def print_estimate(est: Estimate, live: bool) -> None:
    mode = "LIVE" if live else "DRY-RUN (mock providers; estimate only — no real calls)"
    print(f"\n=== Phase 2 live-call ESTIMATE — {mode} ===")
    print(f"  prompts: {est.n_prompts}   strategies: {est.n_strategies}")
    print(f"  unique (prompt, picked_model) pairs across strategies : {est.unique_inference_pairs}")
    print(f"  of which already served (pre-seeded, no call)         : {est.seeded_served_pairs}")
    print(f"  -> INFERENCE calls to make (deduped, minus served)    : {est.live_inference_calls}")
    print(f"  -> JUDGE calls (one per unique answer incl. served)   : {est.judge_calls}")
    print(f"  -> CLASSIFIER calls (deduped by content)              : {est.classifier_calls}")
    for model, count in sorted(est.classifier_calls_by_model.items()):
        print(f"        {model:<34} {count}")
    print(f"  =========================================================")
    print(f"  TOTAL live calls (inference + judge + classifier)     : {est.total_live_calls}")
    print("  distinct models picked per strategy:")
    for name, k in est.picks_per_strategy.items():
        print(f"        {name:<16} {k}")


def print_aggregates(result: PipelineResult) -> None:
    if not result.strategies:
        return
    tag = "LIVE" if result.live else "MOCK (dry-run — numbers are NOT real; wiring check only)"
    print(f"\n=== Phase 2 per-strategy quality vs cost — {tag} ===\n")
    header = (
        f"{'strategy':<16}{'n':>4}{'judge_q':>9}{'infer$':>11}{'clf_tax$':>11}"
        f"{'cost+tax$':>12}{'#models':>9}"
    )
    print(header)
    print("-" * len(header))
    for s in result.strategies:
        print(
            f"{s.name:<16}{s.n:>4}{s.mean_judge_score:>9.3f}{s.mean_infer_cost_usd:>11.6f}"
            f"{s.mean_classifier_tax_usd:>11.6f}{s.mean_cost_with_tax_usd:>12.6f}"
            f"{s.distinct_models_picked:>9}"
        )
    sv = result.served
    print(
        f"{'[served]':<16}{sv.n:>4}{sv.mean_judge_score:>9.3f}{sv.mean_infer_cost_usd:>11.6f}"
        f"{0.0:>11.6f}{sv.mean_infer_cost_usd:>12.6f}{'—':>9}"
    )
    print(f"\n  served feedback: {dict(sv.feedback_counts)}")


def write_csvs(out_dir: Path, result: PipelineResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "estimate.csv").open("w", newline="") as f:
        w = csv.writer(f)
        est = result.estimate
        w.writerow(["metric", "value"])
        for k, v in [
            ("n_prompts", est.n_prompts), ("n_strategies", est.n_strategies),
            ("unique_inference_pairs", est.unique_inference_pairs),
            ("seeded_served_pairs", est.seeded_served_pairs),
            ("live_inference_calls", est.live_inference_calls),
            ("judge_calls", est.judge_calls), ("classifier_calls", est.classifier_calls),
            ("total_live_calls", est.total_live_calls),
        ]:
            w.writerow([k, v])

    with (out_dir / "strategy_aggregate.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "n", "mean_judge_score", "mean_infer_cost_usd",
                    "mean_classifier_tax_usd", "mean_cost_with_tax_usd", "distinct_models_picked"])
        for s in result.strategies:
            w.writerow([s.name, s.n, f"{s.mean_judge_score:.6f}", f"{s.mean_infer_cost_usd:.8f}",
                        f"{s.mean_classifier_tax_usd:.8f}", f"{s.mean_cost_with_tax_usd:.8f}",
                        s.distinct_models_picked])
        sv = result.served
        w.writerow(["[served]", sv.n, f"{sv.mean_judge_score:.6f}", f"{sv.mean_infer_cost_usd:.8f}",
                    "0.0", f"{sv.mean_infer_cost_usd:.8f}", ""])

    # picks.csv keys by prompt INDEX only (no prompt text) — still gitignored out/.
    with (out_dir / "picks.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "prompt_index", "picked_model"])
        for strat, per_prompt in result.picks.items():
            for i, model in enumerate(per_prompt):
                w.writerow([strat, i, model])
