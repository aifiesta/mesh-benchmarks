"""
Phase-2 CLI — python -m router_eval.phase2

DEFAULT IS DRY-RUN: mock providers, no network, no key, no spend. It runs the whole
pipeline (picks → dedupe → estimate → mock answers → mock judge → aggregate), prints the
live-call estimate for the 91 prompts, and writes CSVs to the gitignored out dir. Its
numbers are mock wiring-checks, NOT results.

LIVE (the operator runs this separately with a real key — the prompts are real PII):

    MESH_API_KEY=sk-... python -m router_eval.phase2 --live

Useful flags:
    --estimate-only      stop after printing the estimate (no answer/judge calls)
    --judge-model ID     judge model (default anthropic/claude-opus-4.8)
    --weight-profile P   weighted profile (balanced|quality_first|cost_first|latency_first)
    --seed N             RNG seed for the random baseline
    --out DIR            CSV output dir (default router_eval/phase2/out, gitignored)
    --cache-dir DIR      answer/judge/classify cache (default router_eval/phase2/.cache)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from router_eval.phase2.judge import DEFAULT_JUDGE_MODEL
from router_eval.phase2.pipeline import (
    DEFAULT_OUT_DIR,
    PipelineConfig,
    build_providers,
    execute,
    plan,
)
from router_eval.phase2.cache import DEFAULT_CACHE_ROOT
from router_eval.phase2.mesh_client import LiveCallBlocked
from router_eval.phase2.report import print_aggregates, print_estimate, write_csvs
from router_eval.phase2.traffic import load_traffic


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="router_eval.phase2", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true",
                   help="make REAL Mesh + judge calls (requires MESH_API_KEY). Default: dry-run.")
    p.add_argument("--estimate-only", action="store_true",
                   help="print the live-call estimate and stop (no inference/judge calls)")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="judge model id")
    p.add_argument("--weight-profile",
                   choices=["quality_first", "balanced", "cost_first", "latency_first"],
                   default="balanced")
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--max-answer-tokens", type=int, default=1024)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--traffic", type=Path, default=None,
                   help="traffic jsonl (default: the gitignored mesh_traffic.jsonl). Point at "
                        "fixtures/traffic_sample.jsonl for a PII-free dry-run smoke.")
    args = p.parse_args(argv)

    cfg = PipelineConfig(
        live=args.live, seed=args.seed, weight_profile=args.weight_profile,
        judge_model=args.judge_model, cache_root=args.cache_dir, out_dir=args.out,
        estimate_only=args.estimate_only, max_answer_tokens=args.max_answer_tokens,
        traffic_path=args.traffic,
    )

    try:
        traffic = load_traffic(args.traffic) if args.traffic else load_traffic()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        providers = build_providers(cfg)
    except LiveCallBlocked as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not cfg.live:
        print("DRY RUN — mock providers, no network. Pass --live + MESH_API_KEY for a real run.")

    # PICKS + ESTIMATE first, so spend is predictable before any answer/judge call.
    plan_ = plan(cfg, providers, traffic)
    print_estimate(plan_.estimate, cfg.live)

    if cfg.estimate_only:
        print("\n--estimate-only: stopping before any inference/judge call.")
        return 0

    result = execute(cfg, providers, traffic, plan_)
    print_aggregates(result)
    write_csvs(cfg.out_dir, result)
    print(f"\nWrote CSVs to {cfg.out_dir} (gitignored).")
    if not cfg.live:
        print("Reminder: dry-run numbers are MOCK. Run --live for real judged results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
