"""
Offline RouterBench replay runner (MESH-708 Phase 1).

    python -m router_eval.replay                      # default: synthetic fixture
    python -m router_eval.replay --source routerbench  # real dataset (needs deps+net)
    python -m router_eval.replay --source routerbench --shots 0 --limit 5000

Replays each routing policy over a set of precomputed RouterBench outcomes: for
every item the policy names a model, and we look up that model's KNOWN score and
cost. No live inference happens. Prints a quality/cost table, the gap-to-oracle,
and the AC3 answer (does the frozen benchmark table beat random?), and writes raw
+ aggregate CSVs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from router_eval.data import load_fixture, load_routerbench
from router_eval.metrics import PolicyResult, ac3_benchmark_vs_random, evaluate_all
from router_eval.policies import build_policies, stub_policies

DEFAULT_SEED = 20260821
RESULTS_ROOT = Path(__file__).parent / "results"


def _load_items(args: argparse.Namespace):
    if args.source == "fixture":
        return load_fixture()
    return load_routerbench(shots=args.shots, limit=args.limit)


def _source_tag(args: argparse.Namespace) -> str:
    return "fixture" if args.source == "fixture" else f"routerbench_{args.shots}shot"


def _write_csvs(out_dir: Path, results: list[PolicyResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "aggregate.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["policy", "n", "mean_score", "mean_cost_usd", "classifier_cost_usd",
             "mean_cost_with_classifier_usd", "classifier_latency_ms", "gap_to_oracle",
             "pays_classifier_call"]
        )
        for r in results:
            w.writerow([
                r.name, r.n, f"{r.mean_score:.6f}", f"{r.mean_cost:.8f}",
                f"{r.classifier_cost_usd:.8f}", f"{r.mean_cost_with_classifier:.8f}",
                f"{r.classifier_latency_ms:.1f}",
                "" if r.gap_to_oracle is None else f"{r.gap_to_oracle:.6f}",
                r.pays_classifier_call,
            ])

    with (out_dir / "picks.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "sample_id", "task", "model", "score", "cost_usd",
                    "classifier_cost_usd"])
        for r in results:
            for p in r.picks:
                w.writerow([r.name, p.sample_id, p.task, p.model, f"{p.score:.6f}",
                            f"{p.cost:.8f}", f"{p.classifier_cost:.8f}"])


def _print_report(results: list[PolicyResult], tag: str, n_items: int) -> None:
    print(f"\nRouterBench replay — source={tag}, items={n_items}\n")
    header = (
        f"{'policy':<16}{'n':>7}{'mean_score':>12}{'infer_cost$':>13}"
        f"{'clf_tax$':>11}{'cost+tax$':>12}{'clf_ms':>8}{'gap_oracle':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        gap = "  (ceiling)" if r.name == "oracle" else f"{r.gap_to_oracle:>12.4f}"
        print(
            f"{r.name:<16}{r.n:>7}{r.mean_score:>12.4f}{r.mean_cost:>13.6f}"
            f"{r.classifier_cost_usd:>11.6f}{r.mean_cost_with_classifier:>12.6f}"
            f"{r.classifier_latency_ms:>8.0f}" + gap
        )
    print("\nclf_tax$ = mean per-request classifier LLM call this strategy pays in prod")
    print("  (AC2: benchmark/weighted → gpt-4o-mini; registry → gemini-3-flash-preview;")
    print("  heuristic → benchmark's classifier ONLY on a fast-lane miss). cost+tax$ is the")
    print("  fair cross-strategy cost axis; clf_ms is routing overhead, separate from inference.")


def _print_ac3(results: list[PolicyResult]) -> None:
    ac3 = ac3_benchmark_vs_random(results)
    if ac3 is None:
        return
    verdict = "YES" if ac3.benchmark_beats_random else "NO"
    rel = "n/a" if ac3.relative_uplift is None else f"{ac3.relative_uplift * 100:+.1f}%"
    print("\n=== AC3: does frozen SUPERMODE_BENCHMARKS beat RANDOM? ===")
    print(f"  benchmark mean_score = {ac3.benchmark_mean_score:.4f}")
    print(f"  random    mean_score = {ac3.random_mean_score:.4f}")
    print(f"  delta = {ac3.delta:+.4f}  (relative uplift {rel})")
    print(f"  ANSWER: {verdict} — benchmark {'>' if ac3.benchmark_beats_random else '<='} random")


def _print_stubs() -> None:
    stubs = stub_policies()
    print("\nStubbed strategies (not measurable offline, same interface):")
    for s in stubs:
        clf = "pays classifier call" if s.pays_classifier_call else "no internal classifier call"
        print(f"  - {s.name:<12} {s.reason}  [{clf}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="router_eval.replay", description=__doc__)
    parser.add_argument("--source", choices=["fixture", "routerbench"], default="fixture",
                        help="fixture (default, offline) or the real withmartian/routerbench")
    parser.add_argument("--shots", type=int, choices=[0, 5], default=0,
                        help="RouterBench 0-shot or 5-shot (real source only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap items (real source only) for a quick run")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG seed for random baseline + benchmark tie-breaks")
    parser.add_argument("--benchmark-tier", choices=["premium", "standard"], default="premium",
                        help="which RouterBench tier the benchmark/heuristic/weighted brand map resolves to")
    parser.add_argument("--weight-profile",
                        choices=["quality_first", "balanced", "cost_first", "latency_first"],
                        default="balanced", help="weighted strategy's quality/cost weight profile")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir for CSVs (default: router_eval/results/<tag>/)")
    args = parser.parse_args(argv)

    items = _load_items(args)
    if not items:
        print("No items loaded — aborting.", file=sys.stderr)
        return 1

    policies = build_policies(benchmark_tier=args.benchmark_tier, weight_profile=args.weight_profile)
    results = evaluate_all(policies, items, seed=args.seed)

    tag = _source_tag(args)
    _print_report(results, tag, len(items))
    _print_ac3(results)
    _print_stubs()

    out_dir = args.out or (RESULTS_ROOT / tag)
    _write_csvs(out_dir, results)
    print(f"\nWrote {out_dir / 'aggregate.csv'} and {out_dir / 'picks.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
