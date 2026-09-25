"""
Paired bootstrap over the phase-2 judged run — the confidence interval MESH-941 AC3 asks for.

WHY THIS EXISTS
    `out_*/strategy_aggregate.csv` reports only a MEAN judge score per strategy. A mean
    difference between two arms is not decision-grade on its own: the arms are evaluated on
    the SAME prompts, so the right comparison is paired, and it needs an interval. This
    script recovers the per-prompt judge scores and computes that interval.

HOW THE PER-PROMPT SCORES ARE RECOVERED
    The pipeline never writes per-prompt scores to disk — `picks.csv` has
    (strategy, prompt_index, picked_model) and the aggregate CSV has means. But the run IS
    fully reconstructible from the content-addressed cache, because every cache key is a
    pure function of inputs we still have:

        answer :  sha256("answer" \0 <model_id> \0 <prompt>)                -> {"answer": ...}
        judge  :  sha256("judge"  \0 <judge_model> \0 <model_id> \0 <prompt>
                                  \0 sha256(answer)[:16])                   -> {"raw": ...}

    So for each (prompt_index, picked_model) in picks.csv we re-derive the answer key from
    the traffic file, read the cached answer, hash it exactly as judge.py does, re-derive the
    judge key, read the cached raw judge reply, and parse it with LiveJudge._parse — the same
    parser the run used, including its fail-soft 0.0.

    The recovery is self-verifying: `--verify` re-computes each strategy's mean from the
    recovered per-prompt scores and compares it to the committed strategy_aggregate.csv. If
    the two agree to 1e-6 the recovered vector IS the run's scoring, not a re-scoring of it.

PII
    The cache and the traffic file hold REAL USER PROMPTS AND ANSWERS. They are gitignored
    and must stay local. This script reads them, and prints ONLY aggregate numbers — never a
    prompt, an answer, or a judge rationale. Its output is safe to commit and to paste into
    Jira; its inputs are not.

THE JUDGE MODEL IS PART OF THE KEY
    Pass the judge model the run actually used, not the harness default. The v11 run used
    `anthropic/claude-sonnet-4-6`; the default is `anthropic/claude-opus-4.8`. With the wrong
    one every judge lookup misses — the script says so and refuses to verify, rather than
    silently reading some other judge's scores.

USAGE
    python -m router_eval.phase2.paired_bootstrap \
        --picks router_eval/phase2/out_v11/picks.csv \
        --cache-dir router_eval/phase2/.cache_v11 \
        --aggregate router_eval/phase2/out_v11/strategy_aggregate.csv \
        --judge-model anthropic/claude-sonnet-4-6 \
        --compare weighted_v11:weighted_v10 --compare weighted_v10:weighted

    Results for the v11 run: RESULTS-mesh941-paired.md
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from router_eval.phase2.cache import DiskCache, make_key
from router_eval.phase2.judge import DEFAULT_JUDGE_MODEL, LiveJudge
from router_eval.phase2.traffic import TRAFFIC_PATH, load_traffic

DEFAULT_RESAMPLES = 20000
DEFAULT_SEED = 20260923


def _answer_hash(answer: str) -> str:
    """Identical to judge._answer_hash — kept as a call, not a copy, so it cannot drift."""
    from router_eval.phase2.judge import _answer_hash as _h

    return _h(answer)


# ── Recovery ─────────────────────────────────────────────────────────────────────
@dataclass
class Recovered:
    """Per-strategy per-prompt judge scores, plus what could not be resolved."""

    scores: dict[str, list[float | None]]
    failed: dict[str, list[bool]]  # the picked model returned no answer (dead/4xx/timeout)
    n_prompts: int
    missing_answer: int
    missing_judge: int
    resolved: int

    @property
    def complete(self) -> bool:
        return self.missing_answer == 0 and self.missing_judge == 0


def load_picks(path: Path, n_prompts: int) -> dict[str, list[str]]:
    """picks.csv -> {strategy: [model per prompt_index]}. Empty string = strategy declined."""
    picks: dict[str, list[str]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            strat = row["strategy"]
            idx = int(row["prompt_index"])
            if strat not in picks:
                picks[strat] = [""] * n_prompts
            if idx >= n_prompts:
                raise ValueError(
                    f"picks.csv references prompt_index {idx} but the traffic file has "
                    f"{n_prompts} rows — picks and traffic are from different runs."
                )
            picks[strat][idx] = row["picked_model"]
    return picks


def recover_scores(
    picks: dict[str, list[str]],
    prompts: list[str],
    cache: DiskCache,
    judge_model: str,
) -> Recovered:
    """Re-derive every (prompt, picked_model) judge score from the content-addressed cache."""
    n = len(prompts)
    per_pair: dict[tuple[int, str], float | None] = {}
    per_pair_failed: dict[tuple[int, str], bool] = {}
    missing_answer = 0
    missing_judge = 0

    for per_prompt in picks.values():
        for i, model in enumerate(per_prompt):
            if not model or (i, model) in per_pair:
                continue
            ans = cache.get("answer", make_key("answer", model, prompts[i]))
            if ans is None:
                per_pair[(i, model)] = None
                missing_answer += 1
                continue
            # A model that could not serve the prompt yields an empty answer the judge
            # scores ~0. Counting these separately separates "picked a worse model" from
            # "picked a model that does not work" — different fixes.
            per_pair_failed[(i, model)] = bool(ans.get("failed")) or not (ans.get("answer") or "").strip()
            jkey = make_key(
                "judge", judge_model, model, prompts[i], _answer_hash(ans.get("answer") or "")
            )
            judged = cache.get("judge", jkey)
            if judged is None:
                per_pair[(i, model)] = None
                missing_judge += 1
                continue
            score, _rationale = LiveJudge._parse(judged.get("raw", ""))
            per_pair[(i, model)] = score

    scores = {
        name: [per_pair.get((i, m)) if m else None for i, m in enumerate(per_prompt)]
        for name, per_prompt in picks.items()
    }
    failed = {
        name: [bool(per_pair_failed.get((i, m))) if m else False for i, m in enumerate(per_prompt)]
        for name, per_prompt in picks.items()
    }
    resolved = sum(1 for v in per_pair.values() if v is not None)
    return Recovered(scores, failed, n, missing_answer, missing_judge, resolved)


def verify_against_aggregate(rec: Recovered, aggregate_csv: Path, tol: float = 1e-6) -> list[str]:
    """Compare recovered per-strategy means to the committed aggregate. Returns failures."""
    failures: list[str] = []
    with aggregate_csv.open() as fh:
        for row in csv.DictReader(fh):
            name = row["strategy"]
            if name not in rec.scores:
                continue  # "[served]" has no picks row; not a strategy arm
            vals = [s for s in rec.scores[name] if s is not None]
            published = float(row["mean_judge_score"])
            got = sum(vals) / (len(vals) or 1)
            if abs(got - published) > tol:
                failures.append(f"{name}: recovered {got:.6f} != published {published:.6f}")
    return failures


# ── Paired bootstrap ─────────────────────────────────────────────────────────────
@dataclass
class PairedResult:
    arm_a: str
    arm_b: str
    n_paired: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    resamples: int
    a_wins: int
    b_wins: int
    ties: int
    identical_picks: int
    p_bootstrap: float
    p_sign_test: float
    median_diff_of_differing: float
    failed_a: int
    failed_b: int
    # Conditional on the prompts where the two arms picked DIFFERENT models. This is the
    # scale RESULTS-phase2.md reports for the v7 round ("paired, on the 282 prompts where
    # the arms differed"), so both are printed — they answer different questions. The
    # unconditional mean is the fleet-wide effect (ties correctly dilute it); the
    # conditional mean is the effect size where the change actually bites.
    cond_n: int = 0
    cond_diff: float = 0.0
    cond_ci_low: float = 0.0
    cond_ci_high: float = 0.0
    cond_a_wins: int = 0
    cond_b_wins: int = 0
    cond_ties: int = 0


def _sign_test_p(wins_a: int, wins_b: int) -> float:
    """Exact two-sided binomial sign test on the DISCORDANT prompts only.

    A robustness check that ignores score magnitudes entirely: if arm A is really no
    different from arm B, the prompts where they disagree should split 50/50. It cannot be
    moved by a handful of large deltas, which the mean difference can.
    """
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap(
    rec: Recovered,
    arm_a: str,
    arm_b: str,
    picks: dict[str, list[str]],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    ci: float = 0.95,
) -> PairedResult:
    """Bootstrap mean(a) - mean(b) by resampling PROMPTS (not scores) with replacement.

    Resampling the prompt index keeps each draw paired — both arms are re-weighted by the
    same prompt — which is the whole point: the arms share the 692 prompts, so an unpaired
    interval would be wider than the evidence actually is.
    """
    for arm in (arm_a, arm_b):
        if arm not in rec.scores:
            raise KeyError(f"unknown strategy arm {arm!r}; have: {sorted(rec.scores)}")

    # The prompts both arms have a recovered score for — the pairing, and the index set
    # every statistic below is computed over.
    paired_idx = [
        i
        for i, (a, b) in enumerate(zip(rec.scores[arm_a], rec.scores[arm_b]))
        if a is not None and b is not None
    ]
    pairs = [(rec.scores[arm_a][i], rec.scores[arm_b][i]) for i in paired_idx]
    if not pairs:
        raise ValueError(f"no prompts have a recovered score for both {arm_a} and {arm_b}")

    deltas = [a - b for a, b in pairs]
    n = len(pairs)
    mean_a = sum(a for a, _ in pairs) / n
    mean_b = sum(b for _, b in pairs) / n
    point = mean_a - mean_b

    a_wins = sum(1 for d in deltas if d > 0)
    b_wins = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    identical = sum(
        1
        for i, (pa, pb) in enumerate(zip(picks[arm_a], picks[arm_b]))
        if pa and pa == pb and rec.scores[arm_a][i] is not None
    )
    differing = [d for d in deltas if d != 0]

    failed_a = sum(1 for i in paired_idx if rec.failed[arm_a][i])
    failed_b = sum(1 for i in paired_idx if rec.failed[arm_b][i])

    rng = random.Random(seed)

    def _boot(sample: list[float]) -> tuple[float, float, float]:
        """Percentile CI of the mean of `sample`, plus the two-sided bootstrap p.

        The p is the share of resampled means sitting on, or across, zero — floored at
        2/(resamples+1), which `_fmt_p` prints as an inequality rather than a value.
        """
        if not sample:
            return 0.0, 0.0, 1.0
        k = len(sample)
        draws = sorted(
            sum(sample[rng.randrange(k)] for _ in range(k)) / k for _ in range(resamples)
        )
        centre = sum(sample) / k
        crossed = (
            sum(1 for d in draws if d >= 0) if centre < 0 else sum(1 for d in draws if d <= 0)
        )
        return (
            draws[int((1 - ci) / 2 * resamples)],
            draws[int((1 + ci) / 2 * resamples) - 1],
            min(1.0, 2.0 * (crossed + 1) / (resamples + 1)),
        )

    ci_low, ci_high, p_boot = _boot(deltas)

    # Conditional: only the prompts where the two arms picked DIFFERENT models. The
    # unconditional mean above is the fleet-wide effect, with identical picks correctly
    # diluting it; this is the effect size where the change actually bites, and it is the
    # scale RESULTS-phase2.md quotes for the v7 round.
    cond_idx = [i for i in paired_idx if picks[arm_a][i] != picks[arm_b][i]]
    cond = [rec.scores[arm_a][i] - rec.scores[arm_b][i] for i in cond_idx]
    cond_lo, cond_hi, _cond_p = _boot(cond)

    return PairedResult(
        arm_a=arm_a,
        arm_b=arm_b,
        n_paired=n,
        mean_a=mean_a,
        mean_b=mean_b,
        diff=point,
        ci_low=ci_low,
        ci_high=ci_high,
        resamples=resamples,
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
        identical_picks=identical,
        p_bootstrap=p_boot,
        p_sign_test=_sign_test_p(a_wins, b_wins),
        median_diff_of_differing=statistics.median(differing) if differing else 0.0,
        failed_a=failed_a,
        failed_b=failed_b,
        cond_n=len(cond),
        cond_diff=(sum(cond) / len(cond)) if cond else 0.0,
        cond_ci_low=cond_lo,
        cond_ci_high=cond_hi,
        cond_a_wins=sum(1 for d in cond if d > 0),
        cond_b_wins=sum(1 for d in cond if d < 0),
        cond_ties=sum(1 for d in cond if d == 0),
    )


def _fmt_p(p: float, resamples: int) -> str:
    """The bootstrap p floors at 2/(N+1); print that floor as an inequality, not a value."""
    floor = 2.0 / (resamples + 1)
    return f"<{floor:.5f}" if p <= floor else f"{p:.5f}"


def format_result(r: PairedResult) -> str:
    verdict = (
        "SIGNIFICANT" if (r.ci_low > 0 or r.ci_high < 0) else "NOT significant (CI spans 0)"
    )
    direction = "higher" if r.diff > 0 else "lower"
    return (
        f"{r.arm_a} vs {r.arm_b}\n"
        f"  paired prompts        {r.n_paired}\n"
        f"  mean {r.arm_a:<18} {r.mean_a:.6f}\n"
        f"  mean {r.arm_b:<18} {r.mean_b:.6f}\n"
        f"  paired difference     {r.diff:+.6f}  ({r.arm_a} is {direction})\n"
        f"  95% CI ({r.resamples} resamples)  [{r.ci_low:+.6f}, {r.ci_high:+.6f}]  -> {verdict}\n"
        f"  bootstrap p (2-sided) {_fmt_p(r.p_bootstrap, r.resamples)}\n"
        f"  sign test p (2-sided) {r.p_sign_test:.2e}   (discordant prompts only)\n"
        f"  per-prompt wins       {r.arm_a} {r.a_wins} | {r.arm_b} {r.b_wins} | tie {r.ties}\n"
        f"  identical pick        {r.identical_picks} prompts (same model -> delta 0 by construction)\n"
        f"  median delta over the {r.a_wins + r.b_wins} prompts where the scores differ  "
        f"{r.median_diff_of_differing:+.6f}\n"
        f"  empty/failed answers  {r.arm_a} {r.failed_a} | {r.arm_b} {r.failed_b} "
        f"(of {r.n_paired} prompts)\n"
        f"  --- conditional on the {r.cond_n} prompts where the arms picked DIFFERENT models ---\n"
        f"  conditional difference {r.cond_diff:+.6f}  "
        f"95% CI [{r.cond_ci_low:+.6f}, {r.cond_ci_high:+.6f}]\n"
        f"  conditional wins      {r.arm_a} {r.cond_a_wins} | {r.arm_b} {r.cond_b_wins} "
        f"| tie {r.cond_ties}\n"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="router_eval.phase2.paired_bootstrap",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--picks", type=Path, required=True, help="out_*/picks.csv from the run")
    p.add_argument("--cache-dir", type=Path, required=True, help="the run's .cache_* dir (PII, local)")
    p.add_argument("--traffic", type=Path, default=TRAFFIC_PATH, help="mesh_traffic.jsonl (PII, local)")
    p.add_argument("--aggregate", type=Path, default=None,
                   help="out_*/strategy_aggregate.csv — verify recovery against it")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--compare", action="append", default=[], metavar="A:B",
                   help="paired comparison mean(A)-mean(B); repeatable")
    p.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args(argv)

    traffic = load_traffic(args.traffic)
    prompts = [row.prompt for row in traffic]
    picks = load_picks(args.picks, len(prompts))
    cache = DiskCache(root=args.cache_dir)
    rec = recover_scores(picks, prompts, cache, args.judge_model)

    print(f"prompts {rec.n_prompts} | strategies {len(picks)} | "
          f"resolved (prompt,model) pairs {rec.resolved} | "
          f"missing answer {rec.missing_answer} | missing judgment {rec.missing_judge}")

    if rec.resolved == 0 and rec.missing_judge:
        print(
            f"ERROR: no judgment resolved for judge model {args.judge_model!r}, but every "
            f"answer did. The judge model is part of the cache key — pass the one the run "
            f"used (--judge-model).",
            file=sys.stderr,
        )
        return 1

    if args.aggregate:
        failures = verify_against_aggregate(rec, args.aggregate)
        if failures:
            print("RECOVERY VERIFICATION FAILED — recovered means do not match the run:",
                  file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)
            return 1
        print(f"recovery verified: every strategy mean matches {args.aggregate} to 1e-6")

    if not rec.complete:
        print("WARNING: some (prompt, model) pairs had no cached answer/judgment; "
              "those prompts are dropped from the pairing.", file=sys.stderr)

    for spec in args.compare:
        if ":" not in spec:
            print(f"ERROR: --compare wants A:B, got {spec!r}", file=sys.stderr)
            return 2
        a, b = spec.split(":", 1)
        print()
        print(format_result(paired_bootstrap(rec, a, b, picks, args.resamples, args.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
