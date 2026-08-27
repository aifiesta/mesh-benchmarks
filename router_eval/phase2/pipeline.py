"""
Phase-2 pipeline orchestration.

Flow (identical in dry-run and live; only the providers differ):
  1. PICKS   — every strategy picks a catalog model for each of the 91 prompts. Classifying
               strategies invoke the classifier here (real in live, mock in dry-run).
  2. ESTIMATE— dedupe (prompt, picked_model) across strategies, subtract the already-served
               pairs (seeded from response_raw), and count the live inference + judge +
               classifier calls the run will make. Printed BEFORE any answer/judge call so
               spend is predictable.
  3. ANSWERS — one call per UNIQUE (prompt, model), disk-cached (served answers pre-seeded).
  4. JUDGE   — one judge call per unique answer, disk-cached.
  5. AGGREGATE — per strategy: mean judged quality, mean inference cost, mean classifier tax,
               all-in cost — plus the actually-served model's judged quality + feedback as
               the ground-truth reference. Written to CSVs.

Providers are injected so tests drive the whole thing with mocks. `build_providers` wires
the live-or-mock set from the config. NOTHING here calls the network unless cfg.live and a
MeshClient with a key were built.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from router_eval.metrics import CLASSIFIER_SPECS, classifier_call_cost_usd, estimate_prompt_tokens
from router_eval.phase2.answers import Answerer, LiveAnswerer, MockAnswerer
from router_eval.phase2.catalog import Catalog, fetch_live_catalog, load_sample_catalog
from router_eval.phase2.classifier import LiveClassifier, MockClassifier
from router_eval.phase2.judge import DEFAULT_JUDGE_MODEL, Judge, LiveJudge, MockJudge
from router_eval.phase2.cache import DiskCache, DEFAULT_CACHE_ROOT
from router_eval.phase2.mesh_client import MeshClient
from router_eval.phase2.strategies import Phase2Strategy, RouteContext, build_strategies
from router_eval.phase2.traffic import TrafficRow, load_traffic

# Short served-model ids (as logged) → catalog ids. Best-effort; unmapped served ids keep
# their short form (cost/brand then unknown, still reported as the reference).
SERVED_ALIASES: dict[str, str] = {
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "grok-4-1-fast": "xai/grok-4.1-fast-non-reasoning",
    "mistral-medium-latest": "mistralai/mistral-medium-3.1",
    "deepseek-v4-flash": "deepseek/deepseek-v3.2",
    # "nano-banana-pro" is an image model — intentionally left unmapped.
}

DEFAULT_OUT_DIR = Path(__file__).parent / "out"


@dataclass
class PipelineConfig:
    live: bool = False
    seed: int = 20260821
    weight_profile: str = "balanced"
    judge_model: str = DEFAULT_JUDGE_MODEL
    cache_root: Path = DEFAULT_CACHE_ROOT
    out_dir: Path = DEFAULT_OUT_DIR
    estimate_only: bool = False
    max_answer_tokens: int = 1024
    traffic_path: Path | None = None  # default: the gitignored real mesh_traffic.jsonl


def normalize_served(short_id: str, catalog: Catalog) -> str:
    if catalog.get(short_id) is not None:
        return short_id
    return SERVED_ALIASES.get(short_id, short_id)


# ── Result shapes ────────────────────────────────────────────────────────────────
@dataclass
class Estimate:
    n_prompts: int
    n_strategies: int
    unique_inference_pairs: int
    seeded_served_pairs: int
    live_inference_calls: int
    judge_calls: int
    classifier_calls: int
    total_live_calls: int
    classifier_calls_by_model: dict[str, int] = field(default_factory=dict)
    picks_per_strategy: dict[str, int] = field(default_factory=dict)


@dataclass
class StrategyAggregate:
    name: str
    n: int
    mean_judge_score: float
    mean_infer_cost_usd: float
    mean_classifier_tax_usd: float
    mean_cost_with_tax_usd: float
    distinct_models_picked: int


@dataclass
class ServedReference:
    n: int
    mean_judge_score: float
    mean_infer_cost_usd: float
    feedback_counts: dict[str, int]


@dataclass
class PipelineResult:
    live: bool
    estimate: Estimate
    strategies: list[StrategyAggregate]
    served: ServedReference
    picks: dict[str, list[str]]  # strategy -> per-prompt picked model id


# ── Providers ────────────────────────────────────────────────────────────────────
@dataclass
class Providers:
    catalog: Catalog
    classifier: object
    answerer: Answerer
    judge: Judge


def build_providers(cfg: PipelineConfig) -> Providers:
    """Wire the live-or-mock provider set. Live requires MESH_API_KEY (MeshClient enforces)."""
    cache = DiskCache(root=cfg.cache_root)
    if cfg.live:
        client = MeshClient.from_env(live=True)
        return Providers(
            catalog=fetch_live_catalog(client),
            classifier=LiveClassifier(client=client, cache=cache),
            answerer=LiveAnswerer(client=client, cache=cache, max_tokens=cfg.max_answer_tokens),
            judge=LiveJudge(client=client, judge_model=cfg.judge_model, cache=cache),
        )
    return Providers(
        catalog=load_sample_catalog(),
        classifier=MockClassifier(),
        answerer=MockAnswerer(),
        judge=MockJudge(judge_model=cfg.judge_model),
    )


# ── Cost helpers ─────────────────────────────────────────────────────────────────
def _infer_cost_usd(catalog: Catalog, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    m = catalog.get(model_id)
    if m is None or m.prompt_usd_per_1m is None or m.completion_usd_per_1m is None:
        return 0.0
    return prompt_tokens / 1e6 * m.prompt_usd_per_1m + completion_tokens / 1e6 * m.completion_usd_per_1m


def _classifier_tax_usd(classifier_model: str, prompt: str) -> float:
    spec = CLASSIFIER_SPECS.get(classifier_model)
    if spec is None:
        return 0.0
    return classifier_call_cost_usd(spec, estimate_prompt_tokens(prompt))


# ── Orchestration ────────────────────────────────────────────────────────────────
def compute_picks(
    strategies: list[Phase2Strategy], traffic: list[TrafficRow], providers: Providers, seed: int
) -> tuple[dict[str, list[str | None]], dict[str, list[list[str]]]]:
    """Run every strategy over every prompt. Returns (picks, classifier_calls) keyed by
    strategy name, each a per-prompt list. Classifying strategies hit the classifier here."""
    picks: dict[str, list[str | None]] = {}
    clf_calls: dict[str, list[list[str]]] = {}
    for strat in strategies:
        rng = random.Random(f"{seed}:{strat.name}")
        ctx = RouteContext(catalog=providers.catalog, classifier=providers.classifier, rng=rng)
        picks[strat.name] = [strat.pick(row.prompt, ctx) for row in traffic]
        clf_calls[strat.name] = [strat.classifier_calls(row.prompt, ctx) for row in traffic]
    return picks, clf_calls


def build_estimate(
    traffic: list[TrafficRow],
    picks: dict[str, list[str | None]],
    clf_calls: dict[str, list[list[str]]],
    catalog: Catalog,
) -> Estimate:
    """Dedupe picks + classifier calls and count the live calls the run will make."""
    inference_pairs: set[tuple[int, str]] = set()
    for per_prompt in picks.values():
        for i, model in enumerate(per_prompt):
            if model:
                inference_pairs.add((i, model))

    served_pairs = {(i, normalize_served(row.served_model, catalog)) for i, row in enumerate(traffic)}
    # Answers we must actually generate = unique picks not already served.
    to_generate = inference_pairs - served_pairs
    # Every unique answer (picks ∪ served) is judged once.
    judged_pairs = inference_pairs | served_pairs

    # Classifier calls are content-cached in prod → dedupe by (classifier_model, prompt).
    classifier_pairs: set[tuple[str, int]] = set()
    for per_prompt in clf_calls.values():
        for i, models in enumerate(per_prompt):
            for cm in models:
                classifier_pairs.add((cm, i))
    clf_by_model: dict[str, int] = {}
    for cm, _i in classifier_pairs:
        clf_by_model[cm] = clf_by_model.get(cm, 0) + 1

    picks_per_strategy = {
        name: len({m for m in per if m}) for name, per in picks.items()
    }
    return Estimate(
        n_prompts=len(traffic),
        n_strategies=len(picks),
        unique_inference_pairs=len(inference_pairs),
        seeded_served_pairs=len(served_pairs & inference_pairs),
        live_inference_calls=len(to_generate),
        judge_calls=len(judged_pairs),
        classifier_calls=len(classifier_pairs),
        total_live_calls=len(to_generate) + len(judged_pairs) + len(classifier_pairs),
        classifier_calls_by_model=clf_by_model,
        picks_per_strategy=picks_per_strategy,
    )


def _seed_served_answers(traffic: list[TrafficRow], catalog: Catalog, answerer: Answerer) -> None:
    """Pre-load the already-served answers so a strategy that picks the served model pays
    no live call (only LiveAnswerer persists a cache; MockAnswerer ignores)."""
    seed = getattr(answerer, "seed", None)
    if not callable(seed):
        return
    for row in traffic:
        model = normalize_served(row.served_model, catalog)
        seed(row.prompt, model, row.served_answer, row.input_tokens, row.output_tokens)


@dataclass
class Plan:
    """The result of the cheap PICKS + ESTIMATE phase — computed and printed before any
    answer/judge spend. `estimate_only` runs stop here."""

    strategies: list[Phase2Strategy]
    picks: dict[str, list[str | None]]
    clf_calls: dict[str, list[list[str]]]
    estimate: Estimate


def plan(cfg: PipelineConfig, providers: Providers, traffic: list[TrafficRow]) -> Plan:
    """PICKS + ESTIMATE only. Classifying strategies hit the classifier (real in live),
    but NO inference or judge calls happen — so the estimate can be printed before spend."""
    strategies = build_strategies(weight_profile=cfg.weight_profile)
    picks, clf_calls = compute_picks(strategies, traffic, providers, cfg.seed)
    estimate = build_estimate(traffic, picks, clf_calls, providers.catalog)
    return Plan(strategies, picks, clf_calls, estimate)


def execute(cfg: PipelineConfig, providers: Providers, traffic: list[TrafficRow], plan_: Plan) -> PipelineResult:
    """ANSWERS + JUDGE + AGGREGATE. Runs the (cached) inference + judge calls."""
    strategies, picks, clf_calls, estimate = (
        plan_.strategies, plan_.picks, plan_.clf_calls, plan_.estimate
    )

    # ANSWERS: seed served, then generate/cached-fetch every unique (prompt, model).
    _seed_served_answers(traffic, providers.catalog, providers.answerer)
    answers: dict[tuple[int, str], dict] = {}
    for per_prompt in picks.values():
        for i, model in enumerate(per_prompt):
            if model and (i, model) not in answers:
                answers[(i, model)] = providers.answerer.answer(traffic[i].prompt, model)
    # served answers (from response_raw; judged too)
    served_norm = [normalize_served(r.served_model, providers.catalog) for r in traffic]
    for i, row in enumerate(traffic):
        key = (i, served_norm[i])
        if key not in answers:
            answers[key] = {
                "answer": row.served_answer, "prompt_tokens": row.input_tokens,
                "completion_tokens": row.output_tokens, "model": served_norm[i],
            }

    # JUDGE: one score per unique answer.
    judgments: dict[tuple[int, str], float] = {}
    for (i, model), ans in answers.items():
        judgments[(i, model)] = float(
            providers.judge.score(traffic[i].prompt, ans["answer"], model)["score"]
        )

    # AGGREGATE per strategy.
    strat_aggs: list[StrategyAggregate] = []
    for strat in strategies:
        per_prompt = picks[strat.name]
        clf_per_prompt = clf_calls[strat.name]
        n = 0
        score_sum = infer_sum = tax_sum = 0.0
        distinct: set[str] = set()
        for i, model in enumerate(per_prompt):
            if not model:
                continue
            n += 1
            distinct.add(model)
            ans = answers[(i, model)]
            score_sum += judgments[(i, model)]
            infer_sum += _infer_cost_usd(providers.catalog, model, ans["prompt_tokens"], ans["completion_tokens"])
            tax_sum += sum(_classifier_tax_usd(cm, traffic[i].prompt) for cm in clf_per_prompt[i])
        n = n or 1
        strat_aggs.append(StrategyAggregate(
            name=strat.name, n=len([m for m in per_prompt if m]),
            mean_judge_score=score_sum / n,
            mean_infer_cost_usd=infer_sum / n,
            mean_classifier_tax_usd=tax_sum / n,
            mean_cost_with_tax_usd=(infer_sum + tax_sum) / n,
            distinct_models_picked=len(distinct),
        ))

    # SERVED reference.
    fb_counts: dict[str, int] = {}
    served_score_sum = served_cost_sum = 0.0
    for i, row in enumerate(traffic):
        fb_counts[row.feedback_rating] = fb_counts.get(row.feedback_rating, 0) + 1
        ans = answers[(i, served_norm[i])]
        served_score_sum += judgments[(i, served_norm[i])]
        served_cost_sum += _infer_cost_usd(providers.catalog, served_norm[i], ans["prompt_tokens"], ans["completion_tokens"])
    n_traffic = len(traffic) or 1
    served = ServedReference(
        n=len(traffic), mean_judge_score=served_score_sum / n_traffic,
        mean_infer_cost_usd=served_cost_sum / n_traffic, feedback_counts=fb_counts,
    )
    return PipelineResult(cfg.live, estimate, strat_aggs, served,
                          {k: [m or "" for m in v] for k, v in picks.items()})


def run_pipeline(cfg: PipelineConfig, providers: Providers, traffic: list[TrafficRow]) -> PipelineResult:
    """plan() + execute() in one call. `estimate_only` stops after the plan."""
    plan_ = plan(cfg, providers, traffic)
    if cfg.estimate_only:
        return PipelineResult(
            cfg.live, plan_.estimate, [], ServedReference(0, 0.0, 0.0, {}),
            {k: [m or "" for m in v] for k, v in plan_.picks.items()},
        )
    return execute(cfg, providers, traffic, plan_)


def load_and_run(cfg: PipelineConfig) -> PipelineResult:
    """Convenience: load traffic, wire providers from cfg, run."""
    traffic = load_traffic(cfg.traffic_path) if cfg.traffic_path else load_traffic()
    providers = build_providers(cfg)
    return run_pipeline(cfg, providers, traffic)
