"""
Phase-2 routing data — the ACTIVE routing-data version (v4) brand maps + ranking,
ported from routersvc origin/main (`app/auto_router/benchmarks.py` +
`app/auto_router/versions.py`, `auto_router_data_version = "v4"`).

Phase 1 replays the frozen v1 SUPERMODE table against RouterBench's 11 models. Phase 2
routes over the LIVE Mesh catalog, so it uses the v4 data the production router actually
runs: the v1 table with the six v4 ranking overrides, and the brand→Mesh-model-id maps
(premium/standard tiers). Keep in lockstep with routersvc if v4 changes.

The category/mode come from the classifier (live) or a mock (dry run); this module only
turns a (category, mode) into an ordered list of catalog model ids to route among.
"""

from __future__ import annotations

from router_eval.benchmark_table import SUPERMODE_BENCHMARKS
from router_eval.heuristic_gate import CONVERSATION_CATEGORY

# ── v4 brand → Mesh catalog model id (BENCHMARK_BRAND_TO_{PREMIUM,STANDARD}_MODEL_ID) ──
BRAND_PREMIUM: dict[str, str] = {
    "chatgpt": "openai/gpt-5.4",
    "claude": "anthropic/claude-sonnet-4.6",
    "gemini": "google/gemini-3.1-pro-preview",
    "deepseek": "deepseek/deepseek-r1",
    "grok": "x-ai/grok-4.20",
    "perplexity": "perplexity/sonar-pro",
    "mistral": "mistralai/mistral-medium-3.1",
    "qwen": "qwen/qwen3-max",
    "moonshot": "moonshotai/kimi-k2.5",
    "bytedance": "bytedance-seed/seed-2.0-lite",
}
BRAND_STANDARD: dict[str, str] = {
    "chatgpt": "openai/gpt-5.4-mini",
    "claude": "anthropic/claude-haiku-4.5",
    "gemini": "google/gemini-3-flash-preview",
    "deepseek": "deepseek/deepseek-v3.2",
    "grok": "xai/grok-4.1-fast-non-reasoning",
    "perplexity": "perplexity/sonar",
    "mistral": "mistralai/mistral-medium-3.1",
    "qwen": "qwen/qwen-flash",
    "moonshot": "moonshotai/kimi-k2.5",
    "bytedance": "bytedance-seed/seed-2.0-lite",
}

# ── v4 ranking overrides (versions._V4_RANKING_OVERRIDES) merged over the v1 table ──
_V4_RANKING_OVERRIDES: dict[str, list] = {
    "File generation - pdf, docx, pptx, excel": [["deepseek"], "gemini", "claude"],
    "General reasoning / Q&A - General Conversation, Chatting": [
        ["claude", "deepseek", "chatgpt", "gemini", "grok", "mistral"],
        "qwen", "moonshot", "perplexity",
    ],
    "Web research / citations - Freshness (recency)": [
        ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral",
        "bytedance", "perplexity", "qwen",
    ],
    "Web research / citations - News": [
        ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral",
        "bytedance", "perplexity", "qwen",
    ],
    "Web research / citations - Recent Topics/Latest Information": [
        ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral",
        "bytedance", "perplexity", "qwen",
    ],
    "Web research / citations - currency": [
        ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral",
        "bytedance", "perplexity", "qwen",
    ],
}
V4_BENCHMARKS: dict[str, list] = {**SUPERMODE_BENCHMARKS, **_V4_RANKING_OVERRIDES}

DEFAULT_CATEGORY = CONVERSATION_CATEGORY
CATEGORIES = list(V4_BENCHMARKS.keys())


def _brand_map(mode: str) -> dict[str, str]:
    return BRAND_STANDARD if mode == "standard" else BRAND_PREMIUM


def ranked_models_for_category(
    category: str, mode: str, catalog_ids: set[str]
) -> list[str]:
    """Walk the v4 ranking for `category`, best-first, resolving each ranked brand to
    its tier model id and keeping only ids present in the catalog. Deduped, rank order.
    Tie-group members are ordered lexically (deterministic). Empty when no ranked brand
    resolves into the catalog."""
    brand_to_model = _brand_map(mode)
    ranking = V4_BENCHMARKS.get(category) or V4_BENCHMARKS.get(DEFAULT_CATEGORY, [])
    out: list[str] = []
    seen: set[str] = set()
    for entry in ranking:
        brands = [entry] if isinstance(entry, str) else entry
        models = sorted(
            {brand_to_model[b] for b in brands if b in brand_to_model and brand_to_model[b] in catalog_ids}
        )
        for m in models:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def resolve_benchmark_model(category: str, mode: str, catalog_ids: set[str]) -> str | None:
    """The benchmark strategy's pick: the top-ranked brand's tier model present in the
    catalog (rank-0 of `ranked_models_for_category`), else None (caller falls back)."""
    ranked = ranked_models_for_category(category, mode, catalog_ids)
    return ranked[0] if ranked else None


def conversation_standard_model(catalog_ids: set[str]) -> str | None:
    """The heuristic fast-lane pick over the live catalog: the conversation category's
    rank-0 brand (deterministic first-of-tie), STANDARD tier, if present in the catalog."""
    ranking = V4_BENCHMARKS.get(DEFAULT_CATEGORY) or []
    top = ranking[0] if ranking else None
    brand = top if isinstance(top, str) else (top[0] if top else None)
    model_id = BRAND_STANDARD.get(brand) if brand else None
    return model_id if model_id and model_id in catalog_ids else None
