"""
Bridge between the frozen SUPERMODE_BENCHMARKS brand table and RouterBench's
fixed 11-model universe.

Two mappings live here, and BOTH are modelling choices (documented caveats):

1. brand -> RouterBench model
   The frozen table ranks 10 brands. RouterBench has 11 concrete models. Only
   three brands have a RouterBench representative (claude, chatgpt, mistral); the
   other seven brands the table ranks (gemini, grok, perplexity, qwen, moonshot,
   bytedance, deepseek) have NO RouterBench model and are simply skipped when the
   ranking is walked. Symmetrically, several RouterBench models (WizardLM, the two
   meta/llama models, Yi-34B) belong to no ranked brand and can never be chosen by
   the benchmark policy. See UNMAPPED_BRANDS / UNMAPPED_ROUTERBENCH_MODELS.

2. RouterBench eval_name -> SUPERMODE task category
   RouterBench's `eval_name` is granular (dozens of `mmlu-*`, `grade-school-math`,
   `mtbench*`, plus Chinese-language + misc tasks). We bucket each into a coarse
   family, then map the family to one SUPERMODE category string. In the live router
   this step is done by an LLM classifier reading the PROMPT; the replay instead
   uses the dataset's ground-truth `eval_name`. That means the benchmark policy is
   replayed under PERFECT task classification — an optimistic ceiling that ignores
   classifier error and classifier cost/latency (see the classifier-cost hook in
   metrics.py and the caveats in RESULTS.md).
"""

from __future__ import annotations

import random

from router_eval.benchmark_table import SUPERMODE_BENCHMARKS, SUPERMODE_BRANDS

# ── RouterBench's fixed 11-model universe (exact column names in the .pkl) ─────
ROUTERBENCH_MODELS: list[str] = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]

# ── brand -> RouterBench model, per tier ───────────────────────────────────────
# "premium" = the strongest representative of the brand in RouterBench; "standard"
# = the cheaper/smaller representative. The live benchmark strategy resolves a
# tier from the classifier ("premium"/"standard"); replay defaults to premium.
BRAND_TO_ROUTERBENCH_PREMIUM: dict[str, str] = {
    "claude": "claude-v2",
    "chatgpt": "gpt-4-1106-preview",
    "mistral": "mistralai/mixtral-8x7b-chat",
}

BRAND_TO_ROUTERBENCH_STANDARD: dict[str, str] = {
    "claude": "claude-instant-v1",
    "chatgpt": "gpt-3.5-turbo-1106",
    "mistral": "mistralai/mistral-7b-chat",
}

# Brands the frozen table ranks but that have NO RouterBench model (documented gap).
UNMAPPED_BRANDS: list[str] = sorted(set(SUPERMODE_BRANDS) - set(BRAND_TO_ROUTERBENCH_PREMIUM))

# RouterBench models that belong to no ranked brand — the benchmark policy can
# never select these, though every other policy (random/oracle/...) can.
UNMAPPED_ROUTERBENCH_MODELS: list[str] = [
    "WizardLM/WizardLM-13B-V1.2",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "zero-one-ai/Yi-34B-Chat",
]

# ── eval_name -> coarse family -> SUPERMODE category ───────────────────────────
# Coarse family for a RouterBench eval_name. Order matters (mmlu before math so
# `mmlu-*-mathematics` stays factuality; math-substring before the mtbench catch
# so `mtbench-math` is treated as math).
def eval_to_family(eval_name: str) -> str:
    e = (eval_name or "").lower()
    if e.startswith("mmlu"):
        return "mmlu"
    if e == "grade-school-math" or "math" in e or e == "chinese-remainder-theorem":
        return "math"
    if e == "mbpp":
        return "code"
    if e == "arc-challenge":
        return "arc"
    if e == "hellaswag":
        return "hellaswag"
    if e == "winogrande":
        return "winogrande"
    if e.startswith("mtbench"):
        return "mtbench"
    return "other"


# Which SUPERMODE category each family is routed through. These are deliberate,
# reviewable choices — not ground truth. `other` (Chinese-language + misc evals)
# and the conversational families fall back to a general reasoning category.
FAMILY_TO_CATEGORY: dict[str, str] = {
    "mmlu": "General reasoning / Q&A - Closed-book factuality",
    "math": "Math / logic - Arithmetic & word problems",
    "code": "Coding - Algorithmic / competitive programming",
    "arc": "General reasoning / Q&A - Closed-book factuality",
    "hellaswag": "General reasoning / Q&A - Ambiguity handling",
    "winogrande": "General reasoning / Q&A - Ambiguity handling",
    "mtbench": "General reasoning / Q&A - General Conversation, Chatting",
    "other": "General reasoning / Q&A - General Conversation, Chatting",
}

# The category the benchmark strategy falls back to when a family has no mapping.
DEFAULT_CATEGORY = "General reasoning / Q&A - General Conversation, Chatting"


def eval_to_category(eval_name: str) -> str:
    """Map a RouterBench eval_name to a SUPERMODE_BENCHMARKS category string."""
    return FAMILY_TO_CATEGORY.get(eval_to_family(eval_name), DEFAULT_CATEGORY)


def resolve_benchmark_model(
    eval_name: str,
    candidates: list[str],
    *,
    tier: str = "premium",
    rng: random.Random | None = None,
) -> str | None:
    """
    Port of the routersvc benchmark DECISION LOGIC onto RouterBench models.

    Mirrors `resolve_from_benchmark_category` in routersvc: look up the category's
    brand ranking, walk it best-first, and at the first tier whose brands map to a
    RouterBench model present in `candidates`, pick one (random among ties, seeded).
    Returns None when no ranked+mapped brand is available (caller applies its own
    fallback). This is the same waterfall the live strategy runs; only the
    brand->model universe is swapped from the Mesh catalog to RouterBench.
    """
    rng = rng or random
    brand_to_model = (
        BRAND_TO_ROUTERBENCH_STANDARD if tier == "standard" else BRAND_TO_ROUTERBENCH_PREMIUM
    )
    cand = set(candidates)
    category = eval_to_category(eval_name)
    ranking = SUPERMODE_BENCHMARKS.get(category) or SUPERMODE_BENCHMARKS.get(DEFAULT_CATEGORY, [])

    for entry in ranking:
        brands = [entry] if isinstance(entry, str) else entry
        options = sorted(
            {brand_to_model[b] for b in brands if b in brand_to_model and brand_to_model[b] in cand}
        )
        if options:
            return rng.choice(options)
    return None
