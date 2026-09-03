"""
Phase-2 routing data — the ACTIVE routing-data version (v4) brand maps + ranking, and
the candidate expansion (v7), both ported from routersvc origin/main
(`app/auto_router/benchmarks.py` + `app/auto_router/versions.py`;
`auto_router_data_version = "v4"` is what production serves).

v7 (MESH-232) is a servability-gated coverage expansion derived over v4: 17 new brands
drawn from the 338 catalog models with proven production success, six of which may take
tier-1 (primary) traffic while the rest sit at tier-2+ — reachable as fallover alternates
and in the weighted pool, but never the benchmark strategy's pick. Comparing
`benchmark` (v4) against `benchmark_v7` on the same prompts is what tests whether the
expansion picks BETTER models or merely different ones.

v9 (MESH-232) composes the two: v7's pool with v8's repair. Its pick should match v8's
and its pool should match v7's — `benchmark_v9` is the arm that verifies it.

v8 (MESH-232 follow-up) is a one-entry repair over v4: the `grok` STANDARD model,
`xai/grok-4.1-fast-non-reasoning`, is failing production traffic, and v8 replaces it
with an alias to the premium `x-ai/grok-4.20`. Unlike v7 it is expected to MOVE the
pick, on the prompts where the benchmark strategy lands on standard-tier grok.

Phase 1 replays the frozen v1 SUPERMODE table against RouterBench's 11 models. Phase 2
routes over the LIVE Mesh catalog, so it uses the v4 data the production router actually
runs: the v1 table with the six v4 ranking overrides, and the brand→Mesh-model-id maps
(premium/standard tiers). Keep in lockstep with routersvc if v4 changes.

The category/mode come from the classifier (live) or a mock (dry run); this module only
turns a (category, mode) into an ordered list of catalog model ids to route among.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

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


# ── v7 brand maps + ranking (MESH-232 coverage expansion, derived over v4) ────────
# GENERATED from routersvc `app/auto_router/versions.py::V7` — keep in lockstep.
# The six brands that may take tier-1 (primary) traffic are claude5, deepseekv4, gpt55,
# kimi26, qwencoder and codestral; the other eleven are breadth-only (tier-2+), reachable
# as MESH-497 fallover alternates and in the weighted pool but never the benchmark pick.

BRAND_PREMIUM_V7: dict[str, str] = {
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
    "claude5": "anthropic/claude-sonnet-5",
    "deepseekv4": "deepseek/deepseek-v4-pro",
    "gpt55": "openai/gpt-5.5",
    "kimi26": "moonshotai/kimi-k2.6",
    "qwencoder": "qwen/qwen3-coder-next",
    "codestral": "mistralai/codestral-2508",
    "nova": "amazon/nova-pro-v1",
    "llama": "meta-llama/llama-4-maverick",
    "hunyuan": "tencent/hy3",
    "glm": "z-ai/glm-5.2",
    "minimax": "minimax/minimax-m3",
    "gptoss": "openai/gpt-oss-120b",
    "gemma": "google/gemma-4-26b-a4b-it",
    "ministral": "mistral/ministral-3-8b-instruct",
    "cohere": "cohere/command-a",
    "nemotron": "nvidia/nemotron-3-nano-30b-a3b",
    "mimo": "xiaomi/mimo-v2.5-pro"
}

BRAND_STANDARD_V7: dict[str, str] = {
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
    "claude5": "anthropic/claude-haiku-4.5",
    "deepseekv4": "deepseek/deepseek-v4-flash",
    "gpt55": "openai/gpt-5-mini",
    "kimi26": "moonshotai/kimi-k2.6",
    "qwencoder": "qwen/qwen3-coder-flash",
    "codestral": "mistralai/codestral-2508",
    "nova": "amazon/nova-2-lite-v1",
    "llama": "meta-llama/llama-4-scout",
    "hunyuan": "tencent/hy3",
    "glm": "z-ai/glm-4.7-flash",
    "minimax": "minimax/minimax-m2.5",
    "gptoss": "openai/gpt-oss-20b",
    "gemma": "google/gemma-4-26b-a4b-it",
    "ministral": "mistral/ministral-3-3b-instruct",
    "cohere": "cohere/command-a",
    "nemotron": "nvidia/nemotron-3-nano-30b-a3b",
    "mimo": "xiaomi/mimo-v2.5"
}

# 39 categories differ from v4
_V7_RANKING_OVERRIDES: dict[str, list] = {
    "File generation - pdf, docx, pptx, excel": [
        [
            "deepseek"
        ],
        [
            "gemini",
            "claude5",
            "deepseekv4"
        ],
        "claude"
    ],
    "Creative writing / storytelling - Long-form coherence": [
        "claude",
        [
            "chatgpt",
            "gemini",
            "kimi26",
            "claude5"
        ],
        [
            "moonshot",
            "llama",
            "minimax"
        ],
        "grok",
        "qwen",
        "perplexity",
        "deepseek",
        "mistral"
    ],
    "Creative writing / storytelling - Voice mimicry / style control": [
        "claude",
        [
            "grok",
            "kimi26",
            "claude5"
        ],
        [
            "chatgpt",
            "mimo"
        ],
        "gemini",
        "bytedance",
        "qwen",
        "moonshot",
        "perplexity",
        "deepseek",
        "mistral"
    ],
    "Creative writing / storytelling - Character & world consistency": [
        "claude",
        [
            "gemini",
            "chatgpt",
            "kimi26",
            "claude5"
        ],
        [
            "moonshot",
            "llama"
        ],
        "bytedance",
        "grok",
        "qwen",
        "perplexity",
        "deepseek",
        "mistral"
    ],
    "Creative writing / storytelling - Instruction adherence": [
        "claude",
        [
            "deepseek",
            "llama",
            "claude5"
        ],
        [
            "chatgpt",
            "glm"
        ],
        "gemini",
        "qwen",
        "moonshot",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Creative writing / storytelling- Revision quality": [
        "claude",
        [
            "chatgpt",
            "kimi26",
            "claude5"
        ],
        [
            "gemini",
            "mimo"
        ],
        "moonshot",
        "qwen",
        "grok",
        "deepseek",
        "bytedance",
        "perplexity",
        "mistral"
    ],
    "General reasoning / Q&A - General Conversation, Chatting": [
        [
            "claude",
            "deepseek",
            "chatgpt",
            "gemini",
            "grok",
            "mistral"
        ],
        [
            "qwen",
            "hunyuan",
            "llama",
            "claude5",
            "kimi26"
        ],
        [
            "moonshot",
            "gemma",
            "ministral",
            "minimax"
        ],
        "perplexity"
    ],
    "General reasoning / Q&A - Closed-book factuality": [
        "claude",
        [
            "deepseek",
            "gemini",
            "glm",
            "cohere",
            "claude5"
        ],
        [
            "chatgpt",
            "nova"
        ],
        "qwen",
        "moonshot",
        "grok",
        "perplexity",
        "bytedance",
        "mistral"
    ],
    "General reasoning / Q&A - Decomposition / step-planning": [
        "claude",
        [
            "gemini",
            "deepseek",
            "gptoss",
            "claude5",
            "deepseekv4"
        ],
        [
            "chatgpt",
            "nemotron",
            "glm"
        ],
        "moonshot",
        "qwen",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "General reasoning / Q&A - Numerical reliability": [
        "deepseek",
        [
            "moonshot",
            "chatgpt",
            "nemotron",
            "deepseekv4"
        ],
        [
            "claude",
            "gptoss",
            "glm"
        ],
        [
            "qwen",
            "gemini"
        ],
        "grok",
        "bytedance",
        "perplexity",
        "mistral"
    ],
    "General reasoning / Q&A - Ambiguity handling": [
        "claude",
        [
            "chatgpt",
            "gemini",
            "kimi26",
            "claude5"
        ],
        [
            "moonshot",
            "gemma",
            "nova"
        ],
        "qwen",
        "deepseek",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "General reasoning / Q&A - Uncertainty calibration": [
        "gemini",
        [
            "claude",
            "gptoss",
            "claude5"
        ],
        [
            "chatgpt",
            "cohere",
            "nova"
        ],
        "moonshot",
        "qwen",
        "deepseek",
        "perplexity",
        "grok",
        "mistral",
        "bytedance"
    ],
    "Coding - Bug localization & debugging": [
        [
            "chatgpt",
            "claude"
        ],
        [
            "deepseek",
            "gemini",
            "deepseekv4",
            "codestral",
            "claude5",
            "qwencoder"
        ],
        [
            "qwen",
            "glm"
        ],
        "moonshot",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Coding - Repo comprehension (architecture)": [
        [
            "deepseek",
            "gemini"
        ],
        [
            "chatgpt",
            "claude",
            "deepseekv4",
            "qwencoder",
            "claude5"
        ],
        [
            "qwen",
            "glm"
        ],
        "moonshot",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Coding - Feature scaffolding (greenfield)": [
        [
            "chatgpt",
            "claude"
        ],
        [
            "deepseek",
            "gemini",
            "codestral",
            "deepseekv4",
            "claude5",
            "qwencoder"
        ],
        [
            "qwen",
            "glm"
        ],
        "moonshot",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Coding - Migration / translation": [
        [
            "chatgpt",
            "claude"
        ],
        [
            "deepseek",
            "gemini",
            "deepseekv4",
            "codestral",
            "qwencoder"
        ],
        [
            "qwen",
            "glm"
        ],
        "moonshot",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Coding - Test generation & CI scaffolds": [
        [
            "chatgpt",
            "claude"
        ],
        [
            "deepseek",
            "gemini",
            "codestral",
            "claude5",
            "qwencoder"
        ],
        [
            "qwen",
            "deepseekv4"
        ],
        "moonshot",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Coding - Algorithmic / competitive programming": [
        [
            "chatgpt",
            "claude"
        ],
        [
            "deepseek",
            "gemini",
            "gpt55",
            "deepseekv4",
            "qwencoder"
        ],
        [
            "qwen",
            "glm",
            "minimax"
        ],
        "moonshot",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Coding - Data/ML notebooks (pandas/NumPy/Torch)": [
        [
            "deepseek",
            "gemini"
        ],
        [
            "chatgpt",
            "claude",
            "qwencoder",
            "deepseekv4",
            "claude5"
        ],
        [
            "qwen",
            "gptoss"
        ],
        "moonshot",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Coding - Constrained output (AST/diff/JSON-only)": [
        [
            "chatgpt",
            "claude"
        ],
        [
            "deepseek",
            "gemini",
            "cohere",
            "qwencoder",
            "codestral"
        ],
        [
            "qwen",
            "glm"
        ],
        "moonshot",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Coding - Agentic build-run-fix loops": [
        [
            "chatgpt",
            "claude"
        ],
        [
            "deepseek",
            "gemini",
            "deepseekv4",
            "claude5",
            "qwencoder"
        ],
        [
            "qwen",
            "codestral"
        ],
        "moonshot",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Math / logic - Arithmetic & word problems": [
        "deepseek",
        [
            "chatgpt",
            "nemotron",
            "deepseekv4",
            "gpt55"
        ],
        [
            "claude",
            "glm",
            "gptoss"
        ],
        "qwen",
        "moonshot",
        "gemini",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Math / logic - Symbolic algebra": [
        "deepseek",
        [
            "qwen",
            "gptoss",
            "deepseekv4",
            "gpt55"
        ],
        [
            "chatgpt",
            "glm"
        ],
        "claude",
        "moonshot",
        "gemini",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Math / logic - Combinatorics / graph reasoning": [
        "deepseek",
        [
            "chatgpt",
            "gpt55",
            "deepseekv4"
        ],
        [
            "claude",
            "glm",
            "nemotron"
        ],
        "qwen",
        "moonshot",
        "gemini",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Math / logic - Proof-like explanations": [
        "deepseek",
        [
            "chatgpt",
            "gptoss",
            "deepseekv4",
            "gpt55"
        ],
        [
            "claude",
            "glm"
        ],
        "qwen",
        "moonshot",
        "gemini",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Multimodal - OCR & document QA": [
        "gemini",
        [
            "chatgpt",
            "claude",
            "kimi26"
        ],
        "qwen",
        "moonshot",
        "deepseek",
        "perplexity",
        "bytedance",
        "grok",
        "mistral"
    ],
    "Multimodal - Chart/table reasoning": [
        "gemini",
        [
            "chatgpt",
            "claude",
            "qwen",
            "kimi26"
        ],
        [
            "moonshot",
            "gpt55"
        ],
        "deepseek",
        "perplexity",
        "grok",
        "mistral",
        "bytedance"
    ],
    "Multimodal - UI → code (from screenshot)": [
        "gemini",
        [
            "chatgpt",
            "gpt55"
        ],
        "grok",
        [
            "claude",
            "kimi26"
        ],
        "moonshot",
        "qwen",
        "deepseek",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Multimodal - Image grounding & captions": [
        "gemini",
        [
            "chatgpt",
            "gpt55"
        ],
        [
            "claude",
            "kimi26"
        ],
        "qwen",
        "moonshot",
        "deepseek",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Multimodal - Long-video understanding": [
        "gemini",
        "chatgpt",
        [
            "claude",
            "kimi26"
        ],
        "qwen",
        "moonshot",
        "deepseek",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Safety / compliance - Refusal precision": [
        "claude",
        [
            "chatgpt",
            "nova",
            "claude5"
        ],
        [
            "gemini",
            "cohere"
        ],
        "perplexity",
        "qwen",
        "moonshot",
        "deepseek",
        "grok",
        "mistral",
        "bytedance"
    ],
    "Safety / compliance - Jailbreak resistance": [
        "claude",
        [
            "chatgpt",
            "nova",
            "claude5"
        ],
        [
            "gemini",
            "llama"
        ],
        "perplexity",
        "qwen",
        "moonshot",
        "deepseek",
        "grok",
        "mistral",
        "bytedance"
    ],
    "Safety / compliance - PII & privacy handling": [
        "claude",
        [
            "chatgpt",
            "nova",
            "cohere",
            "claude5"
        ],
        [
            "gemini",
            "hunyuan"
        ],
        "perplexity",
        "qwen",
        "moonshot",
        "deepseek",
        "grok",
        "mistral",
        "bytedance"
    ],
    "Safety / compliance - Policy following": [
        "claude",
        [
            "chatgpt",
            "nova",
            "cohere",
            "claude5"
        ],
        [
            "gemini",
            "llama"
        ],
        "perplexity",
        "qwen",
        "moonshot",
        "deepseek",
        "grok",
        "mistral",
        "bytedance"
    ],
    "Conversational tone / style - Persona control": [
        "claude",
        [
            "chatgpt",
            "kimi26",
            "claude5"
        ],
        [
            "grok",
            "mimo",
            "ministral"
        ],
        "gemini",
        "moonshot",
        "qwen",
        "deepseek",
        "perplexity",
        "mistral",
        "bytedance"
    ],
    "Conversational tone / style - Empathy & prosody": [
        [
            "claude",
            "gemini"
        ],
        [
            "chatgpt",
            "hunyuan",
            "claude5"
        ],
        [
            "grok",
            "mimo",
            "ministral"
        ],
        "moonshot",
        "qwen",
        "deepseek",
        "bytedance",
        "perplexity",
        "mistral"
    ],
    "Conversational tone / style - Humor / wit": [
        [
            "claude",
            "grok"
        ],
        [
            "chatgpt",
            "gemini",
            "moonshot",
            "deepseek",
            "llama",
            "claude5"
        ],
        [
            "qwen",
            "ministral",
            "minimax"
        ],
        "perplexity",
        "bytedance",
        "mistral"
    ],
    "Conversational tone / style - Contextual memory (short-term)": [
        [
            "gemini",
            "claude"
        ],
        [
            "chatgpt",
            "moonshot",
            "hunyuan",
            "claude5",
            "deepseekv4"
        ],
        [
            "deepseek",
            "gemma"
        ],
        "qwen",
        "grok",
        "perplexity",
        "bytedance",
        "mistral"
    ],
    "Conversational tone / style - Long-term persona persistence": [
        [
            "gemini",
            "claude"
        ],
        [
            "chatgpt",
            "kimi26",
            "claude5"
        ],
        [
            "moonshot",
            "hunyuan"
        ],
        "deepseek",
        "qwen",
        "grok",
        "perplexity",
        "mistral",
        "bytedance"
    ]
}

V7_BENCHMARKS: dict[str, list] = {**V4_BENCHMARKS, **_V7_RANKING_OVERRIDES}


DEFAULT_CATEGORY = CONVERSATION_CATEGORY
CATEGORIES = list(V4_BENCHMARKS.keys())


@dataclass(frozen=True)
class RoutingData:
    """One routing-data version: its ranking table plus the two brand→model tier maps.

    Mirrors routersvc's `RoutingDataVersion` for the three fields the benchmark path
    actually reads. Passing one of these to the resolvers below is what lets a single
    strategy implementation be instantiated per version."""

    name: str
    benchmarks: dict[str, list]
    brand_premium: dict[str, str]
    brand_standard: dict[str, str]

    def brand_map(self, mode: str) -> dict[str, str]:
        return self.brand_standard if mode == "standard" else self.brand_premium


V4 = RoutingData("v4", V4_BENCHMARKS, BRAND_PREMIUM, BRAND_STANDARD)
V7 = RoutingData("v7", V7_BENCHMARKS, BRAND_PREMIUM_V7, BRAND_STANDARD_V7)

# ── v8 standard map (MESH-232 follow-up — repair the `grok` STANDARD tier) ────────
# GENERATED from routersvc `app/auto_router/versions.py::V8` — keep in lockstep.
#
# v4 maps `grok` standard to `xai/grok-4.1-fast-non-reasoning`, which is failing live
# traffic (90.1% success over 90 days, 79.5% over 30, 42.7% in September; 429-throttled
# on the vertex `xai/` path). `grok` is tier-1 in "General Conversation, Chatting" and in
# the four web-research categories, so the benchmark strategy reaches it — it is the pick
# on 26 of these 692 prompts, and in the round-2 run it failed 138 of 237 answer attempts.
#
# No cheaper grok clears the servability gate, so v8 aliases the standard tier to the
# premium `x-ai/grok-4.20` (99.6% over 14,870 prod completions, 927ms). Unlike v7 this is
# EXPECTED to move the pick — that is the point of the comparison.
#
# v8 changes exactly ONE map entry: rankings and the premium map are v4's, so
# `benchmark` vs `benchmark_v8` isolates the standard grok model and nothing else.
BRAND_STANDARD_V8: dict[str, str] = {**BRAND_STANDARD, "grok": "x-ai/grok-4.20"}

V8 = RoutingData("v8", V4_BENCHMARKS, BRAND_PREMIUM, BRAND_STANDARD_V8)

# ── v9 — v7's expanded pool WITH v8's grok repair (MESH-232) ─────────────────────
# GENERATED from routersvc `app/auto_router/versions.py::V9` — keep in lockstep.
#
# v7 and v8 are both derived over v4 and neither contains the other, but only ONE data
# version can be active at a time — so shipping just those two forces a choice between
# reaching more models on fallover (v7) and not routing standard-mode grok to a model
# that fails half its calls (v8). v9 is the composition that removes the choice.
#
# The derivation is one line because the changes are disjoint: v7 adds NEW brands at
# tier-2+ and leaves every v4 brand's ids alone; v8 changes ONE existing brand's
# standard model. So v9's PICK should equal v8's (v7's tier-1 is v4's) and its POOL
# should equal v7's. This arm exists to CHECK that rather than assert it — it is the
# live counterpart to the gateway's `test_v9_pick_is_identical_to_v8s`.
BRAND_STANDARD_V9: dict[str, str] = {**BRAND_STANDARD_V7, "grok": "x-ai/grok-4.20"}

V9 = RoutingData("v9", V7_BENCHMARKS, BRAND_PREMIUM_V7, BRAND_STANDARD_V9)


def _brand_map(mode: str) -> dict[str, str]:
    return BRAND_STANDARD if mode == "standard" else BRAND_PREMIUM


def ranked_models_for_category(
    category: str, mode: str, catalog_ids: set[str], data: RoutingData = V4
) -> list[str]:
    """Walk the v4 ranking for `category`, best-first, resolving each ranked brand to
    its tier model id and keeping only ids present in the catalog. Deduped, rank order.
    Tie-group members are ordered lexically (deterministic). Empty when no ranked brand
    resolves into the catalog."""
    brand_to_model = data.brand_map(mode)
    ranking = data.benchmarks.get(category) or data.benchmarks.get(DEFAULT_CATEGORY, [])
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


def resolve_benchmark_model(
    category: str,
    mode: str,
    catalog_ids: set[str],
    data: RoutingData = V4,
    rng: random.Random | None = None,
) -> str | None:
    """The benchmark strategy's pick, mirroring the gateway's
    `resolve_from_benchmark_category`: walk tiers best-first and return a model from the
    FIRST tier that resolves into the catalog — choosing at RANDOM among equally-ranked
    brands in that tier, exactly as the gateway's `random.choice` does.

    Passing `rng=None` keeps the old deterministic first-of-tie behaviour.

    Why the random tie-break matters: the gateway ranks tie-groups, not single brands, so
    the set a category can actually land on is the whole tier-1 group. Taking
    `sorted(...)[0]` collapses every tie to one member, which under-samples the reachable
    pool (it is why an earlier run measured 6 distinct models for a version whose tier-1
    set is 11) and, worse, makes a version that ADDS brands to tie-groups look like it
    changed almost nothing — the added brand only ever shows up when it happens to sort
    first. Any A/B over tie-group membership must use the random rule or it measures the
    alphabet rather than the routing data."""
    ranking = data.benchmarks.get(category) or data.benchmarks.get(DEFAULT_CATEGORY, [])
    brand_to_model = data.brand_map(mode)
    for entry in ranking:
        brands = [entry] if isinstance(entry, str) else entry
        models = sorted(
            {
                brand_to_model[b]
                for b in brands
                if b in brand_to_model and brand_to_model[b] in catalog_ids
            }
        )
        if models:
            return rng.choice(models) if rng is not None else models[0]
    return None


def conversation_standard_model(catalog_ids: set[str]) -> str | None:
    """The heuristic fast-lane pick over the live catalog: the conversation category's
    rank-0 brand (deterministic first-of-tie), STANDARD tier, if present in the catalog."""
    ranking = V4_BENCHMARKS.get(DEFAULT_CATEGORY) or []
    top = ranking[0] if ranking else None
    brand = top if isinstance(top, str) else (top[0] if top else None)
    model_id = BRAND_STANDARD.get(brand) if brand else None
    return model_id if model_id and model_id in catalog_ids else None
