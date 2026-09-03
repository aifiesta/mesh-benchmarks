"""
Frozen SUPERMODE_BENCHMARKS ranking table — ported VERBATIM from routersvc.

Source of truth: routersvc `app/auto_router/benchmarks.py` (routing-data version v1).
This is the hand-curated 48-task-category x 10-model-brand table that the
production `model=auto` `benchmark` strategy uses to rank candidate brands.

We port ONLY the ranking data (the decision logic under test). We deliberately do
NOT port routersvc's `BENCHMARK_BRAND_TO_{PREMIUM,STANDARD}_MODEL_ID` maps, because
those resolve to live Mesh catalog model ids (e.g. `openai/gpt-5.4`). RouterBench
has its own fixed 11-model universe, so the brand -> concrete-model bridge lives in
`routerbench_bridge.py` instead. Everything below is a copy of the frozen table so
the replay exercises the exact ranking a routing decision would see in prod.

Keep this file in lockstep with the routersvc source if v1 ever changes.

RE-SYNCED 2026-09-03: the port had DRIFTED from routersvc in four "General reasoning /
Q&A" categories — Closed-book factuality, Decomposition / step-planning and Ambiguity
handling each demoted `claude` out of tier-1, and Uncertainty calibration ranked
`chatgpt` ahead of `gemini`. Since the benchmark strategy only ever picks from tier-1,
three of those four changed which model the baseline arm actually routed to, so every
`benchmark` number measured before this fix was scored against a table production does
not run. Now regenerated directly from `app.auto_router.benchmarks.SUPERMODE_BENCHMARKS`
and verified equal.
"""

from __future__ import annotations

# fmt: off
SUPERMODE_BENCHMARKS: dict[str, list[str | list[str]]] = {
    "File generation - pdf, docx, pptx, excel": [
        "claude", "gemini", "deepseek",
    ],
    "Creative writing / storytelling - Long-form coherence": [
        "claude", ["chatgpt", "gemini"], "moonshot", "grok", "qwen", "perplexity", "deepseek", "mistral",
    ],
    "Creative writing / storytelling - Voice mimicry / style control": [
        "claude", "grok", "chatgpt", "gemini", "bytedance", "qwen", "moonshot", "perplexity", "deepseek", "mistral",
    ],
    "Creative writing / storytelling - Character & world consistency": [
        "claude", ["gemini", "chatgpt"], "moonshot", "bytedance", "grok", "qwen", "perplexity", "deepseek", "mistral",
    ],
    "Creative writing / storytelling - Instruction adherence": [
        "claude", "deepseek", "chatgpt", "gemini", "qwen", "moonshot", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Creative writing / storytelling- Revision quality": [
        "claude", "chatgpt", "gemini", "moonshot", "qwen", "grok", "deepseek", "bytedance", "perplexity", "mistral",
    ],
    "General reasoning / Q&A - General Conversation, Chatting": [
        ["claude", "chatgpt", "gemini", "grok", "mistral"], "deepseek", "qwen", "moonshot", "perplexity",
    ],
    "General reasoning / Q&A - Closed-book factuality": [
        "claude", ["deepseek", "gemini"], "chatgpt", "qwen", "moonshot", "grok", "perplexity", "bytedance", "mistral",
    ],
    "General reasoning / Q&A - Decomposition / step-planning": [
        "claude", ["gemini", "deepseek"], "chatgpt", "moonshot", "qwen", "grok", "perplexity", "mistral", "bytedance",
    ],
    "General reasoning / Q&A - Numerical reliability": [
        "deepseek", ["moonshot", "chatgpt"], "claude", ["qwen", "gemini"], "grok", "bytedance", "perplexity", "mistral",
    ],
    "General reasoning / Q&A - Ambiguity handling": [
        "claude", ["chatgpt", "gemini"], "moonshot", "qwen", "deepseek", "grok", "perplexity", "mistral", "bytedance",
    ],
    "General reasoning / Q&A - Uncertainty calibration": [
        "gemini", "claude", "chatgpt", "moonshot", "qwen", "deepseek", "perplexity", "grok", "mistral", "bytedance",
    ],
    "Coding - Bug localization & debugging": [
        ["chatgpt", "claude"], ["deepseek", "gemini"], "qwen", "moonshot", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Coding - Repo comprehension (architecture)": [
        ["deepseek", "gemini"], ["chatgpt", "claude"], "qwen", "moonshot", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Coding - Feature scaffolding (greenfield)": [
        ["chatgpt", "claude"], ["deepseek", "gemini"], "qwen", "moonshot", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Coding - Migration / translation": [
        ["chatgpt", "claude"], ["deepseek", "gemini"], "qwen", "moonshot", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Coding - Test generation & CI scaffolds": [
        ["chatgpt", "claude"], ["deepseek", "gemini"], "qwen", "moonshot", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Coding - Algorithmic / competitive programming": [
        ["chatgpt", "claude"], ["deepseek", "gemini"], "qwen", "moonshot", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Coding - Data/ML notebooks (pandas/NumPy/Torch)": [
        ["deepseek", "gemini"], ["chatgpt", "claude"], "qwen", "moonshot", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Coding - Constrained output (AST/diff/JSON-only)": [
        ["chatgpt", "claude"], ["deepseek", "gemini"], "qwen", "moonshot", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Coding - Agentic build-run-fix loops": [
        ["chatgpt", "claude"], ["deepseek", "gemini"], "qwen", "moonshot", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Math / logic - Arithmetic & word problems": [
        "deepseek", "chatgpt", "claude", "qwen", "moonshot", "gemini", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Math / logic - Symbolic algebra": [
        "deepseek", "qwen", "chatgpt", "claude", "moonshot", "gemini", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Math / logic - Combinatorics / graph reasoning": [
        "deepseek", "chatgpt", "claude", "qwen", "moonshot", "gemini", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Math / logic - Proof-like explanations": [
        "deepseek", "chatgpt", "claude", "qwen", "moonshot", "gemini", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Multimodal - OCR & document QA": [
        "gemini", ["chatgpt", "claude"], "qwen", "moonshot", "deepseek", "perplexity", "bytedance", "grok", "mistral",
    ],
    "Multimodal - Chart/table reasoning": [
        "gemini", ["chatgpt", "claude", "qwen"], "moonshot", "deepseek", "perplexity", "grok", "mistral", "bytedance",
    ],
    "Multimodal - UI → code (from screenshot)": [
        "gemini", "chatgpt", "grok", "claude", "moonshot", "qwen", "deepseek", "perplexity", "mistral", "bytedance",
    ],
    "Multimodal - Image grounding & captions": [
        "gemini", "chatgpt", "claude", "qwen", "moonshot", "deepseek", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Multimodal - Long-video understanding": [
        "gemini", "chatgpt", "claude", "qwen", "moonshot", "deepseek", "grok", "perplexity", "mistral", "bytedance",
    ],
    "Safety / compliance - Refusal precision": [
        "claude", "chatgpt", "gemini", "perplexity", "qwen", "moonshot", "deepseek", "grok", "mistral", "bytedance",
    ],
    "Safety / compliance - Jailbreak resistance": [
        "claude", "chatgpt", "gemini", "perplexity", "qwen", "moonshot", "deepseek", "grok", "mistral", "bytedance",
    ],
    "Safety / compliance - PII & privacy handling": [
        "claude", "chatgpt", "gemini", "perplexity", "qwen", "moonshot", "deepseek", "grok", "mistral", "bytedance",
    ],
    "Safety / compliance - Policy following": [
        "claude", "chatgpt", "gemini", "perplexity", "qwen", "moonshot", "deepseek", "grok", "mistral", "bytedance",
    ],
    "Web research / citations - Citation precision": [
        ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral", "bytedance", "perplexity", "qwen",
    ],
    "Web research / citations - Recall / breadth": [
        ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral", "bytedance", "perplexity", "qwen",
    ],
    "Web research / citations - Quote fidelity": [
        ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral", "bytedance", "perplexity", "qwen",
    ],
    "Web research / citations - Freshness (recency)": [
        "perplexity", ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral", "bytedance", "qwen",
    ],
    "Web research / citations - Inline code-check / quick run": [
        ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral", "bytedance", "perplexity", "qwen",
    ],
    "Web research / citations - News": [
        "perplexity", ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral", "bytedance", "qwen",
    ],
    "Web research / citations - Recent Topics/Latest Information": [
        "perplexity", ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral", "bytedance", "qwen",
    ],
    "Web research / citations - currency": [
        "perplexity", ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral", "bytedance", "qwen",
    ],
    "Web research / citations - time": [
        ["chatgpt", "gemini", "claude", "grok"], "deepseek", "moonshot", "mistral", "bytedance", "perplexity", "qwen",
    ],
    "Conversational tone / style - Persona control": [
        "claude", "chatgpt", "grok", "gemini", "moonshot", "qwen", "deepseek", "perplexity", "mistral", "bytedance",
    ],
    "Conversational tone / style - Empathy & prosody": [
        ["claude", "gemini"], "chatgpt", "grok", "moonshot", "qwen", "deepseek", "bytedance", "perplexity", "mistral",
    ],
    "Conversational tone / style - Humor / wit": [
        ["claude", "grok"], ["chatgpt", "gemini", "moonshot", "deepseek"], "qwen", "perplexity", "bytedance", "mistral",
    ],
    "Conversational tone / style - Contextual memory (short-term)": [
        ["gemini", "claude"], ["chatgpt", "moonshot"], "deepseek", "qwen", "grok", "perplexity", "bytedance", "mistral",
    ],
    "Conversational tone / style - Long-term persona persistence": [
        ["gemini", "claude"], "chatgpt", "moonshot", "deepseek", "qwen", "grok", "perplexity", "mistral", "bytedance",
    ],
}
# fmt: on

SUPERMODE_CATEGORIES = list(SUPERMODE_BENCHMARKS.keys())

# The brand universe the frozen table ranks (10 brands). Used by the bridge to
# report which brands have / do not have a RouterBench representative.
SUPERMODE_BRANDS = sorted(
    {b for ranking in SUPERMODE_BENCHMARKS.values() for entry in ranking for b in ([entry] if isinstance(entry, str) else entry)}
)
