"""
Classifier backends for the Phase-2 strategies.

Two classifying decisions the router makes (each an LLM call in prod):
  * category(prompt)            — benchmark/weighted: the task CATEGORY + tier MODE,
                                  via the benchmark classifier (openai/gpt-4o-mini).
  * select_model(prompt, ids)   — registry: an explicit model id chosen from the
                                  catalog, via the registry classifier
                                  (google/gemini-3-flash-preview).

Both are GATED and RECORDED. `LiveClassifier` makes the real Mesh call (only in live
mode) and caches by content. `MockClassifier` is a DETERMINISTIC offline stand-in for
the dry run + tests — it is NOT the real classifier, just enough to exercise the
pipeline and produce a realistic dedupe/estimate. Every call (live or mock) is appended
to `.calls` so the pipeline can count classifier calls and price the classifier tax.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from router_eval.metrics import BENCHMARK_CLASSIFIER, REGISTRY_CLASSIFIER
from router_eval.phase2.cache import DiskCache, make_key
from router_eval.phase2.routing_data import CATEGORIES, DEFAULT_CATEGORY

CATEGORY_CLASSIFIER_MODEL = BENCHMARK_CLASSIFIER.model_id  # openai/gpt-4o-mini
MODEL_CLASSIFIER_MODEL = REGISTRY_CLASSIFIER.model_id  # google/gemini-3-flash-preview


@dataclass(frozen=True)
class ClassifierCall:
    kind: str  # "category" | "model"
    classifier_model: str
    prompt: str  # kept only in-memory for counting; never written to a committed file


class ClassifierBackend:
    """Base: records calls; subclasses implement the two decisions."""

    def __init__(self) -> None:
        self.calls: list[ClassifierCall] = []

    def _record(self, kind: str, model: str, prompt: str) -> None:
        self.calls.append(ClassifierCall(kind, model, prompt))

    def category(self, prompt: str) -> tuple[str, str]:
        raise NotImplementedError

    def select_model(self, prompt: str, candidate_ids: list[str]) -> str | None:
        raise NotImplementedError


# ── Deterministic offline mock (dry run + tests) ────────────────────────────────
_CODE_HINTS = ("code", "function", "bug", "python", "javascript", "sql", "error",
               "compile", "stack trace", "regex", "api", "typescript")
_MATH_HINTS = ("solve", "equation", "integral", "derivative", "calculate", "probability",
               "matrix", "algebra", "theorem", "sum of")
_WRITE_HINTS = ("write", "essay", "story", "poem", "draft", "rephrase", "paraphrase",
                "rewrite", "blog", "email")
_RESEARCH_HINTS = ("latest", "news", "today", "current", "recent", "2026", "price of")

_CODING_CATEGORY = "Coding - Algorithmic / competitive programming"
_MATH_CATEGORY = "Math / logic - Arithmetic & word problems"
_WRITE_CATEGORY = "Creative writing / storytelling - Long-form coherence"
_RESEARCH_CATEGORY = "Web research / citations - Recent Topics/Latest Information"


@dataclass
class MockClassifier(ClassifierBackend):
    """Deterministic keyword router — a stand-in, NOT the production classifier.

    Enough signal to give the dry-run picks realistic diversity (so the estimated
    dedupe is representative) while staying fully deterministic and offline.
    """

    calls: list[ClassifierCall] = field(default_factory=list)

    def category(self, prompt: str) -> tuple[str, str]:
        self._record("category", CATEGORY_CLASSIFIER_MODEL, prompt)
        low = prompt.lower()
        if any(h in low for h in _CODE_HINTS):
            category = _CODING_CATEGORY
        elif any(h in low for h in _MATH_HINTS):
            category = _MATH_CATEGORY
        elif any(h in low for h in _RESEARCH_HINTS):
            category = _RESEARCH_CATEGORY
        elif any(h in low for h in _WRITE_HINTS):
            category = _WRITE_CATEGORY
        else:
            category = DEFAULT_CATEGORY
        # Short, simple asks → standard tier; longer/complex → premium.
        mode = "standard" if len(prompt) <= 160 else "premium"
        return (category if category in CATEGORIES else DEFAULT_CATEGORY), mode

    def select_model(self, prompt: str, candidate_ids: list[str]) -> str | None:
        self._record("model", MODEL_CLASSIFIER_MODEL, prompt)
        if not candidate_ids:
            return None
        # Deterministic, content-dependent pick: hash the prompt onto the sorted ids.
        ordered = sorted(candidate_ids)
        idx = sum(ord(c) for c in prompt) % len(ordered)
        return ordered[idx]


# ── Live classifier (real Mesh calls; live mode only) ───────────────────────────
_CATEGORY_SYSTEM = (
    "You are a task classifier for an LLM router. Read the user request and reply with "
    "ONE line: the single best category from the provided list, a TAB, then either "
    "'premium' or 'standard'. No other text."
)


@dataclass
class LiveClassifier(ClassifierBackend):
    """Real classifier calls through the Mesh API, cached by content. Live mode only.

    The prompts here mirror the intent of routersvc benchmark_classifier /
    registry classifier; parsing is defensive and falls back to the default category /
    a lexical model pick on any miss (never raises), matching the production fail-soft.
    """

    client: object = None  # MeshClient (live)
    cache: DiskCache = field(default_factory=DiskCache)
    calls: list[ClassifierCall] = field(default_factory=list)

    def category(self, prompt: str) -> tuple[str, str]:  # pragma: no cover - live only
        self._record("category", CATEGORY_CLASSIFIER_MODEL, prompt)
        key = make_key("category", CATEGORY_CLASSIFIER_MODEL, prompt)

        def _compute() -> dict:
            listing = "\n".join(f"- {c}" for c in CATEGORIES)
            user = f"Categories:\n{listing}\n\nUser request:\n\"\"\"\n{prompt[:2000]}\n\"\"\""
            raw, _usage = self.client.chat(
                CATEGORY_CLASSIFIER_MODEL, user, system=_CATEGORY_SYSTEM,
                max_tokens=32, temperature=0.0,
            )
            return {"raw": raw}

        raw = self.cache.get_or_compute("classify", key, _compute).get("raw", "")
        return self._parse_category(raw)

    @staticmethod
    def _parse_category(raw: str) -> tuple[str, str]:
        line = (raw or "").strip().splitlines()[0] if raw and raw.strip() else ""
        parts = line.split("\t") if "\t" in line else line.rsplit(" ", 1)
        category = parts[0].strip() if parts else ""
        mode = parts[1].strip().lower() if len(parts) > 1 else "premium"
        if category not in CATEGORIES:
            category = DEFAULT_CATEGORY
        if mode not in ("premium", "standard"):
            mode = "premium"
        return category, mode

    def select_model(self, prompt: str, candidate_ids: list[str]) -> str | None:  # pragma: no cover - live only
        self._record("model", MODEL_CLASSIFIER_MODEL, prompt)
        if not candidate_ids:
            return None
        key = make_key("model", MODEL_CLASSIFIER_MODEL, prompt, str(len(candidate_ids)))

        def _compute() -> dict:
            listing = "\n".join(f"- {mid}" for mid in candidate_ids)
            system = (
                "You are a model router. Reply with exactly ONE model id from the list, "
                "verbatim, and nothing else."
            )
            user = f"Models:\n{listing}\n\nUser request:\n\"\"\"\n{prompt[:2000]}\n\"\"\""
            raw, _usage = self.client.chat(
                MODEL_CLASSIFIER_MODEL, user, system=system, max_tokens=32, temperature=0.0,
            )
            return {"raw": raw}

        raw = self.cache.get_or_compute("classify", key, _compute).get("raw", "")
        picked = (raw or "").strip().splitlines()[0].strip() if raw and raw.strip() else ""
        if picked in set(candidate_ids):
            return picked
        # Fail-soft: lexical fallback (never a bogus id).
        return sorted(candidate_ids)[0]
