"""
LLM judge — scores an answer 0–1 on correctness + helpfulness + instruction-following.

`LiveJudge` calls a configurable strong judge model through the Mesh API (live only) and
caches by (judge_model, answered_model, prompt, answer). `MockJudge` is a deterministic
offline scorer for the dry run + tests (NOT a real judgment). Both return the same shape:
    {"score": float in [0,1], "rationale": str, "judge_model": str}

The judge sees the prompt and the answer — both PII — so the judge model counts as a place
prompts are sent (alongside the Mesh inference call), and its cache is gitignored.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from router_eval.phase2.cache import DiskCache, make_key
from router_eval.phase2.mesh_client import MeshAPIError

# Default judge: a strong model. Configurable via the CLI (--judge-model).
DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-4.8"

_RUBRIC_SYSTEM = (
    "You are a strict evaluation judge. Score the assistant's answer to the user's "
    "request on three axes — correctness, helpfulness, and instruction-following — then "
    "return a SINGLE overall quality score in [0,1]. Reply with ONLY a JSON object: "
    '{\"score\": <float 0..1>, \"rationale\": \"<one sentence>\"}. No other text.'
)


def _answer_hash(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8", "replace")).hexdigest()[:16]


class Judge:
    judge_model: str = DEFAULT_JUDGE_MODEL

    def score(self, prompt: str, answer: str, answered_model: str) -> dict:
        raise NotImplementedError


@dataclass
class MockJudge(Judge):
    """Deterministic offline scorer — a stand-in, NOT a real judgment.

    Produces a stable pseudo-score in [0,1] from a hash of (prompt, answer), nudged so an
    empty answer scores ~0 and a substantive answer scores higher. Enough to exercise the
    aggregation deterministically; never interpret its numbers as quality."""

    judge_model: str = DEFAULT_JUDGE_MODEL
    calls: int = 0

    def score(self, prompt: str, answer: str, answered_model: str) -> dict:
        self.calls += 1
        if not answer.strip():
            return {"score": 0.0, "rationale": "empty answer", "judge_model": self.judge_model}
        h = int(hashlib.sha256(f"{prompt}\x00{answer}".encode()).hexdigest(), 16)
        base = (h % 1000) / 1000.0  # 0..0.999
        # Nudge toward the middle-high band and reward some length (deterministically).
        length_bonus = min(len(answer) / 4000.0, 0.25)
        score = max(0.0, min(1.0, 0.4 + 0.4 * base + length_bonus))
        return {"score": round(score, 4), "rationale": "mock deterministic score",
                "judge_model": self.judge_model}


@dataclass
class LiveJudge(Judge):
    """Real judge calls through the Mesh API, cached by content. Live mode only."""

    client: object = None  # MeshClient (live)
    judge_model: str = DEFAULT_JUDGE_MODEL
    cache: DiskCache = field(default_factory=DiskCache)
    calls: int = 0

    def score(self, prompt: str, answer: str, answered_model: str) -> dict:  # pragma: no cover - live only
        self.calls += 1
        answer = answer or ""  # a null content / missing served answer must not crash the hash
        key = make_key("judge", self.judge_model, answered_model, prompt, _answer_hash(answer))

        def _compute() -> dict:
            user = (
                f"User request:\n\"\"\"\n{prompt[:4000]}\n\"\"\"\n\n"
                f"Assistant answer:\n\"\"\"\n{answer[:6000]}\n\"\"\""
            )
            try:
                raw, _usage = self.client.chat(
                    self.judge_model, user, system=_RUBRIC_SYSTEM, max_tokens=200, temperature=0.0,
                )
            except MeshAPIError:
                # A judge call that fails after the client's retries scores as an empty
                # reply (→ 0.0 in _parse) rather than aborting the whole run. Rare, and
                # visible as a 0.0 in the output.
                return {"raw": ""}
            return {"raw": raw}

        raw = self.cache.get_or_compute("judge", key, _compute).get("raw", "")
        score, rationale = self._parse(raw)
        return {"score": score, "rationale": rationale, "judge_model": self.judge_model}

    @staticmethod
    def _parse(raw: str) -> tuple[float, str]:
        """Extract the JSON {score, rationale}; fail-soft to 0.0 on an unparseable reply."""
        if not raw:
            return 0.0, "empty judge reply"
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return 0.0, "no json in judge reply"
        try:
            obj = json.loads(match.group(0))
            score = float(obj.get("score", 0.0))
        except (ValueError, TypeError, json.JSONDecodeError):
            return 0.0, "unparseable judge score"
        score = max(0.0, min(1.0, score))
        return score, str(obj.get("rationale", ""))[:200]
