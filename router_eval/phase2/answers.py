"""
Answer providers — get a model's answer to a prompt, deduped + cached.

`LiveAnswerer` calls the Mesh chat API (live only) and caches by (model, prompt), so a
(prompt, model) pair picked by several strategies is paid for ONCE. `MockAnswerer` returns
a deterministic offline answer for the dry run + tests.

Return shape: {"answer": str, "prompt_tokens": int, "completion_tokens": int, "model": str}.
Cost is computed by the pipeline from these token counts × the catalog price.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from router_eval.phase2.cache import DiskCache, make_key
from router_eval.phase2.mesh_client import MeshAPIError

# Models that have already proven they cannot serve a chat request. The catalog flags
# ~18% of models `supports_completions_api=true` that in fact 4xx/hang (paddleocr-vl, a
# vision/OCR model, was picked 294x and cost ~2 minutes of timeout EACH). Once a model
# has failed this many times we stop paying the timeout and fail it instantly — the
# outcome is identical, only far cheaper. Per-process; the disk cache persists the rest.
_DEAD_AFTER = 3
_model_failures: dict[str, int] = {}


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Answerer:
    def answer(self, prompt: str, model_id: str) -> dict:
        raise NotImplementedError


@dataclass
class MockAnswerer(Answerer):
    """Deterministic offline answer — a stand-in, NOT a real model output."""

    calls: int = 0

    def answer(self, prompt: str, model_id: str) -> dict:
        self.calls += 1
        text = f"[mock:{model_id}] answer to: {prompt[:80]}"
        return {
            "answer": text,
            "prompt_tokens": _est_tokens(prompt),
            "completion_tokens": _est_tokens(text),
            "model": model_id,
        }


@dataclass
class LiveAnswerer(Answerer):
    """Real Mesh inference, cached by (model, prompt). Live mode only."""

    client: object = None  # MeshClient (live)
    cache: DiskCache = field(default_factory=DiskCache)
    max_tokens: int = 1024
    temperature: float = 0.7
    calls: int = 0

    def answer(self, prompt: str, model_id: str) -> dict:  # pragma: no cover - live only
        key = make_key("answer", model_id, prompt)

        def _compute() -> dict:
            # Short-circuit a model already known dead in this process.
            if _model_failures.get(model_id, 0) >= _DEAD_AFTER:
                return {"answer": "", "prompt_tokens": _est_tokens(prompt),
                        "completion_tokens": 0, "model": model_id,
                        "failed": True, "error": "known-dead model (skipped)"}
            self.calls += 1
            try:
                text, usage = self.client.chat(
                    model_id, prompt, max_tokens=self.max_tokens, temperature=self.temperature
                )
            except MeshAPIError as exc:
                _model_failures[model_id] = _model_failures.get(model_id, 0) + 1
                # A model that can't serve this prompt (capability error, transient 5xx,
                # rate limit) yields a FAILED answer — empty text the judge scores ~0 —
                # instead of aborting the whole ~900-call run. The catalog chat-capable
                # filter should prevent capability errors; this is defence in depth.
                return {
                    "answer": "",
                    "prompt_tokens": _est_tokens(prompt),
                    "completion_tokens": 0,
                    "model": model_id,
                    "failed": True,
                    "error": str(exc)[:200],
                }
            return {
                "answer": text,
                "prompt_tokens": int(usage.get("prompt_tokens") or _est_tokens(prompt)),
                "completion_tokens": int(usage.get("completion_tokens") or _est_tokens(text)),
                "model": model_id,
            }

        return self.cache.get_or_compute("answer", key, _compute)

    def seed(self, prompt: str, model_id: str, answer: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Pre-load a known (prompt, model) answer into the cache — e.g. the already-served
        response — so it costs no live call when a strategy also picks that model."""
        key = make_key("answer", model_id, prompt)
        if self.cache.get("answer", key) is None:
            self.cache.put("answer", key, {
                "answer": answer, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "model": model_id, "seeded": True,
            })
