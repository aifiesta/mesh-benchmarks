"""
Content-addressed disk cache — the dedupe/spend-control layer.

Every expensive result (a model answer, a judge score, a classifier reply) is stored
under sha256(key) so a repeated (prompt, model) — across strategies, or across reruns —
is computed once. Values are small JSON blobs.

The cache holds REAL PROMPTS AND ANSWERS (PII). Its default location
`router_eval/phase2/.cache/` is gitignored. Nothing here ever prints cached content.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CACHE_ROOT = Path(__file__).parent / ".cache"


def make_key(*parts: str) -> str:
    """Canonical cache key from parts, NUL-joined so parts can't collide."""
    return "\x00".join(parts)


@dataclass
class DiskCache:
    """A tiny namespaced JSON cache. `hits`/`misses` track dedupe effectiveness."""

    root: Path = DEFAULT_CACHE_ROOT
    hits: int = 0
    misses: int = 0
    _memo: dict[str, dict] = field(default_factory=dict)

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key: str) -> dict | None:
        memo_key = f"{namespace}\x00{key}"
        if memo_key in self._memo:
            self.hits += 1
            return self._memo[memo_key]
        path = self._path(namespace, key)
        if path.exists():
            value = json.loads(path.read_text())
            self._memo[memo_key] = value
            self.hits += 1
            return value
        self.misses += 1
        return None

    def put(self, namespace: str, key: str, value: dict) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        self._memo[f"{namespace}\x00{key}"] = value

    def get_or_compute(self, namespace: str, key: str, compute):
        """Return the cached value for key, or call `compute()` (a 0-arg callable that
        returns a JSON-able dict), store, and return it."""
        cached = self.get(namespace, key)
        if cached is not None:
            return cached
        value = compute()
        self.put(namespace, key, value)
        return value
