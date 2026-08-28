"""
Gated client for the Mesh OpenAI-compatible API.

THE NETWORK GATE. Every real HTTP call in Phase 2 goes through a MeshClient, and a
MeshClient only touches the network when `live=True` AND an api_key is present. In the
default dry run no MeshClient is constructed at all (the pipeline uses mock providers),
but even if one is, calling a network method without live+key raises `LiveCallBlocked`.

Uses only stdlib `urllib` — no `openai`/`requests` dependency — so the package imports
and tests run with nothing installed and no network.

Base URL: https://api.meshapi.ai/v1   Auth: Authorization: Bearer $MESH_API_KEY
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.meshapi.ai/v1"
API_KEY_ENV = "MESH_API_KEY"

# Retry transient failures (read timeouts, 429, 5xx) a few times with linear backoff.
# A 4xx like 400 model_capability_not_supported is NOT transient and fails fast.
_MAX_ATTEMPTS = 3
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_BACKOFF_S = 1.5


class LiveCallBlocked(RuntimeError):
    """Raised when a network call is attempted while not in live mode (or no key)."""


class MeshAPIError(RuntimeError):
    """A non-2xx response or transport error from the Mesh API (live mode)."""


@dataclass
class MeshClient:
    """Thin OpenAI-compatible client. Network is refused unless live + api_key."""

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    live: bool = False
    timeout_s: float = 90.0

    @classmethod
    def from_env(cls, *, live: bool, base_url: str = DEFAULT_BASE_URL) -> MeshClient:
        """Build from `MESH_API_KEY`. In live mode a missing key is a hard error — we
        never silently proceed to (or fake) a live call."""
        key = os.environ.get(API_KEY_ENV)
        if live and not key:
            raise LiveCallBlocked(
                f"--live requires the {API_KEY_ENV} env var (real operator key). "
                "Refusing to run live without it."
            )
        return cls(api_key=key, base_url=base_url, live=live)

    def _require_live(self, what: str) -> None:
        if not self.live:
            raise LiveCallBlocked(
                f"Refusing to {what}: not in live mode. Pass --live and set "
                f"{API_KEY_ENV} to make real calls."
            )
        if not self.api_key:
            raise LiveCallBlocked(f"Refusing to {what}: no {API_KEY_ENV}.")

    def _do_request(self, req: urllib.request.Request) -> dict:
        """Perform an HTTP request, wrapping EVERY transport failure — including bare
        ``socket.timeout`` (a read timeout is NOT a ``urllib.error.URLError``) — as
        ``MeshAPIError``, and retrying transient failures (timeouts / 429 / 5xx) with
        linear backoff. A 4xx (e.g. 400 model_capability) is not transient → fails fast.
        Wrapping everything is what lets the answerer mark one bad model failed instead
        of aborting the whole ~1200-call run."""
        method, url = req.get_method(), req.full_url
        last: MeshAPIError | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):  # pragma: no cover - live only
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:500]
                last = MeshAPIError(f"{method} {url} -> {exc.code}: {body}")
                if exc.code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS:
                    time.sleep(_BACKOFF_S * attempt)
                    continue
                raise last from exc
            except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
                last = MeshAPIError(f"{method} {url} transport error: {exc!r}")
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_BACKOFF_S * attempt)
                    continue
                raise last from exc
        raise last  # pragma: no cover - the loop always returns or raises

    def _post_json(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode("utf-8"), method="POST"
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        return self._do_request(req)  # pragma: no cover - live only

    def _get_json(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        return self._do_request(req)  # pragma: no cover - live only

    # ── OpenAI-compatible surface ────────────────────────────────────────────────
    def list_models(self) -> list[dict]:
        """GET /models -> full model objects. Accepts BOTH the OpenAI-compat
        ``{"data": [...]}`` envelope and a bare ``[...]`` list — Mesh's live
        ``/v1/models`` returns the bare list."""
        self._require_live("list models")
        body = self._get_json("/models")  # pragma: no cover - live only
        rows = body if isinstance(body, list) else body.get("data", [])
        return [m for m in rows if isinstance(m, dict) and m.get("id")]

    def list_model_ids(self) -> list[str]:
        """GET /models -> the catalog id list."""
        return [str(m["id"]) for m in self.list_models()]

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> tuple[str, dict]:
        """POST /chat/completions -> (answer_text, usage_dict). One user turn."""
        self._require_live("call inference")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        try:  # pragma: no cover - live only
            body = self._post_json("/chat/completions", payload)
        except MeshAPIError as exc:
            # Some models reject a specific sampling param outright (reasoning models:
            # "temperature is deprecated"; o-series: max_tokens). Drop the named param(s)
            # and retry ONCE, rather than dropping the model from the eval entirely.
            msg = str(exc).lower()
            dropped = [p for p in ("temperature", "max_tokens", "top_p") if p in msg and p in payload]
            if not dropped:
                raise
            for p in dropped:
                payload.pop(p, None)
            body = self._post_json("/chat/completions", payload)
        choices = body.get("choices") or []
        # A model can return message.content = null (empty completion, refusal, tool-only
        # reply). `.get("content", "")` returns None for an explicit null — coerce to "".
        content = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        return content, (body.get("usage") or {})
