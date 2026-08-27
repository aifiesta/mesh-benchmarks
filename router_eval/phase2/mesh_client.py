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
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.meshapi.ai/v1"
API_KEY_ENV = "MESH_API_KEY"


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
    timeout_s: float = 60.0

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

    def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - live only
            body = exc.read().decode("utf-8", "replace")[:500]
            raise MeshAPIError(f"POST {path} -> {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - live only
            raise MeshAPIError(f"POST {path} transport error: {exc}") from exc

    def _get_json(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - live only
            body = exc.read().decode("utf-8", "replace")[:500]
            raise MeshAPIError(f"GET {path} -> {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - live only
            raise MeshAPIError(f"GET {path} transport error: {exc}") from exc

    # ── OpenAI-compatible surface ────────────────────────────────────────────────
    def list_model_ids(self) -> list[str]:
        """GET /models -> the catalog id list (OpenAI-compat {data:[{id}]})."""
        self._require_live("list models")
        body = self._get_json("/models")  # pragma: no cover - live only
        return [str(m.get("id")) for m in body.get("data", []) if m.get("id")]

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
        body = self._post_json("/chat/completions", payload)  # pragma: no cover - live only
        choices = body.get("choices") or []
        content = (choices[0].get("message") or {}).get("content", "") if choices else ""
        return content, (body.get("usage") or {})
