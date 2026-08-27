"""
The candidate model catalog the Phase-2 strategies route over.

Two sources, one shape (`CatalogModel`):
  * `load_sample_catalog()` — a small bundled offline catalog (fixtures/catalog_sample.json),
    grounded in the routersvc v4 brand maps (benchmarks.py) + known Mesh prices. Used by
    the DRY-RUN pipeline and tests. No network.
  * `fetch_live_catalog(client)` — the REAL catalog via `GET /v1/models` through the Mesh
    API (live only). Ids come from the endpoint; prices are merged from the bundled price
    map (the /v1/models list is id-only), and each id's SUPERMODE brand is inferred. The
    operator can point `price_map` at the real pricing source.

`brand` is the SUPERMODE_BENCHMARKS brand (claude/chatgpt/gemini/…) used to walk the
frozen ranking; `None` for models outside the ranked brands (they can still be picked by
the price baselines).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_SAMPLE_PATH = Path(__file__).parent / "fixtures" / "catalog_sample.json"

# Cost blend used for the price baselines + weighted's cost term (matches routersvc
# weighted_strategy._COST_ALPHA: blended = 0.3*prompt + 0.7*completion).
_COST_ALPHA = 0.3

# provider-prefix → SUPERMODE brand, for inferring a live model's brand from its id.
_PREFIX_TO_BRAND = {
    "openai": "chatgpt",
    "anthropic": "claude",
    "google": "gemini",
    "deepseek": "deepseek",
    "x-ai": "grok",
    "xai": "grok",
    "perplexity": "perplexity",
    "mistralai": "mistral",
    "qwen": "qwen",
    "moonshotai": "moonshot",
    "bytedance-seed": "bytedance",
    "bytedance": "bytedance",
}


def brand_of(model_id: str) -> str | None:
    """Infer the SUPERMODE brand from a model id's provider prefix (best-effort)."""
    prefix = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
    return _PREFIX_TO_BRAND.get(prefix)


def _to_float(v: object) -> float | None:
    """Best-effort float — the /v1/models pricing block sends strings like '0.15000000'."""
    try:
        return float(v) if v is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CatalogModel:
    model_id: str
    brand: str | None
    prompt_usd_per_1m: float | None
    completion_usd_per_1m: float | None

    @property
    def blended_usd_per_1m(self) -> float | None:
        if self.prompt_usd_per_1m is None or self.completion_usd_per_1m is None:
            return None
        return _COST_ALPHA * self.prompt_usd_per_1m + (1 - _COST_ALPHA) * self.completion_usd_per_1m


class Catalog:
    """A set of CatalogModels with lookup helpers the strategies use."""

    def __init__(self, models: list[CatalogModel]) -> None:
        self.models = models
        self._by_id = {m.model_id: m for m in models}

    def ids(self) -> list[str]:
        return [m.model_id for m in self.models]

    def get(self, model_id: str) -> CatalogModel | None:
        return self._by_id.get(model_id)

    def priced(self) -> list[CatalogModel]:
        """Models with a usable blended price (the pool the price baselines pick from)."""
        return [m for m in self.models if m.blended_usd_per_1m is not None]

    def brand_models(self, brand: str) -> list[CatalogModel]:
        return [m for m in self.models if m.brand == brand]


def _models_from_records(records: list[dict]) -> list[CatalogModel]:
    out: list[CatalogModel] = []
    for r in records:
        mid = str(r["model_id"])
        out.append(
            CatalogModel(
                model_id=mid,
                brand=r.get("brand") if "brand" in r else brand_of(mid),
                prompt_usd_per_1m=r.get("prompt_usd_per_1m"),
                completion_usd_per_1m=r.get("completion_usd_per_1m"),
            )
        )
    return out


def load_sample_catalog(path: Path | str = _SAMPLE_PATH) -> Catalog:
    """Offline sample catalog (no network) — dry-run + tests."""
    records = json.loads(Path(path).read_text())["models"]
    return Catalog(_models_from_records(records))


def price_map_from_sample(path: Path | str = _SAMPLE_PATH) -> dict[str, tuple[float, float]]:
    """model_id -> (prompt_usd_per_1m, completion_usd_per_1m) from the bundled sample,
    used to price a live /v1/models id list. Swap for the real pricing source live."""
    records = json.loads(Path(path).read_text())["models"]
    return {
        str(r["model_id"]): (r["prompt_usd_per_1m"], r["completion_usd_per_1m"])
        for r in records
        if r.get("prompt_usd_per_1m") is not None and r.get("completion_usd_per_1m") is not None
    }


def fetch_live_catalog(client, price_map: dict[str, tuple[float, float]] | None = None) -> Catalog:
    """Build the catalog from the LIVE `GET /v1/models` id list (live only).

    `client` is a MeshClient in live mode (raises if not). Prices are merged from
    `price_map` (defaults to the bundled sample map) since /v1/models is id-only; a
    model with no price maps to None (excluded from the price baselines, still routable
    by the classifier/benchmark strategies). Brand is inferred from the id prefix.
    """
    price_map = price_map if price_map is not None else price_map_from_sample()
    rows = client.list_models()  # network call, gated inside MeshClient
    models: list[CatalogModel] = []
    skipped_non_chat = 0
    for m in rows:
        mid = str(m["id"])
        # Route only over CHAT-capable models: the eval sends /chat/completions, so a
        # video/image/embedding-only model (e.g. luma/ray3-2) would 400 and abort the run.
        # `supports_completions_api` is authoritative; a MISSING flag is treated as
        # chat-capable (fail-open) so a schema change never silently empties the catalog.
        if m.get("supports_completions_api") is False:
            skipped_non_chat += 1
            continue
        # Prefer REAL prices from the /v1/models pricing block; fall back to the sample map.
        pr = m.get("pricing") or {}
        pp = _to_float(pr.get("prompt_usd_per_1m"))
        cp = _to_float(pr.get("completion_usd_per_1m"))
        if pp is None or cp is None:
            fb = price_map.get(mid)
            if fb:
                pp = pp if pp is not None else fb[0]
                cp = cp if cp is not None else fb[1]
        models.append(
            CatalogModel(
                model_id=mid,
                brand=brand_of(mid),
                prompt_usd_per_1m=pp,
                completion_usd_per_1m=cp,
            )
        )
    if skipped_non_chat:
        print(
            f"[catalog] filtered out {skipped_non_chat} non-chat models "
            f"(supports_completions_api=False); {len(models)} chat-capable remain"
        )
    return Catalog(models)
