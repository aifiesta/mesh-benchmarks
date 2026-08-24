"""
RouterBench dataset loading for the offline replay.

Two sources, one in-memory shape (`Item`):

* `load_fixture()` — a tiny synthetic JSONL shipped in the repo. Pure stdlib, no
  network, no heavy deps. This is what CI and the default `python -m
  router_eval.replay` run, so the skeleton is reproducible from a clean checkout.

* `load_routerbench()` — the REAL dataset, `withmartian/routerbench`. It ships as
  pickled pandas DataFrames (`routerbench_0shot.pkl` / `routerbench_5shot.pkl`),
  NOT as a `datasets`-loadable format — `datasets.load_dataset("withmartian/
  routerbench")` fails with "No supported data files". So we fetch the pickle with
  `huggingface_hub.hf_hub_download` and read it with `pandas.read_pickle`. Requires
  `pip install -r router_eval/requirements.txt` and network on first run
  (~99 MB for 0-shot, cached by huggingface_hub afterwards).

Both sources yield the SAME per-item schema so every policy and metric is source
agnostic. A RouterBench row has, per model, three columns:
    "<model>"                -> performance score in [0, 1]  (the quality axis)
    "<model>|total_cost"     -> USD cost of that response    (the cost axis)
    "<model>|model_response" -> raw text (ignored by the replay)
plus metadata columns `sample_id`, `prompt`, `eval_name`, `oracle_model_to_route_to`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Metadata (non-model) columns in the RouterBench schema.
META_COLUMNS = {"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"}

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "routerbench_fixture.jsonl"


@dataclass(frozen=True)
class Item:
    """One replay unit: a task label plus every model's KNOWN score and cost.

    `scores` / `costs` are hindsight ground truth. Realistic policies (random,
    premium, cheapest, benchmark, ...) may read only `task` and the candidate id
    set; ONLY the oracle is allowed to read `scores`/`costs` to pick per item.
    """

    sample_id: str
    task: str  # RouterBench eval_name
    scores: dict[str, float]  # model_id -> performance score in [0, 1]
    costs: dict[str, float]  # model_id -> USD cost

    @property
    def models(self) -> list[str]:
        return list(self.scores.keys())


def _coerce_float(value: object) -> float | None:
    """RouterBench score/cost cells are `object` dtype; coerce, drop non-numeric."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _row_to_item(row: dict, model_ids: list[str]) -> Item | None:
    """Turn a raw RouterBench row dict into an Item, or None if unusable."""
    scores: dict[str, float] = {}
    costs: dict[str, float] = {}
    for m in model_ids:
        s = _coerce_float(row.get(m))
        c = _coerce_float(row.get(f"{m}|total_cost"))
        if s is None or c is None:
            continue
        scores[m] = s
        costs[m] = c
    if not scores:
        return None
    return Item(
        sample_id=str(row.get("sample_id", "")),
        task=str(row.get("eval_name", "")),
        scores=scores,
        costs=costs,
    )


def _infer_model_ids(columns: list[str]) -> list[str]:
    """A model id is any column that also has a `<col>|total_cost` sibling."""
    cost_suffixed = {c[: -len("|total_cost")] for c in columns if c.endswith("|total_cost")}
    return [c for c in columns if c in cost_suffixed and c not in META_COLUMNS]


# ── Fixture source (stdlib only) ───────────────────────────────────────────────
def load_fixture(path: Path | str = FIXTURE_PATH) -> list[Item]:
    """Load the synthetic JSONL fixture. Each line is one RouterBench-shaped row."""
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return []
    model_ids = _infer_model_ids(list(rows[0].keys()))
    items = [_row_to_item(r, model_ids) for r in rows]
    return [it for it in items if it is not None]


# ── Real source (huggingface_hub + pandas) ─────────────────────────────────────
def load_routerbench(shots: int = 0, limit: int | None = None) -> list[Item]:
    """
    Download + load the real `withmartian/routerbench` dataset.

    shots: 0 -> routerbench_0shot.pkl (~99 MB), 5 -> routerbench_5shot.pkl (~171 MB).
    limit: keep only the first N rows (handy for a quick smoke run).

    Lazily imports pandas / huggingface_hub so the fixture path stays dependency
    free. Raises a clear error if they are missing.
    """
    try:
        import pandas as pd  # noqa: PLC0415
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "load_routerbench needs pandas + huggingface_hub. "
            "Install them with: pip install -r router_eval/requirements.txt"
        ) from exc

    if shots not in (0, 5):
        raise ValueError("shots must be 0 or 5 (RouterBench ships 0-shot and 5-shot)")
    filename = f"routerbench_{shots}shot.pkl"
    local_path = hf_hub_download(
        repo_id="withmartian/routerbench", filename=filename, repo_type="dataset"
    )
    df = pd.read_pickle(local_path)
    if limit is not None:
        df = df.head(limit)
    model_ids = _infer_model_ids(list(df.columns))
    items: list[Item] = []
    for record in df.to_dict(orient="records"):
        it = _row_to_item(record, model_ids)
        if it is not None:
            items.append(it)
    return items
