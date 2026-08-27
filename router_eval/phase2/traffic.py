"""
Load the extracted Mesh traffic sample (`mesh_traffic.jsonl`).

Each line is one real Mesh response with the fields the harness needs:
  prompt_raw       — the user's request text (PII; stays local)
  response_raw     — the answer the served model actually produced (PII; used as the
                     'served' baseline's answer so we don't re-pay to reproduce it)
  model            — the model actually served (short id, e.g. "gpt-5.4-mini")
  feedback_rating  — user feedback ("NULL" | "rejected" | "dislike" | ...)
  turns_in_chat    — conversation depth
  input_tokens / output_tokens — served token counts

This file is GITIGNORED — it is real user data. The loader never prints prompt text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TRAFFIC_PATH = Path(__file__).parent / "mesh_traffic.jsonl"


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TrafficRow:
    """One real Mesh request/response the pipeline evaluates strategies against."""

    response_id: str
    prompt: str
    served_model: str  # short id as logged (normalised against catalog ids later)
    served_answer: str  # the answer actually returned (response_raw)
    feedback_rating: str
    turns_in_chat: int
    input_tokens: int
    output_tokens: int

    @property
    def feedback_is_negative(self) -> bool:
        return self.feedback_rating.strip().lower() in {"rejected", "dislike", "thumbs_down", "bad"}


def load_traffic(path: Path | str = TRAFFIC_PATH) -> list[TrafficRow]:
    """Load `mesh_traffic.jsonl` into TrafficRow records. Raises a clear error if the
    (gitignored) file is missing — it must be provided locally by the operator."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is the gitignored real-traffic input; place the "
            "extracted mesh_traffic.jsonl there before running the pipeline."
        )
    rows: list[TrafficRow] = []
    # JSONL is newline-delimited: split on "\n" ONLY. `str.splitlines()` also splits on
    # Unicode line separators (U+2028 etc.) that appear INSIDE these real prompts, which
    # would shatter a record mid-JSON.
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(
            TrafficRow(
                response_id=str(d.get("response_id", "")),
                prompt=str(d.get("prompt_raw", "")),
                served_model=str(d.get("model", "")),
                served_answer=str(d.get("response_raw", "")),
                feedback_rating=str(d.get("feedback_rating", "NULL")),
                turns_in_chat=_to_int(d.get("turns_in_chat")),
                input_tokens=_to_int(d.get("input_tokens")),
                output_tokens=_to_int(d.get("output_tokens")),
            )
        )
    return rows
