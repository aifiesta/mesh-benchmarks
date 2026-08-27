"""
Heuristic fast-lane accept gate — a VERBATIM port of the pure-string logic in
routersvc `app/auto_router/heuristic_strategy.py::_gate` (origin/main).

The production `heuristic` strategy is a rule-based fast lane in front of the
classifier: a trivial conversational prompt (a greeting or a one-line factual
question) is routed straight to the active data version's General-Conversation
*standard-tier* model with zero LLM calls, skipping the classify round-trip that
dominates /v1/router/select latency. Any hint of task work, recency, code,
links, digits, or length makes the gate DECLINE, and the request falls through
to the full benchmark path.

Only the `_gate` predicate is portable with zero dependencies — it reads only
the prompt string. It is copied here byte-for-byte (constants included) so both
the offline RouterBench replay (policies.HeuristicPolicy) and the Phase 2 live
pipeline exercise the exact gate production runs. Keep it in lockstep with the
routersvc source if that gate ever changes.

What is NOT ported here (documented, lives in the caller):
  * the "route to the conversation category's standard-tier model" step — that
    resolves a brand to a concrete model id, which is universe-specific
    (RouterBench's fixed 11 vs the live Mesh catalog), so each caller supplies
    its own brand->model resolution;
  * the async version lookup / candidate-membership fail-soft / structured
    logging — runtime plumbing with no bearing on the decision.
"""

from __future__ import annotations

import re

_MAX_LEN = 140
_MAX_DIGITS = 6

# Recency/volatility markers — these prompts need fresh data or premium routing,
# never the fast lane. Matched as case-insensitive substrings (over-declining is
# harmless: the request just takes the normal benchmark path). "now" is matched
# on word boundaries separately — as a substring it would decline the whole
# "know"/"knowledge" family, which is exactly the traffic the fast lane targets.
_RECENCY_TERMS = (
    "latest", "today", "news", "current", "price",
    "stock", "weather", "score",
)
_NOW_RE = re.compile(r"\bnow\b")
# Years 2020–2099: date-anchored questions need fresh/contextual routing.
_YEAR_RE = re.compile(r"20[2-9][0-9]")

# Task-work verb stems — anything that asks the model to DO work is not trivial.
_TASK_VERBS = (
    "write", "build", "create", "generate", "code", "debug", "fix", "plan",
    "analy", "compare", "translate", "summar", "essay", "report", "design",
    "implement",
)

_SMALL_TALK_RE = re.compile(r"\b(hi|hello|hey|thanks|how are|what is|who is|tell me)\b")

# The SUPERMODE category the heuristic fast lane routes accepted prompts through.
CONVERSATION_CATEGORY = "General reasoning / Q&A - General Conversation, Chatting"


def gate(text: str) -> tuple[bool, str]:
    """Pure-string accept gate. Returns (matched, reason) — must stay cheap.

    Verbatim port of routersvc `heuristic_strategy._gate`. `matched=True` means
    the prompt is trivial-conversational (fast-lane eligible); otherwise `reason`
    names why it declined and the request should fall through to benchmark.
    """
    if not text:
        return False, "empty"
    if len(text) > _MAX_LEN:
        return False, "too_long"
    if "\n" in text:
        return False, "multiline"
    lower = text.lower()
    if "```" in text or "http" in lower:
        return False, "code_or_link"
    if sum(ch.isdigit() for ch in text) > _MAX_DIGITS:
        return False, "digits_dense"
    if (
        _YEAR_RE.search(lower)
        or _NOW_RE.search(lower)
        or any(term in lower for term in _RECENCY_TERMS)
    ):
        return False, "recency_term"
    if any(verb in lower for verb in _TASK_VERBS):
        return False, "task_verb"
    if not (lower.endswith("?") or _SMALL_TALK_RE.search(lower)):
        return False, "not_conversational"
    return True, "trivial_conversational"
