"""
MESH-708 Phase 2 — live-run router evaluation pipeline (BUILD ONLY).

Where Phase 1 replays a fixed public dataset with precomputed scores, Phase 2 runs
each routing strategy over the LIVE Mesh catalog on 91 REAL Mesh prompts, calls the
picked models through the Mesh OpenAI-compatible API, LLM-judges the answers, and
plots judged-quality vs cost (incl. the classifier tax) against the model actually
served + its feedback — on the same plane Phase 1 uses.

SAFETY CONTRACT (enforced throughout this package):
  * DEFAULT IS DRY-RUN. No network, no API key, no LLM call. The pipeline runs with
    deterministic MOCK providers so the whole flow — picks, dedupe, cache, judge,
    aggregation, cost estimate — is exercised offline.
  * Every real call (model catalog, classifier, inference, judge) goes through a
    provider that REFUSES to touch the network unless `--live` is set AND
    `MESH_API_KEY` is present. See `mesh_client.LiveCallBlocked`.
  * The 91 prompts and any generated answers are REAL USER DATA (PII). They stay on
    disk locally and are gitignored (mesh_traffic.jsonl, the cache, any run output).
    The only endpoints prompts may reach are the Mesh API and the judge model.

Run (live — the operator does this separately with an operator key):
    MESH_API_KEY=sk-... python -m router_eval.phase2 --live

Dry run (default; safe, offline):
    python -m router_eval.phase2
"""
