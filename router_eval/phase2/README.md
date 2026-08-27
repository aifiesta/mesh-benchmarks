# router_eval/phase2 — live router evaluation pipeline (MESH-708 Phase 2)

**Question Phase 1 can't answer:** on *our* traffic and *current* models, which routing
strategy actually produces the best answers per dollar? Phase 1 replays a fixed, contaminated
public dataset (great for testing the decision logic, useless for answer quality on Mesh).
Phase 2 runs the strategies over the **live Mesh catalog** on **91 real Mesh prompts**, calls
the picked models, **LLM-judges** the answers, and compares judged-quality vs cost against the
**actually-served** model + its feedback.

**This is BUILD-ONLY.** The default is a dry run with mock providers — no network, no key, no
spend. The live pass is run separately by the operator with a real key (the prompts are real
user PII). See [`RESULTS-phase2.md`](RESULTS-phase2.md) for the (placeholder) results and the
deduped call budget.

## Safety / PII

- **Default is dry-run.** `python -m router_eval.phase2` uses deterministic mocks and never
  touches the network. Every real call goes through `MeshClient`, which **refuses** to network
  unless `--live` is set *and* `MESH_API_KEY` is present.
- **The only places prompts are sent are the Mesh API and the judge model.**
- **Gitignored** (never committed): `mesh_traffic.jsonl` (the real prompts + answers), the
  cache `.cache/`, and all run output `out/`. The committed `fixtures/traffic_sample.jsonl` is
  synthetic and PII-free (used by tests + the CI smoke).

## Run

```bash
# Dry run (safe, offline) — full flow with mocks + the live-call estimate:
python -m router_eval.phase2

# Pre-flight budget only (picks + estimate, no answer/judge calls) — dry or live:
python -m router_eval.phase2 --estimate-only

# LIVE — operator only, with a real key (prompts are PII):
MESH_API_KEY=sk-... python -m router_eval.phase2 --live --judge-model anthropic/claude-opus-4-8
```

Flags: `--live`, `--estimate-only`, `--judge-model ID`, `--weight-profile {quality_first,
balanced,cost_first,latency_first}`, `--seed N`, `--max-answer-tokens N`, `--out DIR`,
`--cache-dir DIR`, `--traffic PATH` (defaults to the gitignored `mesh_traffic.jsonl`; point it
at `fixtures/traffic_sample.jsonl` for a PII-free run).

## Pipeline

```
traffic.py     load mesh_traffic.jsonl -> TrafficRow (prompt, served model+answer, feedback, tokens)
catalog.py     live GET /v1/models (+ price merge) | offline sample catalog
routing_data.py v4 SUPERMODE ranking + brand->Mesh-id maps (ported from routersvc origin/main)
classifier.py  category (gpt-4o-mini) + model-select (gemini-3-flash) — Live | deterministic Mock
strategies.py  benchmark/heuristic/weighted/registry + random/cheapest/premium, over the catalog
mesh_client.py THE NETWORK GATE — urllib client; refuses to call unless live + key
answers.py     inference, deduped + disk-cached (served answers pre-seeded) — Live | Mock
judge.py       LLM judge -> 0..1 (correctness+helpfulness+instruction-following) — Live | Mock
cache.py       content-addressed disk cache (dedupe / spend control)
pipeline.py    plan() = picks + estimate; execute() = answers + judge + aggregate
report.py      print the estimate + the per-strategy table; write CSVs
__main__.py    the CLI
```

**Flow:** picks (classify) → dedupe (prompt, model) and subtract already-served pairs → print
the exact live-call budget → answer each unique pair (cached) → judge each unique answer
(cached) → aggregate per strategy vs the served reference → CSVs in `out/`.

## Strategies

Same set as Phase 1, re-homed on the live catalog + a real classifier, **plus `registry`**
(which needs a live classifier and so is stubbed in Phase 1). **No oracle** — a hindsight
oracle would need every model's answer judged per prompt; the served model + feedback is the
ground-truth reference instead. `heuristic`'s fast-lane gate — dead on RouterBench (0 hits) —
fires on 4 of the 91 real prompts.

## Tests

`tests/test_pipeline.py` — offline, no network, no key: traffic parsing, the network gate,
cache/dedupe, the estimate math, mock classifier/judge, and the full dry-run pipeline over the
synthetic sample. Run with `python -m pytest router_eval/phase2/ -q`.
