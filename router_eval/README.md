# router_eval — offline RouterBench replay harness

**Question:** How good is Mesh's frozen `model=auto` routing table (`SUPERMODE_BENCHMARKS`, routing-data v1) at *actually picking the right model*, compared to trivial baselines and to a hindsight oracle?

**Method:** *Offline replay.* We never call a model. We take [`withmartian/routerbench`](https://huggingface.co/datasets/withmartian/routerbench) — ~36k prompts with the **precomputed** correctness score and cost of **11 LLMs** across MMLU, GSM-8K, MBPP, HellaSwag, Winogrande, ARC, MT-Bench (and more) — and *replay a routing policy*: for each item the policy names a model, and we look up that model's **known** score and cost. A whole routing strategy is evaluated in seconds, with zero inference spend.

This started as **MESH-708 Phase 1 (initial setup)**: a reproducible skeleton that runs and produces a first answer. **Phase-2 Part A** (this branch) completes the offline coverage — the `heuristic` and `weighted` strategies are un-stubbed and the per-request **classifier tax** is charged — and **Phase-2 Part B** adds the live-run pipeline under [`phase2/`](phase2/) (build-only; every network/LLM call gated behind `--live`). See [`RESULTS.md`](RESULTS.md) for the numbers and caveats, and [`phase2/README.md`](phase2/README.md) for the live pipeline.

> Note on the series convention: the other mesh-benchmarks repos use *original* datasets to fight contamination. This one deliberately does the opposite — it replays a *fixed public* dataset — because the thing under test is the router's **decision logic**, not any model's quality. The contamination trade-off is spelled out in `RESULTS.md`.

## Reproduce in 5 minutes

From a clean checkout, the default run uses a tiny synthetic fixture and needs **no dependencies and no network**:

```bash
# from the repo root (the dir containing router_eval/)
python -m router_eval.replay
```

That prints a quality/cost table, the gap-to-oracle, and the AC3 answer ("does the frozen table beat random?"), and writes CSVs to `router_eval/results/fixture/`.

Run the **real** dataset (downloads ~99 MB on first run, then cached by `huggingface_hub`):

```bash
python -m pip install -r router_eval/requirements.txt
python -m router_eval.replay --source routerbench --shots 0
# quick subset:
python -m router_eval.replay --source routerbench --shots 0 --limit 5000
```

Tests (pure stdlib + pytest, no network):

```bash
python -m pip install pytest
python -m pytest router_eval/tests/ -q
```

## What runs vs. what's stubbed

| Policy | Status | What it does |
| --- | --- | --- |
| `random` | ✅ implemented | Uniform random candidate. The floor. |
| `always_cheapest` | ✅ implemented | Always the single cheapest model (by mean cost). |
| `always_premium` | ✅ implemented | Always the single most-expensive model (a stand-in for "best"). |
| `oracle` | ✅ implemented | Best model per item with hindsight (max score, ties → cheapest). The **ceiling**. |
| `benchmark` | ✅ implemented | Port of the routersvc `benchmark` strategy: `SUPERMODE_BENCHMARKS` brand-ranking lookup. Now charges the classifier tax. |
| `heuristic` | ✅ implemented | Port of the routersvc fast lane: `heuristic_gate.gate` (verbatim `_gate`) routes trivial small-talk to the conversation standard model with no classifier call; else falls through to `benchmark`. (0 fast-lane hits on RouterBench — see RESULTS.md.) |
| `weighted` | ✅ implemented (portable subset) | Port of MESH-644 `weighted`: SUPERMODE rank (Q) + a cost proxy (C), argmax over the category pool. Latency term dropped (RouterBench has none). |
| `registry` | ⛔ stub | Needs a **live** classifier free-selecting over the live catalog (Phase 2 / `phase2/`). |
| `not_diamond` | ⛔ stub | Needs an external NotDiamond API call (out of scope). |

Every policy shares one adapter interface (`policies.Policy`): `pick(item, candidates, rng) -> model_id` plus `classifier_calls(item, picked) -> [classifier_model_id]` (the classifier-tax driver). Stubs raise `NotImplementedError` so they can never silently emit a bogus number.

## How the `benchmark` port works

`benchmark_table.py` is a **verbatim copy** of `SUPERMODE_BENCHMARKS` from routersvc `app/auto_router/benchmarks.py` (keep the two in lockstep). The frozen table ranks **10 brands** per task category; RouterBench has **11 concrete models**. `routerbench_bridge.py` provides two documented mappings:

1. **brand → RouterBench model.** Only `claude`, `chatgpt`, `mistral` have a RouterBench representative. The other seven ranked brands (`gemini`, `grok`, `deepseek`, `qwen`, `moonshot`, `perplexity`, `bytedance`) have **no** RouterBench model and are skipped when the ranking is walked. Four RouterBench models (WizardLM, both meta/llama, Yi-34B) belong to no ranked brand and can never be chosen by `benchmark`. These gaps are asserted in tests and listed in `RESULTS.md`.
2. **RouterBench `eval_name` → SUPERMODE category.** The granular eval names are bucketed into coarse families, then mapped to one category string. In the live router an LLM classifier does this from the prompt; the replay uses the ground-truth `eval_name` instead — i.e. `benchmark` is measured under **perfect task classification** (an optimistic ceiling — see caveats).

## Layout

```
router_eval/
  benchmark_table.py     # verbatim SUPERMODE_BENCHMARKS port (frozen v1)
  heuristic_gate.py      # verbatim port of routersvc heuristic _gate (pure string rules)
  routerbench_bridge.py  # brand↔model + eval_name→category mappings, gap sets, resolvers
  data.py                # Item schema (+prompt) + fixture loader (stdlib) + real loader (hf+pandas)
  policies.py            # Policy interface, 4 baselines, benchmark/heuristic/weighted, 2 stubs
  metrics.py             # quality/cost point, gap-to-oracle, classifier tax (AC2), AC3
  replay.py              # CLI: python -m router_eval.replay
  fixtures/routerbench_fixture.jsonl   # 14-item synthetic sample (RouterBench schema)
  results/               # published CSVs (aggregate committed; big real picks.csv gitignored)
  tests/                 # test_replay.py (smoke/invariants) + test_phase2a.py (heuristic/tax/weighted)
  phase2/                # Phase-2 Part B: live-run pipeline (build-only, --live gated)
  requirements.txt
```

## License

MIT, like the rest of the series.
