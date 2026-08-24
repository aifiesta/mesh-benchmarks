# router_eval — offline RouterBench replay harness

**Question:** How good is Mesh's frozen `model=auto` routing table (`SUPERMODE_BENCHMARKS`, routing-data v1) at *actually picking the right model*, compared to trivial baselines and to a hindsight oracle?

**Method:** *Offline replay.* We never call a model. We take [`withmartian/routerbench`](https://huggingface.co/datasets/withmartian/routerbench) — ~36k prompts with the **precomputed** correctness score and cost of **11 LLMs** across MMLU, GSM-8K, MBPP, HellaSwag, Winogrande, ARC, MT-Bench (and more) — and *replay a routing policy*: for each item the policy names a model, and we look up that model's **known** score and cost. A whole routing strategy is evaluated in seconds, with zero inference spend.

This is **MESH-708 Phase 1 (initial setup)**: a reproducible skeleton that runs and produces a first answer. Phase 2 (live catalog runs through Mesh) is out of scope here. See [`RESULTS.md`](RESULTS.md) for the numbers and caveats.

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
| `benchmark` | ✅ implemented | Port of the routersvc `benchmark` strategy: `SUPERMODE_BENCHMARKS` brand-ranking lookup. |
| `registry` | ⛔ stub | Needs routersvc registry + classifier runtime (Phase 2). |
| `not_diamond` | ⛔ stub | Needs an external NotDiamond API call (Phase 2). |
| `heuristic` | ⛔ stub | Needs the routersvc heuristic rules ported (Phase 2). |

All eight share one adapter interface (`policies.Policy`): `pick(item, candidates, rng) -> model_id`. Stubs raise `NotImplementedError` so they can never silently emit a bogus number.

## How the `benchmark` port works

`benchmark_table.py` is a **verbatim copy** of `SUPERMODE_BENCHMARKS` from routersvc `app/auto_router/benchmarks.py` (keep the two in lockstep). The frozen table ranks **10 brands** per task category; RouterBench has **11 concrete models**. `routerbench_bridge.py` provides two documented mappings:

1. **brand → RouterBench model.** Only `claude`, `chatgpt`, `mistral` have a RouterBench representative. The other seven ranked brands (`gemini`, `grok`, `deepseek`, `qwen`, `moonshot`, `perplexity`, `bytedance`) have **no** RouterBench model and are skipped when the ranking is walked. Four RouterBench models (WizardLM, both meta/llama, Yi-34B) belong to no ranked brand and can never be chosen by `benchmark`. These gaps are asserted in tests and listed in `RESULTS.md`.
2. **RouterBench `eval_name` → SUPERMODE category.** The granular eval names are bucketed into coarse families, then mapped to one category string. In the live router an LLM classifier does this from the prompt; the replay uses the ground-truth `eval_name` instead — i.e. `benchmark` is measured under **perfect task classification** (an optimistic ceiling — see caveats).

## Layout

```
router_eval/
  benchmark_table.py     # verbatim SUPERMODE_BENCHMARKS port (frozen v1)
  routerbench_bridge.py  # brand↔model + eval_name→category mappings, gap sets, resolver
  data.py                # Item schema + fixture loader (stdlib) + real loader (hf+pandas)
  policies.py            # Policy interface, 4 baselines, benchmark port, 3 stubs
  metrics.py             # quality/cost point, gap-to-oracle, classifier-cost hook, AC3
  replay.py              # CLI: python -m router_eval.replay
  fixtures/routerbench_fixture.jsonl   # 14-item synthetic sample (RouterBench schema)
  results/               # published CSVs (aggregate committed; big real picks.csv gitignored)
  tests/test_replay.py   # smoke + invariant tests
  requirements.txt
```

## License

MIT, like the rest of the series.
