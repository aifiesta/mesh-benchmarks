# Results — offline RouterBench replay (MESH-708)

First answer to **AC3: does the frozen `SUPERMODE_BENCHMARKS` table (`benchmark` policy) beat the RANDOM baseline on the replay set?**

## Headline

**YES.** On the real dataset (`withmartian/routerbench`, 0-shot, n=36,497):

- `benchmark` mean score **0.7411** vs `random` mean score **0.5192**
- delta **+0.2219**, a **+42.7%** relative uplift over random

So the frozen routing table is doing something real — it is far from a coin flip. But two honest qualifiers sit right next to that win (see interpretation): on RouterBench's fixed model set the table does **not** beat the trivial "always use the most expensive model" baseline on quality, and a hindsight oracle leaves large headroom **on both quality and cost**.

**Phase-2 Part-A update (this branch):** the offline coverage is now complete for every strategy that *can* run without a live call — `heuristic` and `weighted` are un-stubbed and measured, and the **classifier tax (AC2)** is charged, so `benchmark`'s cost is no longer a lower bound.

## Real dataset — RouterBench 0-shot (n=36,497)

| policy | mean_score | infer_cost ($/req) | classifier_tax ($/req) | cost+tax ($/req) | gap_to_oracle | classifier |
| --- | --- | --- | --- | --- | --- | --- |
| random | 0.5192 | 0.000824 | 0.0 | 0.000824 | 0.3929 | — |
| always_cheapest | 0.3061 | 0.000046 | 0.0 | 0.000046 | 0.6060 | — |
| always_premium | 0.7814 | 0.003293 | 0.0 | 0.003293 | 0.1307 | — |
| **benchmark** | **0.7411** | **0.003116** | **0.000151** | **0.003267** | **0.1710** | gpt-4o-mini |
| heuristic | 0.7411 | 0.003116 | 0.000151 | 0.003267 | 0.1710 | gpt-4o-mini (miss→bench) |
| weighted | 0.6999 | 0.003066 | 0.000151 | 0.003217 | 0.2121 | gpt-4o-mini |
| oracle | 0.9121 | 0.000242 | 0.0 | 0.000242 | — (ceiling) | — |

`always_premium` resolved to `gpt-4-1106-preview`; `always_cheapest` to `mistralai/mistral-7b-chat`. Reproduce with `python -m router_eval.replay --source routerbench --shots 0` (needs `pip install -r router_eval/requirements.txt` + the dataset). Raw per-item picks are regenerated locally (not committed — ~27 MB); the aggregate is in `results/routerbench_0shot/aggregate.csv`. Classifier latency is a flat **1300 ms** routing overhead per classified request (separate from inference; see caveat 4).

### The classifier tax, corrected (AC2)

`benchmark`'s all-in cost is now **$0.003267/req = $0.003116 inference + $0.000151 classifier** — a **+4.8%** surcharge over inference alone, no longer `0.0`. The tax is `input_tokens × prompt_price + output_tokens × completion_price` of the classifier model:

- **benchmark / weighted** → `openai/gpt-4o-mini` ($0.15 / $0.60 per 1M). Input ≈ 750 template tokens (system + the 48-category block ≈ 571 tok + wrapper) + the prompt (capped at the 2000-char / ~500-token truncation) → ~$0.000151/req on this set.
- **registry** → `google/gemini-3-flash-preview` ($0.50 / $3.00 per 1M) — pricier per token, so a **higher** tax for the same prompt; charged in Phase 2 (registry can't run offline).
- **heuristic** → pays **nothing on a fast-lane hit**, and benchmark's classifier only on a miss. On RouterBench every prompt misses (below), so its tax equals benchmark's here; on real conversational traffic it will be lower.

Token counts and prices are documented, reviewable constants in `metrics.py` (`BENCHMARK_CLASSIFIER` / `REGISTRY_CLASSIFIER`) — swap them for measured classifier telemetry when available.

## Fixture — synthetic sample (n=14, CI default)

The committed 14-item fixture (`fixtures/routerbench_fixture.jsonl`) exercises the whole pipeline offline. Its numbers are **illustrative only** — hand-authored data, not a measurement:

| policy | mean_score | infer_cost ($/req) | classifier_tax ($/req) | cost+tax ($/req) | gap_to_oracle |
| --- | --- | --- | --- | --- | --- |
| random | 0.4464 | 0.000830 | 0.0 | 0.000830 | 0.5000 |
| always_cheapest | 0.3393 | 0.000052 | 0.0 | 0.000052 | 0.6071 |
| always_premium | 0.8571 | 0.004502 | 0.0 | 0.004502 | 0.0893 |
| benchmark | 0.8214 | 0.004245 | 0.000124 | 0.004369 | 0.1250 |
| heuristic | 0.8214 | 0.004245 | 0.000124 | 0.004369 | 0.1250 |
| weighted | 0.7679 | 0.003345 | 0.000124 | 0.003469 | 0.1786 |
| oracle | 0.9464 | 0.003341 | 0.0 | 0.003341 | — |

AC3 on the fixture: benchmark 0.8214 vs random 0.4464 → **YES**, +84.0%. (The fixture is tuned to resemble the real per-model means; treat only the *real* table above as the result.)

## What runs vs. what's stubbed (updated)

| Policy | Status | What it does |
| --- | --- | --- |
| `random` / `always_cheapest` / `always_premium` / `oracle` | ✅ | The four baselines (floor, two fixed models, hindsight ceiling). |
| `benchmark` | ✅ | Port of the routersvc `benchmark` brand-ranking lookup, now with the classifier tax charged. |
| `heuristic` | ✅ **new** | Port of the routersvc fast lane: string-gate the prompt (`heuristic_gate.py`, a verbatim `_gate` port) → route trivial small-talk to the conversation category's standard model with **no** classifier call; anything else falls through to `benchmark`. |
| `weighted` | ✅ **new (portable subset)** | Port of MESH-644 `weighted`: SUPERMODE rank (Q) + a cost proxy (C), argmax over the category pool. Latency term dropped — RouterBench has none (caveat 10). |
| `registry` | ⛔ stub | Needs a **live** classifier free-selecting over the live catalog — no portable table. Measured in Phase 2 (`phase2/`). |
| `not_diamond` | ⛔ stub | Needs an external NotDiamond API call. Out of scope for both phases here. |

## Interpretation

- **The table beats random, clearly.** +42.7% is not noise — the ranking encodes real signal about which of {claude, gpt, mistral} to send a task to.
- **But it does not beat "always gpt-4" on quality** (0.741 vs 0.781) and saves only ~5% on cost. On RouterBench's *fixed, early-2024* model set, the frozen table's main effect is to sometimes route to `claude-v2` or `mixtral` instead of `gpt-4` — occasionally cheaper, occasionally worse.
- **`heuristic` ≡ `benchmark` on RouterBench — by construction, and it's an honest null.** The fast-lane gate accepts **0 of 36,497** RouterBench prompts (35,801 decline as `too_long` at >140 chars; the rest as `task_verb` / `not_conversational`). RouterBench is *all* benchmark tasks — there is no trivial small-talk for the fast lane to catch — so `heuristic` deterministically falls through to `benchmark` on every item and inherits its exact numbers. The port is real (verified on conversational unit fixtures and exercised in Phase 2 on live Mesh traffic); RouterBench simply cannot test it. This is the clearest example of a rule that **cannot be exercised offline on this dataset**.
- **`weighted` trades ~4 quality points for a slightly cheaper bill.** 0.6999 vs benchmark's 0.7411 at $0.003066 vs $0.003116 inference. The `balanced` profile's cost term pulls picks toward cheaper members of each category pool; with the latency term dropped it is effectively a 0.57·quality + 0.43·cost blend. Whether that trade is good depends on price sensitivity the replay does not weight — and note the cost signal here is a **proxy** (caveat 10).
- **The headroom is large and cost-shaped.** The oracle reaches 0.912 at **$0.000242/req** — higher quality than `always_premium` at ~14x lower cost — because on most RouterBench items a *cheap* model is also correct. A smarter, cost-aware router has a lot to capture.
- **Caveat-laden by construction.** The `benchmark`/`weighted` numbers are *optimistic ceilings* for those strategies — perfect task classification (ground-truth `eval_name`, not a live classifier reading the prompt). Read them as "how good is the ranking *when the classifier is right*", not "how good is the live router". Closing the classification-error gap is Phase 2.

## Coverage gaps (what this replay can and cannot see)

**Ranked brands with no RouterBench model** (the table ranks them, the replay can't test them): `gemini`, `grok`, `deepseek`, `qwen`, `moonshot`, `perplexity`, `bytedance`. Only `claude`, `chatgpt`, `mistral` (3 of 10) have a RouterBench representative — so on RouterBench, `benchmark`/`weighted` effectively choose among `claude-v2` / `gpt-4-1106-preview` / `mixtral-8x7b`. The table's ability to discriminate among the other seven brands is **untested here**.

**RouterBench models no ranked brand maps to** (every baseline can pick them; `benchmark`/`weighted`/`heuristic` never will): `WizardLM/WizardLM-13B-V1.2`, `meta/code-llama-instruct-34b-chat`, `meta/llama-2-70b-chat`, `zero-one-ai/Yi-34B-Chat`.

## Honest caveats

1. **Fixed, dated model set.** RouterBench's 11 models are early-2024 (gpt-4-1106, claude-v2, mixtral-8x7b, …). They do **not** match Mesh's ~1000-model production catalog or the model *versions* the frozen table's brand maps point at (e.g. `openai/gpt-5.4`, `anthropic/claude-sonnet-4.6`). **This phase tests the routing DECISION LOGIC, not our catalog.** Absolute scores here say nothing about how those decisions play out on current models.
2. **Contamination.** MMLU / HellaSwag / GSM-8K / ARC are widely present in pre-training data. This largely **cancels for cross-strategy comparison** (every policy sees the same contaminated items, so the *ranking* of policies is fairer than any absolute number) but is fatal for **any absolute per-model quality claim**. Do not quote a model's RouterBench score as its quality.
3. **Perfect-classification optimism.** In production `benchmark`/`weighted` run an LLM classifier over the *prompt* to guess the task category, and it can be wrong. The replay instead uses RouterBench's ground-truth `eval_name`. So their numbers are **upper bounds** — real routing eats classifier errors this harness does not model.
4. **Classifier tax is a MODEL, not telemetry (AC2).** The per-request classifier cost/latency now charged (`metrics.py`) uses documented token/price constants (template tokens grounded in the real routing prompts; prices from the Mesh catalog; the prompt-token estimate is chars/4 capped at the 2000-char truncation). It is a realistic estimate, not measured usage — swap in classifier telemetry when available. `not_diamond` (were it run) makes an external router call with **no** internal classifier LLM call, a different profile again.
5. **eval_name → category is a hand-made bridge.** A deliberate, reviewable mapping (`routerbench_bridge.FAMILY_TO_CATEGORY`), not ground truth. The `other` bucket and the conversational families fall back to a general-reasoning category, which flatters or penalizes brands unevenly.
6. **Oracle is quality-max (ties → cheapest).** A quality ceiling, not RouterBench's own cost-first `oracle_model_to_route_to`. A different oracle definition gives a different ceiling.
7. **"Premium" = most expensive** by mean cost (a proxy for "best"), and **"cheapest" = least expensive**, both derived from the data at fit time.
8. **Scores are RouterBench performance scores in [0,1]**, fractional for judge-scored tasks (MT-Bench). `mean_score` averages them; it is not pure accuracy for those tasks.
9. **0-shot reported; 5-shot available** via `--shots 5` (larger download, not run here).
10. **`weighted`'s cost signal is a PROXY and its latency signal is ABSENT.** Production `weighted` blends the model's $/1M prompt+completion **price**; RouterBench exposes only per-response $ cost, so the replay uses each model's **mean per-request cost** across the set (log10 + min-max inverted) — directionally the same, but it folds in each model's verbosity. The production **latency** term has no RouterBench analogue and is dropped (the source's own "missing signal → drop the term and renormalize" path). So offline `weighted` = quality + cost only.
11. **`heuristic` is untested on RouterBench** (0 fast-lane accepts, above). Its behaviour is verified on conversational unit fixtures and measured on real traffic in Phase 2.
12. **Seeded.** `random` and `benchmark`/`heuristic` tie-breaks use seed `20260821`; another seed moves those by a hair (not the deterministic policies).

## What is deferred to Phase 2 (`router_eval/phase2/`)

- **Live inference + LLM-judged quality** on real Mesh traffic — the number this offline replay cannot produce (RouterBench's scores are precomputed and contaminated; Phase 2 judges fresh answers).
- **`registry`** measurement (needs the live classifier over the live catalog).
- **Live catalog + real classifier calls** — offline uses ground-truth `eval_name`; Phase 2 classifies the prompt for real (gated behind `--live`).
- No `not_diamond` (external dependency), and no statistical-significance / per-task breakdown here.
