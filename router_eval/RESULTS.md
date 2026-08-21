# Results — offline RouterBench replay (MESH-708 Phase 1)

First answer to **AC3: does the frozen `SUPERMODE_BENCHMARKS` table (`benchmark` policy) beat the RANDOM baseline on the replay set?**

## Headline

**YES.** On the real dataset (`withmartian/routerbench`, 0-shot, n=36,497):

- `benchmark` mean score **0.7411** vs `random` mean score **0.5192**
- delta **+0.2219**, a **+42.7%** relative uplift over random

So the frozen routing table is doing something real — it is far from a coin flip. But two honest qualifiers sit right next to that win (see interpretation below): on RouterBench's fixed model set the table does **not** beat the trivial "always use the most expensive model" baseline on quality, and a hindsight oracle leaves large headroom **on both quality and cost**.

## Real dataset — RouterBench 0-shot (n=36,497)

| policy | mean_score | mean_cost ($/req) | gap_to_oracle | pays classifier call? |
| --- | --- | --- | --- | --- |
| random | 0.5192 | 0.000824 | 0.3929 | no |
| always_cheapest | 0.3061 | 0.000046 | 0.6060 | no |
| always_premium | 0.7814 | 0.003293 | 0.1307 | no |
| **benchmark** | **0.7411** | **0.003116** | **0.1710** | **yes\*** |
| oracle | 0.9121 | 0.000242 | — (ceiling) | no |

`always_premium` resolved to `gpt-4-1106-preview`; `always_cheapest` to `mistralai/mistral-7b-chat`. Reproduce with `python -m router_eval.replay --source routerbench --shots 0`. Raw per-item picks are regenerated locally (not committed — ~17 MB); the aggregate is in `results/routerbench_0shot/aggregate.csv`.

\* `benchmark` pays a per-request classifier LLM call in production that this replay does **not** yet charge (`classifier_cost_usd = 0.0`). Its cost column is therefore a **lower bound**. See caveat 4.

## Fixture — synthetic sample (n=14, CI default)

The committed 14-item fixture (`fixtures/routerbench_fixture.jsonl`) exercises the whole pipeline offline. Its numbers are **illustrative only** — hand-authored data, not a measurement:

| policy | mean_score | mean_cost ($/req) | gap_to_oracle |
| --- | --- | --- | --- |
| random | 0.4464 | 0.000830 | 0.5000 |
| always_cheapest | 0.3393 | 0.000052 | 0.6071 |
| always_premium | 0.8571 | 0.004502 | 0.0893 |
| benchmark | 0.8214 | 0.004245 | 0.1250 |
| oracle | 0.9464 | 0.003341 | — |

AC3 on the fixture: benchmark 0.8214 vs random 0.4464 → **YES**, +84.0%. (The fixture is tuned to resemble the real per-model means; treat only the *real* table above as the result.)

## Interpretation

- **The table beats random, clearly.** +42.7% is not noise — the ranking encodes real signal about which of {claude, gpt, mistral} to send a task to.
- **But it does not beat "always gpt-4" on quality** (0.741 vs 0.781) and saves only ~5% on cost. On RouterBench's *fixed, early-2024* model set, the frozen table's main effect is to sometimes route to `claude-v2` or `mixtral` instead of `gpt-4` — occasionally cheaper, occasionally worse. Whether that trade is good depends on price sensitivity the replay does not weight.
- **The headroom is large and cost-shaped.** The oracle reaches 0.912 at **$0.000242/req** — higher quality than `always_premium` at ~14x lower cost — because on most RouterBench items a *cheap* model is also correct. A smarter, cost-aware router has a lot to capture. (This mirrors RouterBench's own cheapest-correct `oracle_model_to_route_to`, which is dominated by cheap models.)
- **Caveat-laden by construction.** The `benchmark` number here is an *optimistic ceiling* for that strategy — perfect task classification, no classifier cost. Read it as "how good is the ranking *when the classifier is right and free*", not "how good is the live benchmark router". Closing those gaps is Phase 2.

## Coverage gaps (what this replay can and cannot see)

**Ranked brands with no RouterBench model** (the table ranks them, the replay can't test them): `gemini`, `grok`, `deepseek`, `qwen`, `moonshot`, `perplexity`, `bytedance`. Only `claude`, `chatgpt`, `mistral` (3 of 10) have a RouterBench representative — so on RouterBench, `benchmark` effectively chooses among `claude-v2` / `gpt-4-1106-preview` / `mixtral-8x7b`. The table's ability to discriminate among the other seven brands is **untested here**.

**RouterBench models no ranked brand maps to** (every policy but `benchmark` can pick them; `benchmark` never will): `WizardLM/WizardLM-13B-V1.2`, `meta/code-llama-instruct-34b-chat`, `meta/llama-2-70b-chat`, `zero-one-ai/Yi-34B-Chat`.

## Honest caveats

1. **Fixed, dated model set.** RouterBench's 11 models are early-2024 (gpt-4-1106, claude-v2, mixtral-8x7b, …). They do **not** match Mesh's ~1000-model production catalog or the model *versions* the frozen table's brand maps point at (e.g. `openai/gpt-5.4`, `anthropic/claude-sonnet-4.6`). **Phase 1 tests the routing DECISION LOGIC, not our catalog.** Absolute scores here say nothing about how those decisions play out on current models.
2. **Contamination.** MMLU / HellaSwag / GSM-8K / ARC are widely present in pre-training data. This largely **cancels for cross-strategy comparison** (every policy sees the same contaminated items, so the *ranking* of policies is fairer than any absolute number) but is fatal for **any absolute per-model quality claim**. Do not quote a model's RouterBench score as its quality.
3. **Perfect-classification optimism.** In production the `benchmark` strategy runs an LLM classifier over the *prompt* to guess the task category, and it can be wrong. The replay instead uses RouterBench's ground-truth `eval_name`. So `benchmark`'s number is an **upper bound** — real routing eats classifier errors this harness does not model.
4. **Classifier cost/latency not charged.** `benchmark` and `registry` pay an extra per-request classifier LLM call; `not_diamond` makes an external router call but **no internal classifier LLM call**; `heuristic` is rule-based. The replay sets `classifier_cost_usd = 0.0` for all (see the TODO hook in `metrics.py`). Until it is wired from real telemetry, benchmark/registry costs are lower bounds and the cost axis is **not** apples-to-apples with not_diamond/heuristic.
5. **eval_name → category is a hand-made bridge.** A deliberate, reviewable mapping (`routerbench_bridge.FAMILY_TO_CATEGORY`), not ground truth. The `other` bucket (RouterBench's Chinese-language + misc evals) and the conversational families fall back to a general-reasoning category, which flatters or penalizes brands unevenly.
6. **Oracle is quality-max (ties → cheapest).** A quality ceiling, not RouterBench's own cost-first `oracle_model_to_route_to`. A different oracle definition gives a different ceiling.
7. **"Premium" = most expensive** by mean cost (a proxy for "best"), and **"cheapest" = least expensive**, both derived from the data at fit time.
8. **Scores are RouterBench performance scores in [0,1]**, fractional for judge-scored tasks (MT-Bench). `mean_score` averages them; it is not pure accuracy for those tasks.
9. **0-shot reported; 5-shot available** via `--shots 5` (larger download, not run here).
10. **Seeded.** `random` and `benchmark` tie-breaks use seed `20260821`; another seed moves those two by a hair (not the deterministic policies).

## What Phase 1 deliberately does NOT cover

- No live inference and no Mesh routing (Phase 2).
- No `registry` / `not_diamond` / `heuristic` measurement — stubbed behind the shared interface.
- No latency axis (RouterBench has no comparable latency field), and no classifier cost/latency accounting yet.
- No statistical significance testing or per-task breakdown — this is the initial-setup skeleton, not the full readout.
