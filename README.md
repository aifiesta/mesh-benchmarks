# mesh-benchmarks

An index of public benchmark repos that test real LLM behavior routed through [Mesh API](https://meshapi.ai). Each benchmark is one repo, one dataset, one methodology, one question. Reproducible from a fresh clone.

## Why these exist

Public LLM benchmarks usually optimize for what's easy to measure: leaderboard scores on standardized academic tests, raw $/M-token from price cards, single-shot accuracy on contamination-prone problems. The numbers that actually matter to anyone shipping LLM features ($/correct output, $/quality-point, real-world latency, hidden reasoning tokens, tokenizer tax) are not on those leaderboards.

Each repo in this series answers one specific question with full data and reproducible scripts. None of them is the truth of the universe; each is what we found on our prompts with our methodology. If you run them and get different numbers, open an issue with the CSVs.

## The benchmarks

| Repo | Question | Status | Last update |
| --- | --- | --- | --- |
| [mesh-bench-cost-vs-quality](https://github.com/aifiesta/mesh-bench-cost-vs-quality) | What does a real correct answer or quality-point cost across five LLMs? | Pilot (n=5/task). v1 with n=30 in progress. | 2026-05 |
| [router_eval](./router_eval) (in-repo) | How good is Mesh's frozen `model=auto` routing table vs. random / premium / oracle? | Phase 1: offline RouterBench replay skeleton + first result. | 2026-08 |
| _next benchmark slot_ | _coming soon_ | | |

> `router_eval` is the one exception to "one benchmark, one repo, original dataset": it lives in-repo and *replays a fixed public dataset* (RouterBench) offline, because it measures the router's **decision logic**, not any model's quality. Its contamination trade-off is documented in [`router_eval/RESULTS.md`](./router_eval/RESULTS.md).

## Conventions across all benchmarks

Every repo in this series follows the same shape so you can navigate one as easily as another:

- **MIT licensed**, scripts and datasets free to fork.
- **Reproduce in 5 minutes** quickstart in every README.
- **Pilot before scale.** Each benchmark ships a small pilot (typically n=5/task) first; the full run lands when methodology is locked.
- **Honest caveats.** Every result page lists what the benchmark deliberately does NOT cover.
- **Raw CSVs published** alongside aggregated numbers. You can re-aggregate however you want.
- **No emoji, no hype, no obfuscation.** If a number is small or weird, we say so.
- **Datasets are original.** Problems are written from scratch to reduce training-set contamination. The pattern overlap is always honestly flagged.
- **Mesh API is the routing layer**, so the runner, rate-limiting, billing, and observability stay identical across providers. You can plug any Mesh-routable model into the scripts.

## Contributing

Issues on individual benchmark repos welcome. PRs for: additional models, additional task definitions, tokenizer-inflation measurements, judge-bias studies, alternative scoring methods. If you want to add a whole new benchmark to this series, open an issue here describing the question and methodology.

## License

This index repo is MIT. Each benchmark repo carries its own MIT LICENSE.
