#!/usr/bin/env bash
# Nightly routing-quality sample (MESH-708 follow-on).
#
# WHY THIS EXISTS: production carries no per-request quality signal — quality only
# exists when a judge scores an answer. Without this, the "quality" panel on the
# daily dashboard can only show model MIX, not whether answers got better or worse.
# A small nightly judged sample gives a real trend line, and rebuilds the per-model
# scores that a previous run's cache overwrite destroyed.
#
# Cost: ~30 prompts x ~4 strategies, deduped + cached ≈ $1-3/night on a Sonnet judge.
# Opus is deliberately NOT used: it rejects the judge request shape and runs at
# 1-2 calls/min through the gateway, which cannot finish a nightly window.
set -euo pipefail

cd "$(dirname "$0")/../.."                       # repo root
# Under cron there is no interactive env, so source the operator key file if needed.
if [ -z "${MESH_API_KEY:-}" ] && [ -f "$HOME/creds/.env" ]; then
  # shellcheck disable=SC1091
  set -a; . "$HOME/creds/.env"; set +a
fi
: "${MESH_API_KEY:?set MESH_API_KEY, or put it in ~/creds/.env}"
STAMP="$(date -u +%Y%m%d)"
N="${NIGHTLY_SAMPLE:-30}"
JUDGE="${NIGHTLY_JUDGE:-anthropic/claude-sonnet-4-6}"
OUT="router_eval/nightly/results/$STAMP"
mkdir -p "$OUT"

# Run-scoped cache. THE LESSON FROM THE 692 RUN: a shared cache dir let later runs
# overwrite the answers behind already-published numbers, making them unreproducible.
# Never point two runs at one cache again.
CACHE="router_eval/nightly/.cache/$STAMP"
mkdir -p "$CACHE"

# Stable nightly subsample of the traffic corpus, seeded by date so each night differs
# but any night is reproducible.
python3 - "$N" "$STAMP" <<'PY'
import json, random, sys
n, stamp = int(sys.argv[1]), sys.argv[2]
rows = [json.loads(l) for l in open("router_eval/phase2/mesh_traffic.jsonl")]
random.Random(int(stamp)).shuffle(rows)
with open("router_eval/nightly/sample.jsonl", "w") as f:
    for r in rows[:n]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"sampled {min(n,len(rows))} of {len(rows)} prompts (seed={stamp})")
PY

PHASE2_REAL_ONLY=1 python3 -m router_eval.phase2 \
  --live --judge-model "$JUDGE" \
  --traffic router_eval/nightly/sample.jsonl \
  --cache-dir "$CACHE" --out "$OUT" 2>&1 | tail -20

echo "results -> $OUT/strategy_aggregate.csv"
