#!/usr/bin/env bash
set -euo pipefail

# Repeatable local experiment entrypoint.
# If no checkpoint path is supplied, the benchmark downloads the selected
# Hugging Face model into the normal local cache.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"
PY="${PYTHON:-python3}"

case "${1:-help}" in
  smoke)
    "$PY" scripts/residency_smoke_test.py
    ;;
  benchmark)
    MODEL="${2:?model key required: 1.5b or 3b}"
    MODEL_PATH="${3:-}"
    RUNS="${RUNS:-3}"
    MAX_TOKENS="${MAX_TOKENS:-64}"
    PREFETCH="${PREFETCH:-both}"
    EXECUTION_MODES="${EXECUTION_MODES:-selective}"
    LOOKAHEADS="${LOOKAHEADS:-2}"
    CACHE_MODES="${CACHE_MODES:-cold}"
    BUDGETS="${BUDGETS:-}"
    LEARNED_PREDICTOR="${LEARNED_PREDICTOR:-0}"
    ADAPTIVE_LOOKAHEAD="${ADAPTIVE_LOOKAHEAD:-0}"
    ARGS=(--model "$MODEL" --runs "$RUNS" --max-tokens "$MAX_TOKENS" --prefetch "$PREFETCH" \
          --execution-modes "$EXECUTION_MODES" --lookaheads "$LOOKAHEADS" --cache-modes "$CACHE_MODES")
    if [ -n "$BUDGETS" ]; then ARGS+=(--budgets "$BUDGETS"); fi
    if [ "$LEARNED_PREDICTOR" = "1" ]; then ARGS+=(--learned-predictor); fi
    if [ "$ADAPTIVE_LOOKAHEAD" = "1" ]; then ARGS+=(--adaptive-lookahead); fi
    if [ "${COLD_CACHE:-1}" = "0" ]; then ARGS+=(--no-cold-cache); fi
    if [ -n "$MODEL_PATH" ]; then ARGS+=(--model-path "$MODEL_PATH"); fi
    "$PY" -m experiments.residency.benchmark_matrix "${ARGS[@]}"
    ;;
  *)
    cat <<'EOF'
Usage:
  ./scripts/run_residency_experiments.sh smoke
  ./scripts/run_residency_experiments.sh benchmark <1.5b|3b> [local-checkpoint]

If local-checkpoint is omitted, the selected Qwen checkpoint is downloaded
automatically into the Hugging Face cache.

Environment:
  RUNS=3
  MAX_TOKENS=64
  PREFETCH=both|on|off
  COLD_CACHE=1          (0 keeps the OS page cache warm; legacy single-mode switch)
  EXECUTION_MODES=selective   (resident|selective|disk, comma-separated)
  LOOKAHEADS=2            (comma-separated, e.g. 1,2,4)
  CACHE_MODES=cold        (cold|warm, comma-separated)
  BUDGETS=                (VRAM:RAM pairs in GB, e.g. 2:4,4:8)
  LEARNED_PREDICTOR=0     (1 enables the online state-aware predictor)
  ADAPTIVE_LOOKAHEAD=0     (1 adapts speculative prefetch depth from telemetry)
  PYTHON=<interpreter>  (default: python3)
EOF
    exit 2
    ;;
esac
