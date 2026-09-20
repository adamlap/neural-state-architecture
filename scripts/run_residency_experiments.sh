#!/usr/bin/env bash
set -euo pipefail

# Repeatable local experiment entrypoint.
# If no checkpoint path is supplied, the benchmark downloads the selected
# Hugging Face model into the normal local cache.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "${1:-help}" in
  smoke)
    python scripts/residency_smoke_test.py
    ;;
  benchmark)
    MODEL="${2:?model key required: 1.5b or 3b}"
    MODEL_PATH="${3:-}"
    RUNS="${RUNS:-3}"
    MAX_TOKENS="${MAX_TOKENS:-64}"
    PREFETCH="${PREFETCH:-both}"
    ARGS=(--model "$MODEL" --runs "$RUNS" --max-tokens "$MAX_TOKENS" --prefetch "$PREFETCH")
    if [ -n "$MODEL_PATH" ]; then ARGS+=(--model-path "$MODEL_PATH"); fi
    python -m experiments.residency.benchmark_matrix "${ARGS[@]}"
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
EOF
    exit 2
    ;;
esac
