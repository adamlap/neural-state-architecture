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
    ARGS=(--model "$MODEL" --runs "$RUNS" --max-tokens "$MAX_TOKENS" --prefetch "$PREFETCH")
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
  COLD_CACHE=1          (0 keeps the OS page cache warm)
  PYTHON=<interpreter>  (default: python3)
EOF
    exit 2
    ;;
esac
