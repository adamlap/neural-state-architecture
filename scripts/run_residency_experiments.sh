#!/usr/bin/env bash
set -euo pipefail

# Repeatable local experiment entrypoint.
#
# Examples:
#   ./scripts/run_residency_experiments.sh smoke
#   ./scripts/run_residency_experiments.sh benchmark 1.5b /path/to/Qwen2.5-1.5B-Instruct

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "${1:-help}" in
  smoke)
    python scripts/residency_smoke_test.py
    ;;
  benchmark)
    MODEL="${2:?model key required: 1.5b or 3b}"
    MODEL_PATH="${3:?local model path required}"
    RUNS="${RUNS:-3}"
    MAX_TOKENS="${MAX_TOKENS:-64}"
    PREFETCH="${PREFETCH:-both}"
    OUTPUT="${OUTPUT:-results/residency/${MODEL}-matrix.json}"

    python -m experiments.residency.benchmark_matrix \
      --model "$MODEL" \
      --model-path "$MODEL_PATH" \
      --runs "$RUNS" \
      --max-tokens "$MAX_TOKENS" \
      --prefetch "$PREFETCH" \
      --output "$OUTPUT"
    ;;
  *)
    cat <<'EOF'
Usage:
  ./scripts/run_residency_experiments.sh smoke
  ./scripts/run_residency_experiments.sh benchmark <1.5b|3b> <local-checkpoint>

Environment:
  RUNS=3
  MAX_TOKENS=64
  PREFETCH=both|on|off
  OUTPUT=results/residency/<model>-matrix.json
EOF
    exit 2
    ;;
esac
