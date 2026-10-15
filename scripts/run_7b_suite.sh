#!/usr/bin/env bash
# Run the research benchmarks on a larger model (default qwen2.5:7b) with grids sized for a CPU-only box.
# Usage: scripts/run_7b_suite.sh [model]   (the model must already be pulled: `ollama pull qwen2.5:7b`)
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-qwen2.5:7b}"
TAG="$(echo "$MODEL" | tr ':/' '--')"
export PYTHONPATH=.
PY="${PYTHON:-.venv/bin/python}"

# distractors capped at 250 so transcript prompts stay near 5k tokens (a 7B on CPU is slow at 30k).
"$PY" experiments/research_v5/memory_poisoning_benchmark.py --models "$MODEL" --out "results/research-v5-$TAG"
"$PY" experiments/research_v6/tool_injection_benchmark.py   --models "$MODEL" --out "results/research-v6-$TAG"
"$PY" experiments/research_v7/calibrated_abstention_benchmark.py --models "$MODEL" --out "results/research-v7-$TAG"
"$PY" experiments/research_v4/temporal_semantics_benchmark.py --models "$MODEL" --seeds 7 17 37 --distractors 10 250 \
      --trials 1 --timeout 1800 --out "results/research-v4.2-$TAG"
