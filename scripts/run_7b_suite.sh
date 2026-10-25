#!/usr/bin/env bash
# Run the research benchmarks on a larger model (default qwen2.5:7b) with grids sized for a CPU-only box.
# Usage: scripts/run_7b_suite.sh [model]   (the model must already be pulled: `ollama pull qwen2.5:7b`)
#
# MEMORY WARNING: a 7B model (4.7 GB) at a 32k-token context plus other resident models exhausted an 8.9 GB WSL VM
# and froze it. Therefore:
#   * start the server with one resident model and no parallel requests:
#         OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 ollama serve
#   * run nothing else on the Ollama server while this runs;
#   * the benchmarks call experiments/research_common/memory_guard.prepare_model(), which unloads other models and
#     aborts with a message if weights + KV cache + 1.2 GiB margin would not fit in MemAvailable;
#   * contexts are capped at 8192 here (v8 sweeps 2048/8192 only) and long transcripts are capped.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-qwen2.5:7b}"
TAG="$(echo "$MODEL" | tr ':/' '--')"
export PYTHONPATH=.
PY="${PYTHON:-.venv/bin/python}"

"$PY" experiments/research_v5/memory_poisoning_benchmark.py --models "$MODEL" --num-ctx 8192 --out "results/research-v5-$TAG"
"$PY" experiments/research_v6/tool_injection_benchmark.py   --models "$MODEL" --num-ctx 8192 --seeds 7 17 37 --trials 2 --out "results/research-v6-$TAG"
"$PY" experiments/research_v7/calibrated_abstention_benchmark.py --models "$MODEL" --num-ctx 8192 --distractors 10 --out "results/research-v7-$TAG"
"$PY" experiments/research_v9/natural_memory_benchmark.py --models "$MODEL" --num-ctx 8192 --seeds 7 17 --trials 1 --turns 10 --timeout 900 --out "results/research-v9.2-$TAG"
"$PY" experiments/research_v8/resource_bounded_benchmark.py --models "$MODEL" --seeds 7 17 --num-ctx 2048 8192 \
      --distractors 10 100 --transcript-token-cap 2200 --timeout 1800 --out "results/research-v8-$TAG"
