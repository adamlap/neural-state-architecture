#!/usr/bin/env bash
# Run the research benchmarks on a larger model (default qwen2.5:7b) with grids sized for a CPU-only box.
# Usage: scripts/run_7b_suite.sh [model]   (the model must already be pulled: `ollama pull qwen2.5:7b`)
#
# MEMORY: a 7B model needs ~7 GiB resident at num_ctx 8192 (measured 6.86 GB). Use a WSL VM / host with >= 10 GiB, one resident model:
#     OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 ollama serve
# Run nothing else on the Ollama server while this runs. The benchmarks call experiments/research_common/memory_guard.prepare_model(),
# which unloads other models, waits for the unload, and aborts if the estimate would not fit.
# Value names are opaque (--values opaque, the default): "V007_CURRENT"-style names are mis-copied by a 7B (research_common/value_name_probe.py).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-qwen2.5:7b}"
TAG="$(echo "$MODEL" | tr ':/' '--')"
export PYTHONPATH=.
PY="${PYTHON:-.venv/bin/python}"

"$PY" experiments/research_v5/memory_poisoning_benchmark.py --models "$MODEL" --num-ctx 8192 --out "results/research-v5.1-$TAG"
"$PY" experiments/research_v7/calibrated_abstention_benchmark.py --models "$MODEL" --num-ctx 8192 --distractors 10 --out "results/research-v7.1-$TAG"
"$PY" experiments/research_v8/resource_bounded_benchmark.py --models "$MODEL" --seeds 7 17 --num-ctx 2048 8192 \
      --distractors 10 100 --transcript-token-cap 2200 --timeout 1800 --out "results/research-v8.1-$TAG"
# larger samples for the natural-language memory benchmarks
"$PY" experiments/research_v9/natural_memory_benchmark.py --models "$MODEL" --num-ctx 8192 --seeds 7 17 37 47 57 --trials 2 --turns 10 --timeout 900 \
      --out "results/research-v9.2-large-$TAG"
"$PY" -m experiments.research_v10.end_to_end --model "$MODEL" --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --out "results/research-v10.1-end-to-end-large-$TAG.json"
