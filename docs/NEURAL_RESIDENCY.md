# NSA Neural Residency Architecture

NSA has an experimental neural virtual-memory subsystem. Model weights are addressable as *regions* (one per decoder layer today) and their physical placement is tracked separately from the cognitive/control state, so a checkpoint larger than the fast-memory budget can run with most layers on NVMe.

The first target models are the local Qwen2.5 1.5B and 3B checkpoints already used by the NSA experiments.

> **Status: experimental.** Placement and disk offload are performed by Hugging Face Accelerate. NSA adds region-level bookkeeping, transition prediction, an NVMe page-cache prefetcher and telemetry. Measured on real Qwen2.5 checkpoints (see `docs/ACTIVE_RESIDENCY.md`): prefetch overhead on a small model that fits comfortably in the page cache has been eliminated (0.94x, statistically neutral); on a real disk-bound model it is correct (100% hit rate, identical outputs) but not yet reliably beneficial (0.99x median, high run-to-run variance from CPU/disk contention with the main decode thread). Run `make residency-benchmark` and read the `summary.notes` it prints for your own hardware.

## Selective storage vs selective computation

The residency layer answers: *which weights should physically exist in the fast memory tier now?* The model layer answers: *which weights should be executed?* These are deliberately separate decisions. Residency can change where a neural region lives; it cannot change NSA authority, policy, or safety decisions.

## What is implemented

| Piece | Module | Notes |
|---|---|---|
| Region model, tiers, events | `nsa.residency.types` | `MemoryTier` (VRAM/RAM/NVMe), `NeuralRegion`, `ResidencyEvent` |
| Byte-accounted caches | `nsa.residency.cache`, `manager` | LRU per tier; a region larger than a tier's whole budget is rejected, never flushes the cache |
| Policy | `nsa.residency.policy` | score from tag relevance (when known), predicted next-use probability, dependency probability, minus size/latency penalties |
| Predictors | `nsa.residency.predictor`, `learned` | count-based transition + tag co-occurrence models; the online one is fitted from execution telemetry (it is not a neural model) |
| Backend | `nsa.runtime.inference.resident_transformers` | `SelectiveStorageTransformersBackend`: empty-weights skeleton + `load_checkpoint_and_dispatch` with an explicit VRAM/RAM/disk device map |
| Prefetch | `nsa.residency.accelerate_prefetch`, `controller` | warms the OS page cache for the *next* decoder layer's offloaded bytes while the current layer runs |
| Sizing | `nsa.residency.sizing` | region sizes are read from the checkpoint's safetensors headers, with a config-based fallback |
| Telemetry | `nsa.residency.trace` | bounded, thread-safe events with optional JSONL persistence and honest hit-rate metrics |

The remaining deliberate boundary is direct control of Accelerate's parameter lifecycle: Accelerate remains the owner of actual parameter materialization. NSA controls prediction, admission/planning, execution feedback and page-cache warming. This avoids claiming that a page-cache read is equivalent to physical VRAM residency.

The runtime now supports three storage modes:
* **resident** — all decoder layers are placed on the fast execution device; this is the in-memory control condition.
* **selective** — edge layers are admitted to the fast device and adjacent layers to host RAM subject to the configured byte budgets; the remainder is disk-backed.
* **disk** — all decoder layers are disk-backed; this is the strongest offload condition.

State-aware execution feedback is implemented through `NeuralResidencyManager.set_active_state()` and `record_execution()`. The optional `OnlineResidencyPredictor` learns transition and cognitive-tag co-occurrence evidence online without changing authority or safety policy. MoE expert/router regions are already architecture-derived and supported by the residency planner.

Multi-step lookahead is implemented by the active controller's bounded worker pool and benchmark grid; the benchmark treats lookahead as an experimental variable rather than assuming that a larger value is beneficial.

## Running

Install the optional ML stack and run the deterministic smoke checks first:

```bash
pip install "neural-state-architecture[ml-residency]"
make residency-smoke
```

For real hardware, the benchmark can now evaluate the complete matrix rather than one narrow on/off comparison:

```bash
python experiments/residency/benchmark_matrix.py \
  --model 1.5b \
  --execution-modes resident,selective,disk \
  --prefetch both \
  --cache-modes cold,warm \
  --lookaheads 1,2,4 \
  --budgets 2:4,4:8 \
  --runs 3 \
  --max-tokens 64
```

Use `--learned-predictor` to replace the heuristic predictor with the online state-aware predictor. The benchmark reports decode latency, tokens/s, page-cache bytes warmed, hit/coverage/lead-time telemetry, slowest regions, output equivalence, execution mode, cache state and budget. It does not claim a performance improvement until the measured runs establish one.

The three storage modes deliberately provide matched controls:
* `resident` is the no-offload baseline and does not run the prefetch condition.
* `disk + prefetch=off` isolates Accelerate disk offload.
* `disk + prefetch=on` measures NSA's page-cache intervention on the same disk-backed model.
* `selective` measures the intended tiered policy under explicit VRAM/RAM budgets.

The ML extras require `torch>=2.5`. Cached mode never downloads weights; the benchmark downloads only when no checkpoint path is given.

The ML extras require `torch>=2.5` (transformers 5 refuses to use older torch); the backend raises a clear error if the installed torch is too old. Cached mode never downloads weights; the benchmark downloads only when no checkpoint path is given.

## Development phases

1. Foundation - region model, policy, predictor, cache, telemetry. **Done.**
2. Disk residency - empty-model + disk-backed checkpoint execution. **Done.**
3. Active residency - predictive page-cache prefetch around decoder regions. **Done and instrumented.**
4. State coupling - explicit cognitive-state propagation into planning and online execution feedback. **Done.**
5. Learned residency - online transition/tag predictor with deterministic regression coverage. **Done.**
6. MoE specialization - architecture-derived expert/router regions and device-map support. **Done.**
7. Evaluation - controlled resident/selective/disk modes, cold/warm cache, lookahead and VRAM/RAM budget grids, per-region latency and prefetch overlap telemetry. **Implemented; empirical conclusions remain hardware/model dependent.**

The research boundary remains explicit: the benchmark can establish whether a mechanism helps on a stated machine and workload, but a green CI run alone is not evidence of a speedup.
