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

Not implemented: direct control of Accelerate's parameter lifecycle, multi-step lookahead beyond the next layer, coupling the predictor to NSA cognitive state (the plumbing exists in `cognitive_state_features`, but the backend feeds only execution transitions), and MoE expert regions.

## Running

```bash
pip install "neural-state-architecture[ml-residency]"
make residency-smoke                        # no model needed
make residency-benchmark RESIDENCY_MODEL=1.5b
```

The ML extras require `torch>=2.5` (transformers 5 refuses to use older torch); the backend raises a clear error if the installed torch is too old. Cached mode never downloads weights; the benchmark downloads only when no checkpoint path is given.

## Development phases

1. Foundation - region model, policy, predictor, cache, telemetry. **Done.**
2. Disk residency - empty-model + disk-backed checkpoint execution. **Done** (verified end to end on a tiny Qwen2 in CI).
3. Active residency - predictive page-cache prefetch around decoder regions. **Done, measured on real hardware; see status above.**
4. State coupling - use NSA cognitive state instead of transition statistics alone. Planned.
5. Learned residency - richer predictors trained from region transition traces. Planned.
6. MoE specialization - experts as independently resident regions. Planned.
7. Evaluation - peak VRAM/RAM, transfer bandwidth, latency, tokens/s and output quality on real checkpoints. In progress (`experiments/residency/benchmark_matrix.py`).
