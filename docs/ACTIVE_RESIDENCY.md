# Active Neural Residency

The active residency path is intentionally split into two layers:

1. NSA predictor/controller decides which region is likely to execute next.
2. Accelerate remains authoritative for moving model parameters between NVMe, RAM and the execution device.

## Prefetch semantics

AccelerateDiskPrefetcher does not modify model parameters or Accelerate hooks. It reads the safetensor pages belonging to a predicted decoder region on a background worker. The goal is to warm the operating-system/page-cache path before Accelerate's normal disk-offload hook needs the weights.

This is a real I/O prefetch, but it is not the same thing as making the layer GPU-resident. NSA therefore records prefetch/prefetch-complete telemetry separately from resident.

If the prefetcher cannot find an offload index or safetensors dependency, it safely becomes a no-op. Normal Accelerate inference remains the fallback.

## Runtime flow

decoder layer N -> observe transition -> predict likely N+1 -> schedule page-cache prefetch -> continue layer N computation -> Accelerate loads N+1 through its normal hook -> execute N+1 -> record transition and continue learning

The first generation establishes transition statistics; subsequent generations can exploit them.

## Why this boundary matters

Accelerate's disk/CPU offload hooks own parameter lifecycle. Replacing or manually invoking those hooks from a prediction thread would risk races, stale hook state, duplicate tensors, or incorrect offload accounting. The NSA adapter therefore overlaps storage I/O, not parameter ownership.

## Metrics

ResidencyTrace records bounded events and can optionally persist JSONL. Useful metrics include prefetch count and hit count, bytes touched/transferred, execution count, transfer latency, and average event latency.

A future backend can implement a stronger physical prefetch mechanism without changing the NSA predictor or policy APIs.
