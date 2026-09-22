# Active Neural Residency

The active path is split into two layers:

1. The NSA predictor/controller decides which region is likely to execute next.
2. Accelerate remains authoritative for moving parameters between NVMe, RAM and the execution device.

## Prefetch semantics

`AccelerateDiskPrefetcher` does not modify parameters or Accelerate hooks. It reads the bytes of a predicted region from Accelerate's offload folder on a background worker, so the operating-system page cache is warm when Accelerate's own disk-offload hook needs them.

It understands the formats Accelerate writes to `index.json`:

* `{"dtype", "shape"}` entries - weights in `<offload_dir>/<key>.dat`;
* `{"safetensors_file", "weight_name"}` entries - a tensor inside a safetensors file (only that byte range is read; headers are parsed directly);
* Hugging Face `weight_map` indexes and single-file safetensors checkpoints.

A region with no mapped parameters warms nothing (an unmapped region never means "the whole model"). `prefetch()` returns the bytes actually read; `0` is recorded as `prefetch-skipped`, never as a completed prefetch.

This is real I/O prefetch. It is **not** the same as making a layer GPU-resident, and it is only useful for layers that Accelerate offloads to disk. The controller therefore schedules prefetch only for regions the prefetcher can cover (`prefetch_eligible`).

## Runtime flow

decoder layer N starts -> record transition N-1 -> N -> predict N+1 -> schedule page-cache prefetch on a worker -> layer N computes -> Accelerate loads N+1 through its normal hook (now from warm pages) -> record execution.

The first generation only establishes transition statistics; later ones exploit them. For a decoder the transition table is deterministic (layer i -> i+1), so the predictor reaches probability 1.0 after one pass.

Because the instrumentation wraps the *already hooked* `layer.forward`, the recorded `execute` latency includes Accelerate's weight loading. That is what makes a prefetch benefit visible.

## Why this boundary

Accelerate's hooks own the parameter lifecycle. Replacing or invoking them from a prediction thread would risk races, stale hook state, duplicate tensors or wrong offload accounting. The NSA adapter therefore overlaps storage I/O, not parameter ownership. Failures in the prefetch worker are recorded as `prefetch-error` events and never interrupt decoding; after `close()` further prefetches are no-ops.

## Metrics (`ResidencyTrace.metrics()`)

| Metric | Meaning |
|---|---|
| `prefetches`, `prefetch_completed`, `prefetch_skipped`, `prefetch_errors` | issued / finished / warmed-nothing / failed |
| `prefetch_hits` | completed prefetches consumed by a later execution that *started after* they finished (one prefetch serves at most one execution) |
| `prefetch_hit_rate` | `prefetch_hits / prefetches` (always in [0, 1]) |
| `prefetch_coverage` | fraction of executions that had a completed prefetch waiting |
| `bytes_prefetched` | bytes actually read into the page cache |
| `bytes_moved` | placement transfers only (`resident`/`evict`); prefetch bytes are reported separately |

## Evaluation method

`experiments/residency/benchmark_matrix.py` times loading and decoding separately, alternates prefetch on/off order each round, drops the OS page cache for the weight files before decoding (disable with `--no-cold-cache`), hashes the greedy-decoded tokens of every run and refuses to call a comparison meaningful when no bytes were prefetched. Its `summary.notes` lists any caveat that applies to a given run.
