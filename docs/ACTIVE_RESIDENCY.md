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

### Never re-warm what's already warm

`AccelerateDiskPrefetcher.needs_warming(region_id)` checks real OS page-cache residency (`nsa.residency.page_cache`, via `mincore(2)`) before a background prefetch task is even scheduled. This exists because measurement, not intuition, showed it mattered:

1. **First version** (no residency check): re-read every mapped byte on every decoder step regardless of whether it was already cached. On a small model (Qwen2.5-0.5B-Instruct) whose whole weight set stays in the page cache after the first pass, this meant re-reading ~19 GB across a 40-token generation of a 954 MB model -- pure overhead. Measured **23% slower** decoding than no prefetch at all (3 alternated, cold-cache-start runs).
2. **mincore, one mmap per check**: correctly detected "already resident" and skipped the actual read, but `mmap()`/`munmap()` is itself a real syscall pair, done per tensor per decoder step. Measured **21-38% slower** than no prefetch -- the check cost more than the read it was avoiding.
3. **mincore, one mmap per file, reused** (`PageCacheProbe`) **plus a synchronous pre-check gating whether to schedule a background task at all** (`prefetch_eligible` now calls `needs_warming()`, not just `covers()`): for an already-resident region, nothing is scheduled -- no lock, no `ThreadPoolExecutor` handoff, no telemetry write. Measured decode time statistically indistinguishable from prefetch disabled (0.94x, i.e. no measurable cost; stdevs overlap).

On a real, disk-bound model (Qwen2.5-3B-Instruct on this CPU-only, 7.8 GB RAM host, ~85% of layers never fitting in the RAM/VRAM tiers) the same final design measured a 0.99x median (statistically neutral) across 3 alternated runs each way, with individual runs ranging from 30% faster to 15% slower than no-prefetch. The mechanism is correct here (100% prefetch hit rate, identical outputs every run) but not reliably beneficial: the background prefetch thread and the main compute thread compete for the same limited CPU cores and disk bandwidth on this host, so whether the prefetch wins its race against Accelerate's own synchronous read varies run to run. A machine with cores/disk bandwidth to spare for the prefetch thread, or a larger `lookahead` (more predicted regions in flight, more total time budget to win the race), would be the next thing to test -- not yet done.

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

`experiments/residency/benchmark_matrix.py` times loading and decoding separately, alternates prefetch on/off order each round, drops the OS page cache for the weight files before decoding (disable with `--no-cold-cache`), hashes the greedy-decoded tokens of every run and refuses to call a comparison meaningful when neither `bytes_prefetched` nor `bytes_already_resident` is ever nonzero (nothing mapped, or a broken offload index -- distinct from the mechanism correctly finding everything already warm, which legitimately reports 0 for both on a fast enough model; see above). Its `summary.notes` lists any caveat that applies to a given run.
