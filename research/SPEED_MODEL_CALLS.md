# Making the model call itself faster (CPU, Ollama)

Scripts: `experiments/speed/nsa_call_speed.py`, `experiments/speed/prompt_layout_speed.py`. Results: `results/speed_nsa_call_{1.5b,7b}.json`, `results/speed_prompt_layout_{1.5b,7b}.json`.
Machine: 6 logical CPUs (WSL), no GPU. Each number is a median over 8-24 calls with unique values per call; accuracy is checked on every variant.

## Where the time goes

A memory-backed call is ~75-90 evaluated prompt tokens and a ~6-token answer. Prefill (reading the prompt) is about 70 % of the call: 1.5b 344 ms prefill + 141 ms decode;
7b 1,649 ms + 604 ms. So the lever is the number of prompt tokens the model has to read, then threads. Output-side settings barely matter because the answer is already short.

## Request settings (cumulative)

| variant | 1.5b wall ms | 7b wall ms |
|---|---|---|
| baseline (ctx 8192, default threads) | 485 | 2,275 |
| + `num_thread` = all CPUs | 390 | 1,528 |
| + `num_ctx` 2048 | 336 | 1,600 |
| + stop at newline | 333 | 1,927 |
| + instruction in a constant `system` prefix | 363 | 1,516 |
| + 12-token budget | 339 | 1,391 |

Threads are the one clear gain (1.24x / 1.49x). The rest is within run-to-run noise on the 7b (n=8); a smaller `num_ctx` helped the 1.5b a little and mostly saves memory.
Accuracy 1.00 everywhere.

## Prompt layout (all threads, 80-fact memory)

| layout | 1.5b wall ms | 1.5b acc | 7b wall ms | 7b acc |
|---|---|---|---|---|
| 3 retrieved rows | 393 | 1.00 | 1,276 | 1.00 |
| 1 retrieved row | 357 | 1.00 | 1,183 | 1.00 |
| whole memory pinned as a cached prefix | 218 | 0.79 | 796 | 1.00 |
| pinned, new fact appended every 4th call | 226 | 0.71 | 738 | 0.88 |
| pinned, a middle fact updated every 4th call | 244 (mean 707) | 0.75 | 1,029 (mean 3,088) | 0.88 |

* Pinning the whole memory lets the server's KV prefix cache serve it, so only the question is read: 1.6x on the 7b with accuracy unchanged, 1.8x on the 1.5b **but the small model
  loses accuracy** with 80 distractor rows. The first call pays for the prefix once (5 s / 23 s).
* Appending keeps the cache valid; an update in the middle re-reads everything after it, which makes the *mean* worse than retrieval when updates are frequent (7b: 3.1 s).
  The 0.88 cells are 2 misses out of 16, too few to call.
* End to end on the 7b: 2,275 ms (benchmark default) -> 796 ms pinned (2.9x), or -> ~1,180 ms with the always-safe settings (threads + fewer rows, 1.9x).

## What shipped (all opt-in)

`OllamaInferenceBackend(num_thread="auto", keep_alive=...)`, `generate(stop=..., system=...)`, `RuntimeConfig(compact_memory=True)`, `RuntimeConfig(pin_memory_prefix=True, pin_memory_max=200)`.
`compact_memory` is implemented and unit-tested but its speed/accuracy effect has not been measured yet.

Boundary: one machine, CPU only, synthetic lookups, small n; the pinned layout is not yet measured through `NSARuntime.step` or on long noisy memories.
