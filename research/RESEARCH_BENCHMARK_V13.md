# Research benchmark v13: raw-line retrieval through the real runtime, at scale

`experiments/research_v13/runtime_lines_benchmark.py`, results in `results/research-v13-1.5b.json` and `results/research-v13-7b.json`.

Agents are built with `observe()` (rule extractor, routing, one untrusted source, `provenance_window=256`) over 100 / 1,000 / 3,000-line noisy histories and every
question goes through `step(q, auto_memory=True, remember_prompt=False, retrieval_query=key)`. Conditions: `mem_only` (`retrieve_lines=0`), `mem_lines` (`retrieve_lines=3`),
`rag_overlap` (3 lines, scan), `transcript` (only at 100 lines; longer ones skipped). `train` phrasing is readable by the rule extractor; `heldout` is not, so memory is mostly
empty and only the raw lines can answer. 1.5b: 6 seeds x 6 questions = 36 per cell. 7b: 3 seeds x 4 questions = 12 per cell (guarded, ctx 8192).

## Accuracy (lenient value match)

| lines | phrasing | 1.5b mem_only | 1.5b mem+lines | 1.5b rag | 7b mem_only | 7b mem+lines | 7b rag |
|---|---|---|---|---|---|---|---|
| 100 | train | 0.89 | 0.94 | 0.92 | 1.00 | 1.00 | 0.92 |
| 100 | held-out | 0.31 | **0.81** | 0.92 | 0.33 | **0.92** | 1.00 |
| 1,000 | train | 0.78 | 0.92 | 0.53 | 1.00 | 0.83 | 0.33 |
| 1,000 | held-out | 0.61 | 0.72 | 0.64 | 0.67 | 0.67 | 0.67 |
| 3,000 | train | 0.86 | 0.89 | 0.53 | 1.00 | 0.83 | 0.58 |
| 3,000 | held-out | 0.56 | 0.69 | 0.61 | 0.67 | 0.58 | 0.25 |

Transcript at 100 lines: 0.36-0.47 (1.5b), 0.75-0.92 (7b), at ~1.6k prompt tokens and 14 s vs 4 s on the 7b.

Paired exact McNemar, `mem_lines` vs `mem_only`: significant only at 100 lines held-out (1.5b 18:0, p<0.001; 7b 7:0, p=0.016). Every 1,000/3,000-line comparison is not significant
(1.5b: +4 to +7 discordant in favour of lines, p 0.125-1.0; 7b: 0-2 each way, the signs favour memory alone on the train phrasing).

## What it says

* The v10 claim (memory + lines fixes unreadable phrasing) is confirmed through the runtime path at 100 lines and is the only clearly significant gain.
* It does **not** scale cleanly. As the history grows, the most similar raw lines are increasingly typo'd restatements; `rag_overlap` alone collapses (0.25-0.64) and answers
  a typo 17-50 % of the time. Adding lines to a *correct* memory hurt the 7b on known phrasing (1.00 -> 0.83, n.s., n=12): the model follows the noisy lines over the memory entry.
  On the 1.5b it helped (0.78 -> 0.92, n.s.). Treat `retrieve_lines` as useful when memory coverage is poor, not as a free addition; recommended use stays the fallback pattern
  (memory first, lines only when no key matches) rather than always-on.
* Costs are flat: ~185 prompt tokens and the same latency at 100 or 3,000 lines. Ingest is 0.15-0.2 ms/line.
* Retrieval latency (`results/lines_scale.txt`, `experiments/speed/lines_scale.py`): inverted index 0.01 / 0.10 / 1.7 ms at 1k / 10k / 100k lines vs a full scan 2.9 / 30 / 301 ms.
* Boundary: synthetic, rule extractor, opaque values, small n on the 7b.

## Side finding: observe() cost grew with agent age

Every transition hashed the whole provenance history (3x per `observe`), so cost rose linearly (0.09 ms at 500 observes, 1.4 ms at 8,000). `RuntimeConfig(provenance_window=N)` folds old
provenance into a hash chain (still tamper-evident, digest of never-folded states unchanged); the per-call cost is then flat (`tests/test_provenance_folding.py`). Off by default.

## Change detector for the dual-process test (`experiments/organic/dual_process_v2.py`)

CUSUM on the prediction error of queried valenced outcomes; alarm opens a 10-event window of forced queries. Parameters chosen on dev seeds 0-9 (best dev margin over matched random only
+0.055, most settings negative), then scored once on unseen seeds 200-209 (`results/organic_dual_process_v2.json`):

| policy | queries | acc just after shift | acc end | recovery (events) |
|---|---|---|---|---|
| cusum | 67 | 0.71 | 0.79 | 60 |
| gated (v1) | 73 | 0.64 | 0.72 | 60.5 |
| random (matched) | 66 | 0.52 | 0.77 | 140 |
| periodic (matched) | 66 | 0.54 | 0.79 | 110 |

On these seeds the familiarity-gated policies recover faster than matched schedules, and the detector adds ~0.07 over plain gating, but: no paired test was run, seed variance is large (v1's
final seeds did not show the gated-vs-matched gain), the dev grid showed no robust detector advantage, and the pre-shift false-alarm rate was 1-6 per run. Verdict: inconclusive; not claimed.
