# Can a dense model read only "the weights it needs"? Two go/no-go probes (negative)

Scripts: `experiments/sparsity/neuron_sparsity_probe.py`, `experiments/sparsity/layer_skip_probe.py`. Results: `results/sparsity_probe_qwen0.5b.json`,
`results/layer_skip_probe_qwen0.5b.json`. Model: Qwen2.5-0.5B-Instruct (24 blocks, 4,864 feed-forward neurons per block, SiLU), fp32 on CPU. Four topics (code, cooking,
finance, medicine), 4 texts per topic for profiling, 2 held-out texts per topic for scoring (~270 scored tokens). Score: does the reduced model predict the same next token
as the full model (top-1 agreement), and KL divergence.

## 1. Feed-forward neurons

Activity is not concentrated: on average 55 % of a block's neurons are needed for 90 % of its output contribution and 84 % for 99 %.

| kept neurons | oracle (this token's top) | previous token's top | static topic set | static global set | another topic's set | topic set, dropped = mean | random |
|---|---|---|---|---|---|---|---|
| 10 % | 0.38 | 0.06 | 0.06 | 0.07 | 0.03 | 0.08 | 0.00 |
| 25 % | 0.68 | 0.07 | 0.09 | 0.15 | 0.12 | 0.18 | 0.00 |
| 50 % | 0.89 | 0.12 | 0.40 | 0.34 | 0.25 | 0.43 | 0.00 |

* Even the oracle (which needs the full computation, so it is an upper bound, not a speedup) loses 11 % of next-token decisions at half the neurons.
* Any set known in advance is far worse: the best static set at 50 % agrees on 0.43 of tokens. The previous token's active set is a poor predictor (0.12).
* Topic information is real but small: the right topic's set beats another topic's (0.40 vs 0.25 at 50 %), and topic sets overlap only 34-40 % (Jaccard at 25 %). It does not
  come close to making a static hot set usable.

## 2. Whole blocks (the unit the residency module manages)

| blocks skipped (of 24) | chosen per topic | chosen globally | another topic's choice |
|---|---|---|---|
| 1 | 0.78 | 0.79 | 0.78 |
| 2 | 0.75 | 0.75 | 0.72 |
| 4 | 0.66 | 0.66 | 0.62 |
| 6 | 0.50 | 0.57 | 0.52 |

Skipping a single block (4 % of the compute) already changes 21 % of next-token decisions, and a per-topic choice is no better than a global one. The least harmful blocks
are the same middle ones (8-13) for every topic.

## Verdict

For this model, without retraining, there is no context-keyed subset of weights that can be left unread at acceptable accuracy, at neuron or block granularity. This agrees
with the literature: the published hot/cold-neuron and flash-loading speedups rely on ReLU models or on fine-tuning the model to be sparse, and layer skipping on models
trained with layer dropout. Not tested here: larger models (usually more redundant), trained per-token predictors, sparsity-inducing fine-tunes, mixture-of-experts models
(where "only some weights per token" is built in and the expert choice is the natural residency hot set).

Consequence for the residency track: context-keyed hot sets make sense where the unit is optional by construction - experts in a mixture-of-experts model, adapters, or whole
models in a small/large cascade - not inside a dense model.
