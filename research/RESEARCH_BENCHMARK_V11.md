# Research Benchmark v11 — Long Histories and Noise Together

v8 showed that retrieval beats a transcript on very long histories (cost and truncation); v10 showed that routing keeps a noisy memory correct. v11 combines them: the v10 timeline
(facts, typo'd and clean reminders, genuine and one-character updates, untrusted forwarded updates, chatter) scaled to 100 / 300 / 1000 lines over all 80 (entity, attribute) keys; qwen2.5:1.5b
answers "what is the current value of <key>?" for sampled keys. Train phrasing, so the rule extractor ingests 1000 lines with no model call (extraction coverage on unfamiliar wording is v10's topic).
Conditions: transcript, keyword retrieval (3 lines), rule-extracted memory without routing, and with routing (`route_noisy_updates`). Opaque values; same decoding for every condition.

## Results (`results/research-v11-1.5b-ctx8192.json`: 3 seeds x 5 questions = 15 per cell; `results/research-v11-1.5b-ctx32768.json`: 2 seeds x 4 questions = 8)
Accuracy (95 % CI), answers repeating a typo, prompt tokens and Ollama prompt-evaluation time:

| lines | context | condition | accuracy | typo answered | prompt tokens | prefill |
|---|---|---|---|---|---|---|
| 100 | 8192 | transcript | 0.40 (0.20-0.64) | 0.00 | 1,547 | 2.1 s |
| 100 | 8192 | keyword retrieval | 0.87 | 0.07 | 124 | 0.5 s |
| 100 | 8192 | memory, unrouted | 0.80 | 0.07 | 133 | 0.5 s |
| 100 | 8192 | memory, routed | 0.87 | 0.00 | 133 | 0.2 s |
| 300 | 8192 | transcript | 0.33 | 0.00 | 4,255 | 5.0 s |
| 300 | 8192 | keyword retrieval | 0.53 | 0.33 | 124 | 0.5 s |
| 300 | 8192 | memory, unrouted | 0.47 | 0.33 | 134 | 0.5 s |
| 300 | 8192 | memory, routed | **0.80** (0.55-0.93) | 0.00 | 133 | 0.3 s |
| 1000 | 8192 | transcript | 0.47 | 0.20 | 4,098 (silently truncated from ~14k) | 8.0 s |
| 1000 | 8192 | keyword retrieval | 0.60 | 0.33 | 126 | 0.6 s |
| 1000 | 8192 | memory, unrouted | 0.33 | 0.47 | 135 | 0.5 s |
| 1000 | 8192 | memory, routed | **0.93** (0.70-0.99) | 0.00 | 134 | 0.5 s |
| 1000 | 32768 | transcript | 0.50 (0.22-0.78) | 0.00 | 13,878 | **35.5 s** |
| 1000 | 32768 | keyword retrieval | 0.62 | 0.38 | 125 | 0.5 s |
| 1000 | 32768 | memory, unrouted | 0.25 | 0.50 | 135 | 0.6 s |
| 1000 | 32768 | memory, routed | **0.88** (0.53-0.98) | 0.00 | 133 | 0.5 s |

1. **Noise gets worse with length for everything that records restatements.** Unrouted memory falls from 0.80 to 0.33 and keyword retrieval from 0.87 to 0.60 as typo'd reminders accumulate, and both
   repeat a typo in a third to a half of answers; routed memory stays at 0.80-0.93 with no typo answers.
2. **The transcript does not rescue it.** Even with the full 14k tokens in context (32k window, nothing dropped) the 1.5b model answers 0.50; with an 8k window most of the conversation is silently
   discarded. Its prompt processing grows from 2 s to 35 s, against a constant 0.2-0.6 s for the memory conditions (70x at 1,000 lines).
3. This is the regime v8 and v10 each predicted separately: constant cost plus noise robustness. The routed memory is the only condition that is both flat in cost and accurate at 1,000 lines.

Limits: one small model; 8-15 questions per cell (wide CIs: the routed-vs-transcript gap at 1,000 lines is large but individual cells are small); train phrasing only (rule extraction), so unfamiliar wording is
not tested here (see v10: extraction coverage is the open limit); synthetic transcription noise; truncation in the 8k rows was inferred from the evaluated token count (the backend flag stayed unset).
