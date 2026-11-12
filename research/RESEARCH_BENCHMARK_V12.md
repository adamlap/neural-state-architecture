# Research Benchmark v12 — Gating Model Calls Inside the Runtime

NSA's own overhead is tiny (0.07-0.3 ms per observation, 0.2 ms per `step()` with an echo backend, measured with cProfile), so speeding NSA up means avoiding, shrinking and accelerating *model* work.
`RuntimeConfig(abstain_text=..., reuse_familiar_answers=True, require_full_key_match=True)` adds two call-avoiding gates to `NSARuntime.step(..., auto_memory=True)`, both applied **after**
policy enforcement (a denied prompt is blocked before any gate; tested against the shipped `safe_assistant` policy):

* **abstain:** no memory key matches the question -> return `abstain_text` without a model call. `require_full_key_match` accepts only keys whose every token the question covers (one-character
  typos of words of 4+ letters tolerated), so "the database port for zebra" cannot answer with the only entity that happens to have a database port.
* **reuse:** the retrieved memory state (keys and their versions) is unchanged since an earlier call and the question wording is familiar (character-trigram cosine >= 0.95) -> return the earlier
  answer. Any write to a retrieved key bumps its version and invalidates. Only answers that contain one of the retrieved values are cached (grounded caching), so a model's mistake is not repeated.

## Design (`experiments/research_v12/gated_calls_benchmark.py`)
12 facts ingested (train phrasing, rule extraction, no model), then 60 questions per session with 6 genuine updates interleaved: exact repeats (50 %), one-character typos (25 %), paraphrases (25 %), and 15 % about entities
that do not exist. qwen2.5:1.5b, 6 sessions, 360 questions per condition, same decoding and retrieval in both conditions. Staleness is measured against the truth version at the time of the cached answer.

## Result (`results/research-v12-1.5b.json`)
| condition | model calls | reused | abstained | wall time | known-question accuracy | fabrication on unknown entities | stale reuse | reused-wrong | abstained on known |
|---|---|---|---|---|---|---|---|---|---|
| always call | 360 | 0 | 0 | 118 s | 0.60 | **1.00** | 0 | 0 | 0 |
| **gated** | **175** | 121 | 64 | **54 s** | **0.68** | **0.00** | 0 | 0 | 0 |

* 2.2x less time and 51 % fewer model calls, with *higher* accuracy on answerable questions and no fabrication (the always-call path answered every unknown-entity question with an invented value).
* **Process notes (what the first two runs got wrong).** Run 1 (typo tolerance for words of 5+ letters, no grounding check) showed gated calls at 114 but 54 "stale" reuses and 40 wrong abstentions: "cace size" was not recognised as `cache`,
  retrieval returned nothing, and a bad first answer was cached and repeated. The fixes (4-letter tolerance, grounded caching, a separate stale-vs-reused-wrong counter) are in the runtime, not the benchmark, and are covered by tests.
  Reuse repeats whatever the first grounded answer was, so a wrong-but-grounded answer (e.g. a different value present in the retrieved block) would still be repeated.
* Absolute accuracy (0.60-0.68) is limited by the 1.5B model reading its context, not by the gates.

Limits: one small model; synthetic session with rule-extracted memory; paraphrases are rarely reused by design (cosine < 0.95), so reuse savings come mostly from exact repeats and typos; a question whose answer depends on anything but the
retrieved keys (e.g. the conversation) would need those in the cache key.
