# Research Benchmark v7 — Knowing What Is Not Known (Evidence-Gated Abstention)

Half of the questions are answerable from the event stream (`recall`, `supersession`); half ask about a key that never
appeared (`unanswerable`; the correct answer is `VALUE=UNKNOWN`).

| Condition | Mechanism |
|-----------|-----------|
| `transcript` | Full event log; generic escape clause ("if the value is not in the information provided, reply VALUE=UNKNOWN"). |
| `memory_generic` | Canonical memory retrieval (`NO_RELEVANT_MEMORY` when empty); generic escape clause. |
| `memory_keyed` | Same retrieval; clause keyed to the retrieval result ("only if RETRIEVED_MEMORY is exactly NO_RELEVANT_MEMORY…"). |
| `state_gated` | The runtime's soft-state uncertainty is derived from retrieval coverage (0 if memory holds the key, else 1). Above 0.5 the system abstains **without calling the model**; otherwise the model gets a plain instruction with **no escape clause**. |

The two prompt-based memory conditions are there because abstention by instruction proved wording- and model-dependent in
a preliminary check on identical prompts: qwen2.5:1.5b with a generic clause abstains even when the fact is present
(recall 1/6 correct); with the keyed clause 5/6 and correct abstention; with no clause it fabricates 6/6. qwen2.5:3b is
fine with the generic clause and worse with the keyed one.

Metrics: answerable accuracy, over-abstention (UNKNOWN on an answerable question), hallucination rate (any answer other than
UNKNOWN on an unanswerable question), fraction of episodes that needed a model call, latency; Wilson CIs; paired exact McNemar.

## Scientific boundary

`state_gated` abstains on a deterministic rule, so a zero hallucination rate for it is structural. The uncertainty signal is
retrieval coverage, **not** a learned or self-reported confidence; calibrating a signal for genuinely ambiguous or partial
evidence is not tested here. Per-cell samples are small (24 answerable, 12 unanswerable).
