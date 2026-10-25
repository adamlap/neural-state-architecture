# Research Benchmark v8 — Operation Under a Fixed Memory Budget

## Why

On a low-memory device the context window is the knob. The KV cache is allocated for `num_ctx` tokens when the model loads, so
`num_ctx` sets resident memory, and prompt-processing time grows with the prompt. A stateless transcript agent must keep the whole
history inside that window. A retrieval agent only needs what it retrieves. v8 measures what that is worth on this CPU-only box.

## Design

Sweep `num_ctx` ∈ {2048, 4096, 8192, 32768} × history length (distractors) ∈ {10, 100, 250, 1000}, two tasks (`recall`,
`supersession_current`), 3 seeds. Conditions:

| Condition | Prompt |
|-----------|--------|
| `transcript` | The whole history. If it exceeds `num_ctx` the server silently drops the *start* of the prompt, which is where the fact is. |
| `rag_lines` | Generic keyword retrieval over the raw log: the last 3 lines that mention the queried key. No structure, no supersession. |
| `temporal_memory` | The canonical store's current value (versioned, integrity-gated). |

Recorded per call: correctness (lenient), whether the prompt was truncated, Ollama's own `prompt_eval` time, wall time, the loaded model's
size from `/api/ps`, and the resident set of the Ollama processes. Transcripts whose processed prompt would exceed `--transcript-token-cap`
are skipped and excluded from accuracy.

## Scientific boundary

* `temporal_memory` uses the deterministic extractor of v2–v8 and `rag_lines` uses the question text; neither tests LLM memory formation (see v9).
* **Generic keyword retrieval (`rag_lines`) matches canonical memory on these non-adversarial tasks.** Retrieval itself is not what is
  special about NSA's memory; what the canonical store adds is supersession, the integrity gate (v5, v6) and abstention (v7).
* Memory is server-side and includes the weights; differences across `num_ctx` are the KV cache. One CPU-only machine, one run per cell.
