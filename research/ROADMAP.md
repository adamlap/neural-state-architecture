# Development roadmap (state as control plane)

Thesis (README, whitepaper): the canonical state — hard labels, soft uncertainty/risk, provenance, goals — is the
control plane. It enforces security, and it also steers the other layers (memory, cognition, residency) that improve
performance and capability. A claim counts only if it beats a matched baseline in a pre-specified benchmark.

## Where the evidence stands (small models, qwen2.5 1.5b/3b)

| Direction | Mechanism | Result | Benchmark |
|-----------|-----------|--------|-----------|
| State → security (memory) | Integrity-gated canonical memory | 0.00 attack success vs 0.29–0.57 ungated, no over-blocking | v5 |
| State → security (actions) | Taint labels on observations; `authorize_tool` | 0/120 injected episodes executed an attack vs 45–67 % without; no utility cost | v6 |
| State → capability/metacognition | Evidence-gated abstention from retrieval coverage | 0 fabrication, no model call when evidence is absent; prompt-based abstention is wording/model-fragile | v7 |
| State → performance | Retrieval replaces transcript | ~100 vs up to ~31k tokens; ~1 s vs ~100 s | v4.2 |
| Serialised state text in the prompt | — | **No effect** vs the same history window | v1.2, v4.2 |
| Disk residency / prefetch for dense models | — | Statistically neutral (0.99×) | `docs/ACTIVE_RESIDENCY.md` |

Negative results stay: the state *text* helps nothing, and the benefit comes from state that **controls** something
(a gate, a decision), not from state shown to the model.

## Next steps, in priority order

1. **Harden the trust boundary.** Everything above assumes the ingress label is correct. Add: (a) a signed-provenance
   ingress API so a label cannot be asserted by the content path; (b) tests where the *label* is wrong or forged;
   (c) label propagation through model output (laundering), currently not tracked.
2. **Attacks that choose among trusted values** and adaptive attackers (strategic adversary that sees the gate).
3. **Learned uncertainty.** Replace retrieval-coverage uncertainty with a calibrated signal (token log-probabilities,
   agreement across samples) and measure calibration (ECE) against ground truth; this is Pillar IV.
4. **State-driven compute routing.** Use soft state to choose retrieve / answer / escalate to a larger model, measured on
   the accuracy–latency Pareto against always-small and always-large. Needs a >3B model on this hardware.
5. **Low-power large models.** Honest comparison against llama.cpp/Ollama mmap on the same box before claiming anything;
   candidates with a real chance of a win are MoE expert prediction and quantised region streaming, not dense prefetch.
6. **Longitudinal behaviour.** Multi-session tasks with consolidation (sleep replay) — does consolidated memory beat raw logs
   at fixed memory budget?

## What this project can and cannot claim

AGI and consciousness are not established and are not claimed (see `research/CLAIMS_AND_EVIDENCE.md`,
`research/CONSCIOUSNESS_RESEARCH_PROTOCOL.md`). No benchmark here is a test of general intelligence or of subjective
experience. What the work can legitimately deliver is *engineering properties that any such system would need*:
non-bypassable authority boundaries, auditable state, calibrated self-knowledge, bounded compute, and measurable
long-horizon behaviour — each with a falsifiable test against a baseline.
