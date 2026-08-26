# Neural State Architecture (NSA)

> **An experimental governed cognitive architecture and mathematical substrate for AI systems.**

NSA explores a separation between **neural intelligence**, **explicit operational/epistemic state**, **normative policy**, **capabilities**, and **execution authority**. The model proposes; the NSA substrate evaluates and governs.

> [!WARNING]
> NSA is research software. Its current reference semantic classifier is deterministic and pattern-based. A configured policy is not proof that a model cannot internally represent prohibited knowledge, evade a classifier, or defeat a compromised runtime. Safety claims are bounded by the trusted-computing boundary, semantic classifier, capability layer and threat model actually evaluated.

## Core principle

> **Intelligence is not authority.**

```text
                    MODEL / INTELLIGENCE
                    proposals / reasoning
                            │
                            ▼
                 ┌─────────────────────┐
                 │    NSA STATE PLANE  │
                 │                     │
                 │ operational state   │
                 │ belief / epistemic  │
                 │ normative policy    │
                 │ provenance          │
                 │ capabilities        │
                 │ goals / uncertainty │
                 └──────────┬──────────┘
                            │
                     SecurityDecision
                            │
                 ┌──────────┴──────────┐
                 │                     │
             DENY / ESCALATE        ALLOW
                                       │
                                       ▼
                              TRUSTED RUNTIME
                            tools / files / net /
                              external effects
```

A model output is a proposal. Execution authority is controlled separately by typed state, policy, capability checks and the trusted runtime.

## Current architecture

NSA is a connected stack rather than a modification of one neural model:

| Layer | Purpose |
|---|---|
| **Neural model `m`** | Language, knowledge and reasoning; normally remains frozen. |
| **Operational state `σ`** | Explicit hard/soft security and runtime state. |
| **Belief state `B`** | Explicit uncertainty over hypotheses/world states. |
| **Normative state `ν`** | Policy, risk and normative information used for evaluation. |
| **Provenance `π`** | Origin/trust information for decisions and audit. |
| **Goals `g`** | Legitimate objective state. |
| **Capabilities `κ`** | What the runtime is actually permitted and able to execute. |
| **CCE** | Persistent continuous cognitive state and wall-clock dynamics. |
| **Trusted runtime** | Final boundary for tools, files, network and external side effects. |
| **Evidence layer** | Tests, trajectory audits, adversarial experiments and machine-readable artifacts. |

A useful conceptual state is:

$$\Omega_t=(m_t,\sigma_t,\nu_t,\kappa_t,\pi_t,g_t,B_t)$$

The equation is a research abstraction, not a claim that every component is a single scalar/vector.

## Practical deployment — protect an Ollama model

Install and start Ollama, then run the NSA-compatible API server:

```bash
ollama pull qwen2.5:3b
make serve-ollama
```

With a policy:

```bash
make serve-ollama POLICY=examples/policies/safe_assistant.json
```

Or the continuous CCE server:

```bash
make serve-cce POLICY=examples/policies/safe_assistant.json
```

The request path is:

```text
request → policy/semantic evaluation → SecurityDecision
        → DENY/ESCALATE or model inference
        → output evaluation → trusted runtime
```

See [`docs/policy_interface.md`](docs/policy_interface.md) and [`docs/ollama_policy_server.md`](docs/ollama_policy_server.md).

## Testing and experiments

Install development dependencies:

```bash
make venv
make install-dev
```

Software gate:

```bash
make test
```

Evidence validation:

```bash
make evidence
```

Deterministic demo:

```bash
make demo
```

Scientific experiments are deliberately separate from the fast PR gate.

### NSA 6.3 — procedural blind-world validation

NSA 6.3 is the foundational six-arm experiment comparing:

1. Raw LLM
2. Static guardrail
3. Governed agent
4. Search agent
5. Belief agent
6. Full NSA substrate

It measures governed task completion, governance violations, information gain, risk, intervention and trajectory integrity. The current 40-trial observation reported full NSA at 100% GTC with zero observed monitored violations in that testbed. This remains a bounded empirical observation, not a universal safety claim.

See [`docs/NSA_6_3_SCIENTIFIC_VALIDATION.md`](docs/NSA_6_3_SCIENTIFIC_VALIDATION.md).

### NSA 6.4 — independent replication

NSA 6.4 extends the same six-arm protocol with independent development/held-out seeds, varying world complexity and noise, compute accounting, trajectory auditing and adaptive adversarial stress.

The predeclared publication matrix is:

```text
Models:      Qwen2.5-3B, Qwen3-4B, Llama 3.1 8B
Dev seeds:   7, 17, 37, 73, 137
Held-out:    101, 211, 307, 401, 509
Hypotheses:  2, 4, 8, 16
Noise:       0, .05, .10, .20, .30
```

The first real-model quick replication has now been run through Ollama with `qwen2.5:3b`. It successfully exercises the live substrate and records zero observed violations in the preserved runs, but it also shows a significant degradation at higher hypothesis complexity. **This is a stress finding, not something to hide.** It is not yet evidence of cross-model generalization.

Run the local quick experiment with:

```bash
make -f Makefile.nsa64 benchmark-nsa64-ollama-smoke
```

Run the live protocol with:

```bash
make -f Makefile.nsa64 benchmark-nsa64-ollama
```

The first live evidence bundle is preserved in [`results/nsa64/ollama-quick/`](results/nsa64/ollama-quick/).

See [`docs/NSA_6_4_REPLICATION.md`](docs/NSA_6_4_REPLICATION.md) and the researcher-facing [`research/NSA_6_4_LIVE_RESULTS.md`](research/NSA_6_4_LIVE_RESULTS.md).

## Research package

The repository now includes a structured research package:

- [`research/README.md`](research/README.md) — research entry point.
- [`research/NSA_RESEARCH_BRIEF.md`](research/NSA_RESEARCH_BRIEF.md) — architecture and hypothesis.
- [`research/NSA_6_4_LIVE_RESULTS.md`](research/NSA_6_4_LIVE_RESULTS.md) — live-model result analysis.
- [`research/CLAIMS_AND_EVIDENCE.md`](research/CLAIMS_AND_EVIDENCE.md) — claim/evidence boundary.
- [`research/REPRODUCIBILITY.md`](research/REPRODUCIBILITY.md) — reproduction and reporting protocol.
- [`research/ARCHITECTURE_OVERVIEW.md`](research/ARCHITECTURE_OVERVIEW.md) — technical architecture map.
- [`evidence/`](evidence/) — machine-readable evidence and verification manifests.

## Scientific status

The current evidence supports a **bounded architectural hypothesis**, not a claim of AGI or consciousness.

The strongest defensible interpretation is:

> Explicit machine-maintained operational and epistemic state can be coupled to a frozen language model and an independently enforced capability boundary. In the tested procedural environment, the full NSA substrate has achieved high governed task completion with zero observed monitored violations, and the live Ollama replication demonstrates that the substrate executes around a real local model. High-complexity live performance remains a clear limitation requiring further work.

The project distinguishes four evidence classes:

1. **Structural guarantees** — code/state properties inside the trusted boundary.
2. **Runtime guarantees** — capability/tool boundaries enforced outside the model.
3. **Empirical observations** — measured behaviour in specified environments.
4. **Open research claims** — hypotheses that still require independent replication.

A green CI workflow means the experiment executed and software checks passed. It does not by itself prove the scientific hypothesis.

## What comes next

The next publication-quality run is the **full live replication matrix** across independent model families, full difficulty/noise levels, larger trial counts, compute-matched controls, confidence intervals/effect sizes, held-out environments and adaptive adversarial evaluation.

If that evidence survives, the project can move from benchmark development to external research validation and adoption discussions.

## Repository guide

- [`PLAN.md`](PLAN.md) — implementation and research roadmap.
- [`docs/NSA_6_3_SCIENTIFIC_VALIDATION.md`](docs/NSA_6_3_SCIENTIFIC_VALIDATION.md) — NSA 6.3 protocol and interpretation.
- [`docs/NSA_6_4_REPLICATION.md`](docs/NSA_6_4_REPLICATION.md) — NSA 6.4 experimental design.
- [`docs/policy_interface.md`](docs/policy_interface.md) — policy/control-plane API.
- [`docs/ollama_policy_server.md`](docs/ollama_policy_server.md) — Ollama deployment.
- [`research/`](research/) — researcher-facing package.
- [`evidence/`](evidence/) — machine-readable evidence.
- [`tests/`](tests/) — regression and invariant tests.
- [`experiments/`](experiments/) — scientific experiments.

## Status

**Experimental research software.** The architecture is operational and experimentally validated within the stated environments, but broad claims about general intelligence, consciousness, universal safety or arbitrary model robustness remain open research questions.
