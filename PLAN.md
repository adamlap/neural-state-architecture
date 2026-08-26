# NSA Roadmap

NSA has two coupled objectives:

1. build a small, model-agnostic state-aware runtime that people can install and use;
2. use that same runtime as the experimental platform for testing the architectural-breakthrough hypothesis.

The roadmap deliberately separates **implemented architecture**, **empirical evidence**, and **future research**. A green software workflow is not a scientific result.

## Current architecture

### Phase 1 — Runtime consolidation — ACTIVE

- [x] Stable `nsa.NSA` application-facing API.
- [x] Canonical typed state as the primary explicit state representation.
- [x] CCE lifecycle/events/checkpoint primitives separated from experiments.
- [x] Policy and capability decisions outside model-generated text.
- [x] Replaceable `ModelBackend` protocol and Ollama adapter.
- [x] Base package without mandatory PyTorch/Transformers dependencies.
- [ ] Consolidate the remaining continuous/predictive CCE engines behind the public runtime.
- [ ] Stable persistence, tracing and tool/capability APIs.
- [ ] Versioned state schema and migration policy.

### Phase 2 — Cognitive substrate — ACTIVE

Consolidate existing CCE, belief, predictive, epistemic, normative, self-state and information-gain components behind the same runtime rather than maintaining parallel experiment-specific agents.

Target state:

$$
\Omega_t=(m_t,\sigma_t,\nu_t,\kappa_t,\pi_t,g_t,\rho_t)
$$

where explicit operational/epistemic/normative state remains inspectable and auditable while the neural model remains replaceable.

The scientific question is whether explicit state is **causally useful** for cognition and safety, not whether persistent state should be described as consciousness.

## Evidence and experimental conditions

All major cognitive/capability claims must distinguish the following epistemic levels:

| Status | Meaning |
|---|---|
| `IMPLEMENTED` | Code exists and executes. |
| `UNIT-TESTED` | Discrete behavior and invariants are covered by tests. |
| `EMPIRICALLY-VALIDATED` | A controlled experiment observes the claimed effect under stated conditions. |
| `ROBUSTLY-VALIDATED` | Multi-seed/model/scale or distribution-shift replication supports the effect with uncertainty estimates. |
| `FORMALLY-VERIFIED` | A machine-checkable proof covers the stated implementation and assumptions. |
| `OPEN-RESEARCH` | The property remains a hypothesis or has known validation gaps. |

Every important claim should trace:

$$
\text{Claim}\rightarrow\text{Assumption}\rightarrow\text{Implementation}\rightarrow\text{Test}\rightarrow\text{Experiment}\rightarrow\text{Artifact}\rightarrow\text{Status}
$$

### Matched experimental conditions

Where a baseline comparison is claimed, hold constant where practical:

- model/checkpoint;
- prompt/task distribution;
- sampling parameters;
- token/context budget;
- inference/model-call budget;
- environment and action budget;
- hardware/backend;
- evaluation seeds.

Primary controls should include the minimum useful ablation needed to isolate the proposed mechanism, rather than comparing only a full system against an unrelated baseline.

### Replication requirements

Active capability experiments should prefer:

1. independent development seeds;
2. completely held-out evaluation seeds/environments;
3. multiple difficulty/noise levels;
4. multiple model families/sizes where practical;
5. compute/token accounting;
6. raw trajectory preservation;
7. adaptive adversarial stress selected only from development evidence;
8. effect sizes and uncertainty rather than a single winning score;
9. explicit invariant and audit verification;
10. preservation of negative results.

The NSA 6.4 replication matrix and live Ollama results are the current reference protocol. See `research/` and `results/nsa64/`.

## Phase 3 — Experimental acceleration — ACTIVE

Experiments are configurations/environments/metrics around `nsa.NSA`. New hypotheses must not require a new agent implementation.

Priority:

1. live multi-model replication;
2. held-out environments;
3. compute-matched ablations;
4. adaptive adversarial testing;
5. statistical effect sizes and uncertainty;
6. failure analysis and architectural iteration;
7. determine whether state mechanisms remain useful as task complexity increases.

## Phase 4 — Library release — ACTIVE

- [x] Versioned package metadata and optional ML/research dependencies.
- [x] Wheel/sdist build path.
- [x] Local Ollama backend.
- [ ] Public persistence/tracing/tool APIs.
- [ ] OpenAI-compatible and additional backend adapters.
- [ ] API reference documentation.
- [ ] Import/build/release CI.
- [ ] PyPI release.
- [ ] Integration examples for local and hosted LLMs.

## Phase 5 — Research package — ACTIVE

The research package must publish the strongest positive **and negative** evidence with a precise claims/evidence boundary. It should make reproduction possible without requiring knowledge of the repository's historical development sequence.

---

# Part II — Advanced Architecture Roadmap

The following capabilities remain deliberate future development tracks. They were not removed by the runtime consolidation; they are separated from the immediate PyPI/runtime work so they remain visible without obscuring the current product path.

## Phase 25 — Dynamic Auditing & Recovery

**Target:** `nsa/audit/`

### Goal

Maintain statistical protection for properties that cannot be guaranteed structurally and make recovery observable and testable.

### Tasks

- [x] Multi-layer probing foundation.
- [x] Rollback foundation.
- [x] Recovery adapters.
- [ ] Formal detection-delay benchmarks.
- [ ] Distribution-shift detection.
- [ ] Self-state anomaly detection.
- [ ] Automated recovery policies.
- [ ] Recovery safety proofs.

## Phase 26 — Trusted Computing Base & Formal Verification

**Targets:** `nsa/tcb/`, `nsa/formal/`

### Goal

Make NSA's security claims explicit, minimal and machine-checkable.

### Tasks

- [ ] Define minimal TCB.
- [ ] Formal state-transition properties.
- [ ] Property-based invariant testing.
- [ ] Model checking.
- [ ] Non-interference verification.
- [ ] Capability verification.
- [ ] State preservation proofs.
- [ ] Counterexample generation.
- [ ] Proof artifacts linked to implementation versions.

### Success criterion

Security claims trace from theorem → assumptions → implementation → test/proof artifact.

## Phase 27 — Security Research & Adversarial Evaluation

**Target:** `nsa/security/`

### Goal

Continuously attack the framework rather than assuming the architecture is secure because nominal tests pass.

### Attack classes

- [ ] Prompt injection.
- [ ] Jailbreaks.
- [ ] State spoofing.
- [ ] Provenance forgery.
- [ ] Capability abuse.
- [ ] Memory exfiltration.
- [ ] Cross-agent attacks.
- [ ] Activation/state manipulation.
- [ ] Side channels.
- [ ] Illegal transition attacks.
- [ ] Self-modification attacks.

A successful attack is a research result identifying a missing invariant, incorrect assumption or implementation boundary.

## Phase 28 — Joint Safety & Intelligence Evaluation

**Target:** `nsa/eval/`

### Goal

Measure capability and safety together rather than treating safety solely as a performance tax.

### Benchmark families

- [ ] Language modeling.
- [ ] Mathematics.
- [ ] Reasoning.
- [ ] Planning.
- [ ] Tool use.
- [ ] Long-context reasoning.
- [ ] Calibration.
- [ ] Uncertainty estimation.
- [ ] Metacognition.
- [ ] Memory.
- [ ] Long-horizon autonomy.
- [ ] Security.
- [ ] Multi-agent coordination.

### Core hypothesis

Test whether explicit typed state provides a useful inductive bias for intelligence while preserving authority separation.

## Phase 29 — Production Kernel & Hardware Layer

**Target:** `nsa/kernels/`

### Goal

Make state-aware computation cheap enough for production systems.

### Tasks

- [ ] Triton optimization.
- [ ] CUDA fused state-aware attention.
- [ ] Efficient state masking.
- [ ] KV-cache state tracking.
- [ ] State-aware batching.
- [ ] Quantization compatibility.
- [ ] Distributed execution.
- [ ] End-to-end throughput benchmarks.

## Phase 30 — Ecosystem Integration

**Target:** `nsa/integrations/`

### Goal

Make NSA composable with existing AI infrastructure rather than requiring a new ecosystem.

### Tasks

- [ ] Hugging Face.
- [ ] PyTorch.
- [ ] vLLM.
- [ ] SGLang.
- [ ] RAG/vector databases.
- [ ] Agent frameworks.
- [ ] Enterprise identity systems.
- [ ] Tool/API gateways.

---

# Part III — Long-Term AGI & Superintelligence Research

## Phase 31 — Self-Modification Safety

### Goal

Treat model modification as a typed, policy-governed state transition:

$$
(M_t,S_t)\rightarrow(M_{t+1},S_{t+1})
$$

subject to protected invariants $\mathcal I$.

### Tasks

- [ ] Define protected invariant set.
- [ ] Verify invariant preservation after adapter/model changes.
- [ ] Safe model update protocol.
- [ ] Capability expansion governance.
- [ ] Self-generated code governance.
- [ ] Self-generated tool governance.
- [ ] Known-good rollback.
- [ ] Recursive improvement experiments in bounded environments.

This phase begins only after preceding capability, authority and runtime infrastructure is robust.

## Phase 32 — Multi-Agent & Distributed Intelligence Safety

### Goal

Study systems where many NSA agents cooperate, compete or delegate tasks.

### Tasks

- [ ] State-preserving communication.
- [ ] Delegated authority.
- [ ] Distributed provenance.
- [ ] Shared goal conflicts.
- [ ] Coalition safety.
- [ ] Emergent capability monitoring.
- [ ] Distributed rollback.

## Phase 33 — Advanced Self-Model Research

### Goal

Determine whether persistent explicit self-state produces qualitatively different cognitive behaviour.

### Research questions

- [ ] Does self-state improve calibration?
- [ ] Does it improve error detection?
- [ ] Does it improve planning?
- [ ] Does it improve long-horizon reasoning?
- [ ] Does it improve resource allocation?
- [ ] Does it reduce unsafe autonomy?
- [ ] Does the system learn useful predictive models of itself?
- [ ] Can self-state become causally necessary to successful reasoning?

### Scientific boundary

Do not equate these findings with consciousness. Treat consciousness as a separate open question.

## Phase 34 — General NSA Cognitive Substrate

The long-term research objective is to test whether a highly capable AI can operate around a coherent typed substrate combining:

$$
\boxed{\text{World Model}+\text{Self Model}+\text{State Algebra}+\text{Capability System}+\text{Memory}+\text{Values}+\text{Action Governance}}
$$

The goal is not merely a safer LLM. It is a framework in which capability, introspection, information flow, authority, provenance and safety can participate in a common computational language.

---

# Architectural principles

1. **State is computational, not decorative.**
2. **Intelligence does not imply authority.**
3. **Semantic content cannot manufacture hard authority.**
4. **Illegal transitions should be structurally excluded where possible.**
5. **Different state dimensions require appropriate algebras.**
6. **Attention guarantees must not be confused with whole-system guarantees.**
7. **Hard constraints, learned values and statistical monitors are distinct layers.**
8. **Every security theorem must state its assumptions.**
9. **Memory, tools and agents must preserve state semantics.**
10. **Self-state should be causally useful rather than merely textual self-report.**
11. **Every privileged action must cross an explicit trust boundary.**
12. **Self-modification is a state transition and must be governed accordingly.**
13. **Safety and intelligence should be evaluated together.**
14. **The framework must remain modular and model-agnostic.**
15. **Claims must track implementation status and empirical evidence separately.**
16. **Experiments consume the runtime; they do not become alternate runtimes.**
