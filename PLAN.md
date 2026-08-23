# Neural State Architecture (NSA): Full Framework Master Plan

> **Strategic roadmap from intrinsic neural security to a typed cognitive substrate for advanced AI**

---

## Executive Direction

NSA began as a framework for intrinsic information-flow security in neural networks. The long-term research direction is broader: develop a modular computational substrate in which **semantic cognition, hard policy state, uncertainty, provenance, values, authority, memory, tools, self-state and autonomous action** can all participate in a common typed-state architecture.

The central hypothesis is:

> **A sufficiently expressive representation of machine state can simultaneously make important classes of unsafe computation structurally harder or impossible and improve an AI system's ability to reason about its own knowledge, capabilities, constraints and actions.**

This plan therefore evolves NSA from a neural architecture into a **full framework for safe and highly capable AI**.

The project must remain scientifically conservative: consciousness, AGI safety and superintelligence safety are research targets, not claims of solved problems.

---

# Part I — Completed Foundation

The repository already contains substantial work across the original NSA research program.

## Phase 1: Formal Mathematical Whitepaper & Theoretical Rigor — COMPLETE

- [x] Python prototype implementation (`nsa/algebra.py`, `nsa/state.py`, `nsa/attention.py`, `nsa/layers.py`, `nsa/objectives.py`).
- [x] Initial toy privacy experiment and unit test suite.
- [x] Adversarial leakage attack and multi-tier lattice benchmarks.
- [x] Draft theoretical framework and whitepaper.
- [x] Formal LaTeX whitepaper.
- [x] Formal non-interference theorem under stated hard-mask assumptions.

## Phase 2: Fused GPU Kernels & Scale Validation — COMPLETE / CONTINUING VALIDATION

- [x] Fused GPU attention operator.
- [x] Triton state-aware attention kernel with PyTorch fallback.
- [x] GPU benchmark harness.
- [x] KV-cache-aware fused execution path.
- [ ] Large-scale independent reproduction across hardware and model families.

## Phase 3: NSA-LoRA Retrofitting & Security Benchmarks — FOUNDATION COMPLETE

- [x] NSA-LoRA implementation.
- [x] Retrofit experiments.
- [x] Prompt-injection security benchmark infrastructure.
- [x] Open-LLM retrofit simulation.
- [ ] Replace all toy/simulated claims with reproducible real-checkpoint results where possible.

## Phase 4: Open Source Ecosystem — FOUNDATION COMPLETE

- [x] Hugging Face integration.
- [x] KV-cache integration.
- [x] vLLM/SGLang integration direction.
- [x] Production documentation foundation.

## Phase 5: Live Model Showcase & CUDA-Fused Performance — FOUNDATION COMPLETE

- [x] Live Hugging Face retrofit path.
- [x] `NSAMaskInjector`.
- [x] KV-cache generation path.
- [x] Baseline vs NSA demonstration infrastructure.
- [x] Fused/naive comparison infrastructure.
- [ ] Expand validation beyond showcase experiments.

## Phase 6: Neural Metadata Propagation & Enterprise Governance — COMPLETE FOUNDATION

- [x] Multi-dimensional state vectors.
- [x] Multi-dimensional lattice.
- [x] Threat model and information-flow scope.
- [x] Ingress governance concepts.
- [x] Multi-state algebra tests.

## Phase 7: Native TNC vs Retrofit Research — COMPLETE FOUNDATION

- [x] Controlled baseline/retrofit/native harness.
- [x] Native TNC research path.
- [x] Technical research guide.
- [x] Makefile experiment integration.
- [ ] Expand to statistically rigorous multi-seed and multi-model studies.

## Phase 8: Value Layer & Alignment Substrate — FOUNDATION COMPLETE

- [x] Distinction between hard constraints and learned values.
- [x] `$h=(m,\sigma,\nu)$` architecture.
- [x] Value alignment loss/projector.
- [x] Initial four-way benchmark.
- [ ] Heterogeneous algebraic domains.
- [ ] Moral uncertainty representation.
- [ ] Deliberative value revision.

## Phase 9: NSA 2.0 Dynamic Auditing & Recovery — FOUNDATION COMPLETE

- [x] Dynamic state tracking.
- [x] Multi-layer residual probing.
- [x] KV-cache rollback direction.
- [x] Recovery adapters and semantic pivots.
- [x] Compartmented stream routing.
- [x] Modular verifier subsystem.

## Phase 10: Formal NSA Core & Peer-Review Hardening — FOUNDATION COMPLETE

- [x] Authoritative quad-tuple activations.
- [x] Hard/soft state partition.
- [x] Privilege escalation prevention.
- [x] Capability validation foundation.
- [x] Exact legal transition projection.
- [x] Observational-equivalence theorem formulation.
- [x] Tier 1 structural vs Tier 2 statistical distinction.
- [x] Execution-state rollback model.
- [x] Output boundary TCB concept.

---

# Part II — Full NSA Framework Roadmap

The following phases define the new long-term architecture. Each phase should be implemented as an independently testable module and should expose clear interfaces to the existing NSA core.

---

## Phase 11 — Canonical Typed Neural Core — COMPLETE

**Target:** `nsa/core/`

### Goal

Unify the existing state representations into a canonical typed activation model:

$$
h=(m,\sigma_h,\sigma_s,\nu,\kappa,\pi,g)
$$

where optional components represent hard policy, soft operational state, values, capabilities, provenance and goals.

### Tasks

- [x] Define canonical typed activation protocol.
- [x] Separate hard trusted state from model-generated semantic state.
- [x] Define read/write permissions.
- [x] Define state composition semantics.
- [x] Support partial activation/state vectors.
- [ ] Establish compatibility with current `StateVector`, `MultiStateVector` and quad-tuple APIs.
- [x] Add serialization and versioning.

### Success criterion

Every future NSA subsystem can consume and return a well-defined typed state without inventing its own incompatible representation.

---

## Phase 12 — General State Algebra Engine — COMPLETE

**Target:** `nsa/algebra/`

### Goal

Generalize the current security lattice into a heterogeneous algebra framework.

### Tasks

- [x] Lattice domains.
- [x] Boolean/capability domains.
- [x] Probabilistic domains.
- [x] Temporal domains.
- [x] Constraint domains.
- [x] Coordinate-wise and heterogeneous product algebras.
- [x] Join/meet compatibility.
- [x] Legal transition cones.
- [x] Exact algebraic projections.
- [x] Algebra property test suite.

### Research question

Can heterogeneous state dimensions remain compositional without reducing everything to a scalar score?

---

## Phase 13 — Algebra-Preserving State Transition Engine — COMPLETE

**Target:** `nsa/transitions/`

### Goal

Make state invariants structural rather than dependent solely on learned penalties.

### Core pattern

$$
\sigma_{t+1}=\sigma_t\sqcup\Delta_\theta(m_t,\sigma_t)
$$

### Tasks

- [x] Initial algebra-preserving transition prototype.
- [x] Generalize to all heterogeneous state dimensions.
- [x] Prove dimension-specific invariants.
- [x] Benchmark capability cost vs unconstrained transitions.
- [x] Integrate with native TNC.
- [x] Integrate with retrofit path where feasible.

---

## Phase 14 — Whole-System Information Flow — COMPLETE FOUNDATION

**Target:** `nsa/flow/`

### Goal

Move beyond attention-only security to whole-network and whole-runtime information-flow control.

### Tasks

- [x] Residual-stream taint semantics.
- [x] FFN taint semantics.
- [x] Cross-layer flow analysis.
- [x] State-aware residual composition.
- [x] Formal sink/source policies.
- [x] Declassification semantics.
- [x] Whole-model non-interference conditions.
- [x] Counterexample generator for violated assumptions.

### Success criterion

Every security theorem states exactly which computational pathways it covers.

---

## Phase 15 — Capability & Authority Architecture — COMPLETE

**Target:** `nsa/capabilities/`

### Goal

Make authority independent from intelligence.

$$
\boxed{\text{Intelligence}\neq\text{Authority}}
$$

### Tasks

- [x] Signed capability objects.
- [x] Resource/action scopes.
- [x] Purpose binding.
- [x] Expiry and nonce.
- [x] Delegation.
- [x] Revocation.
- [x] Capability-state compatibility.
- [x] External trusted issuer.
- [x] Semantic-to-authority isolation tests.

### Core invariant

$$
m_t\not\rightarrow\sigma_{h,t+1}
$$

without an authorized capability-mediated transition.

---

## Phase 16 — Provenance, Trust & Epistemic State — COMPLETE

**Target:** `nsa/provenance/`

### Goal

Give every important piece of information an explicit lineage and trust state.

### Tasks

- [x] Source identity.
- [x] Transformation history.
- [x] Evidence graphs.
- [x] Confidence propagation.
- [x] Contradiction handling.
- [x] Trust domains.
- [x] Provenance-aware state transitions.
- [x] Audit export.

### Success criterion

The system can distinguish information by origin, transformation history and confidence rather than treating all context as equivalent.

---

## Phase 17 — Persistent Typed Memory — COMPLETE

**Target:** `nsa/memory/`

### Goal

Extend NSA state semantics into RAG, vector stores and long-term memory.

### Tasks

- [x] State-tagged memory records.
- [x] Provenance-preserving retrieval.
- [x] Tenant isolation.
- [x] Read/write policy.
- [x] Memory declassification.
- [x] Temporal validity.
- [x] Confidence decay.
- [x] State-aware vector retrieval.
- [x] Memory audit trail.

### Success criterion

A secure neural core cannot be bypassed by an untyped external memory layer.

---

## Phase 18 — Self-State & Metacognition — COMPLETE

**Target:** `nsa/self_state/`

### Goal

Investigate whether explicit awareness of machine state can improve reasoning and safety.

### Proposed state

$$
S_t=(m_t,\sigma_t,\kappa_t,\pi_t,g_t,\rho_t)
$$

### Tasks

- [x] Persistent self-state representation.
- [x] State introspection interface.
- [x] Self-state prediction.
- [x] Confidence awareness.
- [x] Capability awareness.
- [x] Resource awareness.
- [x] Error-state detection.
- [x] State-conditioned reasoning depth.
- [x] Metacognitive training objectives.

### Core experiment

Compare a baseline model:

$$m_{t+1}=F(m_t)$$

against an explicit-state model:

$$
(m_{t+1},\sigma_{t+1})=F(m_t,\sigma_t)
$$

under matched parameter and compute budgets.

### Success criterion

Demonstrate measurable gains in calibration, error detection, planning or reasoning without relying on textual self-report alone.

---

## Phase 19 — Predictive Self-Model & Internal Simulation — COMPLETE

**Target:** `nsa/self_model/`

### Goal

Allow the AI to predict consequences for both the external world and its own future state.

### Tasks

- [x] Future state prediction.
- [x] Action consequence simulation.
- [x] Capability-state prediction.
- [x] Resource prediction.
- [x] State prediction error.
- [x] Counterfactual simulation.
- [x] Self-model calibration.

### Planning objective

$$
a^*=\arg\max_a U(m_{t+1},\sigma_{t+1},g_{t+1})
$$

subject to legal state constraints.

---

## Phase 20 — Tool & Action Governance

**Target:** `nsa/actions/`

### Goal

Extend state safety to real-world effects.

### Tasks

- [x] Typed tool requests.
- [x] Capability checks.
- [x] State propagation across tool boundaries.
- [x] Action risk classification.
- [x] Human approval gates.
- [x] Reversible/irreversible action distinction.
- [x] Resource limits.
- [x] Transaction boundaries.
- [x] Output sink policy.

### Success criterion

The same security/state semantics apply from neural computation through external action.

---

## Phase 21 — Trusted Cognitive Runtime

**Target:** `nsa/runtime/`

### Goal

Turn NSA into a trusted execution environment for autonomous agents.

### Tasks

- [x] Execution state object.
- [x] State-aware scheduling.
- [x] Tool routing.
- [x] Memory routing.
- [x] Capability lifecycle.
- [x] Checkpointing.
- [x] Rollback.
- [x] Resource budgets.
- [x] Failure containment.
- [x] Output-boundary TCB.

### Success criterion

The runtime, not the model's generated text, remains the final authority over privileged actions.

---

## Phase 22 — Multi-Agent State Protocol

**Target:** `nsa/multi_agent/`

### Goal

Preserve security, provenance and authority when intelligent systems communicate.

### Tasks

- [x] Agent identity.
- [x] Cross-agent state transfer.
- [x] Capability delegation.
- [x] State translation.
- [x] Trust negotiation.
- [x] Information-flow contracts.
- [x] Shared-memory governance.
- [x] Distributed audit logs.

### Success criterion

Agent boundaries do not silently erase state semantics.

---

## Phase 23 — Value & Alignment Substrate

**Target:** `nsa/alignment/`

### Goal

Expand the existing value layer into a modular alignment substrate.

### Tasks

- [x] Separate hard constraints from values.
- [x] Initial value state and preference loss.
- [x] Utility representations.
- [x] Value uncertainty.
- [x] Preference learning interfaces.
- [x] Conflict detection.
- [x] Deliberative value revision.
- [x] Value-state auditability.

### Principle

$$
\text{Hard constraints}\neq\text{learned values}
$$

---

## Phase 24 — Normative & Moral Uncertainty

**Target:** `nsa/normative/`

### Goal

Investigate reasoning under uncertainty about competing value systems without weakening hard constraints.

### Proposed representation

$$
P(T_i\mid x)
$$

### Tasks

- [x] Multiple normative models.
- [x] Moral uncertainty representation.
- [x] Value conflict.
- [x] Deliberation.
- [x] Human preference incorporation.
- [x] Hard-constraint precedence.
- [x] Uncertainty-aware action selection.

---

## Phase 25 — Dynamic Auditing & Recovery

**Target:** `nsa/audit/`

### Goal

Maintain Tier 2 statistical protection for properties that cannot be guaranteed structurally.

### Tasks

- [x] Multi-layer probing foundation.
- [x] Rollback foundation.
- [x] Recovery adapters.
- [ ] Formal detection-delay benchmarks.
- [ ] Distribution-shift detection.
- [ ] Self-state anomaly detection.
- [ ] Automated recovery policies.
- [ ] Recovery safety proofs.

---

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

Security claims can be traced from theorem → assumption → implementation → test/proof artifact.

---

## Phase 27 — Security Research & Adversarial Evaluation

**Target:** `nsa/security/`

### Goal

Continuously attack the framework.

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

### Principle

A successful attack is a research result that identifies a missing invariant, an incorrect assumption or an implementation boundary.

---

## Phase 28 — Joint Safety & Intelligence Evaluation

**Target:** `nsa/eval/`

### Goal

Measure capability and safety together.

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

NSA should not be evaluated only as a security tax. We should test whether state provides a useful inductive bias for intelligence.

---

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

---

## Phase 30 — Ecosystem Integration

**Target:** `nsa/integrations/`

### Tasks

- [ ] Hugging Face.
- [ ] PyTorch.
- [ ] vLLM.
- [ ] SGLang.
- [ ] RAG/vector databases.
- [ ] Agent frameworks.
- [ ] Enterprise identity systems.
- [ ] Tool/API gateways.

### Goal

Make NSA composable with existing AI infrastructure rather than requiring a new ecosystem.

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

This phase must begin only after the preceding capability/authority/runtime infrastructure is robust.

---

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

---

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

---

## Phase 34 — Toward a General NSA Cognitive Substrate

The ultimate research objective is to test whether a highly capable AI can operate around a coherent typed substrate combining:

$$
\boxed{
\text{World Model}
+
\text{Self Model}
+
\text{State Algebra}
+
\text{Capability System}
+
\text{Memory}
+
\text{Values}
+
\text{Action Governance}
}
$$

The desired outcome is not merely a safer LLM. It is a framework in which **capability, introspection, information flow, authority, provenance and safety are represented in the same computational language**.

---

# Part IV — Architectural Principles

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

---

# Part V — Immediate Build Order

The full framework is intentionally larger than what should be implemented at once. The recommended next sequence is:

### Next 1 — Canonical state API

Unify current scalar, multi-dimensional and quad-tuple representations.

### Next 2 — General algebra package

Make heterogeneous algebraic domains first-class.

### Next 3 — Algebra-preserving transitions

Make the new transition mechanism the canonical native path.

### Next 4 — Whole-system information flow

Close the gap between attention-level guarantees and residual/FFN/runtime flow.

### Next 5 — Capability system

Make authority independent from generated semantics.

### Next 6 — Provenance + typed memory

Carry state beyond the neural context.

### Next 7 — Self-state/metacognition prototype

Build a small controlled experiment comparing ordinary cognition against explicit self-state cognition.

### Next 8 — Tool/action governance + runtime

Connect cognition to real-world effects through the capability system.

### Next 9 — Multi-agent protocol

Preserve state semantics across agents.

### Next 10 — Formal verification + adversarial evaluation

Turn the architecture into a continuously tested research platform.

### Next 11 — Self-modification research

Only after the trusted runtime and invariant system are mature.

---

# Part VI — Evidence Standard & Manifest Integration

NSA enforces a formal, machine-traceable **Epistemic Evidence Standard** tracked in [`evidence/manifest.json`](evidence/manifest.json) and validated via `make evidence`:

| Epistemic Status | Definition & Verification Requirement |
|---|---|
| **IMPLEMENTED** | Code exists, imports cleanly, and executes without runtime exceptions. |
| **UNIT-TESTED** | Local unit tests verify discrete mathematical operations, edge cases, and safety bounds (`pytest tests/`). |
| **EMPIRICALLY-VALIDATED** | Controlled empirical experiments demonstrate the claimed phenomenon under tested model configurations and seeds. |
| **ROBUSTLY-VALIDATED** | Multi-seed, multi-scale, and distribution-shift experiments with statistical bootstrap confidence intervals. |
| **FORMALLY-VERIFIED** | Machine-checkable mathematical proof whose explicit assumptions match the execution environment. |
| **OPEN-RESEARCH** | Active research hypothesis, incomplete whole-system property, or open capability question. |

Every major claim must trace:
$$\text{Claim} \longrightarrow \text{Theorem / Proposition} \longrightarrow \text{Assumptions} \longrightarrow \text{Implementation} \longrightarrow \text{Unit Tests} \longrightarrow \text{Empirical Artifacts} \longrightarrow \text{Epistemic Status}$$

No empirical benchmark should be described as a whole-system theorem, and no theorem should be claimed as proven unless its assumptions cover the full implementation.

---

# Final Objective

The long-term mission of NSA is to investigate whether **safe, secure and highly capable AI can be built around a typed computational substrate rather than protected primarily from the outside**.

The central research hypothesis is:

$$
\boxed{
\text{Safety Architecture}
\not\perp
\text{Intelligence Architecture}
}
$$

Instead, the same state representation may become part of the mechanism by which an advanced AI understands its knowledge, capabilities, limitations, authority, goals and consequences.

That hypothesis — and the engineering and mathematics required to test it — is the long-term direction of the Neural State Architecture project.

---

## High-Impact Application Areas

1. Enterprise multi-tenant privacy and licensing.
2. Secure RAG and provenance-aware knowledge systems.
3. Tool-using autonomous agents.
4. Healthcare and financial compliance.
5. Multi-agent systems.
6. High-assurance AI infrastructure.
7. Self-monitoring and metacognitive AI.
8. Long-horizon autonomous systems.
9. Safety infrastructure for increasingly capable AI.
10. Long-term research into AGI and superintelligence safety.

---

# Part VII — CCE / Continuous Cognition Research Track

**Status:** ACTIVE — implemented runtime and live evaluation infrastructure exist; scientific claims remain open.

The Continuous Cognitive Engine (CCE) is a parallel research path built around the authoritative NSA runtime. It must remain modular and must not weaken or duplicate hard NSA authority.

### CCE architecture

```text
REAL / LIVE INPUT
       |
       v
+--------------------------+
| CCE                      |
| persistent cognitive    |
| state + scheduler        |
+------------+-------------+
             |
             v
   authoritative NSA transition
             |
             v
        canonical state
             |
             +------> REAL Ollama inference
             |                 |
             |                 v
             |          observable proposal
             |                 |
             +-----------------+
                       |
                       v
                 NSA governance
                       |
                 ALLOW / HOLD / DENY
```

### Current implementation

- [x] Opt-in wall-clock continuous execution.
- [x] Explicit disabled/clocked control condition.
- [x] Deterministic/manual stepping for reproducible experiments.
- [x] Lifecycle controls and runtime observability.
- [x] Non-overlapping transition execution.
- [x] Fail-closed execution on authoritative transition errors.
- [x] Freeze last valid state on transition failure.
- [x] CCE restricted to scheduling; it cannot grant capabilities or become hard authority.
- [x] CCE connected to the authoritative NSA substrate.
- [x] Real Ollama inference path.
- [x] Matched baseline-vs-NSA live Ollama benchmark.
- [x] Clocked-vs-continuous CCE live Ollama benchmark.
- [x] Core CCE invariant tests in the normal workflow.
- [x] Dedicated live Ollama/CCE workflow with archived JSON artifacts.
- [x] Multi-seed predictive self-model evaluation infrastructure.

### Experimental conditions

All claims about continuous cognition should use matched controls:

1. **Baseline:** live Ollama without persistent NSA state.
2. **Persistent:** live Ollama with persistent canonical state but no autonomous scheduler.
3. **Clocked CCE:** deterministic or finite-step CCE execution.
4. **Continuous CCE:** wall-clock CCE execution enabled.

Hold model, prompt/task, sampling configuration, token budget and inference budget constant wherever practical.

### Next evidence required

- [ ] Run the live workflow across multiple independent seeds.
- [ ] Repeat across at least two model families/sizes where practical.
- [ ] Aggregate predictor-vs-persistence and clocked-vs-continuous results with effect sizes and uncertainty.
- [ ] Measure calibration and error-detection utility separately from prediction MSE.
- [ ] Measure long-duration continuous-state stability and transition latency.
- [ ] Test whether continuous execution improves planning, memory or task performance under matched compute.
- [ ] Add counterfactual action-consequence prediction.
- [ ] Add real asynchronous speech/vision deployment against the canonical runtime.
- [ ] Treat consciousness as an open research question rather than an implementation claim.

### Workflow policy

Feature work in this track should be developed on a branch and submitted as a PR so the repository's GitHub Actions suite exercises the feature before merge. Deterministic CCE invariants belong in the normal test workflow; real Ollama experiments belong in the dedicated live workflow because they require an actual model runtime and model download.

The live workflow must fail if real Ollama execution is unavailable rather than silently substituting a mock backend. Results must be uploaded as workflow artifacts and interpreted from observed data only.

### Scientific interpretation

Continuous execution, persistent memory, self-referential language or predictive state are not by themselves evidence of phenomenal consciousness. The scientific target is measurable causal/computational change: whether persistent and continuously evolving state improves prediction, calibration, error detection, planning, memory, long-horizon performance or safety under matched controls.
