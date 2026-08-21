# Continuous Cognitive State Research Path

> Parallel research and development track for persistent, continuously evolving, self-monitoring cognitive systems governed by NSA.

## Research Status

**Status:** OPEN-RESEARCH

This track investigates whether an AI system can maintain a persistent dynamical cognitive state rather than operating only as independent input-to-output inference calls. It does **not** assume that persistent state constitutes consciousness. Consciousness remains an open scientific question.

## Core Hypothesis

A useful next step beyond stateless or request-scoped inference is a persistent cognitive runtime in which internal state evolves continuously and external inputs act as asynchronous perturbations:

$$
\frac{dX}{dt}=F_\theta(X(t),I(t),G(t))
$$

where $X$ contains cognitive state, memory, goals, self-state and governed operational state.

The security hypothesis is that the cognitive dynamics and the authoritative NSA state should remain distinct:

$$
X_t=(C_t,M_t,G_t,S_t,\Sigma_t)
$$

with $C,M,G,S$ representing learned/semantic state and $\Sigma$ representing authoritative hard/soft governance state.

Candidate state evolution is permitted only through the NSA transition boundary:

$$
\Sigma_{t+\Delta t}=P_{\mathcal T_\Sigma}\left(G_\theta(C_t,\Sigma_t,I_t)\right)
$$

The cognitive system may be highly expressive while security-critical state remains structurally constrained.

## Architectural Model

```text
Sensors / Events
     |
     v
Perception + provenance + confidence
     |
     v
Persistent Cognitive Runtime
     |
     +--> working memory
     +--> long-term typed memory
     +--> goals / intentions
     +--> self-state / self-model
     +--> internal simulation
     +--> LLM reasoning substrate (initially Ollama)
     |
     v
Candidate state / action proposal
     |
     v
NSA Governor
     |
     +--> hard state invariants
     +--> integrity
     +--> authorization / capabilities
     +--> provenance
     +--> license
     +--> confidence / uncertainty
     +--> risk
     |
     +--> ALLOW
     +--> HOLD / REQUEST AUTHORITY
     +--> DENY
     |
     v
Governed output / actuator
```

## Continuous State Clock

The runtime should be able to tick without external input. External events are asynchronous perturbations rather than the only source of computation.

$$
C_{t+\Delta t}=C_t+\Delta t\,F_\theta(C_t,M_t,G_t,S_t,\hat I_t)
$$

where $\hat I_t=\varnothing$ is valid.

The prototype should support configurable internal frequencies and event-driven wakeups. The first implementation may use discrete numerical integration; the research question is persistent state dynamics, not a claim that the underlying hardware is literally continuous.

## LLM / Ollama Integration

The first prototype should **not** require modifying transformer weights or the Ollama server.

Ollama can act as the reasoning substrate inside a persistent runtime:

1. Runtime maintains persistent state.
2. Internal clock evaluates whether cognitive computation is needed.
3. Relevant state is projected into the model's context/interface.
4. Ollama performs reasoning or proposes a state update/action.
5. Runtime commits semantic state updates.
6. NSA independently validates authoritative state transitions and privileged actions.
7. The runtime continues ticking after the inference call completes.

Later research can investigate deeper latent-state coupling, recurrent/SSM architectures and native continuous-time neural dynamics.

## Self-State and Self-Model

A dedicated self-state should represent machine state without relying on textual self-report alone:

$$
S_t=(C_t,\Sigma_t,K_t,G_t,R_t,E_t)
$$

where components may represent current cognitive state, governed state, known capabilities, active goals, resources and recent/error state.

A predictive self-model can estimate future internal state:

$$
\hat S_{t+1}=F_{self}(S_t,a_t,I_t)
$$

Research should test whether self-state becomes causally useful to successful reasoning, planning, calibration and error recovery.

## Action Governance

The model must never be the final authority over privileged effects.

```text
LLM reasoning
    -> typed action proposal
    -> NSA policy/capability evaluation
    -> allow / hold / deny
    -> actuator
```

Actions should carry provenance, confidence, risk, requested capability, target resource and reversibility. High-impact actions should support explicit human approval and transaction boundaries.

## Memory

Persistent memory should be state-aware rather than an untyped side channel.

Each memory record should be capable of carrying:

- provenance
- confidence
- timestamp / temporal validity
- security state
- integrity state
- authorization requirements
- source identity
- transformation history

Retrieval must preserve these semantics into the cognitive state and action boundary.

## Research Program

### R1 — Minimal Persistent Runtime

- [ ] Implement `nsa/runtime/continuous.py` or equivalent.
- [ ] Internal state object with deterministic clock.
- [ ] Event queue for text and sensor inputs.
- [ ] Idle/internal ticks.
- [ ] Checkpoint and restore.
- [ ] Structured state telemetry.

### R2 — Ollama Cognitive Adapter

- [ ] Local Ollama adapter.
- [ ] Structured state/context projection.
- [ ] State update protocol.
- [ ] Action proposal protocol.
- [ ] Streaming input support.
- [ ] Model-agnostic interface so Ollama is replaceable.

### R3 — Persistent Self-State

- [ ] Explicit self-state object.
- [ ] State introspection API.
- [ ] Self-state prediction.
- [ ] Error-state detection.
- [ ] Resource/capability awareness.
- [ ] Self-model calibration tests.

### R4 — Typed Persistent Memory

- [ ] State-tagged episodic memory.
- [ ] Semantic memory.
- [ ] Working memory.
- [ ] Provenance-preserving retrieval.
- [ ] Confidence decay.
- [ ] Temporal validity.
- [ ] Security-aware retrieval.

### R5 — NSA Governance Integration

- [ ] Continuous state transition validation.
- [ ] Hard/soft state separation.
- [ ] Capability-gated actions.
- [ ] Provenance propagation.
- [ ] Risk/confidence gates.
- [ ] Structural illegal-transition tests.
- [ ] Output-boundary enforcement.

### R6 — Sensory Interfaces

- [ ] Speech-to-text input.
- [ ] Text input.
- [ ] Vision/event adapter.
- [ ] Environmental telemetry.
- [ ] Timestamped asynchronous event stream.
- [ ] Input provenance and confidence.

### R7 — Governed Output Interfaces

- [ ] Text output.
- [ ] Text-to-speech output.
- [ ] Tool/API actions.
- [ ] Home/IoT integration in a sandbox.
- [ ] Filesystem actions in a sandbox.
- [ ] Explicit capability classes.
- [ ] Human approval for high-impact actions.

### R8 — Continuous Cognition Experiments

Compare matched systems:

$$
B: y_t=f(x_t)
$$

$$
P: h_{t+1}=F(h_t,x_t)
$$

$$
N: (h_{t+1},\Sigma_{t+1})=F_{NSA}(h_t,\Sigma_t,x_t)
$$

Measure memory, long-horizon reasoning, planning, calibration, error recovery, interruption recovery, goal persistence, tool safety and compute efficiency.

### R9 — Attractor / Internal Simulation Research

- [ ] Persistent internal deliberation.
- [ ] Attractor-state experiments.
- [ ] Multi-timescale state dynamics.
- [ ] Counterfactual simulation.
- [ ] Internal prediction-error signals.
- [ ] Event-driven output thresholds.

### R10 — Deeper Neural Integration

Only after runtime experiments establish a useful effect:

- [ ] Recurrent latent-state model.
- [ ] SSM/state-space integration.
- [ ] Continuous-time neural dynamics.
- [ ] Learned state transition fields.
- [ ] Latent state persistence across inference boundaries.
- [ ] Native NSA-constrained recurrent architecture.

## Consciousness Research Boundary

The project should explicitly separate three claims:

1. **Persistent computation:** the system maintains and evolves internal state over time.
2. **Self-modeling / metacognition:** the system maintains predictive representations of its own state and capabilities.
3. **Conscious experience:** the system has subjective experience.

R1–R10 can experimentally investigate the first two. The third remains an open scientific and philosophical question and must not be inferred from self-reports, conversational fluency or apparent personality alone.

## Safety Principle

The continuous cognitive loop must never weaken NSA's authority model.

In particular:

$$
\text{cognitive state}\not\Rightarrow\text{hard authority}
$$

and:

$$
\text{model output}\not\Rightarrow\text{privileged action}
$$

The runtime and NSA governor remain outside the model's generated semantic state and form the trusted boundary for consequential actions.

## Success Criteria

The research path is successful if it produces reproducible evidence for one or more of the following:

- persistent state improves long-horizon reasoning;
- self-state improves calibration or error detection;
- continuous internal processing improves planning or memory;
- typed state improves safe autonomy;
- NSA governance remains invariant under continuous operation;
- governed persistent agents can operate safely with real sensory and actuator interfaces;
- deeper recurrent/continuous dynamics provide measurable benefits over runtime-only persistence.

No consciousness claim is required for technical success.

## Relationship to Main NSA Roadmap

This is a **parallel R&D path**, not a replacement for the existing NSA roadmap. It directly extends and cross-connects:

- Phase 11 — Canonical Typed Neural Core
- Phase 13 — Algebra-Preserving State Transition Engine
- Phase 17 — Persistent Typed Memory
- Phase 18 — Self-State & Metacognition
- Phase 19 — Predictive Self-Model & Internal Simulation
- Phase 20 — Tool & Action Governance
- Phase 21 — Trusted Cognitive Runtime
- Phase 25 — Dynamic Auditing & Recovery
- Phase 33 — Advanced Self-Model Research
- Phase 34 — General NSA Cognitive Substrate

The purpose of this branch is to develop the continuous-state hypothesis independently while preserving compatibility with the main NSA algebra and governance architecture.