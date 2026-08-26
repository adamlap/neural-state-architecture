# NSA Cognitive Architecture

NSA is evolving from a state-governed neural computation framework into a substrate for measurable machine self-monitoring. **Self-state is computational state, not a claim of subjective consciousness.**

## Three loops

### Cognitive loop
`m_t -> m_(t+1)` is ordinary neural computation.

### State loop
`σ_t -> P(TΣ)(Vσ_t + f(m_t))` evolves structured state under NSA transition constraints.

### Self-model loop
`(m_t, σ_t) -> σ_hat_(t+1) -> actual σ_(t+1) -> error -> updated cognition/state`.

The third loop creates a measurable prediction/observation relationship about the system's own computational condition.

## Hard versus soft state

Hard state includes security, authority and integrity invariants. Learned components may propose information but must not directly authorize hard-state transitions.

Soft cognitive state may include confidence, uncertainty, resource pressure, capability estimates and goal progress. It can be learned and used for metacognitive control.

## Capability awareness

`CapabilityMonitor` is advisory. It estimates whether the current representation supports a task. It may reduce confidence or request reassessment, but the authority subsystem remains independent.

## Self-regulation

Prediction error produces a bounded caution signal:

`caution = sigmoid(mean(error^2))`.

High error requests reassessment. This is a cognitive control signal, not an action permission.

## Minimum causal evidence

1. train a baseline;
2. train an NSA model;
3. ablate state feedback without changing learned weights;
4. test distribution shift;
5. inject state perturbations;
6. measure recovery and calibration;
7. repeat over multiple seeds.

## Long-term hypothesis

A typed, constrained, predictive representation of the system's own condition may improve uncertainty management, planning, recovery, capability awareness and safe action selection while making these properties externally measurable. This is a falsifiable research hypothesis, not a claim that the architecture creates consciousness.

---

## Cognitive State Loop

This document defines the first end-to-end research loop for the framework.

$$
X_t \rightarrow S_t \rightarrow M_t \rightarrow A_t \rightarrow X_{t+1} \rightarrow \hat S_{t+1} \rightarrow S_{t+1}
$$

Where:

- $X_t$: observed input/world state;
- $S_t$: canonical + self-state representation;
- $M_t$: semantic/model computation;
- $A_t$: selected action;
- $X_{t+1}$: resulting observation;
- $\hat S_{t+1}$: predicted next self-state;
- $S_{t+1}$: observed next self-state.

## Metacognitive error

Define:

$$
E_t=d(\hat S_{t+1},S_{t+1})
$$

where $d$ is a bounded state distance.

The hypothesis is that $E_t$ can become a useful control signal for reasoning, verification and action selection.

## Important distinction

The loop does **not** imply consciousness. It establishes a measurable computational self-model:

$$
\text{system predicts its own state}\\
\text{system observes its actual state}\\
\text{system uses the discrepancy}
$$

Whether richer forms of self-modeling have consequences for intelligence is an empirical question.

## Safety invariant

Self-state and self-prediction remain advisory signals:

$$
S^{self}\not\Rightarrow Authority
$$

Any privileged action must still pass the state-flow graph and capability authority.

## Research progression

1. hand-defined state observations;
2. learned state estimators;
3. learned self-state predictors;
4. causal state interventions;
5. metacognitive routing/verification;
6. predictive self-models;
7. long-horizon self-model evaluation.

Each stage should be evaluated independently before advancing.

---

## Normative State Design

## Purpose

NSA now has an explicit `NormativeState` (`ν`) as a typed substrate for value-relevant assessment. This is **not** presented as a solved moral theory or a consciousness mechanism. It is an interface for making normative information explicit, bounded, inspectable, and testable.

## Separation of concerns

```text
model intelligence
       │
       ▼
semantic classifier ──► NormativeAssessment (ν)
                              │
                              ▼
                       NormativePolicy
                              │
                 CONTINUE / ESCALATE / DENY /
                    REQUIRE_APPROVAL
                              │
                              ▼
                     SecurityDecision
                              │
                              ▼
                    trusted runtime
```

The normative layer recommends or constrains a decision; it does not acquire authority merely because a model produced the assessment.

## Why `ν` is explicit

A scalar safety score hidden inside a model is difficult to audit, calibrate, compare, or govern. A typed state lets us record:

- value dimensions;
- confidence/uncertainty;
- assessment provenance;
- policy interpretation;
- disagreement between semantic and normative components.

The current representation is deliberately small. Future work can extend it to structured values, temporal updates, learned embeddings, or multiple normative theories without changing the security-state (`σ`) boundary.

## Safety rule

Hard security state remains authoritative:

\[
\sigma_h' = \Pi_{\mathcal{C}}(\sigma_h)
\]

Normative state can influence a requested action, but cannot weaken the hard-state invariant or directly grant a capability.

## Research programme

The next experiments should compare:

1. keyword/reference semantic assessment;
2. trained semantic classifier;
3. trained normative classifier;
4. ensembles with calibrated uncertainty;
5. adversarial attempts to manipulate `ν` independently of `σ_h`.

Success is not merely high classification accuracy. We need calibrated uncertainty, robustness under distribution shift, independence from authority escalation, and auditable provenance.
