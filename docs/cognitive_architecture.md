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
