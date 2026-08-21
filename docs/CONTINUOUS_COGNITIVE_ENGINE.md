# Continuous Cognitive Engine (CCE)

CCE is the opt-in wall-clock scheduler for NSA's persistent cognitive state.

## Boundary

CCE is deliberately **not** a second cognitive or safety implementation. It schedules calls into an existing authoritative transition function. In the full NSA runtime this is intended to be the existing `CognitiveDynamicsSubstrate.step` path, which already contains epistemic evaluation, predictive simulation, the deliberative governor, the immutable safety kernel, and state commit/rollback.

Therefore:

```text
wall clock
    |
    v
   CCE ----> one authoritative NSA transition ----> new state
    |                         |
    +-------------------------+---- safety kernel remains authoritative
```

The scheduler cannot grant capabilities, mutate hard state directly, or bypass the safety kernel.

## Modes

- **Disabled (default):** no background thread and no automatic ticks.
- **Manual:** enable CCE and call `tick()` explicitly for deterministic experiments.
- **Continuous:** enable CCE, call `start()`, and let the wall-clock loop invoke the transition at the configured cadence.

Calling `set_enabled(False)` stops a running loop and freezes the current state.

## Scientific use

This enables a clean comparison between:

1. stateless/model-call-driven inference,
2. explicit state with manual ticks, and
3. explicit state with continuously evolving internal dynamics.

The scheduler itself is not evidence of consciousness or introspection. Those claims require independent behavioural and causal experiments against matched baselines.
