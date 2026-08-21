# Continuous Cognitive Engine (CCE)

CCE is the opt-in wall-clock scheduler for NSA's persistent cognitive state.

## Boundary

CCE is deliberately **not** a second cognitive or safety implementation. It schedules calls into an existing authoritative transition function. The production binding is `nsa.runtime.cce_adapter.SubstrateTransition`, which delegates each tick to `CognitiveDynamicsSubstrate.step`.

```text
wall clock
    |
    v
   CCE
    |
    v
SubstrateTransition
    |
    v
CognitiveDynamicsSubstrate.step
    |
    +--> epistemic grounding
    +--> predictive self/world simulation
    +--> deliberative governor
    +--> immutable safety kernel
    +--> verified commit / rollback
```

Only `result.new_omega` becomes the next CCE state. The scheduler cannot grant capabilities, mutate hard state directly, or bypass the safety kernel.

## Modes

- **Disabled (default):** no background thread and no automatic ticks.
- **Manual:** enable CCE and call `tick()` explicitly for deterministic experiments.
- **Continuous:** enable CCE, call `start()`, and let the wall-clock loop invoke the transition at the configured cadence.

Calling `set_enabled(False)` stops a running loop and freezes the current state.

## Failure semantics

CCE is **fail-closed by default**. If the authoritative transition raises, the last committed state is preserved, the engine records the error, disables future ticks and stops the automatic loop. `fail_closed=False` remains available only for controlled research experiments.

## Scientific use

This enables a clean comparison between:

1. stateless/model-call-driven inference,
2. explicit state with manual ticks, and
3. explicit state with continuously evolving internal dynamics.

The scheduler itself is not evidence of consciousness or introspection. Those claims require independent behavioural and causal experiments against matched baselines.