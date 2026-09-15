# Unified CCE transaction architecture

This document defines the convergence point for NSA's state, cognition and
continuous execution work.

## Invariant

The model is a proposer, not an authority. A CCE tick follows:

```text
observation
    -> cognitive proposal
    -> action candidates
    -> policy decision
    -> structural transition validation
    -> external execution (if permitted)
    -> canonical state commit
    -> provenance / trajectory record
```

Hard authority can only change through an explicitly authorized
`StateTransition`. Soft cognitive channels may evolve, but they never grant
hard authority.

## Components

- `nsa.core.state.CanonicalState`: stable typed state boundary.
- `nsa.core.transition`: proposal validation, atomic application and state digest.
- `nsa.cce.events`: typed cognitive event vocabulary.
- `nsa.cce.trajectory`: append-only in-memory transition history.
- `nsa.cce.transaction.CognitiveTransactionEngine`: one-tick transaction coordinator.
- `nsa.cce.canonical_runtime.CanonicalCCERuntime`: scheduler + transaction engine.
- `nsa.cognition.interfaces`: model-agnostic prediction, belief, action and
  information-gain protocols.
- `nsa.cognition.deliberation`: uncertainty-sensitive action selection.

## Why this is the convergence layer

The repository contains richer experimental `UnifiedCognitiveState` and
six-layer substrate implementations. They should progressively adapt to this
transaction boundary rather than creating parallel state/commit paths.

A future substrate adapter should translate its rich internal state into a
canonical proposal, invoke the same policy/capability/safety gates, and commit
through the same transaction engine.

## Safety boundary

Execution is deliberately placed after policy and structural validation. A
future capability/safety adapter should become another mandatory gate before
`executor`, not be implemented inside the model or scheduler.

Execution failure produces no canonical state commit. Successful execution is
recorded as an event, while the next state is still created immutably through
the validator.

## Next convergence steps

1. Adapt the existing six-layer `CognitiveDynamicsSubstrate` to emit canonical
   proposals rather than owning a parallel `UnifiedCognitiveState` commit.
2. Add capability and immutable safety-kernel hooks to `CognitiveTransactionEngine`.
3. Persist `CognitiveTrajectory` records alongside existing checkpoints.
4. Add belief/prediction/self-model adapters and prediction-error events.
5. Add goal-driven information-seeking action generation.
6. Make the public `NSA` runtime expose this transaction trajectory without
   making PyTorch mandatory for the base package.
