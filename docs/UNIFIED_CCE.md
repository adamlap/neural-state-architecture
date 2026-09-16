# Unified CCE transaction architecture

This document defines the convergence point for NSA state, cognition and continuous execution.

## Canonical invariant

**Model cognition proposes. Policy and authority govern. Safety validates. Structural validation validates state transitions. Effects execute only after all configured gates pass. Canonical state commits once.**

```text
observation → belief → prediction/self-model → prediction error
→ information need → deliberation → policy → capability verification
→ immutable safety kernel → structural validation → effect prepare
→ capability consume → effect commit → canonical state commit
→ provenance / trajectory
```

Hard authority can only change through an explicitly authorized `StateTransition`. Soft cognitive channels may evolve, but they never grant hard authority.

## Converged components

- `CanonicalState`: stable typed state boundary.
- `TransitionProposal` / `TransitionReceipt` / `TransitionValidator`: atomic proposals, validation and state digests.
- `CapabilityAuthority`: authenticated capability lifecycle with separate verify/consume operations.
- `CapabilityConstraintEvaluator`: typed target, scope, risk, resource and rate constraints.
- `ImmutableSafetyKernel`: deterministic reference monitor.
- `CognitiveTransactionEngine`: canonical transaction coordinator.
- `CognitiveLoop`: observe → believe → predict → error → information seeking → deliberate → govern → commit.
- `TwoPhaseExecutor`: explicit prepare/commit/abort/compensate boundary for non-idempotent effects.
- `TrajectoryJournal`: append-only JSONL durability with complete canonical state snapshots and replay verification.
- `OmegaFeedbackAdapter`: one-way projection of rich neural/self-model telemetry into governed canonical proposals.
- `CanonicalOmegaAdapter`: explicit CanonicalState → Omega bridge.
- `SixLayerCanonicalAdapter`: neural substrate output is a proposal, never an authority path.
- `NSARuntime.canonical_cce`: public runtime access to the canonical transaction plane and trajectory.

## Capability lifecycle

1. **Verify** — authenticate action, tier, expiry, replay status and signed constraints.
2. **Govern** — policy, safety and structural gates may reject without consuming the nonce.
3. **Consume** — all required capabilities are re-verified and consumed immediately before effect commit.

Multiple required capabilities are verified before any are consumed, preventing partial consumption when one capability is invalid.

## Typed capability constraints

Signed constraints can express maximum risk, reversible-only actions, target allow-lists, scope allow-lists, maximum resource cost and bounded calls per time window. Constraint evaluation only narrows an authenticated capability; it never grants authority.

## Two-phase effects

Non-idempotent integrations can implement `prepare`, `commit`, `abort` and `compensate`. Preparation occurs only after governance and structural validation. Capability consumption occurs immediately before effect commit. If effect commit fails, canonical state does not commit. If an unexpected canonical commit failure occurs after an effect succeeds, the engine requests compensation and records the effect receipt.

Legacy callables remain supported, but integrations with externally visible side effects should use `TwoPhaseExecutor`.

## Neural / Omega boundary

The neural six-layer substrate remains useful for simulation, epistemic reasoning and self-modeling, but it is not the canonical state owner. `CanonicalOmegaAdapter` provides a one-way representation bridge. `OmegaFeedbackAdapter` turns epistemic/self-model telemetry into a `TransitionProposal` that must pass canonical gates.

No tensor output can directly mutate canonical hard authority.

## Information seeking

High uncertainty is represented explicitly through `InformationNeed`. `InformationSeekingPlanner` can generate a governed evidence-gathering candidate when no ordinary candidate is available. Information gathering remains subject to policy, capability, safety and structural validation.

## Durable replay

Trajectory records can persist complete canonical state under versioned `nsa.canonical-state.v1`. The codec reconstructs semantic data, hard authority, soft state, provenance, goals and logical step. Journal verification re-hashes reconstructed snapshots and rejects corruption or schema mismatches. Non-JSON semantic objects are rejected rather than silently serialized lossily.

## PyTorch isolation

Canonical `nsa`, `nsa.core`, `nsa.cce` and `nsa.cognition` imports no longer eagerly load tensor-based substrate/Omega modules. Neural integrations are lazy and activate only when explicitly requested, keeping the control plane suitable for deterministic lightweight deployments.

## Production key management

`CapabilityAuthority.from_environment()` requires an explicit `NSA_CAPABILITY_MASTER_SECRET` (configurable variable name). The historical demo key remains only for backwards-compatible research fixtures and must not be used for production deployments.

## Verification strategy

The hardening suite covers state codec round-trips, journal reconstruction, typed capability constraints, rejected-policy nonce preservation, two-phase effect ordering and neural import isolation, in addition to the existing adversarial governance suites.

CI remains authoritative for the complete repository verification; this integration environment cannot execute the GitHub checkout locally.
