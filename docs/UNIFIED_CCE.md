# Unified CCE transaction architecture

This document defines the convergence point for NSA state, cognition and continuous execution.

## Canonical invariant

**Model cognition proposes. Policy and authority govern. Safety validates. Structural validation validates state transitions. Execution happens only after all configured gates pass. Canonical state commits once.**

```text
observation
   ↓
belief update
   ↓
prediction / self-model
   ↓
prediction error + information need
   ↓
deliberation / action proposal
   ↓
policy
   ↓
capability verification
   ↓
immutable safety kernel
   ↓
structural transition validation
   ↓
capability consumption
   ↓
external execution
   ↓
canonical state commit
   ↓
provenance + append-only trajectory
```

Hard authority can only change through an explicitly authorized `StateTransition`.
Soft cognitive channels may evolve, but they never grant hard authority.

## Converged components

- `nsa.core.state.CanonicalState`: stable typed state boundary.
- `nsa.core.transition`: atomic proposals, validation, receipts and state digests.
- `nsa.core.capabilities`: authenticated capability constraints with separate verify/consume operations.
- `nsa.core.safety_kernel.ImmutableSafetyKernel`: deterministic reference monitor; capability consumption is explicit.
- `nsa.cce.events`: typed observation, belief, prediction, prediction-error, cognition, governance and commit events.
- `nsa.cce.trajectory`: append-only in-memory transition history.
- `nsa.cce.persistence.TrajectoryJournal`: durable JSONL trajectory journal with integrity verification.
- `nsa.cce.transaction.CognitiveTransactionEngine`: canonical transaction coordinator.
- `nsa.cce.loop.CognitiveLoop`: observe → believe → predict → error → deliberate → govern → commit orchestration.
- `nsa.cce.canonical_runtime.CanonicalCCERuntime`: wall-clock scheduler over the canonical transaction engine, with optional durable journaling.
- `nsa.cce.omega_adapter.CanonicalOmegaAdapter`: explicit one-way CanonicalState → Omega bridge.
- `nsa.cce.substrate_adapter.SixLayerCanonicalAdapter`: turns the neural six-layer substrate into action proposals rather than authority.
- `nsa.cce.safety_adapter.ImmutableKernelGate`: exposes the immutable kernel as a canonical CCE safety gate.
- `nsa.cognition.deliberation.InformationSeekingPlanner`: makes missing evidence an explicit T1-style action proposal that still passes normal governance.

## Capability lifecycle

Capability authorization has three distinct states:

1. **Verify** — authenticate scope, action, tier, expiry, replay status and constraints.
2. **Govern** — policy, safety and structural gates may still reject the proposal without burning the nonce.
3. **Consume** — consume the nonce immediately before an external effect.

This prevents rejected proposals from consuming capabilities while preserving one-shot replay protection for actually authorized use.

## Neural / Omega boundary

The six-layer neural substrate remains useful for simulation, epistemic reasoning and
self-modeling, but it is not the canonical state owner. `CanonicalOmegaAdapter` provides
a one-way representation bridge and `SixLayerCanonicalAdapter` exposes substrate results
as proposals. No tensor output can directly mutate canonical hard authority.

The public compatibility `CognitiveDynamicsSubstrate.step()` still returns a projected
Omega for older callers; new integrations should use the canonical adapters and transaction engine.

## Information seeking

High uncertainty is represented explicitly through `InformationNeed` and can produce a
governed information-gathering `ActionCandidate`. Information seeking is not an authority
bypass: the resulting action is still subject to policy, capability, safety and structural validation.

## Durability and replay

`TrajectoryJournal` stores append-only JSONL records. `verify()` checks monotonic steps,
well-formed SHA-256 state digests, and optionally the latest digest against a live canonical
state. This is the base for durable replay. Full state reconstruction remains a separate
phase because arbitrary semantic payloads cannot safely be reconstructed from summaries alone.

## Remaining hardening phases

1. **Transactional effect/commit coupling** — introduce an effect receipt or two-phase executor so a non-idempotent external effect cannot succeed while its canonical commit fails.
2. **General capability constraint evaluator** — move beyond `max_risk` and `reversible_only` to typed scopes, targets, rate limits and resource bounds.
3. **Production key management** — require explicit injected key material in production deployments; keep the demo key only for legacy research fixtures.
4. **Full Omega state adapters** — connect learned self-model state, epistemic vectors and prediction-error metrics to canonical soft/provenance channels through explicit typed adapters.
5. **Durable replay** — persist enough canonical state material to reconstruct and verify a trajectory, with schema/version hashes and corruption detection.
6. **Adversarial end-to-end matrix** — continuously prove that model output cannot bypass policy, capability, safety, hard-state authorization or trajectory integrity.
7. **PyTorch isolation** — keep the base canonical CCE usable without importing the neural substrate; neural dependencies should remain optional integration modules.
