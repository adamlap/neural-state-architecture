# NSA Roadmap Status — 2026-08-19

This is a living implementation snapshot for the master roadmap in `PLAN.md`.

## Current Position

NSA has moved from the original neural security prototype into the first implementation layer of the **full typed cognitive substrate**.

### Phases 1–10 — Original NSA Foundation
**Status: MATURE FOUNDATION / VALIDATION CONTINUES**

The repository contains the original lattice, conservation-law, non-interference, multi-dimensional state, native/retrofit, GPU, auditing/recovery, value-layer and formal-core work. Independent real-checkpoint/model/hardware validation remains an important open task.

---

# New Framework Phases

## Phase 11 — Canonical Typed Neural Core
**Status: IMPLEMENTATION STARTED**

Implemented:

- `nsa/core/`
- canonical typed state primitives
- semantic state
- hard trusted state
- soft operational state
- provenance state
- goal state
- explicit hard-state transitions

Remaining:

- compatibility adapter for every legacy state type
- serialization/versioning
- richer typed capability/value fields
- integration into native neural operators

## Phase 12 — General State Algebra
**Status: IMPLEMENTATION STARTED**

Implemented in `nsa/algebra_engine.py`:

- generic algebra contract
- ordered domains
- capability/power-set algebra
- confidence/probability algebra
- risk algebra
- heterogeneous product algebra
- component-wise ordering
- product transition decisions

The original `nsa/algebra.py` remains intact for backward compatibility during migration.

Remaining:

- temporal algebra
- constraint algebra
- richer transition cones
- exact heterogeneous projections
- broader algebraic property suite
- eventual unification with the legacy algebra module

## Phase 13 — Algebra-Preserving Transition Engine
**Status: FOUNDATION IMPLEMENTED**

Implemented in `nsa/transitions/`:

- explicit transition proposals
- policy validation
- authorization boundary
- structured transition results
- authorization-addition protection
- license-transition policy

Remaining:

- integration with every heterogeneous algebra domain
- native TNC integration
- formal proofs
- direct capability-object integration
- performance/capability-cost benchmarks

## Phase 14 — Whole-System Information Flow
**Status: FOUNDATION IMPLEMENTED**

Implemented in `nsa/flow/`:

- declarative flow nodes
- typed flow edges
- dimension-level flow permissions
- destination authorization requirements
- structured flow violations
- state propagation engine
- propagation tests

The graph is intentionally declarative at this stage. Tensor/activation-level enforcement, declassification and whole-model non-interference remain future work.

## Phase 15 — Capability / Authority
**Status: INITIAL MODULAR IMPLEMENTATION**

Implemented in `nsa/capabilities/`:

- scoped capability objects
- issuer/subject/action/scope/purpose
- expiry
- optional nonce
- trusted capability authority
- least-authority checks
- issuer isolation tests

Remaining:

- cryptographic signatures
- revocation
- attenuation/delegation
- replay protection
- capability lifecycle integration with runtime/tool execution

## Phase 16 — Provenance / Epistemic State
**Status: INITIAL MODULAR IMPLEMENTATION**

Implemented in `nsa/provenance/`:

- immutable evidence records
- source identity
- evidence kind
- reliability bounds
- claim records
- parent-claim lineage
- append-only provenance store

Remaining:

- evidence graphs
- contradiction handling
- confidence propagation
- trust domains
- provenance-aware transitions
- audit export

## Phase 17 — Typed Persistent Memory
**Status: INITIAL MODULAR IMPLEMENTATION**

Implemented in `nsa/memory/`:

- typed memory items
- provenance references
- sensitivity metadata
- timestamps
- expiry
- append-only memory store

Remaining:

- vector retrieval integration
- tenant isolation
- read/write flow policy
- declassification
- confidence decay
- memory audit trail

## Phase 18 — Self-State / Metacognition
**Status: RUNNABLE PROTOTYPE / FIRST EXPERIMENT IMPLEMENTED**

Implemented in `nsa/self_state/`:

- explicit self-state representation
- confidence
- uncertainty
- perceived risk
- capability awareness
- resource pressure
- goal progress
- state prediction error
- metacognitive pressure
- immutable observations
- self-state prediction primitive

First experiment implemented in `experiments/self_state/`:

- matched baseline recurrent model
- explicit self-state recurrent model
- approximately matched parameter budgets
- sequential noisy-evidence task
- calibration metrics
- shifted-distribution evaluation
- explicit state-path causal ablation
- smoke tests
- reproducible seed/CLI

Run:

```bash
PYTHONPATH=. python experiments/self_state/run.py --steps 800 --seed 7
```

A positive result is **not** assumed. The experiment is designed to falsify the hypothesis as well as support it.

## Phase 19 — Predictive Self-Model
**Status: FIRST PRIMITIVE IMPLEMENTED / RESEARCH PROTOTYPE**

`nsa/self_state/prediction.py` provides a first prediction/error abstraction. The full predictive self-model remains to be built.

Remaining:

- action-conditioned future-state prediction
- internal simulation
- counterfactuals
- resource/capability prediction
- calibrated state prediction error
- learned predictive model

## Phase 20 — Tool & Action Governance
**Status: ARCHITECTURAL FOUNDATION**

Flow and capability primitives now provide the substrate for typed action requests. A dedicated `nsa/actions/` module and trusted tool gateway remain to be built.

## Phase 21 — Trusted Cognitive Runtime
**Status: DESIGN / FOUNDATION**

The canonical state, flow, capability, provenance and memory primitives now define the ingredients for a trusted runtime. Execution state, scheduling, checkpointing, rollback and resource governance remain future implementation work.

## Phase 22 — Multi-Agent State
**Status: DESIGN ONLY**

## Phase 23 — Alignment / Value Substrate
**Status: FOUNDATION COMPLETE / EXTENSION PENDING**

Existing value-layer work remains. Heterogeneous values, normative uncertainty and auditable value revision remain future work.

## Phase 24 — Normative Uncertainty
**Status: DESIGN ONLY**

## Phase 25 — Dynamic Auditing / Recovery
**Status: FOUNDATION COMPLETE / EXTENSION PENDING**

Existing probing, rollback, recovery and verifier infrastructure remains the foundation.

## Phase 26 — TCB / Formal Verification
**Status: PARTIAL FOUNDATION**

Formal theorem work exists. Machine-checkable verification and end-to-end proofs remain open.

## Phase 27 — Security / Red Teaming
**Status: FOUNDATION EXISTS / EXPANSION PENDING**

Existing adversarial benchmark infrastructure provides the starting point for the new whole-system state substrate.

## Phase 28 — Joint Safety + Intelligence Evaluation
**Status: FIRST RESEARCH PROGRAM STARTED**

The explicit self-state experiment is the first concrete capability-vs-safety research program. A broader benchmark suite remains future work.

## Phases 29–30 — Kernels / Ecosystem
**Status: FOUNDATION COMPLETE / EXPANSION PENDING**

Existing GPU and ecosystem integrations provide the base.

## Phases 31–34 — Self-Modification / Distributed Intelligence / Advanced Self-Model / General Substrate
**Status: LONG-TERM RESEARCH**

These phases should not be treated as implementation-ready until the trusted state, capability, runtime and formal verification layers are substantially mature.

---

# Current Architecture Maturity

```text
Original NSA security              ████████████████████  mature foundation
Multi-dimensional state             ██████████████████░░  strong foundation
Canonical typed state               ███████████████░░░░░  implementation
General state algebra                ████████████░░░░░░░░  implementation
Trusted transitions                  ███████████████░░░░░  foundation
Whole-system flow                   ████████████░░░░░░░░  foundation
Capabilities                        ██████████░░░░░░░░░░  initial module
Provenance                          ████████░░░░░░░░░░░░  initial module
Typed memory                        ███████░░░░░░░░░░░░░  initial module
Self-state                          ████████████░░░░░░░░  runnable prototype
Self-model                          █████░░░░░░░░░░░░░░░  first primitive
Actions                             ████░░░░░░░░░░░░░░░░  architecture
Trusted runtime                     ███░░░░░░░░░░░░░░░░░  design
Multi-agent                         ██░░░░░░░░░░░░░░░░░░  design
Formal verification                 ████░░░░░░░░░░░░░░░░  foundation
```

## Immediate Build Sequence

1. Make the generalized algebra fully algebra-aware across all canonical dimensions.
2. Connect capabilities directly to flow and transition enforcement.
3. Connect provenance to claims, memory and confidence updates.
4. Connect typed memory to the flow graph.
5. Connect self-state directly to an NSA neural layer/TNC rather than only the experimental recurrent model.
6. Run the self-state experiment across multiple seeds and tasks.
7. Build typed tool/action requests.
8. Build the trusted cognitive runtime.
9. Add end-to-end formal information-flow properties.
10. Red-team the complete substrate.

The key milestone is not the number of modules. It is reaching the point where **semantic cognition, self-state, security state, provenance, memory and authority all evolve through one coherent transition algebra**.
