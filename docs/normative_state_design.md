# Normative State Design

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
