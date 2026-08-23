# Practical Policy Control Plane Roadmap

The existing NSA research roadmap establishes the algebraic and runtime foundations, but a deployable safety architecture also needs a developer-facing control plane. This addendum makes that requirement explicit.

## Phase 11.5 — Policy & Configuration Layer

- [x] Declarative `NSAPolicy` schema.
- [x] Prohibited semantic categories with deny/escalate modes.
- [x] Protected-data classes.
- [x] Restricted capabilities/actions.
- [x] Human approval gates.
- [x] Explicit unknown-policy and uncertainty defaults.
- [x] JSON persistence and optional YAML loading.

## Phase 11.6 — Decision & Enforcement API

- [x] Typed `SecurityDecision` result.
- [x] ALLOW / DENY / ESCALATE / REQUIRE_APPROVAL / REDACT decision vocabulary.
- [x] Matched policy categories and hard-constraint audit metadata.
- [x] Risk and uncertainty fields.
- [x] Fail-closed enforcement option.

## Phase 11.7 — Model Adapter Layer

- [x] Model-agnostic `protect_model(...)` wrapper.
- [x] Pre-generation request enforcement.
- [x] Post-generation output enforcement.
- [x] Explicit policy violation exception.
- [ ] Native Hugging Face generation adapter.
- [ ] vLLM/SGLang runtime adapters.
- [ ] Native NSA model integration.

## Phase 11.8 — Reference Policies

- [x] Enterprise reference policy.
- [ ] Developer-agent policy.
- [ ] High-security policy.
- [ ] Autonomous-agent policy.

## Phase 11.9 — Policy Verification

- [x] Unit tests for deny/approval/capability/output boundaries.
- [ ] Automated policy regression corpus.
- [ ] Adversarial semantic-classification benchmark.
- [ ] False-positive/false-negative measurement.
- [ ] Distribution-shift evaluation.
- [ ] Structural guarantee ledger linking each rule to its trusted enforcement boundary.

## Scientific boundary

A declarative policy is not itself a semantic oracle. Learned normative/semantic components can estimate intent and risk, while NSA hard state and trusted capability boundaries enforce what is structurally permitted. Security claims must state the classifier, TCB and enforcement assumptions under which they hold.
