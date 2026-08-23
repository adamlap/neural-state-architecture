# Practical Policy Control Plane Roadmap

The NSA research roadmap now includes a deployable policy/control-plane layer.

## Phase 11.5 — Policy & Configuration

- [x] Declarative `NSAPolicy` schema.
- [x] Prohibited semantic categories with deny/escalate modes.
- [x] Protected-data classes.
- [x] Restricted actions and approval gates.
- [x] Unknown-policy and uncertainty defaults.
- [x] JSON persistence and optional YAML loading.

## Phase 11.6 — Decision & Enforcement

- [x] Typed `SecurityDecision`.
- [x] ALLOW / DENY / ESCALATE / REQUIRE_APPROVAL / REDACT.
- [x] Audit metadata, risk and uncertainty.
- [x] Fail-closed model wrapper.
- [x] Request and output enforcement.

## Phase 11.7 — Model adapters

- [x] Model-agnostic `protect_model` wrapper.
- [x] Ollama/CCE server boundary.
- [ ] Native Hugging Face generation adapter.
- [ ] vLLM/SGLang adapters.
- [ ] Native NSA model integration.

## Phase 11.8 — Reference policies

- [x] Enterprise policy.
- [x] Safe assistant policy.
- [ ] Developer-agent policy.
- [ ] High-security policy.
- [ ] Autonomous-agent policy.

## Phase 11.9 — Policy verification

- [x] Unit tests for denial, approval, capability and output boundaries.
- [ ] Automated policy regression corpus.
- [ ] Adversarial semantic-classification benchmark.
- [ ] False-positive/false-negative measurement.
- [ ] Distribution-shift evaluation.
- [ ] Structural guarantee ledger linking each rule to its trusted enforcement boundary.

A declarative policy is not itself a semantic oracle. Learned normative/semantic components estimate intent and risk while NSA hard state and trusted capability boundaries enforce what is structurally permitted.
