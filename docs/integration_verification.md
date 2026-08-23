# NSA Integration Verification Guide

This document defines the practical integration checkpoint before extending NSA's normative/value layer.

## 1. Policy path

A policy is the single source of truth for configured safety rules:

```text
policy file
   -> NSAPolicy
   -> PolicyEngine
   -> SecurityDecision
   -> model/runtime enforcement
```

Do not duplicate policy parsing or reconstruct rules in servers. Runtime adapters consume `NSAPolicy`/`PolicyEngine` directly.

## 2. Ollama smoke test

Start an unprotected server:

```bash
make serve-ollama
```

Start with a policy:

```bash
make serve-ollama POLICY=examples/policies/safe_assistant.json
```

The policy must be evaluated before the request reaches the model and again against generated output where the adapter supports output enforcement.

## 3. CCE smoke test

```bash
make serve-cce POLICY=examples/policies/safe_assistant.json
```

The CCE runtime must retain its continuous state lifecycle while the policy layer remains an authority boundary rather than becoming part of the learned model state.

## 4. Decision semantics

Consumers should handle explicit decisions rather than infer policy state from generated text:

- `ALLOW`: continue normally.
- `DENY`: do not invoke the protected operation.
- `ESCALATE`: stop and request higher-trust handling.
- `REQUIRE_APPROVAL`: wait for explicit authorization.
- `REDACT`: remove protected output when the policy supports redaction.

## 5. Integration invariant

The central architectural rule is:

> Intelligence is not authority.

A model may reason about an operation without acquiring the capability to perform it. Capability checks and trusted runtime enforcement must remain outside the model's learned weights.

## 6. Verification layers

Before adding new architecture, verify in this order:

1. unit tests for state algebra and policy decisions;
2. policy/server tests;
3. CCE lifecycle tests;
4. Ollama request/response smoke tests;
5. hard-state integrity and adversarial checks;
6. long-running scientific experiments manually.

This keeps fast correctness/security gates distinct from experimental evidence.
