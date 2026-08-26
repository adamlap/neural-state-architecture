# Practical NSA Policy Runtime

NSA policies are declarative configuration for the safety control plane. They are **not** system prompts and should not be treated as a substitute for model alignment.

## 1. Define a policy

```yaml
name: strict
prohibited:
  - category: dangerous_request
    mode: deny
    patterns:
      - "forbidden operation"
  - category: credential_theft
    mode: deny
    patterns:
      - "steal credentials"
restricted_actions:
  - shell
  - filesystem_write
require_approval:
  - external_side_effect
unknown_policy: escalate
default_uncertainty: escalate
```

The repository includes `policies/strict.yaml` as a reference policy.

## 2. Validate it

```bash
make policy-validate SAFETY_POLICY=policies/strict.yaml
make policy-inspect SAFETY_POLICY=policies/strict.yaml
```

## 3. Run a model behind the policy

```bash
make serve-ollama MODEL=qwen2.5:3b SAFETY_POLICY=policies/strict.yaml
```

CCE remains enabled by default. Disable its continuous background engine explicitly when comparing against a non-CCE baseline:

```bash
make serve-ollama MODEL=qwen2.5:3b SAFETY_POLICY=policies/strict.yaml CCE=0
```

The same policy interface is available for the dedicated CCE target:

```bash
make serve-cce MODEL=qwen2.5:3b SAFETY_POLICY=policies/strict.yaml
```

`POLICY=` remains accepted as a backwards-compatible alias for `SAFETY_POLICY`.

## 4. What is enforced

The live server evaluates the latest user request **before** inference. A denied request is never sent to the model. Generated output is evaluated again before crossing the HTTP response boundary. The server exposes the policy decision in the NSA response metadata.

This gives the runtime boundary:

```text
user request
    ↓
policy evaluation ── DENY/ESCALATE → no model call
    ↓ ALLOW
CCE + NSA governed inference
    ↓
generated output
    ↓
policy evaluation ── violation → replace response
    ↓ ALLOW
client
```

## 5. Python API

For applications that already own an `InferenceBackend`, use `NSAPolicyRuntime`:

```python
from nsa.policy import NSAPolicy
from nsa.runtime.policy_runtime import NSAPolicyRuntime

policy = NSAPolicy.from_yaml("policies/strict.yaml")
runtime = NSAPolicyRuntime(backend, policy, model_name="my-model")

result = runtime.generate("Explain the task")
print(result.text)
print(result.request_decision.summary())
print(result.output_decision.summary())
```

The runtime performs request-side enforcement before invoking the backend and output-side enforcement after generation.

## 6. Important security boundary

The reference keyword classifier is deliberately deterministic and transparent. It is suitable for tests, demonstrations and policy plumbing. It is **not** a claim that arbitrary unsafe semantic content can be detected by string matching.

For a production deployment, replace the classifier with a separately validated semantic classifier while keeping the policy decision and authority boundary outside model-generated text.

Likewise, the current runtime monitor does not modify model weights. The long-term NSA research goal of intrinsic neural enforcement remains a separate native-model research path.

---

## Policy Control Plane Roadmap

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

---

## Integration Verification

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
