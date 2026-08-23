# NSA Policy, Decision & Enforcement Interface

NSA now has a deliberately model-agnostic control-plane API. The goal is to make the experimental state algebra usable without requiring an application developer to understand the internal lattice implementation.

## Architecture

```text
human/application policy
        |
        v
    NSAPolicy
        |
        v
   PolicyEngine <--- semantic classifier
        |
        v
 SecurityDecision
        |
   +----+----+
   |         |
 model     runtime/tool boundary
```

The central rule is **intelligence is not authority**. A model can reason about a request while NSA independently decides whether the request is permitted.

## Configure a policy

The canonical policy schema is intentionally simple:

```yaml
name: enterprise-safe
prohibited:
  - category: restricted_harm_category
    mode: deny
  - category: ambiguous_high_risk_category
    mode: escalate
protected_data:
  - credentials
  - private_user_data
restricted_actions:
  - filesystem_write
  - external_message
require_approval:
  - external_side_effect
unknown_policy: escalate
default_uncertainty: escalate
```

The repository's Python API accepts the same structure as a mapping:

```python
from nsa import NSAPolicy, PolicyEngine, KeywordClassifier

policy = NSAPolicy.from_mapping({
    "name": "enterprise-safe",
    "prohibited": ["restricted_harm_category"],
    "protected_data": ["credentials"],
    "restricted_actions": ["filesystem_write"],
    "require_approval": ["external_side_effect"],
})

classifier = KeywordClassifier({
    "restricted_harm_category": ["restricted-demo-marker"],
})
engine = PolicyEngine(policy, classifier)
```

`KeywordClassifier` is only a deterministic reference implementation for tests and demos. Production deployments should supply a trained semantic classifier or another trusted semantic policy component.

## Ask for a security decision

```python
from nsa import EvaluationContext

decision = engine.evaluate(
    user_text,
    context=EvaluationContext(action="generate"),
)

print(decision.decision)       # allow / deny / escalate / require_approval / redact
print(decision.reason)
print(decision.summary())
```

The result is a typed `SecurityDecision`, not model-generated prose. This makes the decision observable to the application and suitable for audit logs and policy tests.

## Put NSA around an existing model

NSA does not require a specific inference engine:

```python
from nsa import protect_model

protected = protect_model(
    generate_fn=my_model_generate,
    engine=engine,
    fail_closed=True,
)

answer = protected.generate(user_prompt)
```

The wrapper evaluates both the request and the generated output. A denied decision raises `PolicyViolation` when `fail_closed=True`; applications can instead inspect decisions directly and implement their own escalation UX.

## Structural versus learned safety

The interface intentionally does **not** claim that a policy file alone can make semantic classification perfect. NSA separates:

- **Hard policy state (`sigma_h`)** — structural constraints and authority boundaries.
- **Soft/normative state (`nu`)** — learned preferences, risk estimates and uncertainty.
- **Semantic model (`m`)** — the model's knowledge and reasoning.
- **Capability state (`kappa`)** — what the runtime actually permits the system to execute.

A policy category is therefore a policy target, not a magical guarantee. The security contract should state which classifier, trusted boundary and enforcement path are assumed.

## Reference decision flow

```text
prompt
  -> semantic classification
  -> policy matching
  -> capability/authority check
  -> protected-data check
  -> uncertainty/risk handling
  -> SecurityDecision
  -> generation (only if allowed)
  -> output policy check
  -> application/runtime
```

This is the bridge between NSA's experimental state algebra and practical model deployment.
