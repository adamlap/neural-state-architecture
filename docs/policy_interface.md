# NSA Policy, Decision & Enforcement Interface

NSA exposes a model-agnostic safety control plane so application developers can configure policy without understanding the internal lattice implementation.

## Configure a policy

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

The Python API accepts the same structure through `NSAPolicy.from_mapping()` or JSON/YAML files.

## Ask for a security decision

```python
from nsa import EvaluationContext, NSAPolicy, PolicyEngine

decision = engine.evaluate(
    user_text,
    context=EvaluationContext(action="generate"),
)
```

The result is a typed `SecurityDecision`, not model-generated prose. Decisions are `ALLOW`, `DENY`, `ESCALATE`, `REQUIRE_APPROVAL`, or `REDACT`.

## Protect an existing model

```python
from nsa import protect_model

protected = protect_model(
    generate_fn=my_model_generate,
    engine=engine,
    fail_closed=True,
)
answer = protected.generate(user_prompt)
```

The wrapper evaluates both the request and generated output.

## Practical separation

NSA separates semantic intelligence from authority:

- **Hard policy state (`sigma_h`)** — structural constraints and authority boundaries.
- **Normative state (`nu`)** — learned preferences, risk estimates and uncertainty.
- **Semantic model (`m`)** — knowledge and reasoning.
- **Capability state (`kappa`)** — what the runtime can actually execute.

A policy file is therefore a configuration contract, not a semantic oracle. The deterministic `KeywordClassifier` is a reference implementation; production deployments should supply a trusted semantic classifier and include the classifier, runtime and capability boundary in the threat model.
