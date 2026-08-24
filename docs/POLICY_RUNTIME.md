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
