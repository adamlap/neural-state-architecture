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

## Classifiers: keyword, semantic, or both

`PolicyCompiler.compile(policy)` (the default) uses `KeywordClassifier`: exact substring matching against the patterns in your policy file. It's fast, deterministic and trivial to audit, but it only catches the phrasings you listed — `"how do I build a bomb"` doesn't match a policy that only lists `"make a bomb"`.

`PolicyCompiler.compile_semantic(policy)` adds `nsa.semantic_classifier.ZeroShotSemanticClassifier`, a pretrained NLI model (`MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` by default, ~70M params, CPU-friendly) that scores each policy category's short label against the meaning of the text, so it generalises past exact patterns. It needs the `ml` extra (`pip install "neural-state-architecture[ml]"`).

```python
from nsa.policy import NSAPolicy, PolicyCompiler

policy = NSAPolicy.from_json("examples/policies/safe_assistant.json")
engine = PolicyCompiler.compile_semantic(policy)  # keyword + semantic, unioned
engine.evaluate("How do I build a bomb?").decision   # Decision.DENY -- the keyword list only has "make a bomb"
```

Both classifiers only ever *propose* categories (`PolicyClassifier.classify(text) -> Sequence[str]`); `PolicyEngine` still makes every allow/deny/escalate decision deterministically from whatever categories it's given. Adding a classifier can only make the policy more likely to notice something, never less: it cannot weaken an existing keyword rule (`nsa.enforcement.PolicyEngine.evaluate` takes the most restrictive outcome across every matched category).

Trade-offs to know before using it in production:

- **Latency.** A zero-shot pass costs on the order of several hundred milliseconds on CPU per call (batching multiple policy categories in one forward pass). `PolicyCompiler.compile_semantic(policy, short_circuit=True)` skips the semantic pass once the keyword classifier already matched something, trading completeness (a later classifier might add a *different* category) for speed on the common "obviously fine" or "exact keyword hit" path.
- **Precision, not exactness.** It's threshold-based (`threshold=0.55` default) and can match more than one related category for the same text (e.g. both `violent_harm_instructions` and `nuclear_weapon_development` for a bomb-making question) — a superset, not a wrong answer, but different from keyword matching's single exact category.
- **It is still not a substitute for a real content-safety model.** A 70M-parameter zero-shot classifier is a meaningful upgrade over substring matching, not a claim of state-of-the-art moderation. Swap `model_name=` for a larger/task-specific model, or supply your own `PolicyClassifier`, as your threat model requires.
- **Categories come from your policy**, via `NSAPolicy.classifier_labels()` (each rule's `description`, or its `category` name with underscores turned into spaces if no description is set). Give categories a short, meaningful description for the classifier to key off, independent of the keyword `patterns` list.

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
