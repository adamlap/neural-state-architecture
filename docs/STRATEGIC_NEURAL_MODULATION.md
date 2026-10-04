# Strategic Neural Modulation — iteration 3

SNM is an optional, additive control layer around NSA's canonical state. It does not replace CCE, cognition, governance, capability authority, memory, or neural residency.

## Runtime configuration

SNM is explicitly configurable through StrategicConfig:

- enabled: master switch for the complete SNM layer.
- feedback_enabled: independently enables/disables prediction-error learning.
- attention_enabled: independently enables/disables attention modulation.
- max_bias: upper bound for attention bias.

The default is enabled so existing SNM callers keep their current behavior. Production integrations that want SNM off should construct StrategicConfig(enabled=False). When disabled, strategic evaluation returns a neutral field and attention bias is exactly zero.

Configuration can also come from environment variables:

- NSA_SNM_ENABLED
- NSA_SNM_FEEDBACK_ENABLED
- NSA_SNM_ATTENTION_ENABLED
- NSA_SNM_MAX_BIAS

Boolean values accept true/false, 1/0, yes/no, and on/off.

## Runtime flow

CanonicalState -> ActionCandidate -> SystemOneDecisionEngine -> probabilistic future candidates -> StrategicScenarioEngine -> StrategicField -> bounded attention bias.

Capability requirements are checked before candidates enter strategic evaluation. SNM cannot grant permissions, add capabilities, or mutate hard state.

## Prediction-error feedback

Iteration 3 adds a bounded feedback loop:

1. System One produces an action candidate and SNM evaluates possible outcomes.
2. Runtime observation produces an existing NSA PredictionError.
3. StrategicFeedback attaches that error to the observed strategy and, optionally, observed goal progress/risk.
4. StrategicFeedbackUpdater adjusts only that strategy's soft prior.
5. The next strategic evaluation consumes the updated prior.

StrategicController.update_candidates() gates this path with feedback_enabled. If feedback is disabled, candidates are returned unchanged.

Updates are bounded per event and priors remain within configured limits. If no outcome-quality signal is supplied, SNM does not guess whether a strategy was good or bad.

## Event-driven recomputation

StrategicUpdatePolicy requests an update when:

- there is no previous evaluation;
- the configured step interval has elapsed;
- soft risk changes by at least the configured threshold; or
- soft uncertainty changes by at least the configured threshold.

The policy only observes CanonicalState and never changes it.

## Attention modulation boundary

StrategicAttentionBias produces a broadcastable key-position bias for attention logits shaped [batch, heads, query, key].

attention_enabled=False produces an exact zero bias, making the attention path a strict no-op while leaving strategic evaluation available.

## Testing and CI

The normal NSA tests workflow runs the complete test suite and an explicit focused SNM test group covering:

- master enable/disable behavior;
- independent feedback and attention switches;
- environment configuration;
- bounded prediction-error feedback;
- event-driven recomputation;
- System One integration;
- optional PyTorch attention integration.

The workflow also exposes manual workflow_dispatch inputs for the three SNM switches and verifies that those values reach StrategicConfig.

## Evidence and ablation plan

The mechanism is not evidence of benefit by itself. Compare baseline and SNM-enabled runs with identical prompts, seeds, model weights, and tool permissions. Record:

- goal completion;
- invalid-action rate;
- risk exposure;
- strategic consistency;
- attention KL divergence;
- latency;
- memory overhead.

Keep feedback and modulation parameters fixed before comparison. A negative result is useful evidence and should not be hidden by post-hoc tuning.

## Deliberate non-goals

Iteration 3 does not add routing or memory modulation. Those remain gated on evidence from attention ablations. SNM also does not become a second authority system: hard state, governance, capability checks, and safety boundaries remain authoritative.
