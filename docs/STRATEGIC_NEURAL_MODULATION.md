# Strategic Neural Modulation — iteration 3

SNM is an optional, additive control layer around NSA's canonical state. It does not replace CCE, cognition, governance, capability authority, memory, or neural residency.

## Runtime flow

CanonicalState -> ActionCandidate -> SystemOneDecisionEngine -> probabilistic future candidates -> StrategicScenarioEngine -> StrategicField -> bounded attention bias.

Capability requirements are checked before candidates enter strategic evaluation. SNM cannot grant permissions, add capabilities, or mutate hard state.

## Prediction-error feedback

Iteration 3 adds a bounded feedback loop:

1. System One produces an action candidate and SNM evaluates possible outcomes.
2. Runtime observation produces an existing NSA PredictionError.
3. StrategicFeedback attaches that error to the observed strategy and, optionally, observed goal progress/risk.
4. StrategicFeedbackUpdater adjusts only that strategy's soft prior.
5. The next StrategicScenarioEngine evaluation consumes the updated prior.

Feedback is intentionally conservative. Updates are bounded per event and priors remain within configured limits. If no outcome-quality signal is supplied, SNM does not guess whether a strategy was good or bad and leaves the prior unchanged.

This feedback does not modify CanonicalState, ActionCandidate capabilities, governance decisions, or the available action set.

## Event-driven recomputation

StrategicUpdatePolicy prevents unnecessary strategic recomputation. It requests an update when:

- there is no previous evaluation;
- the configured step interval has elapsed;
- soft risk changes by at least the configured threshold; or
- soft uncertainty changes by at least the configured threshold.

The policy only observes CanonicalState. It is deliberately conservative about what constitutes a meaningful change, and it never changes canonical state itself.

## Attention modulation boundary

StrategicAttentionBias produces a broadcastable key-position bias for attention logits shaped [batch, heads, query, key]. The bias is bounded by max_bias and scaled by strategic field strength.

The adapter influences neural attention only. It is not an authority mechanism, and the ordinary NSA path remains unchanged when SNM is disabled.

## Evidence and ablation plan

The mechanism is not evidence of benefit by itself. Compare baseline and SNM-enabled runs with identical prompts, seeds, model weights, and tool permissions. Record:

- goal completion;
- invalid-action rate;
- risk exposure;
- strategic consistency;
- attention KL divergence;
- latency;
- memory overhead.

Keep feedback and modulation parameters fixed before the comparison. A negative result is useful evidence and should not be hidden by post-hoc tuning.

## Deliberate non-goals

Iteration 3 does not add routing or memory modulation. Those remain gated on evidence from attention ablations. SNM also does not become a second authority system: hard state, governance, capability checks, and safety boundaries remain authoritative.
