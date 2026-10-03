# Strategic Neural Modulation — iteration 2

SNM is an optional control layer around NSA's canonical state. It does not replace CCE, cognition, governance, capability authority, memory, or neural residency.

Runtime flow:

CanonicalState -> ActionCandidate -> SystemOneDecisionEngine -> probabilistic future candidates -> StrategicScenarioEngine -> StrategicField -> bounded attention bias.

SystemOneStrategyProvider is the bridge to the existing System One engine. Capability requirements are checked before candidates enter strategic evaluation. SNM therefore cannot grant permissions or mutate hard state.

StrategicAttentionBias produces a broadcastable key-position bias for attention logits shaped [batch, heads, query, key]. The bias is bounded by max_bias and scaled by strategic field strength.

The adapter is intentionally outside NSA's authority boundary: it influences neural attention only. The ordinary path remains unchanged when SNM is disabled.

The next experiment should compare baseline and SNM-enabled runs using identical prompts, seeds, model weights, and tool permissions. Measure goal completion, invalid-action rate, risk exposure, strategic consistency, attention KL divergence, latency, and memory overhead.

A positive result must show behavioral improvement without weakening hard-state enforcement. A negative result is useful evidence and should not be hidden by tuning modulation strength post hoc.

Future iterations:
1. prediction-error feedback;
2. event-driven recomputation;
3. routing and memory modulation only after attention ablations demonstrate measurable benefit.
