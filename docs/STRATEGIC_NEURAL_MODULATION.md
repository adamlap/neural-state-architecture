# Strategic Neural Modulation — iteration 2

SNM remains an optional control layer around NSA's canonical state. This iteration connects it to the existing System One decision engine.

Flow:

CanonicalState → ActionCandidate → SystemOneDecisionEngine → probabilistic future candidates → StrategicScenarioEngine → StrategicField → bounded attention bias.

The bridge is intentionally additive. Existing capability checks happen before an action enters the strategic layer; SNM cannot grant permissions or mutate hard state.

The resulting field can be consumed by a model adapter, while the ordinary NSA path remains unchanged when SNM is not enabled.

Next: integrate the field into an actual PyTorch/Transformers attention path and add baseline-vs-modulated behavioral measurements.