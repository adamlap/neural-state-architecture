# Strategic Neural Modulation

Strategic Neural Modulation (SNM) is an **optional NSA control layer**. It does not replace the canonical state, CCE, System 1/System 2 cognition, governance, capability authority, memory, or neural residency layers.

## First usable iteration

The first iteration provides four pieces:

1. **Scenario outcomes** — represent several possible futures for each strategy.
2. **StrategicScenarioEngine** — cheaply evaluates multiple strategies using expected goal progress, risk, cost, time, information gain, and reversibility.
3. **StrategicField** — compresses the selected strategy distribution into a bounded control signal.
4. **StrategicAttentionModulator** — converts the field into a small additive attention bias that can be applied before softmax.

The modulation is deliberately a soft prior. It cannot grant authority or modify hard canonical state.

```python
from nsa.strategy import ScenarioOutcome, StrategyCandidate, StrategicController

strategies = [
    StrategyCandidate(
        "safe",
        (ScenarioOutcome("success", .8, .8, risk=.1, cost=.2),
         ScenarioOutcome("failure", .2, .2, risk=.2, cost=.2)),
        prior=.5,
    ),
    StrategyCandidate(
        "fast",
        (ScenarioOutcome("success", .7, .95, risk=.4, cost=.1),
         ScenarioOutcome("failure", .3, .1, risk=.8, cost=.1)),
        prior=.5,
    ),
]

evaluation = StrategicController().evaluate(agent.state, strategies)
bias = StrategicController().attention_bias(
    evaluation,
    token_affinity=[{"safe": 1.0}, {"fast": 1.0}, {"safe": .2, "fast": .4}],
)
```

## Design invariants

- SNM is opt-in and removable without changing NSA's core contract.
- The strategic field is soft state, not authority.
- Modulation is bounded by `max_bias`.
- The ordinary model path remains valid when SNM is disabled.
- Scenario probabilities are explicit and must sum to one.
- Strategy evaluation is deterministic for a fixed set of inputs.
- A future System 1 predictor can supply outcomes without changing the control API.

## Next iterations

1. Add an event-driven update policy so scenarios are recomputed only after meaningful state/prediction changes.
2. Add a System 1 scenario provider backed by the existing NSA cognitive machinery.
3. Add a real Hugging Face/PyTorch attention adapter with baseline-vs-SNM ablations.
4. Add prediction-error feedback to update the strategic field.
5. Add routing and memory modulation after attention modulation has measurable evidence.

The initial implementation intentionally stops before changing Transformer internals. This keeps the feature usable immediately while giving us a clean, measurable interface for neural integration.
