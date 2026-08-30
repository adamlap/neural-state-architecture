# Go 1 — Cognitive State Research

Go 1 adds a consciousness-inspired cognitive substrate behind the public `nsa.NSA` / CCE boundary. It is a computational architecture and **does not establish consciousness, sentience, subjective experience, or phenomenal awareness**.

## Architecture

```text
perception
    ↓
persistent prediction → prediction error → belief/uncertainty update
    ↓
competitive attention → salience → bounded global workspace
    ↓
global broadcast → recurrent cross-module coupling
    ↓
explicit integration graph ↔ self-model / metacognition
    ↓
action / observation selection → state update
```

The substrate is a deterministic state transition. CCE remains the canonical scheduler and `nsa.NSA` remains the sole application-facing runtime. No experiment owns a second agent loop.

## Mechanism mapping

| Mechanism | Implementation | Research framing |
|---|---|---|
| Competitive attention | salience/confidence/novelty competition | GWT-inspired |
| Workspace | bounded active set + ignition/broadcast history | GWT-inspired |
| Recurrent ignition | explicit recurrence trace and state-dependent transitions | recurrent-processing-inspired |
| Predictive processing | persistent predictions, errors, precision, uncertainty | predictive-processing-inspired |
| Integration graph | cross-module graph, coupling, influence and integration score | IIT-inspired computational analogue |
| Self model | predicted/internal state + error + confidence | higher-order/metacognitive-inspired |
| Information gain | uncertainty-reduction score | active-inference-inspired precursor |

These mappings are **inspired-by mappings**, not claims that the implementation instantiates or validates any scientific theory in full.

## Falsifiable hypotheses

H1. Adding the workspace mechanism improves performance on tasks requiring selective access to competing information at equal model-call/token budget.

H2. Persistent prediction/error state improves calibration and recovery after distribution shifts relative to a matched no-prediction ablation.

H3. Recurrent coupling improves long-horizon task performance or recovery relative to feed-forward state updates under matched compute.

H4. Explicit integration predicts behavioural gains that disappear when integration coupling is ablated, rather than being explained solely by extra context.

H5. The self-model improves self-error detection/calibration when its causal state is available, while self-report alone is not accepted as evidence of accuracy.

H6. Information-gain selection reduces uncertainty more efficiently than random observation selection under the same observation/action budget.

## Required evidence boundary

- `IMPLEMENTED`: code executes.
- `UNIT-TESTED`: deterministic mechanisms and invariants are tested.
- `EMPIRICALLY-VALIDATED`: a controlled experiment supports a hypothesis.
- `ROBUSTLY-VALIDATED`: replication across seeds/models/noise levels supports the effect.
- `OPEN-RESEARCH`: no causal evidence yet.

A passing software test is not evidence that the system is conscious.

## Go 2/3 ablation matrix

Every later experiment should preserve these controls:

| Mechanism | Full | Ablation |
|---|---:|---:|
| Workspace | on | off |
| Recurrence | on | off |
| Predictive processing | on | off |
| Self model | on | off |
| Integration | on | off |
| Active information acquisition | on | off |
| Valuation/homeostasis | on | off |
| Persistent identity/autobiographical state | on | off |

Use factorial or pre-registered subsets where the full combinatorial matrix is too expensive. Report raw trajectories, compute/token budgets, effect sizes and uncertainty.
