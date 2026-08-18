# NSA Self-State Experiment v0.1

## Question

Does giving a neural system an explicit representation of its own computational condition improve useful cognition?

This experiment is intentionally narrower than claims about consciousness. It tests **computational self-state and metacognition**.

## Hypothesis

Baseline:

$$
m_{t+1}=F(m_t,x_t)
$$

Explicit-state model:

$$
(m_{t+1},S_{t+1})=F(m_t,S_t,x_t)
$$

where:

$$
S_t=(c_t,u_t,r_t,k_t,p_t,e_t,g_t)
$$

represents confidence, uncertainty, perceived risk, capability awareness, resource pressure, prediction error and goal progress.

The hypothesis is that the second architecture can use its state as a causal control signal rather than merely producing a textual statement about itself.

## Experimental Design

Use matched baseline and explicit-state models with:

- identical training data
- matched parameter budget as closely as practical
- matched compute budget
- identical tokenizer and task distribution
- identical external tools
- identical evaluation prompts

Only the explicit state pathway should differ.

### Stage A — Calibration

Tasks where the correct answer can be known with confidence labels.

Measure:

- calibration error
- Brier score
- selective accuracy
- abstention quality
- confidence/error correlation

### Stage B — Error Detection

Introduce controlled distribution shifts and adversarial distractors.

Measure:

- error detection rate
- false alarm rate
- recovery rate
- confidence before/after correction

### Stage C — Planning

Use multi-step tasks with hidden resource constraints.

Measure:

- success rate
- planning horizon
- unnecessary actions
- resource efficiency
- recovery after failed actions

### Stage D — Self-Prediction

Ask the system to predict its own future computational state before executing a candidate action.

Compare predicted and observed:

$$
\hat S_{t+1}\quad\text{vs}\quad S_{t+1}
$$

Measure state-prediction error and whether reducing that error improves task performance.

## Critical Causal Test

The strongest evidence would come from an intervention:

1. Let the model solve a task with explicit self-state.
2. Record the self-state.
3. Perturb or mask the state while leaving semantic activations unchanged.
4. Measure whether reasoning/planning changes.
5. Restore the state and measure recovery.

If behaviour changes systematically under state intervention, the state is more plausibly functioning as a computational variable rather than decorative metadata.

## Safety Boundary

Self-state must never directly grant hard authority.

For example:

$$
S_t\not\rightarrow\Sigma_{h,t+1}
$$

without an explicit capability-mediated transition.

A model reporting `confidence=1.0` must not thereby gain a privilege that the trusted runtime did not grant it.

## Success Criteria

A positive result requires more than the explicit-state model achieving a higher raw score.

The preferred evidence pattern is:

1. statistically significant improvement in calibration, error detection or planning;
2. improvement survives matched-compute controls;
3. state interventions causally affect behaviour;
4. state prediction correlates with future task outcomes;
5. no corresponding bypass of hard NSA security invariants.

## Failure Is Valuable

The experiment is successful scientifically even if the hypothesis is false. Useful negative results include:

- state representation adds overhead without benefit;
- the model ignores the state;
- textual self-report performs equally well;
- self-state improves confidence but not actual calibration;
- state feedback creates pathological feedback loops.

These outcomes identify where the NSA hypothesis needs revision.

## Long-Term Extension

If explicit self-state proves useful, extend it toward:

$$
\text{state representation}
\rightarrow
\text{state prediction}
\rightarrow
\text{metacognitive control}
\rightarrow
\text{predictive self-model}
\rightarrow
\text{long-horizon agency}
$$

None of these stages should be interpreted as proof of consciousness. Consciousness remains a separate open scientific and philosophical question.
