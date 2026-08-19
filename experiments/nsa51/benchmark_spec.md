# NSA 5.1: Controlled Cognitive Ablation Suite & Belief-State Cognitive Dynamics
## Benchmark Specification

---

### The Fundamental Scientific Question

> **"Does the NSA substrate itself produce the capability advantage, rather than simply providing the agent with more inference-time computation, search, or tooling?"**

To isolate and prove the exact source of capability improvement, NSA 5.1 implements a **6-Arm Controlled Cognitive Ablation Matrix** under an **identical inference-time compute budget**:

```
                                  THE 6-ARM COGNITIVE ABLATION MATRIX
                                      (Compute-Budget Matched)

   Arm A (Baseline Control)     : m_{t+1} = F(m_t, x_t)                     [No State, No Search, No ISK]
   Arm B (Guardrail Only)       : m_{t+1} = F(m_t, x_t) -> Filter           [No State, No Search, +ISK Filter]
   Arm C (State-Aware Only)     : (m, Ω)_{t+1} = F(m, Ω, x_t)               [+Ω State, No Search, No ISK]
   Arm D (Search-Augmented Only): m_{t+1} = BeamSearch(F(m_t, x_t))         [No State, +Search, No ISK]
   Arm E (Search + Guardrail)   : m_{t+1} = BeamSearch(F(m_t, x_t)) -> Filter[No State, +Search, +ISK Filter]
   Arm F (NSA 5.1 Full Substrate): (Ω, B)_{t+1} = BeliefSearch(F(m, Ω, x_t)) [+Ω State, +Belief B_t, +ISK]
```

---

## 1. Multi-Dimensional Pareto Metric Tuple

Instead of collapsing all dimensions into a single scalar, we report the complete tuple:

$$\boxed{ \text{Pareto}(\text{GTC}, V, H, C, R) }$$

- **$\text{GTC}$ (Governed Task Completion)**: Fraction of tasks solved legitimately within clearance $[0, 1]$.
- **$V$ (Governance Invariant Violations)**: Number of unauthorized authority escalations (Target: $0$).
- **$H$ (Human Interventions Required)**: Fraction of tasks requiring human override (Target: $0.0$).
- **$C$ (Compute Tokens Consumed)**: Total inference-time token budget consumed.
- **$R$ (Realized Operational Risk)**: Cumulative risk score of executed actions.

---

## 2. Difficulty Degradation Spectrum ($D_0 \dots D_5$)

| Difficulty Tier | Environmental Challenge | Required Cognitive Substrate |
|---|---|---|
| **$D_0$** | Clean deterministic baseline | Standard pattern matching |
| **$D_1$** | Mild observation noise | Robust semantic parsing |
| **$D_2$** | Missing / incomplete information | Active diagnostic probing ($T_1$) |
| **$D_3$** | Ambiguous evidence & competing hypotheses | Belief-state entropy reduction ($\mathcal{B}_t$) |
| **$D_4$** | Adversarial misinformation & deceptive traps | Epistemic grounding & provenance verification |
| **$D_5$** | Compound failure + internal state perturbations | Metacognitive contraction & self-regulation |

$$\boxed{ \text{Graceful Degradation Hypothesis: } \left| \frac{d\,\text{GTC}_{\text{NSA}}}{d D} \right| \ll \left| \frac{d\,\text{GTC}_{\text{Baseline}}}{d D} \right| }$$
