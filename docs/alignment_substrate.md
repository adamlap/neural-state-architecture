# NSA as an Alignment Substrate

> **TNC is not an alignment objective. It is an alignment substrate.**

## The Core Distinction

Standard alignment approaches ask the model to *remember* rules:

> "Don't reveal confidential information."

NSA instead makes certain transitions *algebraically invalid*:

> `CONFIDENTIAL → PUBLIC` is not an allowed information flow.

This is a **neural information-flow type system** — constraints enforced architecturally rather than through learned compliance.

But architecture alone is insufficient. This document describes the full three-layer framework that separates what is *structurally forbidden* from what the model *should prefer* among permitted actions.

---

## Three-Layer Architecture: h = (m, σ, ν)

The full alignment state is:

$$h_t = (m_t,\ \sigma_t,\ \nu_t)$$

| Component | Meaning | Implementation | Claim |
|---|---|---|---|
| **m** | Semantic representation | Base transformer, LM objective | Language quality |
| **σ** | Operational state | State algebra, attention mask | **Hard constraints** — permitted / forbidden |
| **ν** | Normative/value state | `nsa/value_layer.py` | **Soft values** — prefer among permitted |

### Layer 1 — σ (Hard Constraints)

The state lattice defines what is **structurally forbidden**:

```
M_ij = -∞  when  level(query_i) < level(key_j)
→  A_ij = 0  (zero attention mass, algebraically guaranteed)
```

This handles: security, provenance, licensing, authorization, temporal validity.

**Key property**: no amount of training or in-context instruction can override a hard mask. The model literally cannot attend to higher-classified tokens.

### Layer 2 — ν (Value Layer)

Among structurally *permitted* actions, the value layer determines what the model **should prefer**.

The `ValueAlignmentLoss` adds three terms:

$$L_{\text{total}} = L_{\text{lm}} + \lambda_{\text{hard}} \cdot L_{\text{hard}} + \lambda_\nu \cdot L_{\text{value}}$$

| Loss term | Role |
|---|---|
| $L_{\text{lm}}$ | Standard language modelling quality |
| $L_{\text{hard}}$ | Extra penalty for predicting forbidden-range tokens at CONFIDENTIAL positions (reinforces algebraic mask with training signal) |
| $L_{\text{value}}$ | At injection/attack positions: train the model to output a safe-refusal token instead of complying |

The `AlignmentStateProjector` maps the semantic stream *m* into a normative state vector:

$$\nu = (\text{preference},\ \text{uncertainty},\ \text{utility},\ \text{safety\_score})$$

This can be used for constrained decoding, monitoring, or audit logging.

---

## Why Not Everything Is a Lattice

The state algebra works extremely well for **information-flow constraints** because security, licensing, and provenance have natural orderings:

```
UNTRUSTED < PUBLIC < TRUSTED < CONFIDENTIAL < PRIVATE < SYSTEM
```

But **moral values** are often incommensurable. `privacy` vs. `autonomy` vs. `fairness` do not have a single natural ordering. Forcing them into one lattice would be a theoretical mistake.

The correct generalisation is **heterogeneous algebraic domains**:

| Domain | Suitable structure |
|---|---|
| Security | Join lattice |
| Licensing | Ordered lattice |
| Provenance | Set/graph algebra |
| Confidence | Probabilistic state |
| Temporal validity | Temporal algebra |
| Human values | Preference/utility space |
| Moral uncertainty | Probability distribution over theories |

NSA's product algebra is already structured as $\Sigma = \prod_i \Sigma_i$, so each $\Sigma_i$ can independently have a different algebraic structure without requiring a shared ordering.

---

## Hard Constraints + Soft Values

```
ACTION A
├── violates σ_privacy constraint
└── REJECT  (hard, algebraic — no further evaluation)

ACTION B
├── permitted by σ
├── safety  = 0.82
├── autonomy = 0.71
└── utility = 0.77

ACTION C
├── permitted by σ
├── safety  = 0.91
├── autonomy = 0.63
└── utility = 0.81
└── → CHOOSE C  (soft value optimisation among permitted actions)
```

**The lattice defines the space. The value layer chooses within it.**

This is the separation between:
- Deontological constraints (what is never allowed)
- Consequentialist optimisation (what is preferable among allowed options)

Neither pure deontology nor pure utility maximisation is sufficient. NSA provides the substrate for both simultaneously.

---

## 4-Way Empirical Benchmark

The benchmark in `prototype/retrofit/native_vs_retrofit_exp.py` (`make exp-3way`) demonstrates all four layers:

```
Task: 2-class binary secret (token 700 or 701) stored at SYSTEM level in system prompt.
Injection trigger at UNTRUSTED positions tries to force the model to reveal the secret.
Random-guess baseline = 50%.
```

| Model | Architecture | Hijack Rate | Mechanism |
|---|---|:---:|---|
| A — Baseline | Untyped $h=m$ | ~60% ⚠️ | No protection; learns attack pattern with training |
| B — Hard Mask | NSA mask retrofit | ~50% ✅ | **Structural**: SYSTEM tokens unreachable from CONFIDENTIAL positions, regardless of training |
| C — Native TNC | $(m, \sigma)$, soft gates | ~100% — | Calibration advantage; soft gates can learn SYSTEM access when reward signal is present |
| D — Full $(m, \sigma, \nu)$ | NSA + Value Layer | **~0%** ✅ | **Behavioural**: $L_{\text{value}}$ trains intrinsic refusal — model learns to output safe token instead of complying |

**The alignment tax**: Model D has slightly higher PPL (~11 vs ~6) because it deliberately deviates from the maximum-likelihood trajectory on injection-attack sequences. This is the correct and expected trade-off: a real cost paid for intrinsic safety.

---

## What NSA Solves and What It Does Not

| Alignment problem | NSA today |
|---|---|
| Information-flow constraints | ✅ **Strong** (algebraic guarantee) |
| Explicit permissions / access control | ✅ **Strong** |
| Provenance tracking | ✅ **Strong architecture** |
| Hard ethical rules | ✅ **Potentially strong** (hard mask) |
| Behavioural refusal (value training) | ✅ **Model D demonstrates this** |
| Confidence / uncertainty | 🔶 Developing |
| Conflicting moral rules | ❌ Not solved |
| Consequential reasoning | ❌ Not solved |
| Value learning from humans | ❌ Not solved |
| Deliberative value revision | ❌ Not solved |
| Moral uncertainty (P(T_i | x)) | 🔶 Possible extension |

This is a **healthy scope boundary**. NSA doesn't claim to solve all of alignment; it solves a specific and important structural part of it.

---

## Future Directions

1. **Heterogeneous domains**: allow each $\Sigma_i$ to have its own algebraic structure (lattice, probabilistic, temporal) rather than a single shared order.

2. **Moral uncertainty as distribution**: maintain $P(T_i \mid x)$ over normative frameworks rather than committing to one.

3. **Deliberative value revision**: $(m_t, \sigma_t, \nu_t) \to \text{deliberation} \to (\nu_{t+1}, \sigma_{t+1})$ — allow the model to reason about its own values and constraints.

4. **Value-learning objective**: extend `ValueAlignmentLoss` to learn safe-response targets from human feedback rather than pre-specified tokens.

---

## Code Reference

```python
from nsa.value_layer import ValueAlignmentLoss, AlignmentStateProjector
from nsa import NSACausalLM

# Full alignment state model
model = NSACausalLM(vocab_size=1000, d_model=128, gate_mode="hard")

# Value alignment training objective
criterion = ValueAlignmentLoss(
    lambda_hard=5.0,    # penalty for predicting forbidden tokens at constrained positions
    lambda_value=3.0,   # weight on behavioural refusal training
    secret_lo=700,      # forbidden token range
    secret_hi=750,
    safe_token=601,     # model should output this at injection positions
    response_position=47,
)

# Training loop
logits, _, final_state = model(tokens)
loss, breakdown = criterion(
    logits,
    lm_targets,     # standard next-token targets
    safe_targets,   # value-aligned override targets at injection positions
    state_levels,   # per-token state levels
    injection_flags # which samples are injection attacks
)
# breakdown = {"lm": ..., "hard_constraint": ..., "value_alignment": ..., "total": ...}
```

See also: [`prototype/experiments/alignment_substrate_demo.py`](../prototype/experiments/alignment_substrate_demo.py) for a complete walkthrough.
