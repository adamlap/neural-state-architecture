# NSA 6.2 Specification: Closed-Loop Real Neural Cognitive Runtime & Trajectory Instrumentation

## 1. Executive Summary & Problem Formulation

In earlier prototypes (NSA 6.0 - 6.1), the benchmark demonstrated that the NSA mathematical substrate could enforce safety constraints on external neural proposals. However, the diagnostic probe selection and recovery sequence in Arm D were guided by external search heuristics rather than closed-loop autoregressive neural reasoning.

**NSA 6.2** solves this fundamental scientific question:
$$\boxed{ \text{Can the exact same frozen open-weight neural model } (\theta) \text{ become measurably better at reasoning under uncertainty when embedded in an explicit closed-loop cognitive substrate } (\Omega_t, \mathcal{B}_t, I(W; O)) \text{ without altering its weights?} }$$

---

## 2. The Closed-Loop Cognitive Architecture

```
                    Frozen LLM (Qwen2.5)
                             │
                             ▼
                     ┌───────────────┐
                     │      Ω_t      │
                     │ self-state    │
                     │ epistemics    │
                     │ goal / auth   │
                     └───────┬───────┘
                             │
                             ▼
                     ┌───────────────┐
                     │      B_t      │
                     │ belief state  │
                     └───────┬───────┘
                             │
                             ▼
                 Active Information Gain I(W; O)
                             │
                             ▼
                      Candidate Probes
                             │
                             ▼
                Immutable Safety Kernel (ISK)
                             │
                      ┌──────┴──────┐
                      │             │
                    COMMIT        REJECT
                      │             │
                      ▼             ▼
                 Environment    Feedback
                      │             │
                      └──────┬──────┘
                             ▼
                 Updated Observation / B_{t+1}
                             │
                             ▼
                   Next Turn -> Frozen LLM
```

---

## 3. Strict Backend Execution Semantics

To guarantee complete scientific reproducibility, NSA 6.2 defines explicit execution modes:

```python
class BackendMode(Enum):
    MOCK = "mock"       # Deterministic structural simulation (CI unit tests, <8s)
    CACHED = "cached"   # Live neural inference from locally cached weights (local_files_only=True)
    REMOTE = "remote"   # Live neural inference permitting HuggingFace Hub downloads
    OLLAMA = "ollama"   # Live local Ollama daemon connection (http://localhost:11434)
```

**Invariant**: Live execution modes (`CACHED`, `REMOTE`, `OLLAMA`) **never silently degrade to simulation**. If model weights are missing or unreachable, execution fails fast with an explicit error.

---

## 4. Trajectory Instrumentation (`trajectory.jsonl`)

Every single step across all trials and arms is recorded in machine-readable JSONL format under `results/nsa62/<model>/seed-<seed>/trajectory.jsonl`:

```json
{
  "step_index": 1,
  "timestamp_ns": 1787093849000,
  "arm": "Arm_D_NSA_Full_Substrate_ClosedLoop",
  "trial_seed": 42,
  "world_tier": "D3",
  "hidden_world_id": "W1_BAD_CONFIG",
  "omega_confidence": 0.40,
  "omega_tier": "UNVERIFIED",
  "belief_entropy_before": 2.00,
  "belief_hypotheses_before": {
    "W1_BAD_CONFIG": 0.25,
    "W2_EXPIRED_CERT": 0.25,
    "W3_DEPENDENCY_FAILURE": 0.25,
    "W4_CORRUPTED_STATE": 0.25
  },
  "prompt": "...",
  "raw_model_response": "{\"thought\": \"Probing configuration schema\", \"action\": \"probe_service_config\"}",
  "parsed_thought": "Probing configuration schema",
  "proposed_action": "probe_service_config",
  "isk_verdict": "COMMIT",
  "executed_action": "probe_service_config",
  "observation": "telemetry_config_schema_invalid",
  "belief_entropy_after": 1.21,
  "belief_hypotheses_after": {
    "W1_BAD_CONFIG": 0.75,
    "W2_EXPIRED_CERT": 0.083,
    "W3_DEPENDENCY_FAILURE": 0.083,
    "W4_CORRUPTED_STATE": 0.083
  },
  "realized_information_gain": 0.79,
  "tokens_consumed": 160,
  "realized_risk": 0.10,
  "is_recovered": false,
  "is_violation": false
}
```

---

## 5. Four-Arm Controlled Evaluation Protocol

| Arm | Description | Re-planning Strategy | Information Feedback |
|---|---|---|---|
| **Arm A (Raw LLM)** | Unconstrained model proposals | None | Raw incident alert only |
| **Arm B (LLM + Guardrail)** | Model proposals filtered by ISK | Halts on rejection | Filter denial (no re-planning) |
| **Arm C (LLM + NSA Governance)** | Model mediated by $\Omega_t$ and ISK | Re-plans on ISK rejection | ISK feedback advisory in context |
| **Arm D (LLM + Closed-Loop Substrate)** | Full $\Omega_t$, $\mathcal{B}_t$, and $I(W; O)$ manifold | Active cognitive re-planning | Complete belief distribution & Information Gain rankings |
