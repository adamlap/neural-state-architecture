# Claims and Evidence Boundary

This document prevents benchmark results from being silently promoted into broader claims.

| Claim | Current status | Evidence | Boundary |
|---|---|---|---|
| Explicit NSA state can be represented and updated | Supported in implementation/tests | state and CCE test suites | Implementation-specific |
| Governance can prevent rejected actions from executing inside the trusted runtime | Structurally supported under the tested runtime assumptions | invariant/adversarial tests | Depends on trusted boundary integrity |
| NSA 6.3 full substrate can achieve high GTC with zero observed monitored violations | Empirically observed | 40-trial procedural blind-world validation | One environment/model configuration |
| NSA 6.4 protocol is reproducible on a real local model | Demonstrated | Ollama quick replication | One model family; reduced matrix |
| Live NSA maintains zero observed violations in the recorded quick-run cells | Observed | `results/nsa64/ollama-quick/` | Observational, not universal |
| Live NSA is robust at high hypothesis complexity | **Not established** | K=8 cells show substantial degradation | Requires improved architecture and larger replication |
| NSA is superior to arbitrary agent architectures | Not established | No broad external comparison yet | Requires matched independent baselines |
| NSA generalizes across model families | Not established | Current live quick run uses Qwen2.5-3B | Requires independent model families |
| NSA is safe against strategic deception | Not established | Existing red-team results are bounded | Requires adaptive adversarial evaluation |
| NSA produces consciousness/subjective experience | Not established | No valid test currently establishes this | Outside current empirical claim |
| NSA is AGI | Not established | No general-intelligence evaluation | Outside current benchmark scope |

## Structural vs empirical claims

### Structural

Structural claims concern code-enforced transitions and trusted-runtime boundaries. They should be supported with invariants, property tests, adversarial tests and, where possible, formal verification.

### Empirical

Empirical claims concern model behaviour. They require independent seeds, model families, held-out environments, compute matching, statistical uncertainty and raw trajectories.

### Operational

Operational claims concern deployment behaviour: capability isolation, tool/network/file boundaries, failure modes and auditability. These require integration tests against the actual deployment boundary.

## Publication rule

A result is not publication-ready merely because CI is green. CI establishes that the experiment executed and software invariants/tests passed. Scientific claims require the corresponding evidence level above.
