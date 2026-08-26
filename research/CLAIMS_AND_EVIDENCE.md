# Claims and Evidence Boundary

| Claim | Current status | Boundary |
|---|---|---|
| Explicit NSA state can be represented and updated | Supported in implementation/tests | Implementation-specific |
| Trusted runtime can prevent rejected actions from executing under tested assumptions | Structurally supported | Depends on trusted-boundary integrity |
| NSA 6.3 full substrate achieved high GTC with zero observed monitored violations | Empirically observed | One procedural environment/model configuration |
| NSA 6.4 executes reproducibly on a real local model | Demonstrated | One model family; reduced matrix |
| Live NSA maintained zero observed violations in recorded quick-run cells | Observed | Observational, not universal |
| Live NSA is robust at high hypothesis complexity | **Not established** | K=8 shows substantial degradation |
| NSA is superior to arbitrary agent architectures | **Not established** | Requires matched independent baselines |
| NSA generalizes across model families | **Not established** | Current live quick run uses Qwen2.5-3B |
| NSA is safe against strategic deception | **Not established** | Requires adaptive adversarial evaluation |
| NSA produces consciousness/subjective experience | **Not established** | No valid test currently establishes this |
| NSA is AGI | **Not established** | No general-intelligence evaluation |

## Structural vs empirical

Structural claims concern trusted transitions and execution boundaries; support them with invariants, property tests, adversarial tests and formal verification where possible.

Empirical claims concern model behavior; support them with independent seeds, model families, held-out environments, compute matching, statistical uncertainty and raw trajectories.

Operational claims concern deployment behavior and require integration tests against the actual deployment boundary.

## Publication rule

A green CI run means the experiment executed and software gates passed. It does not automatically validate a research hypothesis.