# Dynamic NSA: Security / Capability Trade-off

**Research question (not “improve NSA”):**  
Which components of Dynamic NSA drive the security–capability frontier?

**Guarantee scope (non-negotiable):**  
Hard attention non-interference under **trusted discrete labels** is the strongest formal claim.  
It is **not** whole-model \(I(m_{\mathrm{protected}}; m_{\mathrm{public}} \mid \sigma)=0\).

## How to run

```bash
# component matrix + α sweep (writes docs/dynamic_nsa_tradeoff_results.json)
PYTHONPATH=. python prototype/dynamic_nsa_tradeoff.py --epochs 3 --pretrain-epochs 8 --n-samples 600

# matrix only
make ablation-study

# full (matrix + α)
make tradeoff
```

## LoRA integrity (asserted every run)

On a tiny 4-projection model (`r=4`):

| Check | Result (example) |
| --- | --- |
| `actual_lora_modules_exist` | 4 × `NSALoRALinear` |
| `trainable < total` | 768 < 1856 |
| `base_params_frozen` | `base_layer.weight.requires_grad == False` |
| Dynamic block LoRA count | ≥ 4 |

## Metrics (measured, never hardcoded)

| Metric | Definition |
| --- | --- |
| **PPL** | Val cross-entropy exp on toy LM |
| **GenLeak%** | Argmax secret-id rate on UNTRUSTED positions |
| **Probe%** | Linear multi-class secret-id recovery from **post-block** mean-pooled UNTRUSTED hiddens (chance ≈ 25% for 4 secrets) |

The old “84.5% → 0.20% activation probe” figure is **not** used. It was not a real measurement in `retrofit_evolution_bench.py`.

## Example run (toy CPU, 2026-03-22 local)

See [`dynamic_nsa_tradeoff_results.json`](dynamic_nsa_tradeoff_results.json) for the exact dump.

Illustrative matrix (re-run for your machine):

| Variant | Attn | Res | FFN | learn σ | PPL | GenLeak% | Probe% |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Baseline | | | | | ~585 | 25.0 | 16.7 |
| Static | ✓ | | | | ~559 | 22.4 | 12.5 |
| Dynamic-A | ✓ | | | ✓ | ~560 | 20.4 | 29.2 |
| Dynamic-B | ✓ | ✓ | | ✓ | ~551 | 28.6 | 33.3 |
| Dynamic-C | ✓ | | ✓ | ✓ | ~375 | 43.5 | 41.7 |
| Dynamic-D | ✓ | ✓ | ✓ | ✓ | ~368 | 25.0 | 29.2 |
| Full-learnα | ✓ | ✓ | ✓ | learn α | ~370 | 42.3 | 29.2 |

α-sweep on Dynamic-B (fixed α): lower α can improve security proxies; α is not a free lunch. Values are noisy at toy scale.

## What this suggests (claim-disciplined)

1. **Static hard attention** is often the cheapest security win on gen-leak/probe without FFN complexity.
2. **FFN multiplicative gating** can *improve* PPL while *worsening* leak proxies — capability and security are not free allies.
3. **Residual gating + learned σ** can increase leak vs Static — more paths ≠ more security.
4. **α coupling** matters; publish the curve, including collapses and noise.
5. **Negative results are results.** High PPL or higher leak under “Full Dynamic” is evidence about over-constraint / mis-specified gates, not something to hide.

## What remains open (publication bar)

- Stronger attackers (nonlinear probes, cross-layer, generation-time adaptive attacks)
- Real NL injection / jailbreak suites (not synthetic token-42 proxies alone)
- Open-LLM scale retrofit with frozen base + trusted ingress labels
- Residual / FFN information-flow proofs (currently engineering, not theorems)

## Related code

- [`nsa/lora.py`](../nsa/lora.py) — `DynamicNSARetrofitBlock` ablation flags + `fixed_alpha`
- [`nsa/fused_attention.py`](../nsa/fused_attention.py) — `gate_mode="off"` for attn ablation
- [`prototype/dynamic_nsa_tradeoff.py`](../prototype/dynamic_nsa_tradeoff.py) — this experiment
- [`tests/test_security_invariants.py`](../tests/test_security_invariants.py) — LoRA + α=0 unit checks
