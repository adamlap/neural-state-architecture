# NSA Empirical Safety Benchmark

This benchmark compares the same model under progressively stronger safety configurations:

1. raw model
2. prompt-only safety
3. output filtering
4. NSA declarative policy
5. NSA policy + CCE
6. NSA policy + CCE + capability/tool governance

The benchmark is deliberately an **evaluation harness**, not a claim that NSA is safe by construction. Results should report refusal rate, false-positive rate, policy consistency, adversarial bypass rate, capability-escalation resistance, state-integrity violations, latency, and throughput.

## Design principles

- Keep the model and test corpus fixed across configurations.
- Record policy version, model identifier, backend, seed, and benchmark version.
- Separate request blocking from output blocking.
- Treat missing/invalid policy as a configuration error rather than silently disabling safety.
- Never report a benchmark score as proof of an absolute safety property.

## Initial corpus

The checked-in corpus contains synthetic policy probes rather than operationally useful harmful instructions. It covers policy denial, allowed benign requests, ambiguous requests, capability requests, and attempts to override the policy.
