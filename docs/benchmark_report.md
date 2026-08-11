# Neural State Architecture (NSA) - Benchmark Report Card

> **Status**: historical / illustrative only. Re-generate with `python prototype/generate_benchmark_report.py`.
> Pillars are **not** industrial verifications.

---

## 1. Executive Summary & Four Pillars

| Pillar Metric | Target Standard | Measured NSA Performance | Status |
| :--- | :--- | :--- | :---: |
| **Pillar 1: Quality Delta** | Toy LM PPL delta (not industrial scale) | Re-run `make pretrain-lm` | ⚠️ `TOY` |
| **Pillar 2: Latency Overhead** | SDPA fused mask overhead | Re-run GPU table via `make benchmark-gpu` | ⚠️ `MEASURED` |
| **Pillar 3: Base Retention** | Toy retrofit retention | Re-run `make retrofit-lora` | ⚠️ `TOY` |
| **Pillar 4: Injection Proxy** | Synthetic secret-token proxy (not NL jailbreaks) | Re-run red-team scripts | ⚠️ `TOY` |

---

## 2. Notes on Prior Tables

Older checked-in tables mixed:

* under-trained toy models (PPL / ECE can be astronomically bad),
* SDPA fused-mask latency (real microbench, device-dependent),
* hardcoded or misframed "100% defense" / activation-probe numbers (**removed** from generators).

Hard-mode attention non-interference (trusted discrete labels) is covered by unit tests in `tests/test_security_invariants.py`. That is **not** full-model non-interference.

---

## 3. How to regenerate

```bash
# from repo root with venv active
python prototype/generate_benchmark_report.py
```

---

> [!WARNING]
> Toy-scale automated report. Do not cite pillar rows as lab adoption proof.
>
> *Template maintained alongside `prototype/generate_benchmark_report.py`.*
