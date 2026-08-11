# NSA Prototype Folder

Research scripts, benchmarks, and demos. Prefer the subfolder paths; top-level
`*.py` files are thin compatibility shims.

## Layout

| Subfolder | Contents |
|-----------|----------|
| `pillars/` | Pillar 1–4 validation benches (pretrain, GPU attn, LoRA, prompt injection) |
| `security/` | Leakage, multi-tier lattice, NL red-team, multi-probe |
| `retrofit/` | Open-LLM / HF retrofit, Llama showcase, evolution & native-vs-retrofit |
| `experiments/` | Toy LM, dynamic trade-off, ablations, coupling sweeps |
| `demos/` | Web UI, attention visualizer, showcase eval harness |
| `reporting/` | Benchmark report generator |
| `results/` | JSON/HTML outputs (git-ignore-friendly) |

## Examples

```bash
python prototype/experiments/toy_experiment.py
python prototype/security/nl_redteam_suite.py
python prototype/retrofit/hf_nsa_retrofit.py --model sshleifer/tiny-gpt2
make pillar-4
make tradeoff
```

`requirements.txt` stays at `prototype/requirements.txt`.
