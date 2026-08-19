# Legacy `demo/` directory

This directory contains historical NSA demonstrations, including the earlier retrofit/LoRA showcase.

## Current canonical demos

Do not use the legacy scripts as the primary scientific demonstration. Use the current runtime instead:

```bash
make demo
make demo-live-0.5b
make demo-live-3b
make demo-live-ollama
make demo-lmstudio
```

The current runtime is implemented in `experiments/nsa62/live_cognitive_demo.py` and uses the shared inference adapters under `nsa/runtime/inference/`.

The legacy files remain in the repository so historical experiments and comparisons are not lost.
