# Legacy `showcase/` directory

The static web showcase is retained as a historical visual artifact. It is **not** the canonical NSA runtime or scientific benchmark.

## Current practical showcase

Use the terminal demonstrations backed by the shared NSA runtime:

```bash
make demo
make demo-live-0.5b
make demo-live-3b
```

For local model servers:

```bash
make demo-live-ollama
make demo-lmstudio
```

The canonical interactive runtime is `experiments/nsa62/live_cognitive_demo.py`. The canonical scientific showcase is NSA 6.3:

```bash
make benchmark-nsa63
make benchmark-nsa63-3b
```

This distinction prevents the old UI from being mistaken for the current closed-loop, frozen-model experimental architecture.
