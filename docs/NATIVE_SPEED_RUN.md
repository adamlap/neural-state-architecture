# Native (non-WSL) speed run

All timing numbers in this repository so far are from a CPU-only WSL2 VM (6 vCPUs). To measure what the hardware itself can do, run the same speed script against a native Windows
Ollama (and, if you have one, a GPU), and compare with WSL.

## 1. Make the Windows Ollama reachable from WSL
1. In Windows, set the user environment variable `OLLAMA_HOST` to `0.0.0.0:11434`, then quit and restart Ollama (system tray).
2. Allow inbound TCP 11434 in Windows Defender Firewall (private network / the WSL vEthernet adapter).
3. In WSL: `scripts/check_windows_ollama.sh` prints the URL that answers (the default-gateway address, e.g. `http://172.31.208.1:11434`).
4. Pull the models you want on Windows (`ollama pull qwen2.5:3b`; the Windows install already has `llama3.1:8b`, `qwen3:4b`, `qwen2.5-coder`).

## 2. Run the same measurement on both
```bash
# WSL (local server)
PYTHONPATH=. .venv/bin/python -m experiments.speed.remote_speed --ollama-url http://localhost:11434 --label wsl --models qwen2.5:3b --num-thread 6
# Windows (native), URL from check_windows_ollama.sh
PYTHONPATH=. .venv/bin/python -m experiments.speed.remote_speed --ollama-url http://172.31.208.1:11434 --label windows --models qwen2.5:3b
```
Compare `results/speed_remote_wsl.json` and `results/speed_remote_windows.json`: prefill tokens/s, decode tokens/s, and `prefill_seconds` at 4,000 tokens (what a transcript costs) versus 120 tokens
(what a retrieved memory costs). On a GPU host the absolute times fall by an order of magnitude; the ratio shows whether the retrieval advantage is still large.

## 3. Benchmarks against the remote server
Every benchmark accepts `--ollama-url`; the memory guard does not block remote servers (it cannot see their RAM). Server-side memory (`/api/ps`) is recorded by v8.
