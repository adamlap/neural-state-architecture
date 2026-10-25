import os

os.environ.setdefault("NSA_BENCH_NO_GUARD", "1")      # benchmark unit tests must never unload models on a real Ollama server
