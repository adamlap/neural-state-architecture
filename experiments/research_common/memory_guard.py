"""Refuse to load a model that would not fit, and unload the others first.

Why this exists: on an 8.9 GB WSL VM a 7B model (4.7 GB) at a 32k-token context (+~2 GB KV cache), loaded while
smaller models were still resident (Ollama keeps up to three loaded by default), exhausted RAM and froze the
machine. A benchmark must fail with a message *before* that happens, not after.
"""
from __future__ import annotations

import json
import os
import urllib.request

MARGIN_BYTES = int(1.2 * 2**30)           # headroom for the OS, page cache churn and compute buffers
KV_BYTES_PER_TOKEN = {"small": 40_000, "large": 70_000}     # rough upper bounds for GQA models (<4B / >=4B weights)


def available_bytes(meminfo_path: str = "/proc/meminfo") -> int | None:
    try:
        with open(meminfo_path) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _get(url: str, payload: dict | None = None, timeout: float = 10.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def model_weight_bytes(base_url: str, model: str) -> int | None:
    try:
        for m in _get(f"{base_url.rstrip('/')}/api/tags").get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return int(m["size"])
    except Exception:
        return None
    return None


# Measured on this machine (llama.cpp via Ollama, CPU): qwen2.5:7b (4.36 GiB weights) at num_ctx 8192 had a 6.86 GB resident set. Weights + a
# 70 KB/token KV estimate only explains ~4.9 GiB; the rest (~2 GiB) is compute buffers and runtime overhead, so it is added explicitly.
RUNTIME_OVERHEAD_BYTES = {"small": int(0.5 * 2**30), "large": int(2.0 * 2**30)}


def estimate_need_bytes(weights: int, num_ctx: int) -> int:
    size = "large" if weights >= 3.5 * 2**30 else "small"
    return weights + RUNTIME_OVERHEAD_BYTES[size] + num_ctx * KV_BYTES_PER_TOKEN[size] + MARGIN_BYTES


def unload_all(base_url: str) -> list[str]:
    """Ask the server to drop every loaded model (keep_alive=0); returns the names it unloaded."""
    names = []
    try:
        for m in _get(f"{base_url.rstrip('/')}/api/ps").get("models", []):
            name = m.get("name") or m.get("model")
            _get(f"{base_url.rstrip('/')}/api/generate", {"model": name, "keep_alive": 0})
            names.append(name)
    except Exception:
        pass
    return names


def wait_until_unloaded(base_url: str, timeout: float = 30.0, poll: float = 1.0) -> bool:
    """Unloading is asynchronous: wait until the server reports no loaded model, so available memory is measured after it is freed."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if not _get(f"{base_url.rstrip('/')}/api/ps").get("models"):
                time.sleep(poll)                                   # let the OS reclaim the pages
                return True
        except Exception:
            return False
        time.sleep(poll)
    return False


def prepare_model(base_url: str | None, model: str, num_ctx: int, *, meminfo_path: str = "/proc/meminfo") -> dict:
    """Unload other models, then abort (SystemExit) if the model plus its KV cache cannot fit in available RAM."""
    if os.environ.get("NSA_BENCH_NO_GUARD"):               # unit tests: never touch a real server
        return {"unloaded": [], "weights": None, "available": None, "need": None}
    url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not url.startswith("http"):
        url = "http://" + url
    unloaded = unload_all(url)
    if unloaded:
        wait_until_unloaded(url)
    weights = model_weight_bytes(url, model)
    avail = available_bytes(meminfo_path)
    info = {"unloaded": unloaded, "weights": weights, "available": avail, "need": None}
    if weights is None or avail is None:
        return info                                        # cannot judge (remote server or non-Linux): do not block
    need = estimate_need_bytes(weights, num_ctx)
    info["need"] = need
    local = "localhost" in url or "127.0.0.1" in url
    if local and need > avail:
        raise SystemExit(
            f"refusing to load {model} at num_ctx={num_ctx}: estimated {need / 2**30:.1f} GiB needed "
            f"(weights {weights / 2**30:.1f} + KV + {MARGIN_BYTES / 2**30:.1f} margin) but only {avail / 2**30:.1f} GiB is available. "
            "Lower --num-ctx, stop other processes, or use a smaller model.")
    return info
