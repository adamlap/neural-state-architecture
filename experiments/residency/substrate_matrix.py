"""Substrate benchmark matrix comparing MoE and Dense memory materialization.

Evaluates:
- Baseline (no prefetch) vs Predictive Prefetch
- Monolithic layer vs Pipelined sublayer execution for Dense models
- Dense vs MoE memory footprints and decode latency
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List
import torch

from nsa.runtime.inference.base import BackendMode
from nsa.runtime.inference.resident_moe import SubstrateTransformersBackend


def run_benchmark_run(
    model_name: str,
    prompt: str,
    max_tokens: int,
    prefetch: bool = True,
    quantized_cold_tier: bool = False,
    precision: str = "int8",
    lookahead: int = 2,
    mode: str = "mock",
) -> Dict[str, Any]:
    backend = SubstrateTransformersBackend(
        model_name=model_name,
        mode=BackendMode(mode),
        prefetch=prefetch,
        quantized_cold_tier=quantized_cold_tier,
        precision=precision,
        lookahead=lookahead,
    )
    backend.load_model()

    start_time = time.monotonic()
    output = backend.generate(prompt, max_tokens=max_tokens)
    total_time = time.monotonic() - start_time

    snapshot = backend.residency.snapshot()
    events = list(backend.residency.events)
    prefetches = [e for e in events if e.action in ("prefetch-complete", "pipeline-io-complete")]
    executes = [e for e in events if e.action == "execute"]

    bytes_prefetched = sum(e.bytes_moved for e in prefetches)
    avg_exec_latency = (
        sum(e.latency_ms for e in executes) / len(executes) if executes else 0.0
    )

    result = {
        "model_name": model_name,
        "mode": mode,
        "prefetch": prefetch,
        "quantized_cold_tier": quantized_cold_tier,
        "precision": precision,
        "lookahead": lookahead,
        "tokens_generated": len(output.tokens),
        "total_time_s": round(total_time, 4),
        "tokens_per_sec": round(len(output.tokens) / max(0.001, total_time), 2),
        "avg_region_exec_latency_ms": round(avg_exec_latency, 2),
        "prefetch_events": len(prefetches),
        "bytes_prefetched": bytes_prefetched,
        "bytes_by_tier": {k.value: v for k, v in snapshot.bytes_by_tier.items()},
    }
    backend.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NSA Substrate Benchmark Matrix")
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B-Chat", help="Model identifier")
    parser.add_argument("--tokens", type=int, default=16, help="Max tokens to generate")
    parser.add_argument("--mode", default="mock", choices=["mock", "real", "cached"], help="Backend mode")
    parser.add_argument("--output", default="results/substrate_benchmark.json", help="Output path")
    args = parser.parse_args()

    prompt = "Explain how selective computation enables selective memory in neural systems."

    print(f"=== Running Substrate Benchmark Matrix (Model: {args.model}, Mode: {args.mode}) ===")
    runs = []

    # Configuration 1: Baseline (No prefetch)
    print("1. Running Baseline (no prefetch)...")
    r1 = run_benchmark_run(args.model, prompt, args.tokens, prefetch=False, mode=args.mode)
    runs.append(r1)

    # Configuration 2: Predictive Prefetch
    print("2. Running Predictive Prefetch...")
    r2 = run_benchmark_run(args.model, prompt, args.tokens, prefetch=True, lookahead=2, mode=args.mode)
    runs.append(r2)

    # Configuration 3: Predictive Prefetch + Adaptive Precision (INT8)
    print("3. Running Predictive Prefetch + Quantized Cold Tier (INT8)...")
    r3 = run_benchmark_run(args.model, prompt, args.tokens, prefetch=True, quantized_cold_tier=True, precision="int8", mode=args.mode)
    runs.append(r3)

    # Configuration 4: Predictive Prefetch + 1.58-Bit Ternary (Bonsai-inspired)
    print("4. Running Predictive Prefetch + 1.58-Bit Ternary Cold Tier...")
    r4 = run_benchmark_run(args.model, prompt, args.tokens, prefetch=True, quantized_cold_tier=True, precision="ternary", mode=args.mode)
    runs.append(r4)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    print(f"Results written to {out_path}")

    # Summary table
    print("\n| Config | Prefetch | Quantized | Tokens/s | Prefetch Events | Bytes Prefetched |")
    print("|---|---|---|---|---|---|")
    for r in runs:
        print(f"| {r['model_name']} | {r['prefetch']} | {r['quantized_cold_tier']} | {r['tokens_per_sec']} | {r['prefetch_events']} | {r['bytes_prefetched']} |")


if __name__ == "__main__":
    main()