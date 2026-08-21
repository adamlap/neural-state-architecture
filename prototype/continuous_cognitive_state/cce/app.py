"""Runnable CCE local application.

Ollama is a real local dependency. The default actuator is intentionally absent:
ALLOW without an attached real actuator becomes HOLD rather than a side effect.
"""
from __future__ import annotations

import argparse
import asyncio
import os

from .engine import CCEConfig, ContinuousCognitiveEngine
from .memory import JSONMemory
from .ollama import OllamaProposalGenerator, OllamaReasoner
from .state import CCEState
from .governor import CCEGovernor


def build_engine(args: argparse.Namespace):
    state = CCEState()
    memory = JSONMemory(args.memory)
    memory.load(state)
    engine = ContinuousCognitiveEngine(
        state=state,
        reasoner=OllamaReasoner(model=args.model, base_url=args.ollama),
        proposal_generator=OllamaProposalGenerator(model=args.model, base_url=args.ollama),
        governor=CCEGovernor.from_environment(),
        actuator=None,
        config=CCEConfig(continuous=args.continuous, continuous_dt=args.continuous_dt, tick_hz=args.tick_hz),
    )
    return engine, memory


async def run(args: argparse.Namespace) -> None:
    engine, memory = build_engine(args)
    stop = asyncio.Event()

    async def stdin_bridge() -> None:
        while not stop.is_set():
            line = await asyncio.to_thread(input, "CCE> ")
            if line.strip() in {"/quit", "/exit"}:
                stop.set()
                break
            if line.strip():
                await engine.ingest(line, source="stdin")

    try:
        await asyncio.gather(engine.run(stop=stop), stdin_bridge())
    finally:
        memory.save(engine.state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CCE research runtime")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--continuous-dt", type=float, default=0.02)
    parser.add_argument("--tick-hz", type=float, default=1.0)
    parser.add_argument("--model", default=os.getenv("CCE_OLLAMA_MODEL", "llama3.2:3b"))
    parser.add_argument("--ollama", default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--memory", default=".cce/memory.json")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
