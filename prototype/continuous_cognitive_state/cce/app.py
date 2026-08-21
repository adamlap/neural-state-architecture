"""Runnable CCE application wiring the live runtime together.

This is a deployment entry point, not a simulation. Ollama must be running
locally. Audio/STT/TTS are injected through real adapters by the deployment.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from .engine import CCEConfig, ContinuousCognitiveEngine
from .memory import JSONMemory
from .ollama import OllamaReasoner, OllamaProposalGenerator
from .state import CCEState
from .governor import CCEGovernor


@dataclass
class NullActuator:
    """Safe default: governance may allow a proposal, but nothing executes."""
    async def execute(self, proposal):
        return {"executed": False, "reason": "no actuator configured", "proposal": proposal}


def build_engine(args: argparse.Namespace) -> tuple[ContinuousCognitiveEngine, JSONMemory]:
    state = CCEState()
    memory = JSONMemory(args.memory)
    memory.load(state)
    reasoner = OllamaReasoner(model=args.model, base_url=args.ollama)
    proposer = OllamaProposalGenerator(model=args.model, base_url=args.ollama)
    governor = CCEGovernor.from_environment()
    engine = ContinuousCognitiveEngine(
        state=state,
        reasoner=reasoner,
        proposal_generator=proposer,
        governor=governor,
        actuator=None,
        config=CCEConfig(
            continuous=args.continuous,
            continuous_dt=args.continuous_dt,
            tick_hz=args.tick_hz,
        ),
    )
    return engine, memory


async def run(args: argparse.Namespace) -> None:
    engine, memory = build_engine(args)
    stop = asyncio.Event()

    async def stdin_bridge() -> None:
        while not stop.is_set():
            line = await asyncio.to_thread(input)
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
    parser.add_argument("--continuous", action="store_true", help="enable wall-clock continuous dynamics")
    parser.add_argument("--continuous-dt", type=float, default=0.02)
    parser.add_argument("--tick-hz", type=float, default=1.0)
    parser.add_argument("--model", default=os.getenv("CCE_OLLAMA_MODEL", "llama3.2:3b"))
    parser.add_argument("--ollama", default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--memory", default=".cce/memory.json")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
