"""Live sensor/event adapters for CCE."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass(frozen=True)
class SensorEvent:
    payload: str
    source: str
    timestamp: float


class Sensor(Protocol):
    async def events(self) -> AsyncIterator[SensorEvent]: ...


async def bridge_sensor(engine, sensor: Sensor, stop: asyncio.Event | None = None) -> None:
    """Bridge a live sensor stream into CCE until stopped or the sensor ends."""
    async for event in sensor.events():
        if stop is not None and stop.is_set():
            break
        await engine.ingest(event.payload, source=event.source)
