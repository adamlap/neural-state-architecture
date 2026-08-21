"""Speech-to-text interface for CCE."""
from __future__ import annotations
from typing import Protocol

class SpeechToText(Protocol):
    async def transcribe(self, audio: bytes, *, sample_rate: int = 16000) -> str: ...

class AudioInput(Protocol):
    async def frames(self): ...

async def bridge_audio(engine, audio_input: AudioInput, stt: SpeechToText, stop=None) -> None:
    """Consume real audio frames, transcribe them, and inject text into CCE."""
    async for frame in audio_input.frames():
        if stop is not None and stop.is_set():
            break
        text = await stt.transcribe(frame)
        if text.strip():
            await engine.ingest(text, source="speech")
