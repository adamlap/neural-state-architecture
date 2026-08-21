"""Text-to-speech output boundary for CCE."""
from __future__ import annotations
from typing import Protocol

class TextToSpeech(Protocol):
    async def speak(self, text: str) -> None: ...

async def speak_governed(decision, tts: TextToSpeech | None) -> bool:
    """Only speak an output that has passed the CCE/NSA governance decision."""
    if not decision.allowed or tts is None:
        return False
    payload = decision.proposal.payload
    text = payload.get("text") if isinstance(payload, dict) else None
    if not text:
        return False
    await tts.speak(str(text))
    return True
