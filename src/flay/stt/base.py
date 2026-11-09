from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from flay.audio.base import AudioBuffer
from flay.core.language import Language


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: Language


class SpeechToText(Protocol):
    async def transcribe(
        self, audio: AudioBuffer, language_hint: Language | None = None
    ) -> Transcript:
        """Transcribe one utterance; without a hint the language is detected (de/en only)."""
        ...
