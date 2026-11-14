from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from cassi.audio.base import AudioBuffer
from cassi.core.language import Language


class TextToSpeech(Protocol):
    def synthesize(self, text: str, language: Language) -> AsyncIterator[AudioBuffer]:
        """Stream audio for one sentence; the first chunk should arrive as early as possible."""
        ...
