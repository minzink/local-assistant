"""Short sentences Flay says again and again ("Erledigt.", "Soll ich loslegen?"), kept as
audio: the second time they cost no synthesis.

Wraps any TTS. Only short sentences are kept (long ones rarely repeat), at most MAX_ENTRIES,
the least recently used dropped first. A new speed or voice empties the cache.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable
from typing import Any

from flay.audio.base import AudioBuffer
from flay.core.language import Language
from flay.tts.base import TextToSpeech

log = logging.getLogger(__name__)

MAX_CHARS = 60
MAX_ENTRIES = 128
# Said after actions and questions; prepared at the start.
COMMON = {
    Language.DE: [
        "Erledigt.", "Fertig.", "Okay.", "Soll ich loslegen?", "Was soll ich ändern?",
        "Ich habe es freigegeben.", "Ich habe es abgelehnt.", "Was meinst du?",
    ],
    Language.EN: ["Done.", "Okay.", "Shall I start?", "What do you mean?"],
}  # fmt: skip


class CachedTTS:
    def __init__(self, tts: TextToSpeech) -> None:
        self._tts = tts
        self._cache: OrderedDict[tuple[Language, str], list[AudioBuffer]] = OrderedDict()

    def __getattr__(self, name: str) -> Any:
        # sample_rate, load, voices, download … come from the wrapped TTS.
        return getattr(self._tts, name)

    async def synthesize(self, text: str, language: Language) -> AsyncIterator[AudioBuffer]:
        key = (language, text.strip())
        if len(key[1]) > MAX_CHARS:
            async for chunk in self._tts.synthesize(text, language):
                yield chunk
            return
        if key in self._cache:
            self._cache.move_to_end(key)
            for chunk in self._cache[key]:
                yield chunk
            return
        chunks: list[AudioBuffer] = []
        async for chunk in self._tts.synthesize(text, language):
            chunks.append(chunk)
            yield chunk
        # Only a sentence spoken to the end is kept (an interrupted one is incomplete).
        self._cache[key] = chunks
        if len(self._cache) > MAX_ENTRIES:
            self._cache.popitem(last=False)

    async def prepare(self, phrases: dict[Language, list[str]] | None = None) -> None:
        """Synthesize the common sentences ahead."""
        for language, texts in (phrases or COMMON).items():
            for text in texts:
                await self._fill(text, language)

    async def _fill(self, text: str, language: Language) -> None:
        async for _ in self.synthesize(text, language):
            pass

    def set_speed(self, speed: float) -> None:
        self._cache.clear()
        if callable(change := getattr(self._tts, "set_speed", None)):
            change(speed)

    def set_voice(self, name: str) -> None:
        self._cache.clear()
        if callable(change := getattr(self._tts, "set_voice", None)):
            change(name)

    def cached(self) -> Iterable[tuple[Language, str]]:
        return list(self._cache)
