"""Short fillers ("Mhm.", "Moment.") while the answer is still being prepared.

They are synthesized once and cached, so playing one costs no time. A filler only plays if
the answer has not started speaking by its deadline; a fast answer comes without one.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Callable

import numpy as np

from flay.audio.base import AudioBuffer, AudioOutput
from flay.config import FillerSettings
from flay.core.language import Language
from flay.tts.base import TextToSpeech

log = logging.getLogger(__name__)

# Below this amplitude (about -50 dBFS) a sample counts as silence.
SILENCE_LEVEL = 100


class FillerVoice:
    """The filler phrases as ready-made audio."""

    def __init__(
        self, tts: TextToSpeech, settings: FillerSettings, rng: random.Random | None = None
    ) -> None:
        self._tts = tts
        self._settings = settings
        self._rng = rng or random.Random()
        self._cache: dict[tuple[Language, str], list[AudioBuffer]] = {}
        self._last: str | None = None

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    @property
    def delay(self) -> float:
        return self._settings.delay_ms / 1000

    def _phrases(self, language: Language, action: bool) -> list[str]:
        table = self._settings.action_phrases if action else self._settings.phrases
        return table.get(language, [])

    async def prepare(self) -> None:
        """Synthesize all phrases ahead, so the first filler plays without delay."""
        for language in Language:
            for action in (False, True):
                for phrase in self._phrases(language, action):
                    await self._audio(language, phrase)

    async def _audio(self, language: Language, phrase: str) -> list[AudioBuffer]:
        key = (language, phrase)
        if key not in self._cache:
            chunks = [chunk async for chunk in self._tts.synthesize(phrase, language)]
            self._cache[key] = [trimmed] if (trimmed := trim_silence(chunks)) else []
        return self._cache[key]

    async def pick(self, language: Language, action: bool) -> list[AudioBuffer]:
        """Audio of a phrase, not the same as last time if there is a choice."""
        phrases = self._phrases(language, action)
        if not phrases:
            return []
        fresh = [p for p in phrases if p != self._last] or phrases
        phrase = self._rng.choice(fresh)
        self._last = phrase
        log.debug("Füllwort: %s", phrase)
        return await self._audio(language, phrase)


class TurnFiller:
    """The filler of one turn.

    Until the answer is ready to speak, the filler plays on its own once its deadline
    passes. Then the answer takes over (``hand_over``): a filler that is playing finishes,
    one that has not started yet is played by the answer's audio stream if the first
    sentence is still not ready at the deadline (``remaining`` / ``take``).
    """

    def __init__(
        self,
        voice: FillerVoice,
        output: AudioOutput,
        language: Language,
        clock: Callable[[], float],
        on_play: Callable[[bool], None] = lambda playing: None,
    ) -> None:
        self._voice = voice
        self._output = output
        self._clock = clock
        self._on_play = on_play
        self._deadline = clock() + voice.delay
        self.language = language
        # The request asks for an action: "Moment." fits better than "Mhm.".
        self.action = False
        self._used = False
        self._handed_over = False
        self._task = asyncio.get_running_loop().create_task(self._play_alone())

    async def _play_alone(self) -> None:
        await asyncio.sleep(max(0.0, self._deadline - self._clock()))
        if self._handed_over or self._used:
            return
        self._used = True
        audio = await self._voice.pick(self.language, self.action)
        if not audio:
            return
        self._on_play(True)
        try:
            await self._output.play(_iterate(audio))
        finally:
            self._on_play(False)

    async def hand_over(self) -> None:
        """The answer is about to speak: let a playing filler finish, stop a waiting one."""
        self._handed_over = True
        if not self._used:
            self._task.cancel()
        # asyncio.wait, not "await task" under suppress(CancelledError): that would also
        # swallow a cancellation of the turn itself – the user's interruption got lost.
        await asyncio.wait([self._task])

    def remaining(self) -> float | None:
        """Seconds the answer may still take before the filler is due; None if done."""
        if self._used:
            return None
        return max(0.0, self._deadline - self._clock())

    async def take(self) -> list[AudioBuffer]:
        """The filler's audio, for the answer's stream to play first."""
        if self._used:
            return []
        self._used = True
        return await self._voice.pick(self.language, self.action)

    def cancel(self) -> None:
        self._handed_over = True
        self._task.cancel()


def trim_silence(chunks: list[AudioBuffer], pad_seconds: float = 0.05) -> AudioBuffer | None:
    """One buffer without the silence TTS puts around a phrase: the answer follows sooner."""
    if not chunks:
        return None
    rate = chunks[0].sample_rate
    samples = np.frombuffer(b"".join(c.pcm for c in chunks), dtype="<i2")
    loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > SILENCE_LEVEL)
    if loud.size == 0:
        return None
    pad = int(pad_seconds * rate)
    start, end = max(0, int(loud[0]) - pad), min(samples.size, int(loud[-1]) + pad)
    return AudioBuffer(pcm=samples[start:end].tobytes(), sample_rate=rate)


async def _iterate(chunks: list[AudioBuffer]) -> AsyncIterator[AudioBuffer]:
    for chunk in chunks:
        yield chunk
