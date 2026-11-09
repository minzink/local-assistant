"""Always listening: the microphone stream is cut into utterances by Silero VAD.

Speaking while Flay talks interrupts it (barge-in). That needs a headset: through
speakers Flay would hear its own voice. Without barge-in, speech is ignored while Flay
talks.
"""

from __future__ import annotations

import asyncio
import collections
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from flay.audio.base import AudioBuffer
from flay.audio.microphone import to_pcm16
from flay.audio.vad import FRAME, SAMPLE_RATE, SileroVAD

log = logging.getLogger(__name__)

FRAME_MS = FRAME * 1000 // SAMPLE_RATE  # 32 ms


def frames(milliseconds: float) -> int:
    return max(1, round(milliseconds / FRAME_MS))


@dataclass(frozen=True, slots=True)
class SegmenterConfig:
    threshold: float = 0.5
    # Below this a frame counts as silence (hysteresis against flickering).
    stop_threshold: float = 0.35
    start_ms: int = 96
    end_silence_ms: int = 600
    # After this much silence the utterance is handed out tentatively, so transcription can
    # run while Flay still waits for the end (0 = off).
    speculate_ms: int = 250
    min_speech_ms: int = 250
    max_utterance_ms: int = 30_000
    pre_roll_ms: int = 300
    barge_in: bool = True
    # While Flay speaks, only clear and sustained speech interrupts it.
    barge_in_threshold: float = 0.85
    barge_in_ms: int = 250


@dataclass(frozen=True, slots=True)
class SegmentEvent:
    started: bool = False
    utterance: np.ndarray | None = None
    # The utterance so far, at the first pause; the final one starts with exactly these
    # samples if the user does not continue.
    tentative: np.ndarray | None = None


class Segmenter:
    """Turns per-frame speech probabilities into utterances."""

    def __init__(self, config: SegmenterConfig) -> None:
        self._config = config
        self._pre_roll: collections.deque[np.ndarray] = collections.deque(
            maxlen=frames(config.pre_roll_ms)
        )
        self._recording: list[np.ndarray] | None = None
        self._run = 0
        self._silence = 0
        self._speech = 0

    def feed(self, frame: np.ndarray, probability: float, assistant_speaking: bool) -> SegmentEvent:
        config = self._config
        if self._recording is None:
            self._pre_roll.append(frame)
            if assistant_speaking and not config.barge_in:
                self._run = 0
                return SegmentEvent()
            threshold = config.barge_in_threshold if assistant_speaking else config.threshold
            needed = frames(config.barge_in_ms if assistant_speaking else config.start_ms)
            self._run = self._run + 1 if probability >= threshold else 0
            if self._run < needed:
                return SegmentEvent()
            self._recording = list(self._pre_roll)
            self._pre_roll.clear()
            self._silence, self._speech, self._run = 0, self._run, 0
            return SegmentEvent(started=True)

        self._recording.append(frame)
        if probability < config.stop_threshold:
            self._silence += 1
        else:
            self._silence = 0
            self._speech += 1
        too_long = len(self._recording) >= frames(config.max_utterance_ms)
        if self._silence < frames(config.end_silence_ms) and not too_long:
            speculate = config.speculate_ms > 0 and self._silence == frames(config.speculate_ms)
            if speculate and self._speech >= frames(config.min_speech_ms):
                return SegmentEvent(tentative=np.concatenate(self._recording))
            return SegmentEvent()

        recording, self._recording = self._recording, None
        if self._speech < frames(config.min_speech_ms):
            return SegmentEvent()
        return SegmentEvent(utterance=np.concatenate(recording))


class ContinuousListener:
    """Implements ``UtteranceSource`` without a key: speech is detected automatically."""

    def __init__(
        self,
        vad: SileroVAD,
        config: SegmenterConfig,
        on_speech_start: Callable[[], None],
        is_speaking: Callable[[], bool],
        device: str | int | None = None,
        on_tentative: Callable[[AudioBuffer], None] | None = None,
    ) -> None:
        self._vad = vad
        self._config = config
        self._on_speech_start = on_speech_start
        self._is_speaking = is_speaking
        self._device = device
        self._on_tentative = on_tentative

    async def utterances(self) -> AsyncIterator[AudioBuffer]:
        import sounddevice as sd

        loop = asyncio.get_running_loop()
        blocks: asyncio.Queue[np.ndarray] = asyncio.Queue()
        device_rate = int(sd.query_devices(self._device, "input")["default_samplerate"])

        def callback(indata: np.ndarray, frame_count: int, time: Any, status: Any) -> None:
            loop.call_soon_threadsafe(blocks.put_nowait, indata[:, 0].copy())

        segmenter = Segmenter(self._config)
        stream = self._vad.stream()
        pending = np.zeros(0, dtype=np.float32)
        with sd.InputStream(
            samplerate=device_rate,
            blocksize=device_rate // 20,
            device=self._device,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            log.info("Höre zu (dauerhaft) – einfach sprechen.")
            while True:
                block = await blocks.get()
                pcm = to_pcm16(block, device_rate, SAMPLE_RATE)
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                pending = np.concatenate([pending, samples])
                while len(pending) >= FRAME:
                    frame, pending = pending[:FRAME], pending[FRAME:]
                    event = segmenter.feed(frame, stream.process(frame), self._is_speaking())
                    if event.started:
                        self._on_speech_start()
                    if event.tentative is not None and self._on_tentative is not None:
                        pcm16 = to_pcm16(event.tentative, SAMPLE_RATE, SAMPLE_RATE)
                        self._on_tentative(AudioBuffer(pcm=pcm16, sample_rate=SAMPLE_RATE))
                    if event.utterance is not None:
                        audio = to_pcm16(event.utterance, SAMPLE_RATE, SAMPLE_RATE)
                        yield AudioBuffer(pcm=audio, sample_rate=SAMPLE_RATE)
