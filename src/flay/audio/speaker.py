from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

import sounddevice as sd

from flay.audio.base import BYTES_PER_SAMPLE, AudioBuffer

WRITE_SECONDS = 0.05  # granularity at which playback can be stopped


class Speaker:
    """Plays PCM16 chunks; implements ``AudioOutput``.

    The output stream stays open between replies because opening a device costs
    several hundred milliseconds (more on wireless headsets). It is only reopened when
    the sample rate changes.
    """

    def __init__(self, device: str | int | None = None) -> None:
        self._device = device
        self._stopped = threading.Event()
        self._stream: sd.RawOutputStream | None = None
        self._sample_rate: int | None = None

    def prepare(self, sample_rate: int) -> None:
        """Open the stream ahead of the first reply."""
        if self._stream is not None and self._sample_rate == sample_rate:
            return
        self.close()
        self._stream = sd.RawOutputStream(
            samplerate=sample_rate,
            device=self._device,
            channels=1,
            dtype="int16",
            latency="low",
        )
        self._stream.start()
        self._sample_rate = sample_rate

    def close(self) -> None:
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None

    async def play(self, chunks: AsyncIterator[AudioBuffer]) -> bool:
        self._stopped.clear()
        played = False
        async for chunk in chunks:
            if self._stopped.is_set():
                return False
            self.prepare(chunk.sample_rate)
            await asyncio.to_thread(self._write, chunk)
            played = True
        if played and self._stream is not None and not self._stopped.is_set():
            # write() returns once the data is buffered; wait for the tail to be heard.
            await asyncio.sleep(float(self._stream.latency))
        return not self._stopped.is_set()

    def stop(self) -> None:
        self._stopped.set()

    def _write(self, chunk: AudioBuffer) -> None:
        stream = self._stream
        assert stream is not None
        step = int(chunk.sample_rate * WRITE_SECONDS) * BYTES_PER_SAMPLE
        for offset in range(0, len(chunk.pcm), step):
            if self._stopped.is_set():
                return
            stream.write(chunk.pcm[offset : offset + step])
