from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

BYTES_PER_SAMPLE = 2  # 16-bit PCM


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """Mono 16-bit little-endian PCM audio."""

    pcm: bytes
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.pcm) / BYTES_PER_SAMPLE / self.sample_rate

    @property
    def level_dbfs(self) -> float:
        """RMS level in dB relative to full scale; ``-inf`` for digital silence."""
        samples = np.frombuffer(self.pcm, dtype="<i2").astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
        return 20 * float(np.log10(rms)) if rms > 0 else float("-inf")


class AudioOutput(Protocol):
    async def play(self, chunks: AsyncIterator[AudioBuffer]) -> bool:
        """Play chunks gaplessly as they arrive. Returns False if interrupted by ``stop``."""
        ...

    def stop(self) -> None:
        """Stop playback immediately; safe to call from any thread."""
        ...
