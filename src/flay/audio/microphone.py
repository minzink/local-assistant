"""Microphone capture for push-to-talk.

The input stream stays open the whole time so recording starts without device latency;
audio is only kept between ``start_recording`` and ``stop_recording``. A short pre-roll
keeps the first syllable when the key is pressed slightly late.
"""

from __future__ import annotations

import collections
import threading
from typing import Any

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from flay.audio.base import AudioBuffer

PRE_ROLL_SECONDS = 0.25


def to_pcm16(samples: np.ndarray, source_rate: int, target_rate: int) -> bytes:
    """Convert mono float32 samples in [-1, 1] to PCM16 at ``target_rate``."""
    if source_rate != target_rate:
        divisor = np.gcd(source_rate, target_rate)
        samples = resample_poly(samples, target_rate // divisor, source_rate // divisor)
    return bytes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())


class Microphone:
    def __init__(self, target_rate: int, device: str | int | None = None) -> None:
        self._target_rate = target_rate
        self._device = device
        info = sd.query_devices(device, "input")
        self.device_name = str(info["name"])
        self._device_rate = int(info["default_samplerate"])
        blocks_per_second = 20
        self._block_size = self._device_rate // blocks_per_second
        self._pre_roll: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, int(PRE_ROLL_SECONDS * blocks_per_second))
        )
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def open(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self._device_rate,
            blocksize=self._block_size,
            device=self._device,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start_recording(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._frames = list(self._pre_roll)
            self._recording = True

    def stop_recording(self) -> AudioBuffer:
        with self._lock:
            self._recording = False
            frames, self._frames = self._frames, []
        samples = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        return AudioBuffer(
            pcm=to_pcm16(samples, self._device_rate, self._target_rate),
            sample_rate=self._target_rate,
        )

    def _callback(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        block = indata[:, 0].copy()
        with self._lock:
            if self._recording:
                self._frames.append(block)
            else:
                self._pre_roll.append(block)
