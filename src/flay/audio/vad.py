"""Voice activity detection with Silero VAD (ONNX, runs on onnxruntime).

Push-to-talk uses it to trim silence around an utterance (less audio for Whisper) and to
drop recordings without speech, which Whisper would otherwise hallucinate text for.
Continuous listening will segment the microphone stream with the same model.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

from flay.audio.base import AudioBuffer
from flay.offline import MissingModelError

log = logging.getLogger(__name__)

MODEL_URL = "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
MODEL_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
MODEL_FILE = "silero_vad_v5.onnx"

SAMPLE_RATE = 16_000
FRAME = 512  # samples per inference step at 16 kHz (32 ms)
CONTEXT = 64  # samples of the previous frame the v5 model expects in front


class VoiceActivityDetector(Protocol):
    async def trim(self, audio: AudioBuffer) -> AudioBuffer | None:
        """Audio cut to the spoken part, or None if it contains no speech."""
        ...


def download_model(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("Lade Silero-VAD-Modell von github.com herunter ...")
    with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"Silero VAD checksum mismatch: {digest}")
    target.write_bytes(data)


def speech_bounds(
    probabilities: np.ndarray, threshold: float, min_speech_frames: int
) -> tuple[int, int] | None:
    """First and last frame (exclusive) of speech, or None if there is too little."""
    speech = np.flatnonzero(probabilities >= threshold)
    if speech.size < min_speech_frames:
        return None
    return int(speech[0]), int(speech[-1]) + 1


class VadStream:
    """Silero keeps state between frames; one stream per audio source."""

    def __init__(self, session: ort.InferenceSession) -> None:
        self._session = session
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        self._sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)

    def process(self, frame: np.ndarray) -> float:
        """Speech probability of one frame of ``FRAME`` float32 samples at 16 kHz."""
        x = np.concatenate([self._context, frame.reshape(1, FRAME)], axis=1)
        output, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sample_rate}
        )
        self._context = x[:, -CONTEXT:]
        return float(output[0, 0])


class SileroVAD:
    def __init__(
        self,
        model_dir: Path,
        threshold: float = 0.5,
        min_speech_seconds: float = 0.2,
        padding_seconds: float = 0.2,
    ) -> None:
        self._path = model_dir / MODEL_FILE
        self._threshold = threshold
        self._min_speech_frames = max(1, round(min_speech_seconds * SAMPLE_RATE / FRAME))
        self._padding = int(padding_seconds * SAMPLE_RATE)
        self._session: ort.InferenceSession | None = None

    def download(self) -> None:
        """Fetch the model if missing (only used by `flay setup`)."""
        if not self._path.exists():
            download_model(self._path)

    def load(self) -> None:
        if not self._path.exists():
            raise MissingModelError("Silero-VAD-Modell")
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(self._path), sess_options=options)

    def stream(self) -> VadStream:
        """A stateful detector for live audio, fed frame by frame."""
        if self._session is None:
            self.load()
        assert self._session is not None
        return VadStream(self._session)

    def probabilities(self, samples: np.ndarray) -> np.ndarray:
        """Speech probability for each 32 ms frame of 16 kHz float32 audio."""
        stream = self.stream()
        frames = range(0, len(samples) - FRAME + 1, FRAME)
        return np.array([stream.process(samples[i : i + FRAME]) for i in frames], dtype=np.float32)

    async def trim(self, audio: AudioBuffer) -> AudioBuffer | None:
        return await asyncio.to_thread(self._trim, audio)

    def _trim(self, audio: AudioBuffer) -> AudioBuffer | None:
        if audio.sample_rate != SAMPLE_RATE:
            raise ValueError(f"VAD needs {SAMPLE_RATE} Hz, got {audio.sample_rate}")
        pcm = np.frombuffer(audio.pcm, dtype="<i2")
        bounds = speech_bounds(
            self.probabilities(pcm.astype(np.float32) / 32768.0),
            self._threshold,
            self._min_speech_frames,
        )
        if bounds is None:
            return None
        start = max(0, bounds[0] * FRAME - self._padding)
        end = min(len(pcm), bounds[1] * FRAME + self._padding)
        return AudioBuffer(pcm=pcm[start:end].tobytes(), sample_rate=SAMPLE_RATE)
