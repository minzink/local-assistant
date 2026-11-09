"""Wake word "Hey Flay": only what starts with it counts while listening continuously.

There is no free ready-made wake word model (openWakeWord's trained models and its training
features are non-commercial), so Flay trains its own small one (``flay train-wake``). It
sits on openWakeWord's shared feature models, which are Apache 2.0 (a melspectrogram and
Google's speech_embedding): audio → 32 mel bands every 10 ms → a 96-value embedding every
80 ms → the last 16 embeddings (1.28 s) → a two-layer network → probability of "Hey Flay".

``WakeGate`` wraps the continuous listener: an utterance passes if the wake word is heard at
its start; "Hey Flay" alone opens a short window for the next utterance.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import numpy as np

from flay.audio.base import AudioBuffer

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW = 16  # embeddings the classifier sees (1.28 s)
MEL_WINDOW = 76  # mel frames per embedding
MEL_STEP = 8  # mel frames between embeddings (80 ms)
FEATURE_MODELS = ("melspectrogram.onnx", "embedding_model.onnx")
FEATURE_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{}"
# Where in an utterance the wake word is looked for: its start.
SEARCH_SECONDS = 2.5
# "Hey Flay" alone: the next utterance within this time is the command.
FOLLOW_UP_SECONDS = 8.0
# Utterances this short may be the wake word alone.
SHORT_SECONDS = 2.0
# What speech recognition makes of the wake word at the start of a transcript.
_SPOKEN = re.compile(
    r"^\W*(?:(?:hey|hei|hi|he|ey|okay|ok)?[\s,.!-]*(?:flay|fley|flei|flai|fläi|flay's)"
    # Real words only after "Hey": "Hey, Flea, …" is the wake word, "Flee …" alone is not.
    r"|(?:hey|hei|he|ey)[\s,.!-]*(?:flea|flee))\b[\s,.!?-]*",
    re.IGNORECASE,
)


# The trained detector shipped with Flay; ``flay train-wake`` writes one into the data folder,
# which is used instead.
BUNDLED = Path(__file__).parent / "data" / "hey_flay.npz"
FEATURE_SHA256 = {
    "melspectrogram.onnx": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    "embedding_model.onnx": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
}


def classifier_path(data_dir: Path) -> Path:
    own = data_dir / "models" / "hey_flay.npz"
    return own if own.exists() else BUNDLED


def download_features(models: Path) -> None:
    """openWakeWord's feature models (Apache 2.0); only `flay setup` calls this."""
    import hashlib
    import urllib.request

    models.mkdir(parents=True, exist_ok=True)
    for name in FEATURE_MODELS:
        target = models / name
        if target.exists():
            continue
        log.info("Lade %s von github.com herunter ...", name)
        with urllib.request.urlopen(FEATURE_URL.format(name), timeout=30) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != FEATURE_SHA256[name]:
            raise RuntimeError(f"{name}: checksum mismatch {digest}")
        target.write_bytes(data)


def strip_wake_word(text: str) -> str:
    """The transcript without "Hey Flay, " at its start."""
    return _SPOKEN.sub("", text, count=1).strip()


class Features:
    """Speech embeddings of 16 kHz audio (openWakeWord's feature models)."""

    def __init__(self, models: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self._mel = ort.InferenceSession(
            str(models / FEATURE_MODELS[0]), options, providers=providers
        )
        self._embed = ort.InferenceSession(
            str(models / FEATURE_MODELS[1]), options, providers=providers
        )

    def mel(self, audio: np.ndarray) -> np.ndarray:
        """Mel frames (n, 32) of float audio in -1..1."""
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.float32)[None, :]
        spec = self._mel.run(None, {"input": pcm})[0]
        return np.asarray(np.squeeze(spec)).reshape(-1, 32) / 10 + 2  # openWakeWord's scaling

    def embeddings(self, audio: np.ndarray) -> np.ndarray:
        """Embeddings (n, 96), one per 80 ms once 76 mel frames are there."""
        mel = self.mel(audio)
        if len(mel) < MEL_WINDOW:
            mel = np.pad(mel, ((MEL_WINDOW - len(mel), 0), (0, 0)), constant_values=mel.min())
        starts = range(0, len(mel) - MEL_WINDOW + 1, MEL_STEP)
        batch = np.stack([mel[s : s + MEL_WINDOW] for s in starts])[..., None].astype(np.float32)
        return np.asarray(self._embed.run(None, {"input_1": batch})[0]).reshape(-1, 96)


def windows(embeddings: np.ndarray) -> np.ndarray:
    """All runs of 16 embeddings (n, 16 * 96); short audio is padded at the front."""
    if len(embeddings) < WINDOW:
        pad = np.repeat(embeddings[:1], WINDOW - len(embeddings), axis=0)
        embeddings = np.concatenate([pad, embeddings])
    return np.stack(
        [embeddings[i : i + WINDOW].reshape(-1) for i in range(len(embeddings) - WINDOW + 1)]
    )


class Classifier:
    """Two layers on top of the embeddings; weights from ``flay train-wake`` (.npz)."""

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.weights = weights

    @classmethod
    def load(cls, path: Path) -> Classifier:
        with np.load(path) as data:
            return cls({key: data[key] for key in data.files})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.weights)  # type: ignore[arg-type]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        w = self.weights
        x = (x - w["mean"]) / w["std"]
        hidden = np.maximum(x @ w["w1"] + w["b1"], 0)
        logits = hidden @ w["w2"] + w["b2"]
        return np.asarray(1 / (1 + np.exp(-logits.reshape(-1))))


class WakeWord:
    def __init__(self, models: Path, classifier: Path, threshold: float = 0.5) -> None:
        self._features = Features(models)
        self._classifier = Classifier.load(classifier)
        self.threshold = threshold

    def score(self, audio: np.ndarray) -> float:
        """The highest probability of "Hey Flay" in the first seconds of ``audio``."""
        start = audio[: int(SEARCH_SECONDS * SAMPLE_RATE)]
        if len(start) < SAMPLE_RATE // 4:
            return 0.0
        # Listening hands over speech from its first sound; the features need ~0.8 s before
        # it, and training put the wake word after silence too.
        padded = np.concatenate(
            [np.zeros(SAMPLE_RATE, np.float32), start, np.zeros(SAMPLE_RATE * 3 // 10, np.float32)]
        )
        return float(self._classifier(windows(self._features.embeddings(padded))).max())

    def heard(self, audio: np.ndarray) -> bool:
        return self.score(audio) >= self.threshold


class WakeGate:
    """Lets through only utterances that start with the wake word (``UtteranceSource``)."""

    def __init__(
        self,
        source: object,
        wake: WakeWord,
        on_wake: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._wake = wake
        self._on_wake = on_wake
        self._clock = clock
        self._open_until = 0.0

    def __getattr__(self, name: str) -> object:
        return getattr(self._source, name)

    async def utterances(self) -> AsyncIterator[AudioBuffer]:
        import asyncio

        async for utterance in self._source.utterances():  # type: ignore[attr-defined]
            audio = np.frombuffer(utterance.pcm, dtype="<i2").astype(np.float32) / 32768.0
            if self._clock() < self._open_until:
                self._open_until = 0.0
                yield utterance  # the command after "Hey Flay" alone
                continue
            score = await asyncio.to_thread(self._wake.score, audio)
            if score < self._wake.threshold:
                log.debug("Ohne Aktivierungswort (%.2f) – ignoriert", score)
                continue
            log.info("Aktivierungswort erkannt (%.2f)", score)
            if self._on_wake is not None:
                self._on_wake()
            if len(audio) < SHORT_SECONDS * SAMPLE_RATE:
                # Maybe just "Hey Flay" (the transcript tells): then the next utterance is
                # the command, without the wake word again.
                self._open_until = self._clock() + FOLLOW_UP_SECONDS
            yield utterance
