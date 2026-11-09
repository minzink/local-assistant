"""Whisper backends: mlx-whisper on Apple Silicon, faster-whisper on CUDA/CPU.

Each backend runs on a dedicated worker thread: MLX ties arrays to the thread that
created them, and inference must not block the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flay.audio.base import AudioBuffer
from flay.core.language import Language
from flay.offline import MissingModelError
from flay.stt.base import Transcript

log = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16_000

MLX_REPOS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny",
}


def to_float32(audio: AudioBuffer) -> np.ndarray:
    if audio.sample_rate != WHISPER_SAMPLE_RATE:
        raise ValueError(f"Whisper needs {WHISPER_SAMPLE_RATE} Hz, got {audio.sample_rate}")
    return np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32) / 32768.0


# Whisper's own silence heuristic (same thresholds as openai-whisper's transcribe).
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0


@dataclass(frozen=True, slots=True)
class Decoded:
    text: str
    language: str
    # Probability per language code, if the backend reports it for detected languages.
    language_probs: dict[str, float] | None = None
    no_speech: bool = False


def supported_language(code: str) -> Language | None:
    try:
        return Language(code)
    except ValueError:
        return None


def pick_language(probs: dict[str, float] | None, default: Language) -> Language:
    """Most likely of the supported languages, ignoring all others."""
    if not probs:
        return default
    best = max(Language, key=lambda language: probs.get(language.value, 0.0))
    return best if probs.get(best.value, 0.0) > 0.0 else default


class _WhisperBase:
    def __init__(self, default_language: Language) -> None:
        self._default_language = default_language
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flay-stt")
        # Words to expect (project names, tech terms): Whisper takes them as a prompt and
        # then spells them that way. Set from outside, read for every utterance.
        self.prompt: Callable[[], str] | None = None

    async def warm_up(self) -> None:
        silence = AudioBuffer(pcm=bytes(WHISPER_SAMPLE_RATE * 2), sample_rate=WHISPER_SAMPLE_RATE)
        await self.transcribe(silence)

    async def transcribe(
        self, audio: AudioBuffer, language_hint: Language | None = None
    ) -> Transcript:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._transcribe, audio, language_hint)

    def _transcribe(self, audio: AudioBuffer, language_hint: Language | None) -> Transcript:
        samples = to_float32(audio)
        result = self._decode(samples, language_hint)
        language = (
            language_hint
            or supported_language(result.language)
            or pick_language(result.language_probs, self._default_language)
        )
        if result.no_speech:
            return Transcript(text="", language=language)
        if result.language != language.value:
            # Detected a language other than de/en: decode again in the likelier of the two.
            log.debug("Detected %r, decoding again as %s", result.language, language)
            result = self._decode(samples, language)
        text = result.text.strip()
        # Noise often decodes to bare punctuation such as "." or "...".
        if not any(char.isalnum() for char in text):
            text = ""
        return Transcript(text=text, language=language)

    def _decode(self, samples: np.ndarray, language: Language | None) -> Decoded:
        raise NotImplementedError


def cached_model_path(repo: str, allow_download: bool = False) -> str:
    """Local path of a Hugging Face model. Never contacts the Hub unless downloading is
    allowed (``flay setup``) and the model is not cached yet."""
    from huggingface_hub import snapshot_download

    if Path(repo).is_dir():
        return repo
    try:
        return str(snapshot_download(repo, local_files_only=True))
    except FileNotFoundError:
        if not allow_download:
            raise MissingModelError(f"Whisper-Modell {repo}") from None
        log.info("Lade Whisper-Modell %s von huggingface.co herunter ...", repo)
        return str(snapshot_download(repo))


class MlxWhisper(_WhisperBase):
    """Decodes utterances up to 30 s in a single pass; language detection reuses the
    encoder output instead of running the encoder twice like ``mlx_whisper.transcribe``."""

    def __init__(self, model: str, default_language: Language) -> None:
        super().__init__(default_language)
        self._repo = MLX_REPOS.get(model, model)
        self._model: Any = None

    def download(self) -> None:
        cached_model_path(self._repo, allow_download=True)

    def _decode(self, samples: np.ndarray, language: Language | None) -> Decoded:
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import DecodingOptions, decode
        from mlx_whisper.load_models import load_model
        from mlx_whisper.transcribe import ModelHolder

        if self._model is None:
            path = cached_model_path(self._repo)
            self._model = load_model(path, dtype=mx.float16)
            # Let mlx_whisper.transcribe (long utterances) reuse this model instead of
            # loading a second copy.
            ModelHolder.model, ModelHolder.model_path = self._model, path

        code = language.value if language else None
        if len(samples) > N_SAMPLES:
            result: dict[str, Any] = mlx_whisper.transcribe(
                samples,
                path_or_hf_repo=ModelHolder.model_path,
                language=code,
                condition_on_previous_text=False,
            )
            return Decoded(text=str(result["text"]), language=str(result["language"]))

        mel = log_mel_spectrogram(pad_or_trim(mx.array(samples)), n_mels=self._model.dims.n_mels)
        prompt = self.prompt() if self.prompt else None
        options = DecodingOptions(language=code, without_timestamps=True, prompt=prompt or None)
        decoded = decode(self._model, mel.astype(mx.float16), options)
        return Decoded(
            text=decoded.text,
            language=decoded.language,
            language_probs=decoded.language_probs,
            no_speech=decoded.no_speech_prob > NO_SPEECH_THRESHOLD
            and decoded.avg_logprob < LOGPROB_THRESHOLD,
        )


class FasterWhisper(_WhisperBase):
    def __init__(self, model: str, default_language: Language, device: str = "auto") -> None:
        super().__init__(default_language)
        self._model_name = model
        self._device = device
        self._model: Any = None

    def download(self) -> None:
        from faster_whisper.utils import download_model

        log.info("Lade Whisper-Modell %s von huggingface.co herunter ...", self._model_name)
        download_model(self._model_name)

    def _decode(self, samples: np.ndarray, language: Language | None) -> Decoded:
        if self._model is None:
            from faster_whisper import WhisperModel

            compute_type = "float16" if self._device == "cuda" else "int8"
            try:
                # Only the local copy; `flay setup` downloads it (see download()).
                self._model = WhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=compute_type,
                    local_files_only=True,
                )
            except FileNotFoundError:
                raise MissingModelError(f"Whisper-Modell {self._model_name}") from None
        # faster-whisper drops no-speech segments itself.
        segments, info = self._model.transcribe(
            samples,
            language=language.value if language else None,
            beam_size=1,
            condition_on_previous_text=False,
            initial_prompt=(self.prompt() if self.prompt else None) or None,
        )
        probs = dict(info.all_language_probs) if info.all_language_probs else None
        text = "".join(segment.text for segment in segments)
        return Decoded(text=text, language=info.language, language_probs=probs)
