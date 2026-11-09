"""Mistral Voxtral Mini 4B Realtime (Apache 2.0), 4-bit for MLX via mlx-audio (MIT).

macOS (Apple Silicon) only, and mlx-audio is not a dependency of Flay: it pins its own
mlx and transformers versions. For a test: ``uv run --with "mlx-audio[stt]" flay eval-stt
--backends voxtral --download``. The language is read from the text, as for Parakeet.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from flay.audio.base import AudioBuffer
from flay.core.language import Language
from flay.core.text_language import detect_text_language
from flay.offline import MissingModelError
from flay.stt.base import Transcript
from flay.stt.whisper import WHISPER_SAMPLE_RATE

log = logging.getLogger(__name__)

MODEL = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
# How far the realtime model may look ahead (multiples of 80 ms); Mistral recommends 480.
DELAY_MS = 480


class VoxtralSTT:
    def __init__(self, default_language: Language, delay_ms: int = DELAY_MS) -> None:
        self._language = default_language
        self._delay_ms = delay_ms
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flay-stt")
        self._model: Any = None

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        log.info("Voxtral (%s): lade von huggingface.co ...", MODEL)
        snapshot_download(MODEL)

    def _load(self) -> Any:
        if self._model is None:
            try:
                from mlx_audio.stt.utils import load
            except ImportError as exc:
                raise MissingModelError(
                    'Voxtral braucht mlx-audio: uv run --with "mlx-audio[stt]" ...'
                ) from exc
            try:
                self._model = load(MODEL)
            except Exception as exc:
                raise MissingModelError(f"Spracherkennung Voxtral ({exc})") from exc
        return self._model

    async def warm_up(self) -> None:
        silence = AudioBuffer(pcm=bytes(WHISPER_SAMPLE_RATE * 2), sample_rate=WHISPER_SAMPLE_RATE)
        await self.transcribe(silence)

    async def transcribe(
        self, audio: AudioBuffer, language_hint: Language | None = None
    ) -> Transcript:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._transcribe, audio, language_hint)

    def _transcribe(self, audio: AudioBuffer, language_hint: Language | None) -> Transcript:
        from flay.stt_evaluation import save_wav

        model = self._load()
        # mlx-audio documents file input only; a temporary file is quick for short audio.
        with tempfile.TemporaryDirectory(prefix="flay-voxtral-") as directory:
            path = Path(directory) / "utterance.wav"
            save_wav(path, audio)
            result = model.generate(str(path), transcription_delay_ms=self._delay_ms)
        text = str(getattr(result, "text", result)).strip()
        if not any(char.isalnum() for char in text):
            text = ""
        language = language_hint or (
            detect_text_language(text, fallback=self._language) if text else self._language
        )
        if text:
            self._language = language
        return Transcript(text=text, language=language)
