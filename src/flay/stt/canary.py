"""NVIDIA Canary-1B-v2 (CC BY 4.0) via onnx-asr, on the CPU like Parakeet.

Canary does not detect the language; it is told which one it hears. The default language
is assumed first – always, not the last one used: Canary translates what it is told to
expect, so one sentence taken for English would turn all later German ones into English.
If the text reads as the other language, the utterance is decoded again in that one.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from flay.audio.base import AudioBuffer
from flay.core.language import Language
from flay.core.text_language import detect_text_language
from flay.offline import MissingModelError
from flay.stt.base import Transcript
from flay.stt.whisper import WHISPER_SAMPLE_RATE, to_float32

log = logging.getLogger(__name__)

MODEL = "nemo-canary-1b-v2"
QUANTIZATION = "int8"
PROVIDERS = ["CPUExecutionProvider"]


class CanarySTT:
    def __init__(self, default_language: Language) -> None:
        self._default = default_language
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flay-stt")
        self._model: Any = None

    def download(self) -> None:
        import onnx_asr

        log.info("Canary (%s): prüfe lokale Kopie, lade ggf. von huggingface.co ...", MODEL)
        onnx_asr.load_model(MODEL, quantization=QUANTIZATION, providers=PROVIDERS)

    def _load(self) -> Any:
        if self._model is None:
            import onnx_asr

            try:
                self._model = onnx_asr.load_model(
                    MODEL, quantization=QUANTIZATION, providers=PROVIDERS
                )
            except Exception as exc:
                raise MissingModelError(f"Spracherkennung Canary ({exc})") from exc
        return self._model

    async def warm_up(self) -> None:
        silence = AudioBuffer(pcm=bytes(WHISPER_SAMPLE_RATE * 2), sample_rate=WHISPER_SAMPLE_RATE)
        await self.transcribe(silence)

    async def transcribe(
        self, audio: AudioBuffer, language_hint: Language | None = None
    ) -> Transcript:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._transcribe, audio, language_hint)

    def _recognize(self, samples: np.ndarray, language: Language) -> str:
        text = self._load().recognize(
            samples, sample_rate=WHISPER_SAMPLE_RATE, language=language.value
        )
        return str(text).strip()

    def _transcribe(self, audio: AudioBuffer, language_hint: Language | None) -> Transcript:
        samples = to_float32(audio).astype(np.float32)
        language = language_hint or self._default
        text = self._recognize(samples, language)
        if language_hint is None and text:
            heard = detect_text_language(text, fallback=language)
            if heard is not language:
                language, text = heard, self._recognize(samples, heard)
        if not any(char.isalnum() for char in text):
            text = ""
        return Transcript(text=text, language=language)
