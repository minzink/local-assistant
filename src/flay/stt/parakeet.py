"""NVIDIA Parakeet TDT 0.6B v3 (CC BY 4.0) via onnx-asr: fast speech recognition on CPU,
on macOS and Windows alike.

Parakeet transcribes 25 European languages but does not report which one it heard, so
German/English is decided from the text (falling back to the last language used).
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

MODEL = "nemo-parakeet-tdt-0.6b-v3"
QUANTIZATION = "int8"  # 670 MB instead of 2.4 GB, practically the same accuracy
# Plain CPU beats CoreML on Apple Silicon for this model (M2 Pro: 70 vs. 168 ms per
# sentence, 1 s vs. 23 s to load) and is the same path on Windows.
PROVIDERS = ["CPUExecutionProvider"]


class ParakeetSTT:
    def __init__(self, default_language: Language) -> None:
        self._language = default_language
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flay-stt")
        self._model: Any = None

    def download(self) -> None:
        import onnx_asr

        # onnx-asr uses the local copy if there is one and downloads only what is missing.
        log.info("Parakeet (%s): prüfe lokale Kopie, lade ggf. von huggingface.co ...", MODEL)
        onnx_asr.load_model(MODEL, quantization=QUANTIZATION, providers=PROVIDERS)

    def _load(self) -> Any:
        if self._model is None:
            import onnx_asr

            try:
                self._model = onnx_asr.load_model(
                    MODEL, quantization=QUANTIZATION, providers=PROVIDERS
                )
            except Exception as exc:  # offline and not downloaded yet
                raise MissingModelError(f"Spracherkennung Parakeet ({exc})") from exc
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
        samples = to_float32(audio)
        text = str(
            self._load().recognize(samples.astype(np.float32), sample_rate=WHISPER_SAMPLE_RATE)
        )
        text = text.strip()
        if not any(char.isalnum() for char in text):
            text = ""
        language = language_hint or (
            detect_text_language(text, fallback=self._language) if text else self._language
        )
        if text:
            # Short replies like "Ja." carry no language signal; keep the conversation's.
            self._language = language
        return Transcript(text=text, language=language)
