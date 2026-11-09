"""Piper TTS: fast local voices, one ONNX model per language.

Voices live in ``<data_dir>/voices``; ``flay setup`` downloads them. A model trained on
several languages (like Flay's own Ramona voice) may be named for each of them: it is
loaded once and speaks every language with that language's espeak pronunciation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import VOICE_PATTERN, download_voice

from flay.audio.base import AudioBuffer
from flay.core.language import Language
from flay.offline import MissingModelError
from flay.tts.shaping import pitch_shift
from flay.tts.voice import resample

log = logging.getLogger(__name__)

ESPEAK_VOICES = {Language.DE: "de", Language.EN: "en-us"}

_DONE = object()


class PiperTTS:
    def __init__(
        self,
        voices: Mapping[Language, str],
        voice_dir: Path,
        pitch: float = 0.0,
        speed: float = 1.0,
    ) -> None:
        self._voice_names = dict(voices)
        self._voice_dir = voice_dir
        self._pitch = pitch
        self._config = SynthesisConfig(length_scale=1 / speed)
        self._voices: dict[Language, PiperVoice] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flay-tts")

    def download(self) -> None:
        """Fetch missing voices (only used by `flay setup`)."""
        self._voice_dir.mkdir(parents=True, exist_ok=True)
        for name in dict.fromkeys(self._voice_names.values()):
            if (self._voice_dir / f"{name}.onnx").exists():
                continue
            if not VOICE_PATTERN.match(name):
                # Not a published Piper voice, e.g. one trained with tools/voice-training.
                log.warning("Piper-Stimme %s fehlt: %s.onnx und .onnx.json nach %s kopieren",
                            name, name, self._voice_dir)  # fmt: skip
            else:
                log.info("Lade Piper-Stimme %s von huggingface.co herunter ...", name)
                download_voice(name, self._voice_dir)

    def load(self) -> None:
        """Load all configured voices from disk."""
        loaded: dict[str, PiperVoice] = {}
        for language, name in self._voice_names.items():
            if name not in loaded:
                model_path = self._voice_dir / f"{name}.onnx"
                if not model_path.exists():
                    raise MissingModelError(f"Piper-Stimme {name}")
                loaded[name] = PiperVoice.load(model_path)
            voice = loaded[name]
            if list(self._voice_names.values()).count(name) > 1:
                # One model for several languages: pronounce each as that language.
                config = replace(voice.config, espeak_voice=ESPEAK_VOICES[language])
                voice = replace(voice, config=config)
            self._voices[language] = voice

    @property
    def sample_rate(self) -> int:
        """Output rate: the highest of the voices; lower ones are resampled to it."""
        return max(int(voice.config.sample_rate) for voice in self._voices.values())

    async def synthesize(self, text: str, language: Language) -> AsyncIterator[AudioBuffer]:
        if language not in self._voices:
            await asyncio.get_running_loop().run_in_executor(self._executor, self.load)
        voice = self._voices[language]
        chunks = self._chunks(voice, text, self._config, self._pitch, self.sample_rate)
        loop = asyncio.get_running_loop()
        while (
            chunk := await loop.run_in_executor(self._executor, next, chunks, _DONE)
        ) is not _DONE:
            assert isinstance(chunk, AudioBuffer)
            yield chunk

    @staticmethod
    def _chunks(
        voice: PiperVoice, text: str, config: SynthesisConfig, pitch: float, rate: int
    ) -> Iterator[AudioBuffer]:
        # Piper yields one chunk per sentence.
        for chunk in voice.synthesize(text, config):
            pcm = chunk.audio_int16_bytes
            if pitch or chunk.sample_rate != rate:
                # Same rate for all languages: switching would reopen the audio stream.
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                if pitch:
                    samples = pitch_shift(samples, pitch, chunk.sample_rate)
                samples = resample(samples, chunk.sample_rate, rate)
                pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            yield AudioBuffer(pcm=pcm, sample_rate=rate)
