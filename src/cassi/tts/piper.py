"""Piper TTS: fast local voices, one ONNX model per language.

Voices live in ``<data_dir>/voices``; ``cassi setup`` downloads them. A model trained on
several languages (like Cassi's own Ramona voice) may be named for each of them: it is
loaded once and speaks every language with that language's espeak pronunciation.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import AsyncIterator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig
from piper.download_voices import VOICE_PATTERN, download_voice
from scipy.signal import resample_poly

from cassi.audio.base import AudioBuffer
from cassi.core.language import Language
from cassi.offline import MissingModelError

log = logging.getLogger(__name__)

ESPEAK_VOICES = {Language.DE: "de", Language.EN: "en-us"}
# Voices trained for Cassi come with the repository (Git LFS, see voices/README.md).
REPO_VOICES = Path(__file__).resolve().parents[3] / "voices"
# Without Git LFS a checkout has a pointer of ~130 bytes instead of the model.
MIN_MODEL_BYTES = 1_000_000
# Published voices for when a trained one is missing (both female, like Ramona).
FALLBACK_VOICES = {Language.DE: "de_DE-kerstin-low", Language.EN: "en_US-kristin-medium"}

_DONE = object()


class PiperTTS:
    def __init__(
        self,
        voices: Mapping[Language, str],
        voice_dir: Path,
        pitch: float = 0.0,
        speed: float = 1.0,
        repo_voices: Path | None = REPO_VOICES,
    ) -> None:
        self._voice_names = dict(voices)
        self._voice_dir = voice_dir
        self._repo_voices = repo_voices
        # Pitch without artifacts: Piper speaks ``factor`` times slower (its own, clean
        # duration control), and playing that ``factor`` times faster raises the pitch by
        # the factor at the original pace. Reshaping every sentence afterwards (WSOLA, as
        # for Qwen3-TTS's reference) left Flay's +1 semitone audibly rough and warbly.
        self._pitch = Fraction(2 ** (pitch / 12)).limit_denominator(200)
        self._speed = speed
        self._voices: dict[Language, PiperVoice] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cassi-tts")

    def _install_from_repo(self, name: str) -> bool:
        """Copy a voice that comes with the repository into the voice folder."""
        model = self._repo_voices / f"{name}.onnx" if self._repo_voices else None
        config = model.with_suffix(".onnx.json") if model else None
        if model is None or config is None or not config.is_file() or not model.is_file():
            return False
        if model.stat().st_size < MIN_MODEL_BYTES:
            log.warning("%s ist nur ein Git-LFS-Zeiger – 'git lfs pull' ausführen", model)
            return False
        self._voice_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config, self._voice_dir / config.name)
        shutil.copyfile(model, self._voice_dir / model.name)
        log.info("Piper-Stimme %s aus dem Repository übernommen", name)
        return True

    def download(self) -> None:
        """Fetch missing voices (only used by `cassi setup`), and the fallbacks."""
        self._voice_dir.mkdir(parents=True, exist_ok=True)
        wanted = [*self._voice_names.values(), *FALLBACK_VOICES.values()]
        for name in dict.fromkeys(wanted):
            if (self._voice_dir / f"{name}.onnx").exists() or self._install_from_repo(name):
                continue
            if not VOICE_PATTERN.match(name):
                # Not a published Piper voice, e.g. one trained with tools/voice-training.
                log.warning("Piper-Stimme %s fehlt: %s.onnx und .onnx.json nach %s kopieren",
                            name, name, self._voice_dir)  # fmt: skip
            else:
                log.info("Lade Piper-Stimme %s von huggingface.co herunter ...", name)
                download_voice(name, self._voice_dir)

    def _available(self, language: Language, name: str) -> str:
        """``name`` if its model is there (copied from the repository if need be), else the
        fallback voice for ``language`` if that is there."""
        if (self._voice_dir / f"{name}.onnx").exists() or self._install_from_repo(name):
            return name
        fallback = FALLBACK_VOICES.get(language)
        if fallback and fallback != name and (self._voice_dir / f"{fallback}.onnx").exists():
            log.warning("Piper-Stimme %s fehlt – spreche %s mit %s", name, language, fallback)
            return fallback
        raise MissingModelError(f"Piper-Stimme {name}")

    def load(self) -> None:
        """Load all configured voices from disk."""
        self._voice_names = {
            language: self._available(language, name)
            for language, name in self._voice_names.items()
        }
        loaded: dict[str, PiperVoice] = {}
        for language, name in self._voice_names.items():
            if name not in loaded:
                model_path = self._voice_dir / f"{name}.onnx"
                loaded[name] = PiperVoice.load(model_path)
            voice = loaded[name]
            if list(self._voice_names.values()).count(name) > 1:
                # One model for several languages: pronounce each as that language.
                config = replace(voice.config, espeak_voice=ESPEAK_VOICES[language])
                voice = replace(voice, config=config)
            self._voices[language] = voice

    def set_speed(self, speed: float) -> None:
        """Speak faster (>1) or slower (<1) from the next sentence on."""
        self._speed = speed

    @property
    def _config(self) -> SynthesisConfig:
        return SynthesisConfig(length_scale=float(self._pitch) / self._speed)

    def voices(self) -> list[str]:
        """Voices installed in the voice folder (and Cassi's own)."""
        found = {p.name.removesuffix(".onnx") for p in self._voice_dir.glob("*.onnx")}
        if self._repo_voices is not None:
            found |= {
                p.name.removesuffix(".onnx")
                for p in self._repo_voices.glob("*.onnx")
                if p.stat().st_size >= MIN_MODEL_BYTES
            }
        return sorted(found)

    def set_voice(self, name: str) -> None:
        """One voice for all languages, loaded in the background (~0.3 s)."""
        self._voice_names = dict.fromkeys(self._voice_names, name)
        self._executor.submit(self.load)

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
        voice: PiperVoice, text: str, config: SynthesisConfig, pitch: Fraction, rate: int
    ) -> Iterator[AudioBuffer]:
        # Piper yields one chunk per sentence.
        for chunk in voice.synthesize(text, config):
            pcm = chunk.audio_int16_bytes
            # Played at ``pitch`` × its rate (see __init__), converted to the output rate:
            # the same for all languages, as switching would reopen the audio stream.
            ratio = Fraction(rate) / (chunk.sample_rate * pitch)
            if ratio != 1:
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                samples = resample_poly(samples, ratio.numerator, ratio.denominator)
                pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            yield AudioBuffer(pcm=pcm, sample_rate=rate)
