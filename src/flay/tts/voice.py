"""The voice Flay speaks with: a short recording that Qwen3-TTS clones for both languages.

Bundled voices (all free to use for any purpose, see ``voices/SOURCES.md``) or the user's
own recording (``flay voice record``), stored per language in ``<data_dir>/voice``. A
recording in one language speaks both; with a recording per language each language uses
its own, so neither has an accent from the other.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from flay.core.language import Language

BUNDLED_DIR = Path(__file__).parent / "voices"
BUNDLED = ("hokuspokus", "ramona", "thorsten")
DEFAULT = "ramona"
OWN = "own"


class VoiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reference:
    samples: np.ndarray  # mono float32 in [-1, 1]
    sample_rate: int
    text: str  # what is said in the recording


def own_recording(data_dir: Path, language: Language) -> Path:
    return data_dir / "voice" / f"{language.value}.wav"


def resolve(name: str, data_dir: Path) -> dict[Language, Path]:
    """The recording each language clones."""
    if name == "auto":
        own = any(own_recording(data_dir, language).exists() for language in Language)
        name = OWN if own else DEFAULT
    if name == OWN:
        found = {
            language: path
            for language in Language
            if (path := own_recording(data_dir, language)).exists()
        }
        if not found:
            raise VoiceError("Keine eigene Aufnahme – erst 'flay voice record' ausführen.")
        either = next(iter(found.values()))
        return {language: found.get(language, either) for language in Language}
    path = BUNDLED_DIR / f"{name}.wav" if name in BUNDLED else Path(name).expanduser()
    return dict.fromkeys(Language, path)


def read_reference(path: Path, sample_rate: int) -> Reference:
    """A 16-bit WAV (any rate, mono or stereo) and its transcript from the .txt beside it."""
    transcript = path.with_suffix(".txt")
    if not path.exists() or not transcript.exists():
        raise VoiceError(f"Stimme {path} oder ihre Abschrift {transcript.name} fehlt.")
    with wave.open(str(path), "rb") as file:
        if file.getsampwidth() != 2:
            raise VoiceError(f"{path}: nur 16-Bit-WAV wird unterstützt.")
        rate, channels = file.getframerate(), file.getnchannels()
        pcm = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2")
    samples = (pcm.reshape(-1, channels).mean(axis=1) / 32768.0).astype(np.float32)
    text = transcript.read_text("utf-8").strip()
    return Reference(resample(samples, rate, sample_rate), sample_rate, text)


def resample(samples: np.ndarray, rate: int, target_rate: int) -> np.ndarray:
    if rate == target_rate:
        return samples
    ratio = Fraction(target_rate, rate)
    resampled = resample_poly(samples, ratio.numerator, ratio.denominator)
    return np.asarray(resampled, dtype=np.float32)
