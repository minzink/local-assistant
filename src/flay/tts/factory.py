from __future__ import annotations

from flay.config import Settings
from flay.core.platform import Accelerator, HardwareProfile
from flay.tts.piper import PiperTTS
from flay.tts.qwen3 import Engine, Qwen3TTS


def resolve_backend(backend: str, hardware: HardwareProfile) -> str:
    if backend != "auto":
        return backend
    # Qwen3-TTS needs a GPU to answer quickly; Piper is fast on any CPU.
    return "piper" if hardware.accelerator is Accelerator.CPU else "qwen3"


def create_tts(
    settings: Settings, hardware: HardwareProfile, backend: str | None = None
) -> PiperTTS | Qwen3TTS:
    tts = settings.tts
    if resolve_backend(backend or tts.backend, hardware) == "qwen3":
        engine: Engine = "mlx" if hardware.accelerator is Accelerator.APPLE_SILICON else "torch"
        return Qwen3TTS(tts.voice, settings.data_dir, engine)
    return PiperTTS(tts.voices, settings.data_dir / "voices", tts.voice.pitch, tts.voice.speed)
