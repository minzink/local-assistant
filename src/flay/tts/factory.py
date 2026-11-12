from __future__ import annotations

from flay.config import Settings
from flay.core.platform import Accelerator, HardwareProfile
from flay.tts.piper import PiperTTS
from flay.tts.qwen3 import Engine, Qwen3TTS


def resolve_backend(backend: str, hardware: HardwareProfile) -> str:
    if backend != "auto":
        return backend
    # Only MLX streams Qwen3-TTS fast enough (first sound ~0.2 s on an M2 Pro). The PyTorch
    # version renders whole sentences: on an RTX 4070 Ti 4.5 s for "Es ist 18:19 Uhr.", and
    # it competes with the LLM for VRAM, which slowed both to a crawl (90 s for two
    # sentences). Piper is fast on any CPU and leaves the GPU to the LLM.
    return "qwen3" if hardware.accelerator is Accelerator.APPLE_SILICON else "piper"


def create_tts(
    settings: Settings, hardware: HardwareProfile, backend: str | None = None
) -> PiperTTS | Qwen3TTS:
    tts = settings.tts
    if resolve_backend(backend or tts.backend, hardware) == "qwen3":
        engine: Engine = "mlx" if hardware.accelerator is Accelerator.APPLE_SILICON else "torch"
        return Qwen3TTS(tts.voice, settings.data_dir, engine)
    return PiperTTS(tts.voices, settings.data_dir / "voices", tts.voice.pitch, tts.voice.speed)
