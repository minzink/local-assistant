from __future__ import annotations

from flay.config import Settings
from flay.core.platform import Accelerator, HardwareProfile
from flay.stt.canary import CanarySTT
from flay.stt.parakeet import ParakeetSTT
from flay.stt.voxtral import VoxtralSTT
from flay.stt.whisper import FasterWhisper, MlxWhisper


def resolve_backend(backend: str, hardware: HardwareProfile) -> str:
    if backend != "auto":
        return backend
    return "mlx-whisper" if hardware.accelerator is Accelerator.APPLE_SILICON else "faster-whisper"


def create_stt(
    settings: Settings, hardware: HardwareProfile
) -> MlxWhisper | FasterWhisper | ParakeetSTT | CanarySTT | VoxtralSTT:
    backend = resolve_backend(settings.stt.backend, hardware)
    default_language = settings.language.default
    if backend == "parakeet":
        return ParakeetSTT(default_language)
    if backend == "canary":
        return CanarySTT(default_language)
    if backend == "voxtral":
        return VoxtralSTT(default_language)
    if backend == "mlx-whisper":
        return MlxWhisper(settings.stt.model, default_language)
    device = "cuda" if hardware.accelerator is Accelerator.CUDA else "cpu"
    return FasterWhisper(settings.stt.model, default_language, device=device)
