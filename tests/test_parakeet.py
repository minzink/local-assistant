import numpy as np
import pytest

from flay.audio.base import AudioBuffer
from flay.config import Settings
from flay.core.language import Language
from flay.core.platform import Accelerator, HardwareProfile
from flay.offline import MissingModelError
from flay.stt.factory import create_stt
from flay.stt.parakeet import ParakeetSTT

AUDIO = AudioBuffer(pcm=bytes(32_000), sample_rate=16_000)


class FakeModel:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.received: list[np.ndarray] = []

    def recognize(self, waveform: np.ndarray, sample_rate: int = 16_000) -> str:
        self.received.append(waveform)
        return self.texts.pop(0)


def stt_with(*texts: str) -> ParakeetSTT:
    stt = ParakeetSTT(Language.DE)
    stt._model = FakeModel(*texts)
    return stt


async def test_language_comes_from_the_text_and_sticks_for_short_replies() -> None:
    stt = stt_with("What is the capital of France?", "Yes.", "Wie spät ist es?")

    first = await stt.transcribe(AUDIO)
    short = await stt.transcribe(AUDIO)
    german = await stt.transcribe(AUDIO)

    assert (first.text, first.language) == ("What is the capital of France?", Language.EN)
    assert short.language is Language.EN  # "Yes." alone says nothing; keep English
    assert german.language is Language.DE


async def test_hint_and_empty_results() -> None:
    stt = stt_with("Hello there", " . ")

    hinted = await stt.transcribe(AUDIO, Language.DE)
    empty = await stt.transcribe(AUDIO)

    assert hinted.language is Language.DE
    assert empty.text == ""


async def test_audio_is_passed_as_float32() -> None:
    stt = stt_with("Hallo")
    await stt.transcribe(AUDIO)

    received = stt._model.received[0]
    assert received.dtype == np.float32 and len(received) == 16_000


async def test_missing_model_points_to_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnx_asr

    def fail(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(onnx_asr, "load_model", fail)

    with pytest.raises(MissingModelError, match="flay setup"):
        await ParakeetSTT(Language.DE).transcribe(AUDIO)


def test_factory_selects_parakeet() -> None:
    settings = Settings.model_validate({"stt": {"backend": "parakeet"}})
    hardware = HardwareProfile("Darwin", "arm64", Accelerator.APPLE_SILICON)

    assert isinstance(create_stt(settings, hardware), ParakeetSTT)
