from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from flay.config import Settings, TTSSettings, VoiceSettings
from flay.core.language import Language
from flay.core.platform import Accelerator, HardwareProfile
from flay.offline import MissingModelError
from flay.tts import qwen3
from flay.tts.factory import create_tts, resolve_backend
from flay.tts.piper import PiperTTS
from flay.tts.qwen3 import SAMPLE_RATE, Qwen3TTS
from flay.tts.shaping import pitch_shift, time_stretch
from flay.tts.voice import (
    BUNDLED,
    BUNDLED_DIR,
    DEFAULT,
    VoiceError,
    own_recording,
    read_reference,
    resolve,
)

RATE = 24_000


def sine(frequency: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def peak_frequency(samples: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.fft.rfftfreq(len(samples), 1 / RATE)[np.argmax(spectrum)])


def write_wav(path: Path, samples: np.ndarray, rate: int, channels: int = 1) -> None:
    pcm = (np.repeat(samples, channels) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(pcm)


def test_time_stretch_changes_duration_not_pitch() -> None:
    faster = time_stretch(sine(220), 1.25, RATE)
    assert len(faster) == pytest.approx(RATE / 1.25, abs=2)
    assert peak_frequency(faster) == pytest.approx(220, abs=5)


def test_pitch_shift_changes_pitch_not_duration() -> None:
    octave_up = pitch_shift(sine(220), 12, RATE)
    assert len(octave_up) == pytest.approx(RATE, rel=0.01)
    assert peak_frequency(octave_up) == pytest.approx(440, abs=8)
    unchanged = sine(220)
    assert pitch_shift(unchanged, 0, RATE) is unchanged


def test_voice_settings_are_bounded() -> None:
    with pytest.raises(ValidationError):
        VoiceSettings(pitch=20)
    with pytest.raises(ValidationError):
        VoiceSettings(speed=0.1)


def test_bundled_voices_have_transcripts() -> None:
    for name in BUNDLED:
        reference = read_reference(BUNDLED_DIR / f"{name}.wav", RATE)
        assert reference.text
        assert 5 < len(reference.samples) / RATE < 15


def test_auto_voice_is_the_default_without_own_recording(tmp_path: Path) -> None:
    assert resolve("auto", tmp_path) == dict.fromkeys(Language, BUNDLED_DIR / f"{DEFAULT}.wav")
    assert resolve("ramona", tmp_path)[Language.EN] == BUNDLED_DIR / "ramona.wav"
    with pytest.raises(VoiceError):
        resolve("own", tmp_path)


def test_own_recording_speaks_both_languages(tmp_path: Path) -> None:
    german = own_recording(tmp_path, Language.DE)
    german.parent.mkdir()
    german.touch()
    assert resolve("auto", tmp_path) == {Language.DE: german, Language.EN: german}
    english = own_recording(tmp_path, Language.EN)
    english.touch()
    assert resolve("own", tmp_path) == {Language.DE: german, Language.EN: english}


def test_reference_is_read_as_mono_at_the_model_rate(tmp_path: Path) -> None:
    path = tmp_path / "voice.wav"
    write_wav(path, sine(200, 0.5), 48_000, channels=2)
    with pytest.raises(VoiceError):
        read_reference(path, RATE)  # no transcript yet
    path.with_suffix(".txt").write_text("Hallo Welt.\n")
    reference = read_reference(path, RATE)
    assert reference.text == "Hallo Welt."
    assert len(reference.samples) == pytest.approx(RATE * 0.25, abs=2)


def profile(accelerator: Accelerator) -> HardwareProfile:
    return HardwareProfile(system="test", machine="test", accelerator=accelerator)


def test_qwen3_where_a_gpu_makes_it_fast() -> None:
    assert resolve_backend("auto", profile(Accelerator.APPLE_SILICON)) == "qwen3"
    assert resolve_backend("auto", profile(Accelerator.CUDA)) == "qwen3"
    assert resolve_backend("auto", profile(Accelerator.CPU)) == "piper"
    assert resolve_backend("piper", profile(Accelerator.CUDA)) == "piper"


def test_factory_picks_engine_and_passes_voice(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, tts=TTSSettings(voice=VoiceSettings(speed=1.25)))
    assert isinstance(create_tts(settings, profile(Accelerator.APPLE_SILICON)), Qwen3TTS)
    piper = create_tts(settings, profile(Accelerator.CPU))
    assert isinstance(piper, PiperTTS)
    assert piper._config.length_scale == pytest.approx(0.8)


def test_one_piper_model_speaks_both_languages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from piper import PiperConfig, PiperVoice
    from piper.config import PhonemeType

    loads: list[Path] = []

    def load(path: Path) -> PiperVoice:
        loads.append(path)
        config = PiperConfig(
            num_symbols=1, num_speakers=1, sample_rate=22_050, espeak_voice="de",
            phoneme_id_map={}, phoneme_type=PhonemeType.ESPEAK,
        )  # fmt: skip
        return PiperVoice(session=None, config=config)

    monkeypatch.setattr(PiperVoice, "load", staticmethod(load))
    (tmp_path / "ramona.onnx").touch()
    tts = PiperTTS(dict.fromkeys(Language, "ramona"), tmp_path)
    tts.load()
    assert loads == [tmp_path / "ramona.onnx"]
    assert tts._voices[Language.DE].config.espeak_voice == "de"
    assert tts._voices[Language.EN].config.espeak_voice == "en-us"


class FakeModel:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, Language]] = []

    def chunks(self, text: str, language: Language) -> Iterator[np.ndarray]:
        self.spoken.append((text, language))
        yield np.full(100, 0.5, dtype=np.float32)
        yield np.full(50, 2.0, dtype=np.float32)  # clipped


async def test_qwen3_streams_pcm_chunks(tmp_path: Path) -> None:
    tts = Qwen3TTS(VoiceSettings(), tmp_path, "mlx")
    model = FakeModel()
    tts._model = model
    chunks = [chunk async for chunk in tts.synthesize("Hallo.", Language.DE)]
    assert model.spoken == [("Hallo.", Language.DE)]
    assert [c.sample_rate for c in chunks] == [SAMPLE_RATE, SAMPLE_RATE]
    assert len(chunks[0].pcm) == 200
    assert np.frombuffer(chunks[1].pcm, dtype="<i2").max() == 32767


def test_qwen3_without_model_asks_for_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(repo: str) -> str:
        raise MissingModelError(f"Stimme Qwen3-TTS ({repo})")

    monkeypatch.setattr(qwen3, "_cached", missing)
    with pytest.raises(MissingModelError):
        Qwen3TTS(VoiceSettings(), tmp_path, "mlx").load()


class SlowModel:
    def __init__(self) -> None:
        self.events: list[str] = []

    def chunks(self, text: str, language: Language) -> Iterator[np.ndarray]:
        self.events.append(f"start {text}")
        try:
            for _ in range(3):
                yield np.zeros(10, dtype=np.float32)
        finally:
            self.events.append(f"end {text}")


async def test_qwen3_speaks_one_sentence_at_a_time(tmp_path: Path) -> None:
    import asyncio

    tts = Qwen3TTS(VoiceSettings(), tmp_path, "mlx")
    model = SlowModel()
    tts._model = model

    async def speak(text: str, stop_after: int | None = None) -> None:
        stream = tts.synthesize(text, Language.DE)
        count = 0
        async for _ in stream:
            count += 1
            if count == stop_after:
                break  # interrupted, like a reply the user talks over
        await stream.aclose()

    await asyncio.gather(speak("a", stop_after=1), speak("b"))
    assert model.events == ["start a", "end a", "start b", "end b"]


def test_setup_does_not_download_a_trained_piper_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import flay.tts.piper as piper_module

    downloads: list[str] = []
    monkeypatch.setattr(piper_module, "download_voice", lambda name, _: downloads.append(name))
    voices = {Language.DE: "flay-ramona-medium", Language.EN: "flay-ramona-medium"}
    PiperTTS(voices, tmp_path).download()
    assert downloads == []
    assert caplog.text.count("flay-ramona-medium fehlt") == 1


async def test_piper_voices_of_different_rates_play_at_one(tmp_path: Path) -> None:
    from types import SimpleNamespace

    class Voice:
        def __init__(self, rate: int) -> None:
            self.config = SimpleNamespace(sample_rate=rate)

        def synthesize(self, text: str, config: object) -> Iterator[SimpleNamespace]:
            pcm = np.zeros(self.config.sample_rate, dtype="<i2").tobytes()  # one second
            yield SimpleNamespace(audio_int16_bytes=pcm, sample_rate=self.config.sample_rate)

    tts = PiperTTS({}, tmp_path)
    tts._voices = {Language.DE: Voice(16_000), Language.EN: Voice(22_050)}  # type: ignore[dict-item]

    assert tts.sample_rate == 22_050
    for language in Language:
        (chunk,) = [chunk async for chunk in tts.synthesize("Hallo.", language)]
        assert chunk.sample_rate == 22_050
        assert chunk.duration == pytest.approx(1.0)
