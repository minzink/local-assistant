from pathlib import Path

import huggingface_hub
import numpy as np
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from flay.audio.base import AudioBuffer
from flay.core.language import Language
from flay.stt.whisper import Decoded, _WhisperBase, cached_model_path

AUDIO = AudioBuffer(pcm=bytes(32_000), sample_rate=16_000)


class ScriptedWhisper(_WhisperBase):
    def __init__(self, *results: Decoded) -> None:
        super().__init__(Language.DE)
        self._results = list(results)
        self.calls: list[Language | None] = []

    def _decode(self, samples: np.ndarray, language: Language | None) -> Decoded:
        self.calls.append(language)
        return self._results.pop(0)


async def test_detected_language_is_used_directly() -> None:
    stt = ScriptedWhisper(Decoded(" Hello", "en", {"en": 0.9, "de": 0.1}))

    transcript = await stt.transcribe(AUDIO)

    assert (transcript.text, transcript.language) == ("Hello", Language.EN)
    assert stt.calls == [None]


async def test_unsupported_language_is_decoded_again() -> None:
    stt = ScriptedWhisper(
        Decoded("Hallo", "nl", {"nl": 0.5, "de": 0.4, "en": 0.1}),
        Decoded(" Hallo", "de"),
    )

    transcript = await stt.transcribe(AUDIO)

    assert transcript.language is Language.DE
    assert stt.calls == [None, Language.DE]


async def test_hint_skips_detection() -> None:
    stt = ScriptedWhisper(Decoded("Hi", "en"))

    transcript = await stt.transcribe(AUDIO, Language.EN)

    assert transcript.language is Language.EN
    assert stt.calls == [Language.EN]


async def test_no_speech_returns_empty_text() -> None:
    stt = ScriptedWhisper(Decoded("Untertitel der Amara.org-Community", "de", no_speech=True))

    assert (await stt.transcribe(AUDIO)).text == ""


async def test_supported_detection_without_probs_is_accepted() -> None:
    stt = ScriptedWhisper(Decoded("Hello", "en"))

    transcript = await stt.transcribe(AUDIO)

    assert transcript.language is Language.EN
    assert stt.calls == [None]


async def test_unsupported_detection_without_probs_uses_default() -> None:
    stt = ScriptedWhisper(Decoded("Hallo", "nl"), Decoded("Hallo", "de"))

    assert (await stt.transcribe(AUDIO)).language is Language.DE


async def test_punctuation_only_counts_as_silence() -> None:
    stt = ScriptedWhisper(Decoded(" ...", "en"))

    assert (await stt.transcribe(AUDIO)).text == ""


def test_cached_model_path_does_not_contact_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_snapshot_download(repo: str, local_files_only: bool = False) -> str:
        calls.append(local_files_only)
        return "/cache/model"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    assert cached_model_path("mlx-community/whisper-tiny") == "/cache/model"
    assert calls == [True]


def test_missing_model_is_downloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_snapshot_download(repo: str, local_files_only: bool = False) -> str:
        calls.append(local_files_only)
        if local_files_only:
            raise LocalEntryNotFoundError("not cached")
        return "/cache/model"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    assert cached_model_path("mlx-community/whisper-tiny", allow_download=True) == "/cache/model"
    assert calls == [True, False]


def test_missing_model_is_not_downloaded_while_running(monkeypatch: pytest.MonkeyPatch) -> None:
    from flay.offline import MissingModelError

    calls: list[bool] = []

    def fake_snapshot_download(repo: str, local_files_only: bool = False) -> str:
        calls.append(local_files_only)
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    with pytest.raises(MissingModelError, match="flay setup"):
        cached_model_path("mlx-community/whisper-tiny")
    assert calls == [True]


def test_local_directory_is_used_as_is(tmp_path: Path) -> None:
    assert cached_model_path(str(tmp_path)) == str(tmp_path)
