import os
from pathlib import Path

import pytest

from flay.audio.vad import SileroVAD
from flay.core.language import Language
from flay.offline import MissingModelError, enforce_offline, is_local_url
from flay.tts.piper import PiperTTS


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://localhost:11434/v1", True),
        ("http://127.0.0.1:1234/v1", True),
        ("http://[::1]:8080/v1", True),
        ("http://192.168.1.20:11434/v1", False),
        ("https://api.example.com/v1", False),
    ],
)
def test_is_local_url(url: str, local: bool) -> None:
    assert is_local_url(url) is local


def test_enforce_offline_sets_hugging_face_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv first, so monkeypatch restores both variables after the test.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")

    enforce_offline(allow_downloads=False)

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_missing_voice_and_vad_are_not_downloaded(tmp_path: Path) -> None:
    tts = PiperTTS({Language.DE: "de_DE-thorsten-medium"}, tmp_path / "voices")
    vad = SileroVAD(tmp_path / "models")

    with pytest.raises(MissingModelError, match="flay setup"):
        tts.load()
    with pytest.raises(MissingModelError, match="flay setup"):
        vad.load()
    assert not (tmp_path / "voices").exists() or not any((tmp_path / "voices").iterdir())
    assert not (tmp_path / "models").exists()
