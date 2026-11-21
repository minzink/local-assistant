import json
import shutil
import sys
from pathlib import Path

import pytest

from cassi import maintenance, voice_training
from cassi.app import App
from cassi.config import Settings
from cassi.manage import Action, Tab
from cassi.tools.preferences import Preferences
from cassi.voice_training import epoch_reached, read_status, voice_tab


def make_tab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str = "win32"
) -> tuple[Tab, Settings, Path]:
    source = tmp_path / "src"
    (source / "tools" / "voice-training").mkdir(parents=True)
    (source / "tools" / "voice-training" / "run.ps1").touch()
    settings = Settings(data_dir=tmp_path / "data")
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(voice_training, "running", lambda: False)
    return voice_tab(App("flay", None, source), lambda: settings), settings, source


def action(tab: Tab, id_: str) -> Action:
    return next(a for a in tab.actions if a.id == id_)


def test_status_written_by_powershell_is_read(tmp_path: Path) -> None:
    status = {"step": "train", "quality": "high", "ok": False}
    (tmp_path / "status.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(status).encode())

    assert read_status(tmp_path) == status
    assert read_status(tmp_path / "missing") is None


def test_epoch_reached_from_checkpoints(tmp_path: Path) -> None:
    folder = tmp_path / "lightning_logs" / "version_0" / "checkpoints"
    folder.mkdir(parents=True)
    for name in ("epoch=9-step=100.ckpt", "epoch=219-step=2200.ckpt", "last.ckpt"):
        (folder / name).touch()

    assert epoch_reached(tmp_path) == 220
    assert epoch_reached(tmp_path / "none") is None


def test_start_launches_run_script_in_own_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab, settings, source = make_tab(tmp_path, monkeypatch)
    started: list[list[str]] = []

    def launch(command: list[str]) -> bool:
        started.append(command)
        return True

    monkeypatch.setattr(maintenance, "in_own_window", launch)

    assert action(tab, "voice-train").run(lambda line: None, {"high": True})

    command = started[0]
    assert command[command.index("-File") + 1] == str(source / "tools/voice-training/run.ps1")
    assert command[command.index("-Quality") + 1] == "high"
    assert command[command.index("-VoicesDir") + 1] == str(settings.downloads / "voices")


def test_start_refused_without_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tab, _, _ = make_tab(tmp_path, monkeypatch, platform="darwin")
    lines: list[str] = []
    monkeypatch.setattr(maintenance, "in_own_window", lambda c: pytest.fail())

    assert not action(tab, "voice-train").run(lines.append, {})
    assert "nur unter Windows" in lines[0]
    assert tab.sections()[0].rows[0].level == "fail"


def test_use_sets_the_trained_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tab, settings, _ = make_tab(tmp_path, monkeypatch)
    lines: list[str] = []

    assert not action(tab, "voice-use").run(lines.append, {"high": True})  # not trained yet

    voices = settings.downloads / "voices"
    voices.mkdir(parents=True)
    (voices / "cassi-ramona-high.onnx").touch()
    assert action(tab, "voice-use").run(lines.append, {"high": True})
    assert Preferences.load(settings.data_dir).voice == "cassi-ramona-high"
    installed = tab.sections()[-1].rows
    assert [(r.label, r.level) for r in installed] == [("cassi-ramona-high", "ok")]


def test_progress_while_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tab, _, source = make_tab(tmp_path, monkeypatch)
    work = source / "tools" / "voice-training" / "work"
    (work / "runs" / "checkpoints").mkdir(parents=True)
    (work / "runs" / "checkpoints" / "epoch=499-step=1.ckpt").touch()
    status = {
        "step": "train",
        "text": "3/5 Trainieren",
        "quality": "medium",
        "epochs": 2000,
        "started": "2026-11-21T10:00:00",
        "updated": "2026-11-21T12:00:00",
        "done": False,
    }
    (work / "status.json").write_text(json.dumps(status))
    monkeypatch.setattr(voice_training, "running", lambda: True)

    rows = {r.label: r.value for r in tab.sections()[1].rows}

    assert rows["Zustand"] == "läuft"
    assert rows["Epoche"] == "500 von 2000"
