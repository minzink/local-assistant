import plistlib
import sys
from pathlib import Path

import pytest

from cassi import autostart


def test_launch_agent_runs_flay_in_background(tmp_path: Path) -> None:
    plist = plistlib.loads(autostart.launch_agent(tmp_path / "cassi.yaml", tmp_path / "cassi.log"))

    assert plist["Label"] == autostart.LABEL
    assert plist["ProgramArguments"][-2:] == ["run", "--background"]
    assert plist["RunAtLoad"] is True
    assert plist["EnvironmentVariables"]["CASSI_CONFIG"] == str((tmp_path / "cassi.yaml").resolve())
    assert "/opt/homebrew/bin" in plist["EnvironmentVariables"]["PATH"]


def test_startup_script_for_windows(tmp_path: Path) -> None:
    script = autostart.startup_script(None)

    assert script.startswith("@echo off\r\n")
    assert '"run" "--background"' in script


@pytest.mark.skipif(sys.platform not in ("darwin", "win32"), reason="macOS/Windows only")
def test_enable_and_disable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entry = tmp_path / "LaunchAgents" / "entry"
    monkeypatch.setattr(autostart, "entry_path", lambda: entry)

    autostart.enable(None, tmp_path / "data")
    assert autostart.is_enabled()
    assert autostart.disable()
    assert not autostart.is_enabled()
    assert not autostart.disable()
