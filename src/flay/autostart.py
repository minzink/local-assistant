"""Start Flay at login: a LaunchAgent on macOS, the Startup folder on Windows."""

from __future__ import annotations

import os
import plistlib
import shutil
import sys
from pathlib import Path

LABEL = "de.flay.assistant"


def flay_command() -> list[str]:
    flay = shutil.which("flay")
    return [flay] if flay else [sys.executable, "-m", "flay"]


def entry_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    if sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return (
            appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "flay.cmd"
        )
    raise RuntimeError("Autostart wird nur unter macOS und Windows unterstützt.")


def launch_agent(config: Path | None, log_file: Path) -> bytes:
    environment = {
        # launchd starts with a minimal PATH; Flay needs tmux, claude and the IDE launcher.
        "PATH": ":".join(
            [
                str(Path.home() / ".local" / "bin"),
                "/opt/homebrew/bin",
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
            ]
        )
    }
    if config is not None:
        environment["FLAY_CONFIG"] = str(config.resolve())
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": [*flay_command(), "run", "--background"],
            "EnvironmentVariables": environment,
            "RunAtLoad": True,
            "ProcessType": "Interactive",
            "StandardOutPath": str(log_file),
            "StandardErrorPath": str(log_file),
        }
    )


def startup_script(config: Path | None) -> str:
    command = " ".join(f'"{part}"' for part in [*flay_command(), "run", "--background"])
    lines = ["@echo off"]
    if config is not None:
        lines.append(f'set "FLAY_CONFIG={config.resolve()}"')
    lines.append(f'start "Flay" /min {command}')
    return "\r\n".join(lines) + "\r\n"


def enable(config: Path | None, data_dir: Path) -> Path:
    path = entry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        data_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(launch_agent(config, data_dir / "flay.log"))
    else:
        path.write_text(startup_script(config), encoding="utf-8")
    return path


def disable() -> bool:
    path = entry_path()
    if not path.exists():
        return False
    path.unlink()
    return True


def is_enabled() -> bool:
    return entry_path().exists()
