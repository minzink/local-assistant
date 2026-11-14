"""Start Cassi at login: a LaunchAgent on macOS, the Startup folder on Windows."""

from __future__ import annotations

import os
import plistlib
import shutil
import sys
from pathlib import Path

LABEL = "de.cassi.assistant"


def program_command() -> list[str]:
    """The running program (``cassi``, or an adapter's command like ``flay``): its launcher
    if installed, else this Python with the module."""
    from cassi.app import current

    name = current().name
    launcher = shutil.which(name)
    return [launcher] if launcher else [sys.executable, "-m", name]


def entry_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    if sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return (
            appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "cassi.cmd"
        )
    raise RuntimeError("Autostart wird nur unter macOS und Windows unterstützt.")


def launch_agent(config: Path | None, log_file: Path) -> bytes:
    environment = {
        # launchd starts with a minimal PATH; Cassi needs tmux, claude and the IDE launcher.
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
        environment["CASSI_CONFIG"] = str(config.resolve())
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": [*program_command(), "run", "--background"],
            "EnvironmentVariables": environment,
            "RunAtLoad": True,
            "ProcessType": "Interactive",
            "StandardOutPath": str(log_file),
            "StandardErrorPath": str(log_file),
        }
    )


def startup_script(config: Path | None) -> str:
    command = " ".join(f'"{part}"' for part in [*program_command(), "run", "--background"])
    lines = ["@echo off"]
    if config is not None:
        lines.append(f'set "CASSI_CONFIG={config.resolve()}"')
    lines.append(f'start "Cassi" /min {command}')
    return "\r\n".join(lines) + "\r\n"


def enable(config: Path | None, data_dir: Path) -> Path:
    path = entry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        data_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(launch_agent(config, data_dir / "cassi.log"))
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
