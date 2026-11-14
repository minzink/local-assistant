"""Switch to a newer version while running: an old Cassi must not keep running – and keep the
chat port – once a newer one is installed.

While ``cassi run`` runs, ``UpdateWatcher`` compares every minute what is installed with what
was installed at the start. Newer means:

- the version or the git commit of the checkouts changed (``cassi update``, ``git pull``,
  ``uv sync``),
- the command (``~/.local/bin/cassi``) now starts another installation,
- this installation was deleted.

Cassi then says so in the chat, waits until it has nothing to do, and starts the new version
in its place (``restart_command``). The chat in the browser reconnects and shows what
changed. Without anything to start, it ends, so the port is free for the new one.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from cassi.app import App
from cassi.maintenance import SHIM_MARK, quiet_environment

log = logging.getLogger(__name__)

# Set for the new process, so its chat says what it replaced.
SWITCHED_ENV = "CASSI_SWITCHED_FROM"


@dataclass(frozen=True, slots=True)
class Installed:
    # Compared to notice a change: version and commit of every checkout.
    fingerprint: str
    # For people: "Flay 0.2.0 (37bbbb0)".
    label: str


def _commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=quiet_environment(),
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def installed(app: App) -> Installed:
    commits = [c for c in (_commit(root) for root in app.repositories()) if c]
    version = app.version
    label = f"{app.title} {version}" + (f" ({', '.join(commits)})" if commits else "")
    return Installed("|".join([version, *commits]), label)


def launcher_target(name: str) -> Path | None:
    """The project the installed command starts (only Cassi's own launchers count)."""
    launcher = shutil.which(name)
    if launcher is None:
        return None
    try:
        text = Path(launcher).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if SHIM_MARK not in text:
        return None
    match = re.search(r'--project "([^"]+)"', text)
    return Path(match.group(1)) if match else None


def newer(app: App, started: Installed) -> str | None:
    """Why a newer version should take over, or None if this one is current."""
    target = launcher_target(app.name)
    if (
        target is not None
        and app.project is not None
        and target.exists()
        and target.resolve() != app.project.resolve()
    ):
        return f"{app.name} startet jetzt die Installation in {target}"
    if app.project is not None and not app.project.exists():
        return "diese Installation wurde entfernt"
    if not Path(sys.executable).exists():
        return "diese Installation wurde entfernt"
    now = installed(app)
    if now.fingerprint != started.fingerprint:
        return f"{started.label} → {now.label}"
    return None


def restart_arguments(argv: Sequence[str]) -> list[str]:
    """The same command again, without opening another browser tab (the chat reconnects)."""
    arguments = list(argv)
    if "run" not in arguments:  # `cassi` alone runs, too; only a running Cassi switches
        arguments.append("run")
    if "--background" not in arguments:
        arguments.append("--background")
    return arguments


def restart_command(app: App, argv: Sequence[str]) -> list[str] | None:
    """What starts the new version: the installed command, else this Python if still there."""
    arguments = restart_arguments(argv)
    if launcher_target(app.name) is not None:
        launcher = shutil.which(app.name)
        if launcher is not None:
            return [launcher, *arguments]
    if Path(sys.executable).exists() and (app.project is None or app.project.exists()):
        return [sys.executable, "-m", app.name, *arguments]
    return None


def switch(command: Sequence[str], reason: str) -> None:
    """Replace this process with ``command`` (Windows: start it and let this one end)."""
    os.environ[SWITCHED_ENV] = reason
    log.info("Starte neue Version: %s", " ".join(command))
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(list(command), creationflags=flags)
        return
    os.execv(command[0], list(command))


class UpdateWatcher:
    """Checks every ``interval`` seconds; calls ``on_newer(reason)`` once, then stops."""

    def __init__(
        self,
        app: App,
        on_newer: Callable[[str], None],
        interval: float = 60.0,
        check: Callable[[App, Installed], str | None] = newer,
    ) -> None:
        self._app = app
        self._on_newer = on_newer
        self._interval = interval
        self._check = check

    async def run(self) -> None:
        started = await asyncio.to_thread(installed, self._app)
        log.debug("Installiert: %s", started.label)
        while True:
            await asyncio.sleep(self._interval)
            try:
                reason = await asyncio.to_thread(self._check, self._app, started)
            except Exception:  # a failed check must not stop Cassi
                log.debug("Versionsprüfung fehlgeschlagen", exc_info=True)
                continue
            if reason:
                log.info("Neuere Version: %s", reason)
                self._on_newer(reason)
                return
