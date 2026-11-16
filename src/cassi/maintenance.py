"""Install, repair, update and uninstall – what the management UI (``cassi manage``) and the
install scripts do.

- **Command:** a small launcher ``~/.local/bin/cassi`` (Windows ``cassi.cmd``) that runs the
  program from its project with ``uv run``. Updates of the checkout apply at once; there is
  no second copy to keep in step.
- **Repair** runs ``scripts/setup.sh`` / ``setup.ps1`` again: missing programs, Ollama, Python
  packages, models, the command. It is safe to run any time.
- **Update** pulls the checkouts (fast-forward only, never over local changes), then repairs.
- **Uninstall** removes what Cassi put on the computer outside its folder; the folder itself
  is deleted by the user (a running program cannot remove itself on Windows).
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cassi.app import App

if TYPE_CHECKING:
    from cassi.adapters import Adapter
    from cassi.config import Settings

log = logging.getLogger(__name__)

Output = Callable[[str], None]

# Marks launchers written by Cassi, so uninstalling never removes someone else's file.
SHIM_MARK = "Created by Cassi's installer"


def bin_dir() -> Path:
    """Where launchers go: uv's tool folder, which uv puts on the PATH."""
    override = os.environ.get("CASSI_BIN_DIR")
    return Path(override) if override else Path.home() / ".local" / "bin"


def shim_path(app: App, folder: Path | None = None) -> Path:
    folder = folder or bin_dir()
    return folder / (f"{app.name}.cmd" if sys.platform == "win32" else app.name)


def shim_text(app: App, uv: str) -> str:
    if app.project is None:
        raise ValueError("no project folder")
    if sys.platform == "win32":
        return (
            f"@echo off\r\nrem {SHIM_MARK} ({app.project}).\r\n"
            f'"{uv}" run --quiet --project "{app.project}" {app.name} %*\r\n'
        )
    return (
        f"#!/bin/sh\n# {SHIM_MARK} ({app.project}).\n"
        f'exec "{uv}" run --quiet --project "{app.project}" {app.name} "$@"\n'
    )


def install_command(app: App, out: Output, folder: Path | None = None) -> bool:
    uv = shutil.which("uv")
    if uv is None or app.project is None:
        out("✗ Befehl nicht eingerichtet: uv oder Projektordner nicht gefunden")
        return False
    path = shim_path(app, folder)
    if path.exists() and SHIM_MARK not in _read(path):
        out(f"✗ {path} gehört einem anderen Programm – nicht überschrieben")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(shim_text(app, uv), encoding="utf-8")
    if sys.platform != "win32":
        path.chmod(0o755)
    out(f"✓ Befehl {app.name}: {path}")
    on_path = {Path(p).resolve() for p in os.environ.get("PATH", "").split(os.pathsep) if p}
    default_folder = folder is None and "CASSI_BIN_DIR" not in os.environ
    if path.parent.resolve() not in on_path and default_folder:
        # Adds the folder to the shell's PATH (shell profile / Windows user PATH).
        result = subprocess.run([uv, "tool", "update-shell"], capture_output=True, text=True)
        out(
            f"  {path.parent} zum PATH hinzugefügt – in einem neuen Terminal verfügbar"
            if result.returncode == 0
            else f"  ! {path.parent} ist nicht im PATH – dort eintragen"
        )
    return True


def remove_command(app: App, out: Output, folder: Path | None = None) -> None:
    path = shim_path(app, folder)
    if not path.exists():
        return
    if SHIM_MARK not in _read(path):
        out(f"· {path} gehört einem anderen Programm – bleibt")
        return
    path.unlink()
    out(f"✓ Befehl {app.name} entfernt ({path})")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# Git ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepoState:
    root: Path
    commit: str
    branch: str
    # Changed tracked files: updating would overwrite them, so it is refused.
    changes: int
    # Commits upstream that are not here yet, as of the last fetch; None without upstream.
    behind: int | None


def quiet_environment() -> dict[str, str]:
    """No prompts: the management UI runs in the browser, nobody would see ssh or git ask
    for a password – they would wait forever. Without one they fail with a message."""
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"}
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    return environment


def _git(root: Path, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        stdin=subprocess.DEVNULL,
        env=quiet_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def repo_state(root: Path) -> RepoState | None:
    try:
        commit = _git(root, "rev-parse", "--short", "HEAD").stdout.strip()
        if not commit:
            return None
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        status = _git(root, "status", "--porcelain", "--untracked-files=no").stdout
        behind = _git(root, "rev-list", "--count", "HEAD..@{upstream}")
    except (OSError, subprocess.SubprocessError):
        return None
    count = int(behind.stdout.strip()) if behind.returncode == 0 else None
    return RepoState(root, commit, branch, len(status.splitlines()), count)


def check_updates(app: App, out: Output) -> bool:
    """Fetch every checkout and say what is new; False if one could not be fetched."""
    ok = True
    for root in app.repositories():
        try:
            fetched = _git(root, "fetch", "--quiet")
        except (OSError, subprocess.SubprocessError) as exc:
            out(f"✗ {root.name}: {exc}")
            ok = False
            continue
        if fetched.returncode != 0:
            out(f"✗ {root.name}: {fetched.stderr.strip() or 'git fetch fehlgeschlagen'}")
            ok = False
            continue
        state = repo_state(root)
        if state is None or state.behind is None:
            out(f"· {root.name}: kein Upstream-Zweig")
        elif state.behind == 0:
            out(f"✓ {root.name}: aktuell ({state.commit})")
        else:
            out(f"↓ {root.name}: {state.behind} neue Änderung(en) verfügbar")
            log_lines = _git(root, "log", "--oneline", "HEAD..@{upstream}").stdout
            for line in log_lines.splitlines()[:10]:
                out(f"    {line}")
    return ok


def pull(app: App, out: Output) -> bool:
    """Fast-forward every checkout; refuses checkouts with local changes."""
    roots = app.repositories()
    if not roots:
        out("✗ Kein Git-Checkout gefunden – Update nur über eine neue Installation")
        return False
    for root in roots:
        state = repo_state(root)
        if state is not None and state.changes:
            out(
                f"✗ {root.name}: {state.changes} geänderte Datei(en) – erst committen oder "
                "verwerfen, dann updaten"
            )
            return False
    for root in roots:
        out(f"> git pull ({root.name})")
        if not run_process(["git", "-C", str(root), "pull", "--ff-only"], out):
            out(f"✗ {root.name}: Update fehlgeschlagen (siehe oben)")
            return False
        if (root / ".gitattributes").is_file() and "lfs" in _read(root / ".gitattributes"):
            run_process(["git", "-C", str(root), "lfs", "pull"], out)
    return True


# Processes ------------------------------------------------------------------------------


def run_process(command: Sequence[str], out: Output, cwd: Path | None = None) -> bool:
    """Run ``command``, passing every line of its output to ``out``; True on exit code 0."""
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=quiet_environment(),
        )
    except OSError as exc:
        out(f"✗ {command[0]} lässt sich nicht starten: {exc}")
        return False
    assert process.stdout is not None
    for line in process.stdout:
        out(line.rstrip("\n"))
    return process.wait() == 0


def setup_command(app: App, extra: Sequence[str] = ()) -> list[str] | None:
    """The setup script for this program (repair), or None without a source checkout."""
    if app.source is None or app.project is None:
        return None
    if sys.platform == "win32":
        script = app.source / "scripts" / "setup.ps1"
        return [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
            "-Project", str(app.project), "-Command", app.name, *extra,
        ]  # fmt: skip
    script = app.source / "scripts" / "setup.sh"
    return ["bash", str(script), "--project", str(app.project), "--command", app.name, *extra]


def in_own_window(command: Sequence[str]) -> bool:
    """Windows: start ``command`` in a new console window and return at once.

    Updating Python packages replaces files the running program has loaded, which Windows
    does not allow – so repair and update run outside it, and the management UI closes.
    """
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    try:
        subprocess.Popen(["cmd", "/c", *command, "&", "pause"], creationflags=flags)
    except OSError:
        return False
    return True


def open_folder(path: Path) -> bool:
    if not path.exists():
        return False
    if sys.platform == "darwin":
        command = ["open", str(path)]
    elif sys.platform == "win32":
        command = ["explorer", str(path)]
    else:
        command = ["xdg-open", str(path)]
    try:
        subprocess.Popen(command)
    except OSError:
        return False
    return True


def folder_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):  # vanished or unreadable
                total += (Path(root) / name).stat().st_size
    return total


def human_size(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# Uninstall ------------------------------------------------------------------------------


def ollama_models(settings: Settings) -> list[str]:
    """Models Cassi downloaded or created in Ollama (some may be used by others too)."""
    from cassi.config import LLMSettings
    from cassi.llm import ollama

    names = []
    for model in (settings.llm.model, LLMSettings().model):
        names += [model, ollama.main_variant(model).name]
    for variant in (ollama.EMBEDDING, ollama.POINTER):
        names += [variant.name, variant.base]
    return list(dict.fromkeys(names))


def uninstall(
    app: App,
    settings: Settings,
    adapters: Sequence[Adapter],
    out: Output,
    data: bool = False,
    models: bool = False,
) -> bool:
    """Remove what Cassi put outside its folder. ``data``: the data folder (memory,
    recordings, voices); ``models``: Cassi's models in Ollama."""
    from cassi import autostart
    from cassi.llm import ollama

    ok = True
    try:
        if autostart.disable():
            out("✓ Autostart entfernt")
    except RuntimeError:
        pass
    remove_command(app, out)
    for adapter in adapters:
        try:
            for line in adapter.uninstall(settings, models):
                out(line)
        except Exception as exc:  # one adapter must not stop the rest
            out(f"✗ Adapter {adapter.name}: {exc}")
            ok = False
    if models:
        url = settings.llm.base_url
        if not ollama.is_ollama(url):
            out(f"✗ Ollama antwortet nicht ({url}) – Modelle nicht gelöscht")
            ok = False
        else:
            have = ollama.models(url)
            for name in ollama_models(settings):
                if name in have:
                    deleted = ollama.delete(url, name)
                    out(f"✓ Modell {name} gelöscht" if deleted else f"✗ {name} nicht gelöscht")
                    ok = ok and deleted
    if data and settings.data_dir.exists():
        try:
            shutil.rmtree(settings.data_dir)
            out(f"✓ Datenordner gelöscht ({settings.data_dir})")
        except OSError as exc:
            out(f"✗ Datenordner nicht vollständig gelöscht: {exc}")
            ok = False
    elif settings.data_dir.exists():
        out(f"· Datenordner bleibt: {settings.data_dir}")
    if app.project is not None:
        out(f"· Zum Schluss den Programmordner löschen: {app.project}")
    out(
        "· Bleiben (auch von anderen Programmen genutzt): uv, Ollama, Git, "
        "Modelle im Hugging-Face-Cache (~/.cache/huggingface)"
    )
    return ok


# The running assistant -------------------------------------------------------------------


def port_owner(port: int) -> int | None:
    """The process listening on ``port`` on this computer, or None."""
    try:
        if sys.platform == "win32":
            output = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True, timeout=10
            ).stdout
            for line in output.splitlines():
                parts = line.split()
                if (
                    len(parts) >= 5
                    and parts[3].upper() in ("LISTENING", "ABHÖREN")
                    and parts[1].endswith(f":{port}")
                ):
                    return int(parts[4])
            return None
        output = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    pids = [int(line) for line in output.split() if line.isdigit()]
    return pids[0] if pids else None


def process_command(pid: int) -> str:
    """The command line of a process ("" if unknown)."""
    try:
        if sys.platform == "win32":
            query = f'(Get-CimInstance Win32_Process -Filter "ProcessId={pid}").CommandLine'
            command = ["powershell", "-NoProfile", "-Command", query]
        else:
            command = ["ps", "-p", str(pid), "-o", "command="]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def stop_process(pid: int, timeout: float = 8.0) -> bool:
    """End a process: politely first, then for sure. True once it is gone."""
    import signal
    import time

    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        except OSError:
            return False
        return process_gone(pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if process_gone(pid):
            return True
        time.sleep(0.2)
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)
    time.sleep(0.5)
    return process_gone(pid)


def process_gone(pid: int) -> bool:
    """Whether the process ``pid`` has ended."""
    if sys.platform == "win32":
        return not process_command(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    # A zombie (ended, not yet collected by its parent) counts as gone.
    try:
        state = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return not state or state.startswith("Z")


# Management UIs ------------------------------------------------------------------------


def _processes() -> list[tuple[int, int, str]]:
    """(pid, parent pid, command line) of the user's processes."""
    try:
        if sys.platform == "win32":
            query = (
                "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,"
                "CommandLine | ConvertTo-Json -Compress"
            )
            output = subprocess.run(
                ["powershell", "-NoProfile", "-Command", query],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
            import json

            rows = json.loads(output or "[]")
            rows = rows if isinstance(rows, list) else [rows]
            return [
                (int(r["ProcessId"]), int(r["ParentProcessId"] or 0), r.get("CommandLine") or "")
                for r in rows
            ]
        output = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,command="], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    found = []
    for line in output.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            found.append((int(parts[0]), int(parts[1]), parts[2]))
    return found


_PYTHON = re.compile(r"^python(\d+(\.\d+)?)?(\.exe)?$", re.IGNORECASE)


def is_manage_command(command: str, name: str) -> bool:
    """Whether ``command`` is this program's management UI itself: ``<name> manage`` or
    Python running ``<name>`` / ``-m <name>`` with the subcommand ``manage`` – strictly, so
    no other program (a shell whose line merely mentions it) is ever taken for one. The uv
    wrapper does not count: its Python child is the one to end, uv follows it."""
    words = command.replace("\\", "/").split()
    if not words:
        return False
    launchers = {name, f"{name}.exe", f"{name}.cmd"}
    program = words[0].rsplit("/", 1)[-1]
    if program.lower() in launchers:
        rest = words[1:]
    elif _PYTHON.match(program):
        if len(words) > 2 and words[1] == "-m" and words[2] == name:
            rest = words[3:]
        elif len(words) > 1 and words[1].rsplit("/", 1)[-1].lower() in launchers:
            rest = words[2:]
        else:
            return False
    else:
        return False
    index = 0
    while index < len(rest) and rest[index] in ("-c", "--config", "-v", "--verbose"):
        index += 2 if rest[index] in ("-c", "--config") else 1
    return index < len(rest) and rest[index] == "manage"


def other_manage_processes(name: str) -> list[tuple[int, str]]:
    """Running management UIs of this program besides this process."""
    me, parent = os.getpid(), os.getppid()
    return [
        (pid, command)
        for pid, ppid, command in _processes()
        if pid not in (me, parent) and ppid != me and is_manage_command(command, name)
    ]


def ask_for_folder(prompt: str, start: Path | None = None) -> Path | None:
    """Show the system's folder dialog on this computer (the management UI runs here); the
    chosen folder, or None if cancelled or no dialog is available."""
    try:
        if sys.platform == "darwin":
            script = f'POSIX path of (choose folder with prompt "{prompt}"'
            if start is not None and start.is_dir():
                script += f' default location POSIX file "{start}"'
            result = subprocess.run(
                ["osascript", "-e", script + ")"], capture_output=True, text=True, timeout=600
            )
        elif sys.platform == "win32":
            selected = f"$d.SelectedPath = '{start}'; " if start is not None else ""
            script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
                f"$d.Description = '{prompt}'; {selected}"
                "if ($d.ShowDialog() -eq 'OK') { $d.SelectedPath }"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-STA", "-Command", script],
                capture_output=True,
                text=True,
                timeout=600,
            )
        else:
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory", f"--title={prompt}"],
                capture_output=True,
                text=True,
                timeout=600,
            )
    except (OSError, subprocess.SubprocessError):
        return None
    chosen = result.stdout.strip()
    return Path(chosen) if result.returncode == 0 and chosen else None
