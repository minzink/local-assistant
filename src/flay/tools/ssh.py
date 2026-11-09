"""`ssh_connect`: open a new terminal window with ``ssh <host>`` for a server of the ssh config.

Only hosts named in the user's ssh config (``~/.ssh/config`` and its ``Include`` files) can
be connected to; the spoken name is matched against them like a project name. Flay passes
nothing but the host alias to ssh – user, port, keys and jump hosts come from the config.
Nothing is run on the server: the user gets an interactive session in a terminal window.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech, spoken
from flay.tools.phpstorm import match_names

log = logging.getLogger(__name__)

Terminal = Literal["auto", "terminal", "iterm", "windows-terminal", "console"]
# Host aliases Flay hands to ssh: plain names only, never an option ("-o…") or a pattern.
_SAFE_ALIAS = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._@:+-]*$")
# Include depth: configs including each other must not loop forever.
MAX_INCLUDE_DEPTH = 8


@dataclass(frozen=True)
class SshHost:
    alias: str
    # The HostName behind the alias, if the config names one ("Verbinde dich mit db.example.com").
    hostname: str | None = None


def default_config() -> Path:
    return Path.home() / ".ssh" / "config"


def read_ssh_hosts(path: Path | None = None) -> list[SshHost]:
    """The hosts of an ssh config, in order; wildcard patterns ("Host *") are left out."""
    hosts: dict[str, SshHost] = {}
    _read(path or default_config(), hosts, 0)
    return list(hosts.values())


def _read(path: Path, hosts: dict[str, SshHost], depth: int) -> None:
    try:
        lines = path.read_text("utf-8", errors="replace").splitlines()
    except OSError:
        return
    current: list[str] = []
    for line in lines:
        words = _split(line)
        if not words:
            continue
        keyword, values = words[0].lower(), words[1:]
        if keyword == "include" and depth < MAX_INCLUDE_DEPTH:
            for pattern in values:
                for included in _include_paths(pattern, path.parent):
                    _read(included, hosts, depth + 1)
        elif keyword == "host":
            current = [alias for alias in values if _SAFE_ALIAS.match(alias)]
            for alias in current:
                hosts.setdefault(alias, SshHost(alias))
        elif keyword == "match":
            current = []
        elif keyword == "hostname" and values:
            for alias in current:
                if hosts[alias].hostname is None:
                    hosts[alias] = SshHost(alias, values[0])


def _split(line: str) -> list[str]:
    """``Host a b`` or ``HostName=x`` – ssh_config allows both; comments start with #."""
    line = line.split("#", 1)[0].strip()
    found = re.match(r"(\S+?)(?:\s*=\s*|\s+)(.*)$", line)
    if found is None:
        return [line] if line else []
    keyword, rest = found.groups()
    try:
        return [keyword, *shlex.split(rest)]
    except ValueError:  # unbalanced quotes
        return [keyword, *rest.split()]


def _include_paths(pattern: str, base: Path) -> list[Path]:
    expanded = Path(pattern).expanduser()
    if not expanded.is_absolute():
        # Relative includes are relative to ~/.ssh (as ssh does for the user config).
        expanded = base / expanded
    return [Path(found) for found in sorted(glob.glob(str(expanded)))]


def terminal_command(alias: str, terminal: Terminal = "auto") -> list[str]:
    """The command that opens a new terminal window running ``ssh <alias>``."""
    if not _SAFE_ALIAS.match(alias):
        raise ValueError(f"unsafe ssh host alias: {alias!r}")
    if terminal == "auto":
        terminal = _default_terminal()
    if terminal == "terminal":
        return [
            "osascript",
            "-e", 'tell application "Terminal"',
            "-e", "activate",
            "-e", f'do script "ssh {alias}"',
            "-e", "end tell",
        ]  # fmt: skip
    if terminal == "iterm":
        return [
            "osascript",
            "-e", 'tell application "iTerm"',
            "-e", "activate",
            "-e", f'create window with default profile command "ssh {alias}"',
            "-e", "end tell",
        ]  # fmt: skip
    if terminal == "windows-terminal":
        return ["wt", "new-tab", "--title", alias, "ssh", alias]
    # A console window of its own (Windows) or the desktop's terminal emulator (Linux).
    if sys.platform == "win32":
        return ["ssh", alias]
    return ["x-terminal-emulator", "-e", "ssh", alias]


def _default_terminal() -> Terminal:
    if sys.platform == "darwin":
        return "terminal"
    if sys.platform == "win32" and shutil.which("wt"):
        return "windows-terminal"
    return "console"


async def launch(command: list[str]) -> None:
    """Start the terminal and leave it running; never through a shell."""
    if sys.platform == "win32" and command[0] == "ssh":
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(command, creationflags=flags)  # noqa: S603 – fixed argv, no shell
        return
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    if command[0] == "osascript":
        # osascript returns once the window is open; anything else keeps running.
        _, error = await process.communicate()
        if process.returncode:
            raise RuntimeError(error.decode(errors="replace").strip() or "osascript failed")


class SshConnectTool:
    def __init__(
        self,
        hosts: Callable[[], list[SshHost]] | None = None,
        terminal: Terminal = "auto",
        start: Callable[[list[str]], Any] = launch,
    ) -> None:
        # Read anew for each request: the config may change while Flay runs.
        self._hosts = hosts or read_ssh_hosts
        self._terminal = terminal
        self._start = start

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ssh_connect",
            description=(
                "Open a new terminal window with an SSH session to a server from the user's "
                'ssh config ("verbinde dich mit dem Server …", "SSH auf …"). Use the '
                "server name exactly as the user said it; it is matched against the ssh "
                "config. Runs no command on the server."
            ),
            parameters={
                "type": "object",
                "properties": {"host": {"type": "string", "description": "Server name"}},
                "required": ["host"],
            },
        )

    def find(self, name: str) -> list[SshHost]:
        """Hosts the spoken name fits: by alias, else by the HostName behind it."""
        hosts = self._hosts()
        by_alias = {h.alias: h for h in hosts}
        found = match_names(name, list(by_alias))
        if not found:
            by_hostname = {h.hostname: h for h in hosts if h.hostname}
            return [by_hostname[n] for n in match_names(name, list(by_hostname))]
        return [by_alias[alias] for alias in found]

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        name = str(arguments.get("host", "")).strip()
        if not name:
            return ToolResult(
                "No server named; ask which one.",
                ok=False,
                say=speech("Mit welchem Server?", "Which server?"),
            )
        matches = self.find(name)
        if not matches:
            return ToolResult(
                f"No host in the ssh config matches '{name}'.",
                ok=False,
                action=f"SSH: kein Server „{name}“ in der ssh-Config",
                say=speech(
                    f"Einen Server {name} finde ich in deiner ssh-Config nicht.",
                    f"I can't find a server {name} in your ssh config.",
                ),
            )
        if len(matches) > 1:
            return _which_one(name, [h.alias for h in matches])
        host = matches[0]
        try:
            await self._start(terminal_command(host.alias, self._terminal))
        except Exception as exc:
            log.warning("SSH-Terminal für %s ließ sich nicht öffnen: %s", host.alias, exc)
            return ToolResult(
                f"Could not open a terminal for '{host.alias}': {exc}",
                ok=False,
                action=f"SSH: Terminal für „{host.alias}“ nicht geöffnet",
                say=speech("Das Terminal ließ sich nicht öffnen.", "I couldn't open the terminal."),
            )
        return ToolResult(
            f"Opened a terminal window with 'ssh {host.alias}'.",
            action=f"SSH: Terminal mit „ssh {host.alias}“ geöffnet",
            say=speech(
                f"Ich verbinde dich mit {spoken(host.alias)}.",
                f"Connecting you to {spoken(host.alias)}.",
            ),
        )


# Read out at most this many servers when a name fits several.
MAX_LISTED = 3


def _which_one(name: str, aliases: list[str]) -> ToolResult:
    text = f"Several hosts match '{name}': {', '.join(aliases)}. Ask which one."
    if len(aliases) > MAX_LISTED:
        return ToolResult(
            text,
            ok=False,
            say=speech(
                f"Zu {name} passen {len(aliases)} Server. Welchen genau?",
                f"{len(aliases)} servers match {name}. Which one exactly?",
            ),
        )
    options = [spoken(alias) for alias in aliases]
    de = ", ".join(options[:-1]) + f" oder {options[-1]}"
    en = ", ".join(options[:-1]) + f" or {options[-1]}"
    return ToolResult(text, ok=False, say=speech(f"Meinst du {de}?", f"Do you mean {en}?"))
