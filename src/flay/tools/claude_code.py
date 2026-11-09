"""Hand instructions to Claude Code, the coding agent the user runs in PhpStorm's terminal.

Two transports:

- **tmux** (macOS/Linux): ``flay claude`` starts Claude Code inside a tmux session named
  after the project. Flay types into that session directly – no window focus, nothing can
  land in the editor – and can read Claude's screen to report its status.
- **clipboard** (everywhere, fallback): the instruction is copied, PhpStorm is brought to
  the front and the user pastes it into Claude.

If no session runs for the project, Flay can start Claude itself in a background tmux
session; ``flay claude`` in PhpStorm's terminal then attaches to it.

Enter is never pressed: the user reviews the instruction and submits it.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from flay.core.language import Language
from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech, spoken
from flay.tools.phpstorm import Project, ambiguous, match_projects, recent_projects

if TYPE_CHECKING:
    from flay.tools.phpstorm_mcp import PhpStormMcp

log = logging.getLogger(__name__)

SESSION_PREFIX = "flay-claude-"
MAC_BUNDLE_ID = "com.jetbrains.PhpStorm"


def session_name(project_dir: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", project_dir.name.lower()).strip("-") or "project"
    return SESSION_PREFIX + slug


def single_line(text: str) -> str:
    # Enter submits in Claude Code; typed newlines would send the instruction early.
    return " ".join(text.split())


@dataclass(frozen=True, slots=True)
class Session:
    name: str
    last_activity: int
    # Shown in a terminal right now (e.g. PhpStorm's), i.e. where the user is looking.
    attached: bool = False

    @property
    def project(self) -> str:
        return self.name.removeprefix(SESSION_PREFIX)


class Tmux:
    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("tmux")

    @property
    def available(self) -> bool:
        return self.executable is not None

    async def _run(self, *args: str) -> str:
        assert self.executable is not None
        process = await asyncio.create_subprocess_exec(
            self.executable,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr.decode(errors="replace").strip() or "tmux failed")
        return stdout.decode(errors="replace")

    async def sessions(self) -> list[Session]:
        if not self.available:
            return []
        try:
            output = await self._run(
                "list-sessions",
                "-F",
                "#{session_name}\t#{session_activity}\t#{session_attached}",
            )
        except RuntimeError:  # no server running = no sessions
            return []
        sessions = []
        for line in output.splitlines():
            name, activity, attached = (line.split("\t") + ["0", "0"])[:3]
            if name.startswith(SESSION_PREFIX):
                sessions.append(Session(name, int(activity or 0), attached not in ("", "0")))
        return sessions

    async def start(self, session: str, cwd: Path, command: list[str]) -> None:
        """Start ``command`` detached, styled like ``flay claude`` (no status bar, mouse)."""
        await self._run(
            "new-session", "-d", "-s", session, "-x", "200", "-y", "50", "-c", str(cwd), *command
        )
        await self._run("set-option", "-t", session, "status", "off")
        await self._run("set-option", "-t", session, "mouse", "on")

    async def type_text(self, session: str, text: str) -> None:
        # -l sends the text literally; no Enter key is sent.
        await self._run("send-keys", "-t", session, "-l", "--", text)

    async def press(self, session: str, key: str, count: int = 1) -> None:
        await self._run("send-keys", "-t", session, "-N", str(count), key)

    async def screen(self, session: str, lines: int = 40) -> str:
        output = await self._run("capture-pane", "-p", "-t", session, "-S", f"-{lines}")
        return output.rstrip()


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def pick_session(sessions: list[Session], project: str | None) -> Session | None:
    """The session for the named project; without a name the one the user has open.

    "Open" means attached to a terminal (``flay claude`` in PhpStorm). Detached sessions
    are only used if nothing is open, preferring the most recently active one.
    """
    if project:
        matches = [s for s in sessions if _matches(s, project)]
        return max(matches, key=lambda s: s.last_activity, default=None)
    open_sessions = [s for s in sessions if s.attached] or sessions
    return max(open_sessions, key=lambda s: s.last_activity, default=None)


ClaudeState = Literal["ready", "trust", "timeout"]

# Claude asks once per folder whether to trust it; "No, exit" is preselected, so typing
# there must never happen.
_TRUST_PROMPT = re.compile(r"trust this folder|do you trust", re.IGNORECASE)
# Claude puts a no-break space (U+00A0) after the prompt sign; \s matches both kinds.
_INPUT_LINE = re.compile(r"^\s*❯(?:\s|$)", re.MULTILINE)


_BORDER = re.compile(r"^\s*─{10,}")
_CONTEXT_CHIP = re.compile(r"^\[⧉[^\]]*\]\s*")  # IDE selection, e.g. "[⧉ In README.md]"
_PLACEHOLDER = re.compile(r'^Try ".*"$')


def read_input(screen: str) -> str | None:
    """Text in Claude's input box ("" if empty), or None if no input box is visible.

    Wrapped lines lose their line-break spaces, so compare results with ``_compact``.
    """
    lines = screen.splitlines()
    starts = [i for i, line in enumerate(lines) if _INPUT_LINE.match(line)]
    if not starts:
        return None
    start = starts[-1]
    parts = [lines[start].split("❯", 1)[1]]
    for line in lines[start + 1 :]:
        if _BORDER.match(line):
            break
        parts.append(line.strip())
    text = _CONTEXT_CHIP.sub("", " ".join(p for p in parts if p).strip())
    return "" if _PLACEHOLDER.match(text) else text


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def claude_state(screen: str) -> ClaudeState | None:
    if _TRUST_PROMPT.search(screen):
        return "trust"
    if _INPUT_LINE.search(screen):
        return "ready"
    return None


async def wait_for_claude(
    tmux: Tmux, session: str, timeout: float = 30.0, interval: float = 0.5
) -> ClaudeState:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        state = claude_state(await tmux.screen(session))
        if state is not None:
            return state
        await asyncio.sleep(interval)
    return "timeout"


def copy_to_clipboard(text: str) -> None:
    import pyperclip

    pyperclip.copy(text)


def bring_phpstorm_to_front() -> None:
    """Best effort; the clipboard works even if this fails."""
    if sys.platform == "darwin":
        script = f'tell application id "{MAC_BUNDLE_ID}" to activate'
        subprocess.run(["osascript", "-e", script], capture_output=True, check=False)
    elif sys.platform == "win32":
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined, unused-ignore]
        hwnd = user32.FindWindowW(None, None)
        while hwnd:
            length = user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if "PhpStorm" in buffer.value and user32.IsWindowVisible(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.SetForegroundWindow(hwnd)
                return
            hwnd = user32.GetWindow(hwnd, 2)  # GW_HWNDNEXT


_PROJECT_PARAMETER = {
    "type": "string",
    "description": "Project name, only if the user names one; otherwise omit",
}


class SendToClaudeTool:
    def __init__(
        self,
        transport: str = "auto",
        autostart: bool = True,
        tmux: Tmux | None = None,
        find: Callable[[str], list[Project]] = lambda name: match_projects(name, recent_projects()),
        claude_command: list[str] | None = None,
        ready_timeout: float = 30.0,
        phpstorm: PhpStormMcp | None = None,
        confirm_timeout: float = 40.0,
    ) -> None:
        self._transport = transport
        self._autostart = autostart
        self._tmux = tmux or Tmux()
        self._find = find
        claude = shutil.which("claude")
        self._claude_command = claude_command or ([claude] if claude else [])
        self._ready_timeout = ready_timeout
        self._phpstorm = phpstorm
        # What Flay itself typed into each session and is still unsent there.
        self._typed: dict[str, str] = {}
        # Time for PhpStorm's terminal and `flay claude` to come up.
        self._confirm_timeout = confirm_timeout

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="send_to_claude",
            description=(
                "Give an instruction to Claude Code, the coding agent the user runs in PhpStorm. "
                "Use it when the user wants Claude to do something in the code. Phrase a "
                "complete, precise instruction in the user's language. The instruction is only "
                "written into Claude's input; the user checks it and submits it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "Text for Claude Code"},
                    "project": _PROJECT_PARAMETER,
                },
                "required": ["instruction"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        text = single_line(str(arguments.get("instruction", "")))
        if not text:
            return ToolResult(
                "No instruction given.",
                ok=False,
                say=speech("Was soll ich Claude sagen?", "What should I tell Claude?"),
            )
        project = str(arguments.get("project") or "") or None
        if self._transport in ("auto", "tmux") and self._tmux.available:
            sessions = await self._tmux.sessions()
            session = pick_session(sessions, project)
            if session is not None:
                visible = session.attached
                if not visible and self._autostart and self._phpstorm is not None:
                    # Nobody would see a detached session: attach it in a PhpStorm terminal.
                    visible = await self._show_in_phpstorm(session)
                if refused := await self._write(session.name, text):
                    return refused
                if visible:
                    return ToolResult(
                        f"The instruction is written into Claude's input for project "
                        f"'{session.project}' but NOT sent. Tell the user it is waiting in "
                        f"Claude for project '{session.project}' to check and press Enter.",
                        action=(
                            f"Claude ({session.name}): Auftrag eingetippt, nicht abgeschickt – "
                            f"„{text}“"
                        ),
                        say=_typed_speech(session.project),
                    )
                return ToolResult(
                    f"The instruction is written into Claude's input for project "
                    f"'{session.project}', NOT sent, but that Claude runs in the background and "
                    "is not shown anywhere. Tell the user to run 'flay claude' in PhpStorm's "
                    f"terminal of project '{session.project}' to see it, then press Enter.",
                    action=(
                        f"Claude ({session.name}, im Hintergrund, nicht sichtbar): Auftrag "
                        f"eingetippt – mit 'flay claude' im Projekt anzeigen – „{text}“"
                    ),
                    say=speech(
                        f"Der Auftrag steht in Claude für {spoken(session.project)}, das läuft "
                        "aber im Hintergrund. Gib flay claude im Terminal des Projekts ein, um "
                        "es zu sehen, und drück dann Enter.",
                        f"The instruction is in Claude for {spoken(session.project)}, but that "
                        "Claude runs in the background. Run flay claude in the project's "
                        "terminal to see it, then press Enter.",
                    ),
                )
            if self._autostart and self._phpstorm is not None:
                opened = await self._open_in_phpstorm(project, text)
                if opened is not None:
                    return opened
            if self._autostart and self._claude_command:
                started = await self._start_and_type(project, text)
                if started is not None:
                    return started
        if self._transport == "tmux":
            return ToolResult(
                "No Claude session started with 'flay claude' is running.",
                ok=False,
                say=speech(
                    "Ich finde kein Claude, das mit flay claude gestartet wurde.",
                    "I can't find a Claude started with flay claude.",
                ),
            )
        return await self._via_clipboard(text)

    async def _open_in_phpstorm(self, project_name: str | None, text: str) -> ToolResult | None:
        """Start ``flay claude`` in a new PhpStorm terminal tab of an open project.

        Returns None if the project is not open in PhpStorm (or PhpStorm is unreachable),
        so the caller can fall back to a background session.
        """
        assert self._phpstorm is not None
        open_projects = await self._phpstorm.open_projects()
        if not open_projects:
            return None
        if project_name:
            candidates = [Project(p.name, p) for p in open_projects]
            matches = match_projects(project_name, candidates)
            if len(matches) != 1:
                return None
            project = matches[0].path
        elif len(open_projects) == 1:
            project = open_projects[0]
        else:
            names = ", ".join(p.name for p in open_projects)
            return ToolResult(
                "No Claude session is open. Ask the user in which project Claude should work; "
                f"open in PhpStorm: {names}. Then call this tool again with that project.",
                ok=False,
                action=f"Claude: keine offene Sitzung – frage nach dem Projekt ({names})",
                say=speech(
                    "In welchem Projekt soll Claude arbeiten?",
                    "Which project should Claude work in?",
                ),
            )

        name = session_name(project)
        if not await self._open_claude(project, name):
            return ToolResult(
                f"Flay tried to open Claude in a PhpStorm terminal of project '{project.name}', "
                "but it did not come up, so NOTHING was typed. Tell the user to run "
                "'flay claude' in that project's terminal and ask again.",
                ok=False,
                action=(
                    f"PhpStorm ({project.name}): Claude-Terminal nicht gestartet – "
                    "nichts eingetippt"
                ),
                say=speech(
                    f"Ich konnte Claude in {spoken(project.name)} nicht öffnen, es ist nichts "
                    "eingetippt. Starte es bitte mit flay claude im Terminal.",
                    f"I couldn't open Claude in {spoken(project.name)}, nothing was typed. "
                    "Please start it with flay claude in the terminal.",
                ),
            )
        state = await wait_for_claude(self._tmux, name, self._ready_timeout)
        if state != "ready":
            reason = "asks whether to trust the folder" if state == "trust" else "is not ready"
            return ToolResult(
                f"Claude was opened in PhpStorm's terminal of project '{project.name}' but "
                f"{reason}, so NOTHING was typed. Tell the user to look at that terminal.",
                ok=False,
                action=(
                    f"PhpStorm ({project.name}): Claude geöffnet, aber nicht bereit – "
                    "nichts eingetippt"
                ),
                say=_not_ready_speech(project.name, state == "trust"),
            )
        if refused := await self._write(name, text):
            return refused
        return ToolResult(
            f"Claude was opened in a new terminal tab of project '{project.name}' in PhpStorm "
            "and the instruction is written into its input, NOT sent. Tell the user it is "
            "waiting there to check and press Enter.",
            action=(
                f"PhpStorm ({project.name}): Claude im Terminal geöffnet, Auftrag eingetippt, "
                f"nicht abgeschickt – „{text}“"
            ),
            say=_typed_speech(project.name, opened=True),
        )

    async def _write(self, session: str, text: str) -> ToolResult | None:
        """Type ``text`` into Claude's empty input; returns a refusal if that is not safe.

        Text Flay typed earlier and the user never sent is replaced. Anything else in the
        input may be the user's own writing, so Flay leaves it alone.
        """
        current = read_input(await self._tmux.screen(session))
        previous = self._typed.get(session, "")
        if current and previous and _compact(current) == _compact(previous):
            await self._tmux.press(session, "BSpace", len(previous))
            # Claude redraws asynchronously; wait until the input shows as empty.
            for _ in range(20):
                await asyncio.sleep(0.1)
                current = read_input(await self._tmux.screen(session))
                if current == "":
                    break
        if current is None:
            return ToolResult(
                "Claude's input line is not visible (Claude may be showing a question or a "
                "menu), so NOTHING was typed. Tell the user to look at Claude.",
                ok=False,
                action=f"Claude ({session}): Eingabezeile nicht erkennbar – nichts eingetippt",
                say=speech(
                    "Claude zeigt gerade keine Eingabezeile, vielleicht eine Frage oder ein "
                    "Menü. Ich habe nichts eingetippt, schau bitte nach.",
                    "Claude isn't showing its input line, maybe a question or a menu. I "
                    "typed nothing, please have a look.",
                ),
            )
        if current:
            return ToolResult(
                "Claude's input already contains text that Flay did not write, so NOTHING was "
                f"typed. Tell the user to clear Claude's input first. It contains: {current[:200]}",
                ok=False,
                action=(
                    f"Claude ({session}): Eingabe enthält schon fremden Text – nichts eingetippt "
                    f"(„{current[:80]}“)"
                ),
                say=speech(
                    "In Claudes Eingabe steht schon Text von dir, den überschreibe ich nicht. "
                    "Leere sie bitte zuerst.",
                    "Claude's input already has your text in it, and I won't overwrite it. "
                    "Please clear it first.",
                ),
            )
        await self._tmux.type_text(session, text)
        self._typed[session] = text
        return None

    async def _show_in_phpstorm(self, session: Session) -> bool:
        """Attach a detached session in a new terminal tab of its project, if it is open."""
        assert self._phpstorm is not None
        projects = [
            p for p in await self._phpstorm.open_projects() if session_name(p) == session.name
        ]
        if not projects:
            return False
        return await self._open_claude(projects[0], session.name)

    async def _open_claude(self, project: Path, session: str) -> bool:
        """Open ``flay claude`` in a PhpStorm terminal tab and wait until it is shown."""
        assert self._phpstorm is not None
        try:
            launcher = await self._phpstorm.open_claude(project)
        except Exception as exc:
            log.warning("PhpStorm (%s): Claude konnte nicht geöffnet werden: %s", project.name, exc)
            return False
        log.info("PhpStorm (%s): Terminal mit 'flay claude' gestartet", project.name)
        try:
            return await self._wait_for_session(session)
        finally:
            launcher.unlink(missing_ok=True)

    async def _wait_for_session(self, name: str, interval: float = 0.5) -> bool:
        """Wait until the session is shown in a terminal."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._confirm_timeout
        while loop.time() < deadline:
            if any(s.name == name and s.attached for s in await self._tmux.sessions()):
                return True
            await asyncio.sleep(interval)
        return False

    async def _start_and_type(self, project_name: str | None, text: str) -> ToolResult | None:
        if not project_name:
            # PhpStorm's recent-projects file is only written now and then, so "the current
            # project" cannot be derived from it; ask instead of guessing.
            return ToolResult(
                "No Claude session is open. Ask the user which project Claude should work in, "
                "then call this tool again with that project.",
                ok=False,
                action="Claude: keine offene Sitzung – frage nach dem Projekt",
                say=speech(
                    "In welchem Projekt soll Claude arbeiten?",
                    "Which project should Claude work in?",
                ),
            )
        matches = await asyncio.to_thread(self._find, project_name)
        if len(matches) > 1:
            # Asking beats guessing – and beats silently falling back to the clipboard.
            return ambiguous(project_name, matches)
        if not matches:
            return None
        project = matches[0]
        name = session_name(project.path)
        await self._tmux.start(name, project.path, self._claude_command)
        state = await wait_for_claude(self._tmux, name, self._ready_timeout)
        attach = f"run 'flay claude' in PhpStorm's terminal of project '{project.name}'"
        if state == "trust":
            return ToolResult(
                f"Claude Code was started for project '{project.name}' but first asks whether "
                f"to trust this folder, so NOTHING was typed. Tell the user to {attach} and "
                "answer the question, then ask again.",
                ok=False,
                action=(
                    f"Claude ({name}): für „{project.name}“ gestartet, wartet auf "
                    "Vertrauensabfrage – nichts eingetippt"
                ),
                say=_not_ready_speech(project.name, trust=True, background=True),
            )
        if state == "timeout":
            return ToolResult(
                f"Claude Code was started for project '{project.name}' but did not become "
                f"ready, so NOTHING was typed. Tell the user to {attach} and check it.",
                ok=False,
                action=f"Claude ({name}): gestartet, aber nicht bereit – nichts eingetippt",
                say=_not_ready_speech(project.name, trust=False, background=True),
            )
        if refused := await self._write(name, text):
            return refused
        return ToolResult(
            f"No Claude session was running, so Claude Code was started in the background for "
            f"project '{project.name}' and the instruction is written into its input, NOT sent. "
            f"Tell the user: to see it, {attach}; then check it and press Enter.",
            action=(
                f"Claude ({name}): für „{project.name}“ gestartet, Auftrag eingetippt, "
                f"nicht abgeschickt – „{text}“"
            ),
            say=speech(
                f"Ich habe Claude für {spoken(project.name)} im Hintergrund gestartet, der "
                "Auftrag steht in der Eingabe. Gib flay claude im Terminal des Projekts ein, "
                "um es zu sehen, und drück dann Enter.",
                f"I started Claude for {spoken(project.name)} in the background; the "
                "instruction is in its input. Run flay claude in the project's terminal to "
                "see it, then press Enter.",
            ),
        )

    async def _via_clipboard(self, text: str) -> ToolResult:
        await asyncio.to_thread(copy_to_clipboard, text)
        await asyncio.to_thread(bring_phpstorm_to_front)
        return ToolResult(
            "IMPORTANT: nothing was written into Claude, because no Claude session started with "
            "'flay claude' is running. The instruction is only on the clipboard and PhpStorm "
            "is in front. Tell the user exactly that: they have to click into Claude and paste "
            "it (Cmd+V / Ctrl+V), and next time start Claude with 'flay claude'.",
            action=(
                "Claude: keine „flay claude“-Sitzung gefunden → Auftrag in die Zwischenablage "
                f"kopiert, PhpStorm nach vorn geholt – „{text}“"
            ),
            say=speech(
                "Ich finde kein laufendes Claude. Der Auftrag liegt in der Zwischenablage: "
                "klick in Claude und füg ihn ein.",
                "I can't find a running Claude. The instruction is on the clipboard: click "
                "into Claude and paste it.",
            ),
        )


def _typed_speech(project: str, opened: bool = False) -> dict[Language, str]:
    where = spoken(project)
    if opened:
        return speech(
            f"Ich habe Claude in {where} geöffnet, der Auftrag steht in der Eingabe. Prüf ihn "
            "und drück Enter.",
            f"I opened Claude in {where}; the instruction is in its input. Check it and press "
            "Enter.",
        )
    return speech(
        f"Der Auftrag steht in Claude für {where}. Prüf ihn und drück Enter.",
        f"The instruction is in Claude for {where}. Check it and press Enter.",
    )


def _not_ready_speech(project: str, trust: bool, background: bool = False) -> dict[Language, str]:
    where = spoken(project)
    attach_de = " Gib flay claude im Terminal des Projekts ein." if background else ""
    attach_en = " Run flay claude in the project's terminal." if background else ""
    if trust:
        return speech(
            f"Claude in {where} fragt erst, ob es dem Ordner vertrauen soll. Ich habe nichts "
            f"eingetippt.{attach_de} Bestätige das bitte und sag es mir dann nochmal.",
            f"Claude in {where} first asks whether to trust the folder. I typed nothing."
            f"{attach_en} Please confirm and tell me again.",
        )
    return speech(
        f"Claude in {where} ist nicht bereit geworden, ich habe nichts eingetippt.{attach_de}",
        f"Claude in {where} didn't become ready, I typed nothing.{attach_en}",
    )


def _matches(session: Session, project: str) -> bool:
    wanted = _normalize(project)
    return bool(wanted) and wanted in _normalize(session.project)


class ClaudeStatusTool:
    def __init__(self, tmux: Tmux | None = None) -> None:
        self._tmux = tmux or Tmux()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="claude_status",
            description=(
                "Read what Claude Code currently shows in PhpStorm's terminal: whether it is "
                "still working, finished, or waiting for a permission or answer. Summarize it "
                "briefly for the user instead of reading it out."
            ),
            parameters={"type": "object", "properties": {"project": _PROJECT_PARAMETER}},
        )

    async def screen_text(self, project: str | None = None) -> str | None:
        """What Claude's session shows now (e.g. to take a link from its answer)."""
        if not self._tmux.available:
            return None
        session = pick_session(await self._tmux.sessions(), project)
        return await self._tmux.screen(session.name) if session else None

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        session = pick_session(await self._tmux.sessions(), str(arguments.get("project") or ""))
        if session is None:
            return ToolResult(
                "No Claude session started with 'flay claude' is running, so its status "
                "cannot be read.",
                ok=False,
            )
        screen = await self._tmux.screen(session.name)
        return ToolResult(
            f"Claude Code screen (project '{session.project}'):\n{screen}",
            action=f"Claude ({session.name}): Terminal gelesen",
        )
