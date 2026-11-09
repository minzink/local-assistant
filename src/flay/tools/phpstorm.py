"""PhpStorm: find recently opened projects and open one by (spoken) name."""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech, spoken

if TYPE_CHECKING:
    from flay.tools.phpstorm_mcp import PhpStormMcp


@dataclass(frozen=True, slots=True)
class Project:
    name: str
    path: Path


def jetbrains_config_root() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "JetBrains"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "JetBrains"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "JetBrains"


def _version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", path.name))


def latest_config_dir(root: Path) -> Path | None:
    """Config dir of the newest PhpStorm version, e.g. ``PhpStorm2026.2``."""
    candidates = [p for p in root.glob("PhpStorm*") if (p / "options").is_dir()]
    return max(candidates, key=_version_key, default=None)


def read_recent_projects(config_dir: Path) -> list[Project]:
    file = config_dir / "options" / "recentProjects.xml"
    if not file.exists():
        return []
    home = str(Path.home())
    projects = []
    for entry in ET.parse(file).getroot().iter("entry"):
        key = entry.get("key")
        if not key:
            continue
        path = Path(key.replace("$USER_HOME$", home))
        projects.append(Project(name=path.name, path=path))
    return projects


def recent_projects(config_dir: Path | None = None) -> list[Project]:
    directory = config_dir or latest_config_dir(jetbrains_config_root())
    return read_recent_projects(directory) if directory else []


def resolve_project(name: str | None, config_dir: Path | None = None) -> Project | None:
    """The named project if exactly one recent project matches."""
    if not name:
        return None
    matches = match_projects(name, recent_projects(config_dir))
    return matches[0] if len(matches) == 1 else None


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", "", text.lower())


def match_projects(query: str, projects: list[Project]) -> list[Project]:
    """Best matches for a spoken name: "stadt wolkenthal" finds "stadt-wolkenthal"."""
    # Two projects may share a name (different folders): both match, Flay asks which.
    names = match_names(query, list(dict.fromkeys(p.name for p in projects)))
    return [p for name in names for p in projects if p.name == name]


def match_names(query: str, names: list[str]) -> list[str]:
    """Best matches for a spoken name among ``names`` (projects, servers)."""
    wanted = _normalize(query)
    if not wanted:
        return []
    exact = [n for n in names if _normalize(n) == wanted]
    if exact:
        return exact
    contained = [n for n in names if wanted in _normalize(n)]
    if contained:
        return contained
    # Speech recognition garbles names ("KEWO" for "kx-ewo"), and the distinctive part is
    # usually at the end ("project-symfony-kx-ewo"), so the name's tails count too.
    scores = {n: max(_similarity(wanted, tail) for tail in _tails(n)) for n in names}
    best = sorted((n for n in names if scores[n] >= 0.75), key=lambda n: -scores[n])
    if len(best) > 1 and scores[best[0]] - scores[best[1]] >= CLEAR_LEAD:
        # "K K P F B": kx-kpfb fits clearly better than kx-kpf – no need to ask.
        return best[:1]
    return best[:3]


# How much better the best garbled match must fit than the next to be taken alone.
CLEAR_LEAD = 0.08


def _similarity(heard: str, tail: str) -> float:
    """How well a heard name fits a name tail; one lost letter ("K K P F" for "kx-kpf",
    the "D" swallowed) is almost as good as the name itself."""
    one_lost = len(heard) >= 4 and len(tail) == len(heard) + 1
    if one_lost and any(tail[:i] + tail[i + 1 :] == heard for i in range(len(tail))):
        return 0.95
    return difflib.SequenceMatcher(None, heard, tail).ratio()


def _tails(name: str) -> list[str]:
    """``project-symfony-kx-ewo`` → projectsymfonykxewo, symfonykxewo, kxewo, ewo."""
    parts = [part for part in re.split(r"[^a-z0-9äöüß]+", name.lower()) if part]
    return ["".join(parts[i:]) for i in range(len(parts))] or [_normalize(name)]


def mention(name: str, said: str, similarity: float = 0.75) -> Literal["exact", "fuzzy"] | None:
    """How clearly the user's words contain the project name: literally, roughly, or not.

    Speech recognition garbles names ("K X E V O" for "kx-ewo"), so a name counts as
    mentioned if one of its distinctive tails appears in the words with small errors.
    """
    heard = _normalize(said)
    tails = _tails(name)
    if len(tails) > 1 and len(tails[-1]) <= 4:
        # A short last part ("cms", "app") is shared by many projects and proves nothing.
        tails = tails[:-1]
    candidates = [tail for tail in tails if len(tail) >= 3]
    if any(tail in heard for tail in candidates):
        return "exact"
    for tail in candidates:
        if len(tail) < 4:
            continue
        size = len(tail)
        for start in range(max(1, len(heard) - size + 1)):
            window = heard[start : start + size]
            if difflib.SequenceMatcher(None, tail, window).ratio() >= similarity:
                return "fuzzy"
    return None


def mentioned(name: str, said: str) -> bool:
    return mention(name, said) is not None


def find_launcher(configured: Path | None) -> list[str]:
    """Command that opens a path in PhpStorm."""
    if configured is not None:
        return [str(configured)]
    for name in ("phpstorm", "phpstorm.cmd", "pstorm"):
        if found := shutil.which(name):
            return [found]
    if sys.platform == "darwin":
        toolbox = jetbrains_config_root() / "Toolbox" / "scripts" / "phpstorm"
        if toolbox.exists():
            return [str(toolbox)]
        return ["open", "-na", "PhpStorm.app", "--args"]
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        toolbox = local / "JetBrains" / "Toolbox" / "scripts" / "phpstorm.cmd"
        if toolbox.exists():
            return [str(toolbox)]
    raise FileNotFoundError("PhpStorm launcher not found; set tools.phpstorm.launcher in flay.yaml")


class OpenProjectTool:
    def __init__(self, launcher: Path | None = None, config_dir: Path | None = None) -> None:
        self._launcher = launcher
        self._config_dir = config_dir

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="open_project",
            description=(
                "Open a project in the PhpStorm IDE. Use the project name the user says; "
                "it is matched against PhpStorm's recently opened projects."
            ),
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Project name"}},
                "required": ["name"],
            },
        )

    def projects(self) -> list[Project]:
        return recent_projects(self._config_dir)

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        name = str(arguments.get("name", "")).strip()
        if not name:
            return which_project("öffnen", "open")
        projects = self.projects()
        matches = match_projects(name, projects)
        if not matches:
            known = ", ".join(p.name for p in projects[:15]) or "none"
            return ToolResult(
                f"No project matches '{name}'. Recent projects: {known}.",
                ok=False,
                action=f"PhpStorm: kein Projekt „{name}“ gefunden",
                say=speech(
                    f"Ich finde kein Projekt namens {name}.",
                    f"I can't find a project called {name}.",
                ),
            )
        if len(matches) > 1:
            return ambiguous(name, matches)
        project = matches[0]
        if not project.path.exists():
            return ToolResult(
                f"Project folder {project.path} no longer exists.",
                ok=False,
                say=speech(
                    f"Den Ordner von {spoken(project.name)} gibt es nicht mehr.",
                    f"The folder of {spoken(project.name)} no longer exists.",
                ),
            )
        process = await asyncio.create_subprocess_exec(
            *find_launcher(self._launcher),
            str(project.path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        # Launchers usually return at once; never block the conversation on them.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=10)
        return ToolResult(
            f"Opened project '{project.name}' in PhpStorm.",
            action=f"PhpStorm: Projekt „{project.name}“ geöffnet ({project.path})",
            say=speech(
                f"Ich öffne {spoken(project.name)} in PhpStorm.",
                f"Opening {spoken(project.name)} in PhpStorm.",
            ),
        )


def which_project(verb_de: str, verb_en: str) -> ToolResult:
    return ToolResult(
        "No project named; ask which one.",
        ok=False,
        say=speech(f"Welches Projekt soll ich {verb_de}?", f"Which project should I {verb_en}?"),
    )


def ambiguous(name: str, matches: list[Project]) -> ToolResult:
    """Several projects fit: ask which one, naming them."""
    options = ", ".join(p.name for p in matches)
    names = [spoken(p.name) for p in matches]
    de = ", ".join(names[:-1]) + " oder " + names[-1]
    en = ", ".join(names[:-1]) + " or " + names[-1]
    return ToolResult(
        f"'{name}' is ambiguous: {options}. Ask which one.",
        ok=False,
        action=f"PhpStorm: „{name}“ mehrdeutig – {options}",
        say=speech(f"Meinst du {de}?", f"Do you mean {en}?"),
    )


# Opening behaviour ------------------------------------------------------------------------

# PhpStorm's "Open project in" setting: -1 asks every time, 0 = new window, 1 = this window.
_OPEN_SETTING = re.compile(r'name="confirmOpenNewProject2"\s+value="(-?\d+)"')


def open_project_setting(config_dir: Path | None = None) -> int | None:
    """PhpStorm's "Open project in" choice, None if it cannot be read (default: ask)."""
    directory = config_dir or latest_config_dir(jetbrains_config_root())
    if directory is None:
        return None
    file = directory / "options" / "ide.general.xml"
    if not file.exists():
        return -1
    match = _OPEN_SETTING.search(file.read_text(encoding="utf-8", errors="replace"))
    return int(match.group(1)) if match else -1


# Closing ------------------------------------------------------------------------------------


class CloseProjectTool:
    """Close a project window in PhpStorm (via PhpStorm's MCP server)."""

    def __init__(self, phpstorm: PhpStormMcp, poll_interval: float = 0.5) -> None:
        self._phpstorm = phpstorm
        self._poll_interval = poll_interval

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="close_project",
            description=(
                "Close a project that is open in the PhpStorm IDE. Use the project name the user "
                "says; it is matched against the projects currently open in PhpStorm."
            ),
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Project name"}},
                "required": ["name"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        name = str(arguments.get("name", "")).strip()
        if not name:
            return which_project("schließen", "close")
        open_paths = await self._phpstorm.open_projects()
        projects = [Project(name=path.name, path=path) for path in open_paths]
        matches = match_projects(name, projects)
        if not matches:
            known = ", ".join(p.name for p in projects) or "none"
            return ToolResult(
                f"No open project matches '{name}'. Open in PhpStorm: {known}.",
                ok=False,
                action=f"PhpStorm: kein geöffnetes Projekt „{name}“",
                say=speech(
                    f"In PhpStorm ist kein Projekt {name} offen.",
                    f"There is no project {name} open in PhpStorm.",
                ),
            )
        if len(matches) > 1:
            return ambiguous(name, matches)
        project = matches[0]
        if shadow := await self._shadowing_project(project, open_paths):
            return ToolResult(
                f"Not closed: '{shadow.name}' is open too and includes '{project.name}' as part "
                f"of its own project, so PhpStorm would close '{shadow.name}' instead. Tell the "
                f"user to close '{project.name}' by hand (or to close '{shadow.name}' first).",
                ok=False,
                action=(
                    f"PhpStorm: „{project.name}“ nicht geschlossen – steckt auch in „{shadow.name}“"
                ),
                say=speech(
                    f"Ich habe {spoken(project.name)} nicht geschlossen: {spoken(shadow.name)} ist "
                    "auch offen und enthält es, PhpStorm würde sonst das falsche Projekt "
                    "schließen. Bitte schließ es von Hand.",
                    f"I did not close {spoken(project.name)}: {spoken(shadow.name)} is open too "
                    "and contains it, so PhpStorm would close the wrong project. Please close it "
                    "by hand.",
                ),
            )
        await self._phpstorm.invoke_action("CloseProject", project.path)
        # PhpStorm closes asynchronously; confirm by asking which projects are still open.
        for _ in range(10):
            await asyncio.sleep(self._poll_interval)
            still_open = await self._phpstorm.open_projects()
            gone = [path for path in open_paths if path not in still_open]
            if project.path in gone:
                return ToolResult(
                    f"Closed project '{project.name}' in PhpStorm.",
                    action=f"PhpStorm: Projekt „{project.name}“ geschlossen",
                    say=speech(
                        f"{spoken(project.name)} ist geschlossen.",
                        f"{spoken(project.name)} is closed.",
                    ),
                )
            if gone:
                closed = ", ".join(path.name for path in gone)
                return ToolResult(
                    f"PhpStorm closed '{closed}' instead of '{project.name}', which is still "
                    "open. Tell the user honestly.",
                    ok=False,
                    action=f"PhpStorm: statt „{project.name}“ wurde „{closed}“ geschlossen",
                    say=speech(
                        f"Achtung: PhpStorm hat {spoken(closed)} geschlossen statt "
                        f"{spoken(project.name)}.",
                        f"Careful: PhpStorm closed {spoken(closed)} instead of "
                        f"{spoken(project.name)}.",
                    ),
                )
        return ToolResult(
            f"Asked PhpStorm to close '{project.name}', but it is still open (maybe PhpStorm "
            "is asking something, e.g. about running processes). Tell the user to check.",
            ok=False,
            action=f"PhpStorm: Schließen von „{project.name}“ angefordert, Projekt noch offen",
            say=speech(
                f"{spoken(project.name)} ist noch offen. Vielleicht fragt PhpStorm gerade "
                "etwas, schau bitte kurz nach.",
                f"{spoken(project.name)} is still open. PhpStorm may be asking something, "
                "please have a look.",
            ),
        )

    async def _shadowing_project(self, project: Project, open_paths: list[Path]) -> Path | None:
        """An open parent project that PhpStorm would pick instead of ``project``.

        PhpStorm maps a path to the open project with the deepest content root containing
        it. A parent project that lists the subproject's folder as one of its own content
        roots ties with it, and the parent wins – closing would hit the wrong window. The
        project PhpStorm actually picks shows in its module list.
        """
        parents = [p for p in open_paths if p != project.path and project.path.is_relative_to(p)]
        if not parents:
            return None
        picked = await self._phpstorm.project_modules(project.path)
        for parent in parents:
            if await self._phpstorm.project_modules(parent) == picked:
                return parent
        return None
