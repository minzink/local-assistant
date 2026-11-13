"""Working in PhpStorm through its MCP server: open a file, run a configuration, read the
problems of the open file and the Git status.

The project is the one the user names, else the one whose PhpStorm window is in front.
Everything here only reads or opens, except running a configuration – which the user
asks for by name and which PhpStorm shows in its Run window as usual.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech, spoken
from flay.tools.phpstorm import Project, match_names, match_projects
from flay.tools.phpstorm_mcp import PhpStormMcp

log = logging.getLogger(__name__)

ACTIONS = ("open_file", "run", "problems", "git_status")
# A run configuration may take a while (tests); longer runs keep going in PhpStorm.
RUN_TIMEOUT_MS = 120_000
MAX_OUTPUT_CHARS = 3000
MAX_PROBLEMS = 8


def front_title() -> str:
    """Title of the frontmost PhpStorm window ("flay – assistant.py"), if any."""
    from flay.tools.screen import list_windows

    for window in list_windows():
        if "phpstorm" in window.app.casefold():
            return window.title
    return ""


def file_patterns(query: str) -> list[str]:
    """Search patterns for a spoken file name: "User Controller" → "**/UserController*"."""
    words = re.findall(r"[\w.-]+", query)
    joined = "".join(words)
    patterns = [f"**/{joined}*", f"**/*{joined}*"]
    if len(words) > 1:
        patterns.append(f"**/*{'*'.join(words)}*")
    lower = joined.lower()
    if lower != joined:
        patterns.append(f"**/*{lower}*")
    return list(dict.fromkeys(patterns))


def best_file(query: str, paths: list[str]) -> str | None:
    """The path whose file name fits the spoken name best (shortest wins a tie)."""
    if not paths:
        return None
    wanted = re.sub(r"\W", "", query).casefold()

    def score(path: str) -> tuple[int, int]:
        stem = re.sub(r"\W", "", Path(path).stem).casefold()
        return (0 if stem == wanted else 1 if stem.startswith(wanted) else 2, len(path))

    return min(paths, key=score)


class IdeTool:
    def __init__(
        self,
        phpstorm: PhpStormMcp,
        title: Callable[[], str] = front_title,
    ) -> None:
        self._phpstorm = phpstorm
        self._title = title

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="phpstorm",
            description=(
                "Work in PhpStorm: 'open_file' (a file or class named in 'query'), 'run' a run "
                "configuration ('query' = its name, e.g. tests; empty lists them), 'problems' "
                "of the open file, 'git_status'. 'project' only if the user names one."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "query": {"type": "string"},
                    "project": {"type": "string"},
                },
                "required": ["action"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = str(arguments.get("action") or "")
        if action not in ACTIONS:
            return ToolResult(f"Unknown action '{action}'.", ok=False)
        project = await self._project(str(arguments.get("project") or ""))
        if isinstance(project, ToolResult):
            return project
        query = str(arguments.get("query") or "").strip()
        handler = getattr(self, f"_{action}")
        result: ToolResult = await handler(project, query)
        return result

    async def _project(self, name: str) -> Project | ToolResult:
        try:
            paths = await self._phpstorm.open_projects()
        except Exception as exc:
            paths = []
            log.debug("PhpStorm: offene Projekte nicht lesbar: %s", exc)
        projects = [Project(p.name, p) for p in paths]
        if not projects:
            return ToolResult(
                "PhpStorm has no project open (or its MCP server is off).",
                ok=False,
                say=speech(
                    "In PhpStorm ist gerade kein Projekt offen.", "No project is open in PhpStorm."
                ),
            )
        if name:
            found = match_projects(name, projects)
            if len(found) == 1:
                return found[0]
            return ToolResult(
                f"No single open project matches '{name}'. Open: "
                + ", ".join(p.name for p in projects),
                ok=False,
                say=speech(f"Welches Projekt meinst du mit {name}?", f"Which project is {name}?"),
            )
        if len(projects) == 1:
            return projects[0]
        title = await asyncio.to_thread(self._title)
        front = match_names(title.split(" – ")[0].split(" - ")[0], [p.name for p in projects])
        if len(front) == 1:
            return next(p for p in projects if p.name == front[0])
        return ToolResult(
            "Several projects are open and none is in front: "
            + ", ".join(p.name for p in projects),
            ok=False,
            say=speech("In welchem Projekt?", "In which project?"),
        )

    async def _call(self, tool: str, project: Project, **arguments: Any) -> Any:
        text = await self._phpstorm.call(tool, {**arguments, "projectPath": str(project.path)})
        try:
            return json.loads(text)
        except ValueError:
            return text

    async def _open_file(self, project: Project, query: str) -> ToolResult:
        if not query:
            return ToolResult(
                "Name the file to open.", ok=False, say=speech("Welche Datei?", "Which file?")
            )
        paths: list[str] = []
        for pattern in file_patterns(query):
            found = await self._call("search_file", project, q=pattern, limit=20)
            paths = [i["filePath"] for i in (found or {}).get("items", []) if "filePath" in i]
            if paths:
                break
        if not paths:
            # A class or function: PhpStorm's symbol index (PHP, JS, ...).
            found = await self._call("search_symbol", project, q=query.replace(" ", ""), limit=5)
            items = (found or {}).get("items", []) if isinstance(found, dict) else []
            paths = [i["filePath"] for i in items if "filePath" in i]
        path = best_file(query, paths)
        if path is None:
            return ToolResult(
                f"No file like '{query}' in {project.name}.",
                ok=False,
                action=f"PhpStorm: keine Datei „{query}“",
                say=speech(
                    f"Eine Datei {query} finde ich in {spoken(project.name)} nicht.",
                    f"I can't find a file {query} in {spoken(project.name)}.",
                ),
            )
        await self._call("open_file_in_editor", project, filePath=path)
        name = Path(path).name
        return ToolResult(
            f"Opened {path} in {project.name}."
            + (f" Other matches: {len(paths) - 1}." if len(paths) > 1 else ""),
            action=f"PhpStorm: {path} geöffnet",
            say=speech(
                f"{spoken(Path(name).stem)} ist offen.", f"{spoken(Path(name).stem)} is open."
            ),
        )

    async def _run(self, project: Project, query: str) -> ToolResult:
        found = await self._call("get_run_configurations", project)
        configurations = (
            [str(c.get("name")) for c in (found or {}).get("configurations", []) if c.get("name")]
            if isinstance(found, dict)
            else []
        )
        if not configurations:
            return ToolResult(
                f"{project.name} has no run configurations.",
                ok=False,
                say=speech(
                    "Das Projekt hat keine Ausführungskonfiguration.",
                    "The project has no run configuration.",
                ),
            )
        names = match_names(query, configurations) if query else []
        if len(names) != 1:
            listed = ", ".join(configurations)
            return ToolResult(
                f"Which run configuration? Available: {listed}.",
                ok=False,
                say=speech(
                    f"Welche Konfiguration? Es gibt {listed}.",
                    f"Which configuration? There are {listed}.",
                ),
            )
        name = names[0]
        log.info("PhpStorm: starte „%s“ in %s", name, project.name)
        try:
            output = await self._call(
                "execute_run_configuration",
                project,
                configurationName=name,
                waitForExit=True,
                timeout=RUN_TIMEOUT_MS,
            )
        except RuntimeError as failed:
            output = str(failed)
        text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        tail = text[-MAX_OUTPUT_CHARS:]
        return ToolResult(
            f"Ran '{name}' in {project.name}. Output (end):\n{tail}\nSummarize the outcome in "
            "one or two sentences (passed/failed, how many, the first error).",
            action=f"PhpStorm: „{name}“ ausgeführt",
        )

    async def _problems(self, project: Project, query: str) -> ToolResult:
        opened = await self._call("get_all_open_file_paths", project)
        path = (opened or {}).get("activeFilePath") if isinstance(opened, dict) else None
        if query and isinstance(opened, dict):
            path = best_file(query, list(opened.get("openFiles", []))) or path
        if not path:
            return ToolResult(
                "No file is open in the editor.",
                ok=False,
                say=speech("Im Editor ist keine Datei offen.", "No file is open."),
            )
        found = await self._call("get_file_problems", project, filePath=path, errorsOnly=False)
        problems = (found or {}).get("errors", []) if isinstance(found, dict) else []
        if not problems:
            return ToolResult(
                f"PhpStorm reports no problems in {path}.",
                action=f"PhpStorm: keine Probleme in {path}",
                say=speech(
                    f"In {spoken(Path(path).stem)} meldet PhpStorm keine Probleme.",
                    f"PhpStorm reports no problems in {spoken(Path(path).stem)}.",
                ),
            )
        lines = [
            f"line {p.get('line', '?')}: {p.get('description') or p.get('message') or p}"
            for p in problems[:MAX_PROBLEMS]
        ]
        return ToolResult(
            f"{len(problems)} problems in {path}:\n"
            + "\n".join(lines)
            + "\nName the most important ones briefly.",
            action=f"PhpStorm: {len(problems)} Probleme in {path}",
        )

    async def _git_status(self, project: Project, query: str) -> ToolResult:
        found = await self._call("git_status", project, limit=20)
        repositories = (found or {}).get("repositories", []) if isinstance(found, dict) else []
        if not repositories:
            return ToolResult(
                f"{project.name} is not under Git.",
                ok=False,
                say=speech("Das Projekt hat kein Git.", "The project has no Git."),
            )
        lines = []
        for repo in repositories:
            changed = [e.get("pathRelativeToRepository", "") for e in repo.get("entries", [])][:10]
            counts = (
                f"{repo.get('stagedCount', 0)} staged, {repo.get('unstagedCount', 0)} changed, "
                f"{repo.get('untrackedCount', 0)} new, {repo.get('conflictedCount', 0)} "
                f"conflicts; e.g. {', '.join(changed)}"
            )
            state = "clean" if repo.get("isClean") else counts
            lines.append(f"branch {repo.get('currentBranch')}: {state}")
        return ToolResult(
            f"Git status of {project.name}: " + " | ".join(lines),
            action=f"PhpStorm: Git-Status von {project.name}",
        )
