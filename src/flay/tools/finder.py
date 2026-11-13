"""Finding a file by name in the usual folders, newest first, and opening it.

"Öffne die letzte PDF aus Downloads", "Wo ist die Rechnung von Oktober?". Only the
configured folders are searched (Downloads, Desktop, Documents by default), a few levels
deep; the contents of files are not read here (``read_file`` does that).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech

log = logging.getLogger(__name__)

MAX_DEPTH = 4
MAX_FILES = 30_000
SEARCH_SECONDS = 3.0
SHOWN = 5
_SKIP = re.compile(r"^(?:\.|node_modules$|vendor$|__pycache__$|Library$)")
# Spoken types → extensions.
KINDS = {
    "pdf": (".pdf",),
    "bild": (".png", ".jpg", ".jpeg", ".heic", ".gif", ".webp"),
    "image": (".png", ".jpg", ".jpeg", ".heic", ".gif", ".webp"),
    "word": (".docx", ".doc"),
    "excel": (".xlsx", ".xls", ".csv"),
    "tabelle": (".xlsx", ".xls", ".csv", ".numbers"),
    "text": (".txt", ".md"),
    "zip": (".zip",),
    "video": (".mp4", ".mov"),
    "präsentation": (".pptx", ".key"),
}
# Words that say nothing about the name.
_FILLER = re.compile(
    r"\b(?:die|der|das|den|dem|eine?|letzte\w*|neueste\w*|aktuell\w*|datei|file|von|vom|aus|"
    r"im|in|the|latest|last|newest|from|my|meine?\w*|ordner|folder)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Found:
    path: Path
    modified: float


def default_roots() -> list[Path]:
    home = Path.home()
    return [home / "Downloads", home / "Desktop", home / "Documents"]


def walk(roots: list[Path], deadline: float) -> Iterator[Path]:
    count = 0
    for root in roots:
        stack = [(root, 0)]
        while stack:
            folder, depth = stack.pop()
            try:
                entries = list(os.scandir(folder))
            except OSError:
                continue
            for entry in entries:
                if _SKIP.match(entry.name):
                    continue
                count += 1
                if count > MAX_FILES or time.monotonic() > deadline:
                    return
                if entry.is_dir(follow_symlinks=False):
                    if depth < MAX_DEPTH:
                        stack.append((Path(entry.path), depth + 1))
                elif entry.is_file():
                    yield Path(entry.path)


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.casefold()) if len(w) >= 2]


def search(query: str, kind: str, roots: list[Path]) -> list[Found]:
    """Files whose names contain every word of ``query`` (of type ``kind``), newest first."""
    extensions = KINDS.get(kind.casefold(), (f".{kind.casefold().lstrip('.')}",) if kind else ())
    words = _words(_FILLER.sub(" ", query))
    found = []
    for path in walk(roots, time.monotonic() + SEARCH_SECONDS):
        if extensions and path.suffix.casefold() not in extensions:
            continue
        name = path.name.casefold()
        if all(word in name for word in words):
            try:
                found.append(Found(path, path.stat().st_mtime))
            except OSError:
                continue
    return sorted(found, key=lambda f: f.modified, reverse=True)


def open_with_default_app(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=True, timeout=10)
    elif sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined, unused-ignore]
    else:
        subprocess.run(["xdg-open", str(path)], check=True, timeout=10)


class FindFileTool:
    def __init__(
        self,
        roots: list[Path] | None = None,
        opener: Callable[[Path], None] = open_with_default_app,
    ) -> None:
        self._roots = roots or default_roots()
        self._open = opener

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="find_file",
            description=(
                "Find a file by name words in Downloads, Desktop and Documents, newest first; "
                "'open' opens the best match with its app. 'kind' e.g. 'pdf', 'bild', 'excel'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Words of the file name"},
                    "kind": {"type": "string"},
                    "open": {"type": "boolean"},
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "")
        kind = str(arguments.get("kind") or "")
        if not _words(_FILLER.sub(" ", query)) and not kind:
            return ToolResult(
                "Name words of the file or its kind.",
                ok=False,
                say=speech("Welche Datei suchst du?", "Which file are you looking for?"),
            )
        found = await asyncio.to_thread(search, query, kind, self._roots)
        if not found:
            what = " ".join(filter(None, [kind, query])).strip()
            return ToolResult(
                f"No file matching '{what}' in {', '.join(r.name for r in self._roots)}.",
                ok=False,
                action=f"Datei: „{what}“ nicht gefunden",
                say=speech(f"Ich finde keine Datei {what}.", f"I can't find a file {what}."),
            )
        best = found[0]
        listed = "; ".join(
            f"{f.path} ({time.strftime('%Y-%m-%d %H:%M', time.localtime(f.modified))})"
            for f in found[:SHOWN]
        )
        if arguments.get("open"):
            await asyncio.to_thread(self._open, best.path)
            return ToolResult(
                f"Opened {best.path}. Other matches: {len(found) - 1}.",
                action=f"Datei geöffnet: {best.path}",
                say=speech(f"Ich habe {best.path.name} geöffnet.", f"I opened {best.path.name}."),
            )
        return ToolResult(
            f"{len(found)} files found, newest first: {listed}. The path can be passed to "
            "read_file to read it.",
            action=f"Datei gesucht: {len(found)} Treffer",
        )
