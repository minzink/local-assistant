"""Read local files: "Lies die README von flay", "Was steht im PDF auf dem Schreibtisch?".

The file is found by its name in the places the user names (a PhpStorm project, the desktop,
downloads, documents), read as text (plain text and code, Markdown, PDF, Word) and handed to
the same text loop as web pages (``flay.browsing.browse_text``): long files in parts, the
parts about the request first. Nothing leaves the computer.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech
from flay.tools.pages import Link, Page, describe
from flay.tools.phpstorm import Project, match_projects, recent_projects

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {
    ".md", ".txt", ".rst", ".py", ".php", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".conf", ".csv", ".html", ".htm", ".xml", ".css",
    ".scss", ".sh", ".log", ".sql", ".java", ".kt", ".go", ".rs", ".rb", ".c", ".h", ".cpp",
    ".twig", ".example",
}  # fmt: skip
# Text files without a suffix.
TEXT_NAMES = {"readme", "license", "licence", "changelog", "makefile", "dockerfile", "notice"}
_DOT_FILES = {"ai-instruct", "gitignore", "editorconfig", "env.example", "gitattributes"}
DOCUMENT_SUFFIXES = {".pdf", ".docx"}
MAX_BYTES = 5_000_000
SEARCH_SECONDS = 2.0
MAX_ENTRIES = 30_000
_SKIP_DIRS = {
    ".git", "node_modules", "vendor", ".venv", "venv", "__pycache__", ".idea", "dist", "build",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", "Library", ".cache", ".Trash", "var",
    "storage", "cache", "tmp", "coverage", "logs", "target", ".next", ".nuxt", "bower_components",
}  # fmt: skip

# Spoken places → folders in the home directory.
_PLACES = (
    (re.compile(r"schreibtisch|desktop", re.IGNORECASE), "Desktop"),
    (re.compile(r"download", re.IGNORECASE), "Downloads"),
    (re.compile(r"dokumente|documents|unterlagen", re.IGNORECASE), "Documents"),
)
_NEWEST = re.compile(r"neueste|zuletzt|letzte|latest|newest|most recent", re.IGNORECASE)
_KINDS = (
    (re.compile(r"\bpdf\b", re.IGNORECASE), {".pdf"}),
    (re.compile(r"\bword\b|\bdocx\b", re.IGNORECASE), {".docx"}),
    (re.compile(r"markdown|\bmd\b", re.IGNORECASE), {".md"}),
)
# Words of a spoken file name that say nothing about which file.
_FILLER = {
    "die", "der", "das", "datei", "file", "dokument", "document", "von", "vom", "aus", "im",
    "in", "auf", "dem", "den", "the", "of", "from", "my", "meine", "meiner", "mein", "pdf",
    "projekt", "project", "ordner", "folder", "neueste", "zuletzt", "letzte", "latest",
}  # fmt: skip


def readable(path: Path) -> bool:
    name = path.name.lower().lstrip(".")  # ".ai-instruct", ".env.example"
    if name in TEXT_NAMES or name in _DOT_FILES:
        return True
    suffix = Path(name).suffix
    return suffix in TEXT_SUFFIXES or suffix in DOCUMENT_SUFFIXES


def roots(where: str, projects: list[Project], name: str = "") -> list[Path]:
    """Folders to look in: the named project or place, else projects and the usual places.

    A project named among the file's words ("flay example yaml") counts as the place.
    """
    home = Path.home()
    named = [home / folder for pattern, folder in _PLACES if pattern.search(where)]
    if where.strip() and (found := match_projects(where, projects)):
        named += [project.path for project in found]
    if not named:
        words = set(name_words(name))
        named = [p.path for p in projects if p.name.lower() in words]
    if named:
        return [path for path in named if path.is_dir()]
    usual = [home / folder for _, folder in _PLACES]
    return [path for path in [*(p.path for p in projects), *usual] if path.is_dir()]


def walk(root: Path, depth: int, deadline: float) -> Iterator[Path]:
    """Readable files under ``root``, breadth first (files near the top are found first)."""
    queue = [(root, 0)]
    while queue and time.monotonic() < deadline:
        folder, level = queue.pop(0)
        try:
            with os.scandir(folder) as entries:
                listed = list(entries)
        except OSError:
            continue
        for entry in listed:
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                if level < depth and entry.name not in _SKIP_DIRS and entry.name[:1] != ".":
                    queue.append((Path(entry.path), level + 1))
            elif readable(Path(entry.name)):
                yield Path(entry.path)


def name_words(name: str) -> list[str]:
    words = re.findall(r"[a-z0-9äöüß]+", name.lower().replace("punkt", "."))
    return [w for w in words if w not in _FILLER and len(w) > 1]


def score(path: Path, words: list[str]) -> float:
    """How well a file's name fits the spoken words (0: not at all).

    Words count at the start of the name's parts: "ai" is in ".ai-instruct", not in
    "training.py".
    """
    if not words:
        return 0.0
    parts = re.findall(r"[a-z0-9äöüß]+", path.name.lower())
    hits = sum(1 for word in words if any(part.startswith(word) for part in parts))
    if hits == 0:
        return 0.0
    exact = 1.0 if "".join(words) == "".join(parts[: len(words)]) else 0.0
    return hits / len(words) + exact - len(path.name) / 1000


def find_file(
    name: str, where: str, projects: list[Project], seconds: float = SEARCH_SECONDS
) -> tuple[Path | None, list[Path]]:
    """The file the user means and other candidates; the newest one for "die neueste"."""
    text = f"{name} {where}"
    kinds = next((suffixes for pattern, suffixes in _KINDS if pattern.search(text)), None)
    words = name_words(name)
    newest = bool(_NEWEST.search(text))
    deadline = time.monotonic() + seconds
    candidates: list[tuple[float, float, Path]] = []
    seen = 0
    for root in roots(where, projects, name):
        depth = 4 if root.parent != Path.home() else 2
        for path in walk(root, depth, deadline):
            seen += 1
            if seen > MAX_ENTRIES:
                break
            if kinds is not None and path.suffix.lower() not in kinds:
                continue
            fit = score(path, words) if words else (1.0 if newest or kinds else 0.0)
            if fit > 0:
                try:
                    modified = path.stat().st_mtime
                except OSError:
                    continue
                candidates.append((fit, modified, path))
    if not candidates:
        return None, []
    key: Callable[[tuple[float, float, Path]], tuple[float, float]] = (
        (lambda c: (c[1], c[0])) if newest else (lambda c: (c[0], c[1]))
    )
    candidates.sort(key=key, reverse=True)
    return candidates[0][2], [c[2] for c in candidates[1:4]]


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
        paragraphs = re.split(r"</w:p>", xml)
        return "\n".join(
            "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", part)) for part in paragraphs
        ).strip()
    data = path.read_bytes()[:MAX_BYTES]
    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class ReadFileTool:
    """Finds and reads a local file; ``page`` and ``max_chars`` serve the text loop."""

    def __init__(
        self, projects: Callable[[], list[Project]] = recent_projects, max_chars: int = 8000
    ) -> None:
        self._projects = projects
        self.max_chars = max_chars
        self.page: Page | None = None
        self.path: Path | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_file",
            description=(
                "Find and read a local file (text, code, Markdown, PDF, Word) by its name, "
                "e.g. 'README' in project 'flay' or 'Rechnung' on the desktop."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "File name or words of it, e.g. 'README', 'Rechnung pdf'",
                    },
                    "where": {
                        "type": "string",
                        "description": "Project name or place (Schreibtisch, Downloads, "
                        "Dokumente); empty if not said",
                    },
                },
                "required": ["name"],
            },
        )

    def announcement(self, arguments: dict[str, Any]) -> dict[Any, str]:
        return speech("Ich suche die Datei.", "I'm looking for the file.")

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        name = str(arguments.get("name") or "").strip()
        where = str(arguments.get("where") or "").strip()
        self.page = self.path = None
        if not name and not where:
            return ToolResult(
                "No file named.",
                ok=False,
                say=speech("Welche Datei soll ich lesen?", "Which file should I read?"),
            )
        projects = await asyncio.to_thread(self._projects)
        path, others = await asyncio.to_thread(find_file, name, where, projects)
        if path is None:
            return ToolResult(
                f"No file matching '{name}' found ({
                    where or 'projects, desktop, downloads, documents'
                }).",
                ok=False,
                say=speech(
                    f"Ich finde keine Datei {name}.".replace("  ", " "),
                    f"I can't find a file {name}.".replace("  ", " "),
                ),
            )
        try:
            text = await asyncio.to_thread(read_text, path)
        except Exception as exc:  # broken PDF, no permission
            log.info("Datei nicht lesbar: %s (%s)", path, exc)
            return ToolResult(
                f"{path} could not be read.",
                ok=False,
                say=speech(f"Ich kann {path.name} nicht lesen.", f"I can't read {path.name}."),
            )
        self.path = path
        self.page = Page(path.as_uri(), path.name, text.strip(), (), "file")
        others_text = " Other matches: " + ", ".join(str(p) for p in others) + "." if others else ""
        return ToolResult(
            f"Read {path}.{others_text}\n" + describe(self.page, name, self.max_chars, 0)[0],
            action=f"Datei gelesen: {path} ({len(text)} Zeichen)",
        )

    async def follow(self, link: Link) -> Page | None:
        return None

    async def search(self, query: str) -> Page | None:
        return None
