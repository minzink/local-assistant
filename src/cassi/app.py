"""Which program runs and where it is installed: Cassi itself, or an adapter's own command
(``flay``), which is Cassi's command line under another name.

Installing, repairing, updating and uninstalling (``cassi.maintenance``) work on this: the uv
project the program runs from (its ``.venv``), and the local-assistant checkout whose
``scripts/`` set everything up.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

_name = "cassi"


@dataclass(frozen=True, slots=True)
class App:
    # The command and Python package: "cassi", or an adapter's like "flay".
    name: str
    # The uv project the program runs from (its .venv lives there); None if installed otherwise.
    project: Path | None
    # The local-assistant checkout with scripts/setup.*; None if Cassi is not run from source.
    source: Path | None

    @property
    def title(self) -> str:
        return self.name.capitalize()

    @property
    def version(self) -> str:
        try:
            return metadata.version(self.name)
        except metadata.PackageNotFoundError:
            return "?"

    def command(self, *args: str) -> list[str]:
        """This program with ``args``, run by the current Python (works without PATH)."""
        return [sys.executable, "-m", self.name, *args]

    def repositories(self) -> list[Path]:
        """Git checkouts to update: the project and, if it is another one, Cassi's source."""
        roots: list[Path] = []
        for root in (self.project, self.source):
            if root is not None and (root / ".git").exists() and root not in roots:
                roots.append(root)
        return roots


def set_name(name: str) -> None:
    """Called by the command line: ``flay`` runs Cassi's command line under its own name."""
    global _name
    _name = name


def current() -> App:
    prefix = Path(sys.prefix)
    project = prefix.parent if (prefix.parent / "pyproject.toml").is_file() else None
    source: Path | None = Path(__file__).resolve().parents[2]
    if source is None or not (source / "scripts" / "setup.sh").is_file():
        source = None
    return App(_name, project, source)
