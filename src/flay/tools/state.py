"""What is open on the computer right now, as one short line for the model.

Without it the model guesses ("KEWO is not open") and picks project names it has never
seen. The state is read fresh for every request, in parallel to speech recognition.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flay.tools.claude_code import Tmux
    from flay.tools.phpstorm_mcp import PhpStormMcp

log = logging.getLogger(__name__)


class ComputerState:
    def __init__(self, phpstorm: PhpStormMcp | None = None, tmux: Tmux | None = None) -> None:
        self._phpstorm = phpstorm
        self._tmux = tmux

    async def describe(self) -> str | None:
        projects, sessions = await asyncio.gather(self._projects(), self._sessions())
        parts = []
        if projects:
            parts.append(f"Open in PhpStorm: {', '.join(projects)}.")
        if sessions:
            parts.append(f"Claude Code running for: {', '.join(sessions)}.")
        return " ".join(parts) or None

    async def _projects(self) -> list[str]:
        if self._phpstorm is None:
            return []
        try:
            return [path.name for path in await self._phpstorm.open_projects()]
        except Exception as exc:
            log.debug("Offene Projekte nicht lesbar: %s", exc)
            return []

    async def _sessions(self) -> list[str]:
        if self._tmux is None or not self._tmux.available:
            return []
        return [
            f"{s.project} ({'shown in PhpStorm' if s.attached else 'in the background'})"
            for s in await self._tmux.sessions()
        ]
