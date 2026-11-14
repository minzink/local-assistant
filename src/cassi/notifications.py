"""Things Cassi says on its own: a timer is up, Claude is done or waits, a command finished.

Tools and watchers hand what to say to ``Announcements``; the assistant speaks it as soon
as it is not answering anything (``Assistant.announce``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from cassi.core.language import Language
from cassi.tools.base import speech, spoken

log = logging.getLogger(__name__)

Say = Mapping[Language, str]


class Announcements:
    """A channel from tools and watchers to the assistant's voice."""

    def __init__(self) -> None:
        self._listener: Callable[[Say], None] | None = None
        self._waiting: list[Say] = []

    def connect(self, listener: Callable[[Say], None]) -> None:
        self._listener = listener
        waiting, self._waiting = self._waiting, []
        for say in waiting:
            listener(say)

    def say(self, de: str, en: str) -> None:
        message = speech(de, en)
        log.info("Durchsage: %s", de)
        if self._listener is None:
            self._waiting.append(message)
        else:
            self._listener(message)


# Claude Code ------------------------------------------------------------------------------


class ClaudeState(StrEnum):
    WORKING = "working"
    ASKING = "asking"
    IDLE = "idle"


# Shown while Claude works ("esc to interrupt") and while it asks for a permission.
_WORKING = re.compile(r"esc to interrupt|esc zum abbrechen", re.IGNORECASE)
_ASKING = re.compile(
    r"do you want to (?:proceed|make this edit|create|run)|❯\s*1\.\s*yes|"
    r"would you like to|allow .* to|bestätigen",
    re.IGNORECASE,
)


def claude_state(screen: str) -> ClaudeState:
    """What Claude's terminal shows at its bottom."""
    tail = "\n".join(screen.rstrip().splitlines()[-15:])
    if _ASKING.search(tail):
        return ClaudeState.ASKING
    if _WORKING.search(tail):
        return ClaudeState.WORKING
    return ClaudeState.IDLE


class ClaudeWatcher:
    """Says when a Claude session finishes its work or starts waiting for the user."""

    def __init__(
        self,
        tmux: object,
        announcements: Announcements,
        interval: float = 3.0,
    ) -> None:
        self._tmux = tmux
        self._announcements = announcements
        self._interval = interval
        self._states: dict[str, ClaudeState] = {}
        # A session has to work this long before its end is worth a word.
        self._working_since: dict[str, float] = {}
        self.min_work_seconds = 20.0
        # Told about each permission question (and its end, with None): answer_claude.
        self.on_question: Callable[[Any, str], None] | None = None

    async def run(self) -> None:
        sessions = getattr(self._tmux, "sessions", None)
        screen = getattr(self._tmux, "screen", None)
        if not getattr(self._tmux, "available", False) or sessions is None or screen is None:
            return
        while True:
            try:
                for session in await sessions():
                    shown = await screen(session.name)
                    self.update(session.project, claude_state(shown), screen=shown,
                                session=session.name)  # fmt: skip
            except Exception as exc:  # a session ending mid-read
                log.debug("Claude-Wächter: %s", exc)
            await asyncio.sleep(self._interval)

    def update(
        self,
        project: str,
        state: ClaudeState,
        now: float | None = None,
        screen: str = "",
        session: str = "",
    ) -> None:
        now = time.monotonic() if now is None else now
        before = self._states.get(project)
        self._states[project] = state
        if state is ClaudeState.WORKING and before is not ClaudeState.WORKING:
            self._working_since[project] = now
        if before is None or before == state:
            return
        where = spoken(project)
        if before is ClaudeState.ASKING and self.on_question is not None:
            self.on_question(None, project)
        if state is ClaudeState.ASKING:
            from cassi.tools.claude_code import read_question

            question = read_question(screen, project, session) if screen else None
            if self.on_question is not None:
                self.on_question(question, project)
            if question is not None and question.question:
                self._announcements.say(
                    f"Claude in {where} fragt: {question.question} Sag ja, immer oder nein.",
                    f"Claude in {where} asks: {question.question} Say yes, always or no.",
                )
            else:
                self._announcements.say(
                    f"Claude in {where} wartet auf deine Freigabe.",
                    f"Claude in {where} is waiting for your approval.",
                )
        elif state is ClaudeState.IDLE and before is ClaudeState.WORKING:
            worked = now - self._working_since.get(project, now)
            if worked >= self.min_work_seconds:
                self._announcements.say(
                    f"Claude ist in {where} fertig.", f"Claude is done in {where}."
                )


# Timers -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Timer:
    id: int
    due: float  # epoch seconds
    text: str


# Timers missed while Cassi was not running are still mentioned if they are this recent.
MISSED_SECONDS = 12 * 3600


class Timers:
    """Timers and reminders, kept in a JSON file so they survive a restart."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self._path = path
        self._clock = clock
        self._timers: list[Timer] = self._load()

    def _load(self) -> list[Timer]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [Timer(int(t["id"]), float(t["due"]), str(t["text"])) for t in data]
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(t) for t in self._timers]
        self._path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @property
    def pending(self) -> list[Timer]:
        return sorted(self._timers, key=lambda t: t.due)

    def add(self, due: float, text: str) -> Timer:
        timer = Timer(max((t.id for t in self._timers), default=0) + 1, due, text)
        self._timers.append(timer)
        self._save()
        return timer

    def cancel(self, text: str | None = None) -> list[Timer]:
        """Cancel the timers whose text contains ``text`` (all without it)."""
        wanted = (text or "").casefold()
        gone = [t for t in self._timers if wanted in t.text.casefold()]
        self._timers = [t for t in self._timers if t not in gone]
        self._save()
        return gone

    def due(self) -> list[Timer]:
        """Timers that are up (removed); ones missed long ago are dropped silently."""
        now = self._clock()
        up = [t for t in self._timers if t.due <= now]
        if not up:
            return []
        self._timers = [t for t in self._timers if t.due > now]
        self._save()
        return [t for t in up if now - t.due <= MISSED_SECONDS]

    async def run(self, announcements: Announcements, interval: float = 1.0) -> None:
        while True:
            for timer in self.due():
                late = self._clock() - timer.due > 120
                prefix_de = "Verpasste Erinnerung: " if late else "Erinnerung: "
                prefix_en = "Missed reminder: " if late else "Reminder: "
                if timer.text:
                    announcements.say(prefix_de + timer.text, prefix_en + timer.text)
                else:
                    announcements.say("Dein Timer ist abgelaufen.", "Your timer is up.")
            await asyncio.sleep(interval)
