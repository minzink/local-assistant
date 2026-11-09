"""Events the assistant publishes for user interfaces (web UI, later tray icon etc.)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from flay.core.language import Language


class AssistantState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(frozen=True, slots=True)
class StateChanged:
    state: AssistantState


@dataclass(frozen=True, slots=True)
class UserMessage:
    text: str
    language: Language
    source: Literal["voice", "text"]


@dataclass(frozen=True, slots=True)
class AssistantDelta:
    """A streamed piece of the reply that is currently being generated."""

    text: str


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    text: str
    language: Language
    interrupted: bool = False


@dataclass(frozen=True, slots=True)
class MuteChanged:
    microphone: bool
    speaker: bool


@dataclass(frozen=True, slots=True)
class ListeningChanged:
    """Continuous listening was switched on or off."""

    continuous: bool


@dataclass(frozen=True, slots=True)
class WakeWordChanged:
    """Listening for "Hey Flay" was switched on or off (applies to continuous listening)."""

    enabled: bool


@dataclass(frozen=True, slots=True)
class RecordingChanged:
    """A workflow recording started (its name) or ended (None)."""

    workflow: str | None


@dataclass(frozen=True, slots=True)
class ToolUsed:
    name: str
    arguments: dict[str, Any]
    result: str
    ok: bool
    # Human-readable description of what happened on the computer (German).
    action: str | None = None


@dataclass(frozen=True, slots=True)
class LogEntry:
    """A log line of flay, shown in the UI's debug window."""

    time: str
    level: str
    source: str
    message: str
    # True for things done on the computer (tools), so the UI can filter them.
    action: bool = False


@dataclass(frozen=True, slots=True)
class Notice:
    """Something the user should know, e.g. a mute recording or "nothing understood"."""

    text: str


Event = (
    StateChanged
    | UserMessage
    | AssistantDelta
    | AssistantMessage
    | MuteChanged
    | ListeningChanged
    | WakeWordChanged
    | RecordingChanged
    | ToolUsed
    | Notice
    | LogEntry
)


class EventBus:
    """Fan-out to any number of subscribers; slow subscribers never block the assistant."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()

    def publish(self, event: Event) -> None:
        for queue in self._subscribers:
            queue.put_nowait(event)

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[Event]]:
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)
