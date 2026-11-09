"""Forward flay's log records to the UI's debug window."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime

from flay.core.events import EventBus, LogEntry

# Loggers of the tools report what was done on the computer.
ACTION_LOGGER = "flay.tools"


def to_entry(record: logging.LogRecord) -> LogEntry:
    message = record.getMessage()
    if record.exc_info and record.exc_info[1] is not None:
        message = f"{message} ({type(record.exc_info[1]).__name__}: {record.exc_info[1]})"
    return LogEntry(
        time=datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
        level=record.levelname,
        source=record.name.removeprefix("flay."),
        message=message,
        action=record.name.startswith(ACTION_LOGGER) and record.levelno >= logging.INFO,
    )


class UILogHandler(logging.Handler):
    """Keeps the latest entries and publishes new ones on the event loop.

    Records may come from worker threads (STT, TTS, tools), so publishing is handed to the
    loop thread.
    """

    def __init__(self, events: EventBus, loop: asyncio.AbstractEventLoop, capacity: int = 500):
        super().__init__(logging.DEBUG)
        self._events = events
        self._loop = loop
        self.entries: deque[LogEntry] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = to_entry(record)
            self._loop.call_soon_threadsafe(self._publish, entry)
        except RuntimeError:  # loop already closed during shutdown
            pass
        except Exception:
            self.handleError(record)

    def _publish(self, entry: LogEntry) -> None:
        self.entries.append(entry)
        self._events.publish(entry)
