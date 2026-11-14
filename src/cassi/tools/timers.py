"""Timers and reminders: "in 20 Minuten", "um 15 Uhr", "was steht an?", "lösch den Timer"."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from cassi.llm.base import ToolSpec
from cassi.notifications import Timer, Timers
from cassi.tools.base import ToolResult, speech

MAX_MINUTES = 7 * 24 * 60


def due_time(minutes: float | None, at: str | None, now: datetime) -> datetime | None:
    """When a timer is up: in ``minutes`` or at ``at`` ("15:30", "15 Uhr"; tomorrow if past)."""
    if minutes is not None and 0 < minutes <= MAX_MINUTES:
        return now + timedelta(minutes=minutes)
    if at:
        match = re.search(r"(\d{1,2})(?:[:.h ](\d{2}))?", at)
        if match is None:
            return None
        hour, minute = int(match[1]), int(match[2] or 0)
        if hour > 23 or minute > 59:
            return None
        due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return due if due > now else due + timedelta(days=1)
    return None


def _clock(due: datetime, now: datetime) -> str:
    day = (
        ""
        if due.date() == now.date()
        else "morgen "
        if due.date() == (now + timedelta(days=1)).date()
        else f"am {due:%d.%m.} "
    )
    return f"{day}um {due:%H:%M} Uhr"


class TimerTool:
    def __init__(
        self, timers: Timers, now: Callable[[], datetime] = lambda: datetime.now().astimezone()
    ) -> None:
        self._timers = timers
        self._now = now

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="timer",
            description=(
                "Set a timer or reminder ('minutes' from now or 'at' a time like '15:30', with "
                "an optional 'text' to remind of), 'list' them or 'cancel' (text filters)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["set", "list", "cancel"]},
                    "minutes": {"type": "number"},
                    "at": {"type": "string", "description": "e.g. '15:30'"},
                    "text": {"type": "string", "description": "What to remind of"},
                },
                "required": ["action"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = str(arguments.get("action") or "set")
        text = str(arguments.get("text") or "").strip()
        now = self._now()
        if action == "list":
            return self._list(now)
        if action == "cancel":
            gone = self._timers.cancel(text or None)
            if not gone:
                return ToolResult(
                    "No such timer.",
                    ok=False,
                    say=speech("So einen Timer gibt es nicht.", "There is no such timer."),
                )
            return ToolResult(
                f"Cancelled {len(gone)} timer(s).",
                action=f"Timer gelöscht: {len(gone)}",
                say=speech(
                    "Erledigt, der Timer ist gelöscht."
                    if len(gone) == 1
                    else f"Erledigt, {len(gone)} Timer sind gelöscht.",
                    "Done, cancelled.",
                ),
            )
        minutes = arguments.get("minutes")
        due = due_time(
            float(minutes) if isinstance(minutes, (int, float)) else None,
            str(arguments.get("at") or "") or None,
            now,
        )
        if due is None:
            return ToolResult(
                "Give 'minutes' or 'at'.",
                ok=False,
                say=speech("Wann soll ich dich erinnern?", "When should I remind you?"),
            )
        self._timers.add(due.timestamp(), text)
        when_de = _clock(due, now)
        minutes_left = round((due - now).total_seconds() / 60)
        when_de = (
            f"in {minutes_left} Minuten"
            if minutes_left < 60 and not arguments.get("at")
            else when_de
        )
        what_de = f" an {text}" if text else ""
        return ToolResult(
            f"Timer set for {due:%Y-%m-%d %H:%M}" + (f": {text}" if text else "") + ".",
            action=f"Timer: {due:%d.%m. %H:%M}" + (f" – {text}" if text else ""),
            say=speech(
                f"Okay, ich erinnere dich {when_de}{what_de}.",
                f"Okay, I'll remind you at {due:%H:%M}" + (f" about {text}." if text else "."),
            ),
        )

    def _list(self, now: datetime) -> ToolResult:
        pending: list[Timer] = self._timers.pending
        if not pending:
            return ToolResult(
                "No timers.", say=speech("Es läuft kein Timer.", "No timers are set.")
            )
        lines = [
            f"{datetime.fromtimestamp(t.due, now.tzinfo):%Y-%m-%d %H:%M}"
            + (f": {t.text}" if t.text else "")
            for t in pending
        ]
        return ToolResult("Timers: " + "; ".join(lines) + ". Tell the user briefly.")
