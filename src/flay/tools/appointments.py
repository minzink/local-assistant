"""calendar: the user's appointments, read locally ("Was steht heute an?").

macOS: EventKit, i.e. every calendar the Calendar app shows (iCloud, Exchange, Google …);
it asks once for access. Windows: classic Outlook over COM (the new Outlook has no such
interface). Nothing leaves the computer; nothing is changed.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any

from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech

log = logging.getLogger(__name__)

MAX_LISTED = 15
ACCESS_SECONDS = 60


@dataclass(frozen=True, slots=True)
class Appointment:
    title: str
    start: datetime
    end: datetime
    all_day: bool = False
    location: str = ""
    calendar: str = ""

    def describe(self) -> str:
        when = (
            f"{self.start:%a %d.%m.} all day"
            if self.all_day
            else f"{self.start:%a %d.%m. %H:%M}–{self.end:%H:%M}"
        )
        where = f", {self.location}" if self.location else ""
        return f"{when}: {self.title}{where}"


class CalendarAccessError(PermissionError):
    pass


def span(when: str, now: datetime) -> tuple[datetime, datetime]:
    """The period a request means: today, tomorrow, week, next (24 h ahead), or a date."""
    today = datetime.combine(now.date(), time())
    when = when.strip().lower()
    if when == "tomorrow":
        return today + timedelta(days=1), today + timedelta(days=2)
    if when == "week":
        return now, today + timedelta(days=7)
    if when == "next":
        return now, now + timedelta(days=14)
    if found := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", when):
        year, month, number = (int(g) for g in found.groups())
        day = datetime(year, month, number)
        return day, day + timedelta(days=1)
    return today, today + timedelta(days=1)  # today


def mac_appointments(start: datetime, end: datetime) -> list[Appointment]:
    import EventKit
    from Foundation import NSDate

    store = EventKit.EKEventStore.alloc().init()
    status = EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeEvent)
    full = getattr(EventKit, "EKAuthorizationStatusFullAccess", 3)
    if status != full and status != getattr(EventKit, "EKAuthorizationStatusAuthorized", 3):
        if status in (1, 2):  # restricted, denied
            raise CalendarAccessError("calendar access denied")
        granted = threading.Event()
        allowed: list[bool] = []

        def answered(ok: bool, error: Any) -> None:
            allowed.append(bool(ok))
            granted.set()

        request = getattr(store, "requestFullAccessToEventsWithCompletion_", None)
        if request is not None:
            request(answered)
        else:
            store.requestAccessToEntityType_completion_(EventKit.EKEntityTypeEvent, answered)
        granted.wait(ACCESS_SECONDS)
        if not allowed or not allowed[0]:
            raise CalendarAccessError("calendar access not granted")
        store = EventKit.EKEventStore.alloc().init()
    predicate = store.predicateForEventsWithStartDate_endDate_calendars_(
        NSDate.dateWithTimeIntervalSince1970_(start.timestamp()),
        NSDate.dateWithTimeIntervalSince1970_(end.timestamp()),
        None,
    )
    found = []
    for event in store.eventsMatchingPredicate_(predicate) or []:
        found.append(
            Appointment(
                str(event.title() or ""),
                datetime.fromtimestamp(event.startDate().timeIntervalSince1970()),
                datetime.fromtimestamp(event.endDate().timeIntervalSince1970()),
                bool(event.isAllDay()),
                str(event.location() or ""),
                str(event.calendar().title() or "") if event.calendar() else "",
            )
        )
    return found


def outlook_appointments(start: datetime, end: datetime) -> list[Appointment]:
    import comtypes.client

    outlook = comtypes.client.CreateObject("Outlook.Application")
    folder = outlook.GetNamespace("MAPI").GetDefaultFolder(9)  # olFolderCalendar
    items = folder.Items
    items.IncludeRecurrences = True
    items.Sort("[Start]")
    query = f"[Start] < '{end:%m/%d/%Y %H:%M}' AND [End] > '{start:%m/%d/%Y %H:%M}'"
    found = []
    for item in items.Restrict(query):
        found.append(
            Appointment(
                str(item.Subject or ""),
                _plain(item.Start),
                _plain(item.End),
                bool(item.AllDayEvent),
                str(item.Location or ""),
                "Outlook",
            )
        )
        if len(found) > 100:
            break
    return found


def _plain(value: Any) -> datetime:
    """A COM date (aware datetime) as local naive time."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return datetime.fromisoformat(str(value)[:19])


def system_appointments(start: datetime, end: datetime) -> list[Appointment]:
    if sys.platform == "darwin":
        return mac_appointments(start, end)
    if sys.platform == "win32":
        return outlook_appointments(start, end)
    return []


class CalendarTool:
    def __init__(
        self,
        read: Callable[[datetime, datetime], Sequence[Appointment]] = system_appointments,
        hidden: Sequence[str] = (),
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._read = read
        self._hidden = {h.casefold() for h in hidden}
        self._now = now

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calendar",
            description=(
                "Read the user's appointments. 'when': today, tomorrow, week (next 7 days), "
                "next (the next upcoming ones) or a date as YYYY-MM-DD. Read-only."
            ),
            parameters={
                "type": "object",
                "properties": {"when": {"type": "string", "description": "today, tomorrow, …"}},
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        when = str(arguments.get("when") or "today")
        now = self._now()
        start, end = span(when, now)
        try:
            found = await asyncio.to_thread(self._read, start, end)
        except CalendarAccessError:
            return ToolResult(
                "No access to the calendar.",
                ok=False,
                action="Kalender: kein Zugriff",
                say=speech(
                    "Ich darf den Kalender nicht lesen. Erlaube es in den Systemeinstellungen "
                    "unter Datenschutz, Kalender.",
                    "I'm not allowed to read the calendar. Allow it in System Settings under "
                    "Privacy, Calendars.",
                ),
            )
        shown = sorted(
            (a for a in found if a.calendar.casefold() not in self._hidden and a.end > now),
            key=lambda a: (not a.all_day, a.start),
        )
        if when == "next":
            shown = [a for a in shown if not a.all_day][:3] or shown[:3]
        if not shown:
            return ToolResult(
                f"No appointments ({when}, {start:%Y-%m-%d} to {end:%Y-%m-%d}).",
                action=f"Kalender ({when}): keine Termine",
            )
        listed = "; ".join(a.describe() for a in shown[:MAX_LISTED])
        more = f" (+{len(shown) - MAX_LISTED} more)" if len(shown) > MAX_LISTED else ""
        return ToolResult(
            f"Appointments ({when}, now is {now:%a %d.%m. %H:%M}): {listed}{more}",
            action=f"Kalender ({when}): {len(shown)} Termine",
        )
