"""Reading along: the text of the front window, kept current in the background.

A screenshot costs the model ~1000 tokens (seconds) per request; the text of the front
window, read beforehand by the system's OCR, costs a few hundred and no waiting for the
picture. The model understands less of it (no layout, no images) but answers faster.

Once a second a small grey thumbnail of the front window is compared with the last one
(~25 ms); only when it changed – and at most every few seconds – the window is read with
OCR (~0.1–0.3 s). Private windows and a paused screen are never read. The text stays in
memory: it goes along with a request that is about the screen, and is never stored.

Off by default (``tools.screen.watch``); switched in the tray menu or the management UI,
which write ``screen.json`` in the data folder – read again when it changes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

    from cassi.tools.ocr import TextBox
    from cassi.tools.screen import Window

log = logging.getLogger(__name__)

THUMBNAIL = (64, 40)
# Mean grey difference (0–255) from which a thumbnail counts as changed.
CHANGE = 2.0
# Older text is not worth sending: the screen may show something else by now.
FRESH_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class Reading:
    app: str
    title: str
    text: str
    at: float  # time.monotonic()


@dataclass
class WatchSwitch:
    """On or off, for the whole process."""

    enabled: bool = False


WATCH = WatchSwitch()


def switch_file(data_dir: Path) -> Path:
    return data_dir / "screen.json"


def read_switch(path: Path) -> bool | None:
    """The switch as last set in the tray or management UI; None if never."""
    try:
        value = json.loads(path.read_text("utf-8")).get("watch")
    except (OSError, ValueError, AttributeError):
        return None
    return None if value is None else bool(value)


def write_switch(path: Path, on: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"watch": on}), "utf-8")


# The request is about what the screen shows: "Was heißt diese Fehlermeldung?", "Antworte
# auf die Mail", "Was steht da?", "Fass den Artikel zusammen".
_ABOUT_SCREEN = re.compile(
    r"\b(?:bildschirm|screen|fenster|window|seite|page|tab|artikel|article|mail|e-mail|"
    r"nachricht|message|fehler\w*|error|meldung|warnung|warning|dialog|text|code|"
    r"steht|stehen|zeigt|angezeigt|sehe|siehst|see|says|shows|"
    r"hier|da oben|da unten|das da|dies\w*|this|that one)\b",
    re.IGNORECASE,
)
# "Wie sieht das aus?", "Welche Farbe …": the look counts, the picture is needed.
LOOKS = re.compile(
    r"\b(?:sieht|aussehen|aussieht|look\w*|farbe\w*|colou?r\w*|bild\w*|image|picture|foto|"
    r"photo|grafik|diagramm|chart|layout|design|icon)\b",
    re.IGNORECASE,
)


def about_screen(text: str) -> bool:
    return _ABOUT_SCREEN.search(text) is not None


def reading_order(boxes: list[TextBox], limit: int) -> str:
    """OCR lines top to bottom, left to right; repeated lines once; at most ``limit`` chars."""
    rows = sorted(_lines_only(boxes), key=lambda b: (round(b.frame[1] / 8), b.frame[0]))
    seen: set[str] = set()
    lines = []
    size = 0
    for box in rows:
        line = box.text.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
        size += len(line) + 1
        if size >= limit:
            break
    return "\n".join(lines)[:limit]


def _lines_only(boxes: list[TextBox]) -> list[TextBox]:
    """Without the single words OCR reports besides their line (Windows, some macOS text)."""
    kept = []
    for box in boxes:
        x, y = box.centre
        inside = any(
            other is not box
            and len(other.text) > len(box.text)
            and box.text in other.text
            and other.frame[0] <= x <= other.frame[0] + other.frame[2]
            and other.frame[1] <= y <= other.frame[1] + other.frame[3]
            for other in boxes
        )
        if not inside:
            kept.append(box)
    return kept


def describe(reading: Reading | None, now: float | None = None) -> str | None:
    """The reading for the model, if it is fresh."""
    if reading is None or not reading.text:
        return None
    age = (time.monotonic() if now is None else now) - reading.at
    if age > FRESH_SECONDS:
        return None
    where = " – ".join(p for p in (reading.app, reading.title) if p)
    return (
        f"(Text read from the front window ({where}) {age:.0f} s ago, by text recognition – "
        f"no layout or images:\n«{reading.text}»)"
    )


def _thumbnail(image: Image.Image) -> Any:
    import numpy as np

    return np.asarray(image.convert("L").resize(THUMBNAIL), dtype=np.int16)


def _window_image(window: Window) -> Image.Image:
    """The front window's part of its monitor (macOS captures whole displays only)."""
    from PIL import Image

    from cassi.tools.screen import _mss

    x, y, width, height = window.bounds
    with _mss() as screen:
        monitors = screen.monitors[1:] or screen.monitors
        centre = (x + width / 2, y + height / 2)
        monitor = next(
            (
                m
                for m in monitors
                if m["left"] <= centre[0] < m["left"] + m["width"]
                and m["top"] <= centre[1] < m["top"] + m["height"]
            ),
            monitors[0],
        )
        shot = screen.grab(monitor)
    image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    scale = image.width / monitor["width"]
    left = max(0, (x - monitor["left"]) * scale)
    top = max(0, (y - monitor["top"]) * scale)
    right = min(image.width, (x - monitor["left"] + width) * scale)
    bottom = min(image.height, (y - monitor["top"] + height) * scale)
    if right <= left or bottom <= top:
        return image
    return image.crop((round(left), round(top), round(right), round(bottom)))


class ScreenWatcher:
    def __init__(
        self,
        switch: Path | None = None,
        default: bool = False,
        interval: float = 1.0,
        ocr_seconds: float = 2.0,
        chars: int = 1500,
    ) -> None:
        self._switch = switch
        self._default = default
        self.interval = interval
        self.ocr_seconds = ocr_seconds
        self.chars = chars
        self.reading: Reading | None = None
        self._last: Any = None
        self._last_window: tuple[str, str] | None = None
        self._last_ocr = 0.0
        self._stamp: float | None = None
        WATCH.enabled = default
        self._sync()

    def _sync(self) -> None:
        """Take the switch from screen.json when it changed (tray, management UI)."""
        if self._switch is None:
            return
        try:
            stamp = self._switch.stat().st_mtime
        except OSError:
            return
        if stamp == self._stamp:
            return
        self._stamp = stamp
        value = read_switch(self._switch)
        WATCH.enabled = self._default if value is None else value

    def current(self) -> str | None:
        """The front window's text for the model, if reading along and fresh."""
        return describe(self.reading) if WATCH.enabled else None

    async def run(self) -> None:
        from cassi.tools import ocr

        if not ocr.available():
            log.info("Mitlesen nicht möglich: keine Texterkennung auf diesem System")
            return
        while True:
            await asyncio.sleep(self.interval)
            self._sync()
            if not WATCH.enabled:
                self.reading = None
                continue
            try:
                await asyncio.to_thread(self.step)
            except Exception as exc:  # no permission, a window closing mid-read …
                log.debug("Mitlesen: %s", exc)

    def step(self, now: float | None = None) -> None:
        """One look: read the front window again if it changed (blocking)."""
        from cassi.tools import ocr
        from cassi.tools.screen import PRIVACY, is_own, is_private, list_windows, private_window

        now = time.monotonic() if now is None else now
        windows = [w for w in list_windows() if not is_own(w)]
        front = windows[0] if windows else None
        if (
            front is None
            or PRIVACY.paused
            or is_private(front)
            or private_window(front.bounds, windows) is not None
        ):
            self.reading, self._last = None, None
            return
        image = _window_image(front)
        thumbnail = _thumbnail(image)
        window = (front.app, front.title)
        changed = (
            self._last is None
            or window != self._last_window
            or thumbnail.shape != self._last.shape
            or float(abs(thumbnail - self._last).mean()) >= CHANGE
        )
        if not changed:
            if self.reading is not None and self.reading.app == front.app:
                self.reading = replace(self.reading, at=now)  # still what is shown
            return
        if window == self._last_window and now - self._last_ocr < self.ocr_seconds:
            return  # changes again next second: read then
        self._last, self._last_window, self._last_ocr = thumbnail, window, now
        text = reading_order(ocr.read_image(image), self.chars)
        self.reading = Reading(front.app, front.title, text, now)
