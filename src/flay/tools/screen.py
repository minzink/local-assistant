"""Screenshots for the vision model: a monitor, the active window or an app's window.

Everything stays on this computer. macOS needs the "Screen Recording" permission for the
app that runs flay.
"""

from __future__ import annotations

import asyncio
import io
import re
import sys
from dataclasses import dataclass
from typing import Any

from PIL import Image

from flay.core.messages import ImagePart
from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech

# A monitor's or window's left, top, width, height in screen points.
Region = tuple[int, int, int, int]

# The Flay UI in a browser is not what the user wants to show when they say "this window".
OWN_WINDOW_TITLES = ("Flay",)


@dataclass(frozen=True, slots=True)
class Window:
    id: int
    app: str
    title: str
    # x, y, width, height in screen points
    bounds: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Shot:
    """A screenshot and the part of the screen it shows, in screen points (the coordinate
    system of windows and the mouse)."""

    png: bytes
    # left, top, width, height
    region: tuple[float, float, float, float]

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """A point given as 0–1000 of the image's width and height (how Qwen-VL answers)."""
        left, top, width, height = self.region
        return left + x / 1000 * width, top + y / 1000 * height


class ScreenPermissionError(PermissionError):
    pass


def ensure_permission() -> None:
    """Without "Screen Recording", macOS hands out the wallpaper and the menu bar only –
    an image that looks fine but shows none of the windows. Ask macOS instead of guessing."""
    if sys.platform != "darwin":
        return
    import Quartz

    if not Quartz.CGPreflightScreenCaptureAccess():
        Quartz.CGRequestScreenCaptureAccess()  # shows the system prompt once
        raise ScreenPermissionError(
            "screen recording is not allowed – allow 'Screen Recording' for the app running "
            "flay (System Settings → Privacy & Security) and restart it"
        )


def _finish(image: Image.Image, max_size: int) -> bytes:
    low, high = image.convert("L").getextrema()
    if low == high:
        # macOS returns a uniform image instead of an error without the permission.
        raise PermissionError(
            "the screenshot is empty – allow 'Screen Recording' for the app running flay "
            "(System Settings → Privacy & Security) and restart it"
        )
    # Vision models work on a few hundred tokens per image; full Retina size only costs time.
    image.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _mss() -> Any:
    import mss

    # mss 10 renamed the entry point; older versions only have the lowercase one.
    factory = getattr(mss, "MSS", None) or mss.mss
    return factory()


def _grab(where: dict[str, int] | int) -> tuple[Image.Image, dict[str, int]]:
    with _mss() as screen:
        if isinstance(where, int):
            monitors = screen.monitors  # [0] = all monitors combined, [1] = primary, ...
            region: dict[str, int] = monitors[where if where < len(monitors) else 1]
        else:
            region = where
        shot = screen.grab(region)
    return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"), region


def monitor_bounds(monitor: int) -> tuple[float, float, float, float]:
    """A monitor's area in screen points (0 = all monitors together)."""
    with _mss() as screen:
        monitors = screen.monitors
        m = monitors[monitor if monitor < len(monitors) else 1]
    return (m["left"], m["top"], m["width"], m["height"])


def monitor_count() -> int:
    with _mss() as screen:
        return len(screen.monitors) - 1


def monitor_regions() -> list[Region]:
    """Each monitor's left, top, width, height; index 0 is monitor 1 (the main one)."""
    with _mss() as screen:
        return [(m["left"], m["top"], m["width"], m["height"]) for m in screen.monitors[1:]]


def monitor_positions(regions: list[Region]) -> dict[int, str]:
    """Where each monitor stands ("left", "right", "top", ...), by number from 1.

    Side by side they are named left to right, one above the other top to bottom; a
    single monitor gets no name.
    """
    if len(regions) < 2:
        return {}
    numbers = range(1, len(regions) + 1)
    spread_x = max(r[0] for r in regions) - min(r[0] for r in regions)
    spread_y = max(r[1] for r in regions) - min(r[1] for r in regions)
    axis, names = (0, ("left", "right")) if spread_x >= spread_y else (1, ("top", "bottom"))
    ordered = sorted(numbers, key=lambda n: regions[n - 1][axis])
    middle = "middle" if len(ordered) == 3 else None
    positions = {ordered[0]: names[0], ordered[-1]: names[1]}
    if middle:
        positions[ordered[1]] = middle
    return positions


def window_monitor(window: Window, regions: list[Region]) -> int:
    """Number of the monitor the window's centre is on (1 if none)."""
    x, y, width, height = window.bounds
    cx, cy = x + width / 2, y + height / 2
    return next(
        (
            n
            for n, (left, top, w, h) in enumerate(regions, start=1)
            if left <= cx < left + w and top <= cy < top + h
        ),
        1,
    )


# "Was ist das hier?": the area around the mouse pointer, in screen points. Ollama brings
# every image to about the same number of tokens (~1080 for qwen3-vl, measured), so this is
# not faster than a whole monitor – but the model sees what is meant, at about 1.5 times the
# detail, without the rest of the screen to distract it.
POINTER_AREA = (640, 400)
POINTER_MAX_SIZE = 768


def pointer_position() -> tuple[float, float]:
    from pynput.mouse import Controller

    from flay.tools import pointer  # noqa: F401 – Windows: physical pixels, like mss

    x, y = Controller().position
    return float(x), float(y)


def area_around(
    point: tuple[float, float], regions: list[Region], size: tuple[int, int] = POINTER_AREA
) -> tuple[float, float, float, float]:
    """``size`` centred on ``point``, moved inside the point's monitor where it would stick
    out (left, top, width, height)."""
    x, y = point
    left, top, width, height = next(
        (r for r in regions if r[0] <= x < r[0] + r[2] and r[1] <= y < r[1] + r[3]),
        regions[0] if regions else (0, 0, size[0], size[1]),
    )
    w, h = min(size[0], width), min(size[1], height)
    area_left = min(max(x - w / 2, left), left + width - w)
    area_top = min(max(y - h / 2, top), top + height - h)
    return (area_left, area_top, w, h)


# Enough for what is open; a long list would only cost prompt time.
MAX_LISTED_WINDOWS = 15
MAX_TITLE_CHARS = 80


def describe_windows(windows: list[Window], regions: list[Region]) -> str | None:
    """The visible windows per monitor, for the model: "Monitor 1 (main, left): ..."."""
    shown = [w for w in windows if not is_own(w)][:MAX_LISTED_WINDOWS]
    if not shown:
        return None
    positions = monitor_positions(regions)
    lines = []
    for number in range(1, max(len(regions), 1) + 1):
        on_monitor = [w for w in shown if window_monitor(w, regions) == number]
        if not on_monitor:
            continue
        labels = []
        for window in on_monitor:
            title = window.title[:MAX_TITLE_CHARS]
            label = window.app if not title or title == window.app else f"{window.app}: {title}"
            labels.append(label + (" (front)" if window is shown[0] else ""))
        where = ", ".join(
            part for part in ("main" if number == 1 else "", positions.get(number, "")) if part
        )
        heading = f"Monitor {number}" + (f" ({where})" if where else "")
        lines.append(f"{heading}: " + "; ".join(labels))
    return "Open windows – " + " | ".join(lines) + "."


def capture_png(monitor: int, max_size: int) -> bytes:
    return capture_monitor(monitor, max_size).png


def capture_monitor(monitor: int, max_size: int) -> Shot:
    ensure_permission()
    image, region = _grab(monitor)
    png = _finish(image, max_size)  # raises first if the picture is blank
    return Shot(png, (region["left"], region["top"], region["width"], region["height"]))


# Windows ---------------------------------------------------------------------------------


def list_windows() -> list[Window]:
    """Visible app windows, frontmost first."""
    if sys.platform == "darwin":
        return _mac_windows()
    if sys.platform == "win32":
        return _windows_windows()
    return []


def _mac_windows() -> list[Window]:
    import Quartz

    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    windows = []
    for info in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []:
        if info.get("kCGWindowLayer") != 0:
            continue  # menu bar, dock, overlays
        b = info.get("kCGWindowBounds", {})
        bounds = (
            int(b.get("X", 0)),
            int(b.get("Y", 0)),
            int(b.get("Width", 0)),
            int(b.get("Height", 0)),
        )
        if bounds[2] < 50 or bounds[3] < 50:
            continue
        windows.append(
            Window(
                id=int(info["kCGWindowNumber"]),
                app=str(info.get("kCGWindowOwnerName", "")),
                title=str(info.get("kCGWindowName", "") or ""),
                bounds=bounds,
            )
        )
    return windows


def _windows_windows() -> list[Window]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined, unused-ignore]
    windows: list[Window] = []
    hwnd = user32.GetForegroundWindow()
    order = [hwnd]
    next_hwnd = user32.GetWindow(user32.GetDesktopWindow(), 5)  # GW_CHILD: top of z-order
    while next_hwnd:
        order.append(next_hwnd)
        next_hwnd = user32.GetWindow(next_hwnd, 2)  # GW_HWNDNEXT
    seen = set()
    for handle in order:
        if handle in seen or not user32.IsWindowVisible(handle):
            continue
        seen.add(handle)
        length = user32.GetWindowTextLengthW(handle)
        if not length:
            continue
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        rect = wintypes.RECT()
        user32.GetWindowRect(handle, ctypes.byref(rect))
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width < 50 or height < 50:
            continue
        title = buffer.value
        # Window titles usually end with the app name, e.g. "file.php – PhpStorm".
        app = re.split(r"\s[-–—]\s", title)[-1]
        windows.append(Window(int(handle), app, title, (rect.left, rect.top, width, height)))
    return windows


def is_own(window: Window) -> bool:
    return any(window.title == t or window.title.startswith(f"{t} ") for t in OWN_WINDOW_TITLES)


def find_window(query: str | None, windows: list[Window]) -> Window | None:
    """``None``/"active": the frontmost window; otherwise the first matching app or title."""
    candidates = [w for w in windows if not is_own(w)]
    if not query or query.lower() in ("active", "aktiv", "aktuell", "current"):
        return candidates[0] if candidates else None
    wanted = query.lower()
    matching = [w for w in windows if wanted in w.app.lower() or wanted in w.title.lower()]
    # Flay's own UI is skipped – unless it is the only window of the app the user names
    # (e.g. Firefox showing the Flay tab).
    return next((w for w in matching if not is_own(w)), matching[0] if matching else None)


def capture_region(region: tuple[float, float, float, float], max_size: int) -> Shot:
    """Any part of the screen, in screen points (e.g. a closer look around a point)."""
    ensure_permission()
    left, top, width, height = (round(v) for v in region)
    image, _ = _grab({"left": left, "top": top, "width": width, "height": height})
    return Shot(_finish(image, max_size), (left, top, width, height))


def capture_window_png(window: Window, max_size: int) -> bytes:
    return capture_window(window, max_size).png


def capture_window(window: Window, max_size: int) -> Shot:
    """The window's area of the screen. Windows lying on top of it are captured too.

    The whole monitor is grabbed and cropped: on current macOS only full-display capture
    still works (per-window and region capture use APIs Apple retired for
    ScreenCaptureKit), and it behaves the same on Windows.
    """
    ensure_permission()
    x, y, width, height = window.bounds
    with _mss() as screen:
        monitors = screen.monitors[1:]
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
    scale = image.width / monitor["width"]  # 2 on Retina displays
    box = (
        max(0, round((x - monitor["left"]) * scale)),
        max(0, round((y - monitor["top"]) * scale)),
        min(image.width, round((x - monitor["left"] + width) * scale)),
        min(image.height, round((y - monitor["top"] + height) * scale)),
    )
    # The part actually captured: a window reaching past the monitor's edge is cut off.
    region = (
        monitor["left"] + box[0] / scale,
        monitor["top"] + box[1] / scale,
        (box[2] - box[0]) / scale,
        (box[3] - box[1]) / scale,
    )
    return Shot(_finish(image.crop(box), max_size), region)


def is_frontmost(window: Window, windows: list[Window]) -> bool:
    visible = [w for w in windows if not is_own(w)]
    return bool(visible) and visible[0].id == window.id


# Understanding the request ----------------------------------------------------------------

_ORDINALS = {
    "erst": 1, "first": 1, "haupt": 1, "main": 1,
    "zweit": 2, "second": 2, "dritt": 3, "third": 3,
}  # fmt: skip
_MONITOR = re.compile(
    r"\b(erst|zweit|dritt|haupt|first|second|third|main)\w*\s+(?:bildschirm|monitor|screen|display)"
    r"|\b(?:bildschirm|monitor|screen|display)\s+(\d)\b",
    re.IGNORECASE,
)
_SIDES = {
    "link": "left", "left": "left", "recht": "right", "right": "right",
    "mittler": "middle", "middle": "middle", "ober": "top", "top": "top",
    "unter": "bottom", "bottom": "bottom", "ander": "other", "other": "other",
}  # fmt: skip
_SIDE = re.compile(
    r"\b(link|recht|mittler|ober|unter|ander|left|right|middle|top|bottom|other)\w*\s+"
    r"(?:bildschirm|monitor|screen|display)",
    re.IGNORECASE,
)


def _monitor_at(side: str, regions: list[Region]) -> int | None:
    """Number of the monitor at ``side``; "other" is the second of exactly two."""
    if side == "other":
        return 2 if len(regions) == 2 else None
    return next((n for n, pos in monitor_positions(regions).items() if pos == side), None)


_WINDOW = re.compile(r"\b(?:fenster|window)\b", re.IGNORECASE)
# "Was ist das hier?", "unter der Maus": what the mouse points at.
AT_POINTER = re.compile(
    r"\bdas hier\b|\bhier\b.{0,20}\b(?:maus|mauszeiger|zeiger|cursor)\b"
    r"|\b(?:unter|bei|an)\s+(?:der|dem|meiner|meinem)\s+(?:maus|mauszeiger|zeiger|cursor)\b"
    r"|\bwo\s+(?:die|meine)\s+maus\b|\bwo\s+(?:der|mein)\s+(?:mauszeiger|zeiger|cursor)\b"
    r"|\bthis here\b|\bunder\s+(?:the|my)\s+(?:mouse|cursor|pointer)\b"
    r"|\bwhere\s+(?:the|my)\s+(?:mouse|cursor|pointer)\b",
    re.IGNORECASE,
)


def screen_arguments(
    text: str, apps: list[str], regions: list[Region] | None = None
) -> dict[str, Any]:
    """Arguments for look_at_screen from what the user said, without asking the model."""
    if (match := _SIDE.search(text)) and (
        number := _monitor_at(_SIDES[match.group(1).lower()], regions or [])
    ):
        return {"monitor": number}
    if match := _MONITOR.search(text):
        number = _ORDINALS.get(match.group(1).lower()) if match.group(1) else int(match.group(2))
        return {"monitor": number}
    lowered = text.lower()
    named = next(
        (app for app in sorted(set(apps), key=len, reverse=True) if app.lower() in lowered), None
    )
    if named:
        return {"window": named}
    if _WINDOW.search(text):
        return {"window": "active"}
    if AT_POINTER.search(text):
        return {"pointer": True}
    return {}


class LookAtScreenTool:
    def __init__(self, monitor: int = 1, max_size: int = 1280) -> None:
        self._monitor = monitor
        self._max_size = max_size

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="look_at_screen",
            description=(
                "Take a screenshot and look at it. Use it when the user refers to something "
                "visible ('what do you see', 'this error'). By default the whole main screen; "
                "set 'window' to 'active' for the window the user works in, or to an app name "
                "(e.g. 'PhpStorm'); set 'monitor' for another screen (2 = second)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "window": {
                        "type": "string",
                        "description": "'active' or an app/window name; omit for the whole screen",
                    },
                    "monitor": {"type": "integer", "description": "1 = main screen, 2 = second"},
                },
            },
        )

    async def _look_at_pointer(self) -> ToolResult:
        point = await asyncio.to_thread(pointer_position)
        area = area_around(point, await asyncio.to_thread(monitor_regions))
        shot = await asyncio.to_thread(capture_region, area, POINTER_MAX_SIZE)
        left, top, width, height = shot.region
        # Where the pointer is in the picture, as Qwen-VL counts (0–1000): screenshots do
        # not show the pointer itself, and near an edge it is not in the middle.
        x = round((point[0] - left) / width * 1000)
        y = round((point[1] - top) / height * 1000)
        return ToolResult(
            "Screenshot of the area around the mouse pointer; the user points at what is at "
            f"({x}, {y}) on a 0–1000 scale of the image. It is attached as the next message.",
            images=(ImagePart(shot.png, "image/png"),),
            action=f"Bereich um den Mauszeiger aufgenommen ({len(shot.png) // 1024} KB)",
        )

    def apps(self) -> list[str]:
        """Names of apps with visible windows, for matching what the user says."""
        return [w.app for w in list_windows() if w.app]

    def monitors(self) -> list[Region]:
        return monitor_regions()

    def overview(self) -> str | None:
        """The open windows per monitor as text: names without reading them from pixels."""
        return describe_windows(list_windows(), monitor_regions())

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            return await self._look(arguments)
        except ScreenPermissionError as exc:
            return ToolResult(
                str(exc),
                ok=False,
                action="Bildschirm: keine Berechtigung für Bildschirmaufnahmen",
                say=speech(
                    "Ich darf den Bildschirm nicht aufnehmen. Erlaube es bitte in den "
                    "Systemeinstellungen unter Bildschirmaufnahme.",
                    "I'm not allowed to capture the screen. Please allow it in System "
                    "Settings under Screen Recording.",
                ),
            )

    async def _look(self, arguments: dict[str, Any]) -> ToolResult:
        if arguments.get("pointer"):
            return await self._look_at_pointer()
        window_query = arguments.get("window")
        if window_query:
            windows = await asyncio.to_thread(list_windows)
            window = find_window(str(window_query), windows)
            if window is None:
                apps = ", ".join(sorted({w.app for w in windows if w.app})) or "none"
                return ToolResult(
                    f"No window matches '{window_query}'. Visible apps: {apps}.",
                    ok=False,
                    action=f"Bildschirm: kein Fenster „{window_query}“ gefunden",
                    say=speech(
                        f"Ich sehe gerade kein Fenster von {window_query}. Liegt es vielleicht "
                        "auf einem anderen Schreibtisch oder ist minimiert?",
                        f"I can't see a {window_query} window right now. Is it on another "
                        "desktop or minimized?",
                    ),
                )
            png = await asyncio.to_thread(capture_window_png, window, self._max_size)
            label = f"{window.app}{f' – {window.title}' if window.title else ''}"
            covered = (
                ""
                if is_frontmost(window, windows)
                else " It is not the front window, so other windows may cover parts of it."
            )
            return ToolResult(
                f"Screenshot of the window '{label}'; it is attached as the next message.{covered}",
                images=(ImagePart(png, "image/png"),),
                action=f"Fenster aufgenommen: {label} ({len(png) // 1024} KB)",
            )
        monitor = int(arguments.get("monitor") or self._monitor)
        available = await asyncio.to_thread(monitor_count)
        if monitor > available:
            return ToolResult(
                f"There is no monitor {monitor}; this computer has {available}.",
                ok=False,
                action=f"Bildschirm: Monitor {monitor} gibt es nicht ({available} vorhanden)",
                say=speech(
                    f"Einen Monitor {monitor} gibt es nicht, ich sehe {available}.",
                    f"There is no monitor {monitor}; I see {available}.",
                ),
            )
        png = await asyncio.to_thread(capture_png, monitor, self._max_size)
        return ToolResult(
            "Screenshot taken; it is attached as the next message.",
            images=(ImagePart(png, "image/png"),),
            action=f"Bildschirm aufgenommen (Monitor {monitor}, {len(png) // 1024} KB)",
        )
