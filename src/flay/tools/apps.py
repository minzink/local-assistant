"""Apps and their windows: open, switch to, close, move to a monitor, arrange side by side.

macOS: apps are started and brought to the front with ``open -a``, quit through AppKit,
and windows are moved through the accessibility API (the "Accessibility" permission Flay
already needs for clicking). Windows: apps are started from their Start menu shortcut,
windows are found by title and moved, raised and closed with user32.

Only apps the user names are touched; closing asks the app to quit, so it can still ask
to save.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech
from flay.tools.screen import (
    Region,
    Window,
    is_own,
    list_windows,
    monitor_positions,
    monitor_regions,
)

log = logging.getLogger(__name__)

ACTIONS = ("open", "switch", "close", "move", "maximize", "minimize", "side_by_side")
# A name counts as the app from this similarity on (difflib ratio, lower case).
MIN_SIMILARITY = 0.75
# Space the macOS menu bar takes at the top of the main monitor.
MENU_BAR = 25


@dataclass(frozen=True, slots=True)
class App:
    name: str
    # macOS: the .app bundle; Windows: the Start menu shortcut. None: only running.
    path: Path | None = None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def match_app(wanted: str, apps: list[App]) -> App | None:
    """The app ``wanted`` names: exact, as a word of the name ("Code" → "Visual Studio
    Code") or close enough ("Fire Fox" → "Firefox")."""
    target = _norm(wanted)
    if not target:
        return None
    by_name = {_norm(app.name): app for app in apps}
    if target in by_name:
        return by_name[target]
    words = [app for app in apps if target in {_norm(w) for w in app.name.split()}]
    if len(words) == 1:
        return words[0]
    prefix = [app for app in apps if _norm(app.name).startswith(target) and len(target) >= 4]
    if len(prefix) == 1:
        return prefix[0]
    close = difflib.get_close_matches(target, list(by_name), n=1, cutoff=MIN_SIMILARITY)
    return by_name[close[0]] if close else None


def named_app(text: str, apps: list[App]) -> App | None:
    """An app the sentence names, longest name first ("Visual Studio Code" before "Code")."""
    lowered = f" {text.casefold()} "
    for app in sorted(apps, key=lambda a: len(a.name), reverse=True):
        name = app.name.casefold()
        if len(name) >= 3 and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", lowered):
            return app
    return None


def windows_of(app: App, windows: list[Window]) -> list[Window]:
    """The app's visible windows, frontmost first (Flay's own UI left out)."""
    name = app.name.casefold()
    return [
        w
        for w in windows
        if not is_own(w) and (name in w.app.casefold() or _norm(app.name) in _norm(w.title))
    ]


def target_region(where: str | int | None, regions: list[Region]) -> tuple[int, Region] | None:
    """The monitor ``where`` names: its number, "left"/"right"/"top"/"bottom"/"other"."""
    if not regions:
        return None
    if isinstance(where, int) or (isinstance(where, str) and where.isdigit()):
        number = int(where)
        return (number, regions[number - 1]) if 1 <= number <= len(regions) else None
    side = str(where or "").casefold()
    if side == "other" and len(regions) == 2:
        return 2, regions[1]
    for number, position in monitor_positions(regions).items():
        if position == side:
            return number, regions[number - 1]
    return None


def halves(region: Region) -> tuple[Region, Region]:
    left, top, width, height = region
    half = width // 2
    return (left, top, half, height), (left + half, top, width - half, height)


def usable(region: Region, number: int) -> Region:
    """The part of a monitor windows may cover (macOS: below the menu bar)."""
    if sys.platform == "darwin":
        left, top, width, height = region
        return (left, top + MENU_BAR, width, height - MENU_BAR)
    return region


# macOS ----------------------------------------------------------------------------------


def _mac_apps() -> list[App]:
    apps: dict[str, App] = {}
    for folder in (
        Path("/Applications"),
        Path("/Applications/Utilities"),
        Path("/System/Applications"),
        Path("/System/Applications/Utilities"),
        Path.home() / "Applications",
    ):
        if folder.is_dir():
            for bundle in folder.glob("*.app"):
                apps.setdefault(bundle.stem.casefold(), App(bundle.stem, bundle))
    for name in _mac_running():
        apps.setdefault(name.casefold(), App(name))
    return list(apps.values())


def _mac_running() -> dict[str, int]:
    from AppKit import NSWorkspace

    running = {}
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        if app.activationPolicy() == 0 and app.localizedName():  # regular apps only
            running[str(app.localizedName())] = int(app.processIdentifier())
    return running


def _mac_open(app: App) -> None:
    target = str(app.path) if app.path else app.name
    subprocess.run(["open", "-a", target], check=True, capture_output=True, timeout=15)


def _mac_quit(app: App) -> bool:
    from AppKit import NSRunningApplication

    pid = _mac_running().get(app.name)
    if pid is None:
        return False
    running = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    return bool(running is not None and running.terminate())


def _mac_window(app: App) -> Any:
    import ApplicationServices as AX

    pid = _mac_running().get(app.name)
    if pid is None:
        return None
    element = AX.AXUIElementCreateApplication(pid)
    for attribute in ("AXFocusedWindow", "AXMainWindow"):
        error, window = AX.AXUIElementCopyAttributeValue(element, attribute, None)
        if error == 0 and window is not None:
            return window
    error, windows = AX.AXUIElementCopyAttributeValue(element, "AXWindows", None)
    return windows[0] if error == 0 and windows else None


def _mac_set_frame(app: App, region: Region) -> bool:
    import ApplicationServices as AX
    import Quartz

    window = _mac_window(app)
    if window is None:
        return False
    left, top, width, height = region
    position = AX.AXValueCreate(AX.kAXValueCGPointType, Quartz.CGPoint(left, top))
    size = AX.AXValueCreate(AX.kAXValueCGSizeType, Quartz.CGSize(width, height))
    AX.AXUIElementSetAttributeValue(window, "AXMinimized", False)
    # Position first: a window too large for the new monitor would be clamped.
    AX.AXUIElementSetAttributeValue(window, "AXPosition", position)
    AX.AXUIElementSetAttributeValue(window, "AXSize", size)
    AX.AXUIElementSetAttributeValue(window, "AXPosition", position)
    return True


def _mac_minimize(app: App) -> bool:
    import ApplicationServices as AX

    window = _mac_window(app)
    if window is None:
        return False
    return bool(AX.AXUIElementSetAttributeValue(window, "AXMinimized", True) == 0)


# Windows --------------------------------------------------------------------------------


def _start_menu_dirs() -> list[Path]:
    roots = [os.environ.get("PROGRAMDATA", r"C:\ProgramData"), os.environ.get("APPDATA", "")]
    return [Path(r) / "Microsoft" / "Windows" / "Start Menu" / "Programs" for r in roots if r]


def _win_apps() -> list[App]:
    apps: dict[str, App] = {}
    for folder in _start_menu_dirs():
        if folder.is_dir():
            for shortcut in folder.rglob("*.lnk"):
                if not re.search(r"uninstall|deinstall|readme|hilfe|help", shortcut.stem, re.I):
                    apps.setdefault(shortcut.stem.casefold(), App(shortcut.stem, shortcut))
    for window in list_windows():
        if window.app:
            apps.setdefault(window.app.casefold(), App(window.app))
    return list(apps.values())


def _user32() -> Any:
    import ctypes

    return ctypes.windll.user32  # type: ignore[attr-defined, unused-ignore]


def _win_open(app: App) -> None:
    if app.path is None:
        raise FileNotFoundError(f"no Start menu entry for {app.name}")
    os.startfile(app.path)  # type: ignore[attr-defined, unused-ignore]


def _win_front(window: Window) -> None:
    import ctypes

    user32 = _user32()
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined, unused-ignore]
    if user32.IsIconic(window.id):
        user32.ShowWindow(window.id, 9)  # SW_RESTORE
    # Windows only lets the app in front hand over the foreground: attached to its input
    # thread for a moment, Flay counts as that app.
    front = user32.GetForegroundWindow()
    front_thread = user32.GetWindowThreadProcessId(front, None)
    own_thread = kernel32.GetCurrentThreadId()
    attached = front_thread != own_thread and user32.AttachThreadInput(
        own_thread, front_thread, True
    )
    try:
        user32.BringWindowToTop(window.id)
        user32.SetForegroundWindow(window.id)
    finally:
        if attached:
            user32.AttachThreadInput(own_thread, front_thread, False)


def _win_close(window: Window) -> None:
    _user32().PostMessageW(window.id, 0x0010, 0, 0)  # WM_CLOSE


def _win_set_frame(window: Window, region: Region) -> None:
    user32 = _user32()
    user32.ShowWindow(window.id, 9)  # SW_RESTORE: a maximized window cannot be moved
    left, top, width, height = region
    user32.MoveWindow(window.id, left, top, width, height, True)


def _win_minimize(window: Window) -> None:
    _user32().ShowWindow(window.id, 6)  # SW_MINIMIZE


# The tool -------------------------------------------------------------------------------


def installed_apps() -> list[App]:
    if sys.platform == "darwin":
        return _mac_apps()
    if sys.platform == "win32":
        return _win_apps()
    return []


class Desktop:
    """What the tool does to apps and windows; replaced in tests."""

    def apps(self) -> list[App]:
        return installed_apps()

    def windows(self) -> list[Window]:
        return list_windows()

    def monitors(self) -> list[Region]:
        return monitor_regions()

    def open(self, app: App) -> None:
        if sys.platform == "darwin":
            _mac_open(app)
        else:
            _win_open(app)

    def front(self, app: App, window: Window | None) -> None:
        if sys.platform == "darwin":
            _mac_open(app)  # brings a running app to the front
        elif window is not None:
            _win_front(window)
        else:
            _win_open(app)

    def close(self, app: App, window: Window | None) -> bool:
        if sys.platform == "darwin":
            return _mac_quit(app)
        if window is None:
            return False
        _win_close(window)
        return True

    def set_frame(self, app: App, window: Window | None, region: Region) -> bool:
        if sys.platform == "darwin":
            return _mac_set_frame(app, region)
        if window is None:
            return False
        _win_set_frame(window, region)
        return True

    def minimize(self, app: App, window: Window | None) -> bool:
        if sys.platform == "darwin":
            return _mac_minimize(app)
        if window is None:
            return False
        _win_minimize(window)
        return True


class ControlWindowTool:
    def __init__(self, desktop: Desktop | None = None) -> None:
        self._desktop = desktop or Desktop()
        self._apps: list[App] | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="control_window",
            description=(
                "Open, switch to, close, move, maximize or minimize an app (not a PhpStorm "
                "project). 'move' and 'maximize' take 'monitor': a number or 'left'/'right'/"
                "'other'; 'side_by_side' puts 'app' left and 'other_app' right."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "app": {"type": "string", "description": "App name, e.g. 'Firefox'"},
                    "monitor": {"type": "string", "description": "e.g. '2', 'left', 'other'"},
                    "other_app": {"type": "string", "description": "side_by_side: right app"},
                },
                "required": ["action", "app"],
            },
        )

    def app_names(self) -> list[App]:
        """Installed and running apps (read once; the list rarely changes)."""
        if self._apps is None:
            self._apps = self._desktop.apps()
        return self._apps

    def _find(self, name: str) -> App | None:
        found = match_app(name, self.app_names())
        if found is None:
            self._apps = None  # perhaps installed just now
            found = match_app(name, self.app_names())
        return found

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = str(arguments.get("action") or "")
        name = str(arguments.get("app") or "").strip()
        if action not in ACTIONS or not name:
            return ToolResult(
                "Name the app and what to do with it.",
                ok=False,
                say=speech("Welche App meinst du?", "Which app do you mean?"),
            )
        app = await asyncio.to_thread(self._find, name)
        if app is None:
            return ToolResult(
                f"No app called '{name}' is installed or running.",
                ok=False,
                action=f"App: „{name}“ nicht gefunden",
                say=speech(f"Eine App {name} finde ich nicht.", f"I can't find an app {name}."),
            )
        handler: Callable[[App, dict[str, Any]], ToolResult] = getattr(self, f"_{action}")
        return await asyncio.to_thread(handler, app, arguments)

    def _window(self, app: App) -> Window | None:
        found = windows_of(app, self._desktop.windows())
        return found[0] if found else None

    def _open(self, app: App, arguments: dict[str, Any]) -> ToolResult:
        self._desktop.open(app)
        return _done(f"Opened {app.name}.", f"App geöffnet: {app.name}",
                     f"{app.name} ist offen.", f"{app.name} is open.")  # fmt: skip

    def _switch(self, app: App, arguments: dict[str, Any]) -> ToolResult:
        self._desktop.front(app, self._window(app))
        return _done(f"Brought {app.name} to the front.", f"App nach vorn: {app.name}",
                     f"Hier ist {app.name}.", f"Here is {app.name}.")  # fmt: skip

    def _close(self, app: App, arguments: dict[str, Any]) -> ToolResult:
        if not self._desktop.close(app, self._window(app)):
            return _not_running(app)
        return _done(f"Asked {app.name} to close.", f"App geschlossen: {app.name}",
                     f"Ich habe {app.name} geschlossen.", f"I closed {app.name}.")  # fmt: skip

    def _move(self, app: App, arguments: dict[str, Any], maximize: bool = False) -> ToolResult:
        monitors = self._desktop.monitors()
        where = arguments.get("monitor")
        found = target_region(where, monitors) if where else None
        window = self._window(app)
        if found is None:
            if where or not maximize:
                return ToolResult(
                    f"There is no monitor '{where}' ({len(monitors)} connected).",
                    ok=False,
                    say=speech("Welchen Monitor meinst du?", "Which monitor do you mean?"),
                )
            # Maximize where the window is now.
            number = _monitor_of(window, monitors) if window else 1
            found = number, monitors[number - 1]
        number, region = found
        frame = usable(region, number)
        if not maximize and window is not None:
            # Keep the size if it fits; centre it on the new monitor.
            width, height = min(window.bounds[2], frame[2]), min(window.bounds[3], frame[3])
            frame = (frame[0] + (frame[2] - width) // 2, frame[1] + (frame[3] - height) // 2,
                     width, height)  # fmt: skip
        if not self._desktop.set_frame(app, window, frame):
            return _not_running(app)
        self._desktop.front(app, window)
        verb = "maximiert" if maximize else f"auf Monitor {number} verschoben"
        return _done(
            f"{app.name} {'maximized on' if maximize else 'moved to'} monitor {number}.",
            f"Fenster {verb}: {app.name}",
            f"{app.name} ist jetzt {'groß ' if maximize else ''}auf Monitor {number}.",
            f"{app.name} is now {'maximized ' if maximize else ''}on monitor {number}.",
        )

    def _maximize(self, app: App, arguments: dict[str, Any]) -> ToolResult:
        return self._move(app, arguments, maximize=True)

    def _minimize(self, app: App, arguments: dict[str, Any]) -> ToolResult:
        if not self._desktop.minimize(app, self._window(app)):
            return _not_running(app)
        return _done(f"Minimized {app.name}.", f"Fenster minimiert: {app.name}",
                     f"{app.name} ist minimiert.", f"{app.name} is minimized.")  # fmt: skip

    def _side_by_side(self, app: App, arguments: dict[str, Any]) -> ToolResult:
        other_name = str(arguments.get("other_app") or "").strip()
        other = self._find(other_name) if other_name else None
        if other is None:
            return ToolResult(
                "side_by_side needs a second app in 'other_app'.",
                ok=False,
                say=speech("Neben welche App?", "Next to which app?"),
            )
        monitors = self._desktop.monitors()
        window = self._window(app)
        number = _monitor_of(window, monitors) if window else 1
        if arguments.get("monitor") and (found := target_region(arguments["monitor"], monitors)):
            number = found[0]
        left, right = halves(usable(monitors[number - 1], number))
        other_window = self._window(other)
        if not self._desktop.set_frame(app, window, left):
            return _not_running(app)
        if not self._desktop.set_frame(other, other_window, right):
            return _not_running(other)
        self._desktop.front(other, other_window)
        self._desktop.front(app, window)
        return _done(
            f"{app.name} left, {other.name} right on monitor {number}.",
            f"Nebeneinander: {app.name} | {other.name}",
            f"{app.name} links, {other.name} rechts.",
            f"{app.name} on the left, {other.name} on the right.",
        )


def _monitor_of(window: Window, monitors: list[Region]) -> int:
    from flay.tools.screen import window_monitor

    return window_monitor(window, monitors)


def _done(text: str, action: str, de: str, en: str) -> ToolResult:
    return ToolResult(text, action=action, say=speech(de, en))


def _not_running(app: App) -> ToolResult:
    return ToolResult(
        f"{app.name} has no window open.",
        ok=False,
        action=f"App: {app.name} hat kein Fenster",
        say=speech(f"{app.name} hat gerade kein Fenster offen.", f"{app.name} has no window open."),
    )
