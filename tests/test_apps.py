from __future__ import annotations

from pathlib import Path

from cassi.tools.apps import (
    App,
    ControlWindowTool,
    Desktop,
    match_app,
    named_app,
    target_region,
)
from cassi.tools.screen import Region, Window

APPS = [
    App("Firefox", Path("/Applications/Firefox.app")),
    App("Visual Studio Code", Path("/Applications/Visual Studio Code.app")),
    App("PhpStorm", Path("/Applications/PhpStorm.app")),
    App("Slack"),
]
SIDE_BY_SIDE: list[Region] = [(0, 0, 1920, 1080), (-1920, 0, 1920, 1080)]


def test_match_app() -> None:
    assert match_app("firefox", APPS) == APPS[0]
    assert match_app("Fire Fox", APPS) == APPS[0]
    assert match_app("Code", APPS) == APPS[1]
    assert match_app("phpstorm", APPS) == APPS[2]
    assert match_app("Photoshop", APPS) is None


def test_named_app() -> None:
    assert named_app("Wechsel zu Visual Studio Code", APPS) == APPS[1]
    assert named_app("Mach Slack zu", APPS) == APPS[3]
    assert named_app("Öffne das Projekt flay", APPS) is None


def test_target_region() -> None:
    assert target_region("2", SIDE_BY_SIDE) == (2, SIDE_BY_SIDE[1])
    assert target_region("left", SIDE_BY_SIDE) == (2, SIDE_BY_SIDE[1])
    assert target_region("right", SIDE_BY_SIDE) == (1, SIDE_BY_SIDE[0])
    assert target_region("other", SIDE_BY_SIDE) == (2, SIDE_BY_SIDE[1])
    assert target_region("3", SIDE_BY_SIDE) is None


class FakeDesktop(Desktop):
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.open_windows = [
            Window(1, "Firefox", "Docs – Firefox", (100, 100, 1000, 700)),
            Window(2, "Slack", "Slack", (200, 200, 800, 600)),
        ]

    def apps(self) -> list[App]:
        return APPS

    def windows(self) -> list[Window]:
        return self.open_windows

    def monitors(self) -> list[Region]:
        return SIDE_BY_SIDE

    def open(self, app: App) -> None:
        self.calls.append(("open", app.name))

    def front(self, app: App, window: Window | None) -> None:
        self.calls.append(("front", app.name))

    def close(self, app: App, window: Window | None) -> bool:
        self.calls.append(("close", app.name))
        return window is not None

    def set_frame(self, app: App, window: Window | None, region: Region) -> bool:
        self.calls.append(("frame", app.name, str(region)))
        return window is not None

    def minimize(self, app: App, window: Window | None) -> bool:
        self.calls.append(("minimize", app.name))
        return window is not None


async def test_open_and_switch() -> None:
    desktop = FakeDesktop()
    tool = ControlWindowTool(desktop)

    opened = await tool.run({"action": "open", "app": "firefox"})
    switched = await tool.run({"action": "switch", "app": "Slack"})

    assert opened.ok and switched.ok
    assert desktop.calls == [("open", "Firefox"), ("front", "Slack")]


async def test_move_keeps_the_size_and_centres() -> None:
    desktop = FakeDesktop()
    result = await ControlWindowTool(desktop).run(
        {"action": "move", "app": "Firefox", "monitor": "left"}
    )
    assert result.ok and "Monitor 2" in (result.action or "")
    assert ("frame", "Firefox", str((-1460, 190, 1000, 700))) in desktop.calls or (
        "frame",
        "Firefox",
        str((-1460, 202, 1000, 700)),
    ) in desktop.calls


async def test_side_by_side_splits_the_monitor() -> None:
    desktop = FakeDesktop()
    result = await ControlWindowTool(desktop).run(
        {"action": "side_by_side", "app": "Firefox", "other_app": "Slack"}
    )
    assert result.ok
    frames = [c for c in desktop.calls if c[0] == "frame"]
    assert [f[1] for f in frames] == ["Firefox", "Slack"]
    assert frames[0][2].startswith("(0, ") and frames[1][2].startswith("(960, ")


async def test_unknown_app_and_closed_app() -> None:
    tool = ControlWindowTool(FakeDesktop())
    missing = await tool.run({"action": "open", "app": "Photoshop"})
    no_window = await tool.run({"action": "close", "app": "PhpStorm"})
    assert not missing.ok and "Photoshop" in missing.text
    assert not no_window.ok
