from pathlib import Path
from typing import Any

import pytest

from flay.assistant import tool_intent
from flay.tools import workflows
from flay.tools.pointer import Element
from flay.tools.screen import Window
from flay.tools.workflows import Recorder, Step, Store, WorkflowTool, locate, play, risky_steps

PHPSTORM = Window(1, "PhpStorm", "flay – assistant.py", (0, 25, 1000, 800))
FLAY = Window(2, "Firefox", "Flay", (1100, 0, 400, 300))


class Key:
    def __init__(self, name: str | None = None, char: str | None = None) -> None:
        self.name = name
        self.char = char


@pytest.fixture
def screen(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Windows, accessibility and the mouse simulated; records clicks, keys, scrolling."""
    done: dict[str, Any] = {"clicks": [], "keys": [], "scrolls": []}
    elements = {
        (500.0, 40.0): Element("Git", "AXMenuBarItem", (480, 30, 40, 20), "PhpStorm"),
    }
    monkeypatch.setattr(workflows, "list_windows", lambda: [FLAY, PHPSTORM])
    monkeypatch.setattr(workflows, "element_at", lambda point: elements.get(point))
    monkeypatch.setattr(workflows, "click", lambda p, b: done["clicks"].append((p, b)))
    monkeypatch.setattr(workflows, "press_keys", lambda keys: done["keys"].append(keys))
    monkeypatch.setattr(workflows, "scroll_by", lambda dy: done["scrolls"].append(dy))
    monkeypatch.setattr(workflows, "_activate", lambda app: None)
    monkeypatch.setattr(workflows, "MIN_PAUSE", 0)
    monkeypatch.setattr(workflows, "accessibility_allowed", lambda prompt=False: True)
    return done


def test_recording_keeps_clicks_shortcuts_and_special_keys_not_typing(
    screen: dict[str, Any],
) -> None:
    recorder = Recorder(ignore_key="alt_r")

    recorder._on_click(500, 40, "Button.left", True)
    recorder._on_click(500, 40, "Button.left", False)
    recorder._on_click(1200, 100, "Button.left", True)  # Flay's own window
    recorder._on_click(250, 425, "Button.right", True)  # no label: by position
    for key in (Key(char="g"), Key(char="e"), Key("alt_r")):  # typing, the talk key
        recorder._on_press(key)
    recorder._on_press(Key("cmd"))
    recorder._on_press(Key(char="s"))
    recorder._on_release(Key("cmd"))
    recorder._on_press(Key("enter"))

    steps = recorder.steps
    assert [s.kind for s in steps] == ["click", "click", "keys", "keys"]
    assert (steps[0].label, steps[0].app, steps[0].role) == ("Git", "PhpStorm", "AXMenuBarItem")
    assert steps[1].label == "" and steps[1].button == "right" and steps[1].rel == (0.25, 0.5)
    assert steps[2].keys == ["cmd", "s"] and steps[3].keys == ["enter"]


def test_two_quick_clicks_are_a_double_click(screen: dict[str, Any]) -> None:
    recorder = Recorder()
    recorder._on_click(500, 40, "Button.left", True)
    recorder._on_click(500, 40, "Button.left", True)

    assert len(recorder.steps) == 1 and recorder.steps[0].button == "double"


def test_workflows_are_stored(tmp_path: Path) -> None:
    store = Store(tmp_path / "workflows.json")
    steps = [
        Step("click", "PhpStorm", "Git", "AXMenuBarItem", rel=(0.5, 0.0)),
        Step("keys", keys=["esc"]),
    ]

    store.save({"Release bauen": steps})

    assert store.load() == {"Release bauen": steps}
    assert Store(tmp_path / "missing.json").load() == {}


def test_risky_steps_are_named() -> None:
    steps = [
        Step("click", "PhpStorm", "Git"),
        Step("click", "PhpStorm", "Löschen"),
        Step("keys", keys=["enter"]),
    ]

    assert risky_steps(steps) == ["Klick auf „Löschen“ in PhpStorm", "Tasten enter"]


def test_steps_are_found_by_label_else_by_place(
    screen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    git = Element("Git", "AXMenuBarItem", (600, 30, 40, 20), "PhpStorm")  # moved
    monkeypatch.setattr(workflows, "element_groups", lambda app: iter([[git]]))
    monkeypatch.setattr(workflows, "is_visible", lambda element: True)

    assert locate(Step("click", "PhpStorm", "Git", "AXMenuBarItem")) == (620, 40)
    assert locate(Step("click", "PhpStorm", "", rel=(0.5, 0.5))) == (500, 425)
    assert locate(Step("click", "Xcode", "", rel=(0.5, 0.5))) is None


async def test_a_workflow_is_played_and_stops_at_a_missing_step(
    screen: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflows, "element_groups", lambda app: iter([]))
    monkeypatch.setattr(workflows, "FIND_SECONDS", 0.05)
    steps = [
        Step("click", "PhpStorm", "", rel=(0.5, 0.5), button="right"),
        Step("keys", keys=["esc"]),
        Step("scroll", "PhpStorm", dy=-3),
        Step("click", "PhpStorm", "Gibt es nicht"),
    ]

    done, failed = await play(steps)

    assert done == 3 and failed == "Klick auf „Gibt es nicht“ in PhpStorm"
    assert screen["clicks"] == [((500, 425), "right")]
    assert screen["keys"] == [["esc"]] and screen["scrolls"] == [-3]


async def test_the_tool_records_saves_runs_and_deletes(
    screen: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Recorder, "start", lambda self: None)
    monkeypatch.setattr(Recorder, "stop", lambda self: self.steps)
    tool = WorkflowTool(tmp_path / "workflows.json")

    started = await tool.run({"action": "start", "name": "Release bauen"})
    assert started.ok and tool._recording is not None
    tool._recording[1]._on_click(250, 425, "Button.left", True)
    tool._recording[1]._on_press(Key("enter"))
    saved = await tool.run({"action": "stop", "name": "Ablauf"})
    assert saved.ok and "2" in saved.text

    listed = await tool.run({"action": "list"})
    assert listed.say is not None and "Release bauen" in str(listed.say)

    # Enter is risky: asked first, then played.
    asked = await tool.run({"action": "run", "name": "release bauen"})
    assert asked.confirm is not None and screen["clicks"] == []
    ran = await tool.run(asked.confirm.arguments)
    assert ran.ok and screen["clicks"] == [((250, 425), "left")] and screen["keys"] == [["enter"]]

    deleted = await tool.run({"action": "delete", "name": "Release bauen"})
    assert deleted.ok and Store(tmp_path / "workflows.json").load() == {}
    unknown = await tool.run({"action": "run", "name": "Release bauen"})
    assert not unknown.ok


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Merk dir einen Ablauf: Release bauen", "workflow"),
        ("Ablauf fertig", "workflow"),
        ("Mach den Ablauf Release bauen", "workflow"),
        ("Welche Abläufe kennst du?", "workflow"),
        ("Wie läuft der Ablauf in PhpStorm?", None),
        ("Merk dir, dass ich Tee mag", "remember"),
    ],
)
def test_workflow_intent(text: str, expected: str | None) -> None:
    assert tool_intent(text) == expected


async def test_recording_name_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRecorder:
        def __init__(self, ignore: str) -> None: ...
        def start(self) -> None: ...
        def stop(self) -> list[workflows.Step]:
            return []

    monkeypatch.setattr(workflows, "Recorder", FakeRecorder)
    monkeypatch.setattr(workflows, "accessibility_allowed", lambda prompt=False: True)
    tool = WorkflowTool(tmp_path / "w.json")

    assert tool.recording is None
    await tool.run({"action": "start", "name": "Release bauen"})
    assert tool.recording == "Release bauen"
    await tool.run({"action": "stop"})
    assert tool.recording is None
