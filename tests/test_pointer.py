from typing import Any

import pytest

from flay.assistant import Assistant, tool_intent
from flay.config import Settings
from flay.core.messages import ToolCall
from flay.llm.base import ToolSpec
from flay.tools import pointer
from flay.tools.base import ToolRegistry, ToolResult, speech
from flay.tools.pointer import (
    ClickTool,
    Element,
    ScrollTool,
    best_matches,
    label_score,
    parse_point,
)
from flay.tools.screen import Shot, Window
from tests.test_assistant import FakeOutput, FakeSource, FakeSTT, FakeTTS
from tests.test_tools import PickingLLM

GIT_MENU = Element("Git", "AXMenuBarItem", (612, 0, 39, 33), "PhpStorm")
GIT_BUTTON = Element("Git", "AXButton", (1480, 861, 32, 32), "PhpStorm")
SAVE = Element("Speichern", "AXButton", (100, 500, 80, 24), "TextEdit")
DELETE = Element("Löschen", "AXButton", (200, 500, 80, 24), "TextEdit")
FLAY = Window(9, "Firefox", "Flay", (0, 650, 400, 150))


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ('{"point_2d": [214, 12]}', (214.0, 12.0)),
        ('```json\n{"point_2d": [887.5, 79]}\n```', (887.5, 79.0)),
        ('{"point_2d": [0, 0]}', None),  # the model's "not found"
        ('{"point_2d": [1200, 5]}', None),
        ("There is no such button.", None),
    ],
)
def test_parse_point(answer: str, expected: tuple[float, float] | None) -> None:
    assert parse_point(answer) == expected


def test_points_are_relative_to_the_captured_region() -> None:
    # A window at 100,50 of 800×600 points: the middle of the image is its middle.
    shot = Shot(b"png", (100, 50, 800, 600))

    assert shot.to_screen(500, 500) == (500, 350)
    assert shot.to_screen(0, 1000) == (100, 650)


def test_label_matching() -> None:
    assert label_score("git", "Git") == 1.0
    assert label_score("Speicher", "Speichern") > 0
    assert label_score("Git", "Git-Menü") > 0
    assert label_score("it", "Git") == 0
    assert best_matches("Git", "", [GIT_MENU, GIT_BUTTON, SAVE]) == [GIT_MENU, GIT_BUTTON]
    # The description names the kind of element.
    assert best_matches("Git", "das Git-Menü oben", [GIT_MENU, GIT_BUTTON]) == [GIT_MENU]
    assert best_matches("Datei", "", [GIT_MENU, SAVE]) == []


class FakeLocator:
    def __init__(self, *points: tuple[float, float] | None) -> None:
        self.points = list(points)
        self.asked: list[str] = []
        self.refined = 0

    async def locate(self, shot: Shot, target: str) -> tuple[float, float] | None:
        self.asked.append(target)
        return self.points.pop(0)

    async def refine(self, shot: Shot, guess: tuple[float, float], target: str) -> Any:
        self.refined += 1
        return self.points.pop(0) or guess


@pytest.fixture
def screen(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[float, float], str]]:
    """No real screen, mouse or accessibility; returns the clicks made."""
    clicks: list[tuple[tuple[float, float], str]] = []
    monkeypatch.setattr(pointer, "accessibility_allowed", lambda prompt=False: True)
    monkeypatch.setattr(pointer, "list_windows", lambda: [FLAY])
    monkeypatch.setattr(pointer, "click", lambda point, button: clicks.append((point, button)))
    monkeypatch.setattr(pointer, "is_visible", lambda element: True)
    monkeypatch.setattr(pointer, "element_at", lambda point: None)
    monkeypatch.setattr(pointer, "capture_monitor", lambda m, s: Shot(b"png", (0, 0, 1000, 800)))
    return clicks


def click_tool(locator: FakeLocator | None, *groups: list[Element]) -> ClickTool:
    return ClickTool(locator, elements=lambda app: iter(groups))  # type: ignore[arg-type]


async def test_labelled_element_is_clicked_without_the_vision_model(
    screen: list[tuple[tuple[float, float], str]],
) -> None:
    locator = FakeLocator()
    tool = click_tool(locator, [GIT_MENU, SAVE])

    result = await tool.run({"text": "git", "button": "right"})

    assert screen == [((631.5, 16.5), "right")]
    assert locator.asked == []
    assert result.ok and result.say is not None
    assert "Git" in result.say[next(iter(result.say))]


async def test_frontmost_app_with_the_label_decides(
    screen: list[tuple[tuple[float, float], str]],
) -> None:
    # PhpStorm (in front) has no "Speichern", TextEdit behind it does.
    tool = click_tool(FakeLocator(), [GIT_MENU], [SAVE])

    await tool.run({"text": "Speichern"})

    assert screen == [((140, 512), "left")]


async def test_several_matches_are_decided_by_the_picture(
    screen: list[tuple[tuple[float, float], str]],
) -> None:
    # The model points into the toolbar button, not the menu.
    locator = FakeLocator((1490, 870))
    tool = click_tool(locator, [GIT_MENU, GIT_BUTTON])

    await tool.run({"text": "Git"})

    assert screen == [((1496, 877), "left")]
    assert locator.refined == 0


async def test_unlabelled_target_is_found_in_the_picture_and_refined(
    screen: list[tuple[tuple[float, float], str]],
) -> None:
    locator = FakeLocator((900, 120), (905, 118))
    tool = click_tool(locator, [GIT_MENU])

    result = await tool.run({"description": "bell icon at the top right"})

    assert locator.asked == ["bell icon at the top right"] and locator.refined == 1
    assert screen == [((905, 118), "left")]
    assert result.ok


async def test_nothing_found_clicks_nothing(screen: list[tuple[tuple[float, float], str]]) -> None:
    result = await click_tool(FakeLocator(None), []).run({"text": "Löschen"})

    assert not result.ok and screen == []


async def test_risky_left_click_asks_first(screen: list[tuple[tuple[float, float], str]]) -> None:
    tool = click_tool(FakeLocator(), [DELETE], [DELETE])

    asked = await tool.run({"text": "Löschen"})

    assert screen == [] and asked.confirm is not None
    assert asked.confirm.arguments["confirmed"] is True

    done = await tool.run(asked.confirm.arguments)

    assert done.ok and screen == [((240, 512), "left")]


async def test_right_click_on_a_risky_label_needs_no_question(
    screen: list[tuple[tuple[float, float], str]],
) -> None:
    result = await click_tool(FakeLocator(), [DELETE]).run({"text": "Löschen", "button": "right"})

    assert result.ok and screen == [((240, 512), "right")]


async def test_flays_own_window_is_never_clicked(
    screen: list[tuple[tuple[float, float], str]],
) -> None:
    own = Element("Senden", "AXButton", (100, 700, 80, 24), "Firefox")

    result = await click_tool(FakeLocator(), [own]).run({"text": "Senden", "button": "right"})

    assert not result.ok and screen == []


async def test_missing_permission_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pointer, "accessibility_allowed", lambda prompt=False: False)

    result = await click_tool(FakeLocator(), [GIT_MENU]).run({"text": "Git"})

    assert not result.ok and result.say is not None and "Bedienungshilfen" in str(result.say)


async def test_scroll_moves_the_content_under_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    scrolled: list[tuple[object, ...]] = []
    browser = Window(3, "Firefox", "Python Tutorial", (0, 30, 1500, 900))
    monkeypatch.setattr(pointer, "accessibility_allowed", lambda prompt=False: True)
    monkeypatch.setattr(pointer, "list_windows", lambda: [browser])
    monkeypatch.setattr(pointer, "scroll", lambda *args: scrolled.append(args))

    result = await ScrollTool().run({"direction": "down", "pages": 2})

    assert scrolled == [((0, 30, 1500, 900), "down", 2.0)]
    assert result.ok and result.say is not None and not result.images


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Klick auf das Git-Menü", "click_on_screen"),
        ("Mach einen Rechtsklick auf UTF-8", "click_on_screen"),
        ("Drück mal auf Speichern", "click_on_screen"),
        ("Right-click the Git menu", "click_on_screen"),
        ("Scroll mal runter", "scroll_screen"),
        ("Blätter eine Seite weiter", "scroll_screen"),
        ("Wie klicke ich in PhpStorm auf Git?", None),
        ("Sag Claude, er soll auf den Button klicken", "send_to_claude"),
    ],
)
def test_click_and_scroll_intents(text: str, expected: str | None) -> None:
    assert tool_intent(text) == expected


class AskingTool:
    """Asks before acting, like a click on "Löschen"."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec("click_on_screen", "Click", {"type": "object", "properties": {}})

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(arguments)
        if not arguments.get("confirmed"):
            return ToolResult(
                "asked",
                ok=False,
                say=speech("Soll ich wirklich auf Löschen klicken?", "Really click Delete?"),
                confirm=ToolCall("confirmed", "click_on_screen", {**arguments, "confirmed": True}),
            )
        return ToolResult("clicked")


async def test_a_tools_question_is_answered_with_yes() -> None:
    tool = AskingTool()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        PickingLLM("click_on_screen", {"text": "Löschen"}),
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )

    await assistant.handle_text("Klick auf Löschen")

    assert tool.calls == [{"text": "Löschen"}]
    assert assistant.pending_confirmation is not None

    await assistant.handle_text("Ja.")

    assert tool.calls[-1] == {"text": "Löschen", "confirmed": True}
    assert assistant.pending_confirmation is None


async def test_label_given_only_in_the_description_is_found(
    screen: list[tuple[tuple[float, float], str]],
) -> None:
    utf = Element("UTF-8", "AXButton", (1312, 957, 53, 25), "PhpStorm")
    locator = FakeLocator()

    await click_tool(locator, [GIT_MENU, utf]).run(
        {"description": "UTF-8 unten in der Statusleiste", "button": "right"}
    )

    assert screen == [((1338.5, 969.5), "right")] and locator.asked == []


@pytest.mark.parametrize(
    ("label", "risky"),
    [
        ("Löschen", True),
        ("Jetzt kaufen", True),
        ("In den Warenkorb", True),
        ("Absenden", True),
        ("Weitere Informationen", False),
        ("Installation", False),
        ("Alle ablehnen", False),
    ],
)
def test_risky_labels(label: str, risky: bool) -> None:
    assert bool(pointer._RISKY.search(label)) is risky
