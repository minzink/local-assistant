from typing import Any

import pytest

from flay.assistant import Assistant
from flay.config import Settings
from flay.tools import browser_text, forms
from flay.tools.base import ToolRegistry
from flay.tools.forms import FillWebFormTool, match_fields
from flay.tools.pages import Input, Page
from tests.test_assistant import FakeOutput, FakeSource, FakeSTT, FakeTTS
from tests.test_tools import PickingLLM

FORM = Page(
    "https://example.org/kontakt",
    "Kontakt",
    "Schreib uns.",
    (),
    "browser",
    inputs=(
        Input("Name", ref="name"),
        Input("E-Mail-Adresse", ref="mail"),
        Input("Nachricht", ref="text"),
        Input("Passwort", secret=True, ref="pw"),
    ),
)


def test_fields_are_matched_by_their_label() -> None:
    found, missing = match_fields(
        FORM, [("E-Mail", "max@example.org"), ("Name", "Max"), ("Telefon", "123")]
    )

    assert [(field.ref, value) for field, _, value in found] == [
        ("mail", "max@example.org"),
        ("name", "Max"),
    ]
    assert missing == ["Telefon"]


@pytest.fixture
def browser(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A browser showing FORM; records what was filled in and whether it came to the front."""
    seen: dict[str, Any] = {"filled": [], "front": 0, "reads": 0}

    async def read_front(app: str) -> Page:
        seen["reads"] += 1
        return FORM

    async def fill(field: Input, value: str) -> bool:
        seen["filled"].append((field.ref, value))
        return True

    monkeypatch.setattr(forms, "front_browser", lambda: type("W", (), {"app": "Firefox"})())
    monkeypatch.setattr(browser_text, "read_front", read_front)
    monkeypatch.setattr(browser_text, "fill", fill)
    monkeypatch.setattr(
        browser_text, "activate_default", lambda: seen.__setitem__("front", seen["front"] + 1)
    )
    return seen


ARGUMENTS = {
    "fields": [
        {"field": "Name", "value": "Max Mustermann"},
        {"field": "Passwort", "value": "geheim"},
        {"field": "Nachricht", "value": "Hallo"},
    ]
}


async def test_the_form_is_filled_only_after_yes_and_never_passwords(
    browser: dict[str, Any],
) -> None:
    tool = FillWebFormTool()

    asked = await tool.run(ARGUMENTS)

    assert browser["filled"] == [] and asked.confirm is not None and not asked.ok
    question = next(iter(asked.say.values())) if asked.say else ""
    assert "Max Mustermann" in question and "Abschicken musst du selbst" in question
    assert "Passwörter trage ich nicht ein" in question

    done = await tool.run(asked.confirm.arguments)

    assert done.ok and browser["filled"] == [("name", "Max Mustermann"), ("text", "Hallo")]
    assert browser["front"] == 1  # the user checks and sends
    assert browser["reads"] == 1  # the page read for the question is reused


async def test_through_the_assistant_with_yes(browser: dict[str, Any]) -> None:
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        PickingLLM("fill_web_form", ARGUMENTS),
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([FillWebFormTool()]),
    )

    await assistant.handle_text(
        "Trag auf dieser Seite bei Name Max Mustermann und bei Nachricht Hallo ein"
    )

    assert browser["filled"] == [] and assistant.pending_confirmation is not None

    await assistant.handle_text("Ja.")

    assert browser["filled"] == [("name", "Max Mustermann"), ("text", "Hallo")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Trag auf dieser Seite bei Name Max ein", "fill_web_form"),
        ("Füll das Kontaktformular aus: Name Max", "fill_web_form"),
        ("Fill in the form with name Max", "fill_web_form"),
        ("Trag in deinem Gedächtnis ein, dass ich Tee mag", "remember"),
        ("Wie viel Speicher hat mein Gedächtnis?", None),
        ("Wie fülle ich ein Formular aus?", None),
    ],
)
def test_fill_intent(text: str, expected: str | None) -> None:
    from flay.assistant import tool_intent, web_intent

    assert tool_intent(text) == expected
    assert web_intent(text) is None
