"""What reaches Claude: the task itself, in the project the user named."""

from pathlib import Path

import pytest

from cassi.assistant import Assistant
from cassi.config import Settings
from cassi.core.messages import ImagePart, ToolCall, ToolOutcome
from cassi.tools.base import ToolRegistry, ToolResult, speech
from cassi.tools.files import ReadFileTool
from tests.test_assistant import FakeOutput, FakeSource, FakeSTT, FakeTTS
from tests.test_files import home, projects  # noqa: F401 – fixture
from tests.test_tools import PickingLLM
from tests.test_web import PageTool, ScriptedLLM


def assistant_with(llm: ScriptedLLM, *tools: PageTool) -> Assistant:
    return Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry(tools),
        projects=lambda: ["flay", "umx", "project-symfony-kx-forms"],
    )


async def prepared(assistant: Assistant, request: str, instruction: str) -> dict[str, object]:
    assistant._request = request
    call = await assistant._prepare_for_claude(
        ToolCall("c", "send_to_claude", {"instruction": instruction})
    )
    return dict(call.arguments)


async def test_a_task_addressed_to_claude_is_written_as_the_task() -> None:
    llm = ScriptedLLM("Finde heraus, warum der Build fehlschlägt, und erkläre die Ursache.")
    assistant = assistant_with(llm)

    arguments = await prepared(
        assistant,
        "Frag Claude im umx Projekt, warum der Build fehlschlägt.",
        "Frag Claude, warum der Build fehlschlägt.",
    )

    assert arguments == {
        "instruction": "Finde heraus, warum der Build fehlschlägt, und erkläre die Ursache.",
        "project": "umx",
    }


async def test_a_clean_task_goes_as_it_is() -> None:
    llm = ScriptedLLM()  # no model call expected
    assistant = assistant_with(llm)

    arguments = await prepared(
        assistant,
        "Sag Claude, er soll Tests für den Login schreiben.",
        "Schreib Tests für den Login.",
    )

    assert arguments == {"instruction": "Schreib Tests für den Login."}
    assert llm.prompts == []


async def test_what_the_screen_shows_goes_into_the_task() -> None:
    llm = ScriptedLLM("Baue einen CSV-Export für alle Bestellungen im Kundenportal.")
    assistant = assistant_with(llm)
    shot = ImagePart(b"png", "image/png")
    assistant._actions = [
        ToolOutcome(ToolCall("s", "look_at_screen", {}), "Screenshot taken; attached.")
    ]
    assistant._turn_images = [shot]

    arguments = await prepared(
        assistant,
        "Schau dir den Screen an und kopiere die Beschreibung des Tickets in die Claw-Zeile "
        "vom Projekt Flay.",
        "Kopiere die Beschreibung des Tickets in die Claude-Zeile.",
    )

    assert arguments == {
        "instruction": "Baue einen CSV-Export für alle Bestellungen im Kundenportal.",
        "project": "flay",
    }
    request = llm.prompts[0][-1]
    assert shot in request.parts  # the model sees the screenshot again


async def test_a_failed_rewrite_keeps_the_draft() -> None:
    class Broken(ScriptedLLM):
        async def stream(self, messages, tools=()):  # type: ignore[no-untyped-def,override]
            raise RuntimeError("Ollama weg")
            yield  # pragma: no cover

    assistant = assistant_with(Broken())

    arguments = await prepared(assistant, "Frag Claude was los ist", "Frag Claude, was los ist.")

    assert arguments["instruction"] == "Frag Claude, was los ist."


async def test_a_roadmap_for_claude_goes_to_the_named_project(home: Path) -> None:  # noqa: F811
    reader = ReadFileTool(lambda: projects(home))
    claude = PageTool("send_to_claude", ToolResult("Typed.", say=speech("In Claude.", "-")))
    llm = ScriptedLLM("Die Roadmap beschreibt Phase 12: Webseiten.")
    router = PickingLLM("read_file", {"name": "Roadmap", "where": "flay"})

    class Both:
        capabilities = llm.capabilities

        def stream(self, messages, tools=()):  # type: ignore[no-untyped-def]
            names = [spec.name for spec in tools]
            chosen = router if names == ["read_file"] else llm
            return chosen.stream(messages, tools)

    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        Both(),  # type: ignore[arg-type]
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([reader, claude]),
        projects=lambda: ["flay", "umx"],
    )

    await assistant.handle_text("Fasse die Roadmap im Projekt flay zusammen und gib es Claude")

    assert claude.calls[0]["project"] == "flay"


async def test_a_rewrite_that_still_talks_about_claude_is_pointed_out_once() -> None:
    llm = ScriptedLLM(
        "Kopiere den Text 'Baue den CSV-Export.' in die Claude-Zeile.",
        "Baue den CSV-Export.",
    )
    assistant = assistant_with(llm)

    arguments = await prepared(
        assistant, "kopiere das in die Claw-Zeile", "Kopiere das in die Claude-Zeile."
    )

    assert arguments["instruction"] == "Baue den CSV-Export."
    assert "Claude" in llm.prompts[1][-1].text  # told what was still wrong


async def test_a_page_passed_on_loses_only_the_remarks_about_claude() -> None:
    llm = ScriptedLLM("Google zeigt ein Suchfeld und Links zu Gmail und Bilder.")
    assistant = assistant_with(llm)
    assistant._request = "Schick Claude den Inhalt dieser Webseite."

    call = await assistant._prepare_for_claude(
        ToolCall(
            "web-claude",
            "send_to_claude",
            {"instruction": "Google zeigt ein Suchfeld. Claude Code kann hier nichts tun."},
        )
    )

    assert (
        call.arguments["instruction"] == "Google zeigt ein Suchfeld und Links zu Gmail und Bilder."
    )
    assert "exactly the same content" in llm.prompts[0][0].text
    assert llm.prompts[0][1].text.startswith("Google zeigt ein Suchfeld.")


async def test_an_order_to_copy_text_into_claude_types_the_text() -> None:
    order = "Kopiere den Text 'Baue den CSV-Export.' in die Claude-Zeile des Projekts Flay."
    llm = ScriptedLLM(order, order)
    assistant = assistant_with(llm)

    arguments = await prepared(
        assistant, "kopiere das in die Claw-Zeile", "Kopiere das in die Claude-Zeile."
    )

    assert arguments["instruction"] == "Baue den CSV-Export."


async def test_a_stubborn_remark_on_claude_is_left_out_of_a_page() -> None:
    remark = "Google zeigt ein Suchfeld. Kein Inhalt für Claude zu verarbeiten."
    llm = ScriptedLLM(remark, remark)
    assistant = assistant_with(llm)
    assistant._request = "Schick Claude den Inhalt dieser Webseite."

    call = await assistant._prepare_for_claude(
        ToolCall("web-claude", "send_to_claude", {"instruction": remark})
    )

    assert call.arguments["instruction"] == "Google zeigt ein Suchfeld."


@pytest.mark.parametrize(
    "text",
    [
        "Implementiere im Projekt flay einen Export-Button.",
        "Behebe den Bug im Login.",
        "Schreib Tests für den Warenkorb.",
        "Bau im Projekt umx eine Startseite mit drei Buttons.",
        "Fix the login bug in flay.",
        "Refaktoriere die Klasse UserService.",
        "Füge im Projekt flay eine Validierung für die E-Mail-Adresse hinzu.",
        "Setz das Ticket im Projekt flay um.",
        "Kannst du im Projekt flay einen Endpoint für den Export bauen?",
        "Kannst du im Projekt KX Command eine Validierung für die E-Mail-Adresse einbauen?",
        "Füg dem Formular im Projekt umx ein Pflichtfeld hinzu.",
        "Pass den Login-Button im Projekt flay an.",
    ],
)
def test_coding_goes_to_claude_without_naming_it(text: str) -> None:
    from cassi.intents import to_claude, tool_intent

    assert tool_intent(text) == "send_to_claude" and to_claude(text)


@pytest.mark.parametrize(
    "text",
    [
        "Erzähl mir einen Witz über Programmierer.",
        "Ist der Bug behoben?",
        "Warum schlägt der Test fehl?",
        "Wie implementiere ich einen Login?",
        "Ich habe gestern den Bug gefixt.",
        "Erstelle einen Termin morgen um 10.",
        "Schreib eine Mail an Anna über den Bug.",
        "Schreib: Viele Grüße aus Bielefeld.",
        "Bau mir eine Erinnerung für morgen.",
        "Gib mir den Code für das WLAN.",
        "Öffne das Projekt flay.",
    ],
)
def test_talk_about_code_is_no_order(text: str) -> None:
    from cassi.intents import coding_intent

    assert not coding_intent(text)


async def test_a_copy_order_with_dashes_types_the_text() -> None:
    order = "Kopiere genau diesen Text – 'Baue den CSV-Export.' – in die Claude-Zeile von Flay."
    llm = ScriptedLLM(order, order)
    assistant = assistant_with(llm)

    arguments = await prepared(
        assistant, "kopiere das in die Claw-Zeile", "Kopiere das in die Claude-Zeile."
    )

    assert arguments["instruction"] == "Baue den CSV-Export."
