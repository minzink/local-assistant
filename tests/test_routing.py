from collections.abc import AsyncIterator, Sequence
from typing import Any

from flay import routing
from flay.assistant import Assistant
from flay.config import Settings
from flay.core.language import Language
from flay.core.messages import Message
from flay.llm.base import LLMEvent, ModelCapabilities, TextDelta, ToolSpec
from flay.tools.base import ToolRegistry
from tests.test_assistant import FakeOutput, FakeSource, FakeSTT, FakeTTS
from tests.test_tools import SayingCloseTool


def test_schema_only_allows_known_projects() -> None:
    schema = routing.schema(["close_project"], ["flay", "moewe"])

    assert schema["properties"]["project"]["enum"] == ["", "flay", "moewe"]
    assert schema["properties"]["action"]["enum"] == ["close_project", "none"]
    # The model states the request before it decides the action.
    assert list(schema["properties"])[0] == "request"


def test_projects_are_listed_in_the_prompt() -> None:
    system, user = routing.messages("Schließe flay", Language.DE, ["close_project"], ["flay"])

    assert "Projects: flay." in system.text
    assert user.text == "Schließe flay"


def test_to_call() -> None:
    def call(action: str, project: str = "", text: str = "") -> Any:
        found = routing.to_call({"action": action, "project": project, "text": text})
        return found and (found.name, found.arguments)

    assert call("none") is None
    assert call("close_project", "flay") == ("close_project", {"name": "flay"})
    assert call("send_to_claude", "", "Schreib Tests.") == (
        "send_to_claude",
        {"instruction": "Schreib Tests."},
    )
    assert call("remember", text="Tee") == ("remember", {"fact": "Tee"})


class FormLLM:
    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer
        self.forms = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def complete_json(
        self, messages: Sequence[Message], schema: dict[str, Any], name: str = "answer"
    ) -> dict[str, Any]:
        self.forms += 1
        return self.answer

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        yield TextDelta("Gern.")


def assistant_with(llm: FormLLM, tool: SayingCloseTool, structured: bool = True) -> Assistant:
    settings = Settings.model_validate({"llm": {"structured_routing": structured}})
    return Assistant(
        settings,
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
        projects=lambda: ["flay", "project-symfony-kx-ewo"],
    )


async def test_recognized_action_is_filled_in_by_the_form() -> None:
    form = {"request": "close", "action": "close_project", "project": "project-symfony-kx-ewo"}
    llm, tool = FormLLM(form | {"text": ""}), SayingCloseTool()

    await assistant_with(llm, tool).handle_text("Schließe das Projekt K X E W O.")

    assert tool.calls == [{"name": "project-symfony-kx-ewo"}]


async def test_form_saying_no_action_keeps_the_answer_from_acting() -> None:
    form = {"request": "question", "action": "none", "project": "", "text": ""}
    llm, tool = FormLLM(form), SayingCloseTool()
    assistant = assistant_with(llm, tool)

    await assistant.handle_text("Schließe ich das Projekt flay besser jetzt?")

    assert tool.calls == []
    assert assistant.history[-1].text == "Gern."


async def test_form_is_off_by_default() -> None:
    llm, tool = FormLLM({}), SayingCloseTool()

    await assistant_with(llm, tool, structured=False).handle_text("Schließe flay.")

    assert llm.forms == 0
