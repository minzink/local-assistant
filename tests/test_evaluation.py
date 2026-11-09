from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from flay.config import Settings
from flay.core.messages import Message, ToolCall
from flay.evaluation import DEFAULT_CASES, _judge, load_cases, run_case
from flay.llm.base import LLMEvent, ModelCapabilities, TextDelta, ToolSpec
from flay.tools.phpstorm import Project

PROJECTS = [
    Project(name, Path("/") / name)
    for name in ["flay", "moewe", "project-symfony-kx-forms", "lib-react-kx-forms-app"]
]


@pytest.mark.parametrize(
    ("expect", "calls", "problem"),
    [
        ({"tool": "none"}, [], ""),
        ({"tool": "none"}, [{"tool": "open_project", "name": "flay"}], "Werkzeug open_project"),
        (
            {"tool": "open_project", "project": ["flay"]},
            [{"tool": "open_project", "name": "Flay"}],
            "",
        ),
        (
            {"tool": "open_project", "project": ["moewe"]},
            [{"tool": "open_project", "name": "flay"}],
            "Projekt",
        ),
        # Ambiguous between allowed projects is fine: the tool asks back.
        (
            {"tool": "open_project", "project": [PROJECTS[2].name, PROJECTS[3].name]},
            [{"tool": "open_project", "name": "kx forms"}],
            "",
        ),
        (
            {"tool": "send_to_claude", "contains": ["readme"]},
            [{"tool": "send_to_claude", "instruction": "Fasse die README zusammen."}],
            "",
        ),
        ({"tool": ["close_project", "none"]}, [], ""),
        # A wrong action after the right one counts too.
        (
            {"tool": "look_at_screen", "never": ["open_project"]},
            [{"tool": "look_at_screen"}, {"tool": "open_project", "name": "flay"}],
            "außerdem open_project",
        ),
        ({"tool": "look_at_screen", "never": ["open_project"]}, [{"tool": "look_at_screen"}], ""),
    ],
)
def test_judge(expect: dict[str, Any], calls: list[dict[str, Any]], problem: str) -> None:
    result = _judge({"expect": expect}, calls, PROJECTS)

    assert result.startswith(problem) if problem else result == ""


def test_default_cases_are_valid() -> None:
    state, cases = load_cases(DEFAULT_CASES)

    assert state["open_projects"]
    for case in cases:
        assert case["say"] and "tool" in case["expect"]


class OpeningLLM:
    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        if [t.name for t in tools] == ["open_project"]:
            yield ToolCall("1", "open_project", {"name": "Moewe"})
        else:
            yield TextDelta("Okay.")


async def test_run_case_goes_through_the_assistant_with_simulated_tools() -> None:
    case = {
        "say": "Öffne das Projekt Moewe",
        "expect": {"tool": "open_project", "project": ["moewe"]},
    }
    state = {"open_projects": ["flay"], "recent_projects": ["flay", "moewe"]}

    result = await run_case(Settings(), OpeningLLM(), case, state)

    assert result.ok, result.problem
    assert result.tools == [{"tool": "open_project", "name": "Moewe"}]
    assert result.reply == "Erledigt."
    assert result.first_tool is not None


@pytest.mark.parametrize(
    ("reply", "ok"),
    [
        ("Am langsamsten ist Server C.", True),
        ("Ich denke, das ist schwer zu sagen.", False),
        ("Es bleiben 5 Tage.", True),
        ("Es bleiben 15 Tage.", False),
    ],
)
def test_answer_must_appear_as_a_whole_word(reply: str, ok: bool) -> None:
    from flay.evaluation import _judge_answer

    expect = {"answer": ["C", "5"]}

    assert (_judge_answer(expect, reply) == "") is ok


async def test_history_actions_reach_the_conversation() -> None:
    from flay.config import Settings
    from flay.evaluation import run_case
    from tests.test_tools import RoutingLLM

    llm = RoutingLLM()
    case = {
        "say": "Und jetzt KX Command auch.",
        "history": [
            {
                "user": "Schließe flay.",
                "assistant": "flay ist geschlossen.",
                "actions": [{"tool": "close_project", "arguments": {"name": "flay"}}],
            }
        ],
        "expect": {"tool": "close_project"},
    }

    await run_case(Settings(), llm, case, {"open_projects": ["flay"]})

    seen = [call.name for m in llm.prompts[0][0] for call in m.tool_calls]
    assert seen == ["close_project"]
