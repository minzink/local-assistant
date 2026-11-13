from __future__ import annotations

from typing import Any

from flay.assistant import Assistant
from flay.config import Settings
from flay.core.language import Language
from flay.llm.base import ToolSpec
from flay.plans import describe, plan_intents, split_steps
from flay.tools.base import ToolRegistry, ToolResult, speech
from tests.test_assistant import FakeOutput, FakeSTT, FakeTTS
from tests.test_more_tools import RoutingLLM
from tests.test_tools import FakeSource


def test_split_steps() -> None:
    assert split_steps("Öffne Firefox, dann mach lauter und danach sperr den Bildschirm") == [
        "Öffne Firefox",
        "mach lauter",
        "sperr den Bildschirm",
    ]
    assert split_steps("Zuerst öffne flay, anschließend starte die Tests") == [
        "öffne flay",
        "starte die Tests",
    ]
    assert split_steps("Leg Firefox und Slack nebeneinander") == [
        "Leg Firefox und Slack nebeneinander"
    ]


def test_plan_needs_two_actions() -> None:
    known = {"Öffne Firefox": "control_window", "mach lauter": "system_control"}
    assert plan_intents(["Öffne Firefox", "mach lauter"], known.get) == [
        "control_window",
        "system_control",
    ]
    assert plan_intents(["Öffne Firefox", "erzähl was"], known.get) is None
    assert plan_intents(["Öffne Firefox"], known.get) is None


def test_describe() -> None:
    assert describe(["öffne flay", "starte die Tests"], "de") == (
        "Erstens: öffne flay. Zweitens: starte die Tests."
    )


class Recorded:
    def __init__(self, name: str, say: str, ok: bool = True) -> None:
        self.name, self.say, self.ok = name, say, ok
        self.calls: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.name, {"type": "object", "properties": {}})

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult("done", ok=self.ok, say=speech(self.say, self.say))


def _assistant(*tools: Recorded) -> tuple[Assistant, FakeTTS]:
    tts = FakeTTS()
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), RoutingLLM({}), tts, FakeOutput(),
        tools=ToolRegistry(list(tools)),
    )  # fmt: skip
    return assistant, tts


async def test_plan_runs_its_steps_in_order() -> None:
    volume = Recorded("system_control", "Lauter.")
    timer = Recorded("timer", "Timer gestellt.")
    assistant, tts = _assistant(volume, timer)

    await assistant.handle_text("Mach lauter und dann stell einen Timer auf 5 Minuten")

    spoken = " ".join(text for text, _ in tts.spoken)
    assert spoken == (
        "Erstens: Mach lauter. Zweitens: stell einen Timer auf 5 Minuten. Lauter. "
        "Timer gestellt. Fertig."
    )
    assert volume.calls and timer.calls


async def test_plan_stops_at_a_failed_step() -> None:
    volume = Recorded("system_control", "Das ging nicht.", ok=False)
    timer = Recorded("timer", "Timer gestellt.")
    assistant, tts = _assistant(volume, timer)

    await assistant.handle_text("Mach lauter, danach stell einen Timer auf 5 Minuten")

    assert timer.calls == []
    spoken = " ".join(text for text, _ in tts.spoken)
    assert spoken.endswith("Das ging nicht. Die übrigen Schritte habe ich gelassen.")


async def test_far_reaching_plan_waits_for_a_yes() -> None:
    keys = Recorded("use_keyboard", "Erledigt.")
    volume = Recorded("system_control", "Lauter.")
    assistant, tts = _assistant(keys, volume)

    await assistant.handle_text("Schreib: Hallo zusammen, dann mach lauter")
    assert keys.calls == [] and tts.spoken[-1][0].endswith("Soll ich loslegen?")

    await assistant.handle_text("Ja")
    assert keys.calls and volume.calls
    assert tts.spoken[-1] == ("Fertig.", Language.DE)
