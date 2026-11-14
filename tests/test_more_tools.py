from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cassi.config import Settings
from cassi.core.language import Language
from cassi.notifications import Announcements, ClaudeState, ClaudeWatcher, Timers, claude_state
from cassi.tools.commands import Command, Outcome, RunCommandTool, summary
from cassi.tools.finder import FindFileTool, search
from cassi.tools.ide import IdeTool, best_file, file_patterns
from cassi.tools.keyboard import KeyboardTool, describe_selection
from cassi.tools.system import SystemTool
from cassi.tools.timers import TimerTool, due_time
from tests.test_assistant import FakeLLM, FakeOutput, FakeSTT, FakeTTS, make_assistant

NOW = datetime(2026, 11, 13, 14, 0, tzinfo=timezone(timedelta(hours=1)))


class Recorder(Announcements):
    def __init__(self) -> None:
        super().__init__()
        self.said: list[str] = []
        self.connect(lambda say: self.said.append(say[Language.DE]))


# Announcements ---------------------------------------------------------------------------


async def test_announcement_is_spoken_when_idle() -> None:
    tts, output = FakeTTS(), FakeOutput()
    assistant = make_assistant(FakeSTT("x"), FakeLLM("Hallo."), tts, output)
    announcements = Announcements()
    announcements.say("Früh gesagt.", "Said early.")  # before the assistant listens
    announcements.connect(assistant.announce)

    for _ in range(50):
        await asyncio.sleep(0.01)
        if tts.spoken:
            break

    assert tts.spoken == [("Früh gesagt.", Language.DE)]
    assert output.played


def test_claude_state_from_the_screen() -> None:
    assert claude_state("✻ Thinking… (12s · esc to interrupt)") is ClaudeState.WORKING
    assert claude_state("Do you want to proceed?\n❯ 1. Yes\n  2. No") is ClaudeState.ASKING
    assert claude_state("> \n  ? for shortcuts") is ClaudeState.IDLE


def test_claude_watcher_speaks_on_changes() -> None:
    recorder = Recorder()
    watcher = ClaudeWatcher(object(), recorder)
    watcher.update("flay", ClaudeState.IDLE, now=0)
    watcher.update("flay", ClaudeState.WORKING, now=1)
    watcher.update("flay", ClaudeState.IDLE, now=5)  # too short to mention
    watcher.update("flay", ClaudeState.WORKING, now=10)
    watcher.update("flay", ClaudeState.ASKING, now=20)
    watcher.update("flay", ClaudeState.WORKING, now=21)
    watcher.update("flay", ClaudeState.IDLE, now=60)
    assert recorder.said == [
        "Claude in flay wartet auf deine Freigabe.",
        "Claude ist in flay fertig.",
    ]


# Timers ----------------------------------------------------------------------------------


def test_due_time() -> None:
    assert due_time(20, None, NOW) == NOW + timedelta(minutes=20)
    assert due_time(None, "15:30", NOW) == NOW.replace(hour=15, minute=30)
    assert due_time(None, "9 Uhr", NOW) == NOW.replace(hour=9) + timedelta(days=1)
    assert due_time(None, "25:00", NOW) is None
    assert due_time(None, None, NOW) is None


async def test_timer_tool_sets_lists_and_cancels(tmp_path: Path) -> None:
    clock = [NOW.timestamp()]
    timers = Timers(tmp_path / "timers.json", clock=lambda: clock[0])
    tool = TimerTool(timers, now=lambda: NOW)

    set_result = await tool.run({"action": "set", "minutes": 20, "text": "Tee"})
    listed = await tool.run({"action": "list"})

    assert set_result.say and "in 20 Minuten an Tee" in set_result.say[Language.DE]
    assert "Tee" in listed.text
    # Survives a restart.
    assert Timers(tmp_path / "timers.json").pending[0].text == "Tee"
    clock[0] += 21 * 60
    assert [t.text for t in timers.due()] == ["Tee"]
    assert timers.pending == []
    assert not (await tool.run({"action": "cancel"})).ok


async def test_timers_announce_when_due(tmp_path: Path) -> None:
    timers = Timers(tmp_path / "t.json", clock=lambda: NOW.timestamp())
    timers.add(NOW.timestamp() - 1, "Meeting")
    timers.add(NOW.timestamp() - 1, "")
    timers.add(NOW.timestamp() - 24 * 3600, "längst vorbei")
    recorder = Recorder()
    task = asyncio.create_task(timers.run(recorder, interval=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    assert recorder.said == ["Erinnerung: Meeting", "Dein Timer ist abgelaufen."]


# System ----------------------------------------------------------------------------------


async def test_system_tool() -> None:
    done: list[tuple[str, int | None]] = []
    tool = SystemTool(lambda action, level: done.append((action, level)))

    louder = await tool.run({"action": "volume_up"})
    level = await tool.run({"action": "set_volume", "level": 30})
    missing = await tool.run({"action": "set_volume"})

    assert louder.ok and level.ok and not missing.ok
    assert done == [("volume_up", None), ("set_volume", 30)]


# Files -----------------------------------------------------------------------------------


def _file(path: Path, age: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))


def test_search_newest_first(tmp_path: Path) -> None:
    _file(tmp_path / "Downloads" / "Rechnung-Oktober.pdf", 100)
    _file(tmp_path / "Downloads" / "rechnung-september.pdf", 50)
    _file(tmp_path / "Downloads" / "notizen.txt", 10)
    _file(tmp_path / "Downloads" / ".git" / "rechnung.pdf", 1)  # hidden folders are skipped
    roots = [tmp_path / "Downloads"]

    assert [f.path.name for f in search("die letzte", "pdf", roots)] == [
        "rechnung-september.pdf",
        "Rechnung-Oktober.pdf",
    ]
    assert [f.path.name for f in search("Rechnung Oktober", "", roots)] == ["Rechnung-Oktober.pdf"]


async def test_find_file_opens_the_newest(tmp_path: Path) -> None:
    _file(tmp_path / "a.pdf", 100)
    _file(tmp_path / "b.pdf", 1)
    opened: list[Path] = []
    tool = FindFileTool([tmp_path], opener=opened.append)

    result = await tool.run({"kind": "pdf", "open": True})
    nothing = await tool.run({"query": "gibtsnicht"})

    assert result.ok and opened == [tmp_path / "b.pdf"]
    assert not nothing.ok


# Commands --------------------------------------------------------------------------------


async def test_run_command_quick_and_in_the_background() -> None:
    recorder = Recorder()
    commands = {
        "hallo": Command("hallo", "echo hallo"),
        "langsam": Command("langsam", "sleep 0.3; echo 3 failed; exit 1"),
    }
    tool = RunCommandTool(commands, recorder, wait_seconds=0.1)

    quick = await tool.run({"name": "hallo"})
    slow = await tool.run({"name": "langsam"})
    unknown = await tool.run({"name": "rm -rf"})
    await asyncio.sleep(0.5)

    assert quick.ok and "hallo" in quick.text
    assert slow.ok and "Bescheid" in (slow.say or {}).get(Language.DE, "")
    assert not unknown.ok
    assert recorder.said == ["langsam ist fertig, 3 fehlgeschlagen."]


def test_command_summary() -> None:
    assert summary(Outcome(0, ""))[0] == "ist erfolgreich durchgelaufen"
    assert summary(Outcome(None, ""))[0] == "ist nicht rechtzeitig fertig geworden"
    assert summary(Outcome(2, "boom"))[0] == "ist mit Fehler 2 beendet"


def test_run_command_needs_commands_in_the_config() -> None:
    from cassi.tools.factory import create_tools

    assert "run_command" not in create_tools(Settings())
    settings = Settings.model_validate({"tools": {"commands": {"tests": {"command": "true"}}}})
    assert "run_command" in create_tools(settings)


# Keyboard --------------------------------------------------------------------------------


async def test_keyboard_types_and_presses() -> None:
    typed: list[str] = []
    pressed: list[str] = []
    tool = KeyboardTool(typed.append, pressed.append, front_is_own=lambda: False)

    await tool.run({"text": "Grüße aus Köln"})
    await tool.run({"shortcut": "save"})
    enter = await tool.run({"shortcut": "enter"})

    assert typed == ["Grüße aus Köln"] and pressed == ["save"]
    assert not enter.ok  # sending stays with the user


async def test_keyboard_refuses_flays_own_window() -> None:
    typed: list[str] = []
    tool = KeyboardTool(typed.append, lambda _: None, front_is_own=lambda: True)
    assert not (await tool.run({"text": "x"})).ok and typed == []


def test_describe_selection() -> None:
    assert "Nothing is selected" in describe_selection("  ")
    assert "«Hallo»" in describe_selection("Hallo")
    assert "more characters" in describe_selection("x" * 7000)


# PhpStorm --------------------------------------------------------------------------------


def test_file_patterns_and_best_file() -> None:
    assert file_patterns("User Controller")[0] == "**/UserController*"
    assert best_file(
        "user controller", ["src/UserControllerTest.php", "src/UserController.php"]
    ) == ("src/UserController.php")


class FakePhpStorm:
    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def open_projects(self) -> list[Path]:
        return [Path("/p/shop"), Path("/p/blog")]

    async def call(self, tool: str, arguments: dict[str, Any]) -> str:
        self.calls.append((tool, arguments))
        answer = self.answers.get(tool, {})
        if callable(answer):
            answer = answer(arguments)
        return json.dumps(answer)


async def test_ide_opens_a_file_in_the_front_project() -> None:
    phpstorm = FakePhpStorm(
        {
            "search_file": lambda a: (
                {"items": [{"filePath": "src/UserController.php"}]}
                if a["q"] == "**/UserController*"
                else {"items": []}
            )
        }
    )
    tool = IdeTool(phpstorm, title=lambda: "blog – index.php")  # type: ignore[arg-type]

    result = await tool.run({"action": "open_file", "query": "User Controller"})

    assert result.ok
    assert phpstorm.calls[-1] == (
        "open_file_in_editor",
        {"filePath": "src/UserController.php", "projectPath": "/p/blog"},
    )


async def test_ide_runs_a_named_configuration() -> None:
    phpstorm = FakePhpStorm(
        {
            "get_run_configurations": {"configurations": [{"name": "PHPUnit"}, {"name": "Serve"}]},
            "execute_run_configuration": {"output": "OK (12 tests)"},
        }
    )
    tool = IdeTool(phpstorm, title=lambda: "")  # type: ignore[arg-type]

    unclear = await tool.run({"action": "run", "query": "", "project": "shop"})
    ran = await tool.run({"action": "run", "query": "phpunit", "project": "shop"})

    assert not unclear.ok and "PHPUnit, Serve" in unclear.text
    assert ran.ok and "OK (12 tests)" in ran.text


async def test_ide_asks_which_project_when_none_is_in_front() -> None:
    tool = IdeTool(FakePhpStorm({}), title=lambda: "")  # type: ignore[arg-type]
    result = await tool.run({"action": "git_status"})
    assert not result.ok and result.say and "welchem Projekt" in result.say[Language.DE]


# Routing through the assistant -----------------------------------------------------------


class RoutingLLM:
    """Fills in whichever single tool it is offered (as the router prompt asks)."""

    def __init__(self, arguments: dict[str, Any]) -> None:
        self.arguments = arguments
        self.offered: list[list[str]] = []

    @property
    def capabilities(self) -> Any:
        from cassi.core.messages import Modality
        from cassi.llm.base import ModelCapabilities

        return ModelCapabilities(frozenset({Modality.TEXT, Modality.IMAGE}))

    async def stream(self, messages: Any, tools: Any = ()) -> Any:
        from cassi.core.messages import ToolCall
        from cassi.llm.base import TextDelta

        self.offered.append([t.name for t in tools])
        if len(tools) == 1:
            yield ToolCall("c1", tools[0].name, self.arguments)
        else:
            yield TextDelta("Okay.")


async def test_open_firefox_goes_to_the_app_tool_without_a_screenshot() -> None:
    from cassi.assistant import Assistant
    from cassi.tools.apps import ControlWindowTool
    from cassi.tools.base import ToolRegistry
    from tests.test_apps import FakeDesktop
    from tests.test_tools import FakeScreenTool, FakeSource

    desktop = FakeDesktop()
    windows = ControlWindowTool(desktop)
    windows.app_names()  # loaded, as after the first request
    screen = FakeScreenTool()
    llm = RoutingLLM({"action": "move", "app": "Firefox", "monitor": "left"})
    tts = FakeTTS()
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, tts, FakeOutput(),
        tools=ToolRegistry([windows, screen]),
    )  # fmt: skip

    await assistant.handle_text("Schieb Firefox auf den linken Monitor")

    assert llm.offered == []  # app, verb and monitor are clear: no model (cassi.direct)
    assert screen.calls == []  # "Monitor" is no reason to look at the screen here
    assert any(call[0] == "frame" for call in desktop.calls)
    assert tts.spoken[-1][0] == "Firefox ist jetzt auf Monitor 2."
