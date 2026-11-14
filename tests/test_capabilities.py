"""Phase 15: accuracy (garbled verbs, "und" plans, model classification, OCR), operating the
computer freely, Claude (briefs, answering permissions), calendar, settings by voice,
privacy and the request history."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from flay.classify import classify
from flay.config import Settings
from flay.core.events import (
    AssistantMessage,
    AssistantState,
    StateChanged,
    ToolUsed,
    TurnProfiled,
    UserMessage,
)
from flay.core.language import Language
from flay.core.messages import ImagePart, Message, ToolCall
from flay.history import TurnLog, eval_cases, read, stats
from flay.intents import may_be_computer_task, mentions_clipboard
from flay.llm.base import ModelCapabilities, TextDelta, ToolSpec
from flay.notifications import ClaudeState, ClaudeWatcher
from flay.plans import plan_intents, split_steps
from flay.routines import Routines
from flay.tools import screen
from flay.tools.appointments import Appointment, CalendarAccessError, CalendarTool, span
from flay.tools.base import ToolRegistry, ToolResult
from flay.tools.claude_code import AnswerClaudeTool, read_question
from flay.tools.computer import Observation, OperateComputerTool, Solutions
from flay.tools.keyboard import dictated, parse_keys
from flay.tools.ocr import TextBox, describe, find_text
from flay.tools.preferences import AdjustSettingsTool, Preferences
from flay.verbs import heard_verb, phonetic_code
from tests.test_assistant import FakeOutput, FakeSTT, FakeTTS
from tests.test_more_tools import Recorder, RoutingLLM
from tests.test_tools import FakeSource

PROJECTS = [
    "flay", "umx", "project-symfony-kx-ewo", "component-symfony-kx-command", "moewe",
    "stadt-wolkenthal",
]  # fmt: skip


# Garbled verbs ---------------------------------------------------------------------------


def test_phonetic_code() -> None:
    assert phonetic_code("schließe") == "858"
    assert phonetic_code("slie") == "85"
    assert phonetic_code("öffne") == "036"


@pytest.mark.parametrize(
    ("said", "tool"),
    [
        ("Slie K X E V O", "close_project"),
        ("She K Command.", "close_project"),
        ("Mach mal Moewe auf.", "open_project"),
        ("Mach mal flay zu", "close_project"),
        ("Sie haben flay", None),  # "sie" is a word, not a verb
        ("Schau flay", None),  # a real verb
        ("Mach mal Firefox auf", None),  # no project
        ("She is nice", None),
    ],
)
def test_heard_verb(said: str, tool: str | None) -> None:
    found = heard_verb(said, PROJECTS)
    assert (found[0] if found else None) == tool


# Plans joined by "und" ------------------------------------------------------------------


def test_split_steps_at_and_only_on_request() -> None:
    text = "Schau dir meinen Screen an und kopiere den Text in Claude"
    assert split_steps(text) == [text]
    assert split_steps(text, at_and=True) == [
        "Schau dir meinen Screen an",
        "kopiere den Text in Claude",
    ]


def test_plan_intents_distinct_needs_two_different_tools() -> None:
    intents = {"Leg PhpStorm": "control_window", "Firefox nebeneinander": "control_window"}
    assert plan_intents(list(intents), intents.get, distinct=True) is None
    mixed = {"Schau dir den Screen an": "look_at_screen", "kopier es an Claude": "send_to_claude"}
    assert plan_intents(list(mixed), mixed.get, distinct=True) == list(mixed.values())


# Keyboard --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("said", "typed"),
    [
        ("Hallo Komma wie geht es dir Fragezeichen", "Hallo, wie geht es dir?"),
        ("Viele Grüße Komma neue Zeile Timo", "Viele Grüße,\nTimo"),
        ("Der Punkt ist wichtig", "Der Punkt ist wichtig"),
        ("Bis morgen Punkt", "Bis morgen."),
    ],
)
def test_dictated_punctuation(said: str, typed: str) -> None:
    assert dictated(said) == typed


def test_parse_keys() -> None:
    assert parse_keys("enter") == ([], "enter")
    mods, key = parse_keys("cmd+shift+n") or ([], "")
    assert key == "n" and "shift" in mods and len(mods) == 2
    assert parse_keys("hyper+x") is None


# Classification --------------------------------------------------------------------------


def test_may_be_computer_task() -> None:
    assert may_be_computer_task("Wirf Spotify auf den zweiten Bildschirm")
    assert may_be_computer_task("Stell den Dunkelmodus an")
    assert not may_be_computer_task("Erzähl mir einen Witz")
    assert not may_be_computer_task("Wie spät ist es in Tokio?")
    assert mentions_clipboard("Übersetz, was ich kopiert habe")


class JsonLLM:
    """Answers every JSON request with the next prepared answer."""

    def __init__(self, *answers: dict[str, Any], text: str = "") -> None:
        self.answers = list(answers)
        self.requests: list[Sequence[Message]] = []
        self.text = text

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def complete_json(
        self, messages: Sequence[Message], schema: dict[str, Any], name: str = "answer"
    ) -> dict[str, Any]:
        self.requests.append(messages)
        return self.answers.pop(0) if self.answers else {"move": "fail", "text": "leer"}

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> Any:
        yield TextDelta(self.text)


async def test_classify_picks_among_available_tools() -> None:
    llm = JsonLLM({"action": "control_window"})
    assert await classify(llm, "Wirf Spotify rüber", ["control_window", "timer"]) == (
        "control_window"
    )
    assert "- timer:" in llm.requests[0][0].text and "calendar" not in llm.requests[0][0].text
    assert await classify(JsonLLM({"action": "none"}), "Hallo du", ["timer"]) is None


# OCR -------------------------------------------------------------------------------------


def test_find_text_exact_then_close() -> None:
    boxes = [TextBox("Speichern", (0, 0, 50, 10)), TextBox("Refactor", (60, 0, 40, 10))]
    assert find_text("refactor", boxes) == [boxes[1]]
    assert find_text("Speichem", boxes) == [boxes[0]]  # OCR misreading
    assert find_text("Löschen", boxes) == []
    assert "'Refactor' at 60,0" in describe(boxes)


async def test_click_falls_back_to_ocr_before_the_vision_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flay.tools import pointer

    clicked: list[Any] = []
    monkeypatch.setattr(pointer, "accessibility_allowed", lambda prompt=False: True)
    monkeypatch.setattr(pointer, "_window_or_none", lambda query: (None, []))
    monkeypatch.setattr(pointer, "monitor_bounds", lambda monitor: (0, 0, 800, 600))
    monkeypatch.setattr(pointer, "click", lambda point, button: clicked.append(point))
    tool = pointer.ClickTool(
        None,
        elements=lambda app: iter([[]]),  # accessibility knows nothing
        read_text=lambda region: [TextBox("Refactor", (100, 10, 40, 10))],
    )

    result = await tool.run({"text": "Refactor"})

    assert result.ok and clicked == [(120.0, 15.0)]
    assert "Texterkennung" in (result.action or "")


# Privacy ---------------------------------------------------------------------------------


@pytest.fixture
def private(monkeypatch: pytest.MonkeyPatch) -> Any:
    screen.set_privacy(["1Password"], ["Online-Banking"])
    yield
    screen.set_privacy([], [])
    screen.PRIVACY.paused = False


def test_private_window_in_region(private: None) -> None:
    windows = [
        screen.Window(1, "Firefox", "Sparkasse Online-Banking", (0, 0, 800, 600)),
        screen.Window(2, "1Password", "Tresor", (2000, 0, 400, 400)),
        screen.Window(3, "PhpStorm", "flay", (0, 0, 800, 600)),
    ]
    assert screen.private_window((0, 0, 1000, 1000), windows) == windows[0]
    assert screen.private_window((1500, 0, 1000, 1000), windows) == windows[1]
    assert screen.private_window((0, 700, 100, 100), windows) is None


async def test_screenshot_is_refused_while_a_private_window_shows(
    private: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(screen, "ensure_permission", lambda: None)
    monkeypatch.setattr(
        screen, "list_windows", lambda: [screen.Window(1, "1Password", "x", (0, 0, 500, 500))]
    )
    monkeypatch.setattr(screen, "monitor_bounds", lambda monitor: (0, 0, 1000, 800))

    result = await screen.LookAtScreenTool().run({"window": "1Password"})

    assert not result.ok and not result.images
    assert result.say is not None and "1Password" in result.say[Language.DE]


async def test_any_tool_refused_for_privacy_says_so(private: None) -> None:
    class Peeking:
        spec = ToolSpec("peek", "", {"type": "object", "properties": {}})

        async def run(self, arguments: dict[str, Any]) -> ToolResult:
            raise screen.PrivateScreenError("1Password")

    result = await ToolRegistry([Peeking()]).execute(ToolCall("1", "peek", {}))

    assert not result.ok and "privacy" in result.text


def test_paused_screen_refuses_everything(private: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(screen, "ensure_permission", lambda: None)
    screen.PRIVACY.paused = True
    with pytest.raises(screen.PrivateScreenError):
        screen.ensure_allowed((0, 0, 10, 10))


# Operating the computer ------------------------------------------------------------------


class Recording:
    def __init__(self, name: str, result: ToolResult | None = None) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []
        self.result = result or ToolResult(f"{name} done")

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, "", {"type": "object", "properties": {}})

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(arguments)
        return self.result


SEEN = Observation("PhpStorm", "flay", "", ("Run (button)",), ("Refactor",))


def computer(
    llm: JsonLLM, *tools: Recording, solutions: Solutions | None = None
) -> OperateComputerTool:
    registry = ToolRegistry(list(tools))
    return OperateComputerTool(
        llm,  # type: ignore[arg-type]
        lambda: registry,
        solutions=solutions,
        look=lambda: SEEN,
        press=lambda keys: None,
    )


async def test_computer_moves_until_done_and_learns_the_way(tmp_path: Path) -> None:
    windows = Recording("control_window")
    llm = JsonLLM(
        {"thought": "", "move": "app", "target": "Firefox", "text": "move:2"},
        {"thought": "", "move": "done", "target": "", "text": "Firefox ist auf Monitor 2."},
    )
    solutions = Solutions(tmp_path / "solutions.json")
    tool = computer(llm, windows, solutions=solutions)

    result = await tool.run({"goal": "Schieb Firefox auf den zweiten Monitor"})

    assert result.ok and result.say is not None
    assert result.say[Language.DE] == "Firefox ist auf Monitor 2."
    assert windows.calls == [{"action": "move", "app": "Firefox", "monitor": "2"}]
    assert "Front window: PhpStorm" in llm.requests[0][1].text
    # The second time the model sees how it went before.
    again = JsonLLM({"thought": "", "move": "done", "target": "", "text": "Erledigt."})
    await computer(again, windows, solutions=Solutions(tmp_path / "solutions.json")).run(
        {"goal": "Schieb Firefox auf den zweiten Monitor bitte"}
    )
    assert "app Firefox 'move:2'" in again.requests[0][1].text


async def test_computer_asks_before_a_risky_click() -> None:
    risky = ToolResult(
        "asked", ok=False, confirm=ToolCall("confirmed", "click_on_screen", {"text": "Löschen"})
    )
    clicks = Recording("click_on_screen", risky)
    llm = JsonLLM({"thought": "", "move": "click", "target": "Löschen", "text": ""})

    result = await computer(llm, clicks).run({"goal": "Lösch die Datei"})

    assert result.confirm is not None and result.confirm.name == "operate_computer"
    assert result.confirm.arguments == {"goal": "Lösch die Datei", "allow": "Löschen"}


async def test_computer_asks_before_enter_but_not_for_a_search() -> None:
    enter = {"thought": "", "move": "key", "target": "enter", "text": ""}
    asked = await computer(JsonLLM(enter)).run({"goal": "Schick die Nachricht an Max"})
    assert asked.confirm is not None and asked.confirm.arguments["allow"] == "enter"

    done = {"thought": "", "move": "done", "target": "", "text": "Gesucht."}
    searched = await computer(JsonLLM(enter, done)).run({"goal": "Öffne den Bereich Ton"})
    assert searched.ok and searched.confirm is None


async def test_computer_never_types_passwords() -> None:
    keyboard = Recording("use_keyboard")
    llm = JsonLLM(
        {"thought": "", "move": "type", "target": "", "text": "geheim123"},
        {"thought": "", "move": "fail", "target": "", "text": "Das mache ich nicht."},
    )

    result = await computer(llm, keyboard).run({"goal": "Gib mein Passwort ein"})

    assert keyboard.calls == [] and not result.ok
    assert "never typed" in llm.requests[1][1].text


async def test_computer_stops_after_max_steps() -> None:
    llm = JsonLLM(*[{"thought": "", "move": "wait", "target": "", "text": "0"}] * 20)
    tool = computer(llm)
    tool._max_steps = 3

    result = await tool.run({"goal": "Warte"})

    assert not result.ok and "3 moves" in result.text


# Settings by voice -----------------------------------------------------------------------


class SpeedTTS:
    def __init__(self) -> None:
        self.speeds: list[float] = []
        self.voices_set: list[str] = []

    def set_speed(self, speed: float) -> None:
        self.speeds.append(speed)

    def set_voice(self, name: str) -> None:
        self.voices_set.append(name)


async def test_adjust_settings_applies_and_keeps_them(tmp_path: Path) -> None:
    tts = SpeedTTS()
    preferences = Preferences.load(tmp_path)
    tool = AdjustSettingsTool(preferences, lambda: tts, lambda: ["ramona", "thorsten"])

    await tool.run({"speed": "slower"})
    await tool.run({"length": "short", "voice": "thorsten"})

    assert tts.speeds == [0.85] and tts.voices_set == ["thorsten"]
    stored = Preferences.load(tmp_path)
    assert (stored.speed, stored.voice, stored.length) == (0.85, "thorsten", "short")
    assert "one sentence" in stored.style()
    applied = stored.applied(Settings())
    assert applied.tts.voice.speed == 0.85 and applied.tts.voice.name == "thorsten"


async def test_adjust_settings_pauses_the_screen(tmp_path: Path) -> None:
    tool = AdjustSettingsTool(Preferences.load(tmp_path))
    try:
        result = await tool.run({"screen": "pause"})
        assert result.ok and screen.PRIVACY.paused
        await tool.run({"screen": "resume"})
        assert not screen.PRIVACY.paused
    finally:
        screen.PRIVACY.paused = False


async def test_unknown_voice_lists_the_known_ones(tmp_path: Path) -> None:
    tool = AdjustSettingsTool(Preferences.load(tmp_path), voices=lambda: ["ramona"])
    result = await tool.run({"voice": "Darth Vader"})
    assert not result.ok and "ramona" in result.text


async def test_short_answers_reach_the_system_prompt(tmp_path: Path) -> None:
    from flay.assistant import Assistant
    from tests.test_assistant import FakeLLM

    preferences = Preferences(length="short", path=None)
    llm = FakeLLM("Ja.")
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput(),
        preferences=preferences,
    )  # fmt: skip

    await assistant.handle_text("Wie geht es dir?")

    assert "one sentence" in llm.prompts[0][0].text


# Calendar --------------------------------------------------------------------------------

NOW = datetime(2026, 11, 13, 10, 0)


def test_span() -> None:
    assert span("today", NOW) == (datetime(2026, 11, 13), datetime(2026, 11, 14))
    assert span("tomorrow", NOW)[0] == datetime(2026, 11, 14)
    assert span("2026-12-24", NOW) == (datetime(2026, 12, 24), datetime(2026, 12, 25))
    assert span("week", NOW)[1] == datetime(2026, 11, 20)


async def test_calendar_lists_coming_appointments() -> None:
    def read(start: datetime, end: datetime) -> list[Appointment]:
        return [
            Appointment("Daily", NOW - timedelta(hours=1), NOW - timedelta(minutes=45)),
            Appointment("Review", NOW + timedelta(hours=2), NOW + timedelta(hours=3),
                        location="Raum 2"),
            Appointment("Geburtstag Max", datetime(2026, 11, 13), datetime(2026, 11, 14), True,
                        calendar="Geburtstage"),
        ]  # fmt: skip

    tool = CalendarTool(read, hidden=["Geburtstage"], now=lambda: NOW)
    result = await tool.run({"when": "today"})

    assert result.ok and "Review" in result.text and "Raum 2" in result.text
    assert "Daily" not in result.text  # already over
    assert "Geburtstag" not in result.text  # hidden calendar


async def test_calendar_without_access() -> None:
    def denied(start: datetime, end: datetime) -> list[Appointment]:
        raise CalendarAccessError("no")

    result = await CalendarTool(denied, now=lambda: NOW).run({})
    assert not result.ok and result.say is not None


# Claude ----------------------------------------------------------------------------------

ASKING = """
╭───────────────────────────────────────────╮
│ Edit file                                 │
│ Do you want to make this edit to app.py?  │
│ ❯ 1. Yes                                  │
│   2. Yes, allow all edits during this session (shift+tab) │
│   3. No, and tell Claude what to do differently (esc) │
╰───────────────────────────────────────────╯
"""


def test_read_question() -> None:
    question = read_question(ASKING, "flay", "flay-claude-flay")
    assert question is not None
    assert question.question == "Do you want to make this edit to app.py?"
    assert (question.key_for("yes"), question.key_for("always"), question.key_for("no")) == (
        "1",
        "2",
        "3",
    )
    assert read_question("> \n? for shortcuts", "flay", "s") is None


class FakeTmux:
    available = True

    def __init__(self, shown: str) -> None:
        self.shown = shown
        self.pressed: list[tuple[str, str]] = []

    async def screen(self, session: str, lines: int = 40) -> str:
        return self.shown

    async def press(self, session: str, key: str, count: int = 1) -> None:
        self.pressed.append((session, key))


async def test_answer_claude_presses_the_option() -> None:
    tmux = FakeTmux(ASKING)
    tool = AnswerClaudeTool(tmux)  # type: ignore[arg-type]
    tool.asked(read_question(ASKING, "flay", "flay-claude-flay"), "flay")

    result = await tool.run({"answer": "always"})

    assert result.ok and tmux.pressed == [("flay-claude-flay", "2")]
    assert tool.waiting == {}


async def test_answer_claude_when_the_question_is_gone() -> None:
    tool = AnswerClaudeTool(FakeTmux("> "))  # type: ignore[arg-type]
    tool.asked(read_question(ASKING, "flay", "s"), "flay")
    result = await tool.run({"answer": "yes"})
    assert not result.ok and tool.waiting == {}


def test_watcher_reads_out_the_question_and_tells_the_tool() -> None:
    recorder = Recorder()
    told: list[tuple[Any, str]] = []
    watcher = ClaudeWatcher(object(), recorder)
    watcher.on_question = lambda question, project: told.append((question, project))
    watcher.update("flay", ClaudeState.WORKING, now=0)
    watcher.update("flay", ClaudeState.ASKING, now=1, screen=ASKING, session="s")
    watcher.update("flay", ClaudeState.WORKING, now=2)

    assert recorder.said[0].startswith("Claude in flay fragt: Do you want to make this edit")
    assert told[0][0].question.startswith("Do you want") and told[-1] == (None, "flay")


async def test_yes_answers_claude_when_it_waits() -> None:
    from flay.assistant import Assistant

    tmux = FakeTmux(ASKING)
    answer = AnswerClaudeTool(tmux)  # type: ignore[arg-type]
    answer.asked(read_question(ASKING, "flay", "s"), "flay")
    tts = FakeTTS()
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), RoutingLLM({}), tts, FakeOutput(),
        tools=ToolRegistry([answer]),
    )  # fmt: skip

    await assistant.handle_text("Ja, erlaub das.")

    assert tmux.pressed == [("s", "1")]
    assert tts.spoken[-1][0] == "Ich habe es freigegeben."


async def test_claude_brief_carries_the_conversation() -> None:
    from flay.assistant import Assistant

    send = Recording("send_to_claude", ToolResult("typed", say={Language.DE: "Steht drin."}))
    llm = RoutingLLM({"instruction": "Behebe den Fehler"})
    seen: list[str] = []
    stream = llm.stream

    async def spying(messages: Any, tools: Any = ()) -> Any:
        seen.append(messages[0].text)
        async for event in stream(messages, tools):
            yield event

    llm.stream = spying  # type: ignore[method-assign]
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput(),
        tools=ToolRegistry([send]),
    )  # fmt: skip

    await assistant.handle_text("Sag Claude, er soll den Fehler beheben")

    assert send.calls == [{"instruction": "Behebe den Fehler"}]
    assert any("does not hear this conversation" in text for text in seen)


# Routines --------------------------------------------------------------------------------


def test_routines_define_run_forget(tmp_path: Path) -> None:
    routines = Routines(tmp_path / "routines.json")
    assert routines.define("Merk dir als Feierabend: starte die Tests, dann schließ Slack") == (
        "Feierabend",
        "starte die Tests, dann schließ Slack",
    )
    assert routines.define("Merk dir, dass ich Tee mag") is None
    again = Routines(tmp_path / "routines.json")
    assert again.requested("Mach Feierabend.") is not None
    assert again.requested("Starte die Feierabend-Routine") is not None
    assert again.requested("Feierabend ist toll") is None
    assert again.forget("Vergiss die Routine Feierabend") == "Feierabend"
    assert again.names == []


async def test_a_routine_runs_as_a_plan(tmp_path: Path) -> None:
    from flay.assistant import Assistant

    routines = Routines(tmp_path / "routines.json")
    keyboard = Recording("use_keyboard", ToolResult("ok", say={Language.DE: "Gespeichert."}))
    tts = FakeTTS()
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), RoutingLLM({"shortcut": "save"}), tts,
        FakeOutput(), tools=ToolRegistry([keyboard]), routines=routines,
    )  # fmt: skip

    await assistant.handle_text("Merk dir als Sichern: Speichern")
    await assistant.handle_text("Sichern")
    assert assistant.awaiting_answer  # typing reaches far: Flay asks first
    await assistant.handle_text("Ja")

    assert "Ich habe mir Sichern gemerkt." in tts.spoken[0][0]
    assert keyboard.calls == [{"shortcut": "save"}]


# Clipboard -------------------------------------------------------------------------------


async def test_clipboard_goes_with_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from flay import assistant as module
    from flay.assistant import Assistant
    from tests.test_assistant import FakeLLM

    monkeypatch.setattr(module, "clipboard_text", lambda: "Bonjour tout le monde")
    llm = FakeLLM("Hallo Welt.")
    assistant = Assistant(Settings(), FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput())

    await assistant.handle_text("Übersetz die Zwischenablage")

    assert "Bonjour tout le monde" in llm.prompts[0][-1].text


# Plans: a screenshot step hands on what it shows -------------------------------------------


async def test_screenshot_step_is_read_for_the_next_step() -> None:
    from flay.assistant import Assistant

    shot = Recording(
        "look_at_screen",
        ToolResult("Screenshot taken.", images=(ImagePart(b"png", "image/png"),)),
    )
    send = Recording("send_to_claude", ToolResult("typed", say={Language.DE: "Steht drin."}))

    class Llm(RoutingLLM):
        async def stream(self, messages: Any, tools: Any = ()) -> Any:
            if not tools:
                yield TextDelta("Ticket: Login schlägt fehl")
                return
            async for event in super().stream(messages, tools):
                yield event

    llm = Llm({"instruction": "x"})
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput(),
        tools=ToolRegistry([shot, send]),
    )  # fmt: skip

    await assistant.handle_text("Schau dir meinen Screen an und kopiere den Text in Claude")
    assert assistant.awaiting_answer  # sending to Claude reaches far: asked first
    await assistant.handle_text("Ja")

    assert shot.calls and send.calls


# History ---------------------------------------------------------------------------------


PROFILE = {
    "phases": {"state": 0.05, "route": 0.6},
    "marks": {"route": 0.7, "first_audio": 1.1, "end": 1.5},
    "calls": [
        {"purpose": "route", "model": "m", "seconds": 0.6, "first_token": 0.4,
         "prompt_tokens": 800, "output_tokens": 20, "images": 0},
    ],
}  # fmt: skip


def test_turn_log_records_and_summarizes(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    clock = iter([0.0, 2.5, 10.0, 11.0])
    log = TurnLog(path, 30, clock=lambda: next(clock))
    log.handle(UserMessage("Öffne Firefox", Language.DE, "voice"))
    log.handle(ToolUsed("control_window", {"app": "Firefox"}, "opened", True))
    log.handle(AssistantMessage("Firefox ist offen.", Language.DE))
    log.handle(StateChanged(AssistantState.IDLE))
    log.handle(TurnProfiled(PROFILE))
    log.handle(UserMessage("Klick auf Senden", Language.DE, "text"))
    log.handle(ToolUsed("click_on_screen", {"text": "Senden"}, "not found", False))
    log.handle(StateChanged(AssistantState.IDLE))
    log.handle(TurnProfiled(PROFILE))

    turns = list(read(path))

    assert [t.text for t in turns] == ["Öffne Firefox", "Klick auf Senden"]
    assert turns[0].seconds == 2.5 and not turns[0].failed and turns[1].failed
    summary = "\n".join(stats(turns))
    assert "Anfragen: 2" in summary and "click_on_screen" in summary
    assert '- say: "Klick auf Senden"' in eval_cases(turns)


def test_turn_log_drops_old_entries(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    old = {"time": "2020-01-01T10:00:00", "text": "alt", "language": "de", "source": "voice"}
    path.write_text(json.dumps(old) + "\n", "utf-8")
    log = TurnLog(path, 30)
    log.handle(UserMessage("neu", Language.DE, "text"))
    log.handle(StateChanged(AssistantState.IDLE))
    log.handle(TurnProfiled(PROFILE))
    assert [t.text for t in read(path)] == ["neu"]


@pytest.mark.parametrize(
    ("said", "tool"),
    [
        ("Sprich bitte etwas langsamer.", "adjust_settings"),
        ("Antworte ab jetzt kürzer.", "adjust_settings"),
        ("Schau bitte nicht mehr auf meinen Bildschirm.", "adjust_settings"),
        ("Mach mal Slack auf", "control_window"),
        ("Mach in Slack einen neuen Kanal namens test auf.", None),
        ("Streich das letzte Wort.", "use_keyboard"),
    ],
)
def test_computer_intents_of_this_phase(said: str, tool: str | None) -> None:
    from flay.intents import computer_intent
    from flay.tools.apps import App

    assert computer_intent(said, [App("Slack"), App("Firefox")]) == tool


async def test_computer_settings_are_not_flays_settings() -> None:
    llm = JsonLLM({"action": "adjust_settings"})
    found = await classify(
        llm,
        "Stell in den Systemeinstellungen den Dunkelmodus an",
        ["adjust_settings", "operate_computer"],
    )
    assert found == "operate_computer"


def test_turn_log_notices_a_request_failing_again(tmp_path: Path) -> None:
    log = TurnLog(tmp_path / "history.jsonl", 30)
    stuck: list[str] = []
    log.on_repeated_failure = stuck.append
    for text in ("Klick auf Senden", "Klick auf den Senden-Knopf"):
        log.handle(UserMessage(text, Language.DE, "voice"))
        log.handle(ToolUsed("click_on_screen", {}, "not found", False))
        log.handle(StateChanged(AssistantState.IDLE))
        log.handle(TurnProfiled(PROFILE))
    log.handle(TurnProfiled(PROFILE))
    assert stuck == ["Klick auf den Senden-Knopf"]


def test_settle_waits_until_the_screen_is_still(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    from flay.tools import pointer

    monkeypatch.setattr(pointer, "SETTLE_EVERY", 0)
    blank, half, full = (Image.new("L", (10, 10), v) for v in (0, 0, 255))
    half.paste(255, (0, 0, 5, 10))
    frames = iter([half, full, full, full])
    ticks = iter(range(100))

    share = pointer.settle(blank, lambda: next(frames), clock=lambda: next(ticks) * 0.1)

    assert share == 1.0  # waited for the menu to be drawn completely
    nothing = pointer.settle(blank, lambda: blank, clock=lambda: next(ticks) * 0.1)
    assert nothing == 0.0


async def test_solutions_found_by_meaning(tmp_path: Path) -> None:
    import numpy as np

    class Embedder:
        async def embed(self, texts: Sequence[str], query: bool = False) -> Any:
            # "Firefox" goals point one way, everything else another.
            return np.array([[1.0, 0.0] if "Firefox" in t else [0.0, 1.0] for t in texts])

    solutions = Solutions(tmp_path / "s.json", Embedder())
    solutions.add("Schieb Firefox auf den zweiten Monitor", ["app Firefox 'move:2'"])

    found = await solutions.find("Firefox bitte auf den anderen Bildschirm")
    assert found is not None and found.moves == ["app Firefox 'move:2'"]
    assert await solutions.find("Stell einen Wecker") is None


async def test_short_sentences_are_synthesized_once() -> None:
    from flay.tts.cache import CachedTTS

    inner = FakeTTS()
    tts = CachedTTS(inner)  # type: ignore[arg-type]
    for _ in range(3):
        chunks = [c async for c in tts.synthesize("Erledigt.", Language.DE)]
        assert chunks
    long = "Das ist ein langer Satz, der sich so bestimmt nicht noch einmal wiederholt, ehrlich."
    for _ in range(2):
        [c async for c in tts.synthesize(long, Language.DE)]

    assert inner.spoken == [("Erledigt.", Language.DE), (long, Language.DE), (long, Language.DE)]


def test_a_covered_private_window_does_not_count(private: None) -> None:
    editor = screen.Window(1, "PhpStorm", "flay", (0, 0, 1000, 800))
    vault = screen.Window(2, "1Password", "Tresor", (100, 100, 400, 400))
    assert screen.private_window((0, 0, 1000, 800), [editor, vault]) is None
    peeking = screen.Window(3, "1Password", "Tresor", (900, 100, 400, 400))
    assert screen.private_window((0, 0, 1400, 800), [editor, peeking]) == peeking


async def test_computer_does_not_repeat_a_move() -> None:
    keyboard = Recording("use_keyboard")
    typing = {"thought": "", "move": "type", "target": "", "text": "Hallo"}
    llm = JsonLLM(typing, typing, {"thought": "", "move": "done", "target": "", "text": "Ok."})

    await computer(llm, keyboard).run({"goal": "Schreib Hallo"})

    assert keyboard.calls == [{"text": "Hallo"}]
    assert "not done again" in llm.requests[2][1].text


def test_click_label_from_what_the_model_wrote() -> None:
    from flay.tools.computer import _label

    assert _label("Sound (button) or Ton (button) in System Settings menu.") == "Sound"
    assert _label("Search Everywhere (button)") == "Search Everywhere"
    assert _label("Save in Folder") == "Save in Folder"


def test_german_app_names() -> None:
    from flay.tools.apps import App, match_app, windows_of

    settings = App("System Settings")
    assert match_app("Systemeinstellungen", [App("Firefox"), settings]) == settings
    window = screen.Window(1, "Systemeinstellungen", "", (0, 0, 800, 600))
    assert windows_of(settings, [window]) == [window]


def test_perf_stats_sum_up_profiles() -> None:
    from flay.history import perf_stats

    text = "\n".join(perf_stats([PROFILE, PROFILE]))
    assert "first_audio" in text and "1.10 s" in text
    assert "route" in text and "800→20" in text and "2000 tok/s" in text


async def test_a_turn_is_profiled(monkeypatch: pytest.MonkeyPatch) -> None:
    from flay.assistant import Assistant
    from tests.test_assistant import FakeLLM

    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), FakeLLM("Hallo."), FakeTTS(), FakeOutput()
    )
    profiles: list[dict[str, Any]] = []
    publish = assistant.events.publish

    def spy(event: Any) -> None:
        if isinstance(event, TurnProfiled):
            profiles.append(event.profile)
        publish(event)

    monkeypatch.setattr(assistant.events, "publish", spy)
    await assistant.handle_text("Wie geht es dir?")

    assert profiles and "first_audio" in profiles[0]["marks"] and "end" in profiles[0]["marks"]


async def test_a_claimed_action_after_a_story_is_not_carried_out() -> None:
    from flay.assistant import Assistant
    from tests.test_assistant import FakeLLM

    closing = Recording("close_project", ToolResult("closed", say={Language.DE: "Zu."}))
    llm = FakeLLM("Ich habe das Projekt flay geschlossen.")
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput(),
        tools=ToolRegistry([closing]), projects=lambda: ["flay"],
    )  # fmt: skip

    await assistant.handle_text("Ich habe gestern das Projekt flay geschlossen.")

    assert closing.calls == []
