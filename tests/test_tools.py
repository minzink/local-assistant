import asyncio
import os
import shutil
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from flay.assistant import MAX_TOOL_ROUNDS, Assistant
from flay.config import Settings
from flay.core.events import ToolUsed
from flay.core.language import Language
from flay.core.messages import ImagePart, Message, Modality, Role, TextPart, ToolCall
from flay.llm.base import LLMEvent, ModelCapabilities, TextDelta, ToolSpec
from flay.tools.base import ToolRegistry, ToolResult, speech
from flay.tools.claude_code import (
    ClaudeStatusTool,
    SendToClaudeTool,
    Session,
    Tmux,
    claude_state,
    pick_session,
    session_name,
    single_line,
)
from flay.tools.factory import create_tools
from flay.tools.phpstorm import (
    OpenProjectTool,
    Project,
    latest_config_dir,
    match_projects,
    read_recent_projects,
)
from flay.tools.screen import LookAtScreenTool, Window
from tests.test_assistant import ONE_SECOND, FakeOutput, FakeSource, FakeSTT, FakeTTS

RECENT_PROJECTS = """<application>
  <component name="RecentProjectsManager">
    <option name="additionalInfo">
      <map>
        <entry key="$USER_HOME$/phpstormprojects/stadt-wolkenthal"><value/></entry>
        <entry key="$USER_HOME$/phpstormprojects/bvx-cms"><value/></entry>
        <entry key="$USER_HOME$/phpstormprojects/tlr-cms"><value/></entry>
        <entry key="$USER_HOME$/phpstormprojects/ai/flay"><value/></entry>
      </map>
    </option>
  </component>
</application>"""


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    for version in ["PhpStorm2024.1", "PhpStorm2026.2", "PhpStorm2026.10"]:
        (tmp_path / version / "options").mkdir(parents=True)
    newest = tmp_path / "PhpStorm2026.10"
    (newest / "options" / "recentProjects.xml").write_text(RECENT_PROJECTS, encoding="utf-8")
    return newest


# PhpStorm ------------------------------------------------------------------------------


def test_latest_config_dir_compares_versions_numerically(tmp_path: Path, config_dir: Path) -> None:
    assert latest_config_dir(tmp_path) == config_dir


def test_recent_projects_expand_home(config_dir: Path) -> None:
    projects = read_recent_projects(config_dir)

    assert [p.name for p in projects] == ["stadt-wolkenthal", "bvx-cms", "tlr-cms", "flay"]
    assert projects[0].path == Path.home() / "phpstormprojects" / "stadt-wolkenthal"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Stadt Wolkenthal", ["stadt-wolkenthal"]),
        ("flay", ["flay"]),
        ("cms", ["bvx-cms", "tlr-cms"]),
        ("stadt wolkental", ["stadt-wolkenthal"]),
        ("unbekannt", []),
    ],
)
def test_match_projects(query: str, expected: list[str], config_dir: Path) -> None:
    matches = match_projects(query, read_recent_projects(config_dir))

    assert [p.name for p in matches] == expected


@pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script as launcher")
async def test_open_project_runs_launcher(tmp_path: Path) -> None:
    project = tmp_path / "stadt-wolkenthal"
    project.mkdir()
    log = tmp_path / "launched.txt"
    launcher = tmp_path / "phpstorm"
    launcher.write_text(f'#!/bin/sh\necho "$@" > {log}\n')
    launcher.chmod(0o755)
    tool = OpenProjectTool(launcher=launcher)
    tool.projects = lambda: [Project("stadt-wolkenthal", project)]  # type: ignore[method-assign]

    result = await tool.run({"name": "Stadt Wolkenthal"})

    assert result.ok
    assert log.read_text().strip() == str(project)


async def test_open_project_reports_ambiguity_and_unknown(config_dir: Path) -> None:
    tool = OpenProjectTool(launcher=Path("/nonexistent"), config_dir=config_dir)

    ambiguous = await tool.run({"name": "cms"})
    unknown = await tool.run({"name": "gibtsnicht"})

    assert not ambiguous.ok and "bvx-cms" in ambiguous.text and "tlr-cms" in ambiguous.text
    assert not unknown.ok and "stadt-wolkenthal" in unknown.text


# Claude Code ---------------------------------------------------------------------------


class FakeTmux:
    available = True

    def __init__(self, sessions: list[Session], screens: list[str] | None = None) -> None:
        self._sessions = sessions
        self._screens = screens
        self.typed: list[tuple[str, str]] = []
        self.started: list[tuple[str, Path, list[str]]] = []
        self.pressed: list[tuple[str, str, int]] = []

    async def sessions(self) -> list[Session]:
        return self._sessions

    async def start(self, session: str, cwd: Path, command: list[str]) -> None:
        self.started.append((session, cwd, command))

    async def type_text(self, session: str, text: str) -> None:
        self.typed.append((session, text))

    async def press(self, session: str, key: str, count: int = 1) -> None:
        self.pressed.append((session, key, count))

    async def screen(self, session: str, lines: int = 40) -> str:
        if self._screens is not None:
            return self._screens.pop(0) if len(self._screens) > 1 else self._screens[0]
        return f"screen of {session}\n──────────\n❯\u00a0\n──────────"


READY_SCREEN = """ ▐▛███▛█   Claude Code v2.1.310
 ▝▝   ▝▝   ~/phpstormprojects/stadt-wolkenthal
──────────────────────
❯ Try "how does microphone.py work?"
──────────────────────
  ⏵⏵ auto mode on (shift+tab to cycle)"""

TRUST_SCREEN = """ Accessing workspace:
 /Users/x/phpstormprojects/stadt-wolkenthal
 Quick safety check: Is this a project you created or one you trust?
 ❯ No, exit
   Yes, I trust this folder
 Enter to confirm · Esc to cancel"""

STARTING_SCREEN = ""


def autostart_tool(tmux: FakeTmux, tmp_path: Path, timeout: float = 1.0) -> SendToClaudeTool:
    project = Project("stadt-wolkenthal", tmp_path / "stadt-wolkenthal")

    def find(name: str) -> list[Project]:
        return [project] if name == "Stadt Wolkenthal" else []

    return SendToClaudeTool(
        tmux=tmux,  # type: ignore[arg-type]
        find=find,
        claude_command=["claude"],
        ready_timeout=timeout,
    )


@pytest.mark.parametrize(
    ("screen", "state"),
    [(READY_SCREEN, "ready"), (TRUST_SCREEN, "trust"), (STARTING_SCREEN, None)],
)
def test_claude_state(screen: str, state: str | None) -> None:
    assert claude_state(screen) == state


async def test_autostart_starts_claude_and_types_when_ready(tmp_path: Path) -> None:
    tmux = FakeTmux([], screens=[STARTING_SCREEN, STARTING_SCREEN, READY_SCREEN])
    tool = autostart_tool(tmux, tmp_path, timeout=5)

    result = await tool.run({"instruction": "Prüfe die README.", "project": "Stadt Wolkenthal"})

    assert result.ok
    assert tmux.started == [
        ("flay-claude-stadt-wolkenthal", tmp_path / "stadt-wolkenthal", ["claude"])
    ]
    assert tmux.typed == [("flay-claude-stadt-wolkenthal", "Prüfe die README.")]
    assert "flay claude" in result.text
    assert result.action is not None and "gestartet" in result.action


async def test_autostart_types_nothing_into_trust_prompt(tmp_path: Path) -> None:
    tmux = FakeTmux([], screens=[TRUST_SCREEN])

    result = await autostart_tool(tmux, tmp_path).run(
        {"instruction": "x", "project": "Stadt Wolkenthal"}
    )

    assert not result.ok
    assert tmux.typed == []
    assert "trust" in result.text


async def test_autostart_gives_up_when_claude_never_gets_ready(tmp_path: Path) -> None:
    tmux = FakeTmux([], screens=[STARTING_SCREEN])

    result = await autostart_tool(tmux, tmp_path, timeout=0.6).run(
        {"instruction": "x", "project": "Stadt Wolkenthal"}
    )

    assert not result.ok
    assert tmux.typed == []


async def test_named_project_without_session_is_not_typed_into_other_project(
    tmp_path: Path,
) -> None:
    tmux = FakeTmux([FLAY], screens=[READY_SCREEN])

    await autostart_tool(tmux, tmp_path).run({"instruction": "x", "project": "Stadt Wolkenthal"})

    assert tmux.typed == [("flay-claude-stadt-wolkenthal", "x")]


async def test_unresolvable_project_falls_back_to_clipboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied: list[str] = []
    monkeypatch.setattr("flay.tools.claude_code.copy_to_clipboard", copied.append)
    monkeypatch.setattr("flay.tools.claude_code.bring_phpstorm_to_front", lambda: None)
    tmux = FakeTmux([])

    result = await autostart_tool(tmux, tmp_path).run(
        {"instruction": "x", "project": "gibts nicht"}
    )

    assert copied == ["x"]
    assert tmux.started == []
    assert "clipboard" in result.text


async def test_without_session_and_project_flay_asks(tmp_path: Path) -> None:
    tmux = FakeTmux([])

    result = await autostart_tool(tmux, tmp_path).run({"instruction": "x"})

    assert not result.ok
    assert "Ask the user which project" in result.text
    assert tmux.started == [] and tmux.typed == []


FLAY = Session("flay-claude-flay", last_activity=200)
WOLKENTHAL = Session("flay-claude-stadt-wolkenthal", last_activity=100)


def test_session_name() -> None:
    assert session_name(Path("/x/Stadt Wolkenthal_2")) == "flay-claude-stadt-wolkenthal-2"


@pytest.mark.parametrize(
    ("project", "expected"),
    [(None, FLAY), ("Stadt Wolkenthal", WOLKENTHAL), ("unbekannt", None)],
)
def test_pick_session(project: str | None, expected: Session | None) -> None:
    assert pick_session([WOLKENTHAL, FLAY], project) == expected


def test_pick_session_prefers_the_session_the_user_has_open() -> None:
    open_wolkenthal = Session("flay-claude-stadt-wolkenthal", last_activity=100, attached=True)

    assert pick_session([open_wolkenthal, FLAY], None) == open_wolkenthal


async def test_send_to_claude_types_into_tmux_session_without_enter() -> None:
    tmux = FakeTmux([WOLKENTHAL, FLAY])
    tool = SendToClaudeTool(tmux=tmux)  # type: ignore[arg-type]

    result = await tool.run(
        {"instruction": "Füge Tests\nfür den Login hinzu.", "project": "wolkenthal"}
    )

    assert result.ok
    assert tmux.typed == [("flay-claude-stadt-wolkenthal", "Füge Tests für den Login hinzu.")]
    assert "NOT sent" in result.text


async def test_send_to_claude_falls_back_to_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    copied: list[str] = []
    fronted: list[bool] = []
    monkeypatch.setattr("flay.tools.claude_code.copy_to_clipboard", copied.append)
    monkeypatch.setattr(
        "flay.tools.claude_code.bring_phpstorm_to_front", lambda: fronted.append(True)
    )
    tool = SendToClaudeTool(tmux=FakeTmux([]), autostart=False)  # type: ignore[arg-type]

    result = await tool.run({"instruction": "Prüfe die README."})

    assert result.ok
    assert copied == ["Prüfe die README."]
    assert fronted == [True]
    assert "clipboard" in result.text


async def test_tmux_only_transport_reports_missing_session() -> None:
    tool = SendToClaudeTool(transport="tmux", autostart=False, tmux=FakeTmux([]))  # type: ignore[arg-type]

    result = await tool.run({"instruction": "x"})

    assert not result.ok


async def test_claude_status_reads_screen() -> None:
    ok = await ClaudeStatusTool(FakeTmux([FLAY])).run({})  # type: ignore[arg-type]
    missing = await ClaudeStatusTool(FakeTmux([])).run({})  # type: ignore[arg-type]

    assert ok.ok and "screen of flay-claude-flay" in ok.text
    assert not missing.ok


@pytest.mark.skipif(shutil.which("tmux") is None, reason="needs tmux")
async def test_real_tmux_session_receives_text_without_enter() -> None:
    # A separate tmux server, so the user's own sessions are never touched.
    socket = ["-L", f"flay-test-{os.getpid()}"]

    class SocketTmux(Tmux):
        async def _run(self, *args: str) -> str:
            return await super()._run(*socket, *args)

    tmux = SocketTmux()
    name = "flay-claude-test"
    await tmux._run("new-session", "-d", "-s", name, "-x", "80", "-y", "10", "cat")
    try:
        assert [s.name for s in await tmux.sessions()] == [name]
        await tmux.type_text(name, "Hallo Claude; echo nicht ausführen")
        await asyncio.sleep(0.2)
        screen = await tmux.screen(name)
    finally:
        socket_path = (await tmux._run("display-message", "-p", "#{socket_path}")).strip()
        await tmux._run("kill-server")
        Path(socket_path).unlink(missing_ok=True)

    # cat echoes only after Enter: the text sits in the input line, unsent.
    assert screen.splitlines()[0] == "Hallo Claude; echo nicht ausführen"
    assert len([line for line in screen.splitlines() if line]) == 1


def test_single_line() -> None:
    assert single_line("  a\n\nb\tc ") == "a b c"


# Screen ---------------------------------------------------------------------------------


async def test_screen_tool_returns_png(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flay.tools.screen.capture_png", lambda monitor, size: b"\x89PNG")

    result = await LookAtScreenTool().run({})

    assert result.images == (ImagePart(b"\x89PNG", "image/png"),)


# Registry and factory -------------------------------------------------------------------


class EchoTool:
    def __init__(self, fail: bool = False, images: tuple[ImagePart, ...] = ()) -> None:
        self.fail = fail
        self.images = images
        self.calls: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec("echo", "Echo", {"type": "object", "properties": {}})

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(arguments)
        if self.fail:
            raise RuntimeError("kaputt")
        return ToolResult(f"echo {arguments}", images=self.images)


async def test_registry_handles_unknown_and_failing_tools() -> None:
    registry = ToolRegistry([EchoTool(fail=True)])

    unknown = await registry.execute(ToolCall("1", "nope", {}))
    failed = await registry.execute(ToolCall("2", "echo", {}))

    assert not unknown.ok and "Unknown tool" in unknown.text
    assert not failed.ok and "kaputt" in failed.text


def test_factory_drops_screen_tool_for_text_only_models() -> None:
    settings = Settings.model_validate({"llm": {"input_modalities": ["text"]}})

    names = [spec.name for spec in create_tools(settings).specs]

    assert names == [
        "open_project", "close_project", "send_to_claude", "claude_status", "open_url",
        "fill_web_form", "read_file", "workflow", "ssh_connect",
    ]  # fmt: skip


def test_factory_respects_enabled_list() -> None:
    settings = Settings.model_validate({"tools": {"enabled": ["open_project"]}})

    assert [spec.name for spec in create_tools(settings).specs] == ["open_project"]


# Tool loop in the assistant ------------------------------------------------------------


class ToolCallingLLM:
    """Round 1: calls the echo tool. Round 2: answers with text."""

    def __init__(self) -> None:
        self.prompts: list[tuple[list[Message], list[ToolSpec]]] = []

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(frozenset({Modality.TEXT, Modality.IMAGE}))

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.prompts.append((list(messages), list(tools)))
        if len(self.prompts) == 1:
            yield TextDelta("Moment. ")
            yield ToolCall("call_1", "echo", {"x": 1})
        else:
            yield TextDelta("Erledigt.")


async def test_assistant_runs_tool_and_continues_with_result() -> None:
    tool = EchoTool(images=(ImagePart(b"png", "image/png"),))
    llm, tts = ToolCallingLLM(), FakeTTS()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("Mach was"),
        llm,
        tts,
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )
    events: list[object] = []
    assistant.events.publish = events.append  # type: ignore[assignment]

    await assistant.handle(ONE_SECOND)

    assert tool.calls == [{"x": 1}]
    second_prompt, second_tools = llm.prompts[1]
    assistant_call, tool_result, screenshot = second_prompt[-3:]
    assert assistant_call.tool_calls == (ToolCall("call_1", "echo", {"x": 1}),)
    assert tool_result.role is Role.TOOL and tool_result.tool_call_id == "call_1"
    assert screenshot.role is Role.USER and ImagePart(b"png", "image/png") in screenshot.parts
    assert [s.name for s in second_tools] == ["echo"]
    assert ToolUsed("echo", {"x": 1}, "echo {'x': 1}", True) in events
    assert [text for text, _ in tts.spoken] == ["Moment.", "Erledigt."]
    assert assistant.history[-1].text == "Moment. Erledigt."
    assert "act on the user's computer" in llm.prompts[0][0][0].text


async def test_text_of_consecutive_rounds_is_separated() -> None:
    class NoSpaceLLM(ToolCallingLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            self.prompts.append((list(messages), list(tools)))
            if len(self.prompts) == 1:
                yield TextDelta("Ich schaue nach.")
                yield ToolCall("c1", "echo", {})
            else:
                yield TextDelta("Fertig.")

    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        NoSpaceLLM(),
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([EchoTool()]),
    )

    await assistant.handle(ONE_SECOND)

    assert assistant.history[-1].text == "Ich schaue nach. Fertig."


async def test_tool_rounds_are_limited() -> None:
    class AlwaysTools(ToolCallingLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            self.prompts.append((list(messages), list(tools)))
            if tools:
                yield ToolCall(f"c{len(self.prompts)}", "echo", {})
            else:
                yield TextDelta("Fertig.")

    llm, tool = AlwaysTools(), EchoTool()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )

    await assistant.handle(ONE_SECOND)

    assert len(tool.calls) == MAX_TOOL_ROUNDS
    assert llm.prompts[-1][1] == []
    assert assistant.history[-1].text == "Fertig."


def test_blank_screenshot_raises_permission_hint(
    monkeypatch: pytest.MonkeyPatch, permission_granted: None
) -> None:
    import mss

    from flay.tools.screen import capture_png

    class Shot:
        size = (4, 2)
        bgra = bytes([45, 45, 49, 255] * 8)

    class FakeMss:
        monitors = [{}, {}]

        def __enter__(self) -> "FakeMss":
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def grab(self, monitor: object) -> Shot:
            return Shot()

    monkeypatch.setattr(mss, "MSS", FakeMss, raising=False)
    monkeypatch.setattr(mss, "mss", FakeMss)

    with pytest.raises(PermissionError, match="Screen Recording"):
        capture_png(1, 1280)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Schau dir meinen Bildschirm an.", True),
        ("Was siehst du?", True),
        ("Guck mal hier, was ist das für ein Fehler?", True),
        ("What do you see on my screen?", True),
        ("Look at this error", True),
        ("Öffne das Projekt flay", False),
        ("Wie spät ist es?", False),
    ],
)
def test_mentions_screen(text: str, expected: bool) -> None:
    from flay.assistant import mentions_screen

    assert mentions_screen(text) is expected


class ScreenLLM(ToolCallingLLM):
    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.prompts.append((list(messages), list(tools)))
        yield TextDelta("Ich sehe einen Fehler.")


class FakeScreenTool(EchoTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec("look_at_screen", "Screen", {"type": "object", "properties": {}})


async def test_screen_mention_attaches_screenshot_without_tool_round() -> None:
    screen = FakeScreenTool(images=(ImagePart(b"png", "image/png"),))
    llm = ScreenLLM()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([screen]),
    )

    await assistant.handle_text("Schau dir meinen Bildschirm an")

    assert len(llm.prompts) == 1
    user = llm.prompts[0][0][-1]
    assert ImagePart(b"png", "image/png") in user.parts
    # The history keeps the words only, so the image is not sent with every later turn.
    assert assistant.history[0].parts == (TextPart("Schau dir meinen Bildschirm an"),)


async def test_failed_screenshot_is_answered_without_the_model() -> None:
    screen = FakeScreenTool(fail=True)
    llm = ScreenLLM()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([screen]),
    )

    await assistant.handle_text("Was siehst du auf dem Bildschirm?")

    # Without a picture the model would only make up what it "sees".
    assert llm.prompts == []
    reply = assistant.history[-1]
    assert reply.text == "Das hat leider nicht geklappt, Details stehen im Log."
    assert [(a.call.name, a.ok) for a in reply.actions] == [("look_at_screen", False)]


async def test_registry_logs_actions(caplog: pytest.LogCaptureFixture) -> None:
    class ActingTool(EchoTool):
        async def run(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult("done", action="PhpStorm: Projekt geöffnet")

    with caplog.at_level("INFO", logger="flay.tools"):
        await ToolRegistry([ActingTool()]).execute(ToolCall("1", "echo", {}))

    assert "PhpStorm: Projekt geöffnet" in caplog.messages


async def test_clipboard_fallback_says_nothing_was_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flay.tools.claude_code.copy_to_clipboard", lambda text: None)
    monkeypatch.setattr("flay.tools.claude_code.bring_phpstorm_to_front", lambda: None)

    tool = SendToClaudeTool(tmux=FakeTmux([]), autostart=False)  # type: ignore[arg-type]
    result = await tool.run({"instruction": "x"})

    assert "nothing was written into Claude" in result.text
    assert result.action is not None and "Zwischenablage" in result.action


# PhpStorm MCP --------------------------------------------------------------------------


def test_parse_open_projects() -> None:
    from flay.tools.phpstorm_mcp import parse_open_projects

    text = (
        "Unable to determine the target project for the current MCP tool call.\n"
        ' Currently open projects: {"projects":[{"path":"/p/umx"},{"path":"/p/ai/flay"}]}'
    )

    assert parse_open_projects(text) == [Path("/p/umx"), Path("/p/ai/flay")]
    assert parse_open_projects("all fine") == []


def test_projects_from_claude_locks(tmp_path: Path) -> None:
    import json

    from flay.tools.phpstorm_mcp import projects_from_claude_locks

    def lock(name: str, pid: int, ide: str, folder: str) -> None:
        data = {"pid": pid, "ideName": ide, "workspaceFolders": [folder], "authToken": "x"}
        (tmp_path / f"{name}.lock").write_text(json.dumps(data))

    lock("1", os.getpid(), "PhpStorm", "/p/flay")
    lock("2", 999_999_999, "PhpStorm", "/p/closed")
    lock("3", os.getpid(), "PyCharm", "/p/other-ide")
    (tmp_path / "4.lock").write_text("kaputt")

    assert projects_from_claude_locks(tmp_path) == [Path("/p/flay")]


class FakePhpStorm:
    def __init__(self, projects: list[Path], tmux: FakeTmux, starts_session: bool = True) -> None:
        self.projects = projects
        self.tmux = tmux
        self.starts_session = starts_session
        self.commands: list[tuple[str, Path]] = []

    async def open_projects(self) -> list[Path]:
        return self.projects

    async def open_claude(self, project: Path) -> Path:
        self.commands.append(("open_claude", project))
        if self.starts_session:
            self.tmux._sessions.append(Session(session_name(project), 1, attached=True))
        return self.launcher or project / ".flay-claude.sh"

    launcher: Path | None = None


def phpstorm_tool(tmux: FakeTmux, phpstorm: FakePhpStorm, tmp_path: Path) -> SendToClaudeTool:
    tool = autostart_tool(tmux, tmp_path, timeout=2)
    tool._phpstorm = phpstorm  # type: ignore[assignment]
    tool._confirm_timeout = 1.0
    return tool


async def test_single_open_project_gets_claude_terminal(tmp_path: Path) -> None:
    tmux = FakeTmux([], screens=[READY_SCREEN])
    phpstorm = FakePhpStorm([Path("/p/ai/flay")], tmux)

    result = await phpstorm_tool(tmux, phpstorm, tmp_path).run({"instruction": "Prüfe die README."})

    assert result.ok
    assert phpstorm.commands[0][0] == "open_claude"
    assert phpstorm.commands[0][1] == Path("/p/ai/flay")
    assert tmux.typed == [("flay-claude-flay", "Prüfe die README.")]
    assert tmux.started == []  # no hidden background session


async def test_named_open_project_gets_claude_terminal(tmp_path: Path) -> None:
    tmux = FakeTmux([], screens=[READY_SCREEN])
    phpstorm = FakePhpStorm([Path("/p/umx"), Path("/p/ai/flay")], tmux)

    await phpstorm_tool(tmux, phpstorm, tmp_path).run({"instruction": "x", "project": "flay"})

    assert [project for _, project in phpstorm.commands] == [Path("/p/ai/flay")]


async def test_several_open_projects_without_name_asks(tmp_path: Path) -> None:
    tmux = FakeTmux([])
    phpstorm = FakePhpStorm([Path("/p/umx"), Path("/p/ai/flay")], tmux)

    result = await phpstorm_tool(tmux, phpstorm, tmp_path).run({"instruction": "x"})

    assert not result.ok
    assert "umx, flay" in result.text
    assert phpstorm.commands == []


async def test_named_project_not_open_uses_background_session(tmp_path: Path) -> None:
    tmux = FakeTmux([], screens=[READY_SCREEN])
    phpstorm = FakePhpStorm([Path("/p/ai/flay")], tmux)

    result = await phpstorm_tool(tmux, phpstorm, tmp_path).run(
        {"instruction": "x", "project": "Stadt Wolkenthal"}
    )

    assert result.ok
    assert phpstorm.commands == []
    assert tmux.started[0][0] == "flay-claude-stadt-wolkenthal"


async def test_unconfirmed_terminal_types_nothing(tmp_path: Path) -> None:
    tmux = FakeTmux([])
    phpstorm = FakePhpStorm([Path("/p/ai/flay")], tmux, starts_session=False)

    result = await phpstorm_tool(tmux, phpstorm, tmp_path).run({"instruction": "x"})

    assert not result.ok
    assert tmux.typed == []
    assert "did not come up" in result.text


async def test_detached_session_is_shown_in_phpstorm_before_typing(tmp_path: Path) -> None:
    detached = Session("flay-claude-flay", 1, attached=False)
    tmux = FakeTmux([detached])
    phpstorm = FakePhpStorm([Path("/p/ai/flay")], tmux)
    # `flay claude` attaches to the existing session instead of creating another one.
    phpstorm.starts_session = False

    async def attach(project: Path) -> Path:
        phpstorm.commands.append(("open_claude", project))
        tmux._sessions[0] = Session("flay-claude-flay", 2, attached=True)
        return project / ".flay-claude.sh"

    phpstorm.open_claude = attach  # type: ignore[method-assign]

    result = await phpstorm_tool(tmux, phpstorm, tmp_path).run(
        {"instruction": "x", "project": "flay"}
    )

    assert result.ok and "background" not in result.text
    assert [project for _, project in phpstorm.commands] == [Path("/p/ai/flay")]
    assert tmux.typed == [("flay-claude-flay", "x")]


async def test_detached_session_of_closed_project_is_typed_with_hint(tmp_path: Path) -> None:
    tmux = FakeTmux([Session("flay-claude-flay", 1, attached=False)])
    phpstorm = FakePhpStorm([Path("/p/umx")], tmux)

    result = await phpstorm_tool(tmux, phpstorm, tmp_path).run(
        {"instruction": "x", "project": "flay"}
    )

    assert phpstorm.commands == []
    assert tmux.typed == [("flay-claude-flay", "x")]
    assert "background" in result.text and "flay claude" in result.text


def test_describe_flattens_exception_groups() -> None:
    from flay.tools.phpstorm_mcp import describe

    error = ExceptionGroup("outer", [ExceptionGroup("inner", [ConnectionError("refused")])])

    assert describe(error) == "ConnectionError: refused"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sag Claude im Projekt Flay, er soll die README zusammenfassen.", "send_to_claude"),
        (
            "Und so soll's Clodd in dem Projekt Flay sagen, er soll die Readme zusammenfassen.",
            "send_to_claude",
        ),
        ("Du sollst Cloud die Anweisung geben, die README zusammenzufassen.", "send_to_claude"),
        ("Claude, schreib Tests für den Login.", "send_to_claude"),
        ("Tell Claude to add tests.", "send_to_claude"),
        ("Ist Claude schon fertig?", "claude_status"),
        ("Wie ist der Stand bei Claude?", "claude_status"),
        ("Is Claude done?", "claude_status"),
        ("Was ist eine Cloud?", None),
        ("Gib mir Infos zur Cloud.", None),
        ("Sag Claude, er soll die Tests fertig schreiben.", "send_to_claude"),
        ("Öffne das Projekt flay", None),
    ],
)
def test_claude_intent(text: str, expected: str | None) -> None:
    from flay.assistant import claude_intent

    assert claude_intent(text) == expected


class RoutingLLM:
    """Ignores tools in the full conversation (like a model misled by its history), but
    calls the tool when asked in the minimal routing prompt."""

    def __init__(self) -> None:
        self.prompts: list[tuple[list[Message], list[ToolSpec]]] = []

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.prompts.append((list(messages), list(tools)))
        if len(messages) == 2 and [t.name for t in tools] == ["send_to_claude"]:
            yield ToolCall("r1", "send_to_claude", {"instruction": "Fasse die README zusammen."})
        else:
            yield TextDelta("Der Auftrag steht in Claude.")


class SendTool(EchoTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec("send_to_claude", "Claude", {"type": "object", "properties": {}})


async def test_claude_request_is_routed_even_if_history_misleads_the_model() -> None:
    tool, llm = SendTool(), RoutingLLM()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool, EchoTool()]),
    )

    await assistant.handle_text("Sag Claude, er soll die README zusammenfassen.")

    assert tool.calls == [{"instruction": "Fasse die README zusammen."}]
    main_messages, main_tools = llm.prompts[1]
    assert main_messages[-2].tool_calls[0].name == "send_to_claude"
    assert main_messages[-1].role is Role.TOOL
    # Already done: not offered again, so it cannot be typed twice.
    assert "send_to_claude" not in [t.name for t in main_tools]
    assert assistant.history[-1].text == "Der Auftrag steht in Claude."


async def test_routing_without_tool_call_falls_back_to_normal_answer() -> None:
    class NoCall(RoutingLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            self.prompts.append((list(messages), list(tools)))
            yield TextDelta("Was genau soll Claude tun?")

    tool, llm = SendTool(), NoCall()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )

    await assistant.handle_text("Sag Claude Bescheid.")

    assert tool.calls == []
    assert assistant.history[-1].text == "Was genau soll Claude tun?"


async def test_launcher_script_is_deleted_after_start(tmp_path: Path) -> None:
    launcher = tmp_path / ".flay-claude.sh"
    launcher.write_text("#!/bin/sh\n")
    tmux = FakeTmux([], screens=[READY_SCREEN])
    phpstorm = FakePhpStorm([Path("/p/ai/flay")], tmux)
    phpstorm.launcher = launcher

    result = await phpstorm_tool(tmux, phpstorm, tmp_path).run({"instruction": "x"})

    assert result.ok
    assert not launcher.exists()


def test_launcher_script_runs_flay_claude() -> None:
    from flay.tools.phpstorm_mcp import launcher_script

    script = launcher_script()

    assert script.startswith("#!/bin/sh\n")
    assert script.rstrip().endswith("claude")
    assert "exec " in script


LEFTOVER_SCREEN = """ ▝▝   ▝▝   ~/phpstormprojects/ai/flay
──────────────────────────────────────────────────
❯ [⧉ In README.md] Fasse die Readme-Datei des Projekts Flay kurz zusammen.Fasse die Readme des
  Projekts kurz zusammen.
──────────────────────────────────────────────────
  ⏵⏵ auto mode on (shift+tab to cycle)"""


def input_screen(text: str) -> str:
    return f"──────────────\n❯ {text}\n──────────────\n  ⏵⏵ auto mode on"


@pytest.mark.parametrize(
    ("screen", "expected"),
    [
        (READY_SCREEN, ""),
        (input_screen(""), ""),
        (f'{"─" * 20}\n❯\u00a0Try "edit config.py to..."\n{"─" * 20}', ""),
        (f"{'─' * 20}\n❯\u00a0Prüfe das.\n{'─' * 20}", "Prüfe das."),
        ("❯\n───────────────", ""),
        (input_screen("Prüfe die README."), "Prüfe die README."),
        (
            LEFTOVER_SCREEN,
            "Fasse die Readme-Datei des Projekts Flay kurz zusammen.Fasse die Readme des "
            "Projekts kurz zusammen.",
        ),
        ("Claude startet …", None),
    ],
)
def test_read_input(screen: str, expected: str | None) -> None:
    from flay.tools.claude_code import read_input

    assert read_input(screen) == expected


async def test_foreign_text_in_claude_input_is_never_overwritten() -> None:
    tmux = FakeTmux([FLAY], screens=[input_screen("Mein eigener Entwurf")])
    tool = SendToClaudeTool(tmux=tmux, autostart=False)  # type: ignore[arg-type]

    result = await tool.run({"instruction": "Neuer Auftrag."})

    assert not result.ok
    assert tmux.typed == [] and tmux.pressed == []
    assert "Mein eigener Entwurf" in result.text


async def test_own_unsent_text_is_replaced() -> None:
    tmux = FakeTmux(
        [FLAY], screens=[input_screen(""), input_screen("Erster Auftrag."), input_screen("")]
    )
    tool = SendToClaudeTool(tmux=tmux, autostart=False)  # type: ignore[arg-type]

    await tool.run({"instruction": "Erster Auftrag."})
    result = await tool.run({"instruction": "Zweiter Auftrag."})

    assert result.ok
    assert tmux.pressed == [("flay-claude-flay", "BSpace", len("Erster Auftrag."))]
    assert tmux.typed[-1] == ("flay-claude-flay", "Zweiter Auftrag.")


async def test_nothing_is_typed_when_the_input_line_is_not_visible() -> None:
    tmux = FakeTmux([FLAY], screens=["Do you want to proceed?\n 1. Yes\n 2. No"])
    tool = SendToClaudeTool(tmux=tmux, autostart=False)  # type: ignore[arg-type]

    result = await tool.run({"instruction": "x"})

    assert not result.ok and tmux.typed == []


def test_claude_state_with_no_break_space() -> None:
    assert claude_state('──────\n❯\u00a0Try "edit config.py to..."\n──────') == "ready"


# Screen regions --------------------------------------------------------------------------

WINDOWS = [
    Window(1, "Firefox", "Flay", (0, 0, 800, 600)),
    Window(2, "PHPStorm", "flay – assistant.py", (0, 0, 1200, 800)),
    Window(3, "Terminal", "zsh", (0, 0, 600, 400)),
]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (None, 2),
        ("active", 2),
        ("phpstorm", 2),
        ("Terminal", 3),
        ("Slack", None),
        # Named explicitly: Firefox showing the Flay tab is still Firefox.
        ("Firefox", 1),
    ],
)
def test_find_window_skips_flay_ui(query: str | None, expected: int | None) -> None:
    from flay.tools.screen import find_window

    window = find_window(query, WINDOWS)

    assert (window.id if window else None) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Was siehst du auf meinem Bildschirm?", {}),
        ("Was ist auf dem zweiten Bildschirm?", {"monitor": 2}),
        ("Look at my second screen", {"monitor": 2}),
        ("Zeig mir Monitor 3", {"monitor": 3}),
        ("Schau dir mein PhpStorm-Fenster an", {"window": "PHPStorm"}),
        ("Was steht im Terminal?", {"window": "Terminal"}),
        ("Schau dir dieses Fenster an", {"window": "active"}),
    ],
)
def test_screen_arguments(text: str, expected: dict[str, object]) -> None:
    from flay.tools.screen import screen_arguments

    assert screen_arguments(text, ["PHPStorm", "Firefox", "Terminal"]) == expected


async def test_window_capture_and_missing_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flay.tools.screen.list_windows", lambda: WINDOWS)
    monkeypatch.setattr("flay.tools.screen.capture_window_png", lambda window, size: b"png")

    found = await LookAtScreenTool().run({"window": "phpstorm"})
    missing = await LookAtScreenTool().run({"window": "Slack"})

    assert found.ok and found.images == (ImagePart(b"png", "image/png"),)
    assert "PHPStorm" in (found.action or "")
    assert not missing.ok and "Firefox" in missing.text


async def test_unknown_monitor_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flay.tools.screen.monitor_count", lambda: 1)

    result = await LookAtScreenTool().run({"monitor": 2})

    assert not result.ok and "1" in result.text


@pytest.fixture
def permission_granted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flay.tools.screen.ensure_permission", lambda: None)


class GradientMss:
    """A fake Retina monitor: 100x50 points, 200x100 pixels, left half dark, right half bright."""

    monitors = [{}, {"left": 0, "top": 0, "width": 100, "height": 50}]

    def __enter__(self) -> "GradientMss":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def grab(self, monitor: object) -> object:
        pixels = bytearray()
        for _y in range(100):
            for x in range(200):
                value = 30 if x < 100 else 220
                pixels += bytes([value, value, value, 255])

        class Shot:
            size = (200, 100)
            bgra = bytes(pixels)

        return Shot()


def test_window_is_cropped_from_monitor_with_retina_scale(
    monkeypatch: pytest.MonkeyPatch, permission_granted: None
) -> None:
    import io

    from PIL import Image

    from flay.tools.screen import capture_window_png

    monkeypatch.setattr("flay.tools.screen._mss", GradientMss)
    window = Window(9, "PHPStorm", "", (40, 10, 40, 20))  # straddles dark and bright half

    image = Image.open(io.BytesIO(capture_window_png(window, max_size=1000))).convert("L")

    assert image.size == (80, 40)  # 40x20 points at scale 2
    assert image.getpixel((0, 0)) == 30 and image.getpixel((79, 0)) == 220


def test_missing_screen_permission_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from flay.tools.screen import ScreenPermissionError, capture_png

    def deny() -> None:
        raise ScreenPermissionError("screen recording is not allowed")

    monkeypatch.setattr("flay.tools.screen.ensure_permission", deny)

    with pytest.raises(PermissionError, match="not allowed"):
        capture_png(1, 1280)


# Projects: opening behaviour, closing, intents --------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('<option name="confirmOpenNewProject2" value="-1" />', -1),
        ('<option name="confirmOpenNewProject2" value="0" />', 0),
        ("<application />", -1),
    ],
)
def test_open_project_setting(tmp_path: Path, content: str, expected: int) -> None:
    from flay.tools.phpstorm import open_project_setting

    (tmp_path / "options").mkdir()
    (tmp_path / "options" / "ide.general.xml").write_text(content)

    assert open_project_setting(tmp_path) == expected


class FakeIde:
    def __init__(self, projects: list[Path], closes: bool = True) -> None:
        self.projects = list(projects)
        self.closes = closes
        self.actions: list[tuple[str, Path]] = []

    async def open_projects(self) -> list[Path]:
        return list(self.projects)

    async def invoke_action(self, action: str, project: Path) -> str:
        self.actions.append((action, project))
        if self.closes:
            self.projects.remove(project)
        return "{}"


async def test_close_project_closes_matching_open_project() -> None:
    from flay.tools.phpstorm import CloseProjectTool

    ide = FakeIde([Path("/p/umx"), Path("/p/stadt-wolkenthal")])

    result = await CloseProjectTool(ide, poll_interval=0.01).run({"name": "Stadt Wolkenthal"})  # type: ignore[arg-type]

    assert result.ok and "geschlossen" in (result.action or "")
    assert ide.actions == [("CloseProject", Path("/p/stadt-wolkenthal"))]


async def test_close_project_reports_unknown_and_still_open() -> None:
    from flay.tools.phpstorm import CloseProjectTool

    unknown = await CloseProjectTool(FakeIde([Path("/p/umx")])).run({"name": "flay"})  # type: ignore[arg-type]
    stuck_ide = FakeIde([Path("/p/flay")], closes=False)
    stuck = await CloseProjectTool(stuck_ide, poll_interval=0.01).run({"name": "flay"})  # type: ignore[arg-type]

    assert not unknown.ok and "umx" in unknown.text
    assert not stuck.ok and "still open" in stuck.text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Öffne das Projekt Stadt Wolkenthal.", "open_project"),
        ("Open the project flay", "open_project"),
        ("Mach bitte das Projekt umx auf, ich meine öffnen", "open_project"),
        ("Schließe das Projekt Moewe.", "close_project"),
        ("Kannst du das Projekt flay schließen?", "close_project"),
        ("Close the project umx", "close_project"),
        ("Was ist ein Projekt?", None),
    ],
)
def test_project_intents(text: str, expected: str | None) -> None:
    from flay.assistant import tool_intent

    assert tool_intent(text) == expected


def test_match_projects_compares_garbled_names_with_the_distinctive_tail() -> None:
    projects = [
        Project("project-symfony-kx-ewo", Path("/p/ewo")),
        Project("component-symfony-kx-command", Path("/p/command")),
        Project("umx", Path("/p/umx")),
    ]

    # Parakeet heard "KEWO" for "kx-ewo".
    assert [p.name for p in match_projects("KEWO", projects)] == ["project-symfony-kx-ewo"]
    assert match_projects("Moewe", projects) == []


class ProjectRoutingLLM(RoutingLLM):
    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.prompts.append((list(messages), list(tools)))
        if len(messages) == 2 and [t.name for t in tools] == ["close_project"]:
            yield ToolCall("r1", "close_project", {"name": "KEWO"})
        else:
            yield TextDelta("Erledigt.")


class CloseTool(EchoTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec("close_project", "Close", {"type": "object", "properties": {}})


async def test_project_follow_up_without_name_is_routed_with_previous_request() -> None:
    tool, llm = CloseTool(), ProjectRoutingLLM()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )

    await assistant.handle_text("Schließe das Projekt KEWO wieder.")
    await assistant.handle_text("Das Projekt ist geöffnet.")
    await assistant.handle_text("Prüf nochmal, ob du es nicht wirklich schließen kannst.")

    # The statement in between calls nothing; the follow-up calls the tool again.
    assert len(tool.calls) == 2
    routed = [m for m, t in llm.prompts if [x.name for x in t] == ["close_project"]]
    assert "KEWO" in routed[-1][1].text and "nochmal" in routed[-1][1].text


class ShadowIde(FakeIde):
    """PhpStorm with ``umx`` listing the open subproject as one of its own content roots."""

    def __init__(self, projects: list[Path], shadowed: dict[Path, Path]) -> None:
        super().__init__(projects)
        self.shadowed = shadowed

    async def project_modules(self, project: Path) -> str:
        picked = self.shadowed.get(project, project)
        return f'{{"modules":[{{"name":"{picked.name}"}}]}}'


async def test_close_project_refuses_when_a_parent_project_would_be_closed_instead() -> None:
    from flay.tools.phpstorm import CloseProjectTool

    umx, ewo = Path("/p/umx"), Path("/p/umx/central/project-symfony-kx-ewo")
    ide = ShadowIde([umx, ewo], shadowed={ewo: umx})

    result = await CloseProjectTool(ide, poll_interval=0.01).run({"name": "KEWO"})  # type: ignore[arg-type]

    assert not result.ok and "umx" in result.text
    assert ide.actions == []


async def test_close_project_with_separate_parent_project_closes_the_subproject() -> None:
    from flay.tools.phpstorm import CloseProjectTool

    umx, ewo = Path("/p/umx"), Path("/p/umx/central/project-symfony-kx-ewo")
    ide = ShadowIde([umx, ewo], shadowed={})

    result = await CloseProjectTool(ide, poll_interval=0.01).run({"name": "KEWO"})  # type: ignore[arg-type]

    assert result.ok
    assert ide.actions == [("CloseProject", ewo)]


async def test_close_project_reports_when_phpstorm_closed_another_project() -> None:
    from flay.tools.phpstorm import CloseProjectTool

    class WrongIde(FakeIde):
        async def invoke_action(self, action: str, project: Path) -> str:
            self.actions.append((action, project))
            self.projects.remove(Path("/p/umx"))
            return "{}"

    ide = WrongIde([Path("/p/umx"), Path("/p/flay")])

    result = await CloseProjectTool(ide, poll_interval=0.01).run({"name": "flay"})  # type: ignore[arg-type]

    assert not result.ok and "instead" in result.text and "umx" in result.text


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("Das Projekt „KEWO“ ist jetzt geschlossen.", "close_project"),
        ("Ich schließe das Projekt „K Command“ für Sie.", "close_project"),
        ("I've closed the project flay.", "close_project"),
        ("Das Projekt „flay“ ist jetzt in PhpStorm geöffnet.", "open_project"),
        ("Der Auftrag wurde an Claude geschickt.", "send_to_claude"),
        ("Ich habe mir gemerkt, dass du Tee magst.", "remember"),
        ("Das Projekt ist nicht geöffnet, daher kann es nicht geschlossen werden.", None),
        ("Möchtest du das Projekt jetzt schließen?", None),
        ("Du schließt ein Projekt über File, Close Project.", None),
        ("Hallo, wie kann ich helfen?", None),
    ],
)
def test_claimed_tool(sentence: str, expected: str | None) -> None:
    from flay.assistant import claimed_tool

    assert claimed_tool(sentence) == expected


class ClaimingLLM(RoutingLLM):
    """Claims to close the project instead of calling the tool – until the tool ran."""

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.prompts.append((list(messages), list(tools)))
        if len(messages) == 2 and [t.name for t in tools] == ["close_project"]:
            yield ToolCall("r1", "close_project", {"name": "K Command"})
        elif any(m.role is Role.TOOL for m in messages):
            yield TextDelta("PhpStorm hat das Projekt geschlossen.")
        else:
            yield TextDelta("Ich schließe das Projekt „K Command“ für Sie. Erledigt.")


async def test_made_up_action_claim_runs_the_tool_after_all() -> None:
    tool, llm, tts = CloseTool(), ClaimingLLM(), FakeTTS()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        tts,
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )

    await assistant.handle_text("She K Command.")  # "Schließe" misheard

    assert tool.calls == [{"name": "K Command"}]
    assert assistant.history[-1].text == "PhpStorm hat das Projekt geschlossen."
    # The made-up claim is never spoken.
    assert [text for text, _ in tts.spoken] == ["PhpStorm hat das Projekt geschlossen."]


# Fixed replies and the action log -------------------------------------------------------


class SayingCloseTool(CloseTool):
    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(
            "Closed project 'flay'.",
            action="PhpStorm: Projekt „flay“ geschlossen",
            say=speech("flay ist geschlossen.", "flay is closed."),
        )


async def test_action_with_fixed_reply_needs_no_second_model_round() -> None:
    tool, llm, tts = SayingCloseTool(), ProjectRoutingLLM(), FakeTTS()
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, tts, FakeOutput(), tools=ToolRegistry([tool])
    )

    await assistant.handle_text("Schließe das Projekt KEWO.")

    # Only the routing call: the result is spoken word for word, not paraphrased.
    assert len(llm.prompts) == 1
    assert [text for text, _ in tts.spoken] == ["flay ist geschlossen."]
    reply = assistant.history[-1]
    assert reply.text == "flay ist geschlossen."
    assert [(a.call.name, a.result) for a in reply.actions] == [
        ("close_project", "Closed project 'flay'.")
    ]


async def test_past_actions_reach_the_model_as_tool_calls() -> None:
    tool, llm = SayingCloseTool(), ProjectRoutingLLM()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )

    await assistant.handle_text("Schließe das Projekt KEWO.")
    await assistant.handle_text("Hat das geklappt?")

    sent = llm.prompts[-1][0]
    call_message = next(m for m in sent if m.tool_calls)
    assert call_message.tool_calls[0].name == "close_project"
    result = sent[sent.index(call_message) + 1]
    assert result.role is Role.TOOL and result.text == "Closed project 'flay'."
    assert sent[sent.index(call_message) + 2].text == "flay ist geschlossen."


async def test_model_called_tool_with_fixed_reply_ends_the_turn() -> None:
    class CallingLLM(RoutingLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            self.prompts.append((list(messages), list(tools)))
            yield ToolCall("c1", "close_project", {"name": "flay"})

    tool, llm = SayingCloseTool(), CallingLLM()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )

    await assistant.handle_text("Mach flay zu.")  # no routing rule matches

    assert len(llm.prompts) == 1
    assert assistant.history[-1].text == "flay ist geschlossen."


async def test_claude_asks_instead_of_using_the_clipboard_when_project_is_ambiguous(
    tmp_path: Path,
) -> None:
    projects = [
        Project("lib-react-kx-forms-app", tmp_path / "a"),
        Project("project-symfony-kx-forms", tmp_path / "b"),
    ]
    tool = SendToClaudeTool(
        tmux=FakeTmux([]),  # type: ignore[arg-type]
        find=lambda name: projects,
        claude_command=["claude"],
    )

    result = await tool.run({"instruction": "Fasse die README zusammen.", "project": "KX Forms"})

    assert not result.ok and result.say is not None
    assert result.say[Language.DE] == (
        "Meinst du lib react kx forms app oder project symfony kx forms?"
    )


async def test_computer_state_lists_projects_and_claude_sessions() -> None:
    from flay.tools.claude_code import Session
    from flay.tools.state import ComputerState

    class Ide:
        async def open_projects(self) -> list[Path]:
            return [Path("/p/flay"), Path("/p/umx")]

    sessions = [Session("flay-claude-flay", 1, attached=True), Session("flay-claude-umx", 2)]
    state = ComputerState(Ide(), FakeTmux(sessions))  # type: ignore[arg-type]

    assert await state.describe() == (
        "Open in PhpStorm: flay, umx. Claude Code running for: flay (shown in PhpStorm), "
        "umx (in the background)."
    )


async def test_single_sentence_claim_is_caught_too() -> None:
    class OneSentenceClaim(ClaimingLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            self.prompts.append((list(messages), list(tools)))
            if len(messages) == 2 and [t.name for t in tools] == ["close_project"]:
                yield ToolCall("r1", "close_project", {"name": "K Command"})
            elif any(m.role is Role.TOOL for m in messages):
                yield TextDelta("PhpStorm hat das Projekt geschlossen.")
            else:
                # Only the splitter's final flush delivers this sentence.
                yield TextDelta("Ich schließe das Projekt „K Command“ in PhpStorm")

    tool, tts = CloseTool(), FakeTTS()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        OneSentenceClaim(),
        tts,
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )

    await assistant.handle_text("She K Command.")

    assert tool.calls == [{"name": "K Command"}]
    assert [text for text, _ in tts.spoken] == ["PhpStorm hat das Projekt geschlossen."]


@pytest.mark.parametrize(
    ("name", "said", "expected"),
    [
        ("moewe", "Mach mal Moewe auf.", True),
        ("project-symfony-kx-ewo", "Mach mal Moewe auf.", False),
        ("project-symfony-kx-ewo", "Slie K X E V O", True),
        ("component-symfony-kx-command", "She K Command.", True),
        ("umx", "Close the umx project.", True),
        # A shared ending like "cms" does not identify a project.
        ("bvx-cms", "Öffne tlr cms", False),
    ],
)
def test_mentioned(name: str, said: str, expected: bool) -> None:
    from flay.tools.phpstorm import mentioned

    assert mentioned(name, said) is expected


class PickingLLM(RoutingLLM):
    """Picks a project from the ones in view instead of the one the user named."""

    def __init__(self, tool: str, arguments: dict[str, Any]) -> None:
        super().__init__()
        self.call = ToolCall("p1", tool, arguments)

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.prompts.append((list(messages), list(tools)))
        if tools and not any(m.role is Role.TOOL for m in messages):
            yield self.call
        else:
            yield TextDelta("Okay.")


async def test_project_the_user_never_named_is_not_acted_on() -> None:
    tool = SayingCloseTool()
    llm = PickingLLM("close_project", {"name": "project-symfony-kx-ewo"})
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )

    await assistant.handle_text("Mach Moewe zu.")

    assert tool.calls == []
    assert assistant.history[-1].text == "Welches Projekt meinst du?"


async def test_unnamed_optional_project_is_dropped() -> None:
    tool = SendTool()
    llm = PickingLLM("send_to_claude", {"instruction": "Schreib Tests.", "project": "umx"})
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )

    await assistant.handle_text("Sag Claude, er soll Tests schreiben.")

    assert tool.calls == [{"instruction": "Schreib Tests."}]


async def test_request_taken_back_does_nothing() -> None:
    tool, llm = SendTool(), RoutingLLM()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
    )

    await assistant.handle_text("Schreibe Claude, er soll die Readme, ach never mind.")

    assert tool.calls == [] and llm.prompts == []
    assert assistant.history[-1].text == "Okay, dann nicht."


def test_how_to_question_is_no_project_action() -> None:
    from flay.assistant import tool_intent

    assert tool_intent("Wie schließe ich in PhpStorm ein Projekt?") is None
    assert tool_intent("Kannst du das Projekt flay schließen?") == "close_project"


def confirming_assistant(tool: SayingCloseTool, llm: Any) -> Assistant:
    return Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
        projects=lambda: ["flay", "project-symfony-kx-ewo"],
    )


async def test_roughly_named_project_is_closed_only_after_a_yes() -> None:
    tool = SayingCloseTool()
    assistant = confirming_assistant(tool, PickingLLM("close_project", {"name": "KEWO"}))

    await assistant.handle_text("Schließe KEWO.")

    assert tool.calls == []
    assert assistant.history[-1].text == "Meinst du project symfony kx ewo?"

    await assistant.handle_text("Ja, genau.")

    assert tool.calls == [{"name": "project-symfony-kx-ewo"}]
    assert assistant.history[-1].text == "flay ist geschlossen."


async def test_no_or_another_request_drops_the_question() -> None:
    tool = SayingCloseTool()
    assistant = confirming_assistant(tool, PickingLLM("close_project", {"name": "KEWO"}))

    await assistant.handle_text("Schließe KEWO.")
    await assistant.handle_text("Nein.")

    assert tool.calls == [] and assistant.history[-1].text == "Okay, dann nicht."

    await assistant.handle_text("Schließe KEWO.")
    await assistant.handle_text("Wie spät ist es?")

    assert tool.calls == [] and assistant.pending_confirmation is None


async def test_clearly_named_project_is_closed_at_once() -> None:
    tool = SayingCloseTool()
    llm = PickingLLM("close_project", {"name": "project-symfony-kx-ewo"})
    assistant = confirming_assistant(tool, llm)

    await assistant.handle_text("Schließe das Projekt K X E W O.")

    assert tool.calls == [{"name": "project-symfony-kx-ewo"}]


async def test_rules_reach_the_instruction_for_claude() -> None:
    from flay.memory.base import IMPORTANT, Fact, MemoryContext, ShortTermMemory

    class RuledMemory(ShortTermMemory):
        def context(self) -> MemoryContext:
            rule = Fact(1, "Gib Claude nie den Befehl, etwas zu committen.", importance=IMPORTANT)
            return MemoryContext(important=(rule,))

    tool, llm = SendTool(), RoutingLLM()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([tool]),
        memory=RuledMemory(),
    )

    await assistant.handle_text("Sag Claude, er soll die README zusammenfassen und committen.")

    router_prompt = llm.prompts[0][0][0].text
    assert "nie den Befehl, etwas zu committen" in router_prompt
