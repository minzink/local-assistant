from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cassi.config import Settings
from cassi.core.language import Language
from cassi.core.messages import Modality
from cassi.memory.base import FactMemory

if TYPE_CHECKING:
    from cassi.notifications import Announcements, Timers
    from cassi.tools.phpstorm_mcp import PhpStormMcp
    from cassi.tools.preferences import Preferences
    from cassi.tools.state import ComputerState
from cassi.tools.base import Tool, ToolRegistry

log = logging.getLogger(__name__)


def create_state(settings: Settings) -> ComputerState | None:
    """Reads what is open (PhpStorm projects, Claude sessions) for the enabled tools."""
    from cassi.tools.claude_code import Tmux
    from cassi.tools.phpstorm_mcp import PhpStormMcp
    from cassi.tools.state import ComputerState

    config = settings.tools
    phpstorm = None
    if config.phpstorm.use_mcp and {"open_project", "close_project", "send_to_claude"} & set(
        config.enabled
    ):
        token = config.phpstorm.mcp_token
        phpstorm = PhpStormMcp(config.phpstorm.mcp_url, token.get_secret_value() if token else None)
    tmux = Tmux() if "send_to_claude" in config.enabled else None
    if phpstorm is None and tmux is None:
        return None
    return ComputerState(phpstorm, tmux)


def create_tools(
    settings: Settings,
    facts: FactMemory | None = None,
    announcements: Announcements | None = None,
    timers: Timers | None = None,
    preferences: Preferences | None = None,
) -> ToolRegistry:
    """Registry of the tools enabled in the config; nothing else is callable.

    ``facts`` is the persistent memory; without it remember/forget are left out.
    ``announcements`` lets tools speak later (a command finished); ``timers`` keeps the
    timers (without it there is no timer tool).
    """
    config = settings.tools
    tools: list[Tool] = []
    shared: list[PhpStormMcp] = []

    def phpstorm() -> PhpStormMcp:
        """One client for all PhpStorm tools (it remembers the discovered server)."""
        if not shared:
            from cassi.tools.phpstorm_mcp import PhpStormMcp

            token = config.phpstorm.mcp_token
            shared.append(
                PhpStormMcp(config.phpstorm.mcp_url, token.get_secret_value() if token else None)
            )
        return shared[0]

    for name in dict.fromkeys(config.enabled):
        if name == "open_project":
            from cassi.tools.phpstorm import OpenProjectTool

            tools.append(OpenProjectTool(config.phpstorm.launcher, config.phpstorm.config_dir))
        elif name == "send_to_claude":
            from cassi.tools.claude_code import SendToClaudeTool
            from cassi.tools.phpstorm import match_projects, recent_projects

            tools.append(
                SendToClaudeTool(
                    config.claude.transport,
                    autostart=config.claude.autostart,
                    find=lambda name: match_projects(
                        name, recent_projects(config.phpstorm.config_dir)
                    ),
                    phpstorm=phpstorm() if config.phpstorm.use_mcp else None,
                )
            )
        elif name == "close_project":
            if not config.phpstorm.use_mcp:
                log.warning("close_project braucht PhpStorms MCP-Server (use_mcp) – deaktiviert.")
                continue
            from cassi.tools.phpstorm import CloseProjectTool

            tools.append(CloseProjectTool(phpstorm()))
        elif name == "answer_claude":
            if "send_to_claude" not in config.enabled:
                continue
            from cassi.tools.claude_code import AnswerClaudeTool

            tools.append(AnswerClaudeTool())
        elif name == "claude_status":
            from cassi.tools.claude_code import ClaudeStatusTool

            tools.append(ClaudeStatusTool())
        elif name == "look_at_screen":
            if Modality.IMAGE not in settings.llm.input_modalities:
                log.warning("look_at_screen braucht ein Modell mit Bildeingabe – deaktiviert.")
                continue
            from cassi.tools.screen import LookAtScreenTool

            tools.append(LookAtScreenTool(config.screen.monitor, config.screen.max_size))
        elif name in ("click_on_screen", "scroll_screen", "read_webpage"):
            if Modality.IMAGE not in settings.llm.input_modalities:
                log.warning("%s braucht ein Modell mit Bildeingabe – deaktiviert.", name)
                continue
            tools.append(_screen_tool(name, settings))
        elif name == "open_url":
            from cassi.tools.web import OpenUrlTool

            tools.append(OpenUrlTool())
        elif name == "fill_web_form":
            from cassi.tools.forms import FillWebFormTool

            tools.append(FillWebFormTool())
        elif name == "read_file":
            from cassi.tools.files import ReadFileTool

            tools.append(ReadFileTool(max_chars=config.web.max_chars))
        elif name == "workflow":
            from cassi.tools.workflows import WorkflowTool

            tools.append(
                WorkflowTool(settings.data_dir / "workflows.json", settings.input.push_to_talk_key)
            )
        elif name == "ssh_connect":
            from functools import partial

            from cassi.tools.ssh import SshConnectTool, read_ssh_hosts

            hosts = partial(read_ssh_hosts, config.ssh.config_file)
            tools.append(SshConnectTool(hosts, config.ssh.terminal))
        elif name == "control_window":
            from cassi.tools.apps import ControlWindowTool

            tools.append(ControlWindowTool())
        elif name == "use_keyboard":
            from cassi.tools.keyboard import KeyboardTool

            tools.append(KeyboardTool())
        elif name == "phpstorm":
            if not config.phpstorm.use_mcp:
                log.debug("phpstorm braucht PhpStorms MCP-Server (use_mcp) – deaktiviert.")
                continue
            from cassi.tools.ide import IdeTool

            tools.append(IdeTool(phpstorm()))
        elif name == "run_command":
            if not config.commands:
                log.debug("run_command: keine Befehle in tools.commands – deaktiviert.")
                continue
            from cassi.tools.commands import Command, RunCommandTool

            commands = {
                key: Command(
                    key,
                    c.command,
                    c.cwd.expanduser() if c.cwd else None,
                    c.timeout_seconds,
                )
                for key, c in config.commands.items()
            }
            tools.append(RunCommandTool(commands, announcements))
        elif name == "find_file":
            from cassi.tools.finder import FindFileTool

            roots = [r.expanduser() for r in config.files.roots] or None
            tools.append(FindFileTool(roots))
        elif name == "timer":
            if timers is None:
                continue
            from cassi.tools.timers import TimerTool

            tools.append(TimerTool(timers))
        elif name == "system_control":
            from cassi.tools.system import SystemTool

            tools.append(SystemTool())
        elif name == "operate_computer":
            if Modality.IMAGE not in settings.llm.input_modalities:
                log.warning("operate_computer braucht ein Modell mit Bildeingabe – deaktiviert.")
                continue
            from cassi.llm.factory import create_chat_model
            from cassi.tools.computer import OperateComputerTool, Solutions, screen_snapshot

            say = (
                (lambda words: announcements.say(words[Language.DE], words[Language.EN]))
                if announcements is not None
                else None
            )
            mover = settings.llm
            if config.computer.model:
                from cassi.llm.ollama import POINTER, resolve

                model = resolve(
                    config.computer.model, POINTER, settings.llm.model, settings.llm.base_url
                )
                mover = mover.model_copy(update={"model": model})
            computer = OperateComputerTool(
                create_chat_model(mover),
                lambda: registry,
                read_text=_ocr(),
                max_steps=config.computer.max_steps,
                solutions=Solutions(
                    settings.data_dir / "solutions.json" if config.computer.learn else None,
                    _embedder(settings),
                ),
                announce=say,
                snapshot=screen_snapshot,
            )
            tools.append(computer)
        elif name == "calendar":
            from cassi.tools.appointments import CalendarTool

            tools.append(CalendarTool(hidden=config.calendar.hidden))
        elif name == "adjust_settings":
            from cassi.tools.preferences import AdjustSettingsTool, Preferences

            tools.append(AdjustSettingsTool(preferences or Preferences.load(settings.data_dir)))
        elif name in ("remember", "forget"):
            if facts is None:
                log.debug(
                    "%s braucht das dauerhafte Gedächtnis (memory.enabled) – deaktiviert.", name
                )
                continue
            from cassi.tools.memory import ForgetTool, RememberTool

            tools.append(RememberTool(facts) if name == "remember" else ForgetTool(facts))
    registry = ToolRegistry(tools)
    return registry


def _screen_tool(name: str, settings: Settings) -> Tool:
    config = settings.tools
    if name == "click_on_screen":
        from cassi.llm.factory import create_chat_model
        from cassi.llm.ollama import POINTER, resolve
        from cassi.tools.pointer import ClickTool, VisionLocator

        # Deterministic and short: the answer is one point.
        model = resolve(config.pointer.model, POINTER, settings.llm.model, settings.llm.base_url)
        llm = settings.llm.model_copy(
            update={
                "model": model,
                # The 4B variant has a context of its own.
                "context_tokens": POINTER.context
                if model == POINTER.name
                else settings.llm.context_tokens,
                "temperature": 0.0,
                "max_tokens": 60,
                "thinking": settings.llm.thinking.model_copy(update={"enabled": False}),
            }
        )
        locator = VisionLocator(create_chat_model(llm), config.screen.max_size)
        from cassi.tools.pointer import icon_candidates, watch_click

        return ClickTool(
            locator,
            config.screen.monitor,
            config.screen.max_size,
            config.pointer.refine,
            verify=watch_click if config.pointer.verify else None,
            candidates=icon_candidates,
            read_text=_ocr() if config.pointer.ocr else None,
        )
    if name == "scroll_screen":
        from cassi.tools.pointer import ScrollTool

        return ScrollTool(config.screen.monitor, config.screen.max_size)
    from cassi.tools.web import ReadWebpageTool

    return ReadWebpageTool(
        config.web.max_pages,
        config.screen.max_size,
        direct=config.web.direct,
        browser=config.web.browser_text,
        max_chars=config.web.max_chars,
        close_tabs=config.web.close_tabs,
    )


def _ocr() -> Callable[[tuple[float, float, float, float]], list[Any]] | None:
    """The system's text recognition, if this computer has it."""
    from cassi.tools import ocr

    return ocr.read_screen if ocr.available() else None


def _embedder(settings: Settings) -> Any:
    from cassi.memory.embeddings import create_embedder

    try:
        return create_embedder(settings)
    except Exception as exc:
        log.debug("Kein Embedding-Modell für Lösungswege: %s", exc)
        return None
