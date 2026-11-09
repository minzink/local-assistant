from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from flay.config import Settings
from flay.core.messages import Modality
from flay.memory.base import FactMemory

if TYPE_CHECKING:
    from flay.tools.phpstorm_mcp import PhpStormMcp
    from flay.tools.state import ComputerState
from flay.tools.base import Tool, ToolRegistry

log = logging.getLogger(__name__)


def create_state(settings: Settings) -> ComputerState | None:
    """Reads what is open (PhpStorm projects, Claude sessions) for the enabled tools."""
    from flay.tools.claude_code import Tmux
    from flay.tools.phpstorm_mcp import PhpStormMcp
    from flay.tools.state import ComputerState

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


def create_tools(settings: Settings, facts: FactMemory | None = None) -> ToolRegistry:
    """Registry of the tools enabled in the config; nothing else is callable.

    ``facts`` is the persistent memory; without it remember/forget are left out.
    """
    config = settings.tools
    tools: list[Tool] = []
    shared: list[PhpStormMcp] = []

    def phpstorm() -> PhpStormMcp:
        """One client for all PhpStorm tools (it remembers the discovered server)."""
        if not shared:
            from flay.tools.phpstorm_mcp import PhpStormMcp

            token = config.phpstorm.mcp_token
            shared.append(
                PhpStormMcp(config.phpstorm.mcp_url, token.get_secret_value() if token else None)
            )
        return shared[0]

    for name in dict.fromkeys(config.enabled):
        if name == "open_project":
            from flay.tools.phpstorm import OpenProjectTool

            tools.append(OpenProjectTool(config.phpstorm.launcher, config.phpstorm.config_dir))
        elif name == "send_to_claude":
            from flay.tools.claude_code import SendToClaudeTool
            from flay.tools.phpstorm import match_projects, recent_projects

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
            from flay.tools.phpstorm import CloseProjectTool

            tools.append(CloseProjectTool(phpstorm()))
        elif name == "claude_status":
            from flay.tools.claude_code import ClaudeStatusTool

            tools.append(ClaudeStatusTool())
        elif name == "look_at_screen":
            if Modality.IMAGE not in settings.llm.input_modalities:
                log.warning("look_at_screen braucht ein Modell mit Bildeingabe – deaktiviert.")
                continue
            from flay.tools.screen import LookAtScreenTool

            tools.append(LookAtScreenTool(config.screen.monitor, config.screen.max_size))
        elif name in ("click_on_screen", "scroll_screen", "read_webpage"):
            if Modality.IMAGE not in settings.llm.input_modalities:
                log.warning("%s braucht ein Modell mit Bildeingabe – deaktiviert.", name)
                continue
            tools.append(_screen_tool(name, settings))
        elif name == "open_url":
            from flay.tools.web import OpenUrlTool

            tools.append(OpenUrlTool())
        elif name == "fill_web_form":
            from flay.tools.forms import FillWebFormTool

            tools.append(FillWebFormTool())
        elif name == "read_file":
            from flay.tools.files import ReadFileTool

            tools.append(ReadFileTool(max_chars=config.web.max_chars))
        elif name == "workflow":
            from flay.tools.workflows import WorkflowTool

            tools.append(
                WorkflowTool(settings.data_dir / "workflows.json", settings.input.push_to_talk_key)
            )
        elif name == "ssh_connect":
            from functools import partial

            from flay.tools.ssh import SshConnectTool, read_ssh_hosts

            hosts = partial(read_ssh_hosts, config.ssh.config_file)
            tools.append(SshConnectTool(hosts, config.ssh.terminal))
        elif name in ("remember", "forget"):
            if facts is None:
                log.debug(
                    "%s braucht das dauerhafte Gedächtnis (memory.enabled) – deaktiviert.", name
                )
                continue
            from flay.tools.memory import ForgetTool, RememberTool

            tools.append(RememberTool(facts) if name == "remember" else ForgetTool(facts))
    return ToolRegistry(tools)


def _screen_tool(name: str, settings: Settings) -> Tool:
    config = settings.tools
    if name == "click_on_screen":
        from flay.llm.factory import create_chat_model
        from flay.llm.ollama import POINTER, resolve
        from flay.tools.pointer import ClickTool, VisionLocator

        # Deterministic and short: the answer is one point.
        llm = settings.llm.model_copy(
            update={
                "model": resolve(
                    config.pointer.model, POINTER, settings.llm.model, settings.llm.base_url
                ),
                "temperature": 0.0,
                "max_tokens": 60,
                "thinking": settings.llm.thinking.model_copy(update={"enabled": False}),
            }
        )
        locator = VisionLocator(create_chat_model(llm), config.screen.max_size)
        return ClickTool(
            locator, config.screen.monitor, config.screen.max_size, config.pointer.refine
        )
    if name == "scroll_screen":
        from flay.tools.pointer import ScrollTool

        return ScrollTool(config.screen.monitor, config.screen.max_size)
    from flay.tools.web import ReadWebpageTool

    return ReadWebpageTool(
        config.web.max_pages,
        config.screen.max_size,
        direct=config.web.direct,
        browser=config.web.browser_text,
        max_chars=config.web.max_chars,
        close_tabs=config.web.close_tabs,
    )
