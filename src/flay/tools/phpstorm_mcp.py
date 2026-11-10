"""Client for PhpStorm's built-in MCP server (Settings → Tools → MCP Server).

Flay uses it to learn which projects are open and to run commands in a project's
integrated terminal. PhpStorm asks the user to confirm each terminal command unless
"Brave Mode" is enabled.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp
import httpx2
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import MCP_DEFAULT_SSE_READ_TIMEOUT, MCP_DEFAULT_TIMEOUT

log = logging.getLogger(__name__)


def _direct_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx2.Timeout | None = None,
    auth: httpx2.Auth | None = None,
) -> httpx2.AsyncClient:
    """MCP's default client, but never through a proxy: PhpStorm runs on 127.0.0.1."""
    return httpx2.AsyncClient(
        headers=headers,
        timeout=timeout or httpx2.Timeout(MCP_DEFAULT_TIMEOUT, read=MCP_DEFAULT_SSE_READ_TIMEOUT),
        auth=auth,
        trust_env=False,  # see flay.net
    )


# PhpStorm picks a free port in this range when the MCP server is enabled.
DISCOVERY_PORTS = range(64342, 64462)
_OPEN_PROJECTS = re.compile(r"Currently open projects:\s*(\{.*\})", re.DOTALL)


def parse_open_projects(error_text: str) -> list[Path]:
    """PhpStorm lists the open projects when a call does not say which one is meant."""
    match = _OPEN_PROJECTS.search(error_text)
    if not match:
        return []
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return [
        Path(p["path"]) for p in data.get("projects", []) if isinstance(p, dict) and "path" in p
    ]


def projects_from_claude_locks(directory: Path | None = None) -> list[Path]:
    """Workspace folders of open IDE windows, as announced by the Claude Code plugin."""
    directory = directory or Path.home() / ".claude" / "ide"
    projects: list[Path] = []
    for lock in directory.glob("*.lock"):
        try:
            data = json.loads(lock.read_text(encoding="utf-8"))
            os.kill(int(data["pid"]), 0)  # stale lock files belong to closed IDEs
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if "phpstorm" not in str(data.get("ideName", "")).lower():
            continue
        projects.extend(Path(folder) for folder in data.get("workspaceFolders", []))
    return projects


# Launcher script for `flay claude`. PhpStorm runs shell scripts as run configurations in
# a real terminal tab – the only way found to get an interactive terminal from outside:
# `execute_terminal_command` has no TTY, terminal/Claude plugin actions are unavailable or
# do nothing when invoked through MCP.
LAUNCHER = ".flay-claude.sh"


def launcher_script() -> str:
    flay = shutil.which("flay")
    command = (
        f"exec {shlex.quote(flay)} claude"
        if flay
        else f"exec {shlex.quote(sys.executable)} -m flay claude"
    )
    return (
        "#!/bin/sh\n"
        "# Temporary file created by Flay; it opens Claude Code in a tmux session that Flay\n"
        "# can write into. Flay deletes it again after the start.\n"
        f"{command}\n"
    )


class PhpStormMcp:
    def __init__(self, url: str | None = None, token: str | None = None) -> None:
        self._url = url
        self._token = token

    async def url(self) -> str | None:
        if self._url is None:
            self._url = await _discover()
        return self._url

    def use_sse(self) -> None:
        """Fall back to the older SSE transport (e.g. a server without /stream)."""
        if self._url is not None and not self._url.endswith(("/sse", "/stream")):
            self._url = self._url.rstrip("/") + "/sse"

    @contextlib.asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        base = await self.url()
        if base is None:
            raise ConnectionError("PhpStorm MCP server not found (Settings → Tools → MCP Server)")
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
        if base.endswith("/sse"):
            transport = sse_client(base, headers=headers, httpx_client_factory=_direct_http_client)
        else:
            # Streamable HTTP is the current MCP transport; PhpStorm serves it on /stream.
            url = base if base.endswith("/stream") else base.rstrip("/") + "/stream"
            transport = streamable_http_client(
                url, http_client=_direct_http_client(headers=headers)
            )
        async with transport as streams, ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            yield session

    async def call(self, tool: str, arguments: dict[str, Any], attempts: int = 3) -> str:
        """Call a tool of PhpStorm's MCP server; retries when connecting fails."""
        for attempt in range(1, attempts + 1):
            try:
                async with self._session() as session:
                    result = await session.call_tool(tool, arguments)
                text = _text(result)
                if getattr(result, "isError", False) or getattr(result, "is_error", False):
                    raise RuntimeError(text)
                return text
            except RuntimeError:
                raise
            except Exception as exc:
                if attempt == attempts:
                    raise ConnectionError(describe(exc)) from exc
                log.debug("PhpStorm MCP nicht erreichbar (Versuch %d): %s", attempt, describe(exc))
                if attempt == attempts - 1:
                    self.use_sse()
                await asyncio.sleep(0.5 * attempt)
        raise AssertionError("unreachable")

    async def open_projects(self) -> list[Path]:
        """Projects open in PhpStorm right now."""
        projects: list[Path] = []
        try:
            # A read-only call without project: PhpStorm answers with the open projects
            # (as an error) if more than one is open.
            await self.call("get_project_modules", {})
        except RuntimeError as answer:
            projects = parse_open_projects(str(answer))
        except ConnectionError as exc:
            log.debug("PhpStorm MCP: offene Projekte nicht abfragbar: %s", exc)
        if not projects:
            projects = projects_from_claude_locks()
        return list(dict.fromkeys(projects))

    async def project_modules(self, project: Path) -> str:
        """Modules of the project PhpStorm picks for ``project`` (read-only)."""
        return await self.call("get_project_modules", {"projectPath": str(project)})

    async def invoke_action(self, action_id: str, project: Path) -> str:
        return await self.call(
            "invoke_ide_action", {"actionId": action_id, "projectPath": str(project)}
        )

    async def open_claude(self, project: Path) -> Path:
        """Start ``flay claude`` in a new terminal tab of ``project``.

        Returns the temporary launcher script; the caller deletes it once Claude is up.
        """
        await self.call(
            "create_new_file",
            {
                "pathInProject": LAUNCHER,
                "text": launcher_script(),
                "overwrite": True,
                "projectPath": str(project),
            },
        )
        try:
            await self.call(
                "execute_run_configuration",
                {
                    "filePath": LAUNCHER,
                    "line": 1,
                    "waitForExit": False,
                    "timeout": 5000,
                    "projectPath": str(project),
                },
            )
        except RuntimeError as answer:
            # Scripts run in a terminal tab report "failed to start" although they run;
            # the caller checks the tmux session instead.
            log.debug("PhpStorm meldet beim Start: %s", answer)
        return project / LAUNCHER


def describe(exc: BaseException) -> str:
    """Flatten exception groups (anyio/MCP wrap the real cause) into one readable line."""
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(describe(inner) for inner in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


def _text(result: Any) -> str:
    return " ".join(getattr(part, "text", "") for part in getattr(result, "content", []))


async def _discover() -> str | None:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as http:

        async def probe(port: int) -> tuple[int, str] | None:
            base = f"http://127.0.0.1:{port}"
            try:
                async with http.get(base + "/sse") as response:
                    if response.status == 200 and "event-stream" in response.content_type:
                        return 0, base
                    if response.status == 401:  # restricted mode, needs mcp_token
                        return 1, base
            except (aiohttp.ClientError, TimeoutError):
                return None
            return None

        found = [hit for hit in await asyncio.gather(*map(probe, DISCOVERY_PORTS)) if hit]
    # An open server first; a restricted one only works with a configured token.
    return min(found)[1] if found else None
