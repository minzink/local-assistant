"""Local web UI: chat history, text input and mute switches over a WebSocket.

The server only listens on 127.0.0.1. WebSocket connections must come from the UI's own
origin, so other websites open in the browser cannot talk to the assistant (relevant once
it can run actions on the computer).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
from importlib import resources
from typing import Any

from aiohttp import WSMsgType, web

from flay.assistant import Assistant
from flay.core.events import Event
from flay.core.messages import Role
from flay.ui.logs import UILogHandler

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
# flay itself, plus libraries whose warnings matter for debugging (e.g. PhpStorm's MCP link).
LOGGERS = ("flay", "mcp")


def event_to_json(event: Event) -> dict[str, Any]:
    return {"type": type(event).__name__, **dataclasses.asdict(event)}


class WebUI:
    def __init__(self, assistant: Assistant, port: int) -> None:
        self._assistant = assistant
        self._port = port
        self._runner: web.AppRunner | None = None
        self._sockets: set[web.WebSocketResponse] = set()
        self._log_handler: UILogHandler | None = None
        self.app = web.Application()
        self.app.router.add_get("/", self._index)
        self.app.router.add_get("/ws", self._websocket)
        self.app.on_startup.append(self._install_log_handler)
        self.app.on_cleanup.append(self._remove_log_handler)

    async def _install_log_handler(self, app: web.Application) -> None:
        self._log_handler = UILogHandler(self._assistant.events, asyncio.get_running_loop())
        for name in LOGGERS:
            logging.getLogger(name).addHandler(self._log_handler)

    async def _remove_log_handler(self, app: web.Application) -> None:
        if self._log_handler is not None:
            for name in LOGGERS:
                logging.getLogger(name).removeHandler(self._log_handler)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self._port}/"

    @property
    def allowed_origins(self) -> set[str]:
        return {f"http://{HOST}:{self._port}", f"http://localhost:{self._port}"}

    async def start(self) -> None:
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        await web.TCPSite(self._runner, HOST, self._port).start()

    async def stop(self) -> None:
        for socket in list(self._sockets):
            await socket.close()
        if self._runner is not None:
            await self._runner.cleanup()

    async def _index(self, request: web.Request) -> web.Response:
        html = resources.files("flay.ui").joinpath("static/index.html").read_text("utf-8")
        return web.Response(text=html, content_type="text/html")

    def initial_state(self) -> dict[str, Any]:
        assistant = self._assistant
        history = [
            {"role": m.role.value, "text": m.text, "interrupted": m.interrupted}
            for m in assistant.history
            # A reply cut off before its first word has nothing to show.
            if m.role in (Role.USER, Role.ASSISTANT) and (m.text or not m.interrupted)
        ]
        return {
            "type": "Init",
            "history": history,
            "microphone": assistant.microphone_muted,
            "speaker": assistant.speaker_muted,
            "state": assistant.state.value,
            "continuous": assistant.continuous,
            "wake_word": assistant.wake_word,
            "recording": assistant.workflow_recording,
            "logs": [
                dataclasses.asdict(entry)
                for entry in (self._log_handler.entries if self._log_handler else ())
            ],
        }

    async def _websocket(self, request: web.Request) -> web.StreamResponse:
        origin = request.headers.get("Origin")
        if origin not in self.allowed_origins:
            log.warning("UI-Verbindung von fremder Herkunft abgelehnt: %s", origin)
            raise web.HTTPForbidden(text="origin not allowed")

        socket = web.WebSocketResponse(heartbeat=30)
        await socket.prepare(request)
        self._sockets.add(socket)
        async with self._assistant.events.subscribe() as events:
            forward = asyncio.create_task(self._forward(socket, events))
            try:
                await socket.send_json(self.initial_state())
                async for message in socket:
                    if message.type is WSMsgType.TEXT:
                        await self._handle_command(message.data)
            finally:
                forward.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await forward
                self._sockets.discard(socket)
        return socket

    async def _forward(self, socket: web.WebSocketResponse, events: asyncio.Queue[Event]) -> None:
        while True:
            event = await events.get()
            await socket.send_json(event_to_json(event))

    async def _handle_command(self, raw: str) -> None:
        try:
            command = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(command, dict):
            return
        if command.get("type") == "send" and isinstance(command.get("text"), str):
            await self._assistant.submit_text(command["text"][:2000])
        elif command.get("type") == "listen" and isinstance(command.get("continuous"), bool):
            self._assistant.set_continuous(command["continuous"])
        elif command.get("type") == "wake" and isinstance(command.get("enabled"), bool):
            self._assistant.set_wake_word(command["enabled"])
        elif command.get("type") == "mute":
            microphone = command.get("microphone")
            speaker = command.get("speaker")
            microphone = microphone if isinstance(microphone, bool) else None
            speaker = speaker if isinstance(speaker, bool) else None
            if microphone is not None or speaker is not None:
                self._assistant.set_muted(microphone=microphone, speaker=speaker)
