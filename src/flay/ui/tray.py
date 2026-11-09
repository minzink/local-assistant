"""Menu bar (macOS) / tray (Windows) icon: state at a glance, mute switches, quit.

On macOS the status item must be driven from the main thread, so Flay's asyncio loop runs
in a worker thread and every icon update is handed to the main thread.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import sys
import threading
import webbrowser
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw

from flay.core.events import (
    AssistantState,
    ListeningChanged,
    MuteChanged,
    RecordingChanged,
    StateChanged,
    WakeWordChanged,
)

if TYPE_CHECKING:
    from flay.assistant import Assistant

log = logging.getLogger(__name__)

STATE_COLORS = {
    AssistantState.IDLE: (142, 142, 147),
    AssistantState.LISTENING: (255, 149, 0),
    AssistantState.THINKING: (10, 132, 255),
    AssistantState.SPEAKING: (48, 209, 88),
}
STATE_LABELS = {
    AssistantState.IDLE: "Bereit",
    AssistantState.LISTENING: "Hört zu",
    AssistantState.THINKING: "Denkt nach",
    AssistantState.SPEAKING: "Spricht",
}


def icon_image(state: AssistantState, microphone_muted: bool, size: int = 64) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = size // 8
    draw.ellipse((margin, margin, size - margin, size - margin), fill=STATE_COLORS[state])
    if microphone_muted:
        width = max(2, size // 10)
        draw.line((margin, size - margin, size - margin, margin), fill=(255, 59, 48), width=width)
    return image


def _on_main_thread(function: Callable[[], None]) -> None:
    if sys.platform == "darwin":
        from PyObjCTools import AppHelper

        AppHelper.callAfter(function)
    else:
        function()


class Tray:
    def __init__(self, ui_url: str | None) -> None:
        import pystray

        self._ui_url = ui_url
        self._assistant: Assistant | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._main: asyncio.Task[int] | None = None
        self._state = AssistantState.IDLE
        self._microphone_muted = False
        self._speaker_muted = False
        self._continuous: bool | None = None
        self._wake_word: bool | None = None
        self._recording: str | None = None
        item = pystray.MenuItem
        self._icon = pystray.Icon(
            "flay",
            icon_image(self._state, False),
            "Flay",
            menu=pystray.Menu(
                item(lambda _: f"Flay – {STATE_LABELS[self._state]}", None, enabled=False),
                item(
                    lambda _: f"Ablauf „{self._recording}“ beenden und speichern",
                    self._stop_recording,
                    visible=lambda _: self._recording is not None,
                ),
                pystray.Menu.SEPARATOR,
                item(
                    "Dauerhaft zuhören",
                    self._toggle_listening,
                    checked=lambda _: bool(self._continuous),
                    visible=lambda _: self._continuous is not None,
                ),
                item(
                    "Nur auf „Hey Flay“ reagieren",
                    self._toggle_wake_word,
                    checked=lambda _: bool(self._wake_word),
                    visible=lambda _: self._wake_word is not None,
                ),
                item(
                    "Mikrofon stumm",
                    self._toggle_microphone,
                    checked=lambda _: self._microphone_muted,
                ),
                item(
                    "Sprachausgabe stumm",
                    self._toggle_speaker,
                    checked=lambda _: self._speaker_muted,
                ),
                item("Web-UI öffnen", self._open_ui, visible=ui_url is not None),
                pystray.Menu.SEPARATOR,
                item("Beenden", self._quit),
            ),
        )

    def run(self, main: Callable[[Tray], Awaitable[int]]) -> int:
        """Show the icon (blocks the main thread) while ``main`` runs in a worker thread."""
        result = [0]

        def worker(icon: Any) -> None:
            icon.visible = True
            try:
                result[0] = asyncio.run(self._run_main(main))
            except Exception:
                log.exception("Flay wurde unerwartet beendet")
                result[0] = 1
            finally:
                _on_main_thread(icon.stop)

        self._icon.run(setup=worker)
        return result[0]

    async def _run_main(self, main: Callable[[Tray], Awaitable[int]]) -> int:
        self._main = asyncio.current_task()  # type: ignore[assignment, unused-ignore]
        try:
            return await main(self)
        except asyncio.CancelledError:
            return 0

    def attach(self, assistant: Assistant) -> None:
        """Follow the assistant's state; call from within Flay's event loop."""
        self._assistant = assistant
        self._loop = asyncio.get_running_loop()
        self._microphone_muted = assistant.microphone_muted
        self._speaker_muted = assistant.speaker_muted
        self._continuous = assistant.continuous
        self._wake_word = assistant.wake_word
        self._recording = assistant.workflow_recording
        task = self._loop.create_task(self._follow(assistant))
        task.add_done_callback(lambda _: None)
        self._refresh()

    async def _follow(self, assistant: Assistant) -> None:
        async with assistant.events.subscribe() as events:
            while True:
                event = await events.get()
                if isinstance(event, StateChanged):
                    self._state = event.state
                elif isinstance(event, MuteChanged):
                    self._microphone_muted = event.microphone
                    self._speaker_muted = event.speaker
                elif isinstance(event, ListeningChanged):
                    self._continuous = event.continuous
                elif isinstance(event, WakeWordChanged):
                    self._wake_word = event.enabled
                elif isinstance(event, RecordingChanged):
                    self._recording = event.workflow
                else:
                    continue
                self._refresh()

    def _refresh(self) -> None:
        image = icon_image(self._state, self._microphone_muted)

        def update() -> None:
            self._icon.icon = image
            self._icon.update_menu()

        _on_main_thread(update)

    # Menu actions run on the main thread; the assistant lives on Flay's loop.

    def _call(self, function: Callable[[], None]) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(function)

    def _toggle_listening(self) -> None:
        if self._assistant is not None:
            self._call(functools.partial(self._assistant.set_continuous, not self._continuous))

    def _toggle_wake_word(self) -> None:
        if self._assistant is not None:
            self._call(functools.partial(self._assistant.set_wake_word, not self._wake_word))

    def _stop_recording(self) -> None:
        assistant, loop = self._assistant, self._loop
        if assistant is not None and loop is not None:
            # Like saying it: Flay saves the workflow and answers.
            asyncio.run_coroutine_threadsafe(assistant.submit_text("Ablauf fertig"), loop)

    def _toggle_microphone(self) -> None:
        if self._assistant is not None:
            self._call(
                functools.partial(self._assistant.set_muted, microphone=not self._microphone_muted)
            )

    def _toggle_speaker(self) -> None:
        if self._assistant is not None:
            self._call(
                functools.partial(self._assistant.set_muted, speaker=not self._speaker_muted)
            )

    def _open_ui(self) -> None:
        if self._ui_url:
            threading.Thread(target=webbrowser.open, args=(self._ui_url,), daemon=True).start()

    def _quit(self) -> None:
        main = self._main
        if main is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(main.cancel)
        else:
            self._icon.stop()
