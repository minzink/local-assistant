"""Push-to-talk utterance sources.

``HotkeyPushToTalk`` records while a global key is held (pynput). On macOS the terminal
needs the "Input Monitoring"/"Accessibility" permission, otherwise no key events arrive;
``TerminalPushToTalk`` toggles recording with Enter and needs no permission.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

from pynput import keyboard

from flay.audio.base import AudioBuffer
from flay.audio.microphone import Microphone


def parse_key(name: str) -> Any:
    """Map a config name like ``f8``, ``alt_r`` or ``x`` to a pynput key."""
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name.lower())
    try:
        return keyboard.Key[name.lower()]
    except KeyError:
        raise ValueError(f"unknown push-to-talk key: {name!r}") from None


class _PushToTalk:
    def __init__(self, microphone: Microphone, on_press: Callable[[], None]) -> None:
        self._microphone = microphone
        self._on_press = on_press
        self._queue: asyncio.Queue[AudioBuffer] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _press(self) -> None:
        if self._microphone.is_recording:
            return
        assert self._loop is not None
        # Hotkey callbacks run on pynput's thread; the callback belongs on the event loop.
        self._loop.call_soon_threadsafe(self._on_press)
        self._microphone.start_recording()

    def _release(self) -> None:
        if not self._microphone.is_recording:
            return
        utterance = self._microphone.stop_recording()
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._queue.put_nowait, utterance)

    async def utterances(self) -> AsyncIterator[AudioBuffer]:
        self._loop = asyncio.get_running_loop()
        self._microphone.open()
        listener = self._start_listener()
        try:
            while True:
                yield await self._queue.get()
        finally:
            listener()
            self._microphone.close()

    def _start_listener(self) -> Callable[[], None]:
        """Start delivering key events; returns a function that stops it."""
        raise NotImplementedError


class HotkeyPushToTalk(_PushToTalk):
    def __init__(self, microphone: Microphone, key: str, on_press: Callable[[], None]) -> None:
        super().__init__(microphone, on_press)
        self._key = parse_key(key)

    def _start_listener(self) -> Callable[[], None]:
        def matches(key: Any) -> bool:
            return bool(key == self._key)

        listener = keyboard.Listener(
            on_press=lambda key: self._press() if matches(key) else None,
            on_release=lambda key: self._release() if matches(key) else None,
        )
        listener.start()

        def stop() -> None:
            listener.stop()

        return stop


class TerminalPushToTalk(_PushToTalk):
    """Enter starts recording, Enter again stops it."""

    def _start_listener(self) -> Callable[[], None]:
        # A daemon thread works on every platform (Windows' Proactor loop cannot watch
        # stdin) and does not block interpreter shutdown while waiting for input.
        threading.Thread(target=self._read_lines, daemon=True, name="flay-stdin").start()
        return lambda: None

    def _read_lines(self) -> None:
        assert self._loop is not None
        for _ in sys.stdin:
            self._loop.call_soon_threadsafe(self._toggle)

    def _toggle(self) -> None:
        if self._microphone.is_recording:
            self._release()
        else:
            self._press()
