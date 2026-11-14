"""Model requests wait until queued speech is synthesized.

Qwen3-TTS (MLX) and the LLM (Ollama, Metal) share the GPU. Measured on an M2 Pro: alone the
voice is synthesized at 0.31× real time; while the model reads a screenshot (prefill) it
drops to 4.7× – 9 s of speech took 42 s and played with 25 gaps; while it reads a long web
page 1.06× with 4 gaps. While the model only writes (decode) it stays at 0.74×, no gaps.

So a new request – whose prefill is the heavy part – first lets the sentences already
queued for speech be synthesized (about a third of their duration). Their audio then plays
from memory while the model reads. Requests already streaming are not held up, and a
waiting request goes ahead after ``MAX_WAIT_SECONDS`` at the latest.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

log = logging.getLogger(__name__)

MAX_WAIT_SECONDS = 4.0


class SynthesisGate:
    def __init__(self) -> None:
        self._pending = 0
        self._idle: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def pending(self) -> int:
        return self._pending

    def _event(self) -> asyncio.Event:
        # Created lazily: the gate is shared, an event belongs to one running loop.
        loop = asyncio.get_running_loop()
        if self._idle is None or self._loop is not loop:
            self._idle, self._loop = asyncio.Event(), loop
            if self._pending == 0:
                self._idle.set()
        return self._idle

    def queued(self) -> None:
        """A sentence was queued for synthesis."""
        self._pending += 1
        if self._idle is not None:
            self._idle.clear()

    def synthesized(self) -> None:
        """A queued sentence is synthesized (or dropped)."""
        self._pending = max(0, self._pending - 1)
        if self._pending == 0 and self._idle is not None:
            self._idle.set()

    def reset(self) -> None:
        """Speech ended or was cut off: nothing is waiting any more."""
        self._pending = 0
        if self._idle is not None:
            self._idle.set()

    async def wait(self, timeout: float = MAX_WAIT_SECONDS) -> None:
        if self._pending == 0:
            return
        start = time.perf_counter()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._event().wait(), timeout)
        log.debug("Modellanfrage wartete %.2fs auf die Sprachsynthese", time.perf_counter() - start)


# One per process: every model client (main model, pointer model) waits for the same voice.
synthesis = SynthesisGate()
