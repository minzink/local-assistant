"""Push-to-talk plus continuous listening that can be switched on and off at runtime."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable

from flay.audio.base import AudioBuffer
from flay.input.base import UtteranceSource

log = logging.getLogger(__name__)


class SwitchableSource:
    """Implements ``UtteranceSource``.

    Push-to-talk (if given) always works; continuous listening is an additional source
    whose microphone stream only runs while it is switched on.
    """

    def __init__(
        self,
        push_to_talk: UtteranceSource | None,
        continuous: Callable[[], UtteranceSource],
        start_continuous: bool = False,
    ) -> None:
        self._push_to_talk = push_to_talk
        self._make_continuous = continuous
        self._wanted = start_continuous
        self._queue: asyncio.Queue[AudioBuffer] = asyncio.Queue()
        self._continuous_task: asyncio.Task[None] | None = None

    @property
    def continuous(self) -> bool:
        return self._wanted

    def set_continuous(self, enabled: bool) -> None:
        self._wanted = enabled
        running = self._continuous_task is not None and not self._continuous_task.done()
        if enabled and not running:
            self._continuous_task = asyncio.get_running_loop().create_task(
                self._forward(self._make_continuous())
            )
            log.info("Dauerhaftes Zuhören an")
        elif not enabled and running:
            assert self._continuous_task is not None
            self._continuous_task.cancel()  # closes the microphone stream
            log.info("Dauerhaftes Zuhören aus")

    def restart_continuous(self) -> None:
        """Build the continuous source anew (e.g. with or without the wake word)."""
        old = self._continuous_task
        if not self._wanted or old is None or old.done():
            return
        old.cancel()
        self._continuous_task = asyncio.get_running_loop().create_task(self._restart(old))
        log.info("Dauerhaftes Zuhören neu gestartet")

    async def _restart(self, old: asyncio.Task[None]) -> None:
        # One microphone stream at a time: build the new one once the old one is closed
        # (its outcome does not matter; a cancel of this task still ends it here).
        await asyncio.wait([old])
        await self._forward(self._make_continuous())

    async def _forward(self, source: UtteranceSource) -> None:
        async for utterance in source.utterances():
            self._queue.put_nowait(utterance)

    async def utterances(self) -> AsyncIterator[AudioBuffer]:
        tasks: list[asyncio.Task[None]] = []
        if self._push_to_talk is not None:
            tasks.append(asyncio.create_task(self._forward(self._push_to_talk)))
        self.set_continuous(self._wanted)
        try:
            while True:
                yield await self._queue.get()
        finally:
            if self._continuous_task is not None:
                tasks.append(self._continuous_task)
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
