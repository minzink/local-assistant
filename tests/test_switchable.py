import asyncio
from collections.abc import AsyncIterator

from flay.audio.base import AudioBuffer
from flay.input.switchable import SwitchableSource


def buffer(marker: int) -> AudioBuffer:
    return AudioBuffer(pcm=bytes([marker, 0]), sample_rate=16_000)


class QueueSource:
    """Yields whatever is put into its queue; records whether it is running."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[AudioBuffer] = asyncio.Queue()
        self.running = False

    async def utterances(self) -> AsyncIterator[AudioBuffer]:
        self.running = True
        try:
            while True:
                yield await self.queue.get()
        finally:
            self.running = False


async def test_push_to_talk_always_and_continuous_only_while_switched_on() -> None:
    push_to_talk, continuous = QueueSource(), QueueSource()
    source = SwitchableSource(push_to_talk, lambda: continuous)
    received: list[int] = []

    async def collect() -> None:
        async for utterance in source.utterances():
            received.append(utterance.pcm[0])

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)
    assert not continuous.running and not source.continuous

    push_to_talk.queue.put_nowait(buffer(1))
    source.set_continuous(True)
    await asyncio.sleep(0.01)
    assert continuous.running and source.continuous
    continuous.queue.put_nowait(buffer(2))
    await asyncio.sleep(0.01)

    source.set_continuous(False)
    await asyncio.sleep(0.01)
    assert not continuous.running  # microphone stream closed

    task.cancel()
    assert received == [1, 2]


async def test_starts_in_continuous_mode_when_configured() -> None:
    continuous = QueueSource()
    source = SwitchableSource(None, lambda: continuous, start_continuous=True)

    iterator = source.utterances().__aiter__()
    next_item = asyncio.ensure_future(iterator.__anext__())
    await asyncio.sleep(0.01)

    assert continuous.running
    next_item.cancel()


async def test_restart_builds_the_continuous_source_anew_after_the_old_one_closed() -> None:
    built: list[QueueSource] = []

    def make() -> QueueSource:
        # The old microphone stream must be closed before the new one opens.
        assert not any(s.running for s in built)
        built.append(QueueSource())
        return built[-1]

    source = SwitchableSource(None, make, start_continuous=True)
    iterator = source.utterances().__aiter__()
    next_item = asyncio.ensure_future(iterator.__anext__())
    await asyncio.sleep(0.01)

    source.restart_continuous()
    await asyncio.sleep(0.01)
    assert len(built) == 2 and not built[0].running and built[1].running
    built[1].queue.put_nowait(buffer(7))
    assert (await asyncio.wait_for(next_item, 1)).pcm[0] == 7

    source.set_continuous(False)
    source.restart_continuous()  # off: nothing to restart
    await asyncio.sleep(0.01)
    assert len(built) == 2 and not built[1].running
