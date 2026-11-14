import asyncio
import time
from collections.abc import AsyncIterator, Sequence

from cassi.assistant import Assistant
from cassi.audio.base import AudioBuffer
from cassi.config import Settings
from cassi.core.language import Language
from cassi.core.messages import Message
from cassi.llm.base import LLMEvent, ToolSpec
from cassi.llm.gate import SynthesisGate, synthesis
from cassi.tools.base import ToolRegistry
from tests.test_assistant import ONE_SECOND, FakeOutput, FakeSource, FakeSTT, FakeTTS
from tests.test_tools import EchoTool, ToolCallingLLM


async def test_requests_wait_for_queued_speech_but_not_forever() -> None:
    gate = SynthesisGate()
    start = time.perf_counter()
    await gate.wait()
    assert time.perf_counter() - start < 0.01  # nothing queued

    gate.queued()
    gate.queued()
    waiting = asyncio.create_task(gate.wait())
    await asyncio.sleep(0.01)
    gate.synthesized()
    await asyncio.sleep(0.01)
    assert not waiting.done()  # one sentence is still being synthesized
    gate.synthesized()
    await asyncio.wait_for(waiting, 1)

    gate.queued()
    start = time.perf_counter()
    await gate.wait(timeout=0.05)  # synthesis stuck: the request goes ahead anyway
    assert 0.04 < time.perf_counter() - start < 0.5
    gate.reset()
    assert gate.pending == 0


async def test_the_next_model_round_starts_after_the_spoken_sentences_are_synthesized() -> None:
    finished: list[str] = []

    class SlowTTS(FakeTTS):
        async def synthesize(self, text: str, language: Language) -> AsyncIterator[AudioBuffer]:
            async for chunk in super().synthesize(text, language):
                await asyncio.sleep(0.1)  # the GPU is busy with speech
                yield chunk
            finished.append(text)

    seen: list[list[str]] = []

    class GatedLLM(ToolCallingLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            await synthesis.wait()  # as the real client does
            seen.append(list(finished))
            async for event in super().stream(messages, tools):
                yield event

    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("Mach was"),
        GatedLLM(),
        SlowTTS(),
        FakeOutput(),
        tools=ToolRegistry([EchoTool()]),
    )

    await assistant.handle(ONE_SECOND)

    # Round 2 (e.g. reading a screenshot) began once "Moment." was synthesized.
    assert seen == [[], ["Moment."]]
    assert synthesis.pending == 0
