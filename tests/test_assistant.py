import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence

import numpy as np

from flay.assistant import Assistant
from flay.audio.base import AudioBuffer
from flay.config import Settings
from flay.core.language import Language
from flay.core.messages import Message, Role
from flay.llm.base import LLMEvent, ModelCapabilities, TextDelta, ToolSpec
from flay.stt.base import Transcript

# One second of a quiet 440 Hz tone (about -27 dBFS), loud enough to count as speech.
_TONE = (np.sin(np.arange(16_000) * 2 * np.pi * 440 / 16_000) * 2_000).astype("<i2")
ONE_SECOND = AudioBuffer(pcm=_TONE.tobytes(), sample_rate=16_000)
SILENT_SECOND = AudioBuffer(pcm=bytes(32_000), sample_rate=16_000)


class FakeSource:
    def __init__(self, *utterances: AudioBuffer) -> None:
        self._utterances = utterances

    async def utterances(self) -> AsyncIterator[AudioBuffer]:
        for utterance in self._utterances:
            yield utterance


class FakeSTT:
    def __init__(self, text: str, language: Language = Language.DE) -> None:
        self.transcript = Transcript(text, language)
        self.hints: list[Language | None] = []

    async def transcribe(
        self, audio: AudioBuffer, language_hint: Language | None = None
    ) -> Transcript:
        self.hints.append(language_hint)
        return self.transcript


class FakeLLM:
    def __init__(self, *deltas: str) -> None:
        self._deltas = deltas
        self.prompts: list[Sequence[Message]] = []

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.prompts.append(messages)
        for delta in self._deltas:
            yield TextDelta(delta)


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, Language]] = []

    async def synthesize(self, text: str, language: Language) -> AsyncIterator[AudioBuffer]:
        self.spoken.append((text, language))
        yield AudioBuffer(pcm=b"\x00\x00", sample_rate=22_050)


class FakeOutput:
    def __init__(self) -> None:
        self.played: list[AudioBuffer] = []
        self.stopped = 0

    async def play(self, chunks: AsyncIterator[AudioBuffer]) -> bool:
        async for chunk in chunks:
            self.played.append(chunk)
        return True

    def stop(self) -> None:
        self.stopped += 1


def make_assistant(
    stt: FakeSTT, llm: FakeLLM, tts: FakeTTS, output: FakeOutput, settings: Settings | None = None
) -> Assistant:
    return Assistant(settings or Settings(), FakeSource(), stt, llm, tts, output)


async def test_full_turn() -> None:
    stt, llm, tts, output = (
        FakeSTT("Wie geht's?"),
        FakeLLM("Gut, ", "danke!"),
        FakeTTS(),
        FakeOutput(),
    )
    assistant = make_assistant(stt, llm, tts, output)

    await assistant.handle(ONE_SECOND)

    assert tts.spoken == [("Gut, danke!", Language.DE)]
    assert len(output.played) == 1
    assert [m.role for m in assistant.history] == [Role.USER, Role.ASSISTANT]
    system = llm.prompts[0][0]
    assert system.role is Role.SYSTEM
    assert "Always reply in German." in system.text


async def test_answers_in_detected_language() -> None:
    stt, llm, tts = FakeSTT("Hello", Language.EN), FakeLLM("Hi!"), FakeTTS()
    assistant = make_assistant(stt, llm, tts, FakeOutput())

    await assistant.handle(ONE_SECOND)

    assert "Always reply in English." in llm.prompts[0][0].text
    assert llm.prompts[0][-1].text.endswith("Reply in English.)")
    assert tts.spoken == [("Hi!", Language.EN)]
    assert stt.hints == [None]


async def test_fixed_language_is_passed_as_hint() -> None:
    settings = Settings.model_validate({"language": {"default": "en", "auto_detect": False}})
    stt = FakeSTT("Hello", Language.EN)

    await make_assistant(stt, FakeLLM("Hi"), FakeTTS(), FakeOutput(), settings).handle(ONE_SECOND)

    assert stt.hints == [Language.EN]


async def test_short_or_empty_input_is_ignored() -> None:
    llm, tts = FakeLLM("x"), FakeTTS()
    too_short = AudioBuffer(pcm=bytes(3_200), sample_rate=16_000)

    await make_assistant(FakeSTT("Hallo"), llm, tts, FakeOutput()).handle(too_short)
    await make_assistant(FakeSTT(""), llm, tts, FakeOutput()).handle(ONE_SECOND)

    assert llm.prompts == []
    assert tts.spoken == []


async def test_history_is_sent_with_next_turn() -> None:
    llm = FakeLLM("Antwort")
    assistant = make_assistant(FakeSTT("Frage"), llm, FakeTTS(), FakeOutput())

    await assistant.handle(ONE_SECOND)
    await assistant.handle(ONE_SECOND)

    texts = [m.text for m in llm.prompts[1][1:]]
    assert texts[:2] == ["Frage", "Antwort"]
    assert texts[2].startswith("Frage\n(Now: ") and texts[2].endswith("Reply in German.)")


async def test_new_utterance_cancels_running_turn() -> None:
    started = asyncio.Event()

    class SlowLLM(FakeLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            self.prompts.append(messages)
            if not started.is_set():
                started.set()
                await asyncio.sleep(10)
            yield TextDelta("Neue Antwort")

    class TwoUtterances(FakeSource):
        async def utterances(self) -> AsyncIterator[AudioBuffer]:
            yield ONE_SECOND
            await started.wait()
            yield ONE_SECOND

    llm, tts, output = SlowLLM(), FakeTTS(), FakeOutput()
    assistant = Assistant(Settings(), TwoUtterances(), FakeSTT("Hallo"), llm, tts, output)

    await assistant.run()

    assert len(llm.prompts) == 2
    assert output.stopped >= 1
    assert tts.spoken == [("Neue Antwort", Language.DE)]


async def test_mute_recording_is_not_transcribed() -> None:
    stt = FakeSTT("you", Language.EN)

    await make_assistant(stt, FakeLLM("x"), FakeTTS(), FakeOutput()).handle(SILENT_SECOND)

    assert stt.hints == []


def test_level_dbfs() -> None:
    assert SILENT_SECOND.level_dbfs == float("-inf")
    assert -28 < ONE_SECOND.level_dbfs < -26


async def test_speech_starts_before_llm_finishes() -> None:
    release = asyncio.Event()
    spoken_before_llm_done: list[list[tuple[str, Language]]] = []
    tts = FakeTTS()

    class GatedLLM(FakeLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            yield TextDelta("Erster Satz. ")
            await release.wait()
            yield TextDelta("Zweiter Satz.")

    class RecordingOutput(FakeOutput):
        async def play(self, chunks: AsyncIterator[AudioBuffer]) -> bool:
            async for chunk in chunks:
                if not release.is_set():
                    spoken_before_llm_done.append(list(tts.spoken))
                    release.set()
                self.played.append(chunk)
            return True

    output = RecordingOutput()
    assistant = make_assistant(FakeSTT("Erzähl was"), GatedLLM(), tts, output)

    await assistant.handle(ONE_SECOND)

    assert spoken_before_llm_done == [[("Erster Satz.", Language.DE)]]
    assert tts.spoken == [("Erster Satz.", Language.DE), ("Zweiter Satz.", Language.DE)]
    assert len(output.played) == 2
    assert assistant.history[-1].text == "Erster Satz. Zweiter Satz."


async def test_interrupted_playback_stops_generation() -> None:
    generating = asyncio.Event()
    finished = False

    class EndlessLLM(FakeLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            nonlocal finished
            yield TextDelta("Satz eins. ")
            generating.set()
            await asyncio.sleep(10)
            finished = True
            yield TextDelta("Satz zwei.")

    class StoppingOutput(FakeOutput):
        async def play(self, chunks: AsyncIterator[AudioBuffer]) -> bool:
            await generating.wait()
            return False

    assistant = make_assistant(FakeSTT("Hallo"), EndlessLLM(), FakeTTS(), StoppingOutput())

    await asyncio.wait_for(assistant.handle(ONE_SECOND), timeout=2)

    assert not finished
    # The request stays in the conversation; nothing of the reply was heard.
    user, reply = assistant.history
    assert user.text == "Hallo"
    assert reply.interrupted and reply.text == ""


class FakeVAD:
    def __init__(self, result: AudioBuffer | None) -> None:
        self.result = result

    async def trim(self, audio: AudioBuffer) -> AudioBuffer | None:
        return self.result


async def test_vad_without_speech_skips_transcription() -> None:
    stt = FakeSTT("Untertitel im Auftrag des ZDF")
    assistant = Assistant(
        Settings(), FakeSource(), stt, FakeLLM("x"), FakeTTS(), FakeOutput(), vad=FakeVAD(None)
    )

    await assistant.handle(ONE_SECOND)

    assert stt.hints == []


async def test_vad_trimmed_audio_is_transcribed() -> None:
    trimmed = AudioBuffer(pcm=ONE_SECOND.pcm[:16_000], sample_rate=16_000)
    received: list[AudioBuffer] = []

    class RecordingSTT(FakeSTT):
        async def transcribe(
            self, audio: AudioBuffer, language_hint: Language | None = None
        ) -> Transcript:
            received.append(audio)
            return self.transcript

    assistant = Assistant(
        Settings(),
        FakeSource(),
        RecordingSTT("Hallo"),
        FakeLLM("Hi"),
        FakeTTS(),
        FakeOutput(),
        vad=FakeVAD(trimmed),
    )

    await assistant.handle(ONE_SECOND)

    assert received == [trimmed]


async def test_memory_is_paused_during_turn_and_receives_exchange() -> None:
    from flay.memory.base import MemoryContext

    events: list[str] = []

    class SpyMemory:
        def context(self) -> MemoryContext:
            events.append("context")
            return MemoryContext()

        def record(self, user: Message, assistant: Message) -> None:
            events.append(f"record {user.text} -> {assistant.text}")

        def turn_started(self) -> None:
            events.append("started")

        def turn_finished(self) -> None:
            events.append("finished")

        async def close(self) -> None:
            pass

    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("Hallo"),
        FakeLLM("Hi!"),
        FakeTTS(),
        FakeOutput(),
        memory=SpyMemory(),
    )

    await assistant.handle(ONE_SECOND)

    # Read twice: the recent requests for the project check, then the prompt.
    assert events == ["started", "context", "context", "record Hallo -> Hi!", "finished"]


def collect(assistant: Assistant) -> list[object]:
    events: list[object] = []
    original = assistant.events.publish

    def publish(event: object) -> None:
        events.append(event)
        original(event)  # type: ignore[arg-type]

    assistant.events.publish = publish  # type: ignore[method-assign]
    return events


async def test_typed_text_is_answered_without_stt() -> None:
    from flay.core.events import AssistantMessage, UserMessage

    stt, llm, tts = FakeSTT("unbenutzt"), FakeLLM("Paris."), FakeTTS()
    assistant = make_assistant(stt, llm, tts, FakeOutput())
    events = collect(assistant)

    await assistant.handle_text("What is the capital of France?")

    assert stt.hints == []
    assert tts.spoken == [("Paris.", Language.EN)]
    assert UserMessage("What is the capital of France?", Language.EN, "text") in events
    assert AssistantMessage("Paris.", Language.EN) in events


async def test_submit_text_replaces_running_turn() -> None:
    started = asyncio.Event()

    class SlowOnce(FakeLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            self.prompts.append(messages)
            if not started.is_set():
                started.set()
                await asyncio.sleep(10)
            yield TextDelta("Antwort.")

    llm, tts = SlowOnce(), FakeTTS()
    assistant = make_assistant(FakeSTT("x"), llm, tts, FakeOutput())

    await assistant.submit_text("Erste Frage")
    await started.wait()
    await assistant.submit_text("Zweite Frage")
    assert assistant._turn is not None
    await assistant._turn

    assert len(llm.prompts) == 2
    assert tts.spoken == [("Antwort.", Language.DE)]


async def test_muted_speaker_answers_as_text_only() -> None:
    tts, output = FakeTTS(), FakeOutput()
    assistant = make_assistant(FakeSTT("Hallo"), FakeLLM("Hi!"), tts, output)
    assistant.set_muted(speaker=True)

    await assistant.handle(ONE_SECOND)

    assert tts.spoken == []
    assert output.played == []
    assert [m.text for m in assistant.history] == ["Hallo", "Hi!"]


async def test_muting_speaker_mid_reply_keeps_generating_text() -> None:
    from flay.core.events import AssistantMessage

    first_audio = asyncio.Event()

    class SlowLLM(FakeLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            yield TextDelta("Satz eins. ")
            await first_audio.wait()
            yield TextDelta("Satz zwei.")

    class StoppableOutput(FakeOutput):
        def __init__(self) -> None:
            super().__init__()
            self.stopped_event = asyncio.Event()

        async def play(self, chunks: AsyncIterator[AudioBuffer]) -> bool:
            async for _ in chunks:
                first_audio.set()
                await self.stopped_event.wait()
                return False
            return True

        def stop(self) -> None:
            super().stop()
            self.stopped_event.set()

    output = StoppableOutput()
    assistant = make_assistant(FakeSTT("Erzähl"), SlowLLM(), FakeTTS(), output)
    events = collect(assistant)
    turn = asyncio.create_task(assistant.handle(ONE_SECOND))
    await first_audio.wait()

    assistant.set_muted(speaker=True)
    await turn

    assert AssistantMessage("Satz eins. Satz zwei.", Language.DE) in events
    assert assistant.history[-1].text == "Satz eins. Satz zwei."


async def test_muted_microphone_drops_utterances_and_ignores_key() -> None:
    from flay.core.events import StateChanged

    stt, output = FakeSTT("Hallo"), FakeOutput()
    assistant = Assistant(Settings(), FakeSource(ONE_SECOND), stt, FakeLLM("Hi"), FakeTTS(), output)
    assistant.set_muted(microphone=True)
    events = collect(assistant)

    assistant.interrupt()
    await assistant.run()

    assert stt.hints == []
    assert output.stopped == 0
    assert not any(isinstance(e, StateChanged) for e in events)


async def test_states_during_voice_turn() -> None:
    from flay.core.events import AssistantState, StateChanged

    assistant = make_assistant(FakeSTT("Hallo"), FakeLLM("Hi."), FakeTTS(), FakeOutput())
    events = collect(assistant)

    assistant.interrupt()
    await assistant.handle(ONE_SECOND)

    states = [e.state for e in events if isinstance(e, StateChanged)]
    assert states[0] is AssistantState.LISTENING
    assert AssistantState.THINKING in states
    assert AssistantState.SPEAKING in states
    assert states[-1] is AssistantState.IDLE


async def test_keep_warm_sends_the_usual_prompt_when_idle() -> None:
    warmed: list[tuple[list[Message], list[str]]] = []

    class WarmableLLM(FakeLLM):
        async def warm(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> None:
            warmed.append((list(messages), [t.name for t in tools]))

    assistant = make_assistant(FakeSTT("x"), WarmableLLM(), FakeTTS(), FakeOutput())
    assistant._last_activity -= 100  # idle for a while

    task = asyncio.create_task(assistant.keep_warm(idle_seconds=60, check_every=0.01))
    await asyncio.sleep(0.05)
    task.cancel()

    assert len(warmed) == 1  # activity is reset after warming
    assert warmed[0][0][0].role is Role.SYSTEM


async def test_keep_warm_does_nothing_without_support_or_when_disabled() -> None:
    assistant = make_assistant(FakeSTT("x"), FakeLLM(), FakeTTS(), FakeOutput())

    await asyncio.wait_for(assistant.keep_warm(idle_seconds=60, check_every=0.01), 1)
    await asyncio.wait_for(assistant.keep_warm(idle_seconds=0), 1)


async def test_prefetched_transcript_is_used_when_the_user_stopped() -> None:
    calls: list[int] = []

    class CountingSTT(FakeSTT):
        async def transcribe(
            self, audio: AudioBuffer, language_hint: Language | None = None
        ) -> Transcript:
            calls.append(len(audio.pcm))
            return self.transcript

    tts = FakeTTS()
    assistant = make_assistant(CountingSTT("Hallo"), FakeLLM("Hi!"), tts, FakeOutput())
    tentative = AudioBuffer(ONE_SECOND.pcm, 16_000)
    final = AudioBuffer(ONE_SECOND.pcm + bytes(8_000), 16_000)  # plus 0.25 s silence

    assistant.prefetch(tentative)
    await asyncio.sleep(0)
    await assistant.handle(final)

    assert calls == [len(tentative.pcm)]  # recognized once, early
    assert tts.spoken == [("Hi!", Language.DE)]


async def test_prefetch_is_dropped_when_the_user_kept_talking() -> None:
    calls: list[int] = []

    class CountingSTT(FakeSTT):
        async def transcribe(
            self, audio: AudioBuffer, language_hint: Language | None = None
        ) -> Transcript:
            calls.append(len(audio.pcm))
            return self.transcript

    assistant = make_assistant(CountingSTT("Hallo"), FakeLLM("Hi!"), FakeTTS(), FakeOutput())
    assistant.prefetch(AudioBuffer(ONE_SECOND.pcm[:16_000], 16_000))
    await asyncio.sleep(0)
    different = AudioBuffer(bytes(reversed(ONE_SECOND.pcm)), 16_000)

    await assistant.handle(different)

    assert calls[-1] == len(different.pcm)


async def test_interrupted_reply_keeps_only_what_was_heard() -> None:
    class OneSentenceOutput(FakeOutput):
        async def play(self, chunks: AsyncIterator[AudioBuffer]) -> bool:
            async for chunk in chunks:
                self.played.append(chunk)
                return False  # the user cuts in during the first sentence
            return True

    llm = FakeLLM("Satz eins. ", "Satz zwei. ", "Satz drei.")
    assistant = make_assistant(FakeSTT("Hallo"), llm, FakeTTS(), OneSentenceOutput())

    await assistant.handle(ONE_SECOND)

    reply = assistant.history[-1]
    assert reply.interrupted and reply.text == "Satz eins."
    # The next turn tells the model where it was cut off.
    await assistant.handle_text("Nein, anders.")
    sent = llm.prompts[-1]
    assert sent[-2].text == "Satz eins. …"
    assert sent[-1].text.startswith("(I interrupted you there.) Nein, anders.")


async def test_request_cancelled_before_any_answer_is_kept() -> None:
    started = asyncio.Event()

    class HangingLLM(FakeLLM):
        async def stream(
            self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
        ) -> AsyncIterator[LLMEvent]:
            started.set()
            await asyncio.sleep(10)
            yield TextDelta("zu spät")

    assistant = make_assistant(FakeSTT("Erste Frage"), HangingLLM(), FakeTTS(), FakeOutput())
    turn = asyncio.create_task(assistant.handle(ONE_SECOND))
    await started.wait()
    turn.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await turn

    user, reply = assistant.history
    assert user.text == "Erste Frage"
    assert reply.interrupted and reply.text == ""


class LoudTTS(FakeTTS):
    """Audible output: fillers are trimmed to their sound, pure silence is dropped."""

    async def synthesize(self, text: str, language: Language) -> AsyncIterator[AudioBuffer]:
        self.spoken.append((text, language))
        yield AudioBuffer(pcm=_TONE[:800].tobytes(), sample_rate=16_000)


def filler_settings(delay_ms: int) -> Settings:
    return Settings.model_validate(
        {"tts": {"filler": {"delay_ms": delay_ms, "phrases": {"de": ["Mhm."], "en": ["Mhm."]}}}}
    )


class SlowLLM(FakeLLM):
    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        await asyncio.sleep(0.2)
        async for event in super().stream(messages, tools):
            yield event


async def test_filler_plays_while_the_answer_takes_long() -> None:
    tts, output = LoudTTS(), FakeOutput()
    assistant = make_assistant(
        FakeSTT("Hallo"), SlowLLM("Hallo!"), tts, output, filler_settings(delay_ms=50)
    )

    await assistant.handle(ONE_SECOND)

    assert [text for text, _ in tts.spoken] == ["Mhm.", "Hallo!"]
    assert len(output.played) == 2
    # The filler is not part of the conversation.
    assert assistant.history[-1].text == "Hallo!"


async def test_no_filler_when_the_answer_is_fast_or_typed() -> None:
    tts = FakeTTS()
    late = filler_settings(1000)
    fast = make_assistant(FakeSTT("Hallo"), FakeLLM("Hi!"), tts, FakeOutput(), late)
    typed = make_assistant(FakeSTT("x"), SlowLLM("Hi!"), tts, FakeOutput(), filler_settings(0))

    await fast.handle(ONE_SECOND)
    await typed.handle_text("Hallo")

    assert [text for text, _ in tts.spoken] == ["Hi!", "Hi!"]


async def test_filler_plays_on_its_own_while_speech_is_still_recognized() -> None:
    class SlowSTT(FakeSTT):
        async def transcribe(
            self, audio: AudioBuffer, language_hint: Language | None = None
        ) -> Transcript:
            await asyncio.sleep(0.2)
            return self.transcript

    tts = LoudTTS()
    calls: list[int] = []

    class CountingOutput(FakeOutput):
        async def play(self, chunks: AsyncIterator[AudioBuffer]) -> bool:
            calls.append(1)
            return await super().play(chunks)

    output = CountingOutput()
    assistant = make_assistant(
        SlowSTT("Hallo"), FakeLLM("Hallo!"), tts, output, filler_settings(delay_ms=20)
    )

    await assistant.handle(ONE_SECOND)

    assert [text for text, _ in tts.spoken] == ["Mhm.", "Hallo!"]
    assert len(calls) == 2  # the filler, then the answer


async def test_lone_noise_word_is_dropped_while_listening_continuously() -> None:
    from flay.stt.vocabulary import Vocabulary

    class Continuous(FakeSource):
        continuous = True

    llm = FakeLLM("Was meinst du?")
    assistant = Assistant(
        Settings(),
        Continuous(),
        FakeSTT("Matchup."),
        llm,
        FakeTTS(),
        FakeOutput(),
        vocabulary=Vocabulary(lambda: ["flay"]),
    )

    await assistant.handle(ONE_SECOND)

    assert llm.prompts == []
    assert assistant.history == []


async def test_transcript_is_corrected_before_answering() -> None:
    from flay.stt.vocabulary import Vocabulary

    llm = FakeLLM("Okay.")
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("Erzähl was über Stadt Wolkental."),
        llm,
        FakeTTS(),
        FakeOutput(),
        vocabulary=Vocabulary(lambda: ["stadt-wolkenthal"]),
    )

    await assistant.handle(ONE_SECOND)

    assert assistant.history[0].text == "Erzähl was über Stadt Wolkenthal."


async def test_computer_state_goes_into_the_new_message_only() -> None:
    async def state() -> str | None:
        return "Open in PhpStorm: flay."

    llm = FakeLLM("Ja.")
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput(), state=state
    )

    await assistant.handle_text("Ist flay offen?")

    sent = llm.prompts[0]
    assert "Open in PhpStorm: flay." in sent[-1].text
    # Not in the system prompt: that stays the same between turns and is cached.
    assert "PhpStorm: flay" not in sent[0].text
    assert "Open in PhpStorm" not in assistant.history[0].text


async def test_slow_computer_state_does_not_hold_up_the_answer() -> None:
    async def state() -> str | None:
        await asyncio.sleep(5)
        return "zu spät"

    llm = FakeLLM("Ja.")
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput(), state=state
    )

    await asyncio.wait_for(assistant.handle_text("Hallo"), timeout=2)

    assert "zu spät" not in llm.prompts[0][-1].text


async def test_interrupting_while_the_filler_hands_over_cancels_the_turn() -> None:
    import time as clock

    from flay.config import FillerSettings
    from flay.filler import FillerVoice, TurnFiller

    class SlowOutput(FakeOutput):
        async def play(self, chunks: AsyncIterator[AudioBuffer]) -> bool:
            await asyncio.sleep(1)  # the filler is still playing
            return True

    voice = FillerVoice(LoudTTS(), FillerSettings(delay_ms=0))
    filler = TurnFiller(voice, SlowOutput(), Language.DE, clock.monotonic)
    await asyncio.sleep(0.05)  # the filler started playing

    handing_over = asyncio.create_task(filler.hand_over())
    await asyncio.sleep(0.05)
    handing_over.cancel()  # the user interrupts right then

    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(handing_over, timeout=0.5)
    assert handing_over.cancelled()
    filler.cancel()


async def test_recalled_memories_reach_the_model() -> None:
    from flay.memory.base import Fact, ShortTermMemory

    class RecallingMemory(ShortTermMemory):
        async def recall(self, query: str) -> tuple[Fact, ...]:
            return (Fact(1, "Mia hat am Freitag Urlaub."),)

    llm = FakeLLM("Am Freitag.")
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        memory=RecallingMemory(),
    )

    await assistant.handle_text("Wann habe ich frei?")

    assert "Mia hat am Freitag Urlaub." in llm.prompts[0][-1].text


async def test_slow_memory_search_does_not_hold_up_the_answer() -> None:
    from flay.memory.base import Fact, ShortTermMemory

    class SlowMemory(ShortTermMemory):
        async def recall(self, query: str) -> tuple[Fact, ...]:
            await asyncio.sleep(5)
            return (Fact(1, "zu spät"),)

    llm = FakeLLM("Hallo.")
    assistant = Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput(), memory=SlowMemory()
    )

    await asyncio.wait_for(assistant.handle_text("Hallo"), timeout=2)

    assert "zu spät" not in llm.prompts[0][-1].text
