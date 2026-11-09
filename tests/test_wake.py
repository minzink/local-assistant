from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from flay.audio.base import AudioBuffer
from flay.core.language import Language
from flay.voice_recording import Progress, check_take
from flay.wake import WINDOW, Classifier, WakeGate, strip_wake_word, windows
from flay.wake_training import Clip, labelled, train


@pytest.mark.parametrize(
    ("heard", "command"),
    [
        ("Hey Flay, öffne das Projekt flay.", "öffne das Projekt flay."),
        ("Hey Fley öffne flay", "öffne flay"),
        ("hey flay", ""),
        ("Hey Flay?", ""),
        ("Öffne das Projekt flay", "Öffne das Projekt flay"),
        ("Display the logs", "Display the logs"),
        ("Hey, play some music", "Hey, play some music"),
        ("Hey, Flea, sag mir, was du siehst.", "sag mir, was du siehst."),
        ("Flee the building", "Flee the building"),
        ("Hey, fly me to the moon", "Hey, fly me to the moon"),
    ],
)
def test_the_wake_word_is_taken_off_the_request(heard: str, command: str) -> None:
    assert strip_wake_word(heard) == command


def test_windows_cover_16_embeddings() -> None:
    embeddings = np.arange(20 * 96, dtype=np.float32).reshape(20, 96)

    assert windows(embeddings).shape == (5, WINDOW * 96)
    assert windows(embeddings[:3]).shape == (1, WINDOW * 96)  # short audio padded


def test_a_small_network_learns_a_simple_rule(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, WINDOW * 96)).astype(np.float32)
    y = (x[:, 0] > 1).astype(np.float32)

    classifier = train(x, y, hidden=16, epochs=30)

    accuracy = ((classifier(x) > 0.5) == (y > 0.5)).mean()
    assert accuracy > 0.95
    classifier.save(tmp_path / "wake.npz")
    again = Classifier.load(tmp_path / "wake.npz")
    assert np.allclose(again(x[:5]), classifier(x[:5]))


class FakeFeatures:
    """One embedding per 80 ms, its first value the time it ends."""

    def embeddings(self, audio: np.ndarray) -> np.ndarray:
        count = max(1, (len(audio) // 160 - 76) // 8 + 1)
        values = np.zeros((count, 96), np.float32)
        values[:, 0] = np.arange(count)
        return values


def test_windows_are_labelled_by_where_the_wake_word_ends() -> None:
    clip = Clip(np.zeros(16000 * 4, np.float32), "voice", phrase=(1.0, 1.8))

    x, y = labelled(FakeFeatures(), clip)  # type: ignore[arg-type]

    assert y.sum() >= 3  # windows ending 0–0.32 s after the phrase
    assert len(x) == len(y) and set(np.unique(y)) <= {0.0, 1.0}
    negative = Clip(np.zeros(16000 * 3, np.float32), "voice")
    assert labelled(FakeFeatures(), negative)[1].sum() == 0  # type: ignore[arg-type]


class FakeWake:
    threshold = 0.5

    def score(self, audio: np.ndarray) -> float:
        return float(audio.max() > 0.5)  # loud = "Hey Flay"


def utterance(seconds: float, loud: bool) -> AudioBuffer:
    samples = np.full(int(seconds * 16000), 0.9 if loud else 0.1, np.float32)
    return AudioBuffer((samples * 32767).astype("<i2").tobytes(), 16000)


class Source:
    def __init__(self, *items: AudioBuffer) -> None:
        self.items = items

    async def utterances(self) -> AsyncIterator[AudioBuffer]:
        for item in self.items:
            yield item


async def test_only_utterances_with_the_wake_word_pass() -> None:
    now = [0.0]
    woken: list[int] = []
    talk, wake_and_command, wake_alone, command, later = (
        utterance(3, False),
        utterance(3, True),
        utterance(1, True),
        utterance(2, False),
        utterance(2, False),
    )
    source = Source(talk, wake_and_command, wake_alone, command, later)
    gate = WakeGate(source, FakeWake(), on_wake=lambda: woken.append(1), clock=lambda: now[0])  # type: ignore[arg-type]

    passed = []
    async for item in gate.utterances():
        passed.append(item)
        now[0] += 1

    # Background talk is dropped; "Hey Flay" alone passes (Flay answers "Ja?") and lets the
    # next utterance through without it – but only that one.
    assert passed == [wake_and_command, wake_alone, command]
    assert len(woken) == 2


def test_takes_are_checked(tmp_path: Path) -> None:
    text = "Das ist ein ganz normaler Satz zum Vorlesen."
    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(16000 * 3) * 3000).astype("<i2")
    assert check_take(AudioBuffer(speech.tobytes(), 16000), text, -50).ok
    silent = np.zeros(16000 * 3, "<i2")
    assert "Stille" in check_take(AudioBuffer(silent.tobytes(), 16000), text, -50).problem
    clipped = speech.copy()
    clipped[100] = 32767
    assert "übersteuert" in check_take(AudioBuffer(clipped.tobytes(), 16000), text, -50).problem
    short = speech[:4000]
    assert "zu kurz" in check_take(AudioBuffer(short.tobytes(), 16000), text, -50).problem

    progress = Progress(tmp_path)
    (tmp_path / "de_0001.wav").write_bytes(b"")
    progress.add(Language.DE, text, "de_0001.wav", 3.0)
    progress.add(Language.DE, "Verloren.", "de_0002.wav", 1.0)  # its file is gone
    assert progress.done() == {("de", text)} and progress.seconds() == 4.0


class ContinuousSource:
    continuous = True

    async def utterances(self) -> AsyncIterator[AudioBuffer]:
        return
        yield


class ChangingSTT:
    """Hears the next text of a list each time."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)

    async def transcribe(self, audio: AudioBuffer, language_hint: object = None) -> object:
        from flay.stt.base import Transcript

        return Transcript(self.texts.pop(0), Language.DE)


async def test_the_transcript_decides_when_listening_with_the_wake_word() -> None:
    from flay.assistant import Assistant
    from flay.config import Settings
    from tests.test_assistant import ONE_SECOND, FakeLLM, FakeOutput, FakeTTS

    settings = Settings.model_validate({"input": {"continuous": {"wake_word": True}}})
    llm = FakeLLM("Gern.")
    assistant = Assistant(
        settings,
        ContinuousSource(),  # type: ignore[arg-type]
        ChangingSTT(
            "Wir treffen uns um drei.",  # someone nearby: dropped
            "Hey Flay, wie spät ist es?",
            "Hey Flay.",  # alone: "Ja?", the next one needs no wake word
            "Erzähl einen Witz.",
            "Und dann gehen wir essen.",  # the window is used up: dropped
        ),
        llm,
        FakeTTS(),
        FakeOutput(),
    )

    for _ in range(5):
        await assistant._handle_audio(ONE_SECOND)

    asked = [m.text for m in assistant.history if m.role.value == "user"]
    assert asked == ["wie spät ist es?", "Hey Flay.", "Erzähl einen Witz."]
    assert assistant.history[3].text == "Ja?"


async def test_talk_nearby_does_not_cancel_the_answer() -> None:
    import asyncio

    from flay.assistant import Assistant
    from flay.config import Settings
    from tests.test_assistant import ONE_SECOND, FakeLLM, FakeOutput, FakeTTS

    class SlowLLM(FakeLLM):
        async def stream(self, messages, tools=(), think=False):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.2)  # still answering when the next utterance comes
            async for event in super().stream(messages, tools):
                yield event

    class Twice(ContinuousSource):
        async def utterances(self) -> AsyncIterator[AudioBuffer]:
            yield ONE_SECOND
            yield ONE_SECOND

    settings = Settings.model_validate({"input": {"continuous": {"wake_word": True}}})
    assistant = Assistant(
        settings,
        Twice(),  # type: ignore[arg-type]
        ChangingSTT("Hey Flay, erzähl einen Witz.", "Wir treffen uns um drei."),
        SlowLLM("Ein Witz."),
        FakeTTS(),
        FakeOutput(),
    )

    await assistant.run()

    assert [m.text for m in assistant.history] == ["erzähl einen Witz.", "Ein Witz."]
    assert not assistant.history[-1].interrupted
