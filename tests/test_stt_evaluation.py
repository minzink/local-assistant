from pathlib import Path
from typing import Any

import pytest

from flay.audio.base import AudioBuffer
from flay.core.language import Language
from flay.stt.base import Transcript
from flay.stt.canary import CanarySTT
from flay.stt.vocabulary import Vocabulary
from flay.stt_evaluation import (
    DEFAULT_SENTENCES,
    Sentence,
    evaluate_stt,
    load_sentences,
    load_wav,
    missed_terms,
    normalize,
    save_wav,
    word_error_rate,
)
from tests.test_assistant import ONE_SECOND


def test_normalize() -> None:
    assert normalize("Schließe das Projekt KX-EWO!") == "schliesse das projekt kx ewo"


@pytest.mark.parametrize(
    ("heard", "rate"),
    [
        ("Schließe das Projekt KX EWO.", 0.0),
        ("Schließe das Projekt", 0.4),  # two of five words missing
        ("She das Projekt KX EWO", 0.2),
    ],
)
def test_word_error_rate(heard: str, rate: float) -> None:
    assert word_error_rate("Schließe das Projekt KX EWO.", heard) == pytest.approx(rate)


def test_missed_terms_ignore_spacing_and_umlaut_spelling() -> None:
    terms = ["schließe", "kxewo"]

    assert missed_terms(terms, "Schliesse das Projekt K X E W O") == []
    assert missed_terms(terms, "She das Projekt KEWO") == ["schließe", "kxewo"]


def test_default_sentences_are_valid() -> None:
    sentences = load_sentences(DEFAULT_SENTENCES)

    assert len(sentences) >= 20
    assert {s.language for s in sentences} == {Language.DE, Language.EN}
    assert all(s.terms for s in sentences)


def test_wav_round_trip(tmp_path: Path) -> None:
    save_wav(tmp_path / "a.wav", ONE_SECOND)

    assert load_wav(tmp_path / "a.wav") == ONE_SECOND


class EchoSTT:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    async def transcribe(
        self, audio: AudioBuffer, language_hint: Language | None = None
    ) -> Transcript:
        return Transcript(self.texts.pop(0), Language.DE)


async def test_evaluate_stt_scores_each_recording(tmp_path: Path) -> None:
    sentences = [
        Sentence("Schließe KX Command.", Language.DE, ("schließe", "kxcommand")),
        Sentence("Open the project Flay.", Language.EN, ("open", "flay")),
    ]
    for index in (1, 2):
        save_wav(tmp_path / f"{index:02}.wav", ONE_SECOND)
    stt = EchoSTT(["Schließe KX Command.", "Öffne das Projekt Flay."])

    report = await evaluate_stt("echo", stt, sentences, tmp_path, progress=lambda _: None)

    first, second = report.results
    assert first.wer == 0 and first.terms_missed == [] and first.language_ok
    assert second.terms_missed == ["open"] and not second.language_ok
    assert report.summary()["terms_ok"] == "3/4"


class FakeCanary:
    """Hears German or English and, like Canary, translates into the language it is told."""

    def __init__(self, spoken: Language) -> None:
        self.spoken = spoken
        self.asked: list[str] = []

    def recognize(self, samples: Any, sample_rate: int, language: str) -> str:
        self.asked.append(language)
        if language == "en":
            return "What is on my second monitor?"
        return "Was ist auf meinem zweiten Monitor?"


def canary(spoken: Language, first_text_english: bool) -> tuple[CanarySTT, FakeCanary]:
    stt = CanarySTT(Language.DE)
    model = FakeCanary(spoken)
    if first_text_english:
        model.recognize = lambda samples, sample_rate, language: (  # type: ignore[method-assign]
            model.asked.append(language) or "What is on my second monitor?"
        )
    stt._model = model
    return stt, model


async def test_canary_starts_in_the_default_language_every_time() -> None:
    stt, model = canary(Language.EN, first_text_english=True)

    english = await stt.transcribe(ONE_SECOND)
    stt._model.recognize = FakeCanary(Language.DE).recognize  # type: ignore[union-attr]
    german = await stt.transcribe(ONE_SECOND)

    # The English sentence switched once; the next one starts in German again.
    assert english.language is Language.EN and model.asked == ["de", "en"]
    assert german.language is Language.DE


def test_vocabulary_prompt_lists_terms_and_projects() -> None:
    prompt = Vocabulary(lambda: ["stadt-wolkenthal", "flay"]).prompt()

    assert prompt.startswith("Claude, PhpStorm, Symfony")
    assert "stadt-wolkenthal, flay." in prompt


async def test_voxtral_reads_the_language_from_the_text() -> None:
    from flay.stt.voxtral import VoxtralSTT

    class Result:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeVoxtral:
        def __init__(self) -> None:
            self.delays: list[int] = []

        def generate(self, path: str, transcription_delay_ms: int) -> Result:
            assert Path(path).exists()
            self.delays.append(transcription_delay_ms)
            return Result(" What's on my second monitor? ")

    stt = VoxtralSTT(Language.DE)
    model = FakeVoxtral()
    stt._model = model

    transcript = await stt.transcribe(ONE_SECOND)

    assert transcript == Transcript("What's on my second monitor?", Language.EN)
    assert model.delays == [480]
