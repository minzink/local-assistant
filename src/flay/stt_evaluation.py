"""`flay eval-stt` and `flay record`: compare speech recognition on the same recordings.

A set of recordings lives in ``<data_dir>/eval/speech/<set>/`` as ``01.wav``, ``02.wav``, …
in the order of ``eval/speech.yaml``. "synthetic" is spoken by Piper (quick to make, but
cleaner than a real voice); the user's own recordings come from ``flay record``.
"""

from __future__ import annotations

import json
import re
import statistics
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from flay.audio.base import AudioBuffer
from flay.audio.microphone import to_pcm16
from flay.core.language import Language
from flay.stt.base import SpeechToText

DEFAULT_SENTENCES = Path(__file__).resolve().parents[2] / "eval" / "speech.yaml"
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Sentence:
    text: str
    language: Language
    terms: tuple[str, ...]


@dataclass
class Recognition:
    expected: str
    heard: str
    wer: float
    terms_missed: list[str]
    terms_total: int
    language_ok: bool
    seconds: float
    # After Flay's correction against the user's vocabulary – what Flay actually works with.
    corrected: str = ""
    terms_missed_corrected: list[str] = field(default_factory=list)


@dataclass
class SttReport:
    backend: str
    recordings: str
    results: list[Recognition] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        seconds = [r.seconds for r in self.results]
        terms = sum(r.terms_total for r in self.results)
        missed = sum(len(r.terms_missed) for r in self.results)
        missed_corrected = sum(len(r.terms_missed_corrected) for r in self.results)
        return {
            "backend": self.backend,
            "recordings": self.recordings,
            "sentences": len(self.results),
            "wer": round(statistics.mean(r.wer for r in self.results), 3),
            "terms_ok": f"{terms - missed}/{terms}",
            "terms_ok_corrected": f"{terms - missed_corrected}/{terms}",
            "sentences_all_terms": sum(not r.terms_missed for r in self.results),
            "language_ok": sum(r.language_ok for r in self.results),
            "median_s": round(statistics.median(seconds), 3),
            "max_s": round(max(seconds), 3),
        }


def load_sentences(path: Path = DEFAULT_SENTENCES) -> list[Sentence]:
    data = yaml.safe_load(path.read_text("utf-8"))
    return [
        Sentence(item["text"], Language(item["language"]), tuple(item.get("terms", ())))
        for item in data["sentences"]
    ]


def normalize(text: str) -> str:
    """Lower case, umlauts spelled out, punctuation removed: "Schließe!" → "schliesse"."""
    text = text.lower()
    for umlaut, plain in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(umlaut, plain)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def word_error_rate(expected: str, heard: str) -> float:
    """Substituted, missing and extra words relative to the expected words."""
    ref, hyp = normalize(expected).split(), normalize(heard).split()
    if not ref:
        return float(bool(hyp))
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, start=1):
        current = [i]
        for j, other in enumerate(hyp, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != other))
            )
        previous = current
    return previous[-1] / len(ref)


def missed_terms(terms: Sequence[str], heard: str) -> list[str]:
    """Terms not in the recognition; spacing does not matter ("K X E W O" has "kxewo")."""
    compact = normalize(heard).replace(" ", "")
    return [term for term in terms if normalize(term).replace(" ", "") not in compact]


# Recordings ---------------------------------------------------------------------------


def recordings_dir(data_dir: Path, name: str) -> Path:
    return data_dir / "eval" / "speech" / name


def save_wav(path: Path, audio: AudioBuffer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(audio.sample_rate)
        file.writeframes(audio.pcm)


def load_wav(path: Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as file:
        return AudioBuffer(pcm=file.readframes(file.getnframes()), sample_rate=file.getframerate())


async def synthesize_set(tts: Any, sentences: list[Sentence], directory: Path) -> None:
    """Speak the sentences with Piper into ``directory`` (16 kHz like the microphone)."""
    for index, sentence in enumerate(sentences, start=1):
        chunks = [chunk async for chunk in tts.synthesize(sentence.text, sentence.language)]
        rate = chunks[0].sample_rate
        samples = np.frombuffer(b"".join(c.pcm for c in chunks), dtype="<i2")
        audio = to_pcm16(samples.astype(np.float32) / 32768.0, rate, SAMPLE_RATE)
        # Half a second of silence around it, like a real push-to-talk recording.
        pad = bytes(SAMPLE_RATE)
        save_wav(directory / f"{index:02}.wav", AudioBuffer(pad + audio + pad, SAMPLE_RATE))


# Evaluation ---------------------------------------------------------------------------


async def evaluate_stt(
    backend: str,
    stt: SpeechToText,
    sentences: list[Sentence],
    directory: Path,
    progress: Callable[[str], None] = print,
    correct: Callable[[str], str] = lambda text: text,
) -> SttReport:
    report = SttReport(backend, directory.name)
    warm_up = getattr(stt, "warm_up", None)
    if callable(warm_up):
        await warm_up()
    for index, sentence in enumerate(sentences, start=1):
        path = directory / f"{index:02}.wav"
        if not path.exists():
            continue
        audio = load_wav(path)
        start = time.perf_counter()
        transcript = await stt.transcribe(audio)
        seconds = time.perf_counter() - start
        corrected = correct(transcript.text)
        result = Recognition(
            expected=sentence.text,
            heard=transcript.text,
            wer=word_error_rate(sentence.text, transcript.text),
            terms_missed=missed_terms(sentence.terms, transcript.text),
            terms_total=len(sentence.terms),
            language_ok=transcript.language is sentence.language,
            seconds=seconds,
            corrected=corrected,
            terms_missed_corrected=missed_terms(sentence.terms, corrected),
        )
        report.results.append(result)
        mark = "✓" if not result.terms_missed and result.wer < 0.2 else "✗"
        progress(f"{mark} {seconds:5.2f}s  {transcript.text[:70]}")
    return report


def save_report(report: SttReport, data_dir: Path) -> Path:
    directory = data_dir / "eval"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = directory / f"{stamp}-stt-{report.backend}-{report.recordings}.json"
    payload = {"summary": report.summary(), "results": [asdict(r) for r in report.results]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), "utf-8")
    return path
