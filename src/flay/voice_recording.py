"""Recordings of the user's own voice: sentences for a Piper voice, "Hey Flay" for the wake word.

``flay voice dataset`` reads the 965 checked sentences of the voice training aloud, one at a
time; ``tools/voice-training`` turns them into a Piper voice (``dataset.ps1 -Recorded``).
Each take is checked right away (silence, clipping, speaking rate) and can be repeated;
stopping and starting again continues where it ended. 30–60 minutes give a usable voice.

``flay voice wake`` records "Hey Flay" twenty times in different ways for ``flay train-wake``:
the detector then knows the real voice, not only synthetic ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flay.audio.base import AudioBuffer
from flay.core.language import Language

SENTENCES = Path(__file__).parent / "data" / "sentences"
DATASET_RATE = 22_050
WAKE_RATE = 16_000
# Characters per second of natural reading; slower is hesitation, faster is cut off.
PACE = (7.0, 22.0)
WAKE_TAKES = (
    "normal",
    "normal",
    "normal",
    "leise",
    "leise",
    "laut",
    "schnell",
    "schnell",
    "langsam",
    "fragend („Hey Flay?“)",
    "beiläufig",
    "mit etwas Abstand zum Mikrofon",
    "mit etwas Abstand zum Mikrofon",
    "vom Stuhl zurückgelehnt",
    "direkt gefolgt von „öffne flay“",
    "direkt gefolgt von „wie spät ist es“",
    "müde",
    "freundlich",
    "zur Seite gedreht",
    "normal",
)


def sentences(language: Language) -> list[str]:
    path = SENTENCES / f"{language.value}.txt"
    return [line.strip() for line in path.read_text("utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class Check:
    ok: bool
    problem: str = ""


def check_take(audio: AudioBuffer, text: str, silence_dbfs: float) -> Check:
    """Is this take usable for training? Level, clipping and speaking rate."""
    if audio.level_dbfs < silence_dbfs:
        return Check(False, "nur Stille – Mikrofon prüfen")
    samples = np.frombuffer(audio.pcm, dtype="<i2")
    if samples.size and np.abs(samples).max() >= 32700:
        return Check(False, "übersteuert – etwas leiser oder weiter weg sprechen")
    loud = np.flatnonzero(np.abs(samples) > 0.02 * 32767)
    seconds = (loud[-1] - loud[0]) / audio.sample_rate if len(loud) > 1 else 0.0
    pace = len(text) / max(seconds, 0.01)
    if pace > PACE[1]:
        return Check(False, "zu kurz – wurde der ganze Satz aufgenommen?")
    if pace < PACE[0]:
        return Check(False, "sehr langsam oder mit langer Pause – bitte flüssig lesen")
    return Check(True)


class Progress:
    """recorded.jsonl next to the WAVs: what is done, so a new start continues."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._path = folder / "recorded.jsonl"

    def done(self) -> set[tuple[str, str]]:
        if not self._path.exists():
            return set()
        records = [json.loads(line) for line in self._path.read_text("utf-8").splitlines()]
        return {(r["language"], r["text"]) for r in records if (self.folder / r["file"]).exists()}

    def seconds(self) -> float:
        if not self._path.exists():
            return 0.0
        return sum(
            float(json.loads(line)["seconds"])
            for line in self._path.read_text("utf-8").splitlines()
        )

    def add(self, language: Language, text: str, file: str, seconds: float) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        record = {"language": language.value, "text": text, "file": file, "seconds": seconds}
        with self._path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
