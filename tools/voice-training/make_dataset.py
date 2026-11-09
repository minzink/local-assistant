"""Speak the sentence lists with Flay's Qwen3-TTS voice; keep what Whisper understands.

With --recorded <dir> the sentences come from the user's own recordings instead
(``flay voice dataset``): the same checks, trimming and conversion, no Qwen3-TTS.

Writes the Piper training set: <out>/wavs/*.wav (22,050 Hz mono) and <out>/metadata.csv
(file|text|phoneme ids). The ids come from espeak via the Piper Flay runs – German with "de",
English with "en-us" – so one model learns the voice for both languages.
Resumable: sentences already decided are skipped. Downloads the models it needs.

Usage (repo root): uv run python tools/voice-training/make_dataset.py [--voice ramona]
                   uv run python tools/voice-training/make_dataset.py --recorded <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import itertools
import json
import re
import time
import wave
from pathlib import Path

import numpy as np
from piper.phoneme_ids import phonemes_to_ids
from piper.phonemize_espeak import EspeakPhonemizer

import flay
from flay.audio.base import AudioBuffer
from flay.config import VoiceSettings, load_settings
from flay.core.language import Language
from flay.core.platform import detect_hardware
from flay.stt.factory import create_stt
from flay.tts.factory import create_tts
from flay.tts.voice import resample

HERE = Path(__file__).parent
SENTENCES = Path(flay.__file__).parent / "data" / "sentences"
RATE = 22_050
TTS_RATE = 24_000
MAX_WER = 0.1
ESPEAK = {Language.DE: "de", Language.EN: "en-us"}
_WORD = re.compile(r"[\wäöüß']+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'").replace("-", " "))


def word_error(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return 1 - matched / max(len(ref), len(hyp), 1)


def trim(samples: np.ndarray, rate: int, pad: float = 0.12) -> np.ndarray:
    loud = np.flatnonzero(np.abs(samples) > 0.02)
    if not len(loud):
        return samples[:0]
    margin = int(pad * rate)
    return samples[max(0, loud[0] - margin) : loud[-1] + margin]


def own_recordings(folder: Path) -> dict[tuple[str, str], Path]:
    """(language, sentence) → WAV of the user's recordings (recorded.jsonl from flay)."""
    found = {}
    for line in (folder / "recorded.jsonl").read_text("utf-8").splitlines():
        record = json.loads(line)
        if (folder / record["file"]).exists():
            found[(record["language"], record["text"])] = folder / record["file"]
    print(f"Eigene Aufnahmen: {len(found)}")
    return found


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as file:
        rate = file.getframerate()
        data = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2")
    return data.astype(np.float32) / 32767, rate


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="ramona")
    parser.add_argument("--out", type=Path, default=HERE / "work" / "dataset")
    parser.add_argument("--limit", type=int, help="only the first N sentences per language")
    parser.add_argument(
        "--recorded", type=Path, help="own recordings (flay voice dataset) instead of Qwen3-TTS"
    )
    args = parser.parse_args()
    out: Path = args.out
    if args.recorded and out == HERE / "work" / "dataset":
        out = HERE / "work" / "dataset-own"  # not mixed with a synthetic set
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    decided_path = out / "decided.jsonl"
    decided = (
        [json.loads(line) for line in decided_path.read_text("utf-8").splitlines()]
        if (decided_path.exists())
        else []
    )
    done = {(r["language"], r["text"]) for r in decided}

    settings = load_settings()
    voice = settings.tts.voice.model_copy(update={"name": args.voice})
    assert isinstance(voice, VoiceSettings)
    settings = settings.model_copy(update={"tts": settings.tts.model_copy(update={"voice": voice})})
    hardware = detect_hardware()
    recordings = own_recordings(args.recorded) if args.recorded else None
    tts = None if recordings is not None else create_tts(settings, hardware, backend="qwen3")
    stt = create_stt(settings, hardware)
    print("Lade Modelle (beim ersten Mal mit Download) ...", flush=True)
    stt.download()
    if tts is not None:
        tts.download()
        tts.load()
    await stt.warm_up()

    for language in Language:
        sentences = (SENTENCES / f"{language}.txt").read_text("utf-8").splitlines()
        sentences = sentences[: args.limit]
        if recordings is not None:
            sentences = [s for s in sentences if (str(language), s) in recordings]
        todo = [s for s in sentences if s and (str(language), s) not in done]
        start = time.monotonic()
        for index, text in enumerate(todo, 1):
            if recordings is not None:
                samples, rate = read_wav(recordings[(str(language), text)])
                samples = trim(resample(samples, rate, TTS_RATE), TTS_RATE)
            else:
                assert tts is not None
                pcm = b"".join([c.pcm async for c in tts.synthesize(text, language)])
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767
                samples = trim(samples, TTS_RATE)
            seconds = len(samples) / TTS_RATE
            heard, error = "", 1.0
            if seconds > 0.5:
                audio16 = (resample(samples, TTS_RATE, 16_000) * 32767).astype("<i2").tobytes()
                heard = (await stt.transcribe(AudioBuffer(audio16, 16_000), language)).text
                error = word_error(text, heard)
            # Whisper writes "neun Uhr" as "9 Uhr": a little more slack when it used digits.
            limit = MAX_WER + 0.1 if any(c.isdigit() for c in heard) else MAX_WER
            pace = len(text) / max(seconds, 0.01)
            record: dict[str, object] = {
                "language": str(language), "text": text, "wer": round(error, 3),
                "seconds": round(seconds, 2),
                "ok": error <= limit and 7 <= pace <= 22 and seconds <= 15,
            }  # fmt: skip
            if record["ok"]:
                name = f"{language}_{len(decided):05d}.wav"
                with wave.open(str(out / "wavs" / name), "wb") as file:
                    file.setnchannels(1)
                    file.setsampwidth(2)
                    file.setframerate(RATE)
                    data = np.clip(resample(samples, TTS_RATE, RATE), -1, 1)
                    file.writeframes((data * 32767).astype("<i2").tobytes())
                record["file"] = name
            else:
                record["heard"] = heard
            decided.append(record)
            with decided_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(todo):
                elapsed = time.monotonic() - start
                kept = sum(1 for r in decided if r["ok"] and r["language"] == str(language))
                print(f"{language}: {index}/{len(todo)}, behalten {kept} ({elapsed:.0f} s)",
                      flush=True)  # fmt: skip

    phonemizer = EspeakPhonemizer()
    rows = []
    for record in decided:
        if record["ok"]:
            text = str(record["text"])
            phonemes = phonemizer.phonemize(ESPEAK[Language(record["language"])], text)
            ids = itertools.chain(*(phonemes_to_ids(p) for p in phonemes))
            rows.append(f"{record['file']}|{text}|{' '.join(map(str, ids))}")
    (out / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    minutes = sum(float(str(r["seconds"])) for r in decided if r["ok"]) / 60
    print(f"Datensatz: {len(rows)} Sätze, {minutes:.0f} min -> {out}")


asyncio.run(main())
