"""`flay train-wake`: train the "Hey Flay" detector from synthetic and recorded speech.

No speech data with a non-commercial license is used. Positives are "Hey Flay" spoken by the
free Piper voices (Kerstin, Kristin, Thorsten, Ramona) with varied speed, pitch, loudness,
room and noise – plus the user's own recordings, if any (``<data_dir>/wake/*.wav``, from
``flay voice wake``). Negatives are similar sounds ("Hey Fly", "Okay"), everyday sentences
from ``data/wake_negative.txt``, noise, and "Hey" alone (the start of the wake word).

The classifier is small (1536 → 128 → 1) and trained with numpy in minutes on a CPU. One
voice is held out to report how well it works on a voice it never heard. Five synthetic
voices cannot teach it all the speech it will hear (on a held-out voice: 97 % of "Hey Flay"
heard, but ~37 false wake-ups per hour of other speech at 0.7), so the transcript has the
final word: without "Hey Flay" at its start, an utterance is dropped.
"""

from __future__ import annotations

import logging
import random
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flay.tts.shaping import pitch_shift
from flay.tts.voice import resample
from flay.wake import SAMPLE_RATE, WINDOW, Classifier, Features, windows

log = logging.getLogger(__name__)

PHRASES = ("Hey Flay", "Hey Flay!", "Hey, Flay.", "Hey Flay,", "Hey, Flay!")
# Free Piper voices (CC0, public domain, CC BY 4.0, Flay's own Ramona); others may have a
# license that forbids using them for this. The multi-speaker ones bring the variety that
# makes the detector hear voices it was not trained on: LibriTTS-R (904 speakers), MLS
# German (236), VCTK (109).
FREE_VOICES = (
    "kerstin", "kristin", "thorsten", "ramona", "libritts_r", "mls", "vctk",
)  # fmt: skip
# Multi-speaker voices give this many times the examples of a single voice.
MULTI_SPEAKER_SHARE = 4
NEGATIVES = Path(__file__).parent / "data" / "wake_negative.txt"
# When a window counts as having heard the wake word: it ends this soon after the phrase.
POSITIVE_AFTER = 0.32
# Embedding i ends at (8 * i + 76) mel frames of 10 ms; a window ends at its last one.
FIRST_END = (8 * (WINDOW - 1) + 76) * 0.01
STEP = 0.08


@dataclass
class Clip:
    audio: np.ndarray
    voice: str
    # Start and end of the wake word in seconds; None for negatives.
    phrase: tuple[float, float] | None = None
    text: str = ""


def free_voices(*folders: Path) -> list[Path]:
    found: dict[str, Path] = {}
    for folder in folders:
        for path in sorted(folder.glob("*.onnx")) if folder.is_dir() else []:
            if any(name in path.name for name in FREE_VOICES):
                found.setdefault(path.stem, path)
    return list(found.values())


def synthesize(
    voice: object, text: str, speed: float, noise: float, speaker: int | None = None
) -> tuple[np.ndarray, int]:
    from piper import SynthesisConfig

    config = SynthesisConfig(
        speaker_id=speaker, length_scale=1 / speed, noise_scale=noise, noise_w_scale=noise
    )
    parts = []
    rate = SAMPLE_RATE
    for chunk in voice.synthesize(text, config):  # type: ignore[attr-defined]
        rate = chunk.sample_rate
        parts.append(chunk.audio_int16_array.astype(np.float32) / 32768.0)
    audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    return audio, rate


def noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """White, pink or brown noise, scaled to 1."""
    white = rng.standard_normal(length).astype(np.float32)
    kind = rng.integers(3)
    if kind == 1:  # pink: 1/f by filtering the spectrum
        spectrum = np.fft.rfft(white)
        spectrum /= np.sqrt(np.arange(1, len(spectrum) + 1))
        white = np.fft.irfft(spectrum, n=length).astype(np.float32)
    elif kind == 2:  # brown
        white = np.cumsum(white).astype(np.float32)
        white -= np.linspace(white[0], white[-1], length, dtype=np.float32)
    return np.asarray(white / (np.abs(white).max() + 1e-9), dtype=np.float32)


def augment(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Room, background noise and loudness of a real microphone, roughly."""
    if rng.random() < 0.5:  # reverb: decaying noise as impulse response
        length = int(rng.uniform(0.1, 0.5) * SAMPLE_RATE)
        response = rng.standard_normal(length) * np.exp(-np.linspace(0, rng.uniform(4, 9), length))
        response[0] = 1.0
        audio = np.convolve(audio, response.astype(np.float32))[: len(audio)]
        audio /= np.abs(audio).max() + 1e-9
    if rng.random() < 0.8:
        snr = rng.uniform(5, 30)
        level = np.sqrt(np.mean(audio**2)) / (10 ** (snr / 20))
        audio = audio + noise(len(audio), rng) * level * 3
    audio = audio / (np.abs(audio).max() + 1e-9) * rng.uniform(0.1, 0.9)
    return audio.astype(np.float32)


def speakers(voice: object) -> int:
    return int(getattr(getattr(voice, "config", None), "num_speakers", 1) or 1)


def _speech(voice: object, text: str, rng: np.random.Generator) -> np.ndarray:
    count = speakers(voice)
    speaker = int(rng.integers(count)) if count > 1 else None
    audio, rate = synthesize(voice, text, rng.uniform(0.8, 1.3), rng.uniform(0.4, 0.9), speaker)
    if rng.random() < 0.6:
        audio = pitch_shift(audio, float(rng.uniform(-3, 3)), rate)
    return resample(audio, rate, SAMPLE_RATE)


# Openers put before everyday sentences: the start that must not count as "Hey Flay".
_OPENERS = (
    "Hey,", "Hey Leute,", "Hey du,", "Okay,", "Hey guys,", "Hey there,", "Hey Kay,",
    "Hey Fred,", "Hey Lea,", "Hey Ben,", "Hey Paul,", "Hey team,", "Ey,", "Hallo,",
)  # fmt: skip

# What follows the wake word half of the time: "Hey Flay, öffne flay".
COMMANDS = (
    "öffne das Projekt flay", "wie spät ist es", "sag Claude, er soll die Tests reparieren",
    "schließe moewe", "lies mir die README vor", "open the project", "what time is it",
    "tell Claude to fix the build", "merk dir das", "scroll mal runter",
)  # fmt: skip


def positive(voice: object, name: str, rng: np.random.Generator) -> Clip:
    phrase = _speech(voice, str(rng.choice(PHRASES)), rng)
    before = np.zeros(int(rng.uniform(0.3, 1.5) * SAMPLE_RATE), np.float32)
    after = np.zeros(int(rng.uniform(0.1, 0.8) * SAMPLE_RATE), np.float32)
    if rng.random() < 0.5:  # the command right after it, as usually spoken
        gap = np.zeros(int(rng.uniform(0.0, 0.3) * SAMPLE_RATE), np.float32)
        after = np.concatenate([gap, _speech(voice, str(rng.choice(COMMANDS)), rng)])
    audio = augment(np.concatenate([before, phrase, after]), rng)
    start = len(before) / SAMPLE_RATE
    return Clip(audio, name, (start, start + len(phrase) / SAMPLE_RATE))


def recorded(folder: Path) -> Iterator[Clip]:
    """The user's own "Hey Flay" recordings (16 kHz mono WAV, wake word anywhere)."""
    for path in sorted(folder.glob("*.wav")) if folder.is_dir() else []:
        with wave.open(str(path)) as file:
            rate = file.getframerate()
            data = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2")
        audio = resample(data.astype(np.float32) / 32768.0, rate, SAMPLE_RATE)
        # Speech between the first and last loud part.
        loud = np.flatnonzero(np.abs(audio) > np.abs(audio).max() * 0.1)
        if len(loud) == 0:
            continue
        pad = np.zeros(SAMPLE_RATE, np.float32)
        audio = np.concatenate([pad, audio[loud[0] : loud[-1]], pad[: SAMPLE_RATE // 4]])
        yield Clip(audio, "own", (1.0, 1.0 + (loud[-1] - loud[0]) / SAMPLE_RATE))


def labelled(features: Features, clip: Clip) -> tuple[np.ndarray, np.ndarray]:
    """The clip's windows and their labels (1: the wake word just ended in it)."""
    x = windows(features.embeddings(clip.audio))
    ends = FIRST_END + STEP * np.arange(len(x))
    if clip.phrase is None:
        return x, np.zeros(len(x), np.float32)
    start, end = clip.phrase
    labels = np.full(len(x), -1.0, np.float32)  # in between: left out
    labels[(ends >= end) & (ends <= end + POSITIVE_AFTER)] = 1.0
    # Only the start heard ("Hey …"): that is not the wake word yet.
    labels[ends <= start + (end - start) * 0.55] = 0.0
    keep = labels >= 0
    return x[keep], labels[keep]


def train(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int = 128,
    epochs: int = 40,
    seed: int = 0,
) -> Classifier:
    """A two-layer network with Adam and class weights (positives are rare)."""
    rng = np.random.default_rng(seed)
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    x = (x - mean) / std
    weights = {
        "w1": rng.standard_normal((x.shape[1], hidden)).astype(np.float32)
        * np.sqrt(2 / x.shape[1]),
        "b1": np.zeros(hidden, np.float32),
        "w2": rng.standard_normal((hidden, 1)).astype(np.float32) * np.sqrt(1 / hidden),
        "b2": np.zeros(1, np.float32),
    }
    moments = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in weights.items()}
    positives = max(float(y.sum()), 1.0)
    pos_weight = (len(y) - positives) / positives
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for batch in np.array_split(order, max(1, len(x) // 512)):
            xb, yb = x[batch], y[batch]
            h = np.maximum(xb @ weights["w1"] + weights["b1"], 0)
            p = 1 / (1 + np.exp(-(h @ weights["w2"] + weights["b2"]).reshape(-1)))
            weight = np.where(yb > 0.5, pos_weight, 1.0)
            grad_logit = ((p - yb) * weight / len(yb))[:, None]
            grads = {
                "w2": h.T @ grad_logit,
                "b2": grad_logit.sum(axis=0),
            }
            grad_h = grad_logit @ weights["w2"].T * (h > 0)
            grads["w1"] = xb.T @ grad_h + 1e-4 * weights["w1"]
            grads["b1"] = grad_h.sum(axis=0)
            step += 1
            rate = 1e-3 * (0.5 if epoch > epochs * 0.7 else 1.0)
            for key, grad in grads.items():
                m, v = moments[key]
                m[:] = 0.9 * m + 0.1 * grad
                v[:] = 0.999 * v + 0.001 * grad**2
                m_hat = m / (1 - 0.9**step)
                v_hat = v / (1 - 0.999**step)
                weights[key] -= (rate * m_hat / (np.sqrt(v_hat) + 1e-8)).astype(np.float32)
    return Classifier({**weights, "mean": mean.astype(np.float32), "std": std.astype(np.float32)})


@dataclass
class Report:
    voice: str
    recall: float
    false_accepts: int
    negatives: int
    negative_minutes: float
    threshold: float


def evaluate(
    classifier: Classifier, features: Features, clips: list[Clip], threshold: float
) -> tuple[float, int, int, float]:
    """Share of wake words heard; negatives (whole clips) that fired; their length."""
    heard, total, fired, negatives, seconds = 0, 0, 0, 0, 0.0
    for clip in clips:
        score = float(classifier(windows(features.embeddings(clip.audio))).max())
        if clip.phrase is not None:
            total += 1
            heard += score >= threshold
        else:
            negatives += 1
            fired += score >= threshold
            seconds += len(clip.audio) / SAMPLE_RATE
    return heard / max(total, 1), fired, negatives, seconds / 60


def build(
    voices: list[Path],
    models: Path,
    own: Path,
    repeats: int = 120,
    seed: int = 0,
    threshold: float = 0.7,
) -> tuple[Classifier, Report]:
    """Generate the data, check with one voice held out, then train on all voices."""
    from piper import PiperVoice

    rng = np.random.default_rng(seed)
    random.seed(seed)
    features = Features(models)
    texts = [
        line.strip()
        for line in NEGATIVES.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    clips: list[Clip] = []
    for path in voices:
        voice = PiperVoice.load(str(path))
        name = path.stem
        share = MULTI_SPEAKER_SHARE if speakers(voice) > 1 else 1
        log.info("Erzeuge Beispiele mit %s (%d Sprecher) …", name, speakers(voice))
        clips += [positive(voice, name, rng) for _ in range(repeats * share)]
        for _ in range(2 if share > 1 else 1):  # many speakers: each sentence twice
            for text in texts:
                if rng.random() < 0.35:
                    # "Hey Leute, …", "Okay, …": unseen voices fired on any "Hey" before.
                    text = f"{rng.choice(_OPENERS)} {text[0].lower()}{text[1:]}"
                clips.append(Clip(augment(_speech(voice, str(text), rng), rng), name, text=text))
    clips += list(recorded(own))
    for _ in range(max(40, len(clips) // 10)):  # noise without speech
        length = int(rng.uniform(1.5, 4.0) * SAMPLE_RATE)
        clips.append(Clip(noise(length, rng) * rng.uniform(0.01, 0.3), "noise"))

    data = [(clip, *labelled(features, clip)) for clip in clips]
    held = voices[-1].stem if len(voices) > 1 else ""
    train_part = [(x, y) for clip, x, y in data if clip.voice != held]
    x = np.concatenate([x for x, _ in train_part])
    y = np.concatenate([y for _, y in train_part])
    log.info("Prüfe mit zurückgehaltener Stimme %s (%d Fenster, %d positiv)", held, len(y), y.sum())
    check = train(x, y, seed=seed)
    recall, fired, negatives, minutes = evaluate(
        check, features, [clip for clip, _, _ in data if clip.voice == held], threshold
    )
    report = Report(held, recall, fired, negatives, minutes, threshold)

    x = np.concatenate([x for _, x, _ in data])
    y = np.concatenate([y for _, _, y in data])
    return train(x, y, seed=seed), report
