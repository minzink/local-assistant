"""Shape a reference voice before cloning: pitch in semitones, speaking rate as a factor.

The cloned voice follows its reference recording, so changing the reference once changes
every sentence spoken with it – at no cost per sentence. Both use WSOLA (waveform similarity
overlap-add): the recording is cut into overlapping pieces that are laid out again more
densely or sparsely, each shifted slightly so its waveform continues the previous one.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

FRAME_SECONDS = 0.04
# How far a piece may move to line up with its predecessor.
TOLERANCE_SECONDS = 0.012


def time_stretch(samples: np.ndarray, rate: float, sample_rate: int) -> np.ndarray:
    """Play ``samples`` ``rate`` times as fast without changing the pitch."""
    if rate == 1.0 or samples.size == 0:
        return samples
    frame = int(FRAME_SECONDS * sample_rate)
    hop = frame // 2
    tolerance = int(TOLERANCE_SECONDS * sample_rate)
    window = np.hanning(frame)
    length = int(len(samples) / rate)
    source = np.pad(samples.astype(np.float64), (tolerance, frame + hop + tolerance))
    out = np.zeros(length + frame)
    weight = np.zeros(length + frame)
    previous: int | None = None
    for start in range(0, length, hop):
        nominal = int(start * rate) + tolerance
        if previous is None:
            best = nominal
        else:
            # The piece that continues the previous one most smoothly.
            natural = source[previous + hop : previous + hop + frame]
            region = source[nominal - tolerance : nominal + tolerance + frame]
            best = nominal - tolerance + int(np.argmax(np.correlate(region, natural, "valid")))
        out[start : start + frame] += source[best : best + frame] * window
        weight[start : start + frame] += window
        previous = best
    weight[weight < 1e-3] = 1.0
    return (out / weight)[:length].astype(samples.dtype)


def pitch_shift(samples: np.ndarray, semitones: float, sample_rate: int) -> np.ndarray:
    """Raise (or lower) the pitch by ``semitones``; the duration stays the same."""
    if semitones == 0 or samples.size == 0:
        return samples
    factor = Fraction(2 ** (semitones / 12)).limit_denominator(200)
    # Stretch to factor × the length, then resample back: the pitch rises by the factor.
    stretched = time_stretch(samples, 1 / float(factor), sample_rate)
    shifted = resample_poly(stretched, factor.denominator, factor.numerator)
    return np.asarray(shifted, dtype=samples.dtype)


def shape(samples: np.ndarray, sample_rate: int, pitch: float, speed: float) -> np.ndarray:
    return time_stretch(pitch_shift(samples, pitch, sample_rate), speed, sample_rate)
