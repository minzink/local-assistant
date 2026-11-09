import numpy as np

from flay.audio.vad import speech_bounds


def test_speech_bounds() -> None:
    probabilities = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.95, 0.1])

    assert speech_bounds(probabilities, threshold=0.5, min_speech_frames=2) == (2, 6)


def test_too_little_speech_is_rejected() -> None:
    probabilities = np.array([0.1, 0.9, 0.1])

    assert speech_bounds(probabilities, threshold=0.5, min_speech_frames=2) is None
    assert speech_bounds(np.array([]), threshold=0.5, min_speech_frames=1) is None
