import numpy as np

from flay.input.continuous import FRAME_MS, Segmenter, SegmenterConfig, frames

SILENT, SPEECH = 0.05, 0.95
CONFIG = SegmenterConfig(start_ms=96, end_silence_ms=320, min_speech_ms=128, pre_roll_ms=96)


def frame(value: float) -> np.ndarray:
    return np.full(512, value, dtype=np.float32)


def run(segmenter: Segmenter, probabilities: list[float], speaking: bool = False) -> list:
    """Feed frames whose sample value equals their probability; returns the events."""
    return [segmenter.feed(frame(p), p, speaking) for p in probabilities]


def test_frames_conversion() -> None:
    assert FRAME_MS == 32
    assert frames(96) == 3


def test_utterance_with_pre_roll_ends_after_silence() -> None:
    segmenter = Segmenter(CONFIG)
    events = run(segmenter, [SILENT] * 5 + [SPEECH] * 6 + [SILENT] * 10)

    started = [i for i, e in enumerate(events) if e.started]
    utterances = [e.utterance for e in events if e.utterance is not None]
    assert started == [7]  # third speech frame
    assert len(utterances) == 1
    # Pre-roll (3 frames incl. the first speech frames) + rest of speech + 10 silent frames.
    audio = utterances[0]
    assert audio[0] == np.float32(SILENT) or audio[0] == np.float32(SPEECH)
    assert len(audio) == 512 * (3 + 3 + 10)


def test_short_noise_is_not_an_utterance() -> None:
    segmenter = Segmenter(CONFIG)

    events = run(segmenter, [SPEECH] * 2 + [SILENT] * 20)

    assert not any(e.started or e.utterance is not None for e in events)


def test_too_little_speech_after_start_is_dropped() -> None:
    segmenter = Segmenter(SegmenterConfig(start_ms=96, end_silence_ms=96, min_speech_ms=320))

    events = run(segmenter, [SPEECH] * 4 + [SILENT] * 5)

    assert any(e.started for e in events)
    assert all(e.utterance is None for e in events)


def test_long_speech_is_cut_at_max_length() -> None:
    segmenter = Segmenter(SegmenterConfig(start_ms=32, max_utterance_ms=320, min_speech_ms=32))

    events = run(segmenter, [SPEECH] * 30)

    assert sum(e.utterance is not None for e in events) >= 2


def test_barge_in_needs_clear_and_longer_speech_while_assistant_speaks() -> None:
    config = SegmenterConfig(barge_in=True, barge_in_threshold=0.85, barge_in_ms=256)
    segmenter = Segmenter(config)

    weak = run(segmenter, [0.7] * 20, speaking=True)  # e.g. Flay's own voice, quieter
    strong = run(segmenter, [0.95] * 8, speaking=True)

    assert not any(e.started for e in weak)
    assert strong[-1].started


def test_without_barge_in_speech_is_ignored_while_assistant_speaks() -> None:
    segmenter = Segmenter(SegmenterConfig(barge_in=False))

    events = run(segmenter, [SPEECH] * 30, speaking=True)

    assert not any(e.started for e in events)


def test_tentative_utterance_at_first_pause_is_prefix_of_final() -> None:
    segmenter = Segmenter(
        SegmenterConfig(start_ms=96, end_silence_ms=320, speculate_ms=96, min_speech_ms=64)
    )
    events = run(segmenter, [SPEECH] * 6 + [SILENT] * 3 + [SPEECH] * 3 + [SILENT] * 12)

    tentatives = [e.tentative for e in events if e.tentative is not None]
    final = next(e.utterance for e in events if e.utterance is not None)
    # One tentative per pause, the last one is exactly the start of the final utterance.
    assert len(tentatives) == 2
    assert np.array_equal(final[: len(tentatives[-1])], tentatives[-1])
    assert len(final) > len(tentatives[-1])


def test_speculation_can_be_switched_off() -> None:
    segmenter = Segmenter(SegmenterConfig(start_ms=96, end_silence_ms=320, speculate_ms=0))

    events = run(segmenter, [SPEECH] * 6 + [SILENT] * 12)

    assert all(e.tentative is None for e in events)
