from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Protocol

from flay.audio.base import AudioBuffer


class InputMode(StrEnum):
    PUSH_TO_TALK = "push_to_talk"
    # Always listening; utterances are segmented by VAD.
    CONTINUOUS = "continuous"


class UtteranceSource(Protocol):
    """Yields complete user utterances.

    Push-to-talk records while the key is held; a later continuous source segments the
    microphone stream with VAD. The orchestrator only depends on this interface, so
    switching modes does not touch the rest of the pipeline.
    """

    def utterances(self) -> AsyncIterator[AudioBuffer]: ...
