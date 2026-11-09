"""Conversation messages.

Messages are built from content parts so multimodal models (image, audio input) fit the
same conversation model; backends check ``ModelCapabilities`` before accepting non-text
parts. Tool calls and tool results are messages, too.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from flay.core.language import Language


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class TextPart:
    modality: ClassVar[Modality] = Modality.TEXT
    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    modality: ClassVar[Modality] = Modality.IMAGE
    data: bytes
    mime_type: str


@dataclass(frozen=True, slots=True)
class AudioPart:
    modality: ClassVar[Modality] = Modality.AUDIO
    data: bytes
    mime_type: str


ContentPart = TextPart | ImagePart | AudioPart


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The model asks to run a tool; ``id`` links the result message to the call."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """A tool Flay ran for a reply and what it reported.

    Kept with the reply in the conversation, so later turns know what really happened –
    the reply text alone may have been cut off, and a model only sees words otherwise.
    """

    call: ToolCall
    result: str
    ok: bool = True


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    parts: tuple[ContentPart, ...]
    language: Language | None = None
    # Assistant messages that requested tools.
    tool_calls: tuple[ToolCall, ...] = ()
    # Tool result messages: the call they answer.
    tool_call_id: str | None = None
    # Assistant replies the user cut off: the text is what was said until then.
    interrupted: bool = False
    # Assistant replies: the tools that ran for this reply (stored conversation only).
    actions: tuple[ToolOutcome, ...] = ()

    @classmethod
    def from_text(cls, role: Role, text: str, language: Language | None = None) -> Message:
        return cls(role=role, parts=(TextPart(text),), language=language)

    @property
    def modalities(self) -> frozenset[Modality]:
        return frozenset(part.modality for part in self.parts)

    @property
    def text(self) -> str:
        """All text parts joined; non-text parts are ignored."""
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))
