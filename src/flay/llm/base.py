from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from flay.core.messages import Message, Modality, ToolCall


class UnsupportedModalityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """What a configured model can consume. Text-only in version 1."""

    input_modalities: frozenset[Modality] = frozenset({Modality.TEXT})

    def ensure_supported(self, messages: Sequence[Message]) -> None:
        for message in messages:
            unsupported = message.modalities - self.input_modalities
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise UnsupportedModalityError(f"Model does not accept: {names}")


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model may call; ``parameters`` is a JSON schema of its arguments."""

    name: str
    description: str
    parameters: dict[str, Any]


LLMEvent = TextDelta | ToolCall


@dataclass(frozen=True, slots=True)
class ThinkingOptions:
    """How a request with thinking differs from a normal one."""

    model: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 2000


class ChatModel(Protocol):
    @property
    def capabilities(self) -> ModelCapabilities: ...

    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = (), think: bool = False
    ) -> AsyncIterator[LLMEvent]:
        """Stream the reply as text deltas, followed by any tool calls the model makes.

        ``think``: let the model think first, if it is set up for that; the thinking is
        never part of the text deltas. Cancelling the consuming task must abort
        generation.
        """
        ...


@runtime_checkable
class StructuredChat(Protocol):
    """Optional: a model that can answer in a fixed JSON structure.

    The server enforces the schema while generating (OpenAI ``response_format`` with a JSON
    schema; Ollama and llama.cpp support it), so e.g. a project can only be one of the
    names listed in the schema.
    """

    async def complete_json(
        self, messages: Sequence[Message], schema: dict[str, Any], name: str = "answer"
    ) -> dict[str, Any]: ...
