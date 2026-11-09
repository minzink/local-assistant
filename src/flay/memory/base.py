"""Memory interface used by the assistant.

The assistant only asks for a ``MemoryContext`` before answering and reports each finished
exchange; how messages are stored, summarized or turned into long-term facts is up to the
implementation.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from flay.core.messages import Message


@dataclass(frozen=True, slots=True)
class Fact:
    """One memory: a fact ("Lea hat am 3. März Geburtstag.") or an episode – the summary of
    a finished conversation, so later ones can refer back to it."""

    id: int
    text: str
    kind: str = "fact"
    # When it was written into memory (not when the conversation took place).
    created_at: float = 0.0
    importance: int = 2  # IMPORTANT, NORMAL or MINOR
    # Examples: the action the request meant, as JSON {"tool", "arguments", "confirmed"}.
    data: str | None = None

    @property
    def important(self) -> bool:
        return self.importance >= IMPORTANT


# Important memories are lasting rules and instructions ("In Zukunft gib Claude keinen
# Befehl mehr, etwas zu committen") and what the user calls important; they apply always.
# Minor ones are details of past conversations ("die README wurde nicht angepasst").
IMPORTANT, NORMAL, MINOR = 3, 2, 1

# Kinds of memories. Examples are requests as they were heard and the action they meant;
# they teach the model what the user's words mean, they are not knowledge to recall.
# Mistakes are requests whose action the user rejected ("nein, falsch"); they are only kept
# as training data (``flay train-export``), never shown to the model. Verbs are words the
# speech recognition makes of a command verb ("She" for "Schließe", see flay.verbs).
FACT, EPISODE, EXAMPLE, MISTAKE, VERB = "fact", "episode", "example", "mistake", "verb"
KNOWLEDGE = (FACT, EPISODE)
IMPORTANCE_NAMES = {IMPORTANT: "wichtig", NORMAL: "normal", MINOR: "nebensächlich"}

_RULE = re.compile(
    r"\b(?:in zukunft|künftig|zukünftig|ab jetzt|ab sofort|von nun an|nie|niemals|immer|"
    r"grundsätzlich|wichtig|from now on|in future|in the future|never|always|important)\b",
    re.IGNORECASE,
)


def sounds_important(text: str) -> bool:
    """Wording of a lasting rule or of something stressed as important."""
    return bool(_RULE.search(text))


@dataclass(frozen=True, slots=True)
class MemoryContext:
    # Latest messages of the current session, oldest first.
    recent: tuple[Message, ...] = ()
    # Summary of older messages of the current session.
    summary: str | None = None
    # Long-term knowledge about the user, across sessions (without recall: all of it).
    facts: tuple[Fact, ...] = ()
    # Important memories – rules the assistant always follows; always in the prompt.
    important: tuple[Fact, ...] = ()


class Memory(Protocol):
    def context(self) -> MemoryContext: ...

    def record(self, user: Message, assistant: Message) -> None:
        """Store a completed exchange."""
        ...

    async def recall(self, query: str) -> tuple[Fact, ...]:
        """Memories that may help with ``query`` (RAG); empty if there are none."""
        ...

    def turn_started(self) -> None:
        """The user is speaking: pause background work so it does not compete for the LLM."""
        ...

    def turn_finished(self) -> None:
        """The answer is done: background work may run once the user stays quiet."""
        ...

    async def close(self) -> None: ...


class FactMemory(Protocol):
    """Long-term facts the user manages explicitly ("remember …", "forget …")."""

    def facts(self) -> list[Fact]: ...

    def remember(self, text: str, important: bool = False) -> bool:
        """Store a fact; False if it is already known."""
        ...

    def forget(self, ids: list[int]) -> int:
        """Delete facts; returns how many were deleted."""
        ...


class ShortTermMemory:
    """Keeps the last messages in RAM only; used when persistent memory is disabled."""

    def __init__(self, max_messages: int = 12) -> None:
        self._messages: deque[Message] = deque(maxlen=max_messages)

    def context(self) -> MemoryContext:
        return MemoryContext(recent=tuple(self._messages))

    def record(self, user: Message, assistant: Message) -> None:
        self._messages.extend((user, assistant))

    async def recall(self, query: str) -> tuple[Fact, ...]:
        return ()

    def turn_started(self) -> None:
        pass

    def turn_finished(self) -> None:
        pass

    async def close(self) -> None:
        pass
