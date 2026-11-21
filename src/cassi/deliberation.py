"""Thinking a decision through – briefly or thoroughly – with a short-term memory.

For decisions without hurry (a move in a game, a plan), where a quick answer is often wrong:

- "kurz": the model writes its analysis *before* the answer, so the answer follows from the
  analysis instead of being justified afterwards (a few seconds more).
- "gründlich": first a thinking model thinks freely for up to a time budget (its reasoning is
  cut off when the time is up), then the normal model decides with those thoughts.

Notes are a short-term memory, like a plan a person keeps in their head: with every decision
the model rewrites a few short notes ("Plan: …", "Achtung: …") and gets them back with the
next one. They live for one task (a game, a piece of work); in a file they survive a restart.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cassi.core.messages import Message, Role
from cassi.llm.base import ReasoningChat

Depth = Literal["kurz", "gründlich"]
DEPTHS: tuple[Depth, ...] = ("kurz", "gründlich")
# How long a thinking model may think by default ("gründlich"), in seconds.
THINK_SECONDS = 60.0
# Of long thoughts the deciding model gets the end (where the conclusions are).
THOUGHTS_SHOWN = 6000
ANALYSIS_CHARS = 1500


@dataclass
class Notes:
    """A short-term memory: a few short notes, rewritten by the model with each decision."""

    path: Path | None = None
    limit: int = 6
    length: int = 200  # characters per note
    items: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, limit: int = 6) -> Notes:
        notes = cls(path, limit)
        with contextlib.suppress(OSError, ValueError):
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, list):
                notes.items = [str(item) for item in data][:limit]
        return notes

    def keep(self, notes: Iterable[object]) -> None:
        """The model's new notes replace the old ones (it carries over what still holds)."""
        cleaned: list[str] = []
        for note in notes:
            text = " ".join(str(note).split())[: self.length]
            if text and text not in cleaned:
                cleaned.append(text)
        self.items = cleaned[: self.limit]
        self.save()

    def clear(self) -> None:
        self.items = []
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        with contextlib.suppress(OSError):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), "utf-8")

    def prompt(self) -> str:
        if not self.items:
            return "Deine Notizen (Kurzzeitgedächtnis): noch keine."
        lines = "\n".join(f"- {note}" for note in self.items)
        return f"Deine Notizen von vorher (Kurzzeitgedächtnis):\n{lines}"

    def schema(self) -> dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "string", "maxLength": self.length},
            "maxItems": self.limit,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    answer: dict[str, Any]  # the fields of the caller's schema
    analysis: str = ""
    thoughts: str = ""  # "gründlich": what the thinking model thought
    finished: bool = True  # False: the thinking model was stopped by the time budget
    notes: tuple[str, ...] = ()
    seconds: float = 0.0
    thought_seconds: float = 0.0


def schema_with(answer: dict[str, Any], notes: Notes | None) -> dict[str, Any]:
    """The caller's answer schema with "analysis" first and "notes" last: the model writes
    in this order, so it analyses before it answers."""
    properties: dict[str, Any] = {"analysis": {"type": "string", "maxLength": ANALYSIS_CHARS}}
    properties.update(answer.get("properties", {}))
    if notes is not None:
        properties["notes"] = notes.schema()
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


async def deliberate(
    chat: Any,
    messages: Sequence[Message],
    schema: dict[str, Any],
    *,
    name: str = "answer",
    depth: Depth = "kurz",
    notes: Notes | None = None,
    think_seconds: float = THINK_SECONDS,
    max_tokens: int = 1500,
) -> Decision:
    """Answer ``messages`` in ``schema`` after thinking it through (see the module).

    ``chat`` must answer in JSON (``StructuredChat``); for "gründlich" it should also be a
    ``ReasoningChat`` – otherwise it decides as with "kurz"."""
    begin = time.perf_counter()
    added: list[str] = []
    if notes is not None:
        added.append(notes.prompt())
    thoughts, finished, thought_seconds = "", True, 0.0
    if depth == "gründlich" and isinstance(chat, ReasoningChat):
        asked = [*messages, *_user(added)]
        reasoning = await chat.reason(asked, think_seconds)
        thought_seconds = time.perf_counter() - begin
        finished = reasoning.finished
        thoughts = "\n".join(t for t in (reasoning.thoughts, reasoning.answer) if t)
        if thoughts:
            cut = "" if finished else " (die Zeit war um, mittendrin abgebrochen)"
            added.append(f"Deine Überlegungen dazu{cut}:\n…{thoughts[-THOUGHTS_SHOWN:]}")
    rules = (
        "Antworte mit JSON. Zuerst 'analysis': prüfe knapp, was jetzt wichtig ist, wäge zwei, "
        "drei Möglichkeiten ab – die Antwort folgt daraus."
    )
    if notes is not None:
        rules += (
            f" Zuletzt 'notes': bis zu {notes.limit} kurze Notizen für deine nächsten "
            "Entscheidungen (Plan, Gefahren, Absichten des Gegenübers). Sie ersetzen die "
            "bisherigen – übernimm, was noch gilt, streiche, was erledigt ist."
        )
    added.append(rules)
    answer = dict(
        await chat.complete_json(
            [*messages, *_user(added)], schema_with(schema, notes), name, max_tokens=max_tokens
        )
    )
    analysis = str(answer.pop("analysis", "") or "").strip()
    kept = answer.pop("notes", None)
    if notes is not None and isinstance(kept, list):
        notes.keep(kept)
    return Decision(
        answer,
        analysis,
        thoughts,
        finished,
        tuple(notes.items) if notes is not None else (),
        time.perf_counter() - begin,
        thought_seconds,
    )


def _user(texts: list[str]) -> list[Message]:
    return [Message.from_text(Role.USER, "\n\n".join(texts))] if texts else []
