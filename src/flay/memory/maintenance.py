"""LLM tasks that maintain memory: summarizing old messages and extracting facts."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from flay.core.messages import Message, Role
from flay.llm.base import ChatModel, TextDelta
from flay.memory.base import IMPORTANT, MINOR, NORMAL, Fact

log = logging.getLogger(__name__)

SUMMARY_INSTRUCTIONS = (
    "You summarize the earlier part of a conversation between a user and their voice "
    "assistant, so the assistant can continue it. Write at most four sentences in the "
    "language of the conversation. Keep names, numbers, decisions and open requests. "
    "Reply with the summary only."
)

EPISODE_INSTRUCTIONS = (
    "You write the long-term memory of a finished conversation between a software "
    "developer (the user) and their voice assistant. In one to three sentences, in the "
    'language of the conversation and in the third person ("Der Nutzer …"), record what '
    "the user wanted, what was decided or learned, and how it ended. Name projects, people "
    'and dates exactly. Something the assistant says it did only counts if an "Action:" '
    "line confirms it; otherwise write that it was asked for. Leave out greetings and small "
    "talk. Reply with the summary only."
)

FACT_INSTRUCTIONS = """\
You maintain long-term memory for a voice assistant of a software developer.
Given the known facts and new conversation lines, decide what to change.
Add information worth knowing in later conversations: about the user (name, family, \
preferences, appointments, plans), their work and projects (what a project is, its \
technology, decisions made, open tasks), and anything they ask to remember. Never add \
questions, small talk, general knowledge, one-off commands ("open project X") or anything \
the assistant said about itself. Turn relative dates into absolute ones using the date of \
the conversation ("am Freitag" → "am Freitag, 2026-10-30").
Remove facts the user asks to forget or that the new lines contradict.
Every request that starts with "remember" / "merk dir" must become a fact.
Write each new fact as one short sentence in the language the user used.
Rate each new fact's importance:
- "high": a lasting rule or instruction for the assistant ("in future never …", \
"always …") or something the user says is important.
- "normal": lasting knowledge (names, preferences, appointments, how a project works).
- "low": a one-off outcome or passing state ("today the README was not changed").
Reply with JSON only, in this form: \
{"add": [{"text": "...", "importance": "normal"}], "remove": [1, 2]}

Example:
Known facts:
3: Der Nutzer trinkt gern Kaffee.
New conversation lines:
User: Ich bin Anna. Merk dir, dass mein Termin beim Zahnarzt am Freitag ist.
Assistant: Alles klar, Anna!
User: Ich trinke übrigens keinen Kaffee mehr.
Reply:
{"add": [{"text": "Der Nutzer heißt Anna.", "importance": "normal"}, \
{"text": "Annas Zahnarzttermin ist am Freitag.", "importance": "normal"}], "remove": [3]}

Example:
Known facts:
(none)
New conversation lines:
User: Sag Claude in Zukunft nie mehr, dass er etwas committen soll. Die Tests liefen \
heute übrigens nicht durch.
Reply:
{"add": [{"text": "Flay soll Claude nie anweisen, etwas zu committen.", \
"importance": "high"}, {"text": "Die Tests liefen am 2026-10-30 nicht durch.", \
"importance": "low"}], "remove": []}"""


LEVELS = {"high": IMPORTANT, "normal": NORMAL, "low": MINOR}


@dataclass(frozen=True, slots=True)
class FactChanges:
    # New facts with their importance (memory.base: IMPORTANT, NORMAL, MINOR).
    add: tuple[tuple[str, int], ...] = ()
    remove: tuple[int, ...] = ()


async def complete(llm: ChatModel, messages: Sequence[Message]) -> str:
    parts = [event.text async for event in llm.stream(messages) if isinstance(event, TextDelta)]
    return "".join(parts).strip()


def transcript(messages: Sequence[Message]) -> str:
    labels = {Role.USER: "User", Role.ASSISTANT: "Assistant", Role.SYSTEM: "System"}
    lines = []
    for m in messages:
        for action in m.actions:
            lines.append(f"Action: {action.call.name} → {action.result[:120]}")
        cut = " [interrupted by the user]" if m.interrupted else ""
        lines.append(f"{labels[m.role]}: {m.text}{cut}")
    return "\n".join(lines)


async def summarize(llm: ChatModel, previous: str | None, messages: Sequence[Message]) -> str:
    content = f"Conversation:\n{transcript(messages)}"
    if previous:
        content = f"Summary so far:\n{previous}\n\n{content}"
    return await complete(
        llm,
        [
            Message.from_text(Role.SYSTEM, SUMMARY_INSTRUCTIONS),
            Message.from_text(Role.USER, content),
        ],
    )


async def summarize_episode(llm: ChatModel, messages: Sequence[Message]) -> str:
    return await complete(
        llm,
        [
            Message.from_text(Role.SYSTEM, EPISODE_INSTRUCTIONS),
            Message.from_text(Role.USER, f"Conversation:\n{transcript(messages)}"),
        ],
    )


def parse_fact_changes(reply: str, known_ids: set[int]) -> FactChanges:
    """Read the model's JSON leniently; anything malformed means "no changes"."""
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end < start:
        return FactChanges()
    try:
        data = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        log.debug("Unlesbare Fakten-Antwort: %r", reply)
        return FactChanges()
    if not isinstance(data, dict):
        return FactChanges()
    remove = data.get("remove") or []
    added: list[tuple[str, int]] = []
    for item in data.get("add") or []:
        # Plain strings (older answers, small models) count as normal.
        if isinstance(item, str) and item.strip():
            added.append((item.strip(), NORMAL))
        elif isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
            level = LEVELS.get(str(item.get("importance", "")).lower(), NORMAL)
            if item.get("important") is True:
                level = IMPORTANT
            added.append((item["text"].strip(), level))
    return FactChanges(
        add=tuple(added),
        remove=tuple(i for i in remove if isinstance(i, int) and i in known_ids),
    )


async def extract_facts(
    llm: ChatModel, known: Sequence[Fact], messages: Sequence[Message], day: str = ""
) -> FactChanges:
    facts = "\n".join(f"{fact.id}: {fact.text}" for fact in known) or "(none)"
    content = f"Known facts:\n{facts}\n\nNew conversation lines:\n{transcript(messages)}"
    if day:
        content = f"Date of the conversation: {day}\n\n{content}"
    reply = await complete(
        llm,
        [Message.from_text(Role.SYSTEM, FACT_INSTRUCTIONS), Message.from_text(Role.USER, content)],
    )
    return parse_fact_changes(reply, {fact.id for fact in known})
