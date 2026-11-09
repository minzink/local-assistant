"""Builds the chat messages for one turn from persona, memory and the new user message."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from flay.config import PersonaSettings
from flay.core.language import Language
from flay.core.messages import Message, Role, TextPart, ToolCall
from flay.memory.base import Fact, MemoryContext

LANGUAGE_NAMES = {Language.DE: "German", Language.EN: "English"}


TOOL_INSTRUCTIONS = (
    "You can act on the user's computer with the provided tools. When the user asks for such "
    "an action, call the tool right away – do not ask for confirmation first; anything risky "
    "is confirmed by the user elsewhere (e.g. pressing Enter in Claude). Never claim to have "
    "done something without calling the tool, and afterwards say briefly what you did."
)


def system_prompt(
    persona: PersonaSettings,
    memory: MemoryContext,
    language: Language,
    now: datetime,
    tools: bool = False,
) -> str:
    sections = [persona.system_prompt]
    if tools:
        sections.append(TOOL_INSTRUCTIONS)
    if persona.mood:
        levels = ", ".join(f"{name} {level}/10" for name, level in persona.mood.items())
        sections.append(f"Your current mood: {levels}. Let it subtly color your tone.")
    if memory.important:
        rules = "\n".join(f"- {f.text} (noted {_day(f.created_at)})" for f in memory.important)
        sections.append(f"Important – the user told you this; always follow it:\n{rules}")
    if memory.facts:
        facts = "\n".join(f"- {fact.text}" for fact in memory.facts)
        sections.append(f"What you know about the user:\n{facts}")
    if memory.summary:
        sections.append(f"Earlier in this conversation:\n{memory.summary}")
    sections.append(f"Always reply in {LANGUAGE_NAMES[language]}.")
    return "\n\n".join(sections)


def build_messages(
    persona: PersonaSettings,
    memory: MemoryContext,
    user_message: Message,
    language: Language,
    now: datetime,
    tools: bool = False,
    state: str | None = None,
    memories: Sequence[Fact] = (),
    examples: Sequence[Fact] = (),
) -> list[Message]:
    # Small models follow the language of the conversation history over the system prompt
    # (llama3.2: 7/10 after a language switch). A hint in the new message itself fixes that
    # (10/10); it is only sent, never stored.
    name = LANGUAGE_NAMES[language]
    # The time changes every minute, so it goes into the new message: the system prompt and
    # history stay identical between turns and the server can reuse their processed prefix.
    # The same goes for what is open on the computer, which changes all the time.
    situation = f" {state}" if state else ""
    hint = f"\n(Now: {now:%A, %Y-%m-%d %H:%M}.{situation} Reply in {name}.)"
    if examples:
        hint = f"\n({format_examples(examples)})" + hint
    if memories:
        # Found for this request (RAG); dated, so "Freitag" can be placed in time.
        listed = "\n".join(
            f"- {m.text} (noted {_day(m.created_at)}{', minor' if m.importance < 2 else ''})"
            for m in memories
        )
        hint = f"\n(You remember – use only if it helps:\n{listed})" + hint
    hinted = replace(user_message, parts=(*user_message.parts, TextPart(hint)))
    return [
        Message.from_text(Role.SYSTEM, system_prompt(persona, memory, language, now, tools)),
        *mark_interruptions([*memory.recent, hinted]),
    ]


INTERRUPTED_ANSWER = "(I interrupted you there.) "
INTERRUPTED_BEFORE_ANSWER = "(I interrupted you before you answered.) "


# Tool results from earlier turns are shortened; the details mattered only back then.
PAST_RESULT_CHARS = 300


def format_examples(examples: Sequence[Fact]) -> str:
    """Earlier requests as heard and the action each meant – teaches what words mean."""
    lines = []
    for example in examples:
        if not example.data:
            continue
        data = json.loads(example.data)
        arguments = ", ".join(
            f"{key}={str(value)[:60]!r}" for key, value in (data.get("arguments") or {}).items()
        )
        lines.append(f'- "{example.text}" → {data["tool"]}({arguments})')
    if not lines:
        return ""
    listed = "\n".join(lines)
    return f"Earlier requests that sounded like this, and what the user meant:\n{listed}"


def _day(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else "?"


def mark_interruptions(messages: list[Message]) -> list[Message]:
    """The stored conversation as the model should see it.

    - Tools that ran for a reply appear as the tool calls and results they were, so the
      model knows what really happened instead of guessing from the words.
    - Where the user cut the model off, the next user message says so. Otherwise an
      interrupted reply looks as if the user heard all of it, and a request interrupted
      before any answer is missing. The note goes into the user's message, not into the
      reply, so the model does not start imitating it.
    """
    marked: list[Message] = []
    note = ""
    for index, message in enumerate(messages):
        if message.role is Role.ASSISTANT and message.actions:
            marked.extend(_past_actions(message, index))
        if message.role is Role.ASSISTANT and message.interrupted:
            if message.text:
                marked.append(replace(message, parts=(TextPart(message.text + " …"),), actions=()))
                note = INTERRUPTED_ANSWER
            else:
                note = INTERRUPTED_BEFORE_ANSWER
            continue
        if message.role is Role.USER and note:
            message = replace(message, parts=(TextPart(note), *message.parts))
        note = ""
        if message.role is Role.ASSISTANT and message.actions:
            if not message.text:
                continue
            message = replace(message, actions=())
        marked.append(message)
    return marked


def _past_actions(reply: Message, index: int) -> list[Message]:
    # Ids only need to be unique within one request.
    calls = tuple(
        ToolCall(f"past{index}_{n}", action.call.name, action.call.arguments)
        for n, action in enumerate(reply.actions)
    )
    results = [
        Message(Role.TOOL, (TextPart(action.result[:PAST_RESULT_CHARS]),), tool_call_id=call.id)
        for action, call in zip(reply.actions, calls, strict=True)
    ]
    return [Message(Role.ASSISTANT, (TextPart(""),), reply.language, tool_calls=calls), *results]
