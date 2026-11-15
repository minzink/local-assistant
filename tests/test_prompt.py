from datetime import datetime

import pytest
from pydantic import ValidationError

from cassi.config import PersonaSettings
from cassi.core.language import Language
from cassi.core.messages import Message, Role, TextPart
from cassi.memory.base import Fact, MemoryContext
from cassi.prompt import build_messages, system_prompt

NOW = datetime(2026, 10, 24, 21, 30)


def test_minimal_system_prompt() -> None:
    prompt = system_prompt(PersonaSettings(), MemoryContext(), Language.DE, NOW)

    assert prompt.startswith("You are Cassi")
    # The time is part of the user message, so the system prompt stays cacheable.
    assert "2026" not in prompt
    assert prompt.endswith("Always reply in German.")
    assert "mood" not in prompt
    assert "know about the user" not in prompt


def test_mood_facts_and_summary_are_included() -> None:
    persona = PersonaSettings(mood={"cheerful": 7, "curious": 4})
    memory = MemoryContext(summary="Wir sprachen über Tee.", facts=(Fact(1, "Heißt Mia."),))

    prompt = system_prompt(persona, memory, Language.EN, NOW)

    assert "Your current mood: cheerful 7/10, curious 4/10." in prompt
    assert "What you know about the user:\n- Heißt Mia." in prompt
    assert "Earlier in this conversation:\nWir sprachen über Tee." in prompt


def test_messages_contain_history_and_hinted_user_message() -> None:
    history = (Message.from_text(Role.USER, "Hallo"), Message.from_text(Role.ASSISTANT, "Hi"))
    question = Message.from_text(Role.USER, "What time is it?", Language.EN)

    messages = build_messages(
        PersonaSettings(), MemoryContext(recent=history), question, Language.EN, NOW
    )

    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER]
    assert messages[-1].text == (
        "What time is it?\n(Now: Saturday, 2026-10-24 21:30. Reply in English.)"
    )
    assert question.text == "What time is it?"


def test_mood_out_of_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PersonaSettings(mood={"angry": 11})


def test_interruptions_are_marked_in_the_next_user_message() -> None:
    def reply(text: str) -> Message:
        return Message(Role.ASSISTANT, (TextPart(text),), Language.DE, interrupted=True)

    recent = (
        Message.from_text(Role.USER, "Erzähl mir was über Rom."),
        reply("Rom wurde der Sage nach"),
        Message.from_text(Role.USER, "Und Paris?"),
        reply(""),
    )
    new = Message.from_text(Role.USER, "Nein, ich meine Athen.")

    messages = build_messages(
        PersonaSettings(), MemoryContext(recent=recent), new, Language.DE, NOW
    )[1:]

    assert [m.text for m in messages[:3]] == [
        "Erzähl mir was über Rom.",
        "Rom wurde der Sage nach …",
        "(I interrupted you there.) Und Paris?",
    ]
    # The reply that never started is left out; the next message says so.
    assert messages[3].text.startswith("(I interrupted you before you answered.) Nein, ich")
    assert len(messages) == 4


def test_older_replies_reach_the_model_with_their_first_sentence_only() -> None:
    from cassi.prompt import condense_older

    def reply(text: str) -> Message:
        return Message.from_text(Role.ASSISTANT, text)

    history = [
        Message.from_text(Role.USER, "Eins"),
        reply("Erste Antwort. Sag Bescheid, wenn du was brauchst."),
        Message.from_text(Role.USER, "Zwei"),
        reply("Zweite Antwort. Ich bin hier."),
        Message.from_text(Role.USER, "Drei"),
        reply("Dritte Antwort. Was ist dran?"),
    ]

    texts = [m.text for m in condense_older(history, verbatim=2)]

    assert texts[1] == "Erste Antwort."  # its stock ending is no template any more
    assert texts[3:] == ["Zweite Antwort. Ich bin hier.", "Drei", "Dritte Antwort. Was ist dran?"]
    assert [m.text for m in condense_older(history, None)] == [m.text for m in history]
