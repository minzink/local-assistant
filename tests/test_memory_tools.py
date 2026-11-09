from pathlib import Path

import pytest

from flay.config import MemorySettings, Settings
from flay.core.language import Language
from flay.core.messages import Message, Role
from flay.memory.base import Fact
from flay.memory.persistent import PersistentMemory
from flay.memory.store import MemoryStore
from flay.tools.factory import create_tools
from flay.tools.memory import ForgetTool, RememberTool, match_facts
from tests.test_memory import ScriptedLLM

FACTS = [
    Fact(1, "Der Nutzer heißt Mia."),
    Fact(2, "Der Nutzer mag Tee."),
    Fact(3, "Lea hat am 3. März Geburtstag."),
    Fact(4, "Der Nutzer mag grünen Tee am Morgen."),
]


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Lea hat am 3. März Geburtstag", [3]),
        ("Leas Geburtstag", [3]),
        ("dass ich Mia heiße", [1]),
        ("Tee", [2, 4]),
        ("Lieblingsfarbe", []),
    ],
)
def test_match_facts(description: str, expected: list[int]) -> None:
    assert sorted(f.id for f in match_facts(description, FACTS)) == expected


@pytest.fixture
def memory() -> PersistentMemory:
    config = MemorySettings.model_validate({"idle_seconds": 0})
    return PersistentMemory(MemoryStore(":memory:"), ScriptedLLM(), config)


async def test_remember_and_forget(memory: PersistentMemory) -> None:
    remembered = await RememberTool(memory).run({"fact": "Lea hat am 3. März Geburtstag."})
    duplicate = await RememberTool(memory).run({"fact": "lea hat am 3. märz geburtstag."})
    forgotten = await ForgetTool(memory).run({"fact": "Leas Geburtstag"})

    assert remembered.ok and "gemerkt" in (remembered.action or "")
    assert duplicate.ok and "schon bekannt" in (duplicate.action or "")
    assert forgotten.ok and memory.facts() == []


async def test_forget_asks_when_ambiguous_and_reports_unknown(memory: PersistentMemory) -> None:
    memory.remember("Der Nutzer mag Tee.")
    memory.remember("Der Nutzer mag grünen Tee am Morgen.")

    ambiguous = await ForgetTool(memory).run({"fact": "Tee"})
    unknown = await ForgetTool(memory).run({"fact": "Lieblingsfarbe"})

    assert not ambiguous.ok and len(memory.facts()) == 2
    assert "grünen Tee" in ambiguous.text
    assert not unknown.ok


async def test_exchange_handled_by_a_memory_tool_is_not_extracted_again(
    memory: PersistentMemory,
) -> None:
    llm = ScriptedLLM()  # extraction would fail loudly if it were called
    memory._llm = llm
    memory.turn_started()
    memory.remember("Der Nutzer mag Tee.")
    memory.record(
        Message.from_text(Role.USER, "Merk dir, dass ich Tee mag.", Language.DE),
        Message.from_text(Role.ASSISTANT, "Gemerkt.", Language.DE),
    )
    memory.turn_finished()

    await memory.maintain()

    assert llm.prompts == []
    assert [f.text for f in memory.facts()] == ["Der Nutzer mag Tee."]
    await memory.close()


def test_factory_adds_memory_tools_only_with_persistent_memory(
    memory: PersistentMemory,
) -> None:
    settings = Settings.model_validate({"tools": {"enabled": ["remember", "forget"]}})

    assert [s.name for s in create_tools(settings).specs] == []
    assert [s.name for s in create_tools(settings, memory).specs] == ["remember", "forget"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Merk dir, dass Lea am 3. März Geburtstag hat.", "remember"),
        ("Bitte merke dir meine Lieblingsfarbe: Blau.", "remember"),
        ("Remember that my daughter is called Lea.", "remember"),
        ("Vergiss, dass ich Tee mag.", "forget"),
        ("Lösch das bitte aus deinem Gedächtnis.", "forget"),
        ("Forget my favourite colour.", "forget"),
        ("Wie viel Speicher hat mein Mac?", None),
        ("Sag Claude, er soll sich die Tests merken.", "send_to_claude"),
    ],
)
def test_tool_intent(text: str, expected: str | None) -> None:
    from flay.assistant import tool_intent

    assert tool_intent(text) == expected


def test_store_returns_message_ids(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite3")
    ids = store.add_messages(
        1, [Message.from_text(Role.USER, "a"), Message.from_text(Role.ASSISTANT, "b")], now=1
    )
    assert ids == [1, 2]
