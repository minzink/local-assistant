import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from flay.config import MemorySettings
from flay.core.language import Language
from flay.core.messages import Message, Role, TextPart
from flay.llm.base import LLMEvent, ModelCapabilities, TextDelta, ToolSpec
from flay.memory.base import Fact, ShortTermMemory
from flay.memory.maintenance import parse_fact_changes
from flay.memory.persistent import PersistentMemory
from flay.memory.store import MemoryStore


def user(text: str) -> Message:
    return Message.from_text(Role.USER, text, Language.DE)


def assistant(text: str) -> Message:
    return Message.from_text(Role.ASSISTANT, text, Language.DE)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class ScriptedLLM:
    """Answers each request with the next scripted reply and records the prompts."""

    def __init__(self, *replies: str, delay: float = 0.0) -> None:
        self.replies = list(replies)
        self.prompts: list[Sequence[Message]] = []
        self.delay = delay

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.prompts.append(messages)
        await asyncio.sleep(self.delay)
        yield TextDelta(self.replies.pop(0))


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def make_memory(
    store: MemoryStore, llm: ScriptedLLM, clock: Clock, **settings: object
) -> PersistentMemory:
    config = MemorySettings.model_validate({"recent_messages": 2, "idle_seconds": 0, **settings})
    return PersistentMemory(store, llm, config, clock)


# Store ---------------------------------------------------------------------------------


def test_store_persists_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    first = MemoryStore(path)
    first.add_messages(1, [user("Hallo"), assistant("Hi!")], now=1.0)
    first.add_fact("Der Nutzer heißt Mia.", now=1.0)
    first.close()

    second = MemoryStore(path)

    assert [m.message.text for m in second.recent(1, 10)] == ["Hallo", "Hi!"]
    assert second.recent(1, 10)[0].message.language is Language.DE
    assert [f.text for f in second.facts()] == ["Der Nutzer heißt Mia."]


def test_new_session_after_timeout(store: MemoryStore) -> None:
    assert store.session_at(now=0, timeout_seconds=60) == 1
    store.add_messages(1, [user("a")], now=100)

    assert store.session_at(now=150, timeout_seconds=60) == 1
    assert store.session_at(now=200, timeout_seconds=60) == 2


def test_unsummarized_messages_outside_recent_window(store: MemoryStore) -> None:
    store.add_messages(1, [user("1"), assistant("2"), user("3"), assistant("4")], now=1)

    old = store.unsummarized_before_recent(1, keep=2)
    store.mark("summarized", [m.id for m in old])

    assert [m.message.text for m in old] == ["1", "2"]
    assert store.unsummarized_before_recent(1, keep=2) == []


def test_fact_limit_keeps_newest(store: MemoryStore) -> None:
    for text in ["a", "b", "c"]:
        store.add_fact(text, now=1)

    assert [f.text for f in store.facts(limit=2)] == ["b", "c"]


def test_remove_and_clear(store: MemoryStore) -> None:
    store.add_fact("a", now=1)
    store.add_fact("b", now=1)
    store.add_messages(1, [user("x")], now=1)

    assert store.remove_facts([store.facts()[0].id, 999]) == 1
    store.clear(facts=False)

    assert [f.text for f in store.facts()] == ["b"]
    assert store.recent(1, 10) == []


# Persistent memory -----------------------------------------------------------------------


def test_context_contains_recent_messages_of_current_session(store: MemoryStore) -> None:
    clock = Clock()
    memory = make_memory(store, ScriptedLLM(), clock, recent_messages=4)
    memory.record(user("Hallo"), assistant("Hi!"))

    assert [m.text for m in memory.context().recent] == ["Hallo", "Hi!"]

    clock.now += 31 * 60  # default timeout is 30 minutes
    assert memory.context().recent == ()


async def test_maintenance_summarizes_old_messages_and_extracts_facts(store: MemoryStore) -> None:
    llm = ScriptedLLM(
        "Der Nutzer hat sich vorgestellt.",
        '{"add": ["Der Nutzer heißt Mia."], "remove": []}',
    )
    memory = make_memory(store, llm, Clock())
    memory.record(user("Ich heiße Mia."), assistant("Hallo Mia!"))
    memory.record(user("Wie spät ist es?"), assistant("Kurz nach drei."))

    await memory.maintain()

    context = memory.context()
    assert context.summary == "Der Nutzer hat sich vorgestellt."
    assert [m.text for m in context.recent] == ["Wie spät ist es?", "Kurz nach drei."]
    # Facts are no longer in every prompt; they are recalled when they matter.
    assert context.facts == ()
    assert [f.text for f in memory.facts()] == ["Der Nutzer heißt Mia."]
    assert [f.text for f in await memory.recall("Weißt du noch, wie Mia heißt?")] == [
        "Der Nutzer heißt Mia."
    ]
    # Everything is processed: a second run does not call the LLM again.
    await memory.maintain()
    assert len(llm.prompts) == 2


async def test_facts_can_be_removed_and_duplicates_are_skipped(store: MemoryStore) -> None:
    store.add_fact("Lieblingsfarbe ist Blau.", now=1)
    fact_id = store.facts()[0].id
    llm = ScriptedLLM(
        f'Klar: {{"add": ["lieblingsfarbe ist blau.", "Mag Katzen."], "remove": [{fact_id}]}}'
    )
    memory = make_memory(store, llm, Clock(), summarize=False)
    memory.record(user("Vergiss meine Lieblingsfarbe. Ich mag Katzen."), assistant("Okay."))

    await memory.maintain()

    assert [f.text for f in store.facts()] == ["Mag Katzen."]
    assert "Lieblingsfarbe ist Blau." in llm.prompts[0][1].text


async def test_new_turn_cancels_background_work_which_resumes_later(store: MemoryStore) -> None:
    llm = ScriptedLLM('{"add": ["Mag Tee."]}', '{"add": ["Mag Tee."]}', delay=0.05)
    memory = make_memory(store, llm, Clock(), summarize=False)
    memory.record(user("Ich mag Tee."), assistant("Notiert."))

    memory.turn_finished()
    await asyncio.sleep(0.01)  # extraction is running
    memory.turn_started()  # user speaks again
    await asyncio.sleep(0.1)
    assert store.facts() == []

    memory.turn_finished()
    await asyncio.sleep(0.1)
    assert [f.text for f in store.facts()] == ["Mag Tee."]
    await memory.close()


async def test_failed_maintenance_is_logged_not_raised(store: MemoryStore) -> None:
    memory = make_memory(store, ScriptedLLM(), Clock(), summarize=False)  # no replies left
    memory.record(user("Hallo"), assistant("Hi"))

    memory.turn_finished()
    await asyncio.sleep(0.01)

    assert store.facts() == []
    await memory.close()


def test_short_term_memory_keeps_last_messages() -> None:
    memory = ShortTermMemory(max_messages=2)
    memory.record(user("a"), assistant("b"))
    memory.record(user("c"), assistant("d"))

    assert [m.text for m in memory.context().recent] == ["c", "d"]


# Parsing -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "add", "remove"),
    [
        ('{"add": ["A"], "remove": [1]}', (("A", 2),), (1,)),
        ('Here you go:\n```json\n{"add": [], "remove": []}\n```', (), ()),
        ('{"add": ["  ", 3, "B"], "remove": [99, "1"]}', (("B", 2),), ()),
        (
            '{"add": [{"text": "Nie committen.", "importance": "high"}, '
            '{"text": "Heute rot.", "importance": "low"}, {"text": "C"}], "remove": []}',
            (("Nie committen.", 3), ("Heute rot.", 1), ("C", 2)),
            (),
        ),
        ("Keine neuen Fakten.", (), ()),
        ('{"add": [', (), ()),
        ("[1, 2]", (), ()),
    ],
)
def test_parse_fact_changes(
    reply: str, add: tuple[tuple[str, int], ...], remove: tuple[int, ...]
) -> None:
    changes = parse_fact_changes(reply, known_ids={1})

    assert (changes.add, changes.remove) == (add, remove)


def test_fact_is_value_object() -> None:
    assert Fact(1, "x") == Fact(1, "x")


def test_interrupted_flag_is_stored_and_old_databases_are_migrated(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.sqlite3"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session INTEGER NOT NULL, "
        "role TEXT NOT NULL, text TEXT NOT NULL, language TEXT, created_at REAL NOT NULL, "
        "summarized INTEGER NOT NULL DEFAULT 0, facts_done INTEGER NOT NULL DEFAULT 0)"
    )
    old.execute(
        "INSERT INTO messages (session, role, text, created_at) VALUES (1, 'user', 'Alt', 0)"
    )
    old.commit()
    old.close()

    store = MemoryStore(path)
    cut = Message(Role.ASSISTANT, (TextPart("Halb"),), Language.DE, interrupted=True)
    store.add_messages(1, [cut], now=1.0)

    first, second = (stored.message for stored in store.recent(1, 10))
    assert (first.text, first.interrupted) == ("Alt", False)
    assert (second.text, second.interrupted) == ("Halb", True)


def test_actions_are_stored_with_the_reply(tmp_path: Path) -> None:
    from flay.core.messages import ToolCall, ToolOutcome

    store = MemoryStore(tmp_path / "m.sqlite3")
    action = ToolOutcome(ToolCall("x", "close_project", {"name": "flay"}), "Closed.", ok=True)
    reply = Message(Role.ASSISTANT, (TextPart("Zu."),), Language.DE, actions=(action,))
    store.add_messages(1, [reply], now=1.0)

    (stored,) = MemoryStore(tmp_path / "m.sqlite3").recent(1, 10)
    (loaded,) = stored.message.actions
    assert (loaded.call.name, loaded.call.arguments, loaded.result, loaded.ok) == (
        "close_project",
        {"name": "flay"},
        "Closed.",
        True,
    )
