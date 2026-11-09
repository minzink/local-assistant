from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from flay.config import MemorySettings, PersonaSettings, RecallSettings
from flay.core.language import Language
from flay.core.messages import Message, Role
from flay.llm.base import LLMEvent, ModelCapabilities, TextDelta, ToolSpec
from flay.memory.base import Fact, MemoryContext
from flay.memory.embeddings import normalize
from flay.memory.persistent import PersistentMemory
from flay.memory.recall import Recall, keyword_query
from flay.memory.store import MemoryStore
from flay.prompt import build_messages

# Meaning as a few hand-made dimensions: words of one topic share a dimension.
TOPICS = {
    "urlaub": 0, "frei": 0, "freitag": 0, "ferien": 0,
    "cache": 1, "caching": 1, "redis": 1,
    "kaffee": 2, "tee": 2, "trinkt": 2,
    "heißt": 3, "name": 3, "heiße": 3,
}  # fmt: skip


class TopicEmbedder:
    """Deterministic embeddings: a vector per topic the text mentions."""

    model = "topics"

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: Sequence[str], query: bool = False) -> np.ndarray:
        self.calls += 1
        rows = []
        for text in texts:
            vector = np.zeros(5 + 32, dtype=np.float32)
            for word in text.lower().replace("?", " ").replace(".", " ").split():
                if word in TOPICS:
                    vector[TOPICS[word]] += 1.0
                else:  # other words spread a little, so unrelated texts differ
                    vector[5 + sum(map(ord, word)) % 32] += 0.2
            rows.append(vector)
        return normalize(np.array(rows))


class BrokenEmbedder(TopicEmbedder):
    async def embed(self, texts: Sequence[str], query: bool = False) -> np.ndarray:
        raise ConnectionError("Ollama läuft nicht")


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    for text in [
        "Mia hat am Freitag Urlaub.",
        "Im Projekt umx wird Redis als Cache genutzt.",
        "Mia trinkt lieber Tee als Kaffee.",
        "Das Projekt kx-ewo nutzt Symfony 7.",
    ]:
        store.add_fact(text, now=1.0)
    return store


async def test_recall_finds_by_meaning_and_by_keyword(store: MemoryStore) -> None:
    recall = Recall(store, TopicEmbedder(), RecallSettings(limit=2))
    await recall.embed_pending()

    by_meaning = await recall.search("Wann habe ich frei?")
    by_keyword = await recall.search("Was nutzt kx-ewo?")

    assert by_meaning[0].text == "Mia hat am Freitag Urlaub."
    assert by_keyword[0].text == "Das Projekt kx-ewo nutzt Symfony 7."


async def test_unrelated_request_recalls_nothing(store: MemoryStore) -> None:
    recall = Recall(store, TopicEmbedder(), RecallSettings())
    await recall.embed_pending()

    assert await recall.search("Erzähl einen Witz") == []


async def test_recall_falls_back_to_keywords_without_embeddings(store: MemoryStore) -> None:
    recall = Recall(store, BrokenEmbedder(), RecallSettings())

    found = await recall.search("Was ist mit Redis?")

    assert [f.text for f in found] == ["Im Projekt umx wird Redis als Cache genutzt."]


def test_keyword_query_drops_common_words() -> None:
    assert keyword_query("Was hat er mit dem Projekt kx-ewo gemacht?") == "kx ewo gemacht"


async def test_new_or_changed_memories_get_embedded_once(store: MemoryStore) -> None:
    embedder = TopicEmbedder()
    recall = Recall(store, embedder, RecallSettings())

    assert await recall.embed_pending() == 4
    assert await recall.embed_pending() == 0
    episode = store.set_episode(7, "Conversation on 2026-10-29: Cache besprochen.", now=2.0)
    assert await recall.embed_pending() == 1
    store.set_episode(7, "Conversation on 2026-10-29: Cache und Redis besprochen.", now=2.0)
    assert await recall.embed_pending() == 1  # edited: embedded again
    assert store.facts_by_ids([episode])[0].kind == "episode"


def test_keyword_index_covers_memories_from_before_it_existed(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "old.sqlite3"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, text TEXT NOT NULL, created_at REAL)")
    old.execute("INSERT INTO facts (text, created_at) VALUES ('Lea hat am 3. März Geburtstag.', 1)")
    old.commit()
    old.close()

    store = MemoryStore(path)

    assert store.keyword_search("geburtstag lea", 5) == [1]
    assert store.facts()[0].kind == "fact"


class ScriptedLLM:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        yield TextDelta(self.replies.pop(0) if self.replies else "")


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


async def test_finished_conversation_becomes_an_episode(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    clock = Clock()
    settings = MemorySettings(idle_seconds=0, extract_facts=False)
    memory = PersistentMemory(
        store, ScriptedLLM("Es ging um Redis als Cache."), settings, clock, TopicEmbedder()
    )
    memory.record(
        Message.from_text(Role.USER, "Sollen wir Redis nehmen?"),
        Message.from_text(Role.ASSISTANT, "Ja, als Cache."),
    )
    clock.now += 2 * 3600  # a new session starts
    memory.record(Message.from_text(Role.USER, "Hallo"), Message.from_text(Role.ASSISTANT, "Hi"))

    await memory.maintain()

    (episode,) = store.facts(kind="episode")
    assert episode.text.endswith("Es ging um Redis als Cache.")
    assert episode.text.startswith("Conversation on ")
    found = await memory.recall("Was hatten wir zum Caching gesagt?")
    assert episode in found


async def test_remembered_fact_is_findable_at_once(tmp_path: Path) -> None:
    import asyncio

    memory = PersistentMemory(
        MemoryStore(tmp_path / "m.sqlite3"),
        ScriptedLLM(),
        MemorySettings(),
        Clock(),
        TopicEmbedder(),
    )

    memory.remember("Mia hat am Freitag Urlaub.")
    await asyncio.sleep(0.05)  # embedding runs in the background

    assert [f.text for f in await memory.recall("Habe ich bald Ferien?")] == [
        "Mia hat am Freitag Urlaub."
    ]
    await memory.close()


def test_recalled_memories_go_into_the_new_message_with_their_date() -> None:
    memory = Fact(1, "Mia hat am Freitag Urlaub.", created_at=datetime(2026, 10, 27).timestamp())

    messages = build_messages(
        PersonaSettings(),
        MemoryContext(),
        Message.from_text(Role.USER, "Wann habe ich frei?"),
        Language.DE,
        datetime(2026, 10, 31, 15, 0),
        memories=[memory],
    )

    assert "Mia hat am Freitag Urlaub. (noted 2026-10-27)" in messages[-1].text
    assert "Urlaub" not in messages[0].text  # the system prompt stays the same


async def test_rules_are_important_and_always_in_the_prompt(tmp_path: Path) -> None:
    memory = PersistentMemory(
        MemoryStore(tmp_path / "m.sqlite3"), ScriptedLLM(), MemorySettings(), Clock()
    )

    memory.remember("Gib Claude in Zukunft keinen Befehl mehr, etwas zu committen.")
    memory.remember("Die README wurde nicht angepasst.")
    memory.remember("Lea hat am 3. März Geburtstag.", important=True)

    context = memory.context()
    assert [f.text for f in context.important] == [
        "Gib Claude in Zukunft keinen Befehl mehr, etwas zu committen.",
        "Lea hat am 3. März Geburtstag.",
    ]
    system = build_messages(
        PersonaSettings(), context, Message.from_text(Role.USER, "x"), Language.DE, datetime.now()
    )[0].text
    assert "always follow it" in system and "committen" in system
    # Recall leaves them out: they are in the prompt already.
    assert all(not f.important for f in await memory.recall("committen Claude Geburtstag"))
    await memory.close()


async def test_memories_carry_when_they_were_written_and_episodes_are_minor(
    tmp_path: Path,
) -> None:
    from flay.memory.base import MINOR

    store = MemoryStore(tmp_path / "m.sqlite3")
    clock = Clock()
    memory = PersistentMemory(
        store, ScriptedLLM("Es ging um die README."), MemorySettings(extract_facts=False), clock
    )
    memory.record(Message.from_text(Role.USER, "README?"), Message.from_text(Role.ASSISTANT, "Ok"))
    started = clock.now
    clock.now += 2 * 3600
    memory.record(Message.from_text(Role.USER, "Hallo"), Message.from_text(Role.ASSISTANT, "Hi"))

    await memory.maintain()

    (episode,) = store.facts(kind="episode")
    assert episode.importance == MINOR
    assert episode.created_at == clock.now  # written now …
    day = datetime.fromtimestamp(started).strftime("%Y-%m-%d")
    assert episode.text.startswith(f"Conversation on {day}")  # … about that conversation


async def test_extraction_keeps_the_importance_the_model_gives(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m.sqlite3")
    reply = (
        '{"add": [{"text": "Flay soll Claude nie anweisen, etwas zu committen.", '
        '"importance": "high"}, {"text": "Mia nutzt PhpStorm.", "importance": "normal"}, '
        '{"text": "Die README wurde heute nicht angepasst.", "importance": "low"}], "remove": []}'
    )
    memory = PersistentMemory(store, ScriptedLLM(reply), MemorySettings(summarize=False), Clock())
    memory.record(
        Message.from_text(Role.USER, "Sag Claude nie mehr, er soll committen."),
        Message.from_text(Role.ASSISTANT, "Verstanden."),
    )

    await memory.maintain()

    assert [(f.text, f.importance) for f in store.facts()] == [
        ("Flay soll Claude nie anweisen, etwas zu committen.", 3),
        ("Mia nutzt PhpStorm.", 2),
        ("Die README wurde heute nicht angepasst.", 1),
    ]


def test_memory_list_shows_date_kind_and_importance() -> None:
    from flay.cli import _memory_line
    from flay.memory.base import IMPORTANT

    line = _memory_line(
        Fact(
            7,
            "Nie committen.",
            created_at=datetime(2026, 10, 31, 14, 5).timestamp(),
            importance=IMPORTANT,
        )
    )

    assert "2026-10-31 14:05" in line and "Fakt" in line and "wichtig" in line
