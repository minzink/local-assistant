"""SQLite storage for conversations, session summaries and long-term memories.

Memories (facts and episodes) carry an embedding for meaning-based search and are indexed
with FTS5 for keyword search; both together find them again (see ``memory.recall``).
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flay.core.language import Language
from flay.core.messages import Message, Role, TextPart, ToolCall, ToolOutcome
from flay.memory.base import KNOWLEDGE, MINOR, MISTAKE, NORMAL, VERB, Fact

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session INTEGER NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    language TEXT,
    created_at REAL NOT NULL,
    summarized INTEGER NOT NULL DEFAULT 0,
    facts_done INTEGER NOT NULL DEFAULT 0,
    interrupted INTEGER NOT NULL DEFAULT 0,
    actions TEXT
);
CREATE INDEX IF NOT EXISTS messages_session ON messages (session, id);
CREATE TABLE IF NOT EXISTS summaries (
    session INTEGER PRIMARY KEY,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

# Keyword index over the memories, kept in sync by triggers. unicode61 with diacritics
# removed matches "Schließe" and "schliesse" alike.
FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    text, content='facts', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF text ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;
"""

COLUMNS = "id, session, role, text, language, created_at, interrupted, actions"
ADDED_COLUMNS = {
    "interrupted": "INTEGER NOT NULL DEFAULT 0",
    # Tools that ran for an assistant reply, as JSON.
    "actions": "TEXT",
}


ADDED_FACT_COLUMNS = {
    "kind": "TEXT NOT NULL DEFAULT 'fact'",
    # Episodes: the conversation session they summarize.
    "session": "INTEGER",
    "embedding": "BLOB",
    # The embedding model the vector came from; another model means embedding again.
    "embedding_model": "TEXT",
    # 3 important (a rule that always applies), 2 normal, 1 minor (see memory.base).
    "importance": "INTEGER NOT NULL DEFAULT 2",
    # Examples: the action as JSON.
    "data": "TEXT",
}
FACT_COLUMNS = "id, text, kind, created_at, importance, data"


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: int
    session: int
    message: Message
    created_at: float


class MemoryStore:
    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(SCHEMA)
        # Databases from older versions get the columns added since.
        for table, added in (("messages", ADDED_COLUMNS), ("facts", ADDED_FACT_COLUMNS)):
            columns = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
            with self._db:
                for column, definition in added.items():
                    if column not in columns:
                        self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        new_index = not self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'facts_fts'"
        ).fetchone()
        self._db.executescript(FTS)
        if new_index:
            # Memories from before the keyword index existed.
            with self._db:
                self._db.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")

    def close(self) -> None:
        self._db.close()

    # Sessions -------------------------------------------------------------------------

    def session_at(self, now: float, timeout_seconds: float) -> int:
        """Session a message at ``now`` belongs to; a long pause starts a new one."""
        row = self._db.execute(
            "SELECT session, created_at FROM messages ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return 1
        session, last = int(row[0]), float(row[1])
        return session if now - last <= timeout_seconds else session + 1

    # Messages -------------------------------------------------------------------------

    def add_messages(self, session: int, messages: list[Message], now: float) -> list[int]:
        ids = []
        with self._db:
            for m in messages:
                cursor = self._db.execute(
                    "INSERT INTO messages "
                    "(session, role, text, language, created_at, interrupted, actions) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        session,
                        m.role.value,
                        m.text,
                        m.language and m.language.value,
                        now,
                        int(m.interrupted),
                        _dump_actions(m.actions),
                    ),
                )
                ids.append(int(cursor.lastrowid or 0))
        return ids

    def recent(self, session: int, limit: int) -> list[StoredMessage]:
        rows = self._db.execute(
            f"SELECT * FROM (SELECT {COLUMNS} "
            "FROM messages WHERE session = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
            (session, limit),
        ).fetchall()
        return [_stored(row) for row in rows]

    def unsummarized_before_recent(self, session: int, keep: int) -> list[StoredMessage]:
        """Messages that dropped out of the recent window and are not summarized yet."""
        rows = self._db.execute(
            f"SELECT {COLUMNS} FROM messages "
            "WHERE session = ? AND summarized = 0 AND id NOT IN "
            "(SELECT id FROM messages WHERE session = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
            (session, session, keep),
        ).fetchall()
        return [_stored(row) for row in rows]

    def pending_facts(self, limit: int) -> list[StoredMessage]:
        rows = self._db.execute(
            f"SELECT {COLUMNS} FROM messages WHERE facts_done = 0 ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()
        return [_stored(row) for row in rows]

    def mark(self, column: str, ids: list[int]) -> None:
        if column not in ("summarized", "facts_done"):
            raise ValueError(column)
        with self._db:
            self._db.executemany(
                f"UPDATE messages SET {column} = 1 WHERE id = ?", [(i,) for i in ids]
            )

    # Summaries ------------------------------------------------------------------------

    def summary(self, session: int) -> str | None:
        row = self._db.execute(
            "SELECT text FROM summaries WHERE session = ?", (session,)
        ).fetchone()
        return str(row[0]) if row else None

    def set_summary(self, session: int, text: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO summaries (session, text) VALUES (?, ?) "
                "ON CONFLICT (session) DO UPDATE SET text = excluded.text",
                (session, text),
            )

    # Facts ----------------------------------------------------------------------------

    def facts(
        self, limit: int | None = None, kind: str | None = None, min_importance: int = 0
    ) -> list[Fact]:
        """The newest memories, oldest first; only one kind / at least this importance."""
        rows = self._db.execute(
            f"SELECT * FROM (SELECT {FACT_COLUMNS} FROM facts "
            "WHERE (? IS NULL OR kind = ?) AND importance >= ? ORDER BY id DESC LIMIT ?) "
            "ORDER BY id",
            (kind, kind, min_importance, -1 if limit is None else limit),
        ).fetchall()
        return [_fact(row) for row in rows]

    def facts_by_ids(self, ids: list[int]) -> list[Fact]:
        """Memories in the order of ``ids``; ids no longer stored are skipped."""
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self._db.execute(
            f"SELECT {FACT_COLUMNS} FROM facts WHERE id IN ({marks})", ids
        ).fetchall()
        found = {int(row[0]): _fact(row) for row in rows}
        return [found[i] for i in ids if i in found]

    def add_fact(
        self,
        text: str,
        now: float,
        kind: str = "fact",
        session: int | None = None,
        importance: int = NORMAL,
        data: str | None = None,
    ) -> int:
        with self._db:
            cursor = self._db.execute(
                "INSERT INTO facts (text, created_at, kind, session, importance, data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (text, now, kind, session, importance, data),
            )
        return int(cursor.lastrowid or 0)

    def update_data(self, fact_id: int, data: str, now: float) -> None:
        with self._db:
            self._db.execute(
                "UPDATE facts SET data = ?, created_at = ? WHERE id = ?", (data, now, fact_id)
            )

    def set_episode(self, session: int, text: str, now: float) -> int:
        """The memory of a conversation session, replaced when its summary grows."""
        row = self._db.execute(
            "SELECT id FROM facts WHERE kind = 'episode' AND session = ?", (session,)
        ).fetchone()
        if row is None:
            return self.add_fact(text, now, kind="episode", session=session, importance=MINOR)
        with self._db:
            self._db.execute(
                "UPDATE facts SET text = ?, created_at = ?, embedding = NULL WHERE id = ?",
                (text, now, row[0]),
            )
        return int(row[0])

    # Search -----------------------------------------------------------------------------

    def unembedded(self, model: str, limit: int = 64) -> list[Fact]:
        """Memories without a vector from ``model`` (new, edited, or another model)."""
        rows = self._db.execute(
            f"SELECT {FACT_COLUMNS} FROM facts WHERE kind NOT IN (?, ?) AND (embedding IS NULL "
            "OR embedding_model IS NOT ?) ORDER BY id LIMIT ?",
            (MISTAKE, VERB, model, limit),
        ).fetchall()
        return [_fact(row) for row in rows]

    def set_embedding(self, fact_id: int, vector: np.ndarray, model: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE facts SET embedding = ?, embedding_model = ? WHERE id = ?",
                (vector.astype(np.float32).tobytes(), model, fact_id),
            )

    def embeddings(
        self, model: str, kinds: tuple[str, ...] = KNOWLEDGE
    ) -> tuple[list[int], np.ndarray]:
        """Ids and vectors (one row each) of the memories of ``kinds`` embedded with ``model``."""
        marks = ",".join("?" * len(kinds))
        rows = self._db.execute(
            "SELECT id, embedding FROM facts WHERE embedding IS NOT NULL AND embedding_model = ? "
            f"AND kind IN ({marks})",
            (model, *kinds),
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        return [int(r[0]) for r in rows], np.stack(
            [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        )

    def keyword_search(
        self, query: str, limit: int, kinds: tuple[str, ...] = KNOWLEDGE
    ) -> list[int]:
        """Ids of memories of ``kinds`` sharing words with ``query``, best match first."""
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) >= 2]
        if not words:
            return []
        match = " OR ".join(f'"{word}"' for word in dict.fromkeys(words))
        marks = ",".join("?" * len(kinds))
        rows = self._db.execute(
            "SELECT facts_fts.rowid FROM facts_fts JOIN facts ON facts.id = facts_fts.rowid "
            f"WHERE facts_fts MATCH ? AND facts.kind IN ({marks}) "
            "ORDER BY bm25(facts_fts) LIMIT ?",
            (match, *kinds, limit),
        ).fetchall()
        return [int(row[0]) for row in rows]

    def sessions_without_episode(self, current: int) -> list[int]:
        """Finished sessions (before ``current``) not yet stored as an episode, newest first."""
        rows = self._db.execute(
            "SELECT DISTINCT session FROM messages WHERE session < ? AND session NOT IN "
            "(SELECT session FROM facts WHERE kind = 'episode' AND session IS NOT NULL) "
            "ORDER BY session DESC",
            (current,),
        ).fetchall()
        return [int(row[0]) for row in rows]

    def session_messages(self, session: int, limit: int) -> list[StoredMessage]:
        """The last ``limit`` messages of a session, oldest first."""
        rows = self._db.execute(
            f"SELECT * FROM (SELECT {COLUMNS} FROM messages WHERE session = ? "
            "ORDER BY id DESC LIMIT ?) ORDER BY id",
            (session, limit),
        ).fetchall()
        return [_stored(row) for row in rows]

    def episode_ids(self) -> set[int]:
        rows = self._db.execute("SELECT id FROM facts WHERE kind = 'episode'").fetchall()
        return {int(row[0]) for row in rows}

    def latest_episode(self) -> Fact | None:
        row = self._db.execute(
            f"SELECT {FACT_COLUMNS} FROM facts WHERE kind = 'episode' ORDER BY session DESC LIMIT 1"
        ).fetchone()
        return _fact(row) if row else None

    def unsummarized(self, session: int) -> list[StoredMessage]:
        rows = self._db.execute(
            f"SELECT {COLUMNS} FROM messages WHERE session = ? AND summarized = 0 ORDER BY id",
            (session,),
        ).fetchall()
        return [_stored(row) for row in rows]

    def session_started(self, session: int) -> float | None:
        row = self._db.execute(
            "SELECT MIN(created_at) FROM messages WHERE session = ?", (session,)
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def remove_facts(self, ids: list[int]) -> int:
        with self._db:
            cursor = self._db.executemany("DELETE FROM facts WHERE id = ?", [(i,) for i in ids])
        return cursor.rowcount

    def clear(self, conversations: bool = True, facts: bool = True) -> None:
        with self._db:
            if conversations:
                self._db.execute("DELETE FROM messages")
                self._db.execute("DELETE FROM summaries")
            if facts:
                self._db.execute("DELETE FROM facts")


def _fact(row: tuple[int, str, str, float, int, str | None]) -> Fact:
    return Fact(int(row[0]), str(row[1]), str(row[2]), float(row[3]), int(row[4]), row[5])


def _stored(row: tuple[int, int, str, str, str | None, float, int, str | None]) -> StoredMessage:
    id_, session, role, text, language, created_at, interrupted, actions = row
    message = Message(
        Role(role),
        (TextPart(text),),
        Language(language) if language else None,
        interrupted=bool(interrupted),
        actions=_load_actions(actions),
    )
    return StoredMessage(id=id_, session=session, message=message, created_at=created_at)


def _dump_actions(actions: tuple[ToolOutcome, ...]) -> str | None:
    if not actions:
        return None
    return json.dumps(
        [
            {"name": a.call.name, "arguments": a.call.arguments, "result": a.result, "ok": a.ok}
            for a in actions
        ],
        ensure_ascii=False,
    )


def _load_actions(raw: str | None) -> tuple[ToolOutcome, ...]:
    if not raw:
        return ()
    return tuple(
        ToolOutcome(
            ToolCall(f"stored{i}", item["name"], item["arguments"]), item["result"], item["ok"]
        )
        for i, item in enumerate(json.loads(raw))
    )
