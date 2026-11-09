"""Persistent memory: SQLite storage plus background maintenance with the chat model."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from flay.config import MemorySettings
from flay.core.messages import Message, ToolCall
from flay.llm.base import ChatModel
from flay.memory import maintenance
from flay.memory.base import (
    EXAMPLE,
    FACT,
    IMPORTANCE_NAMES,
    IMPORTANT,
    KNOWLEDGE,
    MISTAKE,
    NORMAL,
    VERB,
    Fact,
    MemoryContext,
    sounds_important,
)
from flay.memory.embeddings import Embedder
from flay.memory.recall import Recall
from flay.memory.store import MemoryStore
from flay.verbs import learned_verb

if TYPE_CHECKING:
    from flay.training import Counts

log = logging.getLogger(__name__)

# Messages handed to one fact extraction call.
FACT_BATCH = 20
# Important memories (rules) always in the prompt; more would crowd it.
MAX_IMPORTANT = 20
# Examples of earlier requests shown to the model per request.
EXAMPLES_PER_REQUEST = 3
# Measured with Qwen3-Embedding on the user's requests: chat reached at most 0.41 similarity
# to command examples, garbled commands at least 0.46 ("She K Command"), clear ones 0.69+.
# Below this an example would only lengthen the prompt (and slow the answer down).
EXAMPLE_MIN_SIMILARITY = 0.43
# Known memories shown to the extraction, so it updates instead of duplicating.
KNOWN_FOR_EXTRACTION = 30
# Finished conversations turned into episode memories per maintenance run.
EPISODES_PER_RUN = 3
# Messages of a finished conversation the episode is written from.
EPISODE_MESSAGES = 60


class PersistentMemory:
    """Implements ``Memory``.

    Summaries and fact extraction need the LLM, which local servers usually serve one
    request at a time. They therefore run only after ``idle_seconds`` without a new turn
    and are cancelled as soon as the user speaks again; unfinished work is picked up later
    because the database marks what has been processed.
    """

    def __init__(
        self,
        store: MemoryStore,
        llm: ChatModel,
        settings: MemorySettings,
        clock: Callable[[], float] = time.time,
        embedder: Embedder | None = None,
    ) -> None:
        self._store = store
        self._llm = llm
        self._settings = settings
        self._clock = clock
        self._maintenance: asyncio.Task[None] | None = None
        self._facts_changed_this_turn = False
        self._recall = Recall(store, embedder, settings.recall)
        self._embedding: asyncio.Task[int] | None = None

    @property
    def _timeout(self) -> float:
        return self._settings.session_timeout_minutes * 60

    def context(self) -> MemoryContext:
        session = self._store.session_at(self._clock(), self._timeout)
        recent = self._store.recent(session, self._settings.recent_messages)
        # With recall, memories come per request (``recall``) instead of all of them in
        # the system prompt: that scales, and the system prompt stays the same.
        facts = (
            ()
            if self._settings.recall.enabled
            else self._store.facts(self._settings.max_facts, kind=FACT)
        )
        return MemoryContext(
            recent=tuple(stored.message for stored in recent),
            summary=self._store.summary(session),
            facts=tuple(f for f in facts if not f.important),
            important=tuple(self._store.facts(MAX_IMPORTANT, kind=FACT, min_importance=IMPORTANT)),
        )

    # Examples ---------------------------------------------------------------------------

    def learn(self, request: str, call: ToolCall, confirmed: bool = False) -> None:
        """Keep a request as heard and the action it meant, for similar requests later."""
        known = self._example(request)
        times = 1
        if known is not None and known.data:
            before = json.loads(known.data)
            same = before.get("tool") == call.name and before.get("arguments") == call.arguments
            times = int(before.get("times", 1)) + 1 if same else 1
            confirmed = confirmed or (same and bool(before.get("confirmed")))
        data = json.dumps(
            {
                "tool": call.name,
                "arguments": call.arguments,
                "confirmed": confirmed,
                "times": times,
            },
            ensure_ascii=False,
        )
        if known is not None:
            # Heard the same way again: the latest meaning counts.
            self._store.update_data(known.id, data, self._clock())
        else:
            self._store.add_fact(request, self._clock(), kind=EXAMPLE, data=data)
            self._embed_soon()
        log.info("Beispiel gelernt: „%s“ → %s(%s)", request, call.name, call.arguments)
        if word := learned_verb(request, call.name, str(call.arguments.get("name", ""))):
            self._learn_verb(word, call.name)

    def _learn_verb(self, word: str, tool: str) -> None:
        known = self._example(word, VERB)
        times = 1
        if known is not None and known.data:
            before = json.loads(known.data)
            times = int(before.get("times", 1)) + 1 if before.get("tool") == tool else 1
        data = json.dumps({"tool": tool, "times": times})
        if known is not None:
            self._store.update_data(known.id, data, self._clock())
        else:
            self._store.add_fact(word, self._clock(), kind=VERB, data=data)
        log.info("Verb gelernt: „%s“ → %s", word, tool)

    def verb(self, word: str) -> str | None:
        """The action a learned verb ("she") stands for."""
        known = self._example(word, VERB)
        return str(json.loads(known.data)["tool"]) if known is not None and known.data else None

    async def examples(self, request: str) -> tuple[Fact, ...]:
        """Earlier requests similar to this one, with the action each meant."""
        if not request.strip():
            return ()
        # By meaning only: shared words ("Symfony") would attach command examples to a
        # question about Symfony.
        return tuple(
            await self._recall.search(
                request,
                EXAMPLES_PER_REQUEST,
                kinds=(EXAMPLE,),
                min_similarity=EXAMPLE_MIN_SIMILARITY,
                keywords=False,
            )
        )

    def known_action(self, request: str) -> tuple[ToolCall, bool] | None:
        """The action this exact request (as heard) surely means, and if the user confirmed it.

        Surely: confirmed with "ja", or meant the same action at least twice. A request
        heard once and not contradicted is only a hint for the model (``examples``).
        """
        example = self._example(request)
        if example is None or not example.data:
            return None
        data = json.loads(example.data)
        confirmed = bool(data.get("confirmed"))
        if not confirmed and int(data.get("times", 1)) < 2:
            return None
        call = ToolCall("learned", str(data["tool"]), dict(data.get("arguments") or {}))
        return call, confirmed

    def unlearn(self, request: str, call: ToolCall | None = None) -> None:
        """The user contradicted what this request was taken to mean.

        The example goes; the rejected action is kept as a mistake – training data only.
        """
        if (example := self._example(request)) is not None:
            self._store.remove_facts([example.id])
            log.info("Beispiel verworfen: „%s“", request)
        if call is None:
            return
        word = learned_verb(request, call.name, str(call.arguments.get("name", "")))
        if word and self.verb(word) == call.name and (known := self._example(word, VERB)):
            self._store.remove_facts([known.id])
            log.info("Verb verworfen: „%s“", word)
        known = self._example(request, MISTAKE)
        times = 1
        if known is not None and known.data:
            before = json.loads(known.data)
            same = before.get("tool") == call.name and before.get("arguments") == call.arguments
            times = int(before.get("times", 1)) + 1 if same else 1
        data = json.dumps(
            {"tool": call.name, "arguments": call.arguments, "times": times}, ensure_ascii=False
        )
        if known is not None:
            self._store.update_data(known.id, data, self._clock())
        else:
            self._store.add_fact(request, self._clock(), kind=MISTAKE, data=data)
        log.info("Fehler gemerkt: „%s“ ≠ %s(%s)", request, call.name, call.arguments)

    def training_counts(self) -> Counts:
        """Examples and mistakes collected so far, for ``flay train-export``."""
        from flay.training import counts

        return counts(self._store)

    def _example(self, request: str, kind: str = EXAMPLE) -> Fact | None:
        wanted = _compact(request)
        return next(
            (fact for fact in self._store.facts(kind=kind) if _compact(fact.text) == wanted),
            None,
        )

    async def warm_up(self) -> None:
        """Make recall fast from the first request on (embedding model loaded)."""
        if self._settings.recall.enabled:
            try:
                await self._recall.warm_up()
            except Exception as exc:
                log.warning("Gedächtnis-Suche nur über Stichwörter: %s", exc)

    async def recall(self, query: str) -> tuple[Fact, ...]:
        if not self._settings.recall.enabled or not query.strip():
            return ()
        # Important memories are in the prompt anyway.
        return tuple(f for f in await self._recall.search(query) if not f.important)

    def record(self, user: Message, assistant: Message) -> None:
        now = self._clock()
        session = self._store.session_at(now, self._timeout)
        ids = self._store.add_messages(session, [user, assistant], now)
        if self._facts_changed_this_turn:
            # The remember/forget tools already handled this exchange; extracting facts
            # from it again would only produce duplicates in other words.
            self._store.mark("facts_done", ids)

    # FactMemory -------------------------------------------------------------------------

    def facts(self) -> list[Fact]:
        # What "vergiss …" can refer to: knowledge, not examples or mistakes.
        return [fact for fact in self._store.facts() if fact.kind in KNOWLEDGE]

    def remember(self, text: str, important: bool = False) -> bool:
        self._facts_changed_this_turn = True
        if text.casefold() in {fact.text.casefold() for fact in self._store.facts()}:
            return False
        importance = IMPORTANT if important or sounds_important(text) else NORMAL
        self._store.add_fact(text, self._clock(), importance=importance)
        self._embed_soon()
        return True

    def forget(self, ids: list[int]) -> int:
        self._facts_changed_this_turn = True
        return self._store.remove_facts(ids)

    def turn_started(self) -> None:
        self._facts_changed_this_turn = False
        if self._maintenance is not None and not self._maintenance.done():
            self._maintenance.cancel()

    def turn_finished(self) -> None:
        self.turn_started()
        self._maintenance = asyncio.create_task(self._maintain_when_idle())

    async def close(self) -> None:
        for task in (self._maintenance, self._embedding):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._store.close()

    def _embed_soon(self) -> None:
        """A new memory should be findable at once, not only after the next maintenance."""
        if self._embedding is None or self._embedding.done():
            self._embedding = asyncio.get_running_loop().create_task(self._embed_pending())

    async def _embed_pending(self) -> int:
        try:
            return await self._recall.embed_pending()
        except Exception as exc:
            log.debug("Erinnerungen nicht eingebettet: %s", exc)
            return 0

    async def _maintain_when_idle(self) -> None:
        await asyncio.sleep(self._settings.idle_seconds)
        try:
            await self.maintain()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Gedächtnispflege fehlgeschlagen: %s", exc)

    async def maintain(self) -> None:
        if self._settings.summarize:
            await self._summarize()
            await self._close_sessions()
        if self._settings.extract_facts:
            await self._extract_facts()
        await self._embed_pending()

    async def _summarize(self) -> None:
        session = self._store.session_at(self._clock(), self._timeout)
        old = self._store.unsummarized_before_recent(session, self._settings.recent_messages)
        if not old:
            return
        summary = await maintenance.summarize(
            self._llm, self._store.summary(session), [stored.message for stored in old]
        )
        if summary:
            self._store.set_summary(session, summary)
            self._store.mark("summarized", [stored.id for stored in old])
            log.debug("Zusammenfassung aktualisiert: %s", summary)

    async def _close_sessions(self) -> None:
        """Turn finished conversations into episode memories ("Am 30.10. ging es um …")."""
        current = self._store.session_at(self._clock(), self._timeout)
        for session in self._store.sessions_without_episode(current)[:EPISODES_PER_RUN]:
            # Written from the conversation itself, not from the running summary: that one
            # serves the conversation while it lasts and repeats the assistant's claims.
            messages = self._store.session_messages(session, EPISODE_MESSAGES)
            summary = await maintenance.summarize_episode(
                self._llm, [stored.message for stored in messages]
            )
            if not summary:
                continue
            started = self._store.session_started(session) or self._clock()
            day = time.strftime("%Y-%m-%d", time.localtime(started))
            # The conversation's date is in the text; the memory's date is when it was written.
            self._store.set_episode(session, f"Conversation on {day}: {summary}", self._clock())
            log.info("Gespräch vom %s als Erinnerung gespeichert", day)

    async def _known_for(self, messages: list[Message]) -> list[Fact]:
        """The memories the extraction must know: the related ones once there are many."""
        facts = self._store.facts(kind=FACT)
        if len(facts) <= KNOWN_FOR_EXTRACTION or not self._settings.recall.enabled:
            return facts
        query = " ".join(message.text for message in messages)
        related = await self._recall.search(query, KNOWN_FOR_EXTRACTION)
        return [fact for fact in related if fact.kind == FACT]

    async def _extract_facts(self) -> None:
        pending = self._store.pending_facts(FACT_BATCH)
        if not pending:
            return
        messages = [stored.message for stored in pending]
        day = time.strftime("%Y-%m-%d (%A)", time.localtime(pending[-1].created_at))
        changes = await maintenance.extract_facts(
            self._llm, await self._known_for(messages), messages, day
        )
        known = {fact.text.casefold() for fact in self._store.facts()}
        now = self._clock()
        for text, importance in changes.add:
            if text.casefold() not in known:
                if sounds_important(text):
                    importance = IMPORTANT
                self._store.add_fact(text, now, importance=importance)
                log.info("Gemerkt (%s): %s", IMPORTANCE_NAMES[importance], text)
        if changes.remove:
            self._store.remove_facts(list(changes.remove))
            log.info("Vergessen: Fakten %s", ", ".join(map(str, changes.remove)))
        self._store.mark("facts_done", [stored.id for stored in pending])


def _compact(text: str) -> str:
    """The words only: "She K Command." and "she k command" are the same request."""
    return re.sub(r"[^\w]+", "", text.lower())
