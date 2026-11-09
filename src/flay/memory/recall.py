"""Finding memories again: meaning (embeddings) and words (FTS5) combined.

Embeddings find paraphrases ("Urlaub" for "frei am Freitag"); keywords find names
embeddings blur ("kx-ewo", "Lea"). The two rankings are merged by reciprocal rank fusion,
so neither needs a calibrated score. Without an embedding model (not pulled, server down)
recall falls back to keywords alone.
"""

from __future__ import annotations

import asyncio
import logging
import re

import numpy as np

from flay.config import RecallSettings
from flay.memory.base import EPISODE, KNOWLEDGE, Fact
from flay.memory.embeddings import Embedder
from flay.memory.store import MemoryStore

log = logging.getLogger(__name__)

# Candidates taken from each ranking before fusing.
CANDIDATES = 20
# Reciprocal rank fusion constant (the usual value from the original paper).
RRF_K = 60
# Meaning-based hits must be this close to the best one. Absolute similarities of related
# and unrelated memories overlap (measured with Qwen3-Embedding: related 0.32–0.57,
# unrelated up to 0.47), but within one request the related ones stand out.
MARGIN = 0.12
# Score factor for minor memories when ranking.
MINOR_WEIGHT = 0.7
# Episodes are long and resemble many requests a little (unrelated ones reached 0.49 on
# real conversations, related ones 0.57–0.62), so they need more than short facts do.
EPISODE_MIN_SIMILARITY = 0.50
# "Was haben wir letztes Mal besprochen?" has little to match by meaning; it means the
# latest episode.
_LAST_TIME = re.compile(
    r"\b(?:letzte[sn]? mal|zuletzt|neulich|vorhin|gestern|last time|recently|yesterday)\b",
    re.IGNORECASE,
)

# Words too common to say anything about relevance (incl. "Projekt", which is in nearly
# every request to Flay).
STOPWORDS = frozenset(
    """der die das den dem des ein eine einen einem einer und oder aber nicht ist sind war
    hat habe haben ich du er sie es wir ihr mir mich dir dich uns euch mal bitte auch noch
    schon dass wie was wer wo wann warum für mit von bei aus auf zum zur über unter nach vor
    kann kannst soll sollst will willst wird werden the and for with you your are was this
    that what how why when can could would should have has not but from into about please
    projekt projekte projekts project projects""".split()
)


def keyword_query(text: str) -> str:
    """The text without words that occur in almost every sentence."""
    words = re.findall(r"\w+", text.lower())
    # Two letters count: project names are made of parts like "kx".
    return " ".join(w for w in words if len(w) >= 2 and w not in STOPWORDS)


class Recall:
    def __init__(
        self, store: MemoryStore, embedder: Embedder | None, settings: RecallSettings
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings
        # The last request's vector, being computed or done: memories and examples are
        # searched with the same text at the same time, one embedding call serves both.
        self._last_query: tuple[str, asyncio.Future[np.ndarray]] | None = None

    async def search(
        self,
        query: str,
        limit: int | None = None,
        kinds: tuple[str, ...] = KNOWLEDGE,
        min_similarity: float | None = None,
        keywords: bool = True,
    ) -> list[Fact]:
        """``keywords=False``: by meaning only – keywords serve as fallback when there is
        no embedding model."""
        limit = limit or self._settings.limit
        by_words = self._store.keyword_search(keyword_query(query), CANDIDATES, kinds)
        rankings = [by_words]
        if self._embedder is not None:
            try:
                by_meaning = await self._by_meaning(query, kinds, min_similarity)
                rankings = [by_words, by_meaning] if keywords else [by_meaning]
            except Exception as exc:
                log.debug("Erinnerungen nur über Stichwörter (Embedding: %s)", exc)
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, fact_id in enumerate(ranking):
                scores[fact_id] = scores.get(fact_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        candidates = self._store.facts_by_ids(list(scores))
        # Minor memories (details of past conversations) only win when clearly better.
        weight = {fact.id: MINOR_WEIGHT if fact.importance < 2 else 1.0 for fact in candidates}
        ranked = sorted(candidates, key=lambda fact: -scores[fact.id] * weight[fact.id])
        found = ranked[:limit]
        if (
            EPISODE in kinds
            and _LAST_TIME.search(query)
            and (latest := self._store.latest_episode())
        ):
            found = [latest, *(fact for fact in found if fact.id != latest.id)][:limit]
        return found

    async def _by_meaning(
        self, query: str, kinds: tuple[str, ...], min_similarity: float | None = None
    ) -> list[int]:
        assert self._embedder is not None
        ids, vectors = self._store.embeddings(self._embedder.model, kinds)
        if not ids:
            return []
        vector = await self._query_vector(query)
        similarity = vectors @ vector
        order = np.argsort(-similarity)[:CANDIDATES]
        floor = self._settings.min_similarity if min_similarity is None else min_similarity
        cutoff = max(floor, float(similarity[order[0]]) - MARGIN)
        episodes = self._store.episode_ids()
        # Below that a memory is merely the least unrelated one, not a relevant one.
        return [
            ids[i]
            for i in order
            if similarity[i] >= cutoff
            and (ids[i] not in episodes or similarity[i] >= EPISODE_MIN_SIMILARITY)
        ]

    async def _query_vector(self, query: str) -> np.ndarray:
        assert self._embedder is not None
        if self._last_query is None or self._last_query[0] != query:
            embedder = self._embedder

            async def embed() -> np.ndarray:
                vectors = await embedder.embed([query], query=True)
                vector: np.ndarray = vectors[0]
                return vector

            self._last_query = (query, asyncio.ensure_future(embed()))
        # Shielded: one search being cancelled must not cancel the other's vector.
        return await asyncio.shield(self._last_query[1])

    async def warm_up(self) -> None:
        """Load the embedding model (Ollama: ~5 s the first time) before it is needed."""
        if self._embedder is not None:
            await self.embed_pending()
            await self._embedder.embed(["."], query=True)

    async def embed_pending(self) -> int:
        """Give new or changed memories their vector; returns how many were embedded."""
        if self._embedder is None:
            return 0
        pending = self._store.unembedded(self._embedder.model)
        if not pending:
            return 0
        vectors = await self._embedder.embed([fact.text for fact in pending])
        for fact, vector in zip(pending, vectors, strict=True):
            self._store.set_embedding(fact.id, vector, self._embedder.model)
        log.debug("%d Erinnerungen eingebettet", len(pending))
        return len(pending)
