"""Embeddings: texts as vectors whose closeness reflects closeness in meaning.

Any server with an OpenAI-compatible ``/embeddings`` endpoint works (Ollama, llama.cpp);
the default is Qwen3-Embedding 0.6B (Apache 2.0, multilingual) on the local Ollama.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from openai import AsyncOpenAI

from flay.config import Settings
from flay.llm.openai_compatible import PLACEHOLDER_API_KEY

# Qwen3-Embedding is trained with an instruction in front of queries, not documents.
QUERY_INSTRUCTION = (
    "Instruct: Given a request to a personal voice assistant, retrieve memories that help "
    "with it\nQuery: "
)


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    async def embed(self, texts: Sequence[str], query: bool = False) -> np.ndarray:
        """One unit-length row per text; ``query`` for search requests, not stored text."""
        ...


class OpenAICompatibleEmbedder:
    def __init__(
        self, base_url: str, model: str, api_key: str | None = None, query_prefix: str = ""
    ) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or PLACEHOLDER_API_KEY)
        self._model = model
        self._query_prefix = query_prefix

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: Sequence[str], query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        inputs = [self._query_prefix + t if query else t for t in texts]
        response = await self._client.embeddings.create(model=self._model, input=inputs)
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        return normalize(vectors)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit: np.ndarray = vectors / np.maximum(norms, 1e-12)
    return unit


def query_prefix(model: str) -> str:
    return QUERY_INSTRUCTION if "qwen3-embedding" in model.lower() else ""


def create_embedder(settings: Settings) -> Embedder | None:
    """The embedding model memory search uses; None if recall is switched off."""
    recall = settings.memory.recall
    if not recall.enabled:
        return None
    base_url = recall.base_url or settings.llm.base_url
    model = embedding_model(recall.model, base_url)
    return OpenAICompatibleEmbedder(
        base_url=base_url,
        model=model,
        api_key=recall.api_key.get_secret_value() if recall.api_key else None,
        query_prefix=query_prefix(model),
    )


def embedding_model(setting: str, base_url: str) -> str:
    from flay.llm.ollama import EMBEDDING, resolve

    return resolve(setting, EMBEDDING, EMBEDDING.base, base_url)
