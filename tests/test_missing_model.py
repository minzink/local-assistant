from typing import Any

import httpx2 as httpx
import pytest
from openai import NotFoundError

from cassi.adapters import Adapter
from cassi.cli import replacement_model
from cassi.config import Settings
from cassi.llm import ollama
from cassi.llm.base import ModelCapabilities
from cassi.llm.openai_compatible import OpenAICompatibleChat


def not_found(model: str) -> NotFoundError:
    request = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
    response = httpx.Response(404, request=request)
    return NotFoundError(f"model '{model}' not found", response=response, body=None)


def chat_with(known: set[str]) -> tuple[OpenAICompatibleChat, list[str]]:
    chat = OpenAICompatibleChat("http://127.0.0.1:11434/v1", "flay:old", ModelCapabilities())
    asked: list[str] = []

    async def create(model: str, **request: Any) -> str:
        asked.append(model)
        if model not in known:
            raise not_found(model)
        return "ok"

    chat._client.chat.completions.create = create  # type: ignore[method-assign]
    return chat, asked


async def test_a_missing_model_is_replaced_once_and_for_good() -> None:
    chat, asked = chat_with({"flay:new"})
    chat.resolve_missing = lambda missing: "flay:new"

    assert await chat._create("flay:old", messages=[]) == "ok"
    assert await chat._create(chat.model, messages=[]) == "ok"

    assert asked == ["flay:old", "flay:new", "flay:new"]  # asked about the old one once
    assert chat.model == "flay:new"


async def test_without_a_replacement_the_error_stays() -> None:
    chat, _ = chat_with(set())
    chat.resolve_missing = lambda missing: None

    with pytest.raises(NotFoundError):
        await chat._create("flay:old", messages=[])


class Knowing(Adapter):
    name = "knowing"

    def resolve_model(self, settings: Settings, missing: str) -> str | None:
        return "flay:new" if missing.startswith("flay:") else None


def test_an_adapter_knows_its_own_model_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "is_ollama", lambda url: True)
    monkeypatch.setattr(ollama, "models", lambda url: {"qwen3-vl:8b-instruct"})
    monkeypatch.setattr(ollama, "main_model", lambda url, model: f"{model}-ctx8k")

    assert replacement_model(Settings(), [Knowing()], "flay:old") == "flay:new"
    # Cassi's context variant gone: made again from its base model (here the same name,
    # so there is nothing to switch to – the variant exists again under its old name).
    assert replacement_model(Settings(), [], "qwen3-vl:8b-instruct-ctx8k") is None
    assert replacement_model(Settings(), [], "qwen3-vl:8b-instruct") == "qwen3-vl:8b-instruct-ctx8k"
    # Never just any other model: an unknown one has no replacement.
    assert replacement_model(Settings(), [], "llama3:8b") is None
