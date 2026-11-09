from __future__ import annotations

from flay.config import LLMSettings
from flay.llm.base import ChatModel, ThinkingOptions
from flay.llm.openai_compatible import OpenAICompatibleChat


def create_chat_model(settings: LLMSettings) -> ChatModel:
    # Only one backend so far; further ones are selected by ``settings.backend``.
    return OpenAICompatibleChat(
        base_url=settings.base_url,
        model=settings.model,
        capabilities=settings.capabilities,
        api_key=settings.api_key.get_secret_value() if settings.api_key else None,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        extra_body=settings.extra_body,
        thinking=ThinkingOptions(
            model=settings.thinking.model,
            extra_body=settings.thinking.extra_body,
            max_tokens=settings.thinking.max_tokens,
        )
        if settings.thinking.enabled
        else None,
    )
