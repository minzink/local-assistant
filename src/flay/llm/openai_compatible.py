"""Chat backend for any server exposing the OpenAI chat completions API.

Covers Ollama, LM Studio, llama.cpp (llama-server), mlx-lm, vLLM and others; switching
servers only changes ``base_url`` and ``model``.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from flay.core.messages import AudioPart, ImagePart, Message, Role, TextPart, ToolCall
from flay.llm.base import LLMEvent, ModelCapabilities, TextDelta, ThinkingOptions, ToolSpec
from flay.llm.gate import synthesis

# Local servers ignore the key, but the client requires one.
PLACEHOLDER_API_KEY = "not-needed"

AUDIO_FORMATS = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3"}


def _content_part(part: TextPart | ImagePart | AudioPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    encoded = base64.b64encode(part.data).decode("ascii")
    if isinstance(part, ImagePart):
        url = f"data:{part.mime_type};base64,{encoded}"
        return {"type": "image_url", "image_url": {"url": url}}
    audio_format = AUDIO_FORMATS.get(part.mime_type)
    if audio_format is None:
        raise ValueError(f"unsupported audio type: {part.mime_type}")
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}}


def to_api_message(message: Message) -> dict[str, Any]:
    if message.role is Role.TOOL:
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.text}
    if all(isinstance(part, TextPart) for part in message.parts):
        api: dict[str, Any] = {"role": message.role.value, "content": message.text}
    else:
        api = {"role": message.role.value, "content": [_content_part(p) for p in message.parts]}
    if message.tool_calls:
        api["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return api


def to_api_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def parse_arguments(raw: str) -> dict[str, Any]:
    try:
        arguments = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}
    return arguments if isinstance(arguments, dict) else {}


@dataclass
class _PendingCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


class OpenAICompatibleChat:
    def __init__(
        self,
        base_url: str,
        model: str,
        capabilities: ModelCapabilities,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 400,
        extra_body: dict[str, Any] | None = None,
        thinking: ThinkingOptions | None = None,
    ) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or PLACEHOLDER_API_KEY)
        self._model = model
        self._capabilities = capabilities
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_body = extra_body or None
        self._thinking = thinking

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def warm(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> None:
        """Process a prompt without generating: keeps the model loaded and its prefix cached."""
        extra: dict[str, Any] = {"tools": [to_api_tool(t) for t in tools]} if tools else {}
        await synthesis.wait()  # see flay.llm.gate
        await self._client.chat.completions.create(
            model=self._model,
            messages=cast(list[ChatCompletionMessageParam], [to_api_message(m) for m in messages]),
            max_tokens=1,
            extra_body=self._extra_body,
            **extra,
        )

    async def complete_json(
        self, messages: Sequence[Message], schema: dict[str, Any], name: str = "answer"
    ) -> dict[str, Any]:
        await synthesis.wait()  # see flay.llm.gate
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(list[ChatCompletionMessageParam], [to_api_message(m) for m in messages]),
            # Deterministic: the same request should always be routed the same way.
            temperature=0,
            max_tokens=self._max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
            extra_body=self._extra_body,
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError(f"expected a JSON object, got: {content[:200]}")
        return result

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = (), think: bool = False
    ) -> AsyncIterator[LLMEvent]:
        self._capabilities.ensure_supported(messages)
        extra: dict[str, Any] = {"tools": [to_api_tool(t) for t in tools]} if tools else {}
        model, max_tokens, body = self._model, self._max_tokens, self._extra_body
        if think and self._thinking is not None:
            model = self._thinking.model or model
            max_tokens = self._thinking.max_tokens
            body = (body or {}) | self._thinking.extra_body
        await synthesis.wait()  # see flay.llm.gate
        response = await self._client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], [to_api_message(m) for m in messages]),
            temperature=self._temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body=body,
            **extra,
        )
        # Tool calls arrive in fragments (name, then argument pieces) keyed by index.
        pending: dict[int, _PendingCall] = {}
        # Leaving the context closes the HTTP stream, which stops generation on the
        # server when the consuming task is cancelled.
        # Servers send thinking in a field of its own; some leave it in the text as
        # <think>…</think>. Either way it must never be spoken.
        hidden = ThinkFilter()
        async with response:
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content and (text := hidden.feed(delta.content)):
                    yield TextDelta(text)
                for fragment in delta.tool_calls or ():
                    call = pending.setdefault(fragment.index, _PendingCall())
                    call.id = fragment.id or call.id
                    if fragment.function is not None:
                        call.name += fragment.function.name or ""
                        call.arguments += fragment.function.arguments or ""
        if rest := hidden.flush():
            yield TextDelta(rest)
        for index, call in sorted(pending.items()):
            if call.name:
                yield ToolCall(
                    id=call.id or f"call_{index}",
                    name=call.name,
                    arguments=parse_arguments(call.arguments),
                )


class ThinkFilter:
    """Drops ``<think>…</think>`` from streamed text, also when tags are split up."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._inside = False
        self._held = ""

    def feed(self, text: str) -> str:
        text, self._held = self._held + text, ""
        shown = []
        while text:
            tag = self.CLOSE if self._inside else self.OPEN
            index = text.find(tag)
            if index >= 0:
                if not self._inside:
                    shown.append(text[:index])
                text = text[index + len(tag) :]
                self._inside = not self._inside
                continue
            # Keep a possible beginning of a tag until the next piece arrives.
            keep = next((n for n in range(len(tag) - 1, 0, -1) if text.endswith(tag[:n])), 0)
            if not self._inside:
                shown.append(text[: len(text) - keep])
            self._held = text[len(text) - keep :] if keep else ""
            break
        return "".join(shown)

    def flush(self) -> str:
        """What was held back for a tag that never came."""
        held, self._held = self._held, ""
        return "" if self._inside else held
