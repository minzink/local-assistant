"""Chat backend for any server exposing the OpenAI chat completions API.

Covers Ollama, LM Studio, llama.cpp (llama-server), mlx-lm, vLLM and others; switching
servers only changes ``base_url`` and ``model``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI, DefaultAsyncHttpxClient, NotFoundError
from openai.types.chat import ChatCompletionMessageParam

from cassi import perf
from cassi.core.messages import AudioPart, ImagePart, Message, Role, TextPart, ToolCall
from cassi.llm import budget
from cassi.llm.base import (
    LLMEvent,
    ModelCapabilities,
    Reasoning,
    TextDelta,
    ThinkingOptions,
    ToolSpec,
)
from cassi.llm.gate import synthesis
from cassi.net import trust_env

log = logging.getLogger(__name__)

# Local servers ignore the key, but the client requires one.
PLACEHOLDER_API_KEY = "not-needed"


def create_client(base_url: str, api_key: str | None = None) -> AsyncOpenAI:
    """OpenAI client for ``base_url``; a server on this computer is reached without proxy."""
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or PLACEHOLDER_API_KEY,
        http_client=DefaultAsyncHttpxClient(trust_env=trust_env(base_url)),
    )


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
        context_tokens: int | None = None,
    ) -> None:
        self._client = create_client(base_url, api_key)
        self._context_tokens = context_tokens
        self._model = model
        self._capabilities = capabilities
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_body = extra_body or None
        self._thinking = thinking
        # Which model it is instead, when the server does not know ours (deleted, or replaced
        # by a new one – Flay's after its Modelfile changed); None: no idea.
        self.resolve_missing: Callable[[str], str | None] | None = None

    @property
    def model(self) -> str:
        return self._model

    async def _create(self, model: str, **request: Any) -> Any:
        """``chat.completions.create``; if the server does not know the main model, ask
        which one it is now, switch to it for good and try once more."""
        try:
            return await self._client.chat.completions.create(model=model, **request)
        except NotFoundError as exc:
            if model != self._model or self.resolve_missing is None:
                raise
            replacement = await asyncio.to_thread(self.resolve_missing, model)
            if not replacement or replacement == model:
                raise
            log.warning("Modell %s fehlt (%s) – nutze %s", model, exc.status_code, replacement)
            self._model = replacement
            return await self._client.chat.completions.create(model=replacement, **request)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def limit_context(self, tokens: int) -> None:
        """The server gives the model less context than configured: plan with that."""
        if self._context_tokens is None or tokens < self._context_tokens:
            self._context_tokens = tokens

    def _messages(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        max_tokens: int = 0,
    ) -> list[ChatCompletionMessageParam]:
        """The request's messages, shortened to the context if one is set."""
        if self._context_tokens is not None:
            messages = budget.fit(
                messages, self._context_tokens - max_tokens, budget.tools_tokens(tools)
            )
        return cast(list[ChatCompletionMessageParam], [to_api_message(m) for m in messages])

    async def warm(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> None:
        """Process a prompt without generating: keeps the model loaded and its prefix cached."""
        api_tools = [to_api_tool(t) for t in tools]
        extra: dict[str, Any] = {"tools": api_tools} if tools else {}
        await synthesis.wait()  # see cassi.llm.gate
        with perf.request("warm"):
            await self._create(
                self._model,
                messages=self._messages(messages, api_tools, 1),
                max_tokens=1,
                extra_body=self._extra_body,
                **extra,
            )

    async def complete_json(
        self,
        messages: Sequence[Message],
        schema: dict[str, Any],
        name: str = "answer",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        max_tokens = max_tokens or self._max_tokens
        start = time.perf_counter()
        await synthesis.wait()  # see cassi.llm.gate
        _waited(start)
        sent = time.perf_counter()
        with perf.request(perf.purpose()):
            response = await self._create(
                self._model,
                messages=self._messages(messages, max_tokens=max_tokens),
                # Deterministic: the same request should always be routed the same way.
                temperature=0,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": name, "schema": schema, "strict": True},
                },
                extra_body=self._extra_body,
            )
        usage = response.usage
        perf.llm_call(
            perf.LLMCall(
                perf.purpose(),
                self._model,
                time.perf_counter() - sent,
                None,
                usage.prompt_tokens if usage else None,
                usage.completion_tokens if usage else None,
                _images(messages),
            )
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError(f"expected a JSON object, got: {content[:200]}")
        return result

    async def reason(
        self, messages: Sequence[Message], seconds: float, model: str | None = None
    ) -> Reasoning:
        """Let a model think for at most ``seconds`` (by default the thinking model, if one is
        set up, else the main one): its reasoning, and its answer if it got that far."""
        thinking = self._thinking
        model = model or (thinking.model if thinking else None) or self._model
        max_tokens = thinking.max_tokens if thinking else 4000
        if self._context_tokens is not None:
            # Room for the question: a long budget must not shorten what is asked.
            max_tokens = min(max_tokens, self._context_tokens // 2)
        body = (self._extra_body or {}) | (thinking.extra_body if thinking else {})
        thoughts: list[str] = []
        text: list[str] = []
        finished = True
        await synthesis.wait()  # see cassi.llm.gate
        with perf.request(perf.purpose()):
            try:
                async with asyncio.timeout(seconds):
                    response = await self._create(
                        model,
                        messages=self._messages(messages, max_tokens=max_tokens),
                        temperature=self._temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        extra_body=body or None,
                    )
                    async with response:
                        async for chunk in response:
                            if not chunk.choices:
                                continue
                            delta = chunk.choices[0].delta
                            extra = delta.model_extra or {}
                            # Ollama: "reasoning"; llama.cpp, vLLM: "reasoning_content".
                            for key in ("reasoning", "reasoning_content"):
                                if isinstance(extra.get(key), str):
                                    thoughts.append(extra[key])
                            if delta.content:
                                text.append(delta.content)
            except TimeoutError:
                finished = False
        answer = "".join(text)
        # Some servers leave the thinking in the text as <think>…</think>.
        inside = re.findall(r"<think>(.*?)(?:</think>|$)", answer, flags=re.S)
        if inside:
            thoughts.extend(inside)
            answer = re.sub(r"<think>.*?(?:</think>|$)", "", answer, flags=re.S)
        return Reasoning("".join(thoughts).strip(), answer.strip(), finished)

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = (), think: bool = False
    ) -> AsyncIterator[LLMEvent]:
        self._capabilities.ensure_supported(messages)
        api_tools = [to_api_tool(t) for t in tools]
        extra: dict[str, Any] = {"tools": api_tools} if tools else {}
        model, max_tokens, body = self._model, self._max_tokens, self._extra_body
        if think and self._thinking is not None:
            model = self._thinking.model or model
            max_tokens = self._thinking.max_tokens
            body = (body or {}) | self._thinking.extra_body
        start = time.perf_counter()
        await synthesis.wait()  # see cassi.llm.gate
        _waited(start)
        sent = time.perf_counter()
        # Counted as running while it streams (cassi.perf: the cache guard waits).
        with perf.request(perf.purpose()):
            response = await self._create(
                model,
                messages=self._messages(messages, api_tools, max_tokens),
                temperature=self._temperature,
                max_tokens=max_tokens,
                stream=True,
                # The last chunk then carries the token counts (for cassi.perf).
                stream_options={"include_usage": True},
                extra_body=body,
                **extra,
            )
            measured = perf.LLMCall(perf.purpose(), model, 0.0, images=_images(messages))
            perf.llm_call(measured)
            # Tool calls arrive in fragments (name, then argument pieces) keyed by index.
            pending: dict[int, _PendingCall] = {}
            # Leaving the context closes the HTTP stream, which stops generation on the
            # server when the consuming task is cancelled.
            # Servers send thinking in a field of its own; some leave it in the text as
            # <think>…</think>. Either way it must never be spoken.
            hidden = ThinkFilter()
            async with response:
                async for chunk in response:
                    if chunk.usage is not None:
                        measured.prompt_tokens = chunk.usage.prompt_tokens
                        measured.output_tokens = chunk.usage.completion_tokens
                    measured.seconds = time.perf_counter() - sent
                    if not chunk.choices:
                        continue
                    if measured.first_token is None:
                        measured.first_token = measured.seconds
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


def _waited(start: float) -> None:
    """Time spent waiting for queued speech (cassi.llm.gate) before the request went out."""
    waited = time.perf_counter() - start
    if waited > 0.01 and (profile := perf.current()) is not None:
        profile.add("wait_for_speech", waited)


def _images(messages: Sequence[Message]) -> int:
    return sum(isinstance(part, ImagePart) for m in messages for part in m.parts)


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
