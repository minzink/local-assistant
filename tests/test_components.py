import numpy as np
import pytest
from pynput import keyboard

from flay.audio.base import AudioBuffer
from flay.audio.microphone import to_pcm16
from flay.bench import BenchResult, format_report
from flay.core.language import Language
from flay.core.messages import ImagePart, Message, Role, TextPart
from flay.core.platform import Accelerator, HardwareProfile
from flay.input.push_to_talk import parse_key
from flay.llm.openai_compatible import to_api_message
from flay.stt.factory import resolve_backend
from flay.stt.whisper import pick_language, to_float32

MAC = HardwareProfile("Darwin", "arm64", Accelerator.APPLE_SILICON)
WINDOWS = HardwareProfile("Windows", "amd64", Accelerator.CUDA)


def test_resample_to_16k() -> None:
    one_second = np.zeros(48_000, dtype=np.float32)

    pcm = to_pcm16(one_second, 48_000, 16_000)

    assert AudioBuffer(pcm, 16_000).duration == pytest.approx(1.0)


def test_pcm_roundtrip_is_clipped() -> None:
    pcm = to_pcm16(np.array([2.0, -2.0, 0.5], dtype=np.float32), 16_000, 16_000)

    samples = to_float32(AudioBuffer(pcm, 16_000))

    assert samples == pytest.approx([1.0, -1.0, 0.5], abs=1e-3)


def test_whisper_rejects_wrong_rate() -> None:
    with pytest.raises(ValueError):
        to_float32(AudioBuffer(b"\x00\x00", 44_100))


def test_pick_language_ignores_unsupported() -> None:
    probs = {"nl": 0.6, "de": 0.3, "en": 0.1}

    assert pick_language(probs, Language.EN) is Language.DE
    assert pick_language({"nl": 1.0}, Language.EN) is Language.EN
    assert pick_language(None, Language.DE) is Language.DE


def test_stt_backend_auto_resolution() -> None:
    assert resolve_backend("auto", MAC) == "mlx-whisper"
    assert resolve_backend("auto", WINDOWS) == "faster-whisper"
    assert resolve_backend("faster-whisper", MAC) == "faster-whisper"


def test_parse_key() -> None:
    assert parse_key("alt_r") == keyboard.Key.alt_r
    assert parse_key("F8") == keyboard.Key.f8
    assert parse_key("x") == keyboard.KeyCode.from_char("x")
    with pytest.raises(ValueError):
        parse_key("nope")


def test_text_message_uses_plain_content() -> None:
    message = Message.from_text(Role.USER, "Hallo")

    assert to_api_message(message) == {"role": "user", "content": "Hallo"}


def test_image_message_uses_content_parts() -> None:
    message = Message(Role.USER, (TextPart("Was ist das?"), ImagePart(b"abc", "image/png")))

    content = to_api_message(message)["content"]

    assert content[0] == {"type": "text", "text": "Was ist das?"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64,YWJj"


def test_bench_report() -> None:
    results = [BenchResult("a", 0.2, 1.2, 100), BenchResult("b", 0.4, 1.4, 200)]

    report = format_report(results)

    assert "median 0.30s" in report
    assert "150 Zeichen/s" in report


def test_tool_messages_are_converted() -> None:
    from flay.core.messages import ToolCall

    call = Message(
        Role.ASSISTANT,
        (TextPart(""),),
        tool_calls=(ToolCall("c1", "open_project", {"name": "flay"}),),
    )
    result = Message(Role.TOOL, (TextPart("ok"),), tool_call_id="c1")

    assert to_api_message(call)["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "open_project", "arguments": '{"name": "flay"}'},
        }
    ]
    assert to_api_message(result) == {"role": "tool", "tool_call_id": "c1", "content": "ok"}


async def test_streamed_tool_call_fragments_are_assembled() -> None:
    from openai.types.chat import ChatCompletionChunk

    from flay.core.messages import ToolCall
    from flay.llm.base import ModelCapabilities, TextDelta, ToolSpec
    from flay.llm.openai_compatible import OpenAICompatibleChat

    def chunk(delta: dict[str, object]) -> ChatCompletionChunk:
        return ChatCompletionChunk.model_validate(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )

    chunks = [
        chunk({"content": "Ich öffne es."}),
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "function": {"name": "open_project", "arguments": '{"na'},
                    }
                ]
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'me": "flay"}'}}]}),
    ]

    class Stream:
        async def __aenter__(self) -> "Stream":
            return self

        async def __aexit__(self, *exc: object) -> None:
            pass

        def __aiter__(self):  # type: ignore[no-untyped-def]
            async def gen():  # type: ignore[no-untyped-def]
                for c in chunks:
                    yield c

            return gen()

    sent: dict[str, object] = {}

    async def create(**kwargs: object) -> Stream:
        sent.update(kwargs)
        return Stream()

    chat = OpenAICompatibleChat("http://x/v1", "m", ModelCapabilities())
    chat._client.chat.completions.create = create  # type: ignore[method-assign, assignment]
    spec = ToolSpec("open_project", "Open", {"type": "object"})

    events = [e async for e in chat.stream([Message.from_text(Role.USER, "Öffne flay")], [spec])]

    assert events == [TextDelta("Ich öffne es."), ToolCall("c1", "open_project", {"name": "flay"})]
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "open_project",
                "description": "Open",
                "parameters": {"type": "object"},
            },
        }
    ]
